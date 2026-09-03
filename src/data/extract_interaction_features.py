"""Extract 40-D role-aware interaction features with YOLO pose and ByteTrack."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO

from src.data.feature_schema import (
    TRACK_INTERACTION_COLUMNS,
    TRACK_INTERACTION_FEATURE_SCHEMA,
    TRACK_INTERACTION_SHAPE,
)
from src.data.interaction_tracking import (
    assign_track_roles,
    build_frame_descriptor,
    compute_instruction_alignment_proxy,
    summarize_tracks,
    teacher_zone_membership,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "preprocessed_data" / "yolov5_640x640"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "preprocessed_features" / "interaction_track_features"
)
DEFAULT_DIAGNOSTIC_OUTPUT_DIR = (
    PROJECT_ROOT / "debug_validation" / "interaction_track_features"
)
DEFAULT_TRACK_DETAILS_DIR = PROJECT_ROOT / "debug_validation" / "interaction_tracks"


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_inputs(input_dir: Path) -> list[tuple[str, str, Path]]:
    tasks = []
    for split in ("train", "val", "test"):
        for category in ("low", "mid", "high"):
            directory = input_dir / split / category
            if directory.is_dir():
                tasks.extend(
                    (split, category, path)
                    for path in sorted(directory.glob("*.npz"))
                )
    return tasks


def resolve_model_reference(value: str) -> str:
    """Resolve a local checkpoint consistently regardless of the caller's cwd."""
    candidate = Path(value).expanduser()
    for path in (candidate, PROJECT_ROOT / candidate):
        if path.is_file():
            return str(path.resolve())
    return value


def sha256_file(path: str) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interaction_provenance(args: argparse.Namespace, model_reference: str) -> dict:
    """Record every setting that can change generated interaction values."""
    return {
        "format_version": 2,
        "feature_schema": TRACK_INTERACTION_FEATURE_SCHEMA,
        "shape_per_video": list(TRACK_INTERACTION_SHAPE),
        "columns": list(TRACK_INTERACTION_COLUMNS),
        "implementation": {
            "extractor_sha256": sha256_file(str(Path(__file__).resolve())),
            "pooling_sha256": sha256_file(
                str((SCRIPT_DIR / "interaction_tracking.py").resolve())
            ),
            "schema_sha256": sha256_file(
                str((SCRIPT_DIR / "feature_schema.py").resolve())
            ),
        },
        "detector": {
            "library": "ultralytics",
            "library_version": ultralytics.__version__,
            "model": model_reference,
            "model_sha256": sha256_file(model_reference),
            "confidence": args.confidence,
            "classes": [0],
            "device": select_device(args.device),
            "input_color": "RGB",
            "ultralytics_numpy_color": "BGR",
        },
        "tracker": {
            "library": "ultralytics-bytetrack",
            "config": args.tracker,
            "config_sha256": sha256_file(args.tracker),
            "persist_between_frames": True,
            "reset_between_videos": True,
            "reset_method": "BYTETracker.reset",
            "track_ids_are_clip_local": True,
        },
        "teacher_role": {
            "zone": {
                "x_min": args.zone_x_min,
                "x_max": args.zone_x_max,
                "y_min": args.zone_y_min,
                "y_max": args.zone_y_max,
            },
            "score_weights": list(args.teacher_score_weights),
            "minimum_score": args.minimum_teacher_score,
            "minimum_coverage": args.minimum_teacher_coverage,
            "minimum_detection_confidence": args.minimum_teacher_detection_confidence,
            "minimum_student_coverage": args.minimum_student_coverage,
            "ambiguous_zone_fraction": args.ambiguous_zone_fraction,
            "teacher_is_optional": True,
        },
        "alignment_proxy": {
            "name": "instruction_alignment_proxy",
            "not_gaze_or_attention": True,
            "minimum_keypoint_confidence": args.minimum_keypoint_confidence,
            "aligned_threshold": args.alignment_threshold,
            "missing_value": 0.0,
            "missing_reliability": 0.0,
        },
        "relation_pooling": {
            "permutation_invariant": True,
            "all_valid_student_tracks": True,
            "student_weight": (
                "track_coverage * mean_detection_confidence * "
                "mean_pose_confidence * frame_detection_confidence"
            ),
            "close_pair_distance": args.close_pair_distance,
            "area_ranked_slots": False,
        },
        "sampling_warning": (
            "ByteTrack receives only eight sampled frames; IDs are associations and "
            "are not guaranteed person identities."
        ),
    }


def validate_reusable_provenance(existing: dict | None, requested: dict) -> None:
    """Require an exact generation recipe before reusing existing arrays."""
    if existing is None:
        raise RuntimeError(
            "Existing track-aware interaction files have no provenance manifest. "
            "Use a new --output_dir or --overwrite the complete output set."
        )
    comparable = dict(existing)
    comparable.pop("summary", None)
    if comparable != requested:
        raise RuntimeError(
            "Existing interaction features use an incompatible schema or extraction "
            "configuration. They will not be mixed or relabelled. Use a new "
            "--output_dir or --overwrite the complete output set."
        )


def reusable_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return matrix.shape == TRACK_INTERACTION_SHAPE and np.isfinite(matrix).all()


def reset_tracker_state(model: YOLO) -> None:
    """Reset every Ultralytics tracker so IDs never cross clip boundaries."""
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers is None:
        return
    for tracker in trackers:
        reset = getattr(tracker, "reset", None)
        if not callable(reset):
            raise RuntimeError(
                "Installed Ultralytics tracker has no reset() method; refusing to "
                "risk track IDs leaking across videos."
            )
        reset()
    if hasattr(predictor, "vid_path"):
        predictor.vid_path = [None for _ in trackers]


def _tensor_array(value) -> np.ndarray | None:
    if value is None:
        return None
    return value.detach().cpu().numpy()


def result_observations(
    result,
    frame_index: int,
    zone: tuple[float, float, float, float],
) -> list[dict]:
    """Convert an Ultralytics result into JSON-safe observations."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _tensor_array(boxes.xyxy)
    confidences = _tensor_array(boxes.conf)
    classes = _tensor_array(boxes.cls)
    raw_ids = _tensor_array(boxes.id)
    keypoints = result.keypoints
    keypoint_xy = _tensor_array(keypoints.xy) if keypoints is not None else None
    keypoint_conf = (
        _tensor_array(keypoints.conf)
        if keypoints is not None and keypoints.conf is not None
        else None
    )
    height, width = result.orig_shape
    observations = []
    for index, box in enumerate(xyxy):
        if classes is not None and int(classes[index]) != 0:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        center = [((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height]
        size = [(x2 - x1) / width, (y2 - y1) / height]
        track_id = None if raw_ids is None else int(raw_ids[index])
        points = (
            (keypoint_xy[index] / np.asarray([width, height])).astype(np.float32)
            if keypoint_xy is not None and index < len(keypoint_xy)
            else None
        )
        point_confidences = (
            keypoint_conf[index].astype(np.float32)
            if keypoint_conf is not None and index < len(keypoint_conf)
            else None
        )
        pose_confidence = (
            float(np.clip(point_confidences.mean(), 0.0, 1.0))
            if point_confidences is not None and len(point_confidences)
            else 0.0
        )
        observations.append(
            {
                "frame_index": frame_index,
                "track_id": track_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "center": [float(center[0]), float(center[1])],
                "size": [float(size[0]), float(size[1])],
                "size_x": float(size[0]),
                "size_y": float(size[1]),
                "detection_confidence": float(confidences[index]),
                "pose_confidence": pose_confidence,
                "keypoints": None if points is None else points.tolist(),
                "keypoint_confidences": (
                    None if point_confidences is None else point_confidences.tolist()
                ),
                "inside_teacher_zone": teacher_zone_membership(center[0], center[1], zone),
            }
        )
    return observations


def extract_video(
    model: YOLO,
    source_path: Path,
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, dict, dict]:
    """Track one clip, assign stable roles, then pool all valid student tracks."""
    with np.load(source_path, allow_pickle=False) as archive:
        frames_rgb = archive["frames"]
    if frames_rgb.shape != (8, 640, 640, 3) or frames_rgb.dtype != np.uint8:
        raise ValueError(
            f"frames are {frames_rgb.shape}/{frames_rgb.dtype}; expected "
            "(8, 640, 640, 3)/uint8"
        )

    reset_tracker_state(model)
    zone = (args.zone_x_min, args.zone_x_max, args.zone_y_min, args.zone_y_max)
    frame_observations: list[list[dict]] = []
    tracks: dict[int, list[dict]] = defaultdict(list)
    for frame_index, frame_rgb in enumerate(frames_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = model.track(
            source=frame_bgr,
            persist=True,
            tracker=args.tracker,
            conf=args.confidence,
            classes=[0],
            device=device,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"YOLO.track returned {len(results)} results for one frame")
        observations = result_observations(results[0], frame_index, zone)
        frame_observations.append(observations)
        for observation in observations:
            if observation["track_id"] is not None:
                tracks[int(observation["track_id"])].append(observation)

    summaries = summarize_tracks(tracks, len(frames_rgb), args.teacher_score_weights)
    teacher_track_id, assignments = assign_track_roles(
        summaries,
        minimum_teacher_score=args.minimum_teacher_score,
        minimum_teacher_coverage=args.minimum_teacher_coverage,
        minimum_teacher_detection_confidence=args.minimum_teacher_detection_confidence,
        minimum_student_coverage=args.minimum_student_coverage,
        ambiguous_zone_fraction=args.ambiguous_zone_fraction,
    )

    zone_target = [
        (args.zone_x_min + args.zone_x_max) / 2.0,
        (args.zone_y_min + args.zone_y_max) / 2.0,
    ]
    teacher_summary = summaries.get(teacher_track_id, {})
    for observations in frame_observations:
        teacher_observation = next(
            (item for item in observations if item.get("track_id") == teacher_track_id),
            None,
        )
        if teacher_observation is not None:
            target = teacher_observation["center"]
            target_source = "teacher_observation"
        elif teacher_track_id is not None:
            target = [teacher_summary["mean_x"], teacher_summary["mean_y"]]
            target_source = "teacher_track_mean"
        else:
            target = zone_target
            target_source = "instruction_zone_center"

        for observation in observations:
            value, reliability, method, vector = compute_instruction_alignment_proxy(
                observation["center"],
                target,
                observation["keypoints"],
                observation["keypoint_confidences"],
                minimum_confidence=args.minimum_keypoint_confidence,
            )
            observation["instruction_target"] = [float(target[0]), float(target[1])]
            observation["instruction_target_source"] = target_source
            observation["instruction_alignment_proxy"] = value
            observation["instruction_alignment_reliability"] = reliability
            observation["orientation_method"] = method
            observation["orientation_vector"] = vector
            observation["orientation_reliability"] = reliability
            if observation["track_id"] is None:
                observation["role"] = "unknown"
                observation["role_confidence"] = 0.0
                observation["exclusion_reason"] = "missing_track_id"
            else:
                observation.update(assignments[int(observation["track_id"])])

    summaries = summarize_tracks(tracks, len(frames_rgb), args.teacher_score_weights)
    for track_id, summary in summaries.items():
        summary.update(assignments[track_id])

    descriptors = np.stack(
        [
            build_frame_descriptor(
                observations,
                summaries,
                assignments,
                teacher_track_id,
                close_distance=args.close_pair_distance,
                alignment_threshold=args.alignment_threshold,
            )
            for observations in frame_observations
        ]
    ).astype(np.float32)
    if descriptors.shape != TRACK_INTERACTION_SHAPE:
        raise ValueError(
            f"invalid interaction output shape {descriptors.shape}; expected "
            f"{TRACK_INTERACTION_SHAPE}"
        )
    if not np.isfinite(descriptors).all():
        raise ValueError("interaction output contains NaN or Inf")

    student_summaries = [
        summary for summary in summaries.values() if summary["role"] == "student"
    ]
    alignment_reliabilities = [
        observation["instruction_alignment_reliability"]
        for observations in frame_observations
        for observation in observations
        if observation.get("role") == "student"
    ]
    diagnostics = {
        "detected_track_count": len(summaries),
        "valid_student_track_count": len(student_summaries),
        "teacher_identified": teacher_track_id is not None,
        "mean_student_track_coverage": (
            float(np.mean([item["coverage"] for item in student_summaries]))
            if student_summaries
            else 0.0
        ),
        "mean_student_alignment_reliability": (
            float(np.mean(alignment_reliabilities))
            if alignment_reliabilities
            else 0.0
        ),
        "unknown_track_count": int(
            sum(summary["role"] == "unknown" for summary in summaries.values())
        ),
        "untracked_detection_count": int(
            sum(
                observation["track_id"] is None
                for observations in frame_observations
                for observation in observations
            )
        ),
    }
    details = {
        "video": str(source_path),
        "feature_schema": TRACK_INTERACTION_FEATURE_SCHEMA,
        "feature_columns": list(TRACK_INTERACTION_COLUMNS),
        "track_id_scope": "local_to_this_video_only",
        "tracking_warning": (
            "Eight sparsely sampled frames are tracked. IDs are associations, not "
            "verified person identities."
        ),
        "teacher_zone": {
            "x_min": args.zone_x_min,
            "x_max": args.zone_x_max,
            "y_min": args.zone_y_min,
            "y_max": args.zone_y_max,
        },
        "teacher_track_id": teacher_track_id,
        "tracks": {
            str(track_id): summary for track_id, summary in sorted(summaries.items())
        },
        "frames": [
            {
                "frame_index": frame_index,
                "detections": observations,
                "descriptor": descriptors[frame_index].tolist(),
            }
            for frame_index, observations in enumerate(frame_observations)
        ],
        "diagnostics": diagnostics,
    }
    return descriptors, details, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index")
    parser.add_argument("--zone_x_min", type=float, default=0.0)
    parser.add_argument("--zone_x_max", type=float, default=0.27)
    parser.add_argument("--zone_y_min", type=float, default=0.0)
    parser.add_argument("--zone_y_max", type=float, default=1.0)
    parser.add_argument(
        "--teacher_score_weights", type=float, nargs=3, default=(0.70, 0.20, 0.10)
    )
    parser.add_argument("--minimum_teacher_score", type=float, default=0.60)
    parser.add_argument("--minimum_teacher_coverage", type=float, default=0.50)
    parser.add_argument(
        "--minimum_teacher_detection_confidence", type=float, default=0.45
    )
    parser.add_argument("--minimum_student_coverage", type=float, default=0.25)
    parser.add_argument("--ambiguous_zone_fraction", type=float, default=0.50)
    parser.add_argument("--minimum_keypoint_confidence", type=float, default=0.30)
    parser.add_argument("--alignment_threshold", type=float, default=0.55)
    parser.add_argument("--close_pair_distance", type=float, default=0.15)
    parser.add_argument("--save_track_details", action="store_true")
    parser.add_argument("--track_details_dir", type=Path, default=DEFAULT_TRACK_DETAILS_DIR)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    for name in (
        "zone_x_min",
        "zone_x_max",
        "zone_y_min",
        "zone_y_max",
        "minimum_teacher_score",
        "minimum_teacher_coverage",
        "minimum_teacher_detection_confidence",
        "minimum_student_coverage",
        "ambiguous_zone_fraction",
        "minimum_keypoint_confidence",
        "alignment_threshold",
    ):
        _validate_fraction(f"--{name}", getattr(args, name))
    if args.zone_x_min > args.zone_x_max or args.zone_y_min > args.zone_y_max:
        raise ValueError("teacher-zone minimums cannot exceed maximums")
    if args.close_pair_distance <= 0.0:
        raise ValueError("--close_pair_distance must be positive")
    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("--max_videos must be positive")


def aggregate_diagnostics(items: list[dict]) -> dict:
    if not items:
        return {
            "mean_detected_tracks_per_video": None,
            "mean_valid_student_tracks_per_video": None,
            "teacher_identified_percentage": None,
            "mean_student_track_coverage": None,
            "mean_student_alignment_reliability": None,
            "unknown_track_count": 0,
            "unknown_track_percentage": None,
            "untracked_detection_count": 0,
        }
    detected_tracks = sum(item["detected_track_count"] for item in items)
    unknown_tracks = sum(item["unknown_track_count"] for item in items)
    return {
        "mean_detected_tracks_per_video": float(
            np.mean([item["detected_track_count"] for item in items])
        ),
        "mean_valid_student_tracks_per_video": float(
            np.mean([item["valid_student_track_count"] for item in items])
        ),
        "teacher_identified_percentage": float(
            100.0 * np.mean([item["teacher_identified"] for item in items])
        ),
        "mean_student_track_coverage": float(
            np.mean([item["mean_student_track_coverage"] for item in items])
        ),
        "mean_student_alignment_reliability": float(
            np.mean([item["mean_student_alignment_reliability"] for item in items])
        ),
        "unknown_track_count": int(unknown_tracks),
        "unknown_track_percentage": (
            float(100.0 * unknown_tracks / detected_tracks) if detected_tracks else None
        ),
        "untracked_detection_count": int(
            sum(item["untracked_detection_count"] for item in items)
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_DIAGNOSTIC_OUTPUT_DIR
            if args.max_videos is not None
            else DEFAULT_OUTPUT_DIR
        )
    output_dir = output_dir.expanduser().resolve()
    args.output_dir = output_dir
    args.track_details_dir = args.track_details_dir.expanduser().resolve()
    tasks = collect_inputs(input_dir)
    if args.max_videos is not None:
        tasks = tasks[: args.max_videos]
    if not tasks:
        raise FileNotFoundError(f"No preprocessed .npz files found under {input_dir}")

    model_reference = resolve_model_reference(args.model)
    model = YOLO(model_reference)
    model_reference = resolve_model_reference(
        str(getattr(model, "ckpt_path", model_reference))
    )
    args.tracker = resolve_model_reference(args.tracker)
    requested_provenance = interaction_provenance(args, model_reference)
    manifest_name = (
        "diagnostic_manifest.json"
        if args.max_videos is not None
        else "extraction_manifest.json"
    )
    manifest_path = output_dir / manifest_name
    existing_manifest = None
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open(encoding="utf-8") as file:
            existing_manifest = json.load(file)

    existing_destinations = [
        output_dir / split / category / f"{source_path.stem}.npy"
        for split, category, source_path in tasks
    ]
    if any(path.exists() for path in existing_destinations) and not args.overwrite:
        validate_reusable_provenance(existing_manifest, requested_provenance)
    if args.overwrite:
        manifest_path.unlink(missing_ok=True)

    reusable = [reusable_output(path) for path in existing_destinations]
    if all(reusable) and not args.overwrite and not args.save_track_details:
        summary = {
            "input_videos": len(tasks),
            "processed_videos": 0,
            "skipped_existing_videos": len(tasks),
            "failed_videos": 0,
            "diagnostics": (existing_manifest or {}).get("summary", {}).get(
                "diagnostics"
            ),
        }
        print(json.dumps(summary, indent=2))
        print(f"Preserved existing interaction provenance: {manifest_path}")
        return

    print("=" * 76)
    print("Extracting 40-D Track-Aware Role-Aware Interaction Features")
    print(f"Model: {model_reference}")
    print(f"Tracker: {args.tracker} | device: {select_device(args.device)}")
    print(f"Output: {output_dir}")
    print(f"Videos: {len(tasks)} | diagnostic subset: {args.max_videos is not None}")
    print("No area-ranked person slots; teacher is optional; all valid students are pooled.")
    print("=" * 76)

    device = select_device(args.device)
    processed = 0
    skipped = 0
    failures = []
    diagnostics = []
    for split, category, source_path in tqdm(tasks, desc="Track-aware interaction"):
        destination = output_dir / split / category / f"{source_path.stem}.npy"
        detail_path = args.track_details_dir / split / category / f"{source_path.stem}.json"
        if reusable_output(destination) and not args.overwrite:
            if not args.save_track_details or detail_path.is_file():
                skipped += 1
                continue
        try:
            matrix, details, video_diagnostics = extract_video(
                model, source_path, args, device
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, matrix)
            if args.save_track_details:
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                with detail_path.open("w", encoding="utf-8") as file:
                    json.dump(details, file, indent=2)
            diagnostics.append(video_diagnostics)
            processed += 1
        except Exception as exc:
            failures.append(f"{source_path}: {type(exc).__name__}: {exc}")

    run_summary = {
        "input_videos": len(tasks),
        "processed_videos": processed,
        "skipped_existing_videos": skipped,
        "failed_videos": len(failures),
        "diagnostics_scope": (
            "all_input_videos" if skipped == 0 else "processed_videos_only"
        ),
        "diagnostics": aggregate_diagnostics(diagnostics),
    }
    print(json.dumps(run_summary, indent=2))
    if failures:
        manifest_path.unlink(missing_ok=True)
        for failure in failures[:20]:
            print(f"  - {failure}")
        raise RuntimeError(
            f"Interaction extraction failed for {len(failures)} videos; no "
            "descriptors or roles were silently fabricated."
        )

    manifest = {**requested_provenance, "summary": run_summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved interaction provenance: {manifest_path}")


if __name__ == "__main__":
    main()
