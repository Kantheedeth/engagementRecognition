"""Extract the branch's 32-dimensional YOLO interaction descriptor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.data.feature_schema import INTERACTION_FEATURE_SCHEMA


DEFAULT_INPUT_DIR = PROJECT_ROOT / "preprocessed_data" / "yolov5_640x640"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "preprocessed_features" / "interaction_features"

# VFOA "instructional zone" geometric boundary.
ZONE_X_MIN = 0.0
ZONE_X_MAX = 0.27
ZONE_Y_MIN = 0.0
ZONE_Y_MAX = 1.0
MAX_STUDENTS_TRACKED = 5


def normalize_model_reference(value: str) -> str:
    """Normalize local model paths before comparing extraction provenance."""
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    script_relative = SCRIPT_DIR / candidate
    if not candidate.is_absolute() and script_relative.is_file():
        return str(script_relative.resolve())
    return value


def interaction_provenance(model: str, confidence: float) -> dict:
    """Return the generation settings that must match for reusable features."""
    return {
        "format_version": 1,
        "feature_schema": INTERACTION_FEATURE_SCHEMA,
        "shape_per_video": [8, 32],
        "model": normalize_model_reference(model),
        "confidence": confidence,
        "input_color": "RGB",
        "ultralytics_numpy_color": "BGR",
        "instruction_zone": {
            "x_min": ZONE_X_MIN,
            "x_max": ZONE_X_MAX,
            "y_min": ZONE_Y_MIN,
            "y_max": ZONE_Y_MAX,
        },
        "max_person_states": MAX_STUDENTS_TRACKED,
    }


def validate_reusable_provenance(existing: dict | None, requested: dict) -> None:
    """Reject existing arrays whose recorded generation settings are unknown."""
    if existing is None:
        raise RuntimeError(
            "Existing interaction features have no provenance manifest. Re-run with "
            "--overwrite instead of silently reusing unverifiable arrays."
        )

    comparable_existing = dict(existing)
    comparable_existing.pop("summary", None)
    comparable_existing["model"] = normalize_model_reference(
        str(comparable_existing.get("model", ""))
    )
    if comparable_existing != requested:
        raise RuntimeError(
            "Existing interaction features were generated with different or "
            "incomplete settings. Re-run with --overwrite to replace the full set; "
            "mixed-provenance features are not allowed."
        )


DEFAULT_TARGET = np.array([0.135, 0.50], dtype=np.float32)


def extract_frame_interaction_descriptor(boxes, keypoints=None) -> np.ndarray:
    """Return 7 group geometry + 5x5 area-sorted person-state values with real Zone Attention."""
    if len(boxes) == 0:
        return np.zeros(32, dtype=np.float32)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy()
    kpts = keypoints.xy.detach().cpu().numpy() if keypoints is not None else None
    kpts_conf = (
        keypoints.conf.detach().cpu().numpy()
        if keypoints is not None and keypoints.conf is not None
        else None
    )

    # 1. Identify teacher position in instruction zone if present
    teacher_pos = None
    for i, box in enumerate(xyxy):
        if int(classes[i]) != 0:
            continue
        cx = ((box[0] + box[2]) / 2.0) / 640.0
        cy = ((box[1] + box[3]) / 2.0) / 640.0
        if ZONE_X_MIN <= cx <= ZONE_X_MAX and ZONE_Y_MIN <= cy <= ZONE_Y_MAX:
            teacher_pos = np.array([cx, cy], dtype=np.float32)
            break

    target_pos = teacher_pos if teacher_pos is not None else DEFAULT_TARGET

    persons = []
    for i, box in enumerate(xyxy):
        if int(classes[i]) != 0:
            continue
        cx = ((box[0] + box[2]) / 2.0) / 640.0
        cy = ((box[1] + box[3]) / 2.0) / 640.0
        width = (box[2] - box[0]) / 640.0
        height = (box[3] - box[1]) / 640.0
        area = width * height

        # Target direction vector from student to instruction zone
        v_target = target_pos - np.array([cx, cy], dtype=np.float32)
        dist_target = float(np.linalg.norm(v_target))
        u_target = v_target / dist_target if dist_target > 1e-4 else np.zeros(2, dtype=np.float32)

        # Compute facing vector from pose keypoints
        zone_attention = 0.5  # default neutral orientation
        if cx <= ZONE_X_MAX:
            # Teacher inside the zone is 1.0
            zone_attention = 1.0
        elif kpts is not None and i < len(kpts):
            p_kpts = kpts[i]
            p_conf = kpts_conf[i] if kpts_conf is not None else np.ones(17, dtype=np.float32)

            nose = p_kpts[0] / 640.0
            l_eye, r_eye = p_kpts[1] / 640.0, p_kpts[2] / 640.0
            l_sh, r_sh = p_kpts[5] / 640.0, p_kpts[6] / 640.0

            # Method 1: Head vector (eyes midpoint to nose)
            v_facing = None
            if p_conf[0] > 0.3 and (p_conf[1] > 0.3 or p_conf[2] > 0.3):
                eye_mid = (
                    (l_eye + r_eye) / 2.0
                    if (p_conf[1] > 0.3 and p_conf[2] > 0.3)
                    else (l_eye if p_conf[1] > 0.3 else r_eye)
                )
                v_head = nose - eye_mid
                norm_h = float(np.linalg.norm(v_head))
                if norm_h > 1e-4:
                    v_facing = v_head / norm_h

            # Method 2: Torso vector (orthogonal to shoulders)
            if v_facing is None and p_conf[5] > 0.3 and p_conf[6] > 0.3:
                v_sh = r_sh - l_sh
                # Normal pointing forward
                v_torso = np.array([-v_sh[1], v_sh[0]], dtype=np.float32)
                norm_t = float(np.linalg.norm(v_torso))
                if norm_t > 1e-4:
                    v_facing = v_torso / norm_t

            if v_facing is not None and dist_target > 1e-4:
                cos_sim = float(np.dot(v_facing, u_target))
                zone_attention = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))
            else:
                zone_attention = 0.5

        persons.append((area, cx, cy, width, height, zone_attention))

    person_count = len(persons)
    if person_count == 0:
        return np.zeros(32, dtype=np.float32)

    # Compute student-only attention (students outside instruction zone)
    student_attentions = [p[5] for p in persons if p[1] > ZONE_X_MAX]
    if len(student_attentions) == 0:
        student_attentions = [p[5] for p in persons]

    mean_attention = float(np.mean(student_attentions))
    vfoa_ratio = float(np.mean([att >= 0.55 for att in student_attentions]))

    centers_x = [p[1] for p in persons]
    centers_y = [p[2] for p in persons]
    global_features = [
        person_count / 10.0,
        mean_attention,
        vfoa_ratio,
        float(np.mean(centers_x)),
        float(np.mean(centers_y)),
        float(np.std(centers_x)) if person_count > 1 else 0.0,
        float(np.std(centers_y)) if person_count > 1 else 0.0,
    ]

    persons.sort(key=lambda person: person[0], reverse=True)
    person_features = []
    for index in range(MAX_STUDENTS_TRACKED):
        if index < person_count:
            _, cx, cy, width, height, att = persons[index]
            person_features.extend([cx, cy, width, height, att])
        else:
            person_features.extend([0.0] * 5)

    descriptor = np.asarray(global_features + person_features, dtype=np.float32)
    if descriptor.shape != (32,) or not np.isfinite(descriptor).all():
        raise ValueError(f"Invalid interaction descriptor: {descriptor}")
    return descriptor


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
                    (split, category, path) for path in sorted(directory.glob("*.npz"))
                )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA index")
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0, 1]")
    tasks = collect_inputs(input_dir)
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError("--max_videos must be positive")
        tasks = tasks[: args.max_videos]
    if not tasks:
        raise FileNotFoundError(f"No preprocessed .npz files found under {input_dir}")

    device = select_device(args.device)
    print("=" * 72)
    print("Extracting 32-Dimensional Interaction Features")
    print(f"YOLO model: {args.model} | device: {device} | confidence: {args.confidence}")
    print(f"Instruction zone: x=[{ZONE_X_MIN}, {ZONE_X_MAX}], y=[{ZONE_Y_MIN}, {ZONE_Y_MAX}]")
    print(f"Found {len(tasks)} videos")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extraction_manifest.json"
    requested_provenance = interaction_provenance(args.model, args.confidence)
    existing_manifest = None
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open(encoding="utf-8") as file:
            existing_manifest = json.load(file)
    model = YOLO(args.model)
    processed = 0
    skipped = 0
    failures = []

    for split, category, source_path in tqdm(tasks, desc="Extracting interaction"):
        destination_dir = output_dir / split / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{source_path.stem}.npy"
        if destination_path.is_file() and not args.overwrite:
            existing = np.load(destination_path, allow_pickle=False)
            if existing.shape == (8, 32) and np.isfinite(existing).all():
                validate_reusable_provenance(existing_manifest, requested_provenance)
                skipped += 1
                continue
            print(f"Replacing incompatible existing feature: {destination_path}")

        try:
            with np.load(source_path, allow_pickle=False) as archive:
                frames_rgb = archive["frames"]
            if frames_rgb.shape != (8, 640, 640, 3) or frames_rgb.dtype != np.uint8:
                raise ValueError(
                    f"frames are {frames_rgb.shape}/{frames_rgb.dtype}; expected "
                    "(8, 640, 640, 3)/uint8"
                )
            # Ultralytics treats numpy-array sources as OpenCV BGR images.
            frames_bgr = [
                cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) for frame_rgb in frames_rgb
            ]
            results = model.predict(
                source=frames_bgr,
                conf=args.confidence,
                device=device,
                verbose=False,
            )
            descriptors = np.stack(
                [
                    extract_frame_interaction_descriptor(
                        result.boxes, result.keypoints
                    )
                    for result in results
                ]
            ).astype(np.float32)
            if descriptors.shape != (8, 32) or not np.isfinite(descriptors).all():
                raise ValueError(f"Invalid output matrix {descriptors.shape}")
            np.save(destination_path, descriptors)
            processed += 1
        except Exception as exc:
            failures.append(f"{source_path}: {type(exc).__name__}: {exc}")

    summary = {
        "input_videos": len(tasks),
        "processed_videos": processed,
        "skipped_existing_videos": skipped,
        "failed_videos": len(failures),
    }
    print(json.dumps(summary, indent=2))
    if failures:
        manifest_path.unlink(missing_ok=True)
        for failure in failures[:20]:
            print(f"  - {failure}")
        raise RuntimeError(
            f"Interaction extraction failed for {len(failures)} videos; no "
            "descriptors were silently fabricated."
        )

    if args.max_videos is None:
        if processed == 0 and skipped == len(tasks) and existing_manifest is not None:
            print(f"Preserved existing interaction provenance manifest: {manifest_path}")
        else:
            manifest = {**requested_provenance, "summary": summary}
            with manifest_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)
            print(f"Saved interaction provenance manifest: {manifest_path}")
    else:
        print("Diagnostic subset run: full-dataset manifest was not replaced.")


if __name__ == "__main__":
    main()
