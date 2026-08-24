"""Extract track-aware group affect features from preprocessed RGB frames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Python 3.8 compatibility shim for BooleanOptionalAction
if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help=None, metavar=None):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith('--'):
                    _option_strings.append('--no-' + option_string[2:])
            super().__init__(option_strings=_option_strings, dest=dest, nargs=0, default=default, required=required, help=help)
        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self.option_strings:
                setattr(namespace, self.dest, not option_string.startswith('--no-'))
    argparse.BooleanOptionalAction = _BooleanOptionalAction

import numpy as np
import torch
from tqdm import tqdm

from affect_module import (
    EMOTION_NAMES,
    AffectModule,
    ByteTrackFaceTracker,
    HuggingFaceFERClassifier,
    RetinaFaceDetector,
    TorchScriptFERClassifier,
)
from feature_schema import AFFECT_COLUMNS, AFFECT_FEATURE_SCHEMA


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "preprocessed_data" / "yolov5_640x640"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "preprocessed_features" / "affect_track_features"


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_affect_module(args: argparse.Namespace) -> AffectModule:
    device = select_device(args.device)
    print(f"FER device: {device}")

    detector = RetinaFaceDetector(
        model_name=args.retinaface_model,
        model_root=args.insightface_root,
        det_size=(args.det_size, args.det_size),
        det_threshold=args.det_threshold,
        max_faces=args.max_faces,
    )

    if args.fer_backend == "huggingface":
        fer_model = HuggingFaceFERClassifier(
            model_id=args.fer_model_id,
            device=device,
            local_files_only=args.local_files_only,
        )
    else:
        if not args.fer_checkpoint:
            raise ValueError("--fer_checkpoint is required for --fer_backend torchscript")
        fer_model = TorchScriptFERClassifier(
            checkpoint_path=args.fer_checkpoint,
            device=device,
            input_size=args.fer_input_size,
            source_emotions=args.fer_source_emotions,
        )

    tracker = None
    if args.tracking:
        tracker = ByteTrackFaceTracker(
            track_high_threshold=args.track_high_threshold,
            track_low_threshold=args.track_low_threshold,
            new_track_threshold=args.new_track_threshold,
            track_buffer=args.track_buffer,
            match_threshold=args.track_match_threshold,
        )

    return AffectModule(
        detector=detector,
        fer_model=fer_model,
        tracker=tracker,
        use_tracking=args.tracking,
        expected_faces=args.expected_faces,
        emotion_momentum=args.emotion_momentum,
        missed_detection_decay=args.missed_detection_decay,
        max_feature_age=args.max_feature_age,
        aligned_face_size=args.fer_input_size,
    )


def collect_inputs(input_dir: Path) -> list[tuple[str, str, Path]]:
    tasks = []
    for split in ("train", "val", "test"):
        for category in ("low", "mid", "high"):
            source_dir = input_dir / split / category
            if source_dir.is_dir():
                tasks.extend(
                    (split, category, path)
                    for path in sorted(source_dir.glob("*.npz"))
                )
    return tasks


def extract_video(
    module: AffectModule,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    with np.load(source_path) as data:
        frames = data["frames"]
    if frames.shape != (8, 640, 640, 3) or frames.dtype != np.uint8:
        raise ValueError(
            f"{source_path} has frames {frames.shape}/{frames.dtype}; "
            "expected (8, 640, 640, 3)/uint8"
        )

    module.reset()
    rows = []
    reliability_values = []
    frame_details = []
    for frame_index, frame_rgb in enumerate(frames):
        affect_feat, reliability = module(frame_rgb)
        row = torch.cat(
            [affect_feat.detach().float().cpu(), reliability.detach().float().cpu().reshape(1)]
        )
        rows.append(row.numpy())
        reliability_values.append(float(reliability.item()))
        frame_details.append(
            {
                "frame_index": frame_index,
                "group_affect": {
                    name: float(affect_feat[index].item())
                    for index, name in enumerate(EMOTION_NAMES)
                },
                "group_reliability": float(reliability.item()),
                "tracks": module.last_observations,
            }
        )

    matrix = np.stack(rows).astype(np.float32)
    reliability_array = np.asarray(reliability_values, dtype=np.float32)
    if matrix.shape != (8, 8) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid output for {source_path}: shape={matrix.shape}")
    return matrix, reliability_array, frame_details


def write_manifest(args: argparse.Namespace, output_dir: Path, summary: dict) -> None:
    manifest = {
        "format_version": 2,
        "feature_schema": AFFECT_FEATURE_SCHEMA,
        "shape_per_video": [8, 8],
        "columns": list(AFFECT_COLUMNS),
        "detector": {
            "library": "insightface",
            "model": args.retinaface_model,
            "threshold": args.det_threshold,
            "input_size": args.det_size,
        },
        "tracker": {
            "enabled": args.tracking,
            "library": "ultralytics-bytetrack" if args.tracking else None,
            "high_threshold": args.track_high_threshold,
            "low_threshold": args.track_low_threshold,
            "new_track_threshold": args.new_track_threshold,
            "match_threshold": args.track_match_threshold,
            "track_buffer": args.track_buffer,
        },
        "fer": {
            "backend": args.fer_backend,
            "model": (
                args.fer_model_id
                if args.fer_backend == "huggingface"
                else os.path.abspath(args.fer_checkpoint)
            ),
            "source_emotions": list(args.fer_source_emotions),
        },
        "aggregation": {
            "expected_faces": args.expected_faces,
            "emotion_momentum": args.emotion_momentum,
            "missed_detection_decay": args.missed_detection_decay,
            "max_feature_age": args.max_feature_age,
        },
        "summary": summary,
    }
    with (output_dir / "extraction_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def manifest_matches_args(manifest: dict, args: argparse.Namespace) -> bool:
    """Return whether existing outputs have compatible recorded provenance."""
    if manifest.get("columns") != list(AFFECT_COLUMNS):
        return False
    detector = manifest.get("detector", {})
    tracker = manifest.get("tracker", {})
    fer = manifest.get("fer", {})
    expected_fer_model = (
        args.fer_model_id
        if args.fer_backend == "huggingface"
        else os.path.abspath(args.fer_checkpoint)
    )
    base_matches = (
        detector.get("library") == "insightface"
        and detector.get("model") == args.retinaface_model
        and detector.get("threshold") == args.det_threshold
        and detector.get("input_size") == args.det_size
        and tracker.get("enabled") == args.tracking
        and tracker.get("library")
        == ("ultralytics-bytetrack" if args.tracking else None)
        and tracker.get("high_threshold") == args.track_high_threshold
        and tracker.get("low_threshold") == args.track_low_threshold
        and tracker.get("match_threshold") == args.track_match_threshold
        and fer.get("backend") == args.fer_backend
        and fer.get("model") == expected_fer_model
        and fer.get("source_emotions") == list(args.fer_source_emotions)
    )
    if not base_matches:
        return False
    if manifest.get("format_version") == 1:
        # Version 1 predates recording the remaining defaults. It is accepted
        # only for backward-compatible reuse of the original track-aware run.
        return (
            args.new_track_threshold == 0.45
            and args.track_buffer == 8
            and args.expected_faces == 8
            and args.emotion_momentum == 0.60
            and args.missed_detection_decay == 0.35
            and args.max_feature_age == 1
        )
    aggregation = manifest.get("aggregation", {})
    return (
        manifest.get("feature_schema") == AFFECT_FEATURE_SCHEMA
        and manifest.get("shape_per_video") == [8, 8]
        and tracker.get("new_track_threshold") == args.new_track_threshold
        and tracker.get("track_buffer") == args.track_buffer
        and aggregation.get("expected_faces") == args.expected_faces
        and aggregation.get("emotion_momentum") == args.emotion_momentum
        and aggregation.get("missed_detection_decay") == args.missed_detection_decay
        and aggregation.get("max_feature_age") == args.max_feature_age
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RetinaFace + ByteTrack + PyTorch FER affect extraction"
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument("--retinaface_model", default="buffalo_s")
    parser.add_argument("--insightface_root", default="~/.insightface")
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--det_threshold", type=float, default=0.45)
    parser.add_argument("--max_faces", type=int, default=64)

    parser.add_argument(
        "--fer_backend",
        choices=("huggingface", "torchscript"),
        default="huggingface",
    )
    parser.add_argument(
        "--fer_model_id",
        default="abhilash88/face-emotion-detection",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--fer_checkpoint")
    parser.add_argument("--fer_input_size", type=int, default=224)
    parser.add_argument(
        "--fer_source_emotions",
        nargs=7,
        default=list(EMOTION_NAMES),
        metavar="EMOTION",
    )

    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--track_high_threshold", type=float, default=0.45)
    parser.add_argument("--track_low_threshold", type=float, default=0.10)
    parser.add_argument("--new_track_threshold", type=float, default=0.45)
    parser.add_argument("--track_match_threshold", type=float, default=0.80)
    parser.add_argument("--track_buffer", type=int, default=8)
    parser.add_argument("--expected_faces", type=int, default=8)
    parser.add_argument("--emotion_momentum", type=float, default=0.60)
    parser.add_argument("--missed_detection_decay", type=float, default=0.35)
    parser.add_argument("--max_feature_age", type=int, default=1)
    parser.add_argument(
        "--save_track_details",
        action="store_true",
        help="Save anonymous per-face track IDs, boxes, and emotion probabilities as JSON",
    )
    parser.add_argument(
        "--track_details_dir",
        type=Path,
        default=SCRIPT_DIR / "debug_validation" / "affect_tracks",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        help="Process only the first N videos (intended for diagnostics)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Extracting Track-Aware Affect Features")
    print("RetinaFace + ByteTrack + PyTorch FER")
    print("=" * 72)

    tasks = collect_inputs(args.input_dir)
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError("--max_videos must be positive")
        tasks = tasks[: args.max_videos]
    if not tasks:
        raise FileNotFoundError(f"No preprocessed .npz files found under {args.input_dir}")
    print(f"Found {len(tasks)} videos to process.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = None
    manifest_path = args.output_dir / "extraction_manifest.json"
    existing_outputs = any(
        (args.output_dir / split / category / f"{source_path.stem}.npy").is_file()
        for split, category, source_path in tasks
    )
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as file:
            existing_manifest = json.load(file)
    if existing_outputs and not args.overwrite:
        if existing_manifest is None or not manifest_matches_args(existing_manifest, args):
            raise RuntimeError(
                "Existing affect outputs have missing or incompatible provenance. "
                "Use a new --output_dir, or use --overwrite for a full extraction."
            )
    if (
        existing_outputs
        and args.overwrite
        and args.max_videos is not None
        and (existing_manifest is None or not manifest_matches_args(existing_manifest, args))
    ):
        raise RuntimeError(
            "A diagnostic subset cannot overwrite outputs from an incompatible "
            "configuration. Use a separate --output_dir."
        )
    if args.overwrite and args.max_videos is None:
        # A failed full overwrite must not leave an old manifest claiming that
        # the now-partially-replaced directory is internally consistent.
        manifest_path.unlink(missing_ok=True)
    module = build_affect_module(args)

    processed = 0
    skipped = 0
    reliability_values = []
    failures = []
    for split, category, source_path in tqdm(tasks, desc="Extracting affect"):
        destination_dir = args.output_dir / split / category
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{source_path.stem}.npy"
        details_path = (
            args.track_details_dir / split / category / f"{source_path.stem}.json"
        )
        details_missing = args.save_track_details and not details_path.exists()
        if destination_path.exists() and not args.overwrite and not details_missing:
            skipped += 1
            continue
        try:
            matrix, frame_reliability, frame_details = extract_video(module, source_path)
            np.save(destination_path, matrix)
            if args.save_track_details:
                details_path.parent.mkdir(parents=True, exist_ok=True)
                with details_path.open("w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "video": str(source_path),
                            "note": (
                                "Track IDs are anonymous, local to this video, and "
                                "do not identify a person across videos."
                            ),
                            "frames": frame_details,
                        },
                        file,
                        indent=2,
                    )
            reliability_values.extend(frame_reliability.tolist())
            processed += 1
        except Exception as exc:
            failures.append(f"{source_path}: {type(exc).__name__}: {exc}")

    summary = {
        "input_videos": len(tasks),
        "processed_videos": processed,
        "skipped_existing_videos": skipped,
        "failed_videos": len(failures),
        "mean_frame_reliability": (
            float(np.mean(reliability_values)) if reliability_values else None
        ),
        "zero_reliability_fraction": (
            float(np.mean(np.asarray(reliability_values) == 0.0))
            if reliability_values
            else None
        ),
    }
    if not reliability_values and existing_manifest is not None:
        previous_summary = existing_manifest.get("summary", {})
        summary["mean_frame_reliability"] = previous_summary.get(
            "mean_frame_reliability"
        )
        summary["zero_reliability_fraction"] = previous_summary.get(
            "zero_reliability_fraction"
        )
    print(json.dumps(summary, indent=2))
    if failures:
        print("Failures:")
        for failure in failures[:20]:
            print(f"  - {failure}")
        raise RuntimeError(
            f"Affect extraction failed for {len(failures)} videos; "
            "feature matrices were not silently fabricated."
        )
    if args.max_videos is None:
        write_manifest(args, args.output_dir, summary)
    else:
        print("Diagnostic subset run: preserved the full-dataset extraction manifest.")
    print(f"Saved track-aware affect features under {args.output_dir}")


if __name__ == "__main__":
    main()
