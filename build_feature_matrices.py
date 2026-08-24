"""Build auditable multi-branch matrices from scene, interaction, and affect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from feature_schema import (
    AFFECT_COLUMNS,
    AFFECT_FEATURE_SCHEMA,
    INTERACTION_FEATURE_SCHEMA,
    MULTI_BRANCH_FEATURE_SCHEMA,
    MULTI_BRANCH_SHAPE,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = SCRIPT_DIR / "preprocessed_features"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "feature_matrices"
STREAMS = {
    "scene": ("scene_features", (8, 576)),
    "interaction": ("interaction_features", (8, 32)),
    "track-aware affect": ("affect_track_features", (8, 8)),
}
LABEL_BY_CATEGORY = {"low": 0, "mid": 1, "high": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (8, 616) multi-branch feature matrices"
    )
    parser.add_argument("--feature_dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv_dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    return parser.parse_args()


def read_affect_manifest(feature_dir: Path) -> dict:
    manifest_path = feature_dir / "affect_track_features" / "extraction_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Track-aware affect manifest not found: {manifest_path}. Run "
            "extract_affect_features.py for the full dataset before building matrices."
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    # The first track-aware extraction run used format_version=1 before the
    # explicit schema key was added. Preserve those expensive extracted files
    # only when the recorded components prove they came from this pipeline.
    if manifest.get("feature_schema") is None:
        is_verified_track_v1 = (
            manifest.get("format_version") == 1
            and manifest.get("detector", {}).get("library") == "insightface"
            and manifest.get("tracker", {}).get("enabled") is True
            and manifest.get("tracker", {}).get("library") == "ultralytics-bytetrack"
            and manifest.get("fer", {}).get("backend") in {"huggingface", "torchscript"}
        )
        if is_verified_track_v1:
            manifest["feature_schema"] = AFFECT_FEATURE_SCHEMA
            manifest["shape_per_video"] = [8, 8]
            manifest["normalized_from_format_version"] = 1
    if manifest.get("feature_schema") != AFFECT_FEATURE_SCHEMA:
        raise ValueError(
            f"Unsupported affect schema {manifest.get('feature_schema')!r}; "
            f"expected {AFFECT_FEATURE_SCHEMA!r}. Old YuNet/HSEmotion features cannot "
            "be mixed with the new track-aware features."
        )
    if manifest.get("columns") != list(AFFECT_COLUMNS):
        raise ValueError(
            f"Affect column mismatch in {manifest_path}: {manifest.get('columns')}"
        )
    if manifest.get("shape_per_video") != [8, 8]:
        raise ValueError(
            f"Affect shape contract mismatch in {manifest_path}: "
            f"{manifest.get('shape_per_video')}"
        )
    return manifest


def read_interaction_manifest(feature_dir: Path) -> dict:
    manifest_path = feature_dir / "interaction_features" / "extraction_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Interaction manifest not found: {manifest_path}. Re-run the revised "
            "extract_interaction_features.py before building matrices."
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("feature_schema") != INTERACTION_FEATURE_SCHEMA:
        raise ValueError(
            f"Unsupported interaction schema {manifest.get('feature_schema')!r}; "
            f"expected {INTERACTION_FEATURE_SCHEMA!r}"
        )
    if manifest.get("shape_per_video") != [8, 32]:
        raise ValueError(
            f"Interaction shape contract mismatch in {manifest_path}: "
            f"{manifest.get('shape_per_video')}"
        )
    return manifest


def parse_csv_record(
    line: str,
    csv_path: Path,
    line_number: int,
) -> tuple[str, int, str, str]:
    parts = line.rsplit(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Malformed record at {csv_path}:{line_number}: {line!r}")
    video_path, raw_label = parts
    try:
        label = int(raw_label)
    except ValueError as exc:
        raise ValueError(
            f"Invalid label at {csv_path}:{line_number}: {raw_label!r}"
        ) from exc
    if label not in (0, 1, 2):
        raise ValueError(f"Label must be 0, 1, or 2 at {csv_path}:{line_number}")
    video = Path(video_path)
    if len(video.parts) < 2:
        raise ValueError(
            f"Video path must contain a category at {csv_path}:{line_number}: "
            f"{video_path!r}"
        )
    category = video.parent.name.lower()
    expected_label = LABEL_BY_CATEGORY.get(category)
    if expected_label is None:
        raise ValueError(
            f"Unknown engagement category {category!r} at {csv_path}:{line_number}"
        )
    if label != expected_label:
        raise ValueError(
            f"Category/label mismatch at {csv_path}:{line_number}: "
            f"{category!r} requires label {expected_label}, got {label}"
        )
    return video_path, label, video.stem, category


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_dir = args.csv_dir.expanduser().resolve()
    affect_manifest = read_affect_manifest(feature_dir)
    interaction_manifest = read_interaction_manifest(feature_dir)

    print("=" * 72)
    print("Building Multi-Branch Feature Matrices")
    print("Scene 576 + Interaction 32 + Track-Aware Affect 8 = 616 columns")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    build_manifest_path = output_dir / "build_manifest.json"
    build_manifest_path.unlink(missing_ok=True)

    total_processed = 0
    total_missing = 0
    split_counts: dict[str, int] = {}
    missing_by_stream = {name: 0 for name in STREAMS}
    missing_examples: list[str] = []

    for split in args.splits:
        csv_path = csv_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Required split file not found: {csv_path}")
        lines = [line.strip() for line in csv_path.read_text().splitlines() if line.strip()]
        destination_dir = output_dir / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        split_processed = 0
        seen_destinations: set[Path] = set()

        print(f"Processing split {split!r} ({len(lines)} videos)...")
        for line_number, line in enumerate(
            tqdm(lines, desc=f"Split {split}"), start=1
        ):
            video_path, label, video_name, category = parse_csv_record(
                line, csv_path, line_number
            )
            paths = {
                name: feature_dir / directory / split / category / f"{video_name}.npy"
                for name, (directory, _) in STREAMS.items()
            }
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                total_missing += 1
                for name in missing:
                    missing_by_stream[name] += 1
                if len(missing_examples) < 20:
                    missing_examples.append(f"{video_path}: {', '.join(missing)}")
                continue

            arrays = {}
            for name, path in paths.items():
                array = np.load(path, allow_pickle=False)
                expected_shape = STREAMS[name][1]
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{path} has shape {array.shape}; expected {expected_shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite feature value found in {path}")
                arrays[name] = array

            combined = np.concatenate(
                [arrays["scene"], arrays["interaction"], arrays["track-aware affect"]],
                axis=1,
            ).astype(np.float32, copy=False)
            if combined.shape != MULTI_BRANCH_SHAPE:
                raise AssertionError(f"Internal matrix shape error: {combined.shape}")
            destination = destination_dir / f"{video_name}_label{label}.npy"
            if destination in seen_destinations:
                raise ValueError(f"Duplicate output destination from CSV: {destination}")
            seen_destinations.add(destination)
            np.save(destination, combined)
            total_processed += 1
            split_processed += 1
        split_counts[split] = split_processed

    print("\n" + "=" * 72)
    print(f"Successfully built: {total_processed} matrices")
    print(f"Missing source feature sets: {total_missing}")
    for name, count in missing_by_stream.items():
        if count:
            print(f"  - {name}: {count} missing")
    print("=" * 72)
    if total_missing:
        examples = "\n".join(f"  - {item}" for item in missing_examples)
        raise RuntimeError(
            "Cannot publish an auditable matrix dataset while source features are "
            f"missing. Examples:\n{examples}"
        )

    manifest = {
        "format_version": 2,
        "feature_schema": MULTI_BRANCH_FEATURE_SCHEMA,
        "affect_schema": AFFECT_FEATURE_SCHEMA,
        "shape_per_video": list(MULTI_BRANCH_SHAPE),
        "streams": {
            "scene": {"columns": 576},
            "interaction": {"columns": 32},
            "affect": {"columns": 8, "names": list(AFFECT_COLUMNS)},
        },
        "affect_extraction": {
            key: affect_manifest.get(key)
            for key in ("feature_schema", "columns", "detector", "tracker", "fer")
        },
        "interaction_extraction": interaction_manifest,
        "split_counts": split_counts,
        "total_videos": total_processed,
    }
    with build_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved provenance manifest: {build_manifest_path}")


if __name__ == "__main__":
    main()
