"""Build auditable interaction + track-aware affect matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

try:
    from src.data.build_feature_matrices import (
        parse_csv_record,
        read_affect_manifest,
        read_interaction_manifest,
    )
    from src.data.feature_schema import (
        AFFECT_COLUMNS,
        AFFECT_FEATURE_SCHEMA,
        BEHAVIORAL_FEATURE_SCHEMA,
        BEHAVIORAL_SHAPE,
    )
except ImportError:
    from build_feature_matrices import (
        parse_csv_record,
        read_affect_manifest,
        read_interaction_manifest,
    )
    from feature_schema import (
        AFFECT_COLUMNS,
        AFFECT_FEATURE_SCHEMA,
        BEHAVIORAL_FEATURE_SCHEMA,
        BEHAVIORAL_SHAPE,
    )


DEFAULT_FEATURE_DIR = PROJECT_ROOT / "preprocessed_features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "feature_matrices_behavioral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build (8, 40) interaction + track-aware affect matrices"
    )
    parser.add_argument("--feature_dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_dir = args.csv_dir.expanduser().resolve()
    affect_manifest = read_affect_manifest(feature_dir)
    interaction_manifest = read_interaction_manifest(feature_dir)

    print("=" * 72)
    print("Building Pure Behavioral Feature Matrices")
    print("Interaction 32 + Track-Aware Affect 8 = 40 columns; scene excluded")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    build_manifest_path = output_dir / "build_manifest.json"
    build_manifest_path.unlink(missing_ok=True)
    total_processed = 0
    total_missing = 0
    split_counts: dict[str, int] = {}
    missing_by_stream = {"interaction": 0, "track-aware affect": 0}
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
                "interaction": (
                    feature_dir
                    / "interaction_features"
                    / split
                    / category
                    / f"{video_name}.npy"
                ),
                "track-aware affect": (
                    feature_dir
                    / "affect_track_features"
                    / split
                    / category
                    / f"{video_name}.npy"
                ),
            }
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                total_missing += 1
                for name in missing:
                    missing_by_stream[name] += 1
                if len(missing_examples) < 20:
                    missing_examples.append(f"{video_path}: {', '.join(missing)}")
                continue

            interaction = np.load(paths["interaction"], allow_pickle=False)
            affect = np.load(paths["track-aware affect"], allow_pickle=False)
            for path, array, expected_shape in (
                (paths["interaction"], interaction, (8, 32)),
                (paths["track-aware affect"], affect, (8, 8)),
            ):
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{path} has shape {array.shape}; expected {expected_shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite feature value found in {path}")

            matrix = np.concatenate([interaction, affect], axis=1).astype(
                np.float32, copy=False
            )
            if matrix.shape != BEHAVIORAL_SHAPE:
                raise AssertionError(f"Internal matrix shape error: {matrix.shape}")
            destination = destination_dir / f"{video_name}_label{label}.npy"
            if destination in seen_destinations:
                raise ValueError(f"Duplicate output destination from CSV: {destination}")
            seen_destinations.add(destination)
            np.save(destination, matrix)
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
            "Cannot publish an auditable behavioral dataset while source features "
            f"are missing. Examples:\n{examples}"
        )

    manifest = {
        "format_version": 2,
        "feature_schema": BEHAVIORAL_FEATURE_SCHEMA,
        "affect_schema": AFFECT_FEATURE_SCHEMA,
        "shape_per_video": list(BEHAVIORAL_SHAPE),
        "streams": {
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
