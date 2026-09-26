"""Snapshot selected lecture matrices without decoding or extracting videos."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_schema import BEHAVIORAL_FEATURE_SCHEMA, BEHAVIORAL_SHAPE
from src.models.dataset import load_feature_manifest

LABELS = {"low": 0, "mid": 1, "high": 2}
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_keys(selected_dir: Path) -> set[str]:
    keys = set()
    for path in selected_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        relative = path.relative_to(selected_dir)
        if len(relative.parts) != 2 or relative.parts[0].lower() not in LABELS:
            raise ValueError(f"Expected class/video.mp4 under selection: {relative}")
        key = relative.as_posix().lower()
        if key in keys:
            raise ValueError(f"Duplicate selected class/filename: {key}")
        keys.add(key)
    if not keys:
        raise ValueError(f"No selected videos found under {selected_dir}")
    return keys


def selection_file_keys(path: Path) -> set[str]:
    """Read the portable, class-qualified selection without requiring videos."""
    keys = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        relative = Path(line)
        if (relative.is_absolute() or len(relative.parts) != 2 or
                relative.parts[0].lower() not in LABELS or relative.suffix.lower() != ".mp4"):
            raise ValueError(f"Invalid class/video.mp4 key at {path}:{line_number}")
        key = relative.as_posix().lower()
        if key in keys:
            raise ValueError(f"Duplicate selected class/filename: {key}")
        keys.add(key)
    if not keys:
        raise ValueError(f"No selected clips in {path}")
    return keys


def validate_source_matrix(record: dict) -> dict:
    path = Path(record["source_matrix"])
    data = path.read_bytes()
    matrix = np.load(io.BytesIO(data), allow_pickle=False)
    if matrix.shape != BEHAVIORAL_SHAPE or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix shape or non-finite values: {path}")
    return dict(record, sha256=hashlib.sha256(data).hexdigest())


def prepare_subset(selected_dir: Path, source_dir: Path, csv_dir: Path,
                   output_dir: Path, expected_count: int | None = None,
                   selection_file: Path | None = None) -> dict:
    selected_dir, source_dir, csv_dir, output_dir = (
        Path(p).expanduser().resolve() for p in
        (selected_dir, source_dir, csv_dir, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    selection_file = Path(selection_file).expanduser().resolve() if selection_file else None
    selected = selection_file_keys(selection_file) if selection_file else selection_keys(selected_dir)
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} selected videos, found {len(selected)}")
    source_manifest = load_feature_manifest(
        source_dir, expected_schema=BEHAVIORAL_FEATURE_SCHEMA,
        expected_shape=BEHAVIORAL_SHAPE,
    )
    records, seen = [], set()
    split_lines = {split: [] for split in SPLITS}
    csv_hashes = {}
    for split in SPLITS:
        csv_path = csv_dir / f"{split}.csv"
        csv_hashes[split] = sha256(csv_path)
        for line_number, line in enumerate(csv_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                video_path, label_text = line.rsplit(maxsplit=1)
                label = int(label_text)
                path = Path(video_path)
                category = path.parent.name.lower()
                if LABELS[category] != label or path.suffix.lower() != ".mp4":
                    raise ValueError("Class and label disagree")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid record at {csv_path}:{line_number}: {line}") from exc
            key = f"{category}/{path.name.lower()}"
            if key in seen:
                raise ValueError(f"Duplicate class/filename across split records: {key}")
            seen.add(key)
            if key not in selected:
                continue
            matrix_name = f"{path.stem}_label{label}.npy"
            source_path = source_dir / split / matrix_name
            records.append({"key": key, "split": split, "label": label,
                            "matrix_name": matrix_name,
                            "source_matrix": str(source_path)})
            split_lines[split].append(line)
    missing = selected - {record["key"] for record in records}
    if missing:
        raise ValueError(f"Selected videos absent from source splits: {sorted(missing)}")
    counts = {
        split: {category: sum(r["split"] == split and r["label"] == label for r in records)
                for category, label in LABELS.items()}
        for split in SPLITS
    }
    if any(not count for row in counts.values() for count in row.values()):
        raise ValueError(f"Each split must contain all three classes: {counts}")
    print(f"Validating {len(records)} selected source matrices (no extraction)...", flush=True)
    # Bounded concurrent reads also allow cloud-backed files to hydrate together.
    with ThreadPoolExecutor(max_workers=8) as executor:
        checked = []
        for record in executor.map(validate_source_matrix, records):
            checked.append(record)
            if len(checked) % 25 == 0 or len(checked) == len(records):
                print(f"Validated {len(checked)}/{len(records)} matrices", flush=True)
        records = checked
    selection = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "original_split_membership_filtered_by_activity",
        "limitation": "Preliminary fixed-setup baseline, not recording-held-out evaluation.",
        "selected_video_dir": None if selection_file else str(selected_dir),
        "selection_file": str(selection_file) if selection_file else None,
        "selection_file_sha256": sha256(selection_file) if selection_file else None,
        "source_matrix_dir": str(source_dir),
        "source_manifest_sha256": sha256(source_dir / "build_manifest.json"),
        "source_csv_sha256": csv_hashes, "total_videos": len(records),
        "class_counts": counts, "records": records,
    }
    # Publish only a fully checked snapshot. Never mix with stale destination files.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lecture-subset-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "dataset"
        (staging / "splits").mkdir(parents=True)
        for split in SPLITS:
            (staging / "matrices" / split).mkdir(parents=True)
            (staging / "splits" / f"{split}.csv").write_text(
                "\n".join(split_lines[split]) + "\n", encoding="utf-8"
            )
        for record in records:
            target = staging / "matrices" / record["split"] / record["matrix_name"]
            shutil.copy2(record["source_matrix"], target)
            if sha256(target) != record["sha256"]:
                raise ValueError(f"Copy verification failed: {target}")
        selection["filtered_csv_sha256"] = {
            split: sha256(staging / "splits" / f"{split}.csv") for split in SPLITS
        }
        selection_path = staging / "selection_manifest.json"
        selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        manifest = deepcopy(source_manifest)
        manifest["split_counts"] = {split: sum(counts[split].values()) for split in SPLITS}
        manifest["total_videos"] = len(records)
        manifest["subset"] = {"protocol": selection["protocol"],
                              "selection_manifest_sha256": sha256(selection_path)}
        (staging / "matrices" / "build_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verify_subset(staging)
        if output_dir.exists():
            raise FileExistsError(f"Destination appeared during preparation: {output_dir}")
        staging.rename(output_dir)
    return selection


def verify_subset(dataset_dir: Path) -> dict:
    """Reject stale/altered files before starting a lecture-only run."""
    dataset_dir = Path(dataset_dir).resolve()
    selection_path = dataset_dir / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    manifest = load_feature_manifest(
        dataset_dir / "matrices", expected_schema=BEHAVIORAL_FEATURE_SCHEMA,
        expected_shape=BEHAVIORAL_SHAPE,
    )
    if manifest.get("subset", {}).get("selection_manifest_sha256") != sha256(selection_path):
        raise ValueError("Selection manifest does not match matrix provenance")
    records = selection["records"]
    if len(records) != selection["total_videos"] or len({r["key"] for r in records}) != len(records):
        raise ValueError("Invalid selection record count or duplicate clip keys")
    if manifest["total_videos"] != len(records):
        raise ValueError("Matrix manifest total does not match selection")
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        expected = {r["matrix_name"] for r in selected}
        actual = {p.name for p in (dataset_dir / "matrices" / split).glob("*.npy")}
        if actual != expected or len(expected) != len(selected):
            raise ValueError(f"Matrix membership mismatch in {split}: missing={expected-actual}, extra={actual-expected}")
        counts = Counter(r["label"] for r in selected)
        if (manifest["split_counts"][split] != len(selected) or
                selection["class_counts"][split] != {c: counts[label] for c, label in LABELS.items()}):
            raise ValueError(f"Count mismatch in {split}")
        if sha256(dataset_dir / "splits" / f"{split}.csv") != selection["filtered_csv_sha256"][split]:
            raise ValueError(f"Filtered split list changed: {split}")
        for record in selected:
            path = dataset_dir / "matrices" / split / record["matrix_name"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Matrix content changed: {path}")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_dir", type=Path, default=PROJECT_ROOT / "lecture_only_videos")
    parser.add_argument("--selection_file", type=Path,
                        help="Use a class/video.mp4 text list instead of scanning selected_dir")
    parser.add_argument("--source_dir", type=Path, default=PROJECT_ROOT / "feature_matrices_behavioral")
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "experiments/lecture_only/dataset")
    parser.add_argument("--expected_count", type=int, default=308)
    args = parser.parse_args()
    result = prepare_subset(args.selected_dir, args.source_dir, args.csv_dir,
                            args.output_dir, args.expected_count, args.selection_file)
    print(json.dumps({"total_videos": result["total_videos"], "class_counts": result["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
