"""Audit exact paths and user-supplied session/golden-pair split groups."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_records(csv_dir: Path) -> list[dict]:
    records = []
    for split in SPLITS:
        path = csv_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required split file not found: {path}")
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed split row at {path}:{line_number}")
            video_path, label_text = parts
            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at {path}:{line_number}: {label_text!r}"
                ) from exc
            if label not in (0, 1, 2):
                raise ValueError(f"Label outside 0/1/2 at {path}:{line_number}")
            records.append(
                {
                    "split": split,
                    "video_path": video_path,
                    "label": label,
                }
            )
    return records


def load_group_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Group manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"video_path", "session_id", "golden_pair_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Group manifest must have columns: video_path,session_id,golden_pair_id"
            )
        mapping = {}
        for line_number, row in enumerate(reader, start=2):
            video_path = (row.get("video_path") or "").strip()
            session_id = (row.get("session_id") or "").strip()
            golden_pair_id = (row.get("golden_pair_id") or "").strip()
            if not video_path or not session_id:
                raise ValueError(
                    f"Missing video_path or session_id at {path}:{line_number}"
                )
            if video_path in mapping:
                raise ValueError(f"Duplicate video_path in group manifest: {video_path}")
            mapping[video_path] = {
                "session_id": session_id,
                "golden_pair_id": golden_pair_id,
            }
    return mapping


def audit_split_integrity(csv_dir: Path, group_manifest: Path) -> dict:
    """Fail if an exact video, session, or golden-pair group crosses splits."""
    csv_dir = csv_dir.expanduser().resolve()
    group_manifest = group_manifest.expanduser().resolve()
    records = read_split_records(csv_dir)
    path_splits: dict[str, set[str]] = defaultdict(set)
    path_labels: dict[str, set[int]] = defaultdict(set)
    path_occurrences: dict[str, int] = defaultdict(int)
    for record in records:
        path_splits[record["video_path"]].add(record["split"])
        path_labels[record["video_path"]].add(record["label"])
        path_occurrences[record["video_path"]] += 1
    duplicate_paths = {
        path: {
            "occurrences": path_occurrences[path],
            "splits": sorted(path_splits[path]),
        }
        for path in path_occurrences
        if path_occurrences[path] > 1
    }
    inconsistent_labels = {
        path: sorted(labels) for path, labels in path_labels.items() if len(labels) > 1
    }
    if duplicate_paths or inconsistent_labels:
        raise ValueError(
            "Exact video paths or labels cross splits: "
            f"duplicate_paths={dict(list(duplicate_paths.items())[:5])}, "
            f"inconsistent_labels={dict(list(inconsistent_labels.items())[:5])}"
        )

    groups = load_group_manifest(group_manifest)
    missing = sorted(set(path_splits) - set(groups))
    if missing:
        raise ValueError(
            f"Group manifest is missing {len(missing)} dataset videos; examples: "
            f"{missing[:5]}"
        )
    session_splits: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for video_path, splits in path_splits.items():
        split = next(iter(splits))
        group = groups[video_path]
        session_splits[group["session_id"]].add(split)
        if group["golden_pair_id"]:
            pair_splits[group["golden_pair_id"]].add(split)
    session_leaks = {
        key: sorted(value) for key, value in session_splits.items() if len(value) > 1
    }
    pair_leaks = {
        key: sorted(value) for key, value in pair_splits.items() if len(value) > 1
    }
    if session_leaks or pair_leaks:
        raise ValueError(
            "Session or golden-pair leakage detected: "
            f"sessions={dict(list(session_leaks.items())[:5])}, "
            f"golden_pairs={dict(list(pair_leaks.items())[:5])}"
        )

    report = {
        "status": "verified_no_cross_split_groups",
        "record_count": len(records),
        "unique_video_path_count": len(path_splits),
        "session_count": len(session_splits),
        "golden_pair_count": len(pair_splits),
        "split_counts": {
            split: sum(record["split"] == split for record in records)
            for split in SPLITS
        },
        "csv_sha256": {
            split: sha256_file(csv_dir / f"{split}.csv") for split in SPLITS
        },
        "group_manifest_sha256": sha256_file(group_manifest),
    }
    return report


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2)
