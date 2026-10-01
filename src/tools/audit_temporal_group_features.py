#!/usr/bin/env python3
"""Audit stored interaction/affect quality by frozen temporal group."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_schema import AFFECT_COLUMNS, INTERACTION_COLUMNS
from src.data.prepare_lecture_subset import verify_subset


VIEW_PATTERN = re.compile(r"^view(\d+)\.mp4$", re.IGNORECASE)
QUALITY_FIELDS = (
    "student_count_norm",
    "mean_face_visibility",
    "hidden_face_ratio",
    "posture_observation_ratio",
    "matched_center_displacement_mean_x10",
    "matched_center_displacement_std_x10",
    "matched_stillness_ratio",
    "matched_high_motion_ratio",
    "speaker_in_zone",
    "matched_speaker_displacement_x10",
    "orientation_alignment",
    "affect_reliability",
)


def assign_episode(record: dict, episodes: list[dict]) -> str:
    category, filename = record["key"].split("/", 1)
    match = VIEW_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Cannot parse view ID from {record['key']}")
    number = int(match.group(1))
    matches = [episode for episode in episodes
               if episode["class"].lower() == category
               and episode["start_view"] <= number <= episode["end_view"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one temporal group for {record['key']}; found {len(matches)}")
    return matches[0]["episode_id"]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def pooled_effect(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    variance = ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    if variance <= 1e-12:
        return 0.0
    return float((x.mean() - y.mean()) / math.sqrt(variance))


def run(args: argparse.Namespace) -> dict:
    base_dir = args.base_dataset.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    base = verify_subset(base_dir)
    protocol = json.loads(protocol_path.read_text())
    episodes = protocol["episodes"]
    columns = list(INTERACTION_COLUMNS) + list(AFFECT_COLUMNS)
    rows = []

    for record in base["records"]:
        group = assign_episode(record, episodes)
        matrix_path = base_dir / "matrices" / record["split"] / record["matrix_name"]
        matrix = np.load(matrix_path, allow_pickle=False)
        feature_means = matrix.mean(axis=0)
        row = {
            "key": record["key"], "label": record["label"], "temporal_group": group,
            **{name: float(value) for name, value in zip(columns, feature_means)},
        }
        rows.append(row)
    if len(rows) != base["total_videos"]:
        raise ValueError(f"Expected {base['total_videos']} audited clips, found {len(rows)}")

    numeric_fields = columns
    groups = {}
    for episode in episodes:
        group_id = episode["episode_id"]
        selected = [row for row in rows if row["temporal_group"] == group_id]
        if not selected:
            raise ValueError(f"No selected clips in {group_id}")
        groups[group_id] = {
            "class": episode["class"], "clips": len(selected),
            "metrics": {field: dict(zip(("mean", "sample_sd"),
                                         mean_std([row[field] for row in selected])))
                        for field in numeric_fields},
        }

    h04 = [row for row in rows if row["temporal_group"] == "H04"]
    other_high = [row for row in rows if row["label"] == 2 and row["temporal_group"] != "H04"]
    if not h04 or not other_high:
        raise ValueError("H04 and other High clips are required for the planned comparison")
    comparisons = []
    for field in numeric_fields:
        h_values, other_values = [row[field] for row in h04], [row[field] for row in other_high]
        comparisons.append({
            "metric": field,
            "h04_mean": float(np.mean(h_values)),
            "other_high_mean": float(np.mean(other_values)),
            "difference": float(np.mean(h_values) - np.mean(other_values)),
            "cohens_d": pooled_effect(h_values, other_values),
        })
    comparisons.sort(key=lambda item: abs(item["cohens_d"]), reverse=True)

    report = {
        "protocol": protocol.get("evaluation_protocol"),
        "audited_clips": len(rows),
        "temporal_groups": len(groups),
        "quality_fields": list(QUALITY_FIELDS),
        "groups": groups,
        "h04_vs_other_high": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temporal_group_feature_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (output_dir / "clip_feature_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["key"]))
    with (output_dir / "h04_vs_other_high.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(comparisons[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps({
        "audited_clips": len(rows), "temporal_groups": len(groups),
        "largest_h04_effects": comparisons[:12],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dataset", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_only/dataset")
    parser.add_argument("--protocol", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_temporal_group_cv/temporal_group_protocol.json")
    parser.add_argument("--output_dir", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_temporal_group_cv/diagnostics")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
