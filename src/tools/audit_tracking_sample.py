#!/usr/bin/env python3
"""Bounded stratified audit of raw tracking continuity metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_lecture_subset import verify_subset
from src.tools.audit_temporal_group_features import assign_episode


def parse_tracking_file(path: Path) -> dict:
    details = json.loads(path.read_text())
    samples = details["samples"]
    known_speakers = set(details.get("known_speaker_track_ids", []))
    listener_observations: dict[int, int] = {}
    listener_confidences = []
    listener_counts = []
    speaker_present = []
    for sample in samples:
        listeners = [person for person in sample["people"] if person["role"] == "listener"]
        listener_counts.append(len(listeners))
        speaker_present.append(any(person["track_id"] in known_speakers for person in sample["people"]))
        for person in listeners:
            listener_observations[person["track_id"]] = listener_observations.get(person["track_id"], 0) + 1
            listener_confidences.append(float(person["confidence"]))
    tracks = {track["track_id"]: track for track in details.get("tracks", [])}
    coverages = [count / len(samples) for count in listener_observations.values()]
    gaps = [tracks.get(track_id, {}).get("gap_returns", 0) for track_id in listener_observations]
    return {
        "speaker_target_ratio": float(np.mean([
            sample.get("target_source") == "speaker" for sample in samples
        ])),
        "speaker_track_coverage": float(np.mean(speaker_present)),
        "listener_count_mean": float(np.mean(listener_counts)),
        "listener_count_std": float(np.std(listener_counts)),
        "listener_detection_confidence": float(np.mean(listener_confidences)),
        "listener_unique_tracks": len(listener_observations),
        "listener_track_coverage_mean": float(np.mean(coverages)),
        "listener_long_track_ratio": float(np.mean([coverage >= 0.75 for coverage in coverages])),
        "listener_gap_returns_mean": float(np.mean(gaps)),
    }


def run(args: argparse.Namespace) -> dict:
    base = verify_subset(args.base_dataset)
    protocol = json.loads(args.protocol.read_text())
    grouped: dict[str, list[dict]] = {episode["episode_id"]: [] for episode in protocol["episodes"]}
    for record in base["records"]:
        grouped[assign_episode(record, protocol["episodes"])].append(record)
    selected = []
    for group, records in grouped.items():
        records = sorted(records, key=lambda record: record["key"])
        indices = sorted({0, len(records) - 1})
        for index in indices:
            selected.append((group, records[index]))

    completed, timed_out, failed = [], [], []
    for group, record in selected:
        category, filename = record["key"].split("/", 1)
        path = (args.tracks_root / record["split"] / category /
                filename.replace(".mp4", ".tracks.json"))
        command = [sys.executable, str(Path(__file__).resolve()), "--parse_one", str(path)]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=args.timeout_seconds, check=True)
            completed.append({"temporal_group": group, "key": record["key"],
                              **json.loads(result.stdout)})
        except subprocess.TimeoutExpired:
            timed_out.append({"temporal_group": group, "key": record["key"]})
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            failed.append({"temporal_group": group, "key": record["key"], "error": str(exc)})

    metric_names = []
    group_summary = {}
    if completed:
        metric_names = [key for key in completed[0]
                        if key not in ("temporal_group", "key")]
        for group in grouped:
            rows = [row for row in completed if row["temporal_group"] == group]
            group_summary[group] = {
                "requested": sum(item[0] == group for item in selected),
                "completed": len(rows),
                "metrics": {metric: float(np.mean([row[metric] for row in rows]))
                            for metric in metric_names} if rows else {},
            }
    report = {
        "sampling": "first and last selected clip in each temporal group",
        "timeout_seconds_per_file": args.timeout_seconds,
        "requested": len(selected), "completed": len(completed),
        "timed_out": timed_out, "failed": failed,
        "groups": group_summary, "clips": completed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("requested", "completed", "timed_out", "failed")}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dataset", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_only/dataset")
    parser.add_argument("--protocol", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_temporal_group_cv/temporal_group_protocol.json")
    parser.add_argument("--tracks_root", type=Path,
                        default=PROJECT_ROOT / "preprocessed_features/interaction_features")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_temporal_group_cv/diagnostics/tracking_sample_audit.json")
    parser.add_argument("--timeout_seconds", type=float, default=2.0)
    parser.add_argument("--parse_one", type=Path)
    args = parser.parse_args()
    for name in ("base_dataset", "protocol", "tracks_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.parse_one:
        print(json.dumps(parse_tracking_file(args.parse_one.expanduser().resolve())))
        return
    run(args)


if __name__ == "__main__":
    main()
