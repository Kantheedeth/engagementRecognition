"""Build episode-held-out folds from a frozen lecture-only matrix snapshot."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from src.data.prepare_lecture_subset import LABELS, SPLITS, sha256, verify_subset


VIEW_PATTERN = re.compile(r"^view(\d+)\.mp4$", re.IGNORECASE)


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    folds = protocol.get("outer_folds")
    episodes = protocol.get("episodes")
    if not isinstance(folds, int) or folds < 2 or not isinstance(episodes, list) or not episodes:
        raise ValueError("Episode protocol needs at least two outer folds and a non-empty episode list")
    ids = [episode.get("episode_id") for episode in episodes]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Episode IDs must be unique non-empty strings")
    for episode in episodes:
        category = episode.get("class", "").lower()
        start, end, fold = (episode.get("start_view"), episode.get("end_view"),
                            episode.get("outer_fold"))
        if category not in LABELS or not isinstance(start, int) or not isinstance(end, int) or start > end:
            raise ValueError(f"Invalid episode range: {episode}")
        if not isinstance(fold, int) or not 1 <= fold <= folds:
            raise ValueError(f"Invalid outer fold: {episode}")
    validation = protocol.get("validation_episodes", {})
    known = set(ids)
    for fold in range(1, folds + 1):
        selected = validation.get(str(fold))
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ValueError(f"Fold {fold} needs unique validation episode IDs")
        unknown = set(selected) - known
        if unknown:
            raise ValueError(f"Unknown validation episode IDs for fold {fold}: {sorted(unknown)}")
        test_ids = {e["episode_id"] for e in episodes if e["outer_fold"] == fold}
        if set(selected) & test_ids:
            raise ValueError(f"Fold {fold} validation and test episodes overlap")
    return protocol


def _assign_episodes(base_records: list[dict], protocol: dict) -> tuple[list[dict], dict[str, dict]]:
    episodes = {episode["episode_id"]: episode for episode in protocol["episodes"]}
    assigned = []
    used = Counter()
    numeric_keys = set()
    for record in base_records:
        category, filename = record["key"].split("/", 1)
        match = VIEW_PATTERN.fullmatch(filename)
        if not match:
            raise ValueError(f"Cannot parse view ID from {record['key']}")
        number = int(match.group(1))
        identity = (category, number)
        if identity in numeric_keys:
            raise ValueError(f"Duplicate class/view identity in base snapshot: {identity}")
        numeric_keys.add(identity)
        matches = [episode for episode in episodes.values()
                   if episode["class"].lower() == category
                   and episode["start_view"] <= number <= episode["end_view"]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one episode for {record['key']}; found {len(matches)}")
        episode = matches[0]
        if LABELS[category] != record["label"]:
            raise ValueError(f"Class and label disagree for {record['key']}")
        assigned.append(dict(record, episode_id=episode["episode_id"],
                             outer_fold=episode["outer_fold"], base_split=record["split"]))
        used[episode["episode_id"]] += 1
    empty = set(episodes) - set(used)
    if empty:
        raise ValueError(f"Episodes contain no selected clips: {sorted(empty)}")
    return assigned, episodes


def _split_for(record: dict, fold: int, validation_ids: set[str]) -> str:
    if record["outer_fold"] == fold:
        return "test"
    if record["episode_id"] in validation_ids:
        return "val"
    return "train"


def _class_counts(records: list[dict]) -> dict:
    return {
        split: {category: sum(r["split"] == split and r["label"] == label for r in records)
                for category, label in LABELS.items()}
        for split in SPLITS
    }


def prepare_episode_cv(base_dataset: Path, protocol_path: Path, output_dir: Path) -> dict:
    """Create all outer folds atomically; never modify the frozen base snapshot."""
    base_dataset = Path(base_dataset).expanduser().resolve()
    protocol_path = Path(protocol_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing episode folds: {output_dir}")

    base = verify_subset(base_dataset)
    protocol = _load_protocol(protocol_path)
    assigned, episodes = _assign_episodes(base["records"], protocol)
    base_manifest = json.loads((base_dataset / "matrices/build_manifest.json").read_text())
    expected_test_keys = {record["key"] for record in assigned}
    observed_test_keys = set()
    summaries = {}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".episode-cv-", dir=output_dir.parent) as temporary:
        staging_root = Path(temporary) / "folds"
        for fold in range(1, protocol["outer_folds"] + 1):
            validation_ids = set(protocol["validation_episodes"][str(fold)])
            fold_dir = staging_root / f"fold_{fold}"
            for split in SPLITS:
                (fold_dir / "matrices" / split).mkdir(parents=True)
            (fold_dir / "splits").mkdir()

            records = []
            split_lines = {split: [] for split in SPLITS}
            for source_record in assigned:
                split = _split_for(source_record, fold, validation_ids)
                record = dict(source_record, split=split)
                source = (base_dataset / "matrices" / source_record["base_split"] /
                          source_record["matrix_name"])
                target = fold_dir / "matrices" / split / source_record["matrix_name"]
                shutil.copy2(source, target)
                if sha256(target) != source_record["sha256"]:
                    raise ValueError(f"Copy verification failed: {target}")
                records.append(record)
                category, filename = record["key"].split("/", 1)
                split_lines[split].append(f"videos/{category}/{filename} {record['label']}")
                if split == "test":
                    if record["key"] in observed_test_keys:
                        raise ValueError(f"Clip is a test item in multiple folds: {record['key']}")
                    observed_test_keys.add(record["key"])

            counts = _class_counts(records)
            if any(not value for row in counts.values() for value in row.values()):
                raise ValueError(f"Every fold split must contain every class; fold {fold}: {counts}")
            split_episodes = {
                split: sorted({r["episode_id"] for r in records if r["split"] == split})
                for split in SPLITS
            }
            if any(set(split_episodes[a]) & set(split_episodes[b])
                   for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
                raise ValueError(f"Episode leakage detected in fold {fold}")
            for split in SPLITS:
                path = fold_dir / "splits" / f"{split}.csv"
                path.write_text("\n".join(split_lines[split]) + "\n", encoding="utf-8")

            selection = {
                "format_version": 2,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol.get("evaluation_protocol", "episode_held_out_cv"),
                "limitation": protocol["limitation"],
                "fold": fold,
                "episode_protocol": str(protocol_path),
                "episode_protocol_sha256": sha256(protocol_path),
                "base_dataset": str(base_dataset),
                "base_selection_manifest_sha256": sha256(base_dataset / "selection_manifest.json"),
                "total_videos": len(records),
                "class_counts": counts,
                "split_episodes": split_episodes,
                "records": records,
            }
            selection["filtered_csv_sha256"] = {
                split: sha256(fold_dir / "splits" / f"{split}.csv") for split in SPLITS
            }
            selection_path = fold_dir / "selection_manifest.json"
            selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
            manifest = deepcopy(base_manifest)
            manifest["split_counts"] = {split: sum(counts[split].values()) for split in SPLITS}
            manifest["total_videos"] = len(records)
            manifest["subset"] = {
                "protocol": selection["protocol"],
                "fold": fold,
                "selection_manifest_sha256": sha256(selection_path),
            }
            (fold_dir / "matrices/build_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            verify_subset(fold_dir)
            summaries[str(fold)] = {"class_counts": counts, "split_episodes": split_episodes}

        if observed_test_keys != expected_test_keys:
            raise ValueError("Outer test folds do not cover every selected clip exactly once")
        staging_root.rename(output_dir)

    summary = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "base_selection_manifest_sha256": sha256(base_dataset / "selection_manifest.json"),
        "total_videos": len(assigned),
        "episodes": len(episodes),
        "grouping_name": protocol.get("grouping_name", "episode groups"),
        "folds": summaries,
    }
    (output_dir / "cv_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify_episode_cv(output_dir: Path, protocol_path: Path | None = None) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    summary = json.loads((output_dir / "cv_manifest.json").read_text())
    if protocol_path and sha256(Path(protocol_path).expanduser().resolve()) != summary["protocol_sha256"]:
        raise ValueError("Episode protocol changed after fold preparation")
    test_keys = set()
    for fold_text in summary["folds"]:
        selection = verify_subset(output_dir / f"fold_{fold_text}")
        train_eps = set(selection["split_episodes"]["train"])
        val_eps = set(selection["split_episodes"]["val"])
        test_eps = set(selection["split_episodes"]["test"])
        if train_eps & val_eps or train_eps & test_eps or val_eps & test_eps:
            raise ValueError(f"Episode leakage in fold {fold_text}")
        for record in selection["records"]:
            if record["split"] == "test":
                if record["key"] in test_keys:
                    raise ValueError(f"Duplicate outer test clip: {record['key']}")
                test_keys.add(record["key"])
    if len(test_keys) != summary["total_videos"]:
        raise ValueError("Outer folds do not cover every clip exactly once")
    return summary


def write_assignment_csv(output_dir: Path, destination: Path) -> None:
    """Write a human-readable assignment table from the verified fold manifests."""
    output_dir = Path(output_dir).resolve()
    summary = verify_episode_cv(output_dir)
    rows = []
    for fold_text in summary["folds"]:
        selection = json.loads((output_dir / f"fold_{fold_text}/selection_manifest.json").read_text())
        rows.extend({"fold": int(fold_text), "split": r["split"], "episode_id": r["episode_id"],
                     "key": r["key"], "label": r["label"]} for r in selection["records"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "split", "episode_id", "key", "label"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["fold"], row["split"], row["key"])))
