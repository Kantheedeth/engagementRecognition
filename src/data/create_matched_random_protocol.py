"""Create a frozen clip-randomized protocol matching temporal-fold class counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_episode_cv import verify_episode_cv
from src.data.prepare_lecture_subset import LABELS, SPLITS, sha256, verify_subset


VIEW_PATTERN = re.compile(r"^view(\d+)\.mp4$", re.IGNORECASE)
LABEL_NAMES = {label: category for category, label in LABELS.items()}


def _episode_id(record: dict) -> str:
    category, filename = record["key"].split("/", 1)
    match = VIEW_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Cannot parse view ID from {record['key']}")
    return f"C-{category.upper()}-{int(match.group(1)):04d}"


def create_matched_random_protocol(base_dataset: Path, target_folds: Path,
                                   output_path: Path, split_seed: int = 20260926) -> dict:
    """Randomize clips while exactly matching every target fold's class counts."""
    base_dataset = Path(base_dataset).expanduser().resolve()
    target_folds = Path(target_folds).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {output_path}")
    base = verify_subset(base_dataset)
    target = verify_episode_cv(target_folds)
    fold_numbers = sorted(int(value) for value in target["folds"])
    if fold_numbers != list(range(1, len(fold_numbers) + 1)):
        raise ValueError("Target folds must be consecutively numbered from one")

    target_counts = {}
    for fold in fold_numbers:
        selection = json.loads(
            (target_folds / f"fold_{fold}/selection_manifest.json").read_text()
        )
        target_counts[str(fold)] = selection["class_counts"]

    records_by_label = {label: [] for label in LABEL_NAMES}
    for record in base["records"]:
        records_by_label[record["label"]].append(record)
    test_fold_by_key = {}
    for label, records in records_by_label.items():
        category = LABEL_NAMES[label]
        shuffled = sorted(records, key=lambda record: record["key"])
        random.Random(split_seed + label).shuffle(shuffled)
        cursor = 0
        for fold in fold_numbers:
            count = target_counts[str(fold)]["test"][category]
            selected = shuffled[cursor:cursor + count]
            if len(selected) != count:
                raise ValueError(f"Insufficient {category} clips for fold {fold} test")
            for record in selected:
                test_fold_by_key[record["key"]] = fold
            cursor += count
        if cursor != len(shuffled):
            raise ValueError(
                f"Target test counts cover {cursor}/{len(shuffled)} {category} clips"
            )

    episode_by_key = {record["key"]: _episode_id(record) for record in base["records"]}
    validation = {}
    for fold in fold_numbers:
        validation_ids = []
        for label, records in records_by_label.items():
            category = LABEL_NAMES[label]
            available = sorted(
                (record for record in records if test_fold_by_key[record["key"]] != fold),
                key=lambda record: record["key"],
            )
            random.Random(split_seed + 1000 * fold + label).shuffle(available)
            count = target_counts[str(fold)]["val"][category]
            selected = available[:count]
            if len(selected) != count:
                raise ValueError(f"Insufficient {category} clips for fold {fold} validation")
            validation_ids.extend(episode_by_key[record["key"]] for record in selected)
        validation[str(fold)] = sorted(validation_ids)

    episodes = []
    for record in sorted(base["records"], key=lambda item: item["key"]):
        category, filename = record["key"].split("/", 1)
        number = int(VIEW_PATTERN.fullmatch(filename).group(1))
        episodes.append({
            "episode_id": episode_by_key[record["key"]],
            "class": category,
            "start_view": number,
            "end_view": number,
            "outer_fold": test_fold_by_key[record["key"]],
        })

    protocol = {
        "format_version": 1,
        "name": "lecture_size_matched_random_cv_v1",
        "evaluation_protocol": "size_matched_random_clip_three_fold_control",
        "grouping_name": "randomized single-clip groups",
        "description": (
            "Clip-randomized control with per-fold, per-class train/validation/test "
            "counts exactly matched to the conservative temporal-group protocol."
        ),
        "grouping_rule": "Every selected clip is its own split group.",
        "grouping_assumption": (
            "This deliberately permits temporally adjacent clips in different splits "
            "and is a leakage-sensitive control, not the main evaluation."
        ),
        "limitation": (
            "All clips come from one fixed classroom recording and clip-level "
            "randomization permits strong temporal correlation across splits."
        ),
        "split_seed": split_seed,
        "target_cv_manifest_sha256": sha256(target_folds / "cv_manifest.json"),
        "target_class_counts": target_counts,
        "outer_folds": len(fold_numbers),
        "validation_episodes": validation,
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dataset", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_only/dataset")
    parser.add_argument("--target_folds", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_temporal_group_cv/folds")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_matched_random_cv/matched_random_protocol.json")
    parser.add_argument("--split_seed", type=int, default=20260926)
    args = parser.parse_args()
    protocol = create_matched_random_protocol(
        args.base_dataset, args.target_folds, args.output, args.split_seed
    )
    print(json.dumps({"clips": len(protocol["episodes"]),
                      "folds": protocol["outer_folds"],
                      "split_seed": protocol["split_seed"]}, indent=2))


if __name__ == "__main__":
    main()
