#!/usr/bin/env python3
"""Reproducible Random-Forest ablations for old and track-aware interactions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_schema import (
    AFFECT_COLUMNS,
    BEHAVIORAL_FEATURE_SCHEMA,
    BEHAVIORAL_SHAPE,
    LEGACY_BEHAVIORAL_FEATURE_SCHEMA,
    TRACK_INTERACTION_COLUMNS,
)
from src.data.split_integrity import audit_split_integrity
from src.models.dataset import EngagementDataset, load_feature_manifest


def load_temporal_means(
    data_dir: Path,
    schema: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    manifest = load_feature_manifest(
        data_dir, expected_schema=schema, expected_shape=shape
    )
    train = EngagementDataset(data_dir / "train", expected_shape=shape)
    test = EngagementDataset(data_dir / "test", expected_shape=shape)
    if not train or not test:
        raise RuntimeError(f"Missing train or test matrices under {data_dir}")
    recorded_counts = manifest.get("split_counts", {})
    if recorded_counts.get("train") != len(train) or recorded_counts.get("test") != len(test):
        raise ValueError(
            f"Matrix counts under {data_dir} do not match its build manifest"
        )
    train_x = np.stack([train[index][0].numpy() for index in range(len(train))]).mean(1)
    train_y = np.asarray([train[index][1].item() for index in range(len(train))])
    test_x = np.stack([test[index][0].numpy() for index in range(len(test))]).mean(1)
    test_y = np.asarray([test[index][1].item() for index in range(len(test))])
    return (
        train_x,
        train_y,
        test_x,
        test_y,
        [path.name for path in train.file_paths],
        [path.name for path in test.file_paths],
    )


def score_predictions(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
    class_names = ("Low", "Mid", "High")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "ordinal_mae": float(mean_absolute_error(y_true, predictions)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1, 2]
        ).tolist(),
    }


def fit_baseline(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    indices,
    seed: int,
) -> tuple[dict, RandomForestClassifier]:
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(train_x[:, indices], train_y)
    return score_predictions(test_y, model.predict(test_x[:, indices])), model


def run_ablation(args: argparse.Namespace) -> None:
    split_integrity = audit_split_integrity(
        Path(args.csv_dir), Path(args.group_manifest)
    )
    new_dir = Path(args.data_dir).expanduser().resolve()
    legacy_dir = Path(args.legacy_data_dir).expanduser().resolve()
    new_manifest = load_feature_manifest(
        new_dir,
        expected_schema=BEHAVIORAL_FEATURE_SCHEMA,
        expected_shape=BEHAVIORAL_SHAPE,
    )
    if new_manifest.get("split_csv_sha256") != split_integrity["csv_sha256"]:
        raise ValueError(
            "New matrices were not built from the currently audited split CSVs"
        )
    new_data = load_temporal_means(
        new_dir, BEHAVIORAL_FEATURE_SCHEMA, BEHAVIORAL_SHAPE
    )
    legacy_data = load_temporal_means(
        legacy_dir, LEGACY_BEHAVIORAL_FEATURE_SCHEMA, (8, 40)
    )
    (
        new_train_x,
        new_train_y,
        new_test_x,
        new_test_y,
        new_train_names,
        new_test_names,
    ) = new_data
    (
        old_train_x,
        old_train_y,
        old_test_x,
        old_test_y,
        old_train_names,
        old_test_names,
    ) = legacy_data
    if new_train_names != old_train_names or new_test_names != old_test_names:
        raise ValueError(
            "Old and new datasets do not contain the identical ordered matrix "
            "filenames; refusing a sample-misaligned comparison."
        )
    if not np.array_equal(new_train_y, old_train_y) or not np.array_equal(
        new_test_y, old_test_y
    ):
        raise ValueError(
            "Old and new datasets do not have identical ordered labels; refusing an "
            "invalid comparison."
        )

    interaction_dim = len(TRACK_INTERACTION_COLUMNS)
    affect_dim = len(AFFECT_COLUMNS)
    coordinate_names = {
        "teacher_position_x",
        "teacher_position_y",
        "mean_student_x",
        "mean_student_y",
    }
    no_absolute_coordinates = [
        index
        for index, name in enumerate(TRACK_INTERACTION_COLUMNS)
        if name not in coordinate_names
    ] + list(range(interaction_dim, interaction_dim + affect_dim))

    experiments = []
    majority = np.full_like(new_test_y, np.bincount(new_train_y).argmax())
    experiments.append(("Majority-class baseline", score_predictions(new_test_y, majority)))

    specifications = [
        (
            "Affect only",
            new_train_x,
            new_train_y,
            new_test_x,
            new_test_y,
            slice(interaction_dim, interaction_dim + affect_dim),
        ),
        (
            "Old interaction only (32-D, no track roles)",
            old_train_x,
            old_train_y,
            old_test_x,
            old_test_y,
            slice(0, 32),
        ),
        (
            "Affect + old interaction",
            old_train_x,
            old_train_y,
            old_test_x,
            old_test_y,
            slice(0, 40),
        ),
        (
            "New interaction only (40-D track roles)",
            new_train_x,
            new_train_y,
            new_test_x,
            new_test_y,
            slice(0, interaction_dim),
        ),
        (
            "Affect + new interaction",
            new_train_x,
            new_train_y,
            new_test_x,
            new_test_y,
            slice(0, interaction_dim + affect_dim),
        ),
        (
            "New fusion without absolute positions",
            new_train_x,
            new_train_y,
            new_test_x,
            new_test_y,
            no_absolute_coordinates,
        ),
    ]
    full_model = None
    for name, train_x, train_y, test_x, test_y, indices in specifications:
        metrics, model = fit_baseline(
            train_x, train_y, test_x, test_y, indices, args.seed
        )
        experiments.append((name, metrics))
        if name == "Affect + new interaction":
            full_model = model

    print("=" * 105)
    print("TRACK-AWARE INTERACTION ABLATION (TEMPORAL-MEAN RANDOM FOREST)")
    print(f"Seed: {args.seed} | train={len(new_train_y)} | test={len(new_test_y)}")
    print(
        "Strict split audit: "
        f"{split_integrity['session_count']} sessions, "
        f"{split_integrity['golden_pair_count']} golden-pair groups"
    )
    print("=" * 105)
    print(
        f"{'Configuration':<48} {'Accuracy':>10} {'Macro-F1':>10} "
        f"{'Balanced':>10} {'Ord MAE':>10}"
    )
    for name, metrics in experiments:
        print(
            f"{name:<48} {metrics['accuracy']*100:>9.2f}% "
            f"{metrics['macro_f1']*100:>9.2f}% "
            f"{metrics['balanced_accuracy']*100:>9.2f}% "
            f"{metrics['ordinal_mae']:>10.4f}"
        )

    print("\nPer-class metrics and confusion matrices (rows=true, columns=predicted):")
    for name, metrics in experiments:
        print(f"\n{name}")
        for class_name, class_metrics in metrics["per_class"].items():
            print(
                f"  {class_name:<4} precision={class_metrics['precision']:.4f} "
                f"recall={class_metrics['recall']:.4f} "
                f"f1={class_metrics['f1']:.4f} "
                f"support={class_metrics['support']}"
            )
        print(f"  confusion_matrix={metrics['confusion_matrix']}")

    if full_model is not None:
        names = list(TRACK_INTERACTION_COLUMNS) + list(AFFECT_COLUMNS)
        top = np.argsort(full_model.feature_importances_)[::-1][:10]
        print("\nTop Random-Forest feature associations (not causal importance):")
        for rank, index in enumerate(top, start=1):
            print(
                f"  {rank:2d}. {names[index]:<42} "
                f"{full_model.feature_importances_[index]*100:6.2f}%"
            )
    print(
        "\nInterpretation warning: this diagnostic does not prove attention, "
        "engagement causality, or freedom from session/camera leakage."
    )
    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed,
            "split_integrity": split_integrity,
            "train_samples": len(new_train_y),
            "test_samples": len(new_test_y),
            "experiments": [
                {"configuration": name, **metrics} for name, metrics in experiments
            ],
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        print(f"Saved computed ablation report: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "feature_matrices_behavioral_track",
    )
    parser.add_argument(
        "--legacy_data_dir",
        type=Path,
        default=PROJECT_ROOT / "feature_matrices_behavioral",
    )
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--group_manifest",
        type=Path,
        required=True,
        help="Authoritative CSV with video_path,session_id,golden_pair_id",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=Path)
    run_ablation(parser.parse_args())


if __name__ == "__main__":
    main()
