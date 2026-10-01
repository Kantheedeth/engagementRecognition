#!/usr/bin/env python3
"""Prepare, train and evaluate strict episode-held-out lecture folds."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from src.data.prepare_episode_cv import prepare_episode_cv, verify_episode_cv, write_assignment_csv
from src.data.prepare_lecture_subset import verify_subset


PROJECT_ROOT = Path(__file__).resolve().parent


def run_logged(command: list[str], log_path: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def parse_folds(text: str, available: set[int]) -> list[int]:
    try:
        folds = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from exc
    if not folds or len(folds) != len(set(folds)) or not set(folds) <= available:
        raise argparse.ArgumentTypeError(f"folds must be unique values from {sorted(available)}")
    return folds


def aggregate(run_root: Path, fold_root: Path, folds: list[int], total_expected: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from src.training.evaluate_behavioral import plot_confusion_matrix

    y_true, y_pred, predictions, per_fold = [], [], [], {}
    seen = set()
    evaluation_protocol = None
    for fold in folds:
        metrics_path = run_root / f"fold_{fold}/reports/metrics.json"
        metrics = json.loads(metrics_path.read_text())
        selection = json.loads((fold_root / f"fold_{fold}/selection_manifest.json").read_text())
        if evaluation_protocol is None:
            evaluation_protocol = selection["protocol"]
        elif evaluation_protocol != selection["protocol"]:
            raise ValueError("Fold protocol mismatch")
        by_matrix = {r["matrix_name"]: r for r in selection["records"] if r["split"] == "test"}
        per_fold[str(fold)] = {
            "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
            "test_samples": metrics["test_samples"],
        }
        for item in metrics["predictions"]:
            record = by_matrix.get(item["matrix"])
            if record is None or record["key"] in seen:
                raise ValueError(f"Invalid or duplicate pooled prediction: fold={fold}, item={item}")
            seen.add(record["key"])
            true, predicted = item["true_label"], item["predicted_label"]
            y_true.append(true)
            y_pred.append(predicted)
            predictions.append({"fold": fold, "episode_id": record["episode_id"],
                                "key": record["key"], "true_label": true,
                                "predicted_label": predicted})
    if len(folds) == 3 and len(seen) != total_expected:
        raise ValueError(f"Expected {total_expected} pooled predictions, found {len(seen)}")
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)
    labels, names = [0, 1, 2], ["Low", "Mid", "High"]
    cm = confusion_matrix(y_true_array, y_pred_array, labels=labels)
    result = {
        "protocol": evaluation_protocol,
        "folds": folds,
        "pooled_out_of_fold_samples": len(y_true),
        "pooled_macro_f1": float(f1_score(y_true_array, y_pred_array, labels=labels,
                                          average="macro", zero_division=0)),
        "pooled_accuracy": float(np.mean(y_true_array == y_pred_array)),
        "always_low_macro_f1": float(f1_score(y_true_array, np.zeros_like(y_true_array),
                                               labels=labels, average="macro", zero_division=0)),
        "class_order": names,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true_array, y_pred_array, labels=labels, target_names=names,
            zero_division=0, output_dict=True,
        ),
        "per_fold": per_fold,
        "predictions": predictions,
    }
    reports = run_root / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "pooled_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (reports / "pooled_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(predictions[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(predictions)
    plot_confusion_matrix(cm, names, filename=reports / "pooled_confusion_matrix.png")
    print("\nPOOLED OUT-OF-FOLD EPISODE-CV RESULT")
    print(classification_report(y_true_array, y_pred_array, labels=labels,
                                target_names=names, zero_division=0))
    print(f"Pooled Macro-F1: {result['pooled_macro_f1'] * 100:.2f}%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "train", "eval", "all"], default="prepare")
    parser.add_argument("--experiment_dir", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_episode_cv")
    parser.add_argument("--base_dataset", type=Path,
                        default=PROJECT_ROOT / "experiments/lecture_only/dataset")
    parser.add_argument("--episode_protocol", type=Path, default=None)
    parser.add_argument("--folds", default="1,2,3")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--class_weight_mode", choices=["smoothed", "balanced", "none"],
                        default="smoothed")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--branch_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--branch_mode", choices=["both", "interaction", "affect"], default="both")
    parser.add_argument(
        "--drop_interaction_indices", default="",
        help="Comma-separated zero-based interaction columns omitted during training",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    protocol = (args.episode_protocol or experiment / "episode_protocol.json").expanduser().resolve()
    fold_root = experiment / "folds"
    run_name = args.run_name or f"{args.class_weight_mode}_seed{args.seed}"
    if Path(run_name).name != run_name or run_name in ("", ".", ".."):
        parser.error("run_name must be a single directory name")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience and batch_size must be positive")
    if not 0 <= args.label_smoothing <= 1 or not 0 <= args.dropout < 1:
        parser.error("invalid smoothing or dropout")

    if args.stage in ("prepare", "all"):
        if args.stage == "all" and fold_root.exists():
            print("Reusing verified episode folds.")
        else:
            summary = prepare_episode_cv(args.base_dataset, protocol, fold_root)
            write_assignment_csv(fold_root, experiment / "episode_assignments.csv")
            print(json.dumps(summary, indent=2))
    summary = verify_episode_cv(fold_root, protocol)
    available = {int(value) for value in summary["folds"]}
    try:
        folds = parse_folds(args.folds, available)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    print(f"Verified {summary['episodes']} {summary.get('grouping_name', 'episode groups')} "
          f"and {summary['total_videos']} clips.")
    print("Limitation: all folds remain within one fixed classroom recording.")
    if args.stage == "prepare":
        return

    run_root = experiment / "runs" / run_name
    if args.stage in ("train", "all"):
        run_root.mkdir(parents=True, exist_ok=False)
        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "folds": folds,
            "protocol_sha256": summary["protocol_sha256"],
        }
        (run_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        for fold in folds:
            dataset = fold_root / f"fold_{fold}"
            selection = verify_subset(dataset)
            fold_run = run_root / f"fold_{fold}"
            checkpoints, reports = fold_run / "checkpoints", fold_run / "reports"
            checkpoints.mkdir(parents=True)
            reports.mkdir()
            print(f"Fold {fold}: {selection['class_counts']}")
            command = [sys.executable, "-u", str(PROJECT_ROOT / "src/training/train_behavioral.py"),
                       "--data_dir", str(dataset / "matrices"),
                       "--checkpoint_dir", str(checkpoints)]
            for option in ("class_weight_mode", "label_smoothing", "seed", "epochs", "patience",
                           "batch_size", "lr", "weight_decay", "branch_dim", "num_heads",
                           "dropout", "branch_mode", "drop_interaction_indices", "device"):
                command += [f"--{option}", str(getattr(args, option))]
            run_logged(command, reports / "train.log")
    if args.stage in ("eval", "all"):
        for fold in folds:
            dataset = fold_root / f"fold_{fold}"
            fold_run = run_root / f"fold_{fold}"
            checkpoints, reports = fold_run / "checkpoints", fold_run / "reports"
            if not (checkpoints / "best_model_behavioral.pth").is_file():
                raise FileNotFoundError(f"Train fold {fold} first: {fold_run}")
            reports.mkdir(exist_ok=True)
            command = [sys.executable, "-u", str(PROJECT_ROOT / "src/training/evaluate_behavioral.py"),
                       "--data_dir", str(dataset / "matrices"),
                       "--checkpoint_dir", str(checkpoints),
                       "--output_cm_file", str(reports / "confusion_matrix.png"),
                       "--output_metrics_file", str(reports / "metrics.json"),
                       "--batch_size", str(args.batch_size), "--device", args.device]
            run_logged(command, reports / "evaluate.log")
        aggregate(run_root, fold_root, folds, summary["total_videos"])


if __name__ == "__main__":
    main()
