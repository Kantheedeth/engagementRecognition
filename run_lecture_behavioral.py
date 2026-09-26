#!/usr/bin/env python3
"""Isolated lecture-only preparation, training and evaluation; no extraction."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from src.data.prepare_lecture_subset import prepare_subset, verify_subset

PROJECT_ROOT = Path(__file__).resolve().parent


def run_logged(command, log_path):
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "train", "eval", "all"], default="prepare")
    parser.add_argument("--experiment_dir", type=Path, default=PROJECT_ROOT / "experiments/lecture_only")
    parser.add_argument("--selected_dir", type=Path, default=PROJECT_ROOT / "lecture_only_videos")
    parser.add_argument("--selection_file", type=Path,
                        help="Portable class/video.mp4 selection list; overrides selected_dir during preparation")
    parser.add_argument("--source_dir", type=Path, default=PROJECT_ROOT / "feature_matrices_behavioral")
    parser.add_argument("--csv_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected_count", type=int, default=308)
    parser.add_argument("--run_name", default=None, help="New run name for training; existing run name for evaluation")
    parser.add_argument("--class_weight_mode", choices=["smoothed", "balanced", "none"], default="smoothed")
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
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience and batch_size must be positive")
    if not 0 <= args.label_smoothing <= 1:
        parser.error("label_smoothing must be between 0 and 1")
    if args.lr < 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid learning rate, weight decay or dropout")
    if args.branch_dim < 1 or args.num_heads < 1 or (args.branch_dim * 2) % args.num_heads:
        parser.error("Twice branch_dim must be divisible by num_heads")
    experiment = args.experiment_dir.expanduser().resolve()
    dataset = experiment / "dataset"
    run_name = args.run_name or f"{args.class_weight_mode}_seed{args.seed}"
    if Path(run_name).name != run_name or run_name in ("", ".", ".."):
        parser.error("run_name must be a single directory name")
    if args.stage in ("prepare", "all"):
        if args.stage == "all" and dataset.exists():
            print("Reusing the existing frozen selection snapshot.")
        else:
            prepare_subset(args.selected_dir, args.source_dir, args.csv_dir, dataset,
                           args.expected_count, args.selection_file)
    selection = verify_subset(dataset)
    if selection["total_videos"] != args.expected_count:
        raise ValueError("Snapshot count differs from --expected_count")
    print(f"Verified {selection['total_videos']} copied matrices: {selection['class_counts']}")
    print("Protocol: filtered original splits; not a recording-held-out test.")
    if args.stage == "prepare":
        print(f"Prepared dataset: {dataset}")
        return

    run_dir = experiment / "runs" / run_name
    checkpoints = run_dir / "checkpoints"
    reports = run_dir / "reports"
    if args.stage in ("train", "all"):
        # Every training invocation must have a fresh run name.
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoints.mkdir()
        reports.mkdir()
        command = [sys.executable, "-u", str(PROJECT_ROOT / "src/training/train_behavioral.py"),
                   "--data_dir", str(dataset / "matrices"), "--checkpoint_dir", str(checkpoints)]
        for option in ("class_weight_mode", "label_smoothing", "seed", "epochs", "patience",
                       "batch_size", "lr", "weight_decay", "branch_dim", "num_heads", "dropout", "device"):
            command += [f"--{option}", str(getattr(args, option))]
        config = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "training_command": command, "protocol": selection["protocol"]}
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        run_logged(command, reports / "train.log")
    if args.stage in ("eval", "all"):
        if not (checkpoints / "best_model_behavioral.pth").is_file():
            raise FileNotFoundError(f"Train this run first: {run_dir}")
        reports.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(PROJECT_ROOT / "src/training/evaluate_behavioral.py"),
                   "--data_dir", str(dataset / "matrices"), "--checkpoint_dir", str(checkpoints),
                   "--output_cm_file", str(reports / "confusion_matrix.png"),
                   "--output_metrics_file", str(reports / "metrics.json"),
                   "--batch_size", str(args.batch_size), "--device", args.device]
        run_logged(command, reports / "evaluate.log")
    print(f"Run outputs: {run_dir}")


if __name__ == "__main__":
    main()
