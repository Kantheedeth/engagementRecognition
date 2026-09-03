#!/usr/bin/env python3
"""Run the versioned 40-D interaction + 8-D group-affect pipeline.

Usage:
    # Complete run: audit first, then reuse/extract, build, train, and evaluate.
    python run_behavioral_track.py --stage all --group_manifest split_groups.csv

    # Run one part of the pipeline.
    python run_behavioral_track.py --stage extract
    python run_behavioral_track.py --stage build
    python run_behavioral_track.py --stage audit --group_manifest split_groups.csv
    python run_behavioral_track.py --stage train --group_manifest split_groups.csv
    python run_behavioral_track.py --stage eval --group_manifest split_groups.csv

Existing interaction and affect arrays are reused only when their shape and
recorded extraction provenance match. They are recomputed only when the
corresponding --overwrite_* flag is supplied. The golden_pair_id column in the
group manifest may be empty; session_id must come from authoritative metadata
and is never guessed from randomized OUC-CGE clip names.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def run_command(command: list[str], description: str) -> None:
    print("\n" + "=" * 72)
    print(f"STAGE: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 72)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def stage_extract(args: argparse.Namespace) -> None:
    interaction_command = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src/data/extract_interaction_features.py"),
        "--device",
        args.device,
    ]
    if args.overwrite_interaction:
        interaction_command.append("--overwrite")
    if args.save_interaction_track_details:
        interaction_command.append("--save_track_details")
    run_command(interaction_command, "40-D pose + ByteTrack interaction extraction")

    affect_command = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src/data/extract_affect_features.py"),
        "--device",
        args.device,
    ]
    if args.no_affect_tracking:
        affect_command.append("--no-tracking")
    if args.overwrite_affect:
        affect_command.append("--overwrite")
    run_command(affect_command, "8-D track-aware group-affect extraction or reuse")


def stage_build(args: argparse.Namespace) -> None:
    run_command(
        [
            PYTHON_EXE,
            str(PROJECT_ROOT / "src/data/build_behavioral_matrices.py"),
            "--csv_dir",
            str(args.csv_dir),
        ],
        "Build versioned (8, 48) behavioral matrices",
    )


def stage_audit(args: argparse.Namespace) -> None:
    """Fail before training if an exact video, session, or known pair leaks."""
    run_command(
        [
            PYTHON_EXE,
            str(PROJECT_ROOT / "src/tools/audit_split_integrity.py"),
            "--csv_dir",
            str(args.csv_dir),
            "--group_manifest",
            str(args.group_manifest),
        ],
        "Audit video/session/golden-pair split integrity",
    )


def stage_train(args: argparse.Namespace) -> None:
    run_command(
        [
            PYTHON_EXE,
            str(PROJECT_ROOT / "src/training/train_behavioral.py"),
            "--csv_dir",
            str(args.csv_dir),
            "--group_manifest",
            str(args.group_manifest),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--branch_dim",
            str(args.branch_dim),
            "--num_heads",
            str(args.num_heads),
            "--seed",
            str(args.seed),
        ],
        "Train the new-schema behavioral model",
    )


def stage_evaluate(args: argparse.Namespace) -> None:
    run_command(
        [
            PYTHON_EXE,
            str(PROJECT_ROOT / "src/training/evaluate_behavioral.py"),
            "--csv_dir",
            str(args.csv_dir),
            "--group_manifest",
            str(args.group_manifest),
            "--branch_dim",
            str(args.branch_dim),
            "--num_heads",
            str(args.num_heads),
        ],
        "Evaluate the matching new-schema checkpoint",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "build", "audit", "train", "eval"),
        default="all",
        help=(
            "all audits first, then runs extract, build, train, and evaluation"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Compatibility alias: overwrite both interaction and affect features",
    )
    parser.add_argument("--overwrite_interaction", action="store_true")
    parser.add_argument("--overwrite_affect", action="store_true")
    parser.add_argument("--save_interaction_track_details", action="store_true")
    parser.add_argument("--no_affect_tracking", action="store_true")
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory containing train.csv, val.csv, and test.csv",
    )
    parser.add_argument(
        "--group_manifest",
        type=Path,
        help=(
            "Authoritative CSV with video_path, session_id, golden_pair_id; "
            "golden_pair_id may be blank"
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--branch_dim", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.overwrite:
        args.overwrite_interaction = True
        args.overwrite_affect = True
    if args.stage in ("all", "audit", "train", "eval") and args.group_manifest is None:
        parser.error(
            "--group_manifest is required for training/evaluation so session and "
            "golden-pair leakage can be checked"
        )
    args.csv_dir = args.csv_dir.expanduser().resolve()
    if args.group_manifest is not None:
        args.group_manifest = args.group_manifest.expanduser().resolve()
        if not args.group_manifest.is_file():
            parser.error(f"group manifest not found: {args.group_manifest}")
    return args


def main() -> None:
    args = parse_args()
    print("Track-aware zero-scene behavioral pipeline: 40 interaction + 8 affect")
    print(f"Selected stage: {args.stage}")

    # Audit first during a complete run, before any potentially expensive work.
    # Training and evaluation independently repeat this audit when invoked alone.
    if args.stage == "all":
        stage_audit(args)
    if args.stage in ("all", "extract"):
        print(
            "Reuse policy: valid existing features are preserved; overwrite is "
            "opt-in per modality."
        )
        stage_extract(args)
    if args.stage in ("all", "build"):
        stage_build(args)
    if args.stage == "audit":
        stage_audit(args)
    if args.stage in ("all", "train"):
        stage_train(args)
    if args.stage in ("all", "eval"):
        stage_evaluate(args)


if __name__ == "__main__":
    main()
