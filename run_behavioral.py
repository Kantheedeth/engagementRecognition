#!/usr/bin/env python3
"""Master entrypoint to run the Pure Behavioral Engagement Recognition Pipeline.

Usage:
    python run_behavioral.py --stage all        # Run full pipeline end-to-end
    python run_behavioral.py --stage extract    # Extract interaction & affect features
    python run_behavioral.py --stage build      # Build 40-dim behavioral matrices
    python run_behavioral.py --stage train      # Train pure behavioral attention model
    python run_behavioral.py --stage eval       # Evaluate test set & plot confusion matrix
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def run_command(cmd, desc):
    print("\n" + "=" * 70)
    print(f"▶ STAGE: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n❌ Error: Stage '{desc}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def stage_extract(args):
    print("\n📦 Extracting Behavioral Features (Interaction 32-dim + Affect 8-dim)...")
    # 1. Interaction features (YOLOv8)
    cmd_inter = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "extract_interaction_features.py"),
        "--device", args.device,
    ]
    run_command(cmd_inter, "Extracting YOLOv8 32-dim Interaction Features")

    # 2. Track-Aware Affect features (RetinaFace + ByteTrack + ViT FER)
    cmd_affect = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "extract_affect_features.py"),
        "--device", args.device,
    ]
    if args.no_tracking:
        cmd_affect.append("--no-tracking")
    run_command(cmd_affect, "Extracting Track-Aware 8-dim Affect Features")


def stage_build(args):
    print("\n🧱 Building Pure Behavioral Feature Matrices (40-dim)...")
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "build_behavioral_matrices.py"),
    ]
    run_command(cmd, "Building (8, 40) Pure Behavioral Matrices")


def stage_train(args):
    print("\n🏋️ Training Pure Behavioral Attention Classifier...")
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "training" / "train_behavioral.py"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--branch_dim", str(args.branch_dim),
        "--num_heads", str(args.num_heads),
        "--seed", str(args.seed),
    ]
    run_command(cmd, "Training Pure Behavioral Attention Model")


def stage_eval(args):
    print("\n📊 Evaluating Pure Behavioral Model on Test Set...")
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "training" / "evaluate_behavioral.py"),
        "--branch_dim", str(args.branch_dim),
        "--num_heads", str(args.num_heads),
    ]
    run_command(cmd, "Evaluating Behavioral Model & Plotting Confusion Matrix")


def main():
    parser = argparse.ArgumentParser(
        description="Master Runner for Pure Behavioral Pipeline (Zero Scene Shortcut)"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "build", "train", "eval"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use for feature extraction (auto, mps, cuda, cpu)",
    )
    parser.add_argument(
        "--no_tracking",
        action="store_true",
        help="Disable ByteTrack temporal smoothing during affect extraction",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--branch_dim", type=int, default=48, help="Subspace projection dimension per branch"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🚀 PURE BEHAVIORAL PIPELINE RUNNER")
    print(f"  • Selected Stage : {args.stage.upper()}")
    print(f"  • Modalities     : 32 Interaction + 8 Affect (Zero Scene)")
    print(f"  • Subspaces      : Inter(48) + Affect(48) = 96 Fused Embedding")
    print("=" * 70)

    if args.stage in ("all", "extract"):
        stage_extract(args)

    if args.stage in ("all", "build"):
        stage_build(args)

    if args.stage in ("all", "train"):
        stage_train(args)

    if args.stage in ("all", "eval"):
        stage_eval(args)

    print("\n" + "=" * 70)
    print("✅ PURE BEHAVIORAL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
