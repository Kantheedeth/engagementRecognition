#!/usr/bin/env python3
"""Master entrypoint to run the Multi-Branch Balanced Pipeline (Scene + Interaction + Affect).

Usage:
    python run_multimodal.py --stage all        # Run full pipeline end-to-end
    python run_multimodal.py --stage extract    # Extract scene, interaction & affect features
    python run_multimodal.py --stage build      # Build 616-dim multi-branch matrices
    python run_multimodal.py --stage train      # Train multi-branch balanced attention model
    python run_multimodal.py --stage eval       # Evaluate test set & plot confusion matrix
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
    print("\n📦 Extracting All 3 Modalities (Scene 576 + Interaction 32 + Affect 8)...")
    # 1. Scene features (MobileNetV3)
    cmd_scene = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "extract_scene_features.py"),
    ]
    run_command(cmd_scene, "Extracting MobileNetV3 576-dim Scene Features")

    # 2. Interaction features (YOLOv8)
    cmd_inter = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "extract_interaction_features.py"),
        "--device", args.device,
    ]
    run_command(cmd_inter, "Extracting YOLOv8 32-dim Interaction Features")

    # 3. Track-Aware Affect features (RetinaFace + ByteTrack + ViT FER)
    cmd_affect = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "extract_affect_features.py"),
        "--device", args.device,
    ]
    if args.no_tracking:
        cmd_affect.append("--no-tracking")
    run_command(cmd_affect, "Extracting Track-Aware 8-dim Affect Features")


def stage_build(args):
    print("\n🧱 Building Multi-Branch Feature Matrices (616-dim)...")
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "data" / "build_feature_matrices.py"),
    ]
    run_command(cmd, "Building (8, 616) Multi-Branch Matrices")


def stage_train(args):
    print("\n🏋️ Training Multi-Branch Balanced Attention Classifier...")
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "training" / "train.py"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--scene_branch_dim", str(args.scene_branch_dim),
        "--inter_branch_dim", str(args.inter_branch_dim),
        "--affect_branch_dim", str(args.affect_branch_dim),
        "--num_heads", str(args.num_heads),
        "--seed", str(args.seed),
    ]
    run_command(cmd, "Training Multi-Branch Attention Model (16/32/32)")


def stage_eval(args):
    print("\n📊 Evaluating Multi-Branch Model on Test Set...")
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "src" / "training" / "evaluate.py"),
        "--scene_branch_dim", str(args.scene_branch_dim),
        "--inter_branch_dim", str(args.inter_branch_dim),
        "--affect_branch_dim", str(args.affect_branch_dim),
        "--num_heads", str(args.num_heads),
    ]
    run_command(cmd, "Evaluating Multi-Branch Model & Plotting Confusion Matrix")


def main():
    parser = argparse.ArgumentParser(
        description="Master Runner for Multi-Branch Balanced Pipeline (Scene + Interaction + Affect)"
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
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--scene_branch_dim", type=int, default=16, help="Scene subspace projection dimension"
    )
    parser.add_argument(
        "--inter_branch_dim", type=int, default=32, help="Interaction subspace projection dimension"
    )
    parser.add_argument(
        "--affect_branch_dim", type=int, default=32, help="Affect subspace projection dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    fused_dim = args.scene_branch_dim + args.inter_branch_dim + args.affect_branch_dim
    behavioral_pct = ((args.inter_branch_dim + args.affect_branch_dim) / fused_dim) * 100.0
    scene_pct = (args.scene_branch_dim / fused_dim) * 100.0

    print("\n" + "=" * 70)
    print("🚀 MULTI-BRANCH BALANCED PIPELINE RUNNER")
    print(f"  • Selected Stage : {args.stage.upper()}")
    print(f"  • Modalities     : Scene(576) + Interaction(32) + Affect(8)")
    print(f"  • Subspaces      : Scene({args.scene_branch_dim}, {scene_pct:.1f}%) + Inter({args.inter_branch_dim}) + Affect({args.affect_branch_dim}) = {fused_dim}-dim (Behavioral: {behavioral_pct:.1f}%)")
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
    print("✅ MULTI-BRANCH PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
