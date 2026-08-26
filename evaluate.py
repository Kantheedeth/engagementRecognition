import os
import argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from model import MultiBranchTemporalAttentionClassifier
from dataset import (
    EngagementDataset,
    feature_manifests_compatible,
    load_feature_manifest,
)
from feature_schema import MULTI_BRANCH_FEATURE_SCHEMA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_confusion_matrix(cm, classes, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Multi-Branch Engagement Confusion Matrix",
           ylabel="True Label",
           xlabel="Predicted Label")
           
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion matrix plot successfully saved to {filename}")

def evaluate(args):
    fused_dim = args.scene_branch_dim + args.inter_branch_dim + args.affect_branch_dim
    behavioral_pct = ((args.inter_branch_dim + args.affect_branch_dim) / fused_dim) * 100.0
    scene_pct = (args.scene_branch_dim / fused_dim) * 100.0

    print("=" * 60)
    print("Running Multi-Branch Pipeline Evaluation Phase...")
    print(f"  • Scene: {args.scene_branch_dim} ({scene_pct:.1f}%) | Interaction: {args.inter_branch_dim} | Affect: {args.affect_branch_dim} (Behavioral: {behavioral_pct:.1f}%)")
    print(f"  • Total Fused State: {fused_dim}-dim")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load test dataset only after validating matrix provenance.
    expected_shape = (8, args.dim_scene + args.dim_inter + args.dim_affect)
    feature_manifest = load_feature_manifest(
        args.data_dir,
        expected_schema=MULTI_BRANCH_FEATURE_SCHEMA,
        expected_shape=expected_shape,
    )
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        print(f"Error: Test dataset directory '{test_dir}' not found!")
        return

    test_dataset = EngagementDataset(test_dir, expected_shape=expected_shape)
    if not test_dataset:
        raise RuntimeError(f"No test matrices found under {test_dir}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples.")

    # Multi-Branch Attention Model
    model = MultiBranchTemporalAttentionClassifier(
        dim_scene=args.dim_scene,
        dim_inter=args.dim_inter,
        dim_affect=args.dim_affect,
        scene_branch_dim=args.scene_branch_dim,
        inter_branch_dim=args.inter_branch_dim,
        affect_branch_dim=args.affect_branch_dim,
        num_heads=args.num_heads,
        num_classes=3
    ).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'! Train the model first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_manifest = checkpoint.get("feature_manifest")
    if not feature_manifests_compatible(checkpoint_manifest, feature_manifest):
        raise ValueError(
            "Checkpoint feature provenance does not match the current matrices. "
            "Rebuild, retrain, and then evaluate."
        )
    if checkpoint_manifest != feature_manifest:
        print(
            "Accepted legacy affect provenance: omitted ByteTrack defaults were "
            "verified as new_track_threshold=0.45 and track_buffer=8."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_names = ["Low", "Mid", "High"]
    baseline_preds = np.zeros_like(y_true)
    baseline_f1 = f1_score(y_true, baseline_preds, average='macro')
    
    model_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "-" * 50)
    print("CLASSIFICATION METRICS REPORT (MULTI-BRANCH 16/32/32)")
    print("-" * 50)
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    )
    print(f"Model Macro-F1 Score:     {model_f1*100:.2f}%")
    print(f"Baseline (Always Low) F1:  {baseline_f1*100:.2f}%")
    print("-" * 50 + "\n")

    plot_confusion_matrix(cm, class_names, filename=args.output_cm_file)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "feature_matrices"))
    parser.add_argument("--checkpoint_dir", default=os.path.join(SCRIPT_DIR, "checkpoints"))
    parser.add_argument("--dim_scene", type=int, default=576)
    parser.add_argument("--dim_inter", type=int, default=32)
    parser.add_argument("--dim_affect", type=int, default=8)
    parser.add_argument("--scene_branch_dim", type=int, default=16)
    parser.add_argument("--inter_branch_dim", type=int, default=32)
    parser.add_argument("--affect_branch_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_cm_file", default=os.path.join(SCRIPT_DIR, "confusion_matrix.png"))
    
    args = parser.parse_args()
    evaluate(args)
