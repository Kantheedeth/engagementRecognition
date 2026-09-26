import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_behavioral import PureBehavioralAttentionClassifier
from src.models.dataset import (
    EngagementDataset,
    feature_manifests_compatible,
    load_feature_manifest,
)
from src.data.feature_schema import BEHAVIORAL_FEATURE_SCHEMA

def plot_confusion_matrix(cm, classes, filename="confusion_matrix_behavioral.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Pure Behavioral Engagement Confusion Matrix",
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
    print("=" * 65)
    print("Running Pure Behavioral Pipeline Evaluation Phase...")
    print(f"  • Data Directory : {args.data_dir}")
    print(f"  • Features       : 32 Interaction + 8 Affect (Zero Scene)")
    print("=" * 65)

    requested_device = getattr(args, "device", "auto")
    device = torch.device(
        requested_device if requested_device != "auto" else
        ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    print(f"Using device: {device}")

    expected_shape = (8, args.dim_inter + args.dim_affect)
    feature_manifest = load_feature_manifest(
        args.data_dir,
        expected_schema=BEHAVIORAL_FEATURE_SCHEMA,
        expected_shape=expected_shape,
    )
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test dataset directory not found: {test_dir}")

    test_dataset = EngagementDataset(test_dir, expected_shape=expected_shape)
    if not test_dataset:
        raise RuntimeError(f"No test matrices found under {test_dir}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples.")

    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model_behavioral.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Train first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_manifest = checkpoint.get("feature_manifest")
    if not feature_manifests_compatible(checkpoint_manifest, feature_manifest):
        raise ValueError(
            "Checkpoint feature provenance does not match the current behavioral "
            "matrices. Rebuild, retrain, and then evaluate."
        )
    if checkpoint_manifest != feature_manifest:
        print(
            "Accepted legacy affect provenance: omitted ByteTrack defaults were "
            "verified as new_track_threshold=0.45 and track_buffer=8."
        )

    model_config = checkpoint.get("model_config", {})
    branch_dim = model_config.get("branch_dim", args.branch_dim)
    num_heads = model_config.get("num_heads", args.num_heads)
    dropout = model_config.get("dropout", 0.15)

    model = PureBehavioralAttentionClassifier(
        dim_inter=args.dim_inter,
        dim_affect=args.dim_affect,
        branch_dim=branch_dim,
        num_heads=num_heads,
        num_classes=3,
        dropout=dropout
    ).to(device)
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
    baseline_f1 = f1_score(y_true, baseline_preds, labels=[0, 1, 2], average='macro', zero_division=0)
    
    model_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    print("\n" + "-" * 55)
    print("CLASSIFICATION METRICS REPORT (PURE BEHAVIORAL FEATURES)")
    print("-" * 55)
    print(
        classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=class_names, zero_division=0
        )
    )
    print(f"Model Macro-F1 Score:     {model_f1*100:.2f}%")
    print(f"Baseline (Always Low) F1:  {baseline_f1*100:.2f}%")
    print("-" * 55 + "\n")

    plot_confusion_matrix(cm, class_names, filename=args.output_cm_file)
    if getattr(args, "output_metrics_file", None):
        metrics = {
            "macro_f1": float(model_f1), "accuracy": float(np.mean(y_true == y_pred)),
            "always_low_macro_f1": float(baseline_f1), "test_samples": len(y_true),
            "class_order": class_names, "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=[0, 1, 2], target_names=class_names,
                zero_division=0, output_dict=True,
            ),
            "checkpoint": os.path.abspath(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "feature_manifest": feature_manifest,
            "predictions": [{"matrix": path.name, "true_label": int(true), "predicted_label": int(pred)}
                            for path, true, pred in zip(test_dataset.file_paths, y_true, y_pred)],
        }
        with open(args.output_metrics_file, "w") as file:
            json.dump(metrics, file, indent=2)
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "feature_matrices_behavioral"))
    parser.add_argument("--checkpoint_dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--dim_inter", type=int, default=32)
    parser.add_argument("--dim_affect", type=int, default=8)
    parser.add_argument("--branch_dim", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_cm_file", default=os.path.join(PROJECT_ROOT, "confusion_matrix_behavioral.png"))
    parser.add_argument("--output_metrics_file", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    
    args = parser.parse_args()
    evaluate(args)
