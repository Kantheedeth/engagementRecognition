import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
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
from src.data.split_integrity import audit_split_integrity

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

    split_integrity = audit_split_integrity(
        Path(args.csv_dir), Path(args.group_manifest)
    )
    print("Strict video/session/golden-pair split integrity verified.")
    print("Running Pure Behavioral Pipeline Evaluation Phase...")
    print(f"  • Data Directory : {args.data_dir}")
    print(f"  • Features       : 40 Track Interaction + 8 Affect (Zero Scene)")
    print("=" * 65)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    expected_shape = (8, args.dim_inter + args.dim_affect)
    feature_manifest = load_feature_manifest(
        args.data_dir,
        expected_schema=BEHAVIORAL_FEATURE_SCHEMA,
        expected_shape=expected_shape,
    )
    if feature_manifest.get("split_csv_sha256") != split_integrity["csv_sha256"]:
        raise ValueError(
            "Behavioral matrices were not built from the currently audited split "
            "CSVs. Rebuild the matrices before evaluation."
        )
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test dataset directory not found: {test_dir}")

    test_dataset = EngagementDataset(test_dir, expected_shape=expected_shape)
    if not test_dataset:
        raise RuntimeError(f"No test matrices found under {test_dir}")
    train_dataset = EngagementDataset(
        os.path.join(args.data_dir, "train"), expected_shape=expected_shape
    )
    if not train_dataset:
        raise RuntimeError("Training matrices are required to compute the baseline")
    recorded_counts = feature_manifest.get("split_counts", {})
    actual_counts = {"train": len(train_dataset), "test": len(test_dataset)}
    if any(recorded_counts.get(split) != count for split, count in actual_counts.items()):
        raise ValueError(
            "Matrix counts do not match the build manifest; rebuild before evaluation. "
            f"recorded={recorded_counts}, actual={actual_counts}"
        )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples.")

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. Train the new schema first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("split_integrity") != split_integrity:
        raise ValueError(
            "Checkpoint split-integrity evidence does not match the current CSVs "
            "and group manifest. Evaluation was stopped."
        )
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
    expected_model_config = {
        "dim_inter": args.dim_inter,
        "dim_affect": args.dim_affect,
        "branch_dim": args.branch_dim,
        "num_heads": args.num_heads,
    }
    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, dict) or any(
        checkpoint_model_config.get(key) != value
        for key, value in expected_model_config.items()
    ):
        raise ValueError(
            "Checkpoint architecture does not match the requested interaction, "
            "affect, branch, or attention dimensions."
        )
    model = PureBehavioralAttentionClassifier(
        dim_inter=args.dim_inter,
        dim_affect=args.dim_affect,
        branch_dim=args.branch_dim,
        num_heads=args.num_heads,
        num_classes=3,
        dropout=float(checkpoint_model_config.get("dropout", 0.15)),
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
    train_labels = np.asarray(
        [train_dataset[index][1].item() for index in range(len(train_dataset))]
    )
    baseline_class = int(np.bincount(train_labels, minlength=3).argmax())
    baseline_preds = np.full_like(y_true, baseline_class)
    baseline_f1 = f1_score(
        y_true, baseline_preds, labels=[0, 1, 2], average="macro", zero_division=0
    )
    
    model_f1 = f1_score(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    model_accuracy = accuracy_score(y_true, y_pred)
    model_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    ordinal_mae = mean_absolute_error(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    print("\n" + "-" * 55)
    print("CLASSIFICATION METRICS REPORT (PURE BEHAVIORAL FEATURES)")
    print("-" * 55)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=class_names,
            zero_division=0,
        )
    )
    print(f"Model Macro-F1 Score:     {model_f1*100:.2f}%")
    print(f"Model Accuracy:           {model_accuracy*100:.2f}%")
    print(f"Balanced Accuracy:        {model_balanced_accuracy*100:.2f}%")
    print(f"Ordinal MAE (0/1/2):      {ordinal_mae:.4f}")
    print(
        f"Train-majority baseline ({class_names[baseline_class]}) Macro-F1: "
        f"{baseline_f1*100:.2f}%"
    )
    print("-" * 55 + "\n")

    plot_confusion_matrix(cm, class_names, filename=args.output_cm_file)
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(PROJECT_ROOT, "feature_matrices_behavioral_track"),
    )
    parser.add_argument("--checkpoint_dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--checkpoint_name", default="best_model_behavioral_track.pth")
    parser.add_argument(
        "--group_manifest",
        required=True,
        help="CSV containing video_path,session_id,golden_pair_id for split auditing",
    )
    parser.add_argument(
        "--csv_dir",
        default=PROJECT_ROOT,
        help="Directory containing the exact train.csv, val.csv, and test.csv files",
    )
    parser.add_argument("--dim_inter", type=int, default=40)
    parser.add_argument("--dim_affect", type=int, default=8)
    parser.add_argument("--branch_dim", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_cm_file",
        default=os.path.join(PROJECT_ROOT, "confusion_matrix_behavioral_track.png"),
    )
    
    args = parser.parse_args()
    evaluate(args)
