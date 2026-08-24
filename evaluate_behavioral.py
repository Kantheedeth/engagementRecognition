import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from model_behavioral import PureBehavioralAttentionClassifier
from dataset import EngagementDataset

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

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        print(f"Error: Test dataset directory '{test_dir}' not found!")
        return

    test_dataset = EngagementDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples.")

    model = PureBehavioralAttentionClassifier(
        dim_inter=args.dim_inter,
        dim_affect=args.dim_affect,
        branch_dim=args.branch_dim,
        num_heads=args.num_heads,
        num_classes=3
    ).to(device)

    weights_path = os.path.join(args.checkpoint_dir, "model_weights_behavioral.pth")
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at '{weights_path}'! Train the model first.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
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
    
    print("\n" + "-" * 55)
    print("CLASSIFICATION METRICS REPORT (PURE BEHAVIORAL FEATURES)")
    print("-" * 55)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Model Macro-F1 Score:     {model_f1*100:.2f}%")
    print(f"Baseline (Always Low) F1:  {baseline_f1*100:.2f}%")
    print("-" * 55 + "\n")

    plot_confusion_matrix(cm, class_names, filename=args.output_cm_file)
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="feature_matrices_behavioral")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--dim_inter", type=int, default=32)
    parser.add_argument("--dim_affect", type=int, default=8)
    parser.add_argument("--branch_dim", type=int, default=48)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_cm_file", default="confusion_matrix_behavioral.png")
    
    args = parser.parse_args()
    evaluate(args)
