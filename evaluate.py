import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from model import TemporalAttentionClassifier
from dataset import EngagementDataset

def plot_confusion_matrix(cm, classes, filename="confusion_matrix.png"):
    """Beautiful confusion matrix plotter using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Engagement Classification Confusion Matrix",
           ylabel="True Label",
           xlabel="Predicted Label")
           
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate counts inside matrix cells
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
    print("=" * 60)
    print("Running Pipeline Evaluation Phase...")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load test dataset
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        print(f"Error: Test dataset directory '{test_dir}' not found!")
        return

    test_dataset = EngagementDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples.")

    # Load model architecture
    model = TemporalAttentionClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=3
    ).to(device)

    # Load trained model weights
    weights_path = os.path.join(args.checkpoint_dir, "model_weights.pth")
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

    # Print baseline majority class guesser comparison (Class 0 is Low, count = 72)
    # class mapping: 0: Low, 1: Mid, 2: High
    class_names = ["Low", "Mid", "High"]
    
    # Calculate baseline
    baseline_preds = np.zeros_like(y_true) # always predict majority class (0)
    baseline_f1 = f1_score(y_true, baseline_preds, average='macro')
    
    # Calculate model metrics
    model_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "-" * 50)
    print("CLASSIFICATION METRICS REPORT")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Model Macro-F1 Score:     {model_f1*100:.2f}%")
    print(f"Baseline (Always Low) F1:  {baseline_f1*100:.2f}%")
    print("-" * 50 + "\n")

    # Save visual confusion matrix
    plot_confusion_matrix(cm, class_names, filename=args.output_cm_file)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="feature_matrices")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--input_dim", type=int, default=585)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_cm_file", default="confusion_matrix.png")
    
    args = parser.parse_args()
    evaluate(args)
