import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import TemporalAttentionClassifier
from dataset import EngagementDataset

class DummyAutocast:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc_val, exc_tb): pass

def get_autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    elif device.type == "mps" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("mps")
        except Exception:
            return DummyAutocast()
    else:
        return DummyAutocast()

def calculate_class_weights(dataset):
    # Count frequency of each class in the dataset
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.item())
        
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    num_classes = len(classes)
    
    # Calculate inverse frequency weights
    weights = total / (num_classes * counts)
    print(f"Class counts: {dict(zip(classes, counts))}")
    print(f"Calculated class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)

def train(args):
    print("=" * 60)
    print("Starting Model Training Phase...")
    print("=" * 60)

    # Output directory for checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Device selection
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Initialize Datasets and Loaders
    train_dataset = EngagementDataset(os.path.join(args.data_dir, "train"))
    val_dataset = EngagementDataset(os.path.join(args.data_dir, "val"))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")

    # Calculate class weights for loss function to handle class imbalance
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Instantiate model
    model = TemporalAttentionClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_classes=3,
        dropout=args.dropout
    ).to(device)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Autocast context
    autocast_ctx = get_autocast_context(device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # -------------------------------------------------------------
        # Training loop
        # -------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast_ctx:
                logits = model(x)
                loss = criterion(logits, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += torch.sum(preds == y).item()
            train_total += x.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # -------------------------------------------------------------
        # Validation loop
        # -------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += torch.sum(preds == y).item()
                val_total += x.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        # Step scheduler
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")

        # Early Stopping & Best model checkpoint save
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': epoch_val_acc
            }, best_model_path)
            # Save raw model weights for simple evaluation loading
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_weights.pth"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print("=" * 60)
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="feature_matrices")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--input_dim", type=int, default=585)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    train(args)
