import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_behavioral import PureBehavioralAttentionClassifier
from src.models.dataset import EngagementDataset, load_feature_manifest
from src.data.feature_schema import BEHAVIORAL_FEATURE_SCHEMA

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

import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dropped_interaction_indices(text, dim_inter):
    """Return (kept, dropped) indices for a matrix-preserving ablation."""
    if text is None or not str(text).strip():
        dropped = []
    else:
        try:
            dropped = [int(value.strip()) for value in str(text).split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError(
                "drop_interaction_indices must be comma-separated integers"
            ) from exc
    if len(dropped) != len(set(dropped)):
        raise ValueError("drop_interaction_indices must not contain duplicates")
    if any(index < 0 or index >= dim_inter for index in dropped):
        raise ValueError(
            f"drop_interaction_indices must be between 0 and {dim_inter - 1}"
        )
    kept = [index for index in range(dim_inter) if index not in set(dropped)]
    if not kept:
        raise ValueError("At least one interaction feature must remain")
    return kept, dropped

def calculate_class_weights(dataset, mode="smoothed"):
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(y.item())
        
    classes, counts = np.unique(labels, return_counts=True)
    if classes.tolist() != [0, 1, 2]:
        raise ValueError(f"Training split must contain classes 0, 1, and 2; got {classes}")
    total = len(labels)
    num_classes = len(classes)
    
    if mode == "smoothed":
        # Square-root inverse frequency (not effective-number class balancing).
        # Balances sensitivity without excessive false positives on minority class
        raw_weights = np.sqrt(total / (num_classes * counts))
        weights = raw_weights / np.mean(raw_weights)
    elif mode == "balanced":
        weights = total / (num_classes * counts)
    else:
        weights = np.ones(num_classes, dtype=np.float32)

    print(f"Class counts: {dict(zip(classes, counts))}")
    print(f"Calculated class weights ({mode}): {np.round(weights, 4)}")
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossMeter:
    """Aggregate weighted CE by its weight denominator, not the batch size."""
    def __init__(self):
        self.numerator = 0.0
        self.denominator = 0.0

    def update(self, loss, targets, class_weights):
        denominator = class_weights[targets].sum().item()
        self.numerator += loss.item() * denominator
        self.denominator += denominator

    @property
    def mean(self):
        return self.numerator / self.denominator

def train(args):
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    interaction_indices, dropped_interaction_indices = parse_dropped_interaction_indices(
        getattr(args, "drop_interaction_indices", ""), args.dim_inter
    )

    print("=" * 65)
    print("Pure Behavioral Engagement Training (ZERO Scene Shortcut)")
    print(f"  • Branch Mode        : {args.branch_mode}")
    if args.branch_mode in ("both", "interaction"):
        print(f"  • Interaction Branch : {len(interaction_indices)} -> {args.branch_dim}")
        print(f"  • Interaction Kept   : {interaction_indices}")
        print(f"  • Interaction Dropped: {dropped_interaction_indices}")
    if args.branch_mode in ("both", "affect"):
        print(f"  • Affect Branch      : {args.dim_affect} -> {args.branch_dim}")
    embed_dim = args.branch_dim * (2 if args.branch_mode == "both" else 1)
    print(f"  • Temporal Embed Dim : {embed_dim}")
    print("=" * 65)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    requested_device = getattr(args, "device", "auto")
    device = torch.device(
        requested_device if requested_device != "auto" else
        ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    print(f"Using device: {device}")

    # Load from feature_matrices_behavioral after validating provenance.
    expected_shape = (8, args.dim_inter + args.dim_affect)
    feature_manifest = load_feature_manifest(
        args.data_dir,
        expected_schema=BEHAVIORAL_FEATURE_SCHEMA,
        expected_shape=expected_shape,
    )
    train_dataset = EngagementDataset(
        os.path.join(args.data_dir, "train"), expected_shape=expected_shape
    )
    val_dataset = EngagementDataset(
        os.path.join(args.data_dir, "val"), expected_shape=expected_shape
    )
    if not train_dataset or not val_dataset:
        raise RuntimeError("Training and validation splits must both contain matrices")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")

    # Weighted CrossEntropyLoss
    class_weights = calculate_class_weights(train_dataset, mode=args.class_weight_mode).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=getattr(args, "label_smoothing", 0.0)
    )

    # Pure Behavioral Model
    model = PureBehavioralAttentionClassifier(
        dim_inter=args.dim_inter,
        dim_affect=args.dim_affect,
        branch_dim=args.branch_dim,
        num_heads=args.num_heads,
        num_classes=3,
        dropout=args.dropout,
        branch_mode=args.branch_mode,
        interaction_indices=interaction_indices,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    autocast_ctx = get_autocast_context(device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = WeightedLossMeter()
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

            train_loss.update(loss, y, class_weights)
            preds = torch.argmax(logits, dim=1)
            train_correct += torch.sum(preds == y).item()
            train_total += x.size(0)

        epoch_train_loss = train_loss.mean
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = WeightedLossMeter()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss.update(loss, y, class_weights)
                preds = torch.argmax(logits, dim=1)
                val_correct += torch.sum(preds == y).item()
                val_total += x.size(0)

        epoch_val_loss = val_loss.mean
        epoch_val_acc = val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")
        history.append({"epoch": epoch, "train_loss": epoch_train_loss,
                        "train_accuracy": epoch_train_acc, "val_loss": epoch_val_loss,
                        "val_accuracy": epoch_val_acc})
        with open(os.path.join(args.checkpoint_dir, "training_history.json"), "w") as file:
            json.dump(history, file, indent=2)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_path = os.path.join(args.checkpoint_dir, "best_model_behavioral.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': epoch_val_acc,
                'feature_manifest': feature_manifest,
                'training_config': {key: str(value) if isinstance(value, os.PathLike) else value
                                    for key, value in vars(args).items()},
                'class_weights': class_weights.detach().cpu().tolist(),
                'model_config': {
                    'dim_inter': args.dim_inter,
                    'dim_affect': args.dim_affect,
                    'branch_dim': args.branch_dim,
                    'num_heads': args.num_heads,
                    'dropout': args.dropout,
                    'branch_mode': args.branch_mode,
                    'interaction_indices': interaction_indices,
                    'drop_interaction_indices': dropped_interaction_indices,
                },
            }, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print("=" * 65)
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "feature_matrices_behavioral"))
    parser.add_argument("--checkpoint_dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--dim_inter", type=int, default=32)
    parser.add_argument("--dim_affect", type=int, default=8)
    parser.add_argument("--branch_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--branch_mode", choices=["both", "interaction", "affect"], default="both")
    parser.add_argument(
        "--drop_interaction_indices", default="",
        help="Comma-separated zero-based interaction columns to omit without rebuilding matrices",
    )
    parser.add_argument("--class_weight_mode", choices=["smoothed", "balanced", "none"], default="smoothed",
                        help="smoothed (square-root, balanced), balanced (inverse freq), or none")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Optional CE label smoothing; keep 0 for the initial baseline")
    
    args = parser.parse_args()
    train(args)
