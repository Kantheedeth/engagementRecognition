"""V2 training and evaluation around the unchanged legacy behavioral model."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from experiments_v2.core.artifacts import (
    create_exclusive_dir,
    file_size_mb,
    new_id,
    utc_now,
    write_json_exclusive,
)
from experiments_v2.core.contracts import ModelArtifact, PairDefinition
from experiments_v2.pipeline.engagement_model import create_engagement_model
from experiments_v2.registry.model_registry import ModelRegistry


def _load_ml_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader

        from src.models.dataset import EngagementDataset
        from src.training.train_behavioral import (
            calculate_class_weights,
            get_autocast_context,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The existing project ML dependencies are required for engagement "
            "training: numpy, torch, and scikit-learn"
        ) from exc
    return locals()


def _select_device(torch: Any) -> Any:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def train_engagement(
    *,
    run_id: str,
    run_dir: Path,
    pair: PairDefinition,
    matrix_dir: Path,
    pair_manifest: Mapping[str, Any],
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    model_registry: ModelRegistry,
    git_commit: str | None,
) -> tuple[ModelArtifact, dict[str, Any]]:
    deps = _load_ml_dependencies()
    np = deps["np"]
    torch = deps["torch"]
    nn = deps["nn"]
    DataLoader = deps["DataLoader"]
    AdamW = deps["AdamW"]
    CosineAnnealingLR = deps["CosineAnnealingLR"]
    EngagementDataset = deps["EngagementDataset"]
    calculate_class_weights = deps["calculate_class_weights"]
    get_autocast_context = deps["get_autocast_context"]
    set_seed = deps["set_seed"]

    seed = int(training_config["seed"])
    set_seed(seed)
    device = _select_device(torch)
    expected_shape = (pair.temporal_frames, pair.matrix_dim)
    train_dataset = EngagementDataset(matrix_dir / "train", expected_shape=expected_shape)
    val_dataset = EngagementDataset(matrix_dir / "val", expected_shape=expected_shape)
    if not train_dataset or not val_dataset:
        raise RuntimeError("Training and validation pair matrices must not be empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
    )
    class_weights = calculate_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model, resolved_model_config = create_engagement_model(
        pair=pair,
        model_config=model_config,
    )
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    autocast_context = get_autocast_context(device)

    model_id = new_id("MODEL")
    checkpoint_dir = create_exclusive_dir(run_dir / "checkpoints" / model_id)
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_accuracy = 0.0
    best_checkpoint: Path | None = None
    patience_counter = 0
    history = []
    _synchronize(torch, device)
    training_started = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast_context:
                logits = model(features)
                loss = criterion(logits, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=1)
            train_correct += torch.sum(predictions == targets).item()
            train_total += features.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                with autocast_context:
                    logits = model(features)
                    loss = criterion(logits, targets)
                val_loss += loss.item() * features.size(0)
                predictions = torch.argmax(logits, dim=1)
                val_correct += torch.sum(predictions == targets).item()
                val_total += features.size(0)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_correct / val_total,
        }
        history.append(epoch_metrics)
        scheduler.step()

        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            best_val_accuracy = epoch_metrics["val_accuracy"]
            best_epoch = epoch
            patience_counter = 0
            candidate = checkpoint_dir / f"best_epoch_{epoch:03d}.pt"
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {candidate}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_id": model_id,
                    "run_id": run_id,
                    "pair_id": pair.pair_id,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": best_val_accuracy,
                    "pair_manifest": dict(pair_manifest),
                    "model_config": resolved_model_config,
                    "training_config": dict(training_config),
                },
                candidate,
            )
            best_checkpoint = candidate
        else:
            patience_counter += 1
        if patience_counter >= int(training_config["patience"]):
            break

    _synchronize(torch, device)
    training_seconds = perf_counter() - training_started
    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a validation checkpoint")

    training_result = {
        "model_id": model_id,
        "device": str(device),
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_val_accuracy,
        "training_seconds": training_seconds,
        "parameter_count": parameter_count,
        "checkpoint_path": str(best_checkpoint.resolve()),
        "checkpoint_size_mb": file_size_mb(best_checkpoint),
        "model_config": resolved_model_config,
        "training_config": dict(training_config),
        "history": history,
        "completed_at": utc_now(),
    }
    write_json_exclusive(run_dir / "training.json", training_result)
    artifact = model_registry.register_engagement_checkpoint(
        model_id=model_id,
        pair_id=pair.pair_id,
        run_id=run_id,
        checkpoint_path=best_checkpoint,
        model_config=resolved_model_config,
        parameter_count=parameter_count,
        validation_metric={
            "best_epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_accuracy": best_val_accuracy,
        },
        git_commit=git_commit,
    )
    return artifact, training_result


def evaluate_engagement(
    *,
    run_dir: Path,
    pair: PairDefinition,
    matrix_dir: Path,
    engagement_model: ModelArtifact,
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    deps = _load_ml_dependencies()
    np = deps["np"]
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    EngagementDataset = deps["EngagementDataset"]
    accuracy_score = deps["accuracy_score"]
    confusion_matrix = deps["confusion_matrix"]
    precision_recall_fscore_support = deps["precision_recall_fscore_support"]

    device = _select_device(torch)
    checkpoint_path = Path(str(engagement_model.manifest["checkpoint_path"]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model, resolved_model_config = create_engagement_model(
        pair=pair,
        model_config=model_config,
    )
    if resolved_model_config != model_config:
        raise ValueError("Checkpoint engagement input contract is incomplete or stale")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    expected_shape = (pair.temporal_frames, pair.matrix_dim)
    test_dataset = EngagementDataset(matrix_dir / "test", expected_shape=expected_shape)
    if not test_dataset:
        raise RuntimeError("Test pair matrices must not be empty")
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(evaluation_config["batch_size"]),
        shuffle=False,
    )

    warmup_batches = int(evaluation_config.get("warmup_batches", 1))
    if warmup_batches > 0:
        with torch.no_grad():
            for index, (features, _) in enumerate(test_loader):
                model(features.to(device))
                if index + 1 >= warmup_batches:
                    break
    _synchronize(torch, device)
    inference_started = perf_counter()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for features, targets in test_loader:
            logits = model(features.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(int(value) for value in targets.numpy())
            y_pred.extend(int(value) for value in predictions)
    _synchronize(torch, device)
    inference_seconds = perf_counter() - inference_started

    labels = [0, 1, 2]
    class_names = ["Low", "Mid", "High"]
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    class_metrics = {
        class_names[index]: {
            "precision": float(per_class[0][index]),
            "recall": float(per_class[1][index]),
            "f1": float(per_class[2][index]),
            "support": int(per_class[3][index]),
        }
        for index in range(len(labels))
    }
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": class_metrics,
        "confusion_matrix": confusion.astype(int).tolist(),
        "test_samples": len(test_dataset),
        "inference_seconds": inference_seconds,
        "inference_ms_per_video": inference_seconds * 1000.0 / len(test_dataset),
        "engagement_fps": len(test_dataset) / inference_seconds if inference_seconds else None,
        "device": str(device),
        "warmup_batches": warmup_batches,
    }
    write_json_exclusive(
        run_dir / "confusion_matrix.json",
        {"labels": class_names, "matrix": result["confusion_matrix"]},
    )
    with (run_dir / "predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "true_label", "predicted_label"])
        writer.writerows((index, truth, prediction) for index, (truth, prediction) in enumerate(zip(y_true, y_pred)))
    _save_confusion_plot(run_dir / "confusion_matrix.png", confusion, class_names)
    return result


def _save_confusion_plot(path: Path, matrix: Any, class_names: list[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite confusion matrix plot: {path}")
    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Greens)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="V2 Affect x Interaction Engagement Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(int(matrix[row, column]), "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
