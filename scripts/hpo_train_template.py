#!/usr/bin/env python3
"""HPO Template Training Script for DAPIDL.

This script is used as a template by ClearML HyperParameterOptimizer.
Hyperparameters are connected to ClearML and modified by the optimizer.

Usage (for creating template task):
    uv run python scripts/hpo_train_template.py --create-template

Usage (for local testing):
    uv run python scripts/hpo_train_template.py --local
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from clearml import Logger, Task

from dapidl.models.classifier import CellTypeClassifier
from dapidl.hpo.search_space import resolve_dataset_path, get_valid_batch_size, get_num_classes

# Coarse class mapping
COARSE_MAPPING = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "Mast_Cells": "Immune",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
    "Stromal": "Stromal",
}
COARSE_CLASSES = ["Epithelial", "Immune", "Stromal"]


def remap_labels_to_coarse(
    labels: np.ndarray, class_mapping: dict[str, int]
) -> tuple[np.ndarray, dict[str, int]]:
    """Remap fine-grained labels to coarse labels."""
    idx_to_name = {v: k for k, v in class_mapping.items()}
    coarse_class_mapping = {name: i for i, name in enumerate(COARSE_CLASSES)}

    coarse_labels = np.zeros_like(labels)
    for idx in range(len(labels)):
        fine_name = idx_to_name[labels[idx]]
        coarse_name = COARSE_MAPPING.get(fine_name, "Stromal")
        coarse_labels[idx] = coarse_class_mapping[coarse_name]

    return coarse_labels, coarse_class_mapping


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        if isinstance(batch, dict):
            images = batch["data"].to(device)
            labels = batch["label"].squeeze().to(device)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """Run model on validation/test set and return metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            if isinstance(batch, dict):
                images = batch["data"].to(device)
                labels = batch["label"].squeeze().to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": conf_matrix,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HPO Template Training")
    parser.add_argument("--create-template", action="store_true",
                        help="Create template task and exit")
    parser.add_argument("--local", action="store_true",
                        help="Run locally without ClearML remote execution")
    args = parser.parse_args()

    # Initialize ClearML task
    task = Task.init(
        project_name="DAPIDL/HPO",
        task_name="HPO-Template",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )

    # Define hyperparameters (these will be overwritten by HPO)
    config = {
        # Dataset selection
        "centering": "cellpose",
        "granularity": "coarse",
        "patch_size": 128,

        # Model architecture
        "backbone": "efficientnetv2_rw_s",
        "dropout": 0.3,

        # Training hyperparameters
        "batch_size": 64,
        "learning_rate": 3e-4,
        "max_weight_ratio": 10.0,

        # Fixed parameters
        "epochs": 50,
        "warmup_epochs": 5,
        "early_stopping_patience": 15,
        "label_smoothing": 0.1,
        "weight_decay": 1e-5,
    }

    # Connect to ClearML (HPO will modify these)
    task.connect(config, name="General")

    if args.create_template:
        print("Template task created. Use this task ID for HPO.")
        print(f"Task ID: {task.id}")
        task.close()
        return

    logger = Logger.current_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve dataset path from hyperparameters
    try:
        dataset_path = resolve_dataset_path(
            centering=config["centering"],
            granularity=config["granularity"],
            patch_size=config["patch_size"],
        )
    except FileNotFoundError:
        dataset_name = f"xenium-breast-{config['centering']}-{config['granularity']}-p{config['patch_size']}"
        dataset_path = Path("/mnt/work/git/dapidl/datasets") / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Using dataset: {dataset_path}")

    # Check batch size for backbone
    effective_batch_size = get_valid_batch_size(config["backbone"], config["batch_size"])
    if effective_batch_size != config["batch_size"]:
        print(f"Adjusted batch size from {config['batch_size']} to {effective_batch_size} for {config['backbone']}")
        config["batch_size"] = effective_batch_size

    # Load dataset metadata
    with open(dataset_path / "class_mapping.json") as f:
        class_mapping = json.load(f)

    labels = np.load(dataset_path / "labels.npy")

    # Handle coarse vs finegrained
    if config["granularity"] == "coarse":
        is_already_coarse = len(class_mapping) == 3 and set(class_mapping.keys()) == set(COARSE_CLASSES)
        if not is_already_coarse:
            labels, class_mapping = remap_labels_to_coarse(labels, class_mapping)
        class_names = COARSE_CLASSES
        num_classes = 3
    else:
        class_names = list(class_mapping.keys())
        num_classes = len(class_names)

    print(f"\nConfiguration:")
    print(f"  Centering: {config['centering']}")
    print(f"  Granularity: {config['granularity']}")
    print(f"  Patch size: {config['patch_size']}")
    print(f"  Backbone: {config['backbone']}")
    print(f"  Classes: {num_classes}")

    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        if u < len(class_names):
            print(f"  {class_names[u]}: {c:,} ({c/len(labels)*100:.1f}%)")
            logger.report_scalar("Class Distribution", class_names[u], c, iteration=0)

    # Create effective dataset directory
    temp_dir = Path(tempfile.mkdtemp(prefix="dapidl_hpo_"))
    print(f"Temp dataset: {temp_dir}")

    lmdb_path = dataset_path / "patches.lmdb"
    (temp_dir / "patches.lmdb").symlink_to(lmdb_path)
    np.save(temp_dir / "labels.npy", labels)

    with open(temp_dir / "class_mapping.json", "w") as f:
        json.dump({name: i for i, name in enumerate(class_names)}, f)

    shutil.copy(dataset_path / "normalization_stats.json", temp_dir / "normalization_stats.json")

    # Create dataloaders
    from dapidl.data.dataset import create_dataloaders_with_backend

    train_loader, val_loader, test_loader, metadata = create_dataloaders_with_backend(
        data_path=temp_dir,
        batch_size=config["batch_size"],
        backend="dali-lmdb",
        num_workers=8,
        seed=42,
        device_id=0,
    )

    # Compute class weights
    class_weights = torch.tensor(metadata["class_weights"], dtype=torch.float32)
    class_weights = class_weights / class_weights.min()
    class_weights = torch.minimum(class_weights, torch.tensor(config["max_weight_ratio"]))
    class_weights = class_weights.to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")

    # Create model
    model = CellTypeClassifier(
        backbone_name=config["backbone"],
        num_classes=num_classes,
        pretrained=True,
        dropout=config["dropout"],
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] - config["warmup_epochs"],
    )

    # Training loop
    best_val_f1 = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        # Warmup learning rate
        if epoch < config["warmup_epochs"]:
            warmup_factor = (epoch + 1) / config["warmup_epochs"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = config["learning_rate"] * warmup_factor

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Check on validation set
        val_metrics = run_validation(model, val_loader, criterion, device, class_names)

        if epoch >= config["warmup_epochs"]:
            scheduler.step()

        # Log metrics - IMPORTANT: use "val" title for HPO objective
        logger.report_scalar("train", "loss", train_loss, iteration=epoch)
        logger.report_scalar("train", "accuracy", train_acc, iteration=epoch)
        logger.report_scalar("val", "loss", val_metrics["loss"], iteration=epoch)
        logger.report_scalar("val", "accuracy", val_metrics["accuracy"], iteration=epoch)
        logger.report_scalar("val", "macro_f1", val_metrics["macro_f1"], iteration=epoch)
        logger.report_scalar("lr", "value", optimizer.param_groups[0]["lr"], iteration=epoch)

        # Log per-class F1
        for class_name in class_names:
            class_f1 = val_metrics["per_class"].get(class_name, {}).get("f1-score", 0)
            logger.report_scalar("val_per_class_f1", class_name, class_f1, iteration=epoch)

        print(
            f"Epoch {epoch+1}/{config['epochs']}: "
            f"train_loss={train_loss:.4f}, val_f1={val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"  -> New best! F1={best_val_f1:.4f}")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Log confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.report_confusion_matrix(
                title="Confusion Matrix",
                series=f"epoch_{epoch+1}",
                matrix=val_metrics["confusion_matrix"],
                xlabels=class_names,
                ylabels=class_names,
                iteration=epoch,
            )

    # Load best model for test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test run
    print("\nRunning final test...")
    test_metrics = run_validation(model, test_loader, criterion, device, class_names)

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")

    # Log final metrics
    logger.report_single_value("test_accuracy", test_metrics["accuracy"])
    logger.report_single_value("test_macro_f1", test_metrics["macro_f1"])
    logger.report_single_value("best_val_f1", best_val_f1)

    # Log final confusion matrix
    logger.report_confusion_matrix(
        title="Test Confusion Matrix",
        series="final",
        matrix=test_metrics["confusion_matrix"],
        xlabels=class_names,
        ylabels=class_names,
        iteration=config["epochs"],
    )

    # Generate and save sample predictions
    try:
        from dapidl.hpo.visualization import (
            generate_sample_predictions,
            create_class_prediction_grid,
            save_prediction_grid,
            compute_per_class_metrics,
        )

        print("\nGenerating sample predictions...")
        samples = generate_sample_predictions(
            model, test_loader, class_names,
            n_samples_per_class=5, device=str(device)
        )

        grid = create_class_prediction_grid(
            samples, class_names,
            patch_size=config["patch_size"]
        )

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        grid_path = output_dir / "prediction_grid.png"
        save_prediction_grid(grid, grid_path, task)
        print(f"Saved prediction grid to {grid_path}")

    except Exception as e:
        print(f"Warning: Could not generate prediction grid: {e}")

    # Save model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_model_state,
        "config": config,
        "class_names": class_names,
        "best_val_f1": best_val_f1,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        },
    }, model_path)

    task.upload_artifact(name="best_model", artifact_object=model_path)

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\nTraining complete!")
    task.close()


if __name__ == "__main__":
    main()
