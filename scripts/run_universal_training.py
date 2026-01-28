#!/usr/bin/env python3
"""Universal training on combined Xenium + MERSCOPE datasets.

Uses MultiTissueDataset to combine multiple LMDB datasets with:
- sqrt-balanced sampling across platforms
- Cell Ontology label standardization
- Confidence-weighted training
"""

import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from dapidl.data.multi_tissue_dataset import (
    MultiTissueConfig,
    MultiTissueDataset,
    create_multi_tissue_splits,
)
from dapidl.models.classifier import CellTypeClassifier


def main():
    # Configuration
    OUTPUT_DIR = Path("./experiment_universal_xenium_merscope")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Training hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 1e-4
    BACKBONE = "efficientnetv2_rw_s"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Configure multi-tissue dataset
    config = MultiTissueConfig(
        sampling_strategy="sqrt",  # Square-root balanced sampling
        standardize_labels=True,    # Use Cell Ontology standardization
        use_hierarchical=False,     # Use flat labels for now
        min_samples_per_class=20,
    )

    # Add Xenium breast dataset
    config.add_dataset(
        path="/home/chrism/datasets/derived/xenium-breast-xenium-finegrained-p128",
        tissue="breast",
        platform="xenium",
        confidence_tier=2,  # Consensus annotations
        weight_multiplier=1.0,
    )

    # Add MERSCOPE breast dataset
    config.add_dataset(
        path="/home/chrism/datasets/raw/merscope/breast/pipeline_outputs/patches",
        tissue="breast",
        platform="merscope",
        confidence_tier=2,  # Consensus annotations
        weight_multiplier=1.0,
    )

    logger.info("Creating train/val/test splits...")

    # Create stratified splits
    train_ds, val_ds, test_ds = create_multi_tissue_splits(
        config,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    logger.info(f"Train: {len(train_ds)} samples")
    logger.info(f"Val: {len(val_ds)} samples")
    logger.info(f"Test: {len(test_ds)} samples")

    # Get class info
    num_classes = train_ds.num_classes
    class_names = train_ds.class_names
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = CellTypeClassifier(
        num_classes=num_classes,
        backbone_name=BACKBONE,
        pretrained=True,
        dropout=0.3,
    )
    model = model.to(device)

    logger.info(f"Model: {BACKBONE} with {num_classes} classes")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{EPOCHS} "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss /= n_batches

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        # Update scheduler
        scheduler.step()

        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, OUTPUT_DIR / "best_model.pt")
            logger.info(f"Saved best model with Val Acc: {val_acc:.2f}%")

    # Save training history
    with open(OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f)

    # Final test set computation
    model.eval()
    test_correct = 0
    test_total = 0

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * test_correct / test_total
    logger.info(f"Test Accuracy: {test_acc:.2f}%")

    # Save final results
    results = {
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "epochs": EPOCHS,
        "backbone": BACKBONE,
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
