"""HEIST Trainer for supervised cell type classification.

This module provides the training loop for HEIST, including:
- Supervised cross-entropy loss with class weighting
- W&B logging
- Early stopping and checkpointing
- Learning rate scheduling

No pretraining - directly trains for classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dapidl.data.heist_dataset import HEISTDataset, heist_collate_fn
from dapidl.models.heist import HEISTClassifier

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class HEISTTrainer:
    """Trainer for HEIST classifier.

    Args:
        model: HEISTClassifier model.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Optional test dataset.
        grn_edge_index: GRN edges to use for all batches.
        output_dir: Directory for checkpoints and logs.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        epochs: Number of training epochs.
        batch_size: Number of partitions per batch.
        patience: Early stopping patience.
        class_weights: Optional class weights for loss.
        device: Device to train on.
        use_wandb: Whether to log to W&B.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
    """

    def __init__(
        self,
        model: HEISTClassifier,
        train_dataset: HEISTDataset,
        val_dataset: HEISTDataset,
        test_dataset: HEISTDataset | None = None,
        grn_edge_index: torch.Tensor | None = None,
        output_dir: str | Path = "./heist_output",
        learning_rate: float = 1e-3,
        weight_decay: float = 3e-3,
        epochs: int = 50,
        batch_size: int = 4,
        patience: int = 10,
        class_weights: torch.Tensor | None = None,
        device: str = "cuda",
        use_wandb: bool = True,
        wandb_project: str = "dapidl-heist",
        wandb_run_name: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=heist_collate_fn,
            num_workers=0,  # Graph data doesn't parallelize well
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=heist_collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=heist_collate_fn,
                num_workers=0,
                pin_memory=True,
            )

        # GRN for all batches
        self.grn_edge_index = grn_edge_index
        if self.grn_edge_index is not None:
            self.grn_edge_index = self.grn_edge_index.to(device)

        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01,
        )

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training state
        self.epochs = epochs
        self.patience = patience
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        # W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model": "HEIST",
                    "n_genes": model.n_genes,
                    "hidden_dim": model.hidden_dim,
                    "n_layers": model.n_layers,
                    "num_classes": model.num_classes,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "batch_size": batch_size,
                },
            )

        # Training log
        self.training_log = {
            "train_losses": [],
            "val_losses": [],
            "val_f1_history": [],
            "best_val_f1": 0.0,
            "best_epoch": 0,
        }

        logger.info(
            f"HEISTTrainer: {len(train_dataset)} train partitions, "
            f"{len(val_dataset)} val partitions, lr={learning_rate}"
        )

    def train(self) -> dict[str, Any]:
        """Run training loop.

        Returns:
            Training metrics dict.
        """
        logger.info(f"Starting HEIST training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            # Training epoch
            train_loss = self._train_epoch()
            self.training_log["train_losses"].append(train_loss)

            # Validation
            val_loss, val_f1, val_acc = self._validate()
            self.training_log["val_losses"].append(val_loss)
            self.training_log["val_f1_history"].append(val_f1)

            # Learning rate step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_f1={val_f1:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}"
            )

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                })

            # Early stopping and checkpointing
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.training_log["best_val_f1"] = val_f1
                self.training_log["best_epoch"] = epoch + 1
                self.epochs_without_improvement = 0

                # Save best model
                self.model.save_checkpoint(
                    self.output_dir / "best_model.pt",
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics={"val_f1": val_f1, "val_acc": val_acc},
                )
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        # Save final model
        self.model.save_checkpoint(
            self.output_dir / "final_model.pt",
            optimizer=self.optimizer,
            epoch=self.epochs,
            metrics={"val_f1": self.best_val_f1},
        )

        # Run test if available
        if self.test_loader is not None:
            test_metrics = self._test()
            self.training_log["test_metrics"] = test_metrics
            logger.info(
                f"Test results: F1={test_metrics['f1']:.4f}, "
                f"Acc={test_metrics['accuracy']:.4f}"
            )

        # Save training log
        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)

        if self.use_wandb:
            wandb.finish()

        return self.training_log

    def _train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            expression = batch["expression"].to(self.device)
            coords = batch["coords"].to(self.device)
            labels = batch["labels"].to(self.device)
            spatial_edge_index = batch["spatial_edge_index"].to(self.device)

            # Get GRN (use provided or empty)
            grn = self.grn_edge_index
            if grn is None:
                grn = torch.zeros((2, 0), dtype=torch.long, device=self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(expression, coords, spatial_edge_index, grn)

            # Loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self) -> tuple[float, float, float]:
        """Run validation.

        Returns:
            Tuple of (loss, f1, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                expression = batch["expression"].to(self.device)
                coords = batch["coords"].to(self.device)
                labels = batch["labels"].to(self.device)
                spatial_edge_index = batch["spatial_edge_index"].to(self.device)

                grn = self.grn_edge_index
                if grn is None:
                    grn = torch.zeros((2, 0), dtype=torch.long, device=self.device)

                logits = self.model(expression, coords, spatial_edge_index, grn)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        avg_loss = total_loss / max(len(self.val_loader), 1)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        accuracy = (all_preds == all_labels).mean()

        return avg_loss, f1, accuracy

    def _test(self) -> dict[str, Any]:
        """Run test set inference.

        Returns:
            Dict with test metrics.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                expression = batch["expression"].to(self.device)
                coords = batch["coords"].to(self.device)
                labels = batch["labels"].to(self.device)
                spatial_edge_index = batch["spatial_edge_index"].to(self.device)

                grn = self.grn_edge_index
                if grn is None:
                    grn = torch.zeros((2, 0), dtype=torch.long, device=self.device)

                logits = self.model(expression, coords, spatial_edge_index, grn)
                preds = logits.argmax(dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        accuracy = (all_preds == all_labels).mean()

        # Per-class report
        report = classification_report(
            all_labels, all_preds, zero_division=0, output_dict=True
        )

        return {
            "f1": float(f1),
            "accuracy": float(accuracy),
            "classification_report": report,
        }


def compute_class_weights(
    labels: np.ndarray,
    max_weight_ratio: float = 10.0,
) -> torch.Tensor:
    """Compute class weights for imbalanced data.

    Args:
        labels: (N,) class labels.
        max_weight_ratio: Maximum ratio between largest and smallest weight.

    Returns:
        (num_classes,) weight tensor.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)

    # Inverse frequency weights
    weights = n_samples / (n_classes * counts)

    # Cap weights
    min_weight = weights.min()
    max_weight = min_weight * max_weight_ratio
    weights = np.clip(weights, None, max_weight)

    # Normalize
    weights = weights / weights.sum() * n_classes

    # Create full weight tensor
    full_weights = torch.zeros(unique.max() + 1, dtype=torch.float32)
    for cls, w in zip(unique, weights, strict=True):
        full_weights[cls] = w

    logger.info(f"Class weights (capped at {max_weight_ratio}x): {weights}")
    return full_weights
