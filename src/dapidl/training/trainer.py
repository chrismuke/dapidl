"""Training loop for DAPIDL."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from dapidl.data.dataset import DAPIDLDataset, create_data_splits, create_dataloaders
from dapidl.models.classifier import CellTypeClassifier
from dapidl.training.losses import get_class_weights


class Trainer:
    """Training orchestrator for DAPIDL.

    Handles training loop, validation, checkpointing, and W&B logging.
    """

    def __init__(
        self,
        data_path: str | Path,
        output_path: str | Path = "outputs",
        # Model params
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        # Training params
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Scheduler params
        warmup_epochs: int = 5,
        # Other params
        use_wandb: bool = True,
        project_name: str = "dapidl",
        seed: int = 42,
        num_workers: int = 4,
        early_stopping_patience: int = 15,
        label_smoothing: float = 0.1,
        use_class_weights: bool = True,
    ) -> None:
        """Initialize trainer.

        Args:
            data_path: Path to prepared dataset
            output_path: Path for outputs (checkpoints, logs)
            backbone_name: Name of timm backbone
            pretrained: Use pretrained weights
            dropout: Dropout probability
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            use_wandb: Enable W&B logging
            project_name: W&B project name
            seed: Random seed
            num_workers: DataLoader workers
            early_stopping_patience: Epochs to wait before stopping
            label_smoothing: Label smoothing factor
            use_class_weights: Use class weights for loss
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.seed = seed
        self.num_workers = num_workers
        self.early_stopping_patience = early_stopping_patience
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights

        # Model params
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.dropout = dropout

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # These will be initialized in train()
        self.model: CellTypeClassifier | None = None
        self.optimizer: AdamW | None = None
        self.scheduler: CosineAnnealingWarmRestarts | None = None
        self.criterion: nn.Module | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        # Tracking
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

    def _setup_data(self) -> None:
        """Set up data loaders."""
        logger.info("Setting up data loaders...")

        train_ds, val_ds, test_ds = create_data_splits(
            self.data_path,
            seed=self.seed,
        )

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_ds,
            val_ds,
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            use_weighted_sampler=True,
        )

        self.num_classes = train_ds.num_classes
        self.class_names = train_ds.class_names

        logger.info(f"Data setup complete: {self.num_classes} classes")

    def _setup_model(self) -> None:
        """Set up model, optimizer, scheduler, and loss."""
        logger.info("Setting up model...")

        # Model
        self.model = CellTypeClassifier(
            num_classes=self.num_classes,
            backbone_name=self.backbone_name,
            pretrained=self.pretrained,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        # Loss with class weights
        if self.use_class_weights:
            train_labels = self.train_loader.dataset.labels[
                self.train_loader.dataset.indices
            ]
            class_weights = get_class_weights(
                train_labels, self.num_classes, method="inverse"
            ).to(self.device)
            logger.info(f"Class weights: {class_weights}")
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.label_smoothing,
        )

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not self.use_wandb:
            return

        try:
            import wandb

            wandb.init(
                project=self.project_name,
                config={
                    "backbone": self.backbone_name,
                    "pretrained": self.pretrained,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "label_smoothing": self.label_smoothing,
                    "num_classes": self.num_classes,
                    "class_names": self.class_names,
                },
            )
            logger.info("W&B initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        return {
            "train_loss": total_loss / total,
            "train_acc": correct / total,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader, prefix: str = "val") -> dict[str, float]:
        """Run validation.

        Args:
            loader: DataLoader to evaluate on
            prefix: Metric prefix ('val' or 'test')

        Returns:
            Dictionary of validation metrics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(loader, desc=f"Evaluating ({prefix})", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        macro_precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        macro_recall = recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        )

        return {
            f"{prefix}_loss": total_loss / len(all_labels),
            f"{prefix}_acc": accuracy,
            f"{prefix}_macro_f1": macro_f1,
            f"{prefix}_weighted_f1": weighted_f1,
            f"{prefix}_precision": macro_precision,
            f"{prefix}_recall": macro_recall,
        }

    def train(self) -> dict[str, Any]:
        """Run full training loop.

        Returns:
            Final metrics dictionary
        """
        # Setup
        self._setup_data()
        self._setup_model()
        self._setup_wandb()

        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate(self.val_loader, prefix="val")

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            logger.info(
                f"  train_loss: {metrics['train_loss']:.4f}, "
                f"train_acc: {metrics['train_acc']:.4f}"
            )
            logger.info(
                f"  val_loss: {metrics['val_loss']:.4f}, "
                f"val_acc: {metrics['val_acc']:.4f}, "
                f"val_macro_f1: {metrics['val_macro_f1']:.4f}"
            )

            # W&B logging
            if self.use_wandb:
                try:
                    import wandb

                    wandb.log(metrics)
                except Exception:
                    pass

            # Check for improvement
            if val_metrics["val_macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_macro_f1"]
                self.epochs_without_improvement = 0

                # Save best checkpoint
                self.model.save_checkpoint(
                    str(self.output_path / "best_model.pt"),
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics=metrics,
                )
                logger.info(f"  New best model saved (F1: {self.best_val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping after {self.early_stopping_patience} epochs "
                    f"without improvement"
                )
                break

        # Final evaluation on test set
        logger.info("\nFinal evaluation on test set:")
        test_metrics = self.validate(self.test_loader, prefix="test")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Save final checkpoint
        self.model.save_checkpoint(
            str(self.output_path / "final_model.pt"),
            optimizer=self.optimizer,
            epoch=self.epochs,
            metrics=test_metrics,
        )

        # Close W&B
        if self.use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        return test_metrics
