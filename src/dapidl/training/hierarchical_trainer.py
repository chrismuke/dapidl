"""Hierarchical Trainer with Curriculum Learning.

Training loop for hierarchical multi-head classification with:
- Curriculum learning: progressively activate heads
- Multi-level loss weighting
- Per-level metrics tracking
- Confidence-based fallback inference
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

from dapidl.data.hierarchical_dataset import (
    HierarchicalDataset,
    HierarchicalLabels,
    create_hierarchical_data_splits,
    create_hierarchical_dataloaders,
)
from dapidl.models.hierarchical import (
    HierarchicalClassifier,
    HierarchyConfig,
    HierarchicalOutput,
)
from dapidl.training.hierarchical_loss import (
    HierarchicalLoss,
    CurriculumScheduler,
)


class HierarchicalTrainer:
    """Training orchestrator for hierarchical multi-head classification.

    Implements curriculum learning where classification heads are
    progressively activated during training:
    - Phase 1 (epochs 1-20): Coarse only
    - Phase 2 (epochs 21-50): Coarse + Medium
    - Phase 3 (epochs 51+): All heads

    Attributes:
        model: HierarchicalClassifier model
        criterion: HierarchicalLoss function
        curriculum: CurriculumScheduler for head activation
    """

    def __init__(
        self,
        data_path: str | Path,
        output_path: str | Path = "outputs",
        # Model params
        backbone_name: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        dropout: float = 0.3,
        use_shared_projection: bool = True,
        projection_dim: int = 512,
        # Training params
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Curriculum params
        coarse_only_epochs: int = 20,
        coarse_medium_epochs: int = 50,
        warmup_epochs: int = 5,
        # Loss params
        coarse_weight: float = 1.0,
        medium_weight: float = 0.5,
        fine_weight: float = 0.3,
        consistency_weight: float = 0.1,
        label_smoothing: float = 0.1,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        # Class balancing
        max_weight_ratio: float = 10.0,
        min_samples_per_class: int | None = 20,
        # Other params
        use_wandb: bool = True,
        project_name: str = "dapidl-hierarchical",
        seed: int = 42,
        num_workers: int = 8,
        early_stopping_patience: int = 15,
        use_amp: bool = True,
    ) -> None:
        """Initialize hierarchical trainer.

        Args:
            data_path: Path to prepared dataset
            output_path: Path for outputs
            backbone_name: Name of backbone model
            pretrained: Use pretrained weights
            dropout: Dropout probability
            use_shared_projection: Add shared projection before heads
            projection_dim: Dimension of shared projection
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            coarse_only_epochs: Epochs to train coarse only
            coarse_medium_epochs: Epochs to train coarse+medium
            warmup_epochs: Warmup when activating new heads
            coarse_weight: Loss weight for coarse level
            medium_weight: Loss weight for medium level
            fine_weight: Loss weight for fine level
            consistency_weight: Weight for consistency penalty
            label_smoothing: Label smoothing factor
            use_focal: Use focal loss
            focal_gamma: Focal loss gamma
            max_weight_ratio: Max ratio between class weights
            min_samples_per_class: Min samples per class
            use_wandb: Enable W&B logging
            project_name: W&B project name
            seed: Random seed
            num_workers: DataLoader workers
            early_stopping_patience: Epochs without improvement before stopping
            use_amp: Use automatic mixed precision
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Store all params
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.seed = seed
        self.num_workers = num_workers
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.dropout = dropout
        self.use_shared_projection = use_shared_projection
        self.projection_dim = projection_dim

        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.max_weight_ratio = max_weight_ratio
        self.min_samples_per_class = min_samples_per_class

        # Loss weights
        self.base_loss_weights = (coarse_weight, medium_weight, fine_weight, consistency_weight)

        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            coarse_only_epochs=coarse_only_epochs,
            coarse_medium_epochs=coarse_medium_epochs,
            warmup_epochs=warmup_epochs,
        )

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

        # Will be initialized in train()
        self.model: HierarchicalClassifier | None = None
        self.optimizer: AdamW | None = None
        self.scheduler: CosineAnnealingWarmRestarts | None = None
        self.criterion: HierarchicalLoss | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.hierarchy_config: HierarchyConfig | None = None
        self.hierarchical_labels: HierarchicalLabels | None = None

        # Tracking
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        # AMP
        self.scaler = GradScaler("cuda") if self.use_amp and self.device.type == "cuda" else None

    def _setup_data(self) -> None:
        """Set up data loaders with hierarchical labels."""
        logger.info("Setting up hierarchical data loaders...")

        train_ds, val_ds, test_ds, hierarchical_labels = create_hierarchical_data_splits(
            self.data_path,
            seed=self.seed,
            min_samples_per_class=self.min_samples_per_class,
            stratify_level="coarse",  # Stratify by coarse for balanced splits
        )

        self.hierarchical_labels = hierarchical_labels
        self.hierarchy_config = hierarchical_labels.hierarchy_config

        self.train_loader, self.val_loader, self.test_loader = create_hierarchical_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            use_weighted_sampler=True,
            sample_weight_level="coarse",  # Balance by coarse for stability
        )

        logger.info(
            f"Data setup: coarse={self.hierarchy_config.num_coarse}, "
            f"medium={self.hierarchy_config.num_medium}, "
            f"fine={self.hierarchy_config.num_fine}"
        )

    def _setup_model(self) -> None:
        """Set up model, optimizer, scheduler, and loss."""
        logger.info("Setting up hierarchical model...")

        # Model
        self.model = HierarchicalClassifier(
            hierarchy_config=self.hierarchy_config,
            backbone_name=self.backbone_name,
            pretrained=self.pretrained,
            dropout=self.dropout,
            use_shared_projection=self.use_shared_projection,
            projection_dim=self.projection_dim,
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

        # Get class weights at each level
        class_weights = self.train_loader.dataset.get_class_weights(
            max_weight_ratio=self.max_weight_ratio
        )

        # Loss
        self.criterion = HierarchicalLoss(
            hierarchy_config=self.hierarchy_config,
            coarse_weight=self.base_loss_weights[0],
            medium_weight=self.base_loss_weights[1],
            fine_weight=self.base_loss_weights[2],
            consistency_weight=self.base_loss_weights[3],
            coarse_class_weights=class_weights["coarse_weights"].to(self.device),
            medium_class_weights=class_weights["medium_weights"].to(self.device),
            fine_class_weights=class_weights["fine_weights"].to(self.device),
            label_smoothing=self.label_smoothing,
            use_focal=self.use_focal,
            focal_gamma=self.focal_gamma,
        )

    def _setup_wandb(self) -> None:
        """Initialize W&B logging."""
        if not self.use_wandb:
            return

        try:
            import wandb

            config = {
                # Model
                "backbone": self.backbone_name,
                "pretrained": self.pretrained,
                "dropout": self.dropout,
                "use_shared_projection": self.use_shared_projection,
                "projection_dim": self.projection_dim,
                # Training
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                # Hierarchy
                "num_coarse": self.hierarchy_config.num_coarse,
                "num_medium": self.hierarchy_config.num_medium,
                "num_fine": self.hierarchy_config.num_fine,
                "coarse_names": self.hierarchy_config.coarse_names,
                "medium_names": self.hierarchy_config.medium_names,
                # Curriculum
                "coarse_only_epochs": self.curriculum.coarse_only_epochs,
                "coarse_medium_epochs": self.curriculum.coarse_medium_epochs,
                # Loss
                "base_loss_weights": self.base_loss_weights,
                "label_smoothing": self.label_smoothing,
                "use_focal": self.use_focal,
                # Other
                "seed": self.seed,
            }

            wandb.init(
                project=self.project_name,
                config=config,
            )

            logger.info("W&B initialized for hierarchical training")

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch with curriculum-based head activation.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Get active heads and loss weights for this epoch
        active_heads = self.curriculum.get_active_heads(epoch)
        loss_weights = self.curriculum.get_loss_weights(epoch, self.base_loss_weights)

        # Update model's active heads
        self.model.set_active_heads(active_heads)

        # Update loss weights
        self.criterion.coarse_weight = loss_weights["coarse_weight"]
        self.criterion.medium_weight = loss_weights["medium_weight"]
        self.criterion.fine_weight = loss_weights["fine_weight"]
        self.criterion.consistency_weight = loss_weights["consistency_weight"]

        total_loss = 0.0
        coarse_correct = 0
        medium_correct = 0
        fine_correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Training ({self.curriculum.get_phase_name(epoch)})", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            coarse_labels = labels["coarse"].to(self.device, non_blocking=True)
            medium_labels = labels["medium"].to(self.device, non_blocking=True) if "medium" in active_heads else None
            fine_labels = labels["fine"].to(self.device, non_blocking=True) if "fine" in active_heads else None

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast("cuda"):
                    output = self.model(images)
                    loss, loss_dict = self.criterion(
                        output, coarse_labels, medium_labels, fine_labels
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                loss, loss_dict = self.criterion(
                    output, coarse_labels, medium_labels, fine_labels
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            # Accuracy at each level
            coarse_correct += (output.coarse_logits.argmax(1) == coarse_labels).sum().item()
            if output.medium_logits is not None and medium_labels is not None:
                medium_correct += (output.medium_logits.argmax(1) == medium_labels).sum().item()
            if output.fine_logits is not None and fine_labels is not None:
                fine_correct += (output.fine_logits.argmax(1) == fine_labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "coarse_acc": coarse_correct / total})

        metrics = {
            "train_loss": total_loss / total,
            "train_coarse_acc": coarse_correct / total,
        }
        if "medium" in active_heads:
            metrics["train_medium_acc"] = medium_correct / total
        if "fine" in active_heads:
            metrics["train_fine_acc"] = fine_correct / total

        return metrics

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
        prefix: str = "val",
        return_predictions: bool = False,
    ) -> dict[str, Any]:
        """Run validation at all active hierarchy levels.

        Args:
            loader: DataLoader to validate on
            prefix: Metric prefix ('val' or 'test')
            return_predictions: Include predictions in return

        Returns:
            Dictionary of validation metrics
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Set model to inference mode
        self.model.training = False

        total_loss = 0.0

        all_coarse_preds = []
        all_coarse_labels = []
        all_medium_preds = []
        all_medium_labels = []
        all_fine_preds = []
        all_fine_labels = []

        for images, labels in tqdm(loader, desc=f"Validating ({prefix})", leave=False):
            images = images.to(self.device)
            coarse_labels = labels["coarse"].to(self.device)
            medium_labels = labels["medium"].to(self.device)
            fine_labels = labels["fine"].to(self.device)

            output = self.model(images)

            # Loss (with all levels for fair comparison)
            loss, _ = self.criterion(output, coarse_labels, medium_labels, fine_labels)
            total_loss += loss.item() * images.size(0)

            # Collect predictions
            all_coarse_preds.extend(output.coarse_logits.argmax(1).cpu().numpy())
            all_coarse_labels.extend(coarse_labels.cpu().numpy())

            if output.medium_logits is not None:
                all_medium_preds.extend(output.medium_logits.argmax(1).cpu().numpy())
                all_medium_labels.extend(medium_labels.cpu().numpy())

            if output.fine_logits is not None:
                all_fine_preds.extend(output.fine_logits.argmax(1).cpu().numpy())
                all_fine_labels.extend(fine_labels.cpu().numpy())

        # Convert to arrays
        coarse_preds = np.array(all_coarse_preds)
        coarse_labels_arr = np.array(all_coarse_labels)
        medium_preds = np.array(all_medium_preds) if all_medium_preds else None
        medium_labels_arr = np.array(all_medium_labels) if all_medium_labels else None
        fine_preds = np.array(all_fine_preds) if all_fine_preds else None
        fine_labels_arr = np.array(all_fine_labels) if all_fine_labels else None

        # Compute metrics
        result = {
            f"{prefix}_loss": total_loss / len(coarse_labels_arr),
            f"{prefix}_coarse_acc": (coarse_preds == coarse_labels_arr).mean(),
            f"{prefix}_coarse_f1": f1_score(coarse_labels_arr, coarse_preds, average="macro", zero_division=0),
        }

        if medium_preds is not None:
            result[f"{prefix}_medium_acc"] = (medium_preds == medium_labels_arr).mean()
            result[f"{prefix}_medium_f1"] = f1_score(medium_labels_arr, medium_preds, average="macro", zero_division=0)

        if fine_preds is not None:
            result[f"{prefix}_fine_acc"] = (fine_preds == fine_labels_arr).mean()
            result[f"{prefix}_fine_f1"] = f1_score(fine_labels_arr, fine_preds, average="macro", zero_division=0)

        if return_predictions:
            result["_coarse_preds"] = coarse_preds
            result["_coarse_labels"] = coarse_labels_arr
            result["_medium_preds"] = medium_preds
            result["_medium_labels"] = medium_labels_arr
            result["_fine_preds"] = fine_preds
            result["_fine_labels"] = fine_labels_arr

        return result

    def _log_confusion_matrices(
        self,
        coarse_preds: np.ndarray,
        coarse_labels: np.ndarray,
        medium_preds: np.ndarray | None,
        medium_labels: np.ndarray | None,
        epoch: int,
        prefix: str = "val",
    ) -> None:
        """Log confusion matrices at each level to W&B."""
        if not self.use_wandb:
            return

        try:
            import wandb
            from sklearn.metrics import confusion_matrix

            # Coarse confusion matrix
            cm_coarse = confusion_matrix(
                coarse_labels, coarse_preds,
                labels=range(self.hierarchy_config.num_coarse)
            )
            cm_coarse_norm = cm_coarse.astype("float") / (cm_coarse.sum(axis=1, keepdims=True) + 1e-8)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_coarse_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=self.hierarchy_config.coarse_names,
                yticklabels=self.hierarchy_config.coarse_names,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Coarse Confusion Matrix - Epoch {epoch}")
            plt.tight_layout()
            wandb.log({f"{prefix}_coarse_confusion": wandb.Image(fig)})
            plt.close(fig)

            # Medium confusion matrix (if available)
            if medium_preds is not None and medium_labels is not None:
                cm_medium = confusion_matrix(
                    medium_labels, medium_preds,
                    labels=range(self.hierarchy_config.num_medium)
                )
                cm_medium_norm = cm_medium.astype("float") / (cm_medium.sum(axis=1, keepdims=True) + 1e-8)

                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    cm_medium_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=self.hierarchy_config.medium_names,
                    yticklabels=self.hierarchy_config.medium_names,
                    ax=ax,
                    annot_kws={"size": 8},
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(f"Medium Confusion Matrix - Epoch {epoch}")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)
                plt.tight_layout()
                wandb.log({f"{prefix}_medium_confusion": wandb.Image(fig)})
                plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to log confusion matrices: {e}")

    def train(self) -> dict[str, Any]:
        """Run full training loop with curriculum learning.

        Returns:
            Final test metrics dictionary
        """
        # Setup
        self._setup_data()
        self._setup_model()
        self._setup_wandb()

        logger.info(f"Starting hierarchical training for {self.epochs} epochs")
        logger.info(f"Curriculum: coarse_only={self.curriculum.coarse_only_epochs}, "
                    f"coarse_medium={self.curriculum.coarse_medium_epochs}")

        detailed_log_interval = max(1, min(5, self.epochs // 10))

        for epoch in range(1, self.epochs + 1):
            phase_name = self.curriculum.get_phase_name(epoch)
            logger.info(f"\nEpoch {epoch}/{self.epochs} - {phase_name}")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            should_log_detailed = (
                epoch % detailed_log_interval == 0
                or epoch == 1
                or epoch == self.epochs
            )
            val_metrics = self.validate(
                self.val_loader, prefix="val", return_predictions=should_log_detailed
            )

            # Extract predictions for detailed logging
            coarse_preds = val_metrics.pop("_coarse_preds", None)
            coarse_labels = val_metrics.pop("_coarse_labels", None)
            medium_preds = val_metrics.pop("_medium_preds", None)
            medium_labels = val_metrics.pop("_medium_labels", None)
            fine_preds = val_metrics.pop("_fine_preds", None)
            fine_labels = val_metrics.pop("_fine_labels", None)

            # Update scheduler
            self.scheduler.step()

            # Combine metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "phase": phase_name,
            }

            # Log
            logger.info(
                f"  train_loss: {metrics['train_loss']:.4f}, "
                f"coarse_acc: {metrics['train_coarse_acc']:.4f}"
            )
            logger.info(
                f"  val_loss: {metrics['val_loss']:.4f}, "
                f"coarse_f1: {metrics['val_coarse_f1']:.4f}"
            )
            if "val_medium_f1" in metrics:
                logger.info(f"  medium_f1: {metrics['val_medium_f1']:.4f}")
            if "val_fine_f1" in metrics:
                logger.info(f"  fine_f1: {metrics['val_fine_f1']:.4f}")

            # W&B logging
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log(metrics)
                except Exception:
                    pass

            # Detailed logging
            if should_log_detailed and coarse_preds is not None:
                self._log_confusion_matrices(
                    coarse_preds, coarse_labels,
                    medium_preds, medium_labels,
                    epoch, "val"
                )

            # Check for improvement (use coarse F1 as primary metric - most stable)
            primary_metric = metrics["val_coarse_f1"]
            if primary_metric > self.best_val_f1:
                self.best_val_f1 = primary_metric
                self.epochs_without_improvement = 0

                # Save best checkpoint
                best_path = str(self.output_path / "best_model.pt")
                self.model.save_checkpoint(
                    best_path,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=metrics,
                )
                logger.info(f"  New best model saved (coarse F1: {self.best_val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping after {self.early_stopping_patience} epochs "
                    f"without improvement"
                )
                break

        # Final test set validation
        logger.info("\nFinal test set validation:")

        # Enable all heads for final validation
        self.model.set_active_heads({"coarse", "medium", "fine"})

        test_metrics = self.validate(self.test_loader, prefix="test", return_predictions=True)

        # Extract predictions
        coarse_preds = test_metrics.pop("_coarse_preds", None)
        coarse_labels = test_metrics.pop("_coarse_labels", None)
        medium_preds = test_metrics.pop("_medium_preds", None)
        medium_labels = test_metrics.pop("_medium_labels", None)
        fine_preds = test_metrics.pop("_fine_preds", None)
        fine_labels = test_metrics.pop("_fine_labels", None)

        for k, v in test_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

        # Log final confusion matrices
        if coarse_preds is not None:
            self._log_confusion_matrices(
                coarse_preds, coarse_labels,
                medium_preds, medium_labels,
                self.epochs, "test"
            )

        # Save final checkpoint
        final_path = str(self.output_path / "final_model.pt")
        self.model.save_checkpoint(
            final_path,
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
