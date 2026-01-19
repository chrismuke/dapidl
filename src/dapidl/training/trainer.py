"""Training loop for DAPIDL."""

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

from dapidl.data.dataset import DAPIDLDataset, DAPIDLDatasetWithHeavyAug, create_data_splits, create_dataloaders
from dapidl.models.classifier import CellTypeClassifier
from dapidl.training.losses import get_class_weights

# Import types for type hints (avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dapidl.data.multi_tissue_dataset import MultiTissueConfig


class Trainer:
    """Training orchestrator for DAPIDL.

    Handles training loop, validation, checkpointing, and W&B logging.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
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
        # Class balancing params
        max_weight_ratio: float = 10.0,
        min_samples_per_class: int | None = None,
        # Performance params
        use_amp: bool = True,
        # Data loading backend
        backend: str = "pytorch",
        # Augmentation options
        use_heavy_aug: bool = False,
        # Multi-tissue training
        multi_tissue_config: "MultiTissueConfig | None" = None,
    ) -> None:
        """Initialize trainer.

        Args:
            data_path: Path to prepared dataset (None if using multi_tissue_config)
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
            max_weight_ratio: Max ratio between class weights (prevents mode collapse)
            min_samples_per_class: Filter classes with fewer samples (None to keep all)
            use_amp: Use automatic mixed precision (fp16) for faster training
            backend: Data loading backend ("pytorch" or "dali")
            use_heavy_aug: Use heavy augmentation for rare classes
            multi_tissue_config: Configuration for multi-tissue training (alternative to data_path)
        """
        self.data_path = Path(data_path) if data_path else None
        self.multi_tissue_config = multi_tissue_config

        # Validate data source
        if self.data_path is None and self.multi_tissue_config is None:
            raise ValueError("Either data_path or multi_tissue_config must be provided")
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
        self.max_weight_ratio = max_weight_ratio
        self.min_samples_per_class = min_samples_per_class
        self.use_amp = use_amp
        self.backend = backend
        self.use_heavy_aug = use_heavy_aug

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

        # Mixed precision training (AMP)
        self.scaler = GradScaler("cuda") if self.use_amp and self.device.type == "cuda" else None
        if self.use_amp and self.device.type == "cuda":
            logger.info("Mixed precision training (AMP) enabled")

    def _setup_data(self) -> None:
        """Set up data loaders."""
        # Handle multi-tissue config (runtime combination of multiple LMDB datasets)
        if self.multi_tissue_config is not None:
            self._setup_multi_tissue_data()
            return

        logger.info(f"Setting up data loaders (backend={self.backend})...")

        # Type narrowing - if multi_tissue_config is None, data_path must be set (validated in __init__)
        assert self.data_path is not None, "data_path required for single-dataset training"

        # Auto-filter classes with too few samples for stratified splitting
        # With 70/15/15 split, need at least ceil(2/0.15) ≈ 14 samples per class
        # Use 20 as default, but allow override via min_samples_per_class
        min_samples = self.min_samples_per_class if self.min_samples_per_class is not None else 20

        if self.backend in ("dali", "dali-lmdb"):
            # Use DALI backend for GPU-accelerated data loading
            from dapidl.data.dataset import create_dataloaders_with_backend

            self.train_loader, self.val_loader, self.test_loader, metadata = create_dataloaders_with_backend(
                data_path=self.data_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                backend=self.backend,
                use_weighted_sampler=True,  # Note: DALI doesn't support weighted sampling
                seed=self.seed,
                min_samples_per_class=min_samples,
            )
            self.num_classes = metadata["num_classes"]
            self.class_names = metadata["class_names"]
            self._class_weights = metadata["class_weights"]  # Store for loss function
        else:
            # Use PyTorch backend (default)
            train_ds, val_ds, test_ds = create_data_splits(
                self.data_path,
                seed=self.seed,
                min_samples_per_class=min_samples,
                use_heavy_aug=self.use_heavy_aug,
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
            self._class_weights = None  # Computed in _setup_model for PyTorch

        logger.info(f"Data setup complete: {self.num_classes} classes")

    def _setup_multi_tissue_data(self) -> None:
        """Set up data loaders for multi-tissue training."""
        from dapidl.data.multi_tissue_dataset import (
            create_multi_tissue_splits,
            create_multi_tissue_dataloaders,
        )

        # Type narrowing - this is validated in __init__
        assert self.multi_tissue_config is not None

        logger.info("Setting up multi-tissue data loaders...")
        logger.info(f"  Datasets: {len(self.multi_tissue_config.datasets)}")
        for ds in self.multi_tissue_config.datasets:
            logger.info(f"    - {ds.tissue} ({ds.platform}): {ds.path}")

        # Create train/val/test splits
        train_ds, val_ds, test_ds = create_multi_tissue_splits(
            self.multi_tissue_config,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.seed,
            stratify_by="both",  # Stratify by tissue and label
        )

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_multi_tissue_dataloaders(
            train_ds,
            val_ds,
            test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.num_classes = train_ds.num_classes
        self.class_names = train_ds.class_names
        self._class_weights = None  # Will be computed in _setup_model

        logger.info(f"Multi-tissue data setup complete: {self.num_classes} classes")
        logger.info(f"  Train: {len(train_ds)} samples")
        logger.info(f"  Val: {len(val_ds)} samples")
        logger.info(f"  Test: {len(test_ds)} samples")

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
            if self._class_weights is not None:
                # Use pre-computed weights (DALI backend)
                class_weights = self._class_weights.to(self.device)
            else:
                # Compute from dataset (PyTorch backend)
                train_labels = self.train_loader.dataset.labels[
                    self.train_loader.dataset.indices
                ]
                class_weights = get_class_weights(
                    train_labels,
                    self.num_classes,
                    method="inverse",
                    max_weight_ratio=self.max_weight_ratio,
                ).to(self.device)
            logger.info(f"Class weights (max_ratio={self.max_weight_ratio}): {class_weights}")
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.label_smoothing,
        )

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging with full reproducibility info."""
        if not self.use_wandb:
            return

        try:
            import wandb
            import json

            from dapidl.tracking.reproducibility import (
                get_reproducibility_info,
                get_cli_command,
            )

            # Load dataset info for class distribution
            dataset_info = {}
            if self.data_path is not None:
                dataset_info_path = self.data_path / "dataset_info.json"
                if dataset_info_path.exists():
                    with open(dataset_info_path) as f:
                        dataset_info = json.load(f)

            # Capture reproducibility info
            if self.data_path is not None:
                repro_info = get_reproducibility_info(
                    dataset_path=self.data_path,
                    cli_command=get_cli_command(),
                )
            else:
                # Multi-tissue mode - capture basic info
                repro_info = {
                    "mode": "multi_tissue",
                    "datasets": [str(ds.path) for ds in self.multi_tissue_config.datasets] if self.multi_tissue_config else [],
                    "cli_command": get_cli_command(),
                }

            # Build config with training params + reproducibility
            config = {
                # Training parameters
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
                "seed": self.seed,
                # Dataset info
                "n_samples": dataset_info.get("n_samples", 0),
                "fine_grained": dataset_info.get("fine_grained", False),
                "confidence_threshold": dataset_info.get("confidence_threshold", 0.5),
                "class_distribution": dataset_info.get("class_distribution", {}),
                # Reproducibility info (flat keys for easy filtering)
                **repro_info.to_flat_dict(),
            }

            wandb.init(
                project=self.project_name,
                config=config,
            )

            # Store reproducibility info for later access
            self._reproducibility_info = repro_info

            # Log full reproducibility as JSON artifact if git is dirty
            if repro_info.git.is_dirty and repro_info.git.diff:
                # Save diff to file and log as artifact
                diff_path = self.output_path / "git_diff.patch"
                diff_path.write_text(repro_info.git.diff)

                diff_artifact = wandb.Artifact(
                    name="code-diff",
                    type="code",
                    description="Uncommitted code changes (git diff)",
                )
                diff_artifact.add_file(str(diff_path))
                wandb.log_artifact(diff_artifact)
                logger.warning(
                    f"Git repo has uncommitted changes! Diff saved to {diff_path}"
                )

            # Log environment packages as artifact
            env_path = self.output_path / "environment.json"
            with open(env_path, "w") as f:
                json.dump(repro_info.environment.to_dict(), f, indent=2)

            env_artifact = wandb.Artifact(
                name="environment",
                type="environment",
                description="Python environment snapshot",
            )
            env_artifact.add_file(str(env_path))
            wandb.log_artifact(env_artifact)

            # Log reproduce command as notes
            wandb.run.notes = repro_info.get_reproduce_command()

            # Log class distribution as a bar chart
            if dataset_info.get("class_distribution"):
                class_dist = dataset_info["class_distribution"]
                data = [[name, count] for name, count in class_dist.items()]
                table = wandb.Table(data=data, columns=["cell_type", "count"])
                wandb.log({
                    "cell_type_distribution": wandb.plot.bar(
                        table, "cell_type", "count",
                        title="Cell Type Distribution"
                    )
                })

            logger.info(f"W&B initialized (git: {repro_info.git.commit[:8]})")
            if repro_info.git.is_dirty:
                logger.warning("  WARNING: Running with uncommitted changes!")

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def _log_dataset_artifact(self) -> None:
        """Log dataset with train/val/test splits as a W&B Artifact."""
        if not self.use_wandb:
            return

        try:
            import wandb

            # Create dataset artifact with metadata
            artifact = wandb.Artifact(
                name="dapidl-dataset",
                type="dataset",
                description="DAPIDL cell type classification dataset with train/val/test splits",
                metadata={
                    "num_classes": self.num_classes,
                    "class_names": self.class_names,
                    "train_size": len(self.train_loader.dataset),
                    "val_size": len(self.val_loader.dataset),
                    "test_size": len(self.test_loader.dataset),
                    "seed": self.seed,
                    "data_path": str(self.data_path),
                },
            )

            # Add the full dataset directory (patches.zarr, labels.npy, metadata.parquet, etc.)
            artifact.add_dir(str(self.data_path), name="data")

            # Save and add split indices for reproducibility
            split_dir = self.output_path / "splits"
            split_dir.mkdir(exist_ok=True)

            np.save(split_dir / "train_indices.npy", self.train_loader.dataset.indices)
            np.save(split_dir / "val_indices.npy", self.val_loader.dataset.indices)
            np.save(split_dir / "test_indices.npy", self.test_loader.dataset.indices)

            artifact.add_dir(str(split_dir), name="splits")

            # Log artifact and link to registry
            wandb.log_artifact(artifact)

            # Link to the Datasets registry if available
            try:
                wandb.run.link_artifact(artifact, target_path="wandb-registry-Datasets/dapidl")
            except Exception:
                pass  # Registry linking may not be available in all deployments

            logger.info("Dataset artifact logged to W&B")

        except Exception as e:
            logger.warning(f"Failed to log dataset artifact: {e}")

    def _log_model_artifact(
        self, checkpoint_path: str, artifact_name: str, metrics: dict, is_best: bool = False
    ) -> None:
        """Log model checkpoint as a W&B Artifact.

        Args:
            checkpoint_path: Path to the saved checkpoint file
            artifact_name: Name for the artifact (e.g., 'best-model', 'final-model')
            metrics: Training/validation metrics to include in metadata
            is_best: Whether this is the best model (for registry linking)
        """
        if not self.use_wandb:
            return

        try:
            import wandb

            # Convert numpy types to Python types for JSON serialization
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    clean_metrics[k] = float(v)
                elif isinstance(v, (int, float, str, bool)):
                    clean_metrics[k] = v

            artifact = wandb.Artifact(
                name=f"dapidl-{artifact_name}",
                type="model",
                description=f"DAPIDL cell type classifier - {artifact_name}",
                metadata={
                    "backbone": self.backbone_name,
                    "pretrained": self.pretrained,
                    "dropout": self.dropout,
                    "num_classes": self.num_classes,
                    "class_names": self.class_names,
                    **clean_metrics,
                },
            )

            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

            # Link best model to the Models registry
            if is_best:
                try:
                    wandb.run.link_artifact(artifact, target_path="wandb-registry-Models/dapidl")
                except Exception:
                    pass  # Registry linking may not be available

            logger.info(f"Model artifact '{artifact_name}' logged to W&B")

        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")

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
            images = images.to(self.device, non_blocking=True)
            # Handle dict labels from MultiTissueDataset
            if isinstance(labels, dict):
                labels = labels["label"]
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with optional AMP
            self.optimizer.zero_grad()

            if self.scaler is not None:
                # Mixed precision training
                with autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
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
    def validate(
        self, loader: DataLoader, prefix: str = "val", return_predictions: bool = False
    ) -> dict[str, Any]:
        """Run validation.

        Args:
            loader: DataLoader to evaluate on
            prefix: Metric prefix ('val' or 'test')
            return_predictions: If True, include predictions and labels in return dict

        Returns:
            Dictionary of validation metrics, optionally with predictions/labels
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(loader, desc=f"Evaluating ({prefix})", leave=False):
            images = images.to(self.device)
            # Handle dict labels from MultiTissueDataset
            if isinstance(labels, dict):
                labels = labels["label"]
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

        result = {
            f"{prefix}_loss": total_loss / len(all_labels),
            f"{prefix}_acc": accuracy,
            f"{prefix}_macro_f1": macro_f1,
            f"{prefix}_weighted_f1": weighted_f1,
            f"{prefix}_precision": macro_precision,
            f"{prefix}_recall": macro_recall,
        }

        if return_predictions:
            result["_predictions"] = all_preds
            result["_labels"] = all_labels

        return result

    def _log_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int, prefix: str = "val"
    ) -> None:
        """Log confusion matrix to W&B as a heatmap image.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            epoch: Current epoch number
            prefix: Metric prefix ('val' or 'test')
        """
        if not self.use_wandb:
            return

        try:
            import wandb
            from sklearn.metrics import confusion_matrix

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

            # Normalize by row (true labels) - shows recall per class
            cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

            # Create figure with two subplots: raw counts and normalized
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Raw counts
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[0],
            )
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True")
            axes[0].set_title(f"Confusion Matrix (Counts) - Epoch {epoch}")
            axes[0].tick_params(axis="x", rotation=45)
            axes[0].tick_params(axis="y", rotation=0)

            # Normalized (percentage)
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[1],
                vmin=0,
                vmax=1,
            )
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")
            axes[1].set_title(f"Confusion Matrix (Normalized) - Epoch {epoch}")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].tick_params(axis="y", rotation=0)

            plt.tight_layout()

            # Log to W&B
            wandb.log({f"{prefix}_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

            logger.debug(f"Confusion matrix logged for epoch {epoch}")

        except Exception as e:
            logger.warning(f"Failed to log confusion matrix: {e}")

    def _log_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int, prefix: str = "val"
    ) -> None:
        """Log per-class F1, precision, recall to W&B.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            epoch: Current epoch number
            prefix: Metric prefix ('val' or 'test')
        """
        if not self.use_wandb:
            return

        try:
            import wandb
            from sklearn.metrics import (
                f1_score,
                precision_score,
                recall_score,
                classification_report,
            )

            # Per-class metrics
            f1_per_class = f1_score(
                y_true, y_pred, average=None, zero_division=0, labels=range(self.num_classes)
            )
            precision_per_class = precision_score(
                y_true, y_pred, average=None, zero_division=0, labels=range(self.num_classes)
            )
            recall_per_class = recall_score(
                y_true, y_pred, average=None, zero_division=0, labels=range(self.num_classes)
            )

            # Count samples per class in true labels
            true_counts = np.bincount(y_true, minlength=self.num_classes)
            pred_counts = np.bincount(y_pred, minlength=self.num_classes)

            # Log as individual metrics for each class
            per_class_metrics = {}
            for i, name in enumerate(self.class_names):
                safe_name = name.replace(" ", "_").replace("+", "plus")
                per_class_metrics[f"{prefix}_f1/{safe_name}"] = f1_per_class[i]
                per_class_metrics[f"{prefix}_precision/{safe_name}"] = precision_per_class[i]
                per_class_metrics[f"{prefix}_recall/{safe_name}"] = recall_per_class[i]

            wandb.log(per_class_metrics)

            # Create a table with all per-class metrics
            table_data = []
            for i, name in enumerate(self.class_names):
                table_data.append([
                    name,
                    true_counts[i],
                    pred_counts[i],
                    f1_per_class[i],
                    precision_per_class[i],
                    recall_per_class[i],
                ])

            # Sort by F1 score (ascending) to highlight worst classes
            table_data_sorted = sorted(table_data, key=lambda x: x[3])

            table = wandb.Table(
                columns=["Class", "True Count", "Pred Count", "F1", "Precision", "Recall"],
                data=table_data_sorted,
            )
            wandb.log({f"{prefix}_per_class_metrics": table})

            # Create bar chart showing F1 scores by class
            fig, ax = plt.subplots(figsize=(12, 6))
            sorted_names = [row[0] for row in table_data_sorted]
            sorted_f1 = [row[3] for row in table_data_sorted]

            colors = ["red" if f1 < 0.2 else "orange" if f1 < 0.4 else "green" for f1 in sorted_f1]
            bars = ax.barh(sorted_names, sorted_f1, color=colors)
            ax.set_xlabel("F1 Score")
            ax.set_title(f"Per-Class F1 Scores - Epoch {epoch}")
            ax.set_xlim(0, 1)

            # Add value labels
            for bar, f1 in zip(bars, sorted_f1):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{f1:.3f}", va="center", fontsize=8)

            plt.tight_layout()
            wandb.log({f"{prefix}_f1_by_class": wandb.Image(fig)})
            plt.close(fig)

            # Log worst classes summary
            worst_classes = table_data_sorted[:5]  # Bottom 5
            logger.info(f"  Worst 5 classes by F1:")
            for name, true_n, pred_n, f1, prec, rec in worst_classes:
                logger.info(f"    {name}: F1={f1:.3f} (P={prec:.3f}, R={rec:.3f}, n={true_n})")

        except Exception as e:
            logger.warning(f"Failed to log per-class metrics: {e}")

    def _log_misclassification_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int, prefix: str = "val"
    ) -> None:
        """Log analysis of most common misclassifications.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            epoch: Current epoch number
            prefix: Metric prefix ('val' or 'test')
        """
        if not self.use_wandb:
            return

        try:
            import wandb
            from collections import Counter

            # Find misclassifications
            misclassified_mask = y_true != y_pred
            misclassified_pairs = list(zip(y_true[misclassified_mask], y_pred[misclassified_mask]))

            if not misclassified_pairs:
                return

            # Count most common confusion pairs
            confusion_counts = Counter(misclassified_pairs)
            top_confusions = confusion_counts.most_common(20)

            # Create table
            table_data = []
            for (true_idx, pred_idx), count in top_confusions:
                true_name = self.class_names[true_idx]
                pred_name = self.class_names[pred_idx]
                # Calculate what percentage of true class this represents
                true_total = (y_true == true_idx).sum()
                pct_of_class = (count / true_total * 100) if true_total > 0 else 0
                table_data.append([true_name, pred_name, count, f"{pct_of_class:.1f}%"])

            table = wandb.Table(
                columns=["True Class", "Predicted As", "Count", "% of True Class"],
                data=table_data,
            )
            wandb.log({f"{prefix}_top_misclassifications": table})

            logger.info(f"  Top 5 misclassifications:")
            for i, ((true_idx, pred_idx), count) in enumerate(top_confusions[:5]):
                logger.info(
                    f"    {self.class_names[true_idx]} → {self.class_names[pred_idx]}: {count}"
                )

        except Exception as e:
            logger.warning(f"Failed to log misclassification analysis: {e}")

    def _log_sample_patches(self, samples_per_class: int = 5) -> None:
        """Log sample patches from the validation set to W&B.

        Samples N patches per class and logs them as PNG images organized by class.

        Args:
            samples_per_class: Number of sample patches to log per class (default: 5)
        """
        if not self.use_wandb:
            return

        try:
            import wandb
            from PIL import Image
            import io

            logger.info(f"Logging {samples_per_class} sample patches per class to W&B...")

            # Load labels from dataset
            labels_path = self.data_path / "labels.npy"
            if not labels_path.exists():
                logger.warning("labels.npy not found, skipping sample patch logging")
                return

            all_labels = np.load(labels_path)

            # Get validation indices
            if hasattr(self.val_loader.dataset, 'indices'):
                # PyTorch Subset
                val_indices = self.val_loader.dataset.indices
            else:
                # DALI or other - use a fraction of indices
                n_total = len(all_labels)
                np.random.seed(self.seed)
                val_indices = np.random.choice(n_total, size=min(n_total // 5, 10000), replace=False)

            val_labels = all_labels[val_indices]

            # Try to load patches from LMDB first, then Zarr
            lmdb_path = self.data_path / "patches.lmdb"
            zarr_path = self.data_path / "patches.zarr"

            patches_by_class = {i: [] for i in range(self.num_classes)}

            if lmdb_path.exists():
                # Load from LMDB
                import lmdb
                import struct

                env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
                with env.begin() as txn:
                    for idx_in_val, global_idx in enumerate(val_indices):
                        label = val_labels[idx_in_val]
                        if len(patches_by_class[label]) >= samples_per_class:
                            continue

                        data = txn.get(str(global_idx).encode())
                        if data:
                            # LMDB stores: 4-byte height + 4-byte width + raw data
                            h = struct.unpack('I', data[:4])[0]
                            w = struct.unpack('I', data[4:8])[0]
                            patch = np.frombuffer(data[8:], dtype=np.uint16).reshape(h, w)
                            patches_by_class[label].append((global_idx, patch))

                        # Check if we have enough samples for all classes
                        if all(len(patches_by_class[i]) >= samples_per_class for i in range(self.num_classes)):
                            break
                env.close()

            elif zarr_path.exists():
                # Load from Zarr
                import zarr

                patches = zarr.open(str(zarr_path), mode='r')
                for idx_in_val, global_idx in enumerate(val_indices):
                    label = val_labels[idx_in_val]
                    if len(patches_by_class[label]) >= samples_per_class:
                        continue

                    patch = patches[global_idx]
                    patches_by_class[label].append((global_idx, patch))

                    # Check if we have enough samples for all classes
                    if all(len(patches_by_class[i]) >= samples_per_class for i in range(self.num_classes)):
                        break
            else:
                logger.warning("No patches.lmdb or patches.zarr found, skipping sample patch logging")
                return

            # Convert patches to wandb.Image objects and log
            wandb_images = []
            for class_idx in range(self.num_classes):
                class_name = self.class_names[class_idx]
                patches = patches_by_class[class_idx]

                for sample_idx, (global_idx, patch) in enumerate(patches[:samples_per_class]):
                    # Normalize uint16 to uint8 for visualization
                    patch_normalized = patch.astype(np.float32)
                    patch_normalized = (patch_normalized - patch_normalized.min()) / (patch_normalized.max() - patch_normalized.min() + 1e-8)
                    patch_uint8 = (patch_normalized * 255).astype(np.uint8)

                    # Create PIL Image
                    img = Image.fromarray(patch_uint8, mode='L')

                    # Add to list with caption
                    wandb_images.append(wandb.Image(
                        img,
                        caption=f"{class_name} (idx={global_idx})"
                    ))

            # Log as a media panel grouped by class
            if wandb_images:
                wandb.log({"sample_patches_by_class": wandb_images})

                # Also create a table for better organization
                table_data = []
                for class_idx in range(self.num_classes):
                    class_name = self.class_names[class_idx]
                    patches = patches_by_class[class_idx]

                    for sample_idx, (global_idx, patch) in enumerate(patches[:samples_per_class]):
                        # Normalize for visualization
                        patch_normalized = patch.astype(np.float32)
                        patch_normalized = (patch_normalized - patch_normalized.min()) / (patch_normalized.max() - patch_normalized.min() + 1e-8)
                        patch_uint8 = (patch_normalized * 255).astype(np.uint8)

                        img = Image.fromarray(patch_uint8, mode='L')
                        table_data.append([
                            class_name,
                            sample_idx + 1,
                            global_idx,
                            wandb.Image(img)
                        ])

                table = wandb.Table(
                    columns=["Cell Type", "Sample #", "Index", "Patch"],
                    data=table_data
                )
                wandb.log({"sample_patches_table": table})

            logger.info(f"  Logged {len(wandb_images)} sample patches to W&B")

        except Exception as e:
            logger.warning(f"Failed to log sample patches: {e}")

    def train(self) -> dict[str, Any]:
        """Run full training loop.

        Returns:
            Final metrics dictionary
        """
        # Setup
        self._setup_data()
        self._setup_model()
        self._setup_wandb()

        # Log dataset artifact (once at start of training)
        self._log_dataset_artifact()

        # Log sample patches per class to W&B
        self._log_sample_patches(samples_per_class=5)

        logger.info(f"Starting training for {self.epochs} epochs")

        # Determine how often to log detailed metrics (every 5 epochs, or more for short runs)
        detailed_log_interval = max(1, min(5, self.epochs // 10))

        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate - get predictions for detailed analysis
            should_log_detailed = (
                (epoch + 1) % detailed_log_interval == 0
                or epoch == 0  # First epoch
                or epoch == self.epochs - 1  # Last epoch
            )
            val_metrics = self.validate(
                self.val_loader, prefix="val", return_predictions=should_log_detailed
            )

            # Extract predictions if returned
            val_preds = val_metrics.pop("_predictions", None)
            val_labels = val_metrics.pop("_labels", None)

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

            # Log detailed metrics (confusion matrix, per-class, misclassifications)
            if should_log_detailed and val_preds is not None and val_labels is not None:
                logger.info(f"  Logging detailed metrics to W&B...")
                self._log_confusion_matrix(val_labels, val_preds, epoch + 1, prefix="val")
                self._log_per_class_metrics(val_labels, val_preds, epoch + 1, prefix="val")
                self._log_misclassification_analysis(val_labels, val_preds, epoch + 1, prefix="val")

            # Check for improvement
            if val_metrics["val_macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_macro_f1"]
                self.epochs_without_improvement = 0

                # Save best checkpoint
                best_path = str(self.output_path / "best_model.pt")
                self.model.save_checkpoint(
                    best_path,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics=metrics,
                )
                logger.info(f"  New best model saved (F1: {self.best_val_f1:.4f})")

                # Log best model artifact to W&B
                self._log_model_artifact(best_path, "best-model", metrics, is_best=True)
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
        test_metrics = self.validate(self.test_loader, prefix="test", return_predictions=True)

        # Extract predictions for detailed analysis
        test_preds = test_metrics.pop("_predictions", None)
        test_labels = test_metrics.pop("_labels", None)

        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Log detailed test metrics
        if test_preds is not None and test_labels is not None:
            logger.info("\nLogging detailed test metrics to W&B...")
            self._log_confusion_matrix(test_labels, test_preds, self.epochs, prefix="test")
            self._log_per_class_metrics(test_labels, test_preds, self.epochs, prefix="test")
            self._log_misclassification_analysis(test_labels, test_preds, self.epochs, prefix="test")

        # Save final checkpoint
        final_path = str(self.output_path / "final_model.pt")
        self.model.save_checkpoint(
            final_path,
            optimizer=self.optimizer,
            epoch=self.epochs,
            metrics=test_metrics,
        )

        # Log final model artifact to W&B
        self._log_model_artifact(final_path, "final-model", test_metrics, is_best=False)

        # Close W&B
        if self.use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        return test_metrics
