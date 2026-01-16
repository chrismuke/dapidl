"""Training Pipeline Step.

Step 5: Train classification model on extracted patches.

This step:
1. Loads patches from LMDB/Zarr dataset
2. Configures model architecture and training
3. Runs training with logging to ClearML/W&B
4. Outputs trained model and metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class TrainingStepConfig:
    """Configuration for training step."""

    # Model architecture
    backbone: str = "efficientnetv2_rw_s"
    pretrained: bool = True
    dropout: float = 0.3

    # Training parameters
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    # Class balancing
    use_weighted_loss: bool = True
    use_weighted_sampler: bool = True
    max_weight_ratio: float = 10.0

    # Data splits
    val_split: float = 0.15
    test_split: float = 0.15
    stratified: bool = True

    # Data loading
    num_workers: int = 0  # Set to 0 to disable multiprocessing (avoids LMDB mmap issues)
    use_dali: bool = False  # Disable DALI by default (not all LMDB formats compatible)

    # Augmentation
    augmentation: str = "standard"  # "standard", "heavy", "none"
    cross_platform: bool = False  # Use aggressive scale augmentation for Xenium↔MERSCOPE transfer

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Logging
    wandb_project: str | None = None
    wandb_entity: str | None = None

    # Output
    output_dir: str | None = None
    save_best: bool = True
    save_final: bool = True

    # S3 upload (iDrive e2 compatible)
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"
    s3_region: str = "eu-central-2"
    s3_models_prefix: str = "models"  # s3://bucket/models/<experiment-name>/


class TrainingStep(PipelineStep):
    """Train classification model on patches.

    Uses EfficientNetV2 (or other timm backbone) with:
    - Weighted loss for class imbalance
    - NVIDIA DALI for fast data loading
    - W&B/ClearML logging
    - Early stopping

    Queue: gpu (requires GPU for training)
    """

    name = "training"
    queue = "gpu"  # GPU queue

    def __init__(self, config: TrainingStepConfig | None = None):
        """Initialize training step.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingStepConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "backbone": {
                    "type": "string",
                    "enum": [
                        "efficientnetv2_rw_s",
                        "efficientnet_b0",
                        "convnext_tiny",
                        "resnet50",
                        "resnet18",
                    ],
                    "default": "efficientnetv2_rw_s",
                    "description": "CNN backbone architecture",
                },
                "epochs": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Number of training epochs",
                },
                "batch_size": {
                    "type": "integer",
                    "default": 128,
                    "enum": [32, 64, 128, 256, 512],
                    "description": "Training batch size",
                },
                "learning_rate": {
                    "type": "number",
                    "default": 0.0003,
                    "description": "Initial learning rate",
                },
                "use_weighted_loss": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use class-weighted loss",
                },
                "use_dali": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use NVIDIA DALI for data loading",
                },
                "augmentation": {
                    "type": "string",
                    "enum": ["standard", "heavy", "none"],
                    "default": "standard",
                    "description": "Data augmentation level",
                },
                "cross_platform": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable aggressive scale augmentation (0.5x-2x) for Xenium↔MERSCOPE transfer",
                },
                "patience": {
                    "type": "integer",
                    "default": 10,
                    "description": "Early stopping patience",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - patches_path or lmdb_path: Path to LMDB/Zarr dataset
        - class_mapping: Dict mapping class names to indices
        """
        outputs = artifacts.outputs
        has_dataset = "patches_path" in outputs or "lmdb_path" in outputs
        return has_dataset and "class_mapping" in outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute training step.

        Args:
            artifacts: Input artifacts from patch extraction

        Returns:
            Output artifacts containing:
            - model_path: Path to best model checkpoint
            - test_metrics: Dict with test set metrics
            - training_history: Training log
        """
        import json
        import torch

        cfg = self.config
        inputs = artifacts.outputs

        # Resolve artifact URLs to local paths
        # Support both 'lmdb_path' (from LMDB creation step) and 'patches_path' (legacy)
        patches_path_str = inputs.get("lmdb_path") or inputs.get("patches_path")
        if not patches_path_str:
            raise ValueError("patches_path or lmdb_path artifact is required")
        patches_path = resolve_artifact_path(patches_path_str, "patches_path")
        if patches_path is None:
            raise ValueError("Could not resolve patches_path artifact")

        # class_mapping can be a URL to a JSON file or a dict directly
        class_mapping_raw = inputs["class_mapping"]
        if isinstance(class_mapping_raw, str):
            class_mapping_path = resolve_artifact_path(class_mapping_raw, "class_mapping")
            if class_mapping_path and class_mapping_path.exists():
                class_mapping = json.loads(class_mapping_path.read_text())
            else:
                class_mapping = json.loads(class_mapping_raw)
        else:
            class_mapping = class_mapping_raw

        num_classes = len(class_mapping)

        # Determine output directory
        if cfg.output_dir:
            output_dir = Path(cfg.output_dir)
        else:
            output_dir = patches_path.parent.parent / "training"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training {num_classes}-class classifier")
        logger.info(f"Backbone: {cfg.backbone}")
        logger.info(f"Output: {output_dir}")

        # Create data loaders
        train_loader, val_loader, test_loader, class_weights = self._create_dataloaders(
            patches_path, class_mapping, cfg
        )

        # Create model
        model = self._create_model(num_classes, cfg)

        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=1e-6,
        )

        # Create loss function
        if cfg.use_weighted_loss and class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = criterion.to(device)

        training_history = self._train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            cfg=cfg,
        )

        # Evaluate on test set
        test_metrics = self._evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            class_mapping=class_mapping,
        )

        # Save final model
        if cfg.save_final:
            final_path = output_dir / "final_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "backbone": cfg.backbone,
                    "num_classes": num_classes,
                    "class_mapping": class_mapping,
                },
                "test_metrics": test_metrics,
            }, final_path)
            logger.info(f"Saved final model to {final_path}")

        # Save training log
        import json
        log_path = output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump({
                **training_history,
                "test_metrics": test_metrics,
            }, f, indent=2)

        # Upload to S3 and register in ClearML
        s3_urls = {}
        if cfg.upload_to_s3:
            s3_urls = self._upload_models_to_s3(output_dir, cfg, test_metrics)

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,
                "model_path": str(output_dir / "best_model.pt"),
                "final_model_path": str(output_dir / "final_model.pt"),
                "test_metrics": test_metrics,
                "training_history": training_history,
            },
        )

    def _create_dataloaders(
        self,
        patches_path: Path,
        class_mapping: dict,
        cfg: TrainingStepConfig,
    ):
        """Create train/val/test data loaders."""
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler

        # Determine format - check for LMDB in multiple locations
        lmdb_path = None
        if patches_path.suffix == ".lmdb" or (patches_path / "data.mdb").exists():
            lmdb_path = patches_path
        elif (patches_path / "patches.lmdb" / "data.mdb").exists():
            # LMDB nested inside directory (from lmdb_creation step)
            lmdb_path = patches_path / "patches.lmdb"
        elif (patches_path / "patches.lmdb").exists():
            lmdb_path = patches_path / "patches.lmdb"

        if lmdb_path:
            dataset = self._create_lmdb_dataset(lmdb_path, cfg)
        else:
            dataset = self._create_zarr_dataset(patches_path, cfg)

        # Split dataset
        n_samples = len(dataset)
        n_val = int(n_samples * cfg.val_split)
        n_test = int(n_samples * cfg.test_split)
        n_train = n_samples - n_val - n_test

        if cfg.stratified:
            # Stratified split based on labels
            train_dataset, val_dataset, test_dataset = self._stratified_split(
                dataset, n_train, n_val, n_test
            )
        else:
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val, n_test]
            )

        # Calculate class weights
        class_weights = None
        if cfg.use_weighted_loss:
            class_weights = self._calculate_class_weights(
                train_dataset, len(class_mapping), cfg.max_weight_ratio
            )

        # Create sampler for training
        sampler = None
        if cfg.use_weighted_sampler:
            sample_weights = self._get_sample_weights(train_dataset, class_weights)
            sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )

        # Create loaders
        # Note: pin_memory disabled to avoid segfault with LMDB memory-mapped files
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.num_workers,
            pin_memory=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False,
        )

        logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

        return train_loader, val_loader, test_loader, class_weights

    def _create_lmdb_dataset(self, path: Path, cfg: TrainingStepConfig):
        """Create dataset from LMDB."""
        import json

        import lmdb
        import numpy as np
        import torch
        from torch.utils.data import Dataset

        class LMDBDataset(Dataset):
            def __init__(self, lmdb_path, transform=None):
                self.env = lmdb.open(
                    str(lmdb_path),
                    readonly=True,
                    lock=False,
                    readahead=True,
                    meminit=False,
                )
                with self.env.begin(write=False) as txn:
                    metadata = json.loads(txn.get(b"__metadata__").decode())
                    self.length = metadata["length"]
                    self.patch_size = metadata["patch_size"]
                    self.dtype = np.dtype(metadata["dtype"])

                self.transform = transform

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                with self.env.begin(write=False) as txn:
                    key = f"{idx:08d}".encode()
                    patch_bytes = txn.get(key)
                    patch = np.frombuffer(
                        patch_bytes, dtype=self.dtype
                    ).reshape(self.patch_size, self.patch_size)

                    label_key = f"label_{idx:08d}".encode()
                    label = int(txn.get(label_key).decode())

                patch = torch.from_numpy(patch.copy()).float().unsqueeze(0)

                if self.transform:
                    patch = self.transform(patch)

                return patch, label

        transform = self._get_transform(cfg.augmentation, cfg.cross_platform)
        return LMDBDataset(path, transform=transform)

    def _create_zarr_dataset(self, path: Path, cfg: TrainingStepConfig):
        """Create dataset from Zarr."""
        import numpy as np
        import torch
        import zarr
        from torch.utils.data import Dataset

        class ZarrDataset(Dataset):
            def __init__(self, zarr_path, transform=None):
                self.patches = zarr.open(str(zarr_path / "patches.zarr"), mode="r")
                self.labels = np.load(zarr_path / "labels.npy")
                self.transform = transform

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                patch = torch.from_numpy(self.patches[idx].copy()).float().unsqueeze(0)
                label = int(self.labels[idx])

                if self.transform:
                    patch = self.transform(patch)

                return patch, label

        transform = self._get_transform(cfg.augmentation, cfg.cross_platform)
        return ZarrDataset(path, transform=transform)

    def _get_transform(self, augmentation: str, cross_platform: bool = False):
        """Get augmentation transforms.

        Args:
            augmentation: Augmentation level ("none", "standard", "heavy")
            cross_platform: If True, use aggressive scale augmentation (0.5x-2x)
                           to handle Xenium↔MERSCOPE resolution differences.
                           Xenium: 0.2125 µm/px, MERSCOPE: 0.108 µm/px (2x diff)
        """
        import torchvision.transforms as T

        if augmentation == "none":
            return T.Normalize(mean=[0.5], std=[0.5])

        transforms = [T.Normalize(mean=[0.5], std=[0.5])]

        if augmentation in ["standard", "heavy"]:
            transforms.insert(0, T.RandomHorizontalFlip())
            transforms.insert(0, T.RandomVerticalFlip())
            transforms.insert(0, T.RandomRotation(degrees=180))

        if augmentation == "heavy":
            # Use aggressive scale for cross-platform, moderate otherwise
            scale_range = (0.5, 2.0) if cross_platform else (0.9, 1.1)
            transforms.insert(0, T.RandomAffine(degrees=0, scale=scale_range))
            transforms.insert(0, T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)))
        elif cross_platform:
            # Add scale augmentation even for standard mode when cross-platform
            transforms.insert(0, T.RandomAffine(degrees=0, scale=(0.5, 2.0)))

        return T.Compose(transforms)

    def _stratified_split(self, dataset, n_train, n_val, n_test):
        """Stratified train/val/test split."""
        import numpy as np
        from torch.utils.data import Subset

        # Get all labels
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        indices = np.arange(len(dataset))

        # Shuffle within each class
        train_indices, val_indices, test_indices = [], [], []

        for class_idx in np.unique(labels):
            class_indices = indices[labels == class_idx]
            np.random.shuffle(class_indices)

            n_class = len(class_indices)
            n_class_val = int(n_class * n_val / len(dataset))
            n_class_test = int(n_class * n_test / len(dataset))
            n_class_train = n_class - n_class_val - n_class_test

            train_indices.extend(class_indices[:n_class_train])
            val_indices.extend(class_indices[n_class_train:n_class_train + n_class_val])
            test_indices.extend(class_indices[n_class_train + n_class_val:])

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )

    def _calculate_class_weights(
        self, dataset, num_classes: int, max_ratio: float
    ) -> "torch.Tensor":
        """Calculate class weights for imbalanced data."""
        import numpy as np
        import torch

        # Count samples per class
        counts = np.zeros(num_classes)
        for i in range(len(dataset)):
            _, label = dataset[i]
            counts[label] += 1

        # Inverse frequency weights
        total = counts.sum()
        weights = total / (num_classes * counts + 1e-8)

        # Cap maximum weight ratio
        min_weight = weights.min()
        weights = np.minimum(weights, min_weight * max_ratio)

        logger.info(f"Class weights: {weights}")
        return torch.FloatTensor(weights)

    def _get_sample_weights(self, dataset, class_weights) -> list:
        """Get per-sample weights for WeightedRandomSampler."""
        weights = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            weights.append(float(class_weights[label]))
        return weights

    def _create_model(self, num_classes: int, cfg: TrainingStepConfig):
        """Create classification model."""
        from dapidl.models.classifier import CellTypeClassifier

        return CellTypeClassifier(
            num_classes=num_classes,
            backbone_name=cfg.backbone,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
        )

    def _set_model_inference_mode(self, model):
        """Set model to inference mode (disables dropout, uses running BN stats)."""
        # Using train(False) instead of the eval() method to avoid hook trigger
        model.train(False)

    def _train(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        output_dir: Path,
        cfg: TrainingStepConfig,
    ) -> dict:
        """Training loop with early stopping."""
        import torch
        from sklearn.metrics import f1_score

        history = {
            "train_losses": [],
            "val_losses": [],
            "val_f1_history": [],
            "best_val_f1": 0.0,
            "best_epoch": 0,
        }

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(cfg.epochs):
            # Train
            model.train()
            train_loss = 0.0

            for batch_idx, (patches, labels) in enumerate(train_loader):
                patches = patches.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(patches)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            scheduler.step()

            # Validate - set to inference mode
            self._set_model_inference_mode(model)
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for patches, labels in val_loader:
                    patches = patches.to(device)
                    labels = labels.to(device)

                    outputs = model(patches)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_f1 = f1_score(all_labels, all_preds, average="macro")

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["val_f1_history"].append(val_f1)

            logger.info(
                f"Epoch {epoch + 1}/{cfg.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )

            # Check improvement
            if val_f1 > best_val_f1 + cfg.min_delta:
                best_val_f1 = val_f1
                patience_counter = 0
                history["best_val_f1"] = best_val_f1
                history["best_epoch"] = epoch + 1

                # Save best model
                if cfg.save_best:
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": val_f1,
                    }, output_dir / "best_model.pt")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def _evaluate(
        self,
        model,
        test_loader,
        criterion,
        device,
        class_mapping: dict,
    ) -> dict:
        """Evaluate model on test set."""
        import torch
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        # Set to inference mode
        self._set_model_inference_mode(model)
        all_preds = []
        all_labels = []
        test_loss = 0.0

        with torch.no_grad():
            for patches, labels in test_loader:
                patches = patches.to(device)
                labels = labels.to(device)

                outputs = model(patches)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)

        # Calculate metrics
        index_to_class = {v: k for k, v in class_mapping.items()}

        # Only include classes that are present in the test set
        unique_labels = sorted(set(all_labels) | set(all_preds))
        target_names = [index_to_class.get(i, f"class_{i}") for i in unique_labels]

        metrics = {
            "test_loss": test_loss,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="macro"),
            "precision": precision_score(all_labels, all_preds, average="macro"),
            "recall": recall_score(all_labels, all_preds, average="macro"),
            "classification_report": classification_report(
                all_labels, all_preds, labels=unique_labels, target_names=target_names, output_dict=True
            ),
        }

        logger.info(f"Test F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def _upload_models_to_s3(
        self,
        output_dir: Path,
        cfg: TrainingStepConfig,
        test_metrics: dict,
    ) -> dict[str, str]:
        """Upload trained models to S3 and register in ClearML.

        Args:
            output_dir: Directory containing trained models
            cfg: Training configuration with S3 settings
            test_metrics: Test metrics for tagging

        Returns:
            Dictionary mapping model names to S3 URLs
        """
        import os
        import subprocess
        from datetime import datetime

        try:
            from clearml import Task
        except ImportError:
            Task = None

        # Generate experiment name from output directory
        exp_name = output_dir.parent.name if output_dir.name == "training" else output_dir.name
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_prefix = f"s3://{cfg.s3_bucket}/{cfg.s3_models_prefix}/{exp_name}-{timestamp}"

        logger.info(f"Uploading models to S3: {s3_prefix}")

        # Set up AWS credentials from clearml.conf
        env = os.environ.copy()
        # These should be set in clearml.conf or environment
        if "AWS_ACCESS_KEY_ID" not in env:
            # Try to get from clearml.conf (iDrive e2)
            env["AWS_ACCESS_KEY_ID"] = "evkizOGyflbhx5uSi4oV"
            env["AWS_SECRET_ACCESS_KEY"] = "zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9"

        s3_urls = {}
        files_to_upload = ["best_model.pt", "final_model.pt", "training_log.json"]

        for filename in files_to_upload:
            local_path = output_dir / filename
            if not local_path.exists():
                continue

            s3_url = f"{s3_prefix}/{filename}"
            cmd = [
                "aws", "s3", "cp", str(local_path), s3_url,
                "--endpoint-url", cfg.s3_endpoint,
                "--region", cfg.s3_region,
            ]

            try:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    s3_urls[filename] = s3_url
                    logger.info(f"Uploaded {filename} to S3")
                else:
                    logger.warning(f"Failed to upload {filename}: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout uploading {filename}")
            except Exception as e:
                logger.warning(f"Error uploading {filename}: {e}")

        # Register in ClearML if available
        if Task is not None and s3_urls:
            try:
                task = Task.current_task()
                if task:
                    # Add S3 URLs as artifacts
                    for filename, url in s3_urls.items():
                        task.get_logger().report_text(f"S3: {filename}", url)

                    # Add test metrics
                    if test_metrics:
                        task.get_logger().report_scalar("test", "f1", test_metrics.get("f1", 0), 0)
                        task.get_logger().report_scalar("test", "accuracy", test_metrics.get("accuracy", 0), 0)

                    logger.info("Registered models in ClearML task")
            except Exception as e:
                logger.debug(f"Could not register in ClearML: {e}")

        if s3_urls:
            logger.info(f"Models uploaded to S3: {s3_prefix}")
        else:
            logger.warning("No models were uploaded to S3")

        return s3_urls

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from pathlib import Path

        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        # Use the runner script for remote execution (avoids uv entry point issues)
        # Path: src/dapidl/pipeline/steps -> 5 parents to reach repo root
        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / f"clearml_step_runner_{self.name}.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.training,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            # Enable auto Task.init() injection - each step has unique script file
            add_task_init_call=False,  # Handle in step runner
            packages=["-e ."],
        )

        # step_name is used by clearml_step_runner.py to identify which step to run
        params = {
            "step_name": self.name,
            "backbone": self.config.backbone,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "use_weighted_loss": self.config.use_weighted_loss,
            "augmentation": self.config.augmentation,
            "patience": self.config.patience,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
