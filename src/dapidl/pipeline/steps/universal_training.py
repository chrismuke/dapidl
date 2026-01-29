"""Universal DAPI Training Pipeline Step.

Step for training cross-tissue universal classifiers with:
- Multi-tissue dataset support (breast, lung, liver, etc.)
- Multiple platform support (Xenium, MERSCOPE)
- Tissue-balanced sampling strategies
- Confidence-weighted training (ground truth > consensus > predicted)
- Cell Ontology standardized labels
- Hierarchical classification (coarse/medium/fine)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class TissueDatasetSpec:
    """Specification for a single tissue dataset."""

    path: str
    tissue: str
    platform: str = "xenium"
    confidence_tier: int = 2  # 1=ground truth, 2=consensus, 3=predicted
    weight_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "tissue": self.tissue,
            "platform": self.platform,
            "confidence_tier": self.confidence_tier,
            "weight_multiplier": self.weight_multiplier,
        }


@dataclass
class UniversalTrainingConfig:
    """Configuration for universal cross-tissue training."""

    # Dataset specifications
    datasets: list[TissueDatasetSpec] = field(default_factory=list)

    # Sampling strategy
    sampling_strategy: str = "sqrt"  # "equal", "proportional", "sqrt"

    # Confidence tier weights
    tier1_weight: float = 1.0   # Ground truth
    tier2_weight: float = 0.8   # Consensus
    tier3_weight: float = 0.5   # Single predictor

    # Model architecture
    backbone: str = "efficientnetv2_rw_s"
    pretrained: bool = True
    dropout: float = 0.3
    use_shared_projection: bool = True
    projection_dim: int = 512

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Curriculum learning schedule
    coarse_only_epochs: int = 20
    coarse_medium_epochs: int = 50
    warmup_epochs: int = 5

    # Hierarchical loss weights
    coarse_weight: float = 1.0
    medium_weight: float = 0.5
    fine_weight: float = 0.3
    consistency_weight: float = 0.1

    # Loss options
    label_smoothing: float = 0.1
    use_focal: bool = False
    focal_gamma: float = 2.0

    # Class balancing
    max_weight_ratio: float = 10.0
    min_samples_per_class: int = 20

    # Cell Ontology standardization
    standardize_labels: bool = True

    # Data loading
    num_workers: int = 8
    use_amp: bool = True

    # Early stopping
    patience: int = 15

    # Logging
    wandb_project: str = "dapidl-universal"
    use_wandb: bool = True

    # Output
    output_dir: str | None = None

    # S3 upload
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"
    s3_region: str = "eu-central-2"
    s3_models_prefix: str = "models/universal"

    def add_dataset(
        self,
        path: str,
        tissue: str,
        platform: str = "xenium",
        confidence_tier: int = 2,
        weight_multiplier: float = 1.0,
    ) -> "UniversalTrainingConfig":
        """Add a dataset to the training configuration.

        Args:
            path: Path to LMDB dataset
            tissue: Tissue type (breast, lung, liver, etc.)
            platform: Platform (xenium, merscope)
            confidence_tier: Label confidence (1-3)
            weight_multiplier: Additional weight multiplier

        Returns:
            Self for chaining
        """
        self.datasets.append(TissueDatasetSpec(
            path=path,
            tissue=tissue,
            platform=platform,
            confidence_tier=confidence_tier,
            weight_multiplier=weight_multiplier,
        ))
        return self


class UniversalDAPITrainingStep(PipelineStep):
    """Train universal cross-tissue DAPI classifier.

    Combines multiple tissue datasets for training a universal model:
    - Supports datasets from different tissues and platforms
    - Uses Cell Ontology to standardize labels across datasets
    - Implements tissue-balanced sampling (equal, proportional, sqrt)
    - Supports confidence-weighted training
    - Hierarchical classification with curriculum learning

    Queue: gpu (requires GPU for training)
    """

    name = "universal_training"
    queue = "gpu"

    def __init__(self, config: UniversalTrainingConfig | None = None):
        """Initialize universal training step.

        Args:
            config: Training configuration
        """
        self.config = config or UniversalTrainingConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                # Dataset configuration
                "sampling_strategy": {
                    "type": "string",
                    "enum": ["equal", "proportional", "sqrt"],
                    "default": "sqrt",
                    "description": "Tissue sampling strategy",
                },
                "tier1_weight": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Weight for ground truth labels",
                },
                "tier2_weight": {
                    "type": "number",
                    "default": 0.8,
                    "description": "Weight for consensus labels",
                },
                "tier3_weight": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Weight for predicted labels",
                },
                "standardize_labels": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use Cell Ontology standardization",
                },
                # Model architecture
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
                # Training parameters
                "epochs": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Total training epochs",
                },
                "batch_size": {
                    "type": "integer",
                    "default": 64,
                    "enum": [32, 64, 128, 256],
                    "description": "Training batch size",
                },
                "learning_rate": {
                    "type": "number",
                    "default": 0.0001,
                    "description": "Initial learning rate",
                },
                # Curriculum learning
                "coarse_only_epochs": {
                    "type": "integer",
                    "default": 20,
                    "description": "Epochs to train coarse head only",
                },
                "coarse_medium_epochs": {
                    "type": "integer",
                    "default": 50,
                    "description": "Epochs to train coarse+medium heads",
                },
                # Loss weights
                "coarse_weight": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Loss weight for coarse classification",
                },
                "medium_weight": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Loss weight for medium classification",
                },
                "fine_weight": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Loss weight for fine classification",
                },
                "consistency_weight": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Weight for hierarchy consistency penalty",
                },
                "patience": {
                    "type": "integer",
                    "default": 15,
                    "description": "Early stopping patience",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        For universal training, inputs can come from:
        1. Multiple patches_path_N keys (from parallel patch extraction)
        2. A dataset_configs list with paths and metadata

        Returns:
            True if valid inputs found
        """
        outputs = artifacts.outputs

        # Check for dataset_configs list
        if "dataset_configs" in outputs:
            configs = outputs["dataset_configs"]
            if isinstance(configs, list) and len(configs) > 0:
                return all("path" in c and "tissue" in c for c in configs)

        # Check for patches_path_N pattern
        has_patches = any(
            k.startswith("patches_path") for k in outputs.keys()
        )
        return has_patches

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute universal training step.

        Args:
            artifacts: Input artifacts from patch extraction step(s)

        Returns:
            Output artifacts containing:
            - model_path: Path to best model checkpoint
            - test_metrics: Dict with test set metrics
            - tissue_metrics: Per-tissue metrics breakdown
        """
        import json

        # Initialize ClearML tracking (if not already in a task context)
        clearml_task = None
        clearml_logger = None
        try:
            from clearml import Task
            # Check if we're already in a task context
            clearml_task = Task.current_task()
            if clearml_task is None:
                # Create new task for tracking
                clearml_task = Task.init(
                    project_name="DAPIDL/training",
                    task_name=f"universal-training-{self.config.backbone}",
                    task_type=Task.TaskTypes.training,
                    reuse_last_task_id=False,
                )
                # Log hyperparameters
                clearml_task.connect({
                    "backbone": self.config.backbone,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "patience": self.config.patience,
                    "sampling_strategy": self.config.sampling_strategy,
                    "use_hierarchical": True,
                    "coarse_weight": self.config.coarse_weight,
                    "medium_weight": self.config.medium_weight,
                    "fine_weight": self.config.fine_weight,
                })
            clearml_logger = clearml_task.get_logger()
            logger.info(f"ClearML tracking enabled: {clearml_task.task_id}")
        except Exception as e:
            logger.warning(f"ClearML init failed: {e}")

        from dapidl.data.multi_tissue_dataset import (
            MultiTissueConfig,
            MultiTissueDataset,
            TissueDatasetConfig,
            create_multi_tissue_splits,
            create_multi_tissue_dataloaders,
        )
        from dapidl.training.hierarchical_trainer import HierarchicalTrainer

        cfg = self.config
        inputs = artifacts.outputs

        # Build MultiTissueConfig from inputs or config
        mt_config = MultiTissueConfig(
            sampling_strategy=cfg.sampling_strategy,
            use_hierarchical=True,
            confidence_weights={
                1: cfg.tier1_weight,
                2: cfg.tier2_weight,
                3: cfg.tier3_weight,
            },
            min_samples_per_class=cfg.min_samples_per_class,
            standardize_labels=cfg.standardize_labels,
        )

        # Add datasets from config
        for ds in cfg.datasets:
            resolved_path = resolve_artifact_path(ds.path, f"dataset_{ds.tissue}")
            if resolved_path is None:
                raise ValueError(f"Could not resolve path: {ds.path}")
            mt_config.add_dataset(
                path=str(resolved_path),
                tissue=ds.tissue,
                platform=ds.platform,
                confidence_tier=ds.confidence_tier,
                weight_multiplier=ds.weight_multiplier,
            )

        # Also add datasets from artifacts (if not already in config)
        # Deduplicate by checking existing paths
        existing_paths = {str(ds.path) for ds in mt_config.datasets}
        if "dataset_configs" in inputs:
            for ds_spec in inputs["dataset_configs"]:
                # Handle both TissueDatasetSpec objects and dicts
                if isinstance(ds_spec, TissueDatasetSpec):
                    path = ds_spec.path
                    tissue = ds_spec.tissue
                    platform = ds_spec.platform
                    confidence_tier = ds_spec.confidence_tier
                    weight_multiplier = ds_spec.weight_multiplier
                else:
                    path = ds_spec["path"]
                    tissue = ds_spec["tissue"]
                    platform = ds_spec.get("platform", "xenium")
                    confidence_tier = ds_spec.get("confidence_tier", 2)
                    weight_multiplier = ds_spec.get("weight_multiplier", 1.0)

                resolved_path = resolve_artifact_path(path, f"dataset_{tissue}")
                if resolved_path and str(resolved_path) not in existing_paths:
                    mt_config.add_dataset(
                        path=str(resolved_path),
                        tissue=tissue,
                        platform=platform,
                        confidence_tier=confidence_tier,
                        weight_multiplier=weight_multiplier,
                    )
                    existing_paths.add(str(resolved_path))

        # Add from patches_path_N pattern
        for key, value in inputs.items():
            if key.startswith("patches_path"):
                resolved_path = resolve_artifact_path(value, key)
                if resolved_path:
                    # Try to infer tissue/platform from path
                    parts = str(resolved_path).lower()
                    tissue = "unknown"
                    platform = "xenium"
                    for t in ["breast", "lung", "liver", "ovarian", "brain"]:
                        if t in parts:
                            tissue = t
                            break
                    if "merscope" in parts or "vizgen" in parts:
                        platform = "merscope"

                    # Check if not already added
                    existing_paths = [str(d.path) for d in mt_config.datasets]
                    if str(resolved_path) not in existing_paths:
                        mt_config.add_dataset(
                            path=str(resolved_path),
                            tissue=tissue,
                            platform=platform,
                            confidence_tier=2,
                        )

        if len(mt_config.datasets) == 0:
            raise ValueError("No datasets found in inputs or config")

        logger.info(f"Universal training with {len(mt_config.datasets)} datasets:")
        for ds in mt_config.datasets:
            logger.info(f"  - {ds.tissue}/{ds.platform}: {ds.path} (tier {ds.confidence_tier})")

        # Determine output directory
        if cfg.output_dir:
            output_dir = Path(cfg.output_dir)
        else:
            output_dir = Path(mt_config.datasets[0].path).parent.parent / "universal_training"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create multi-tissue datasets
        train_dataset, val_dataset, test_dataset = create_multi_tissue_splits(
            mt_config,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_by="both",
        )

        # Create dataloaders
        train_loader, val_loader, test_loader = create_multi_tissue_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            use_weighted_sampler=True,
        )

        # Log dataset stats
        logger.info(f"Train: {len(train_dataset)} samples")
        logger.info(f"Val: {len(val_dataset)} samples")
        logger.info(f"Test: {len(test_dataset)} samples")
        logger.info(f"Classes: {train_dataset.num_classes}")

        # Get hierarchy config
        hierarchy_config = train_dataset.hierarchy_config
        if hierarchy_config:
            logger.info(f"Hierarchy: coarse={hierarchy_config.num_coarse}, "
                       f"medium={hierarchy_config.num_medium}, "
                       f"fine={hierarchy_config.num_fine}")

        # Run training using custom training loop
        test_metrics, tissue_metrics = self._train_universal_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_dataset=train_dataset,
            output_dir=output_dir,
            cfg=cfg,
            clearml_logger=clearml_logger,
            clearml_task=clearml_task,
        )

        # Close datasets
        train_dataset.close()
        val_dataset.close()
        test_dataset.close()

        # Save configuration
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "datasets": [d.to_dict() for d in cfg.datasets],
                "sampling_strategy": cfg.sampling_strategy,
                "backbone": cfg.backbone,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "standardize_labels": cfg.standardize_labels,
            }, f, indent=2)

        # Upload to S3 if configured
        s3_urls = {}
        if cfg.upload_to_s3:
            s3_urls = self._upload_models_to_s3(output_dir, cfg, test_metrics)

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,
                "model_path": str(output_dir / "best_model.pt"),
                "final_model_path": str(output_dir / "final_model.pt"),
                "hierarchy_config_path": str(output_dir / "hierarchy_config.json"),
                "test_metrics": test_metrics,
                "tissue_metrics": tissue_metrics,
                "s3_urls": s3_urls,
            },
        )

    def _train_universal_model(
        self,
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        output_dir: Path,
        cfg: UniversalTrainingConfig,
        clearml_logger=None,  # ClearML Logger for tracking
        clearml_task=None,    # ClearML Task for artifacts
    ) -> tuple[dict, dict]:
        """Train the universal model with multi-tissue data.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            train_dataset: Training dataset for metadata
            output_dir: Output directory
            cfg: Training configuration

        Returns:
            Tuple of (test_metrics, tissue_metrics)
        """
        import json
        import torch
        import torch.nn as nn
        from torch.cuda.amp import GradScaler, autocast

        from dapidl.models.hierarchical import HierarchicalClassifier
        from dapidl.training.hierarchical_loss import HierarchicalLoss, CurriculumScheduler
        from dapidl.data.hierarchical_dataset import HierarchicalLabels

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {device}")

        # Get hierarchy config from dataset
        hierarchy_config = train_dataset.hierarchy_config

        # Create model
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name=cfg.backbone,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
            use_shared_projection=cfg.use_shared_projection,
            projection_dim=cfg.projection_dim,
        )
        model = model.to(device)

        # Get class weights at each hierarchy level
        class_weights = train_dataset.get_hierarchical_class_weights(
            max_weight_ratio=cfg.max_weight_ratio
        )
        coarse_weights = class_weights.get("coarse_weights", None)
        medium_weights = class_weights.get("medium_weights", None)
        fine_weights = class_weights.get("fine_weights", None)

        if coarse_weights is not None:
            coarse_weights = coarse_weights.to(device)
        if medium_weights is not None:
            medium_weights = medium_weights.to(device)
        if fine_weights is not None:
            fine_weights = fine_weights.to(device)

        # Create loss and curriculum scheduler
        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            coarse_class_weights=coarse_weights,
            medium_class_weights=medium_weights,
            fine_class_weights=fine_weights,
            coarse_weight=cfg.coarse_weight,
            medium_weight=cfg.medium_weight,
            fine_weight=cfg.fine_weight,
            consistency_weight=cfg.consistency_weight,
            label_smoothing=cfg.label_smoothing,
        )

        scheduler_curriculum = CurriculumScheduler(
            coarse_only_epochs=cfg.coarse_only_epochs,
            coarse_medium_epochs=cfg.coarse_medium_epochs,
            warmup_epochs=cfg.warmup_epochs,
            transition_lr_factor=0.1,  # 10% LR during phase transitions
            freeze_backbone_epochs=2,  # Freeze backbone for 2 epochs at phase start
        )

        # Optimizer and LR scheduler
        # Use base LR - will be modulated by curriculum scheduler
        base_lr = cfg.learning_rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )

        # Mixed precision scaler
        scaler = GradScaler() if cfg.use_amp else None

        # W&B logging
        if cfg.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.wandb_project,
                    config={
                        "backbone": cfg.backbone,
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "learning_rate": cfg.learning_rate,
                        "sampling_strategy": cfg.sampling_strategy,
                        "num_datasets": len(train_dataset.config.datasets),
                        "total_samples": len(train_dataset),
                        "num_classes": train_dataset.num_classes,
                    },
                )
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")
                cfg.use_wandb = False

        # Training history
        history = {
            "train_losses": [],
            "val_losses": [],
            "val_f1_coarse": [],
            "val_f1_medium": [],
            "val_f1_fine": [],
            "curriculum_phase": [],
        }

        best_val_f1 = 0.0
        patience_counter = 0
        current_phase = 0  # Track phase for reset on transition
        phase_best_f1 = {}  # Track best F1 per phase

        # Training loop
        for epoch in range(cfg.epochs):
            # Get curriculum phase
            active_heads = scheduler_curriculum.get_active_heads(epoch)
            phase_num = scheduler_curriculum.get_phase_number(epoch)

            # Reset patience and best_f1 on phase transition
            if phase_num != current_phase:
                if current_phase > 0:
                    # Save best model from previous phase
                    phase_best_f1[current_phase] = best_val_f1
                    logger.info(
                        f"Phase {current_phase} complete. Best F1={best_val_f1:.4f}. "
                        f"Resetting for Phase {phase_num}."
                    )
                current_phase = phase_num
                best_val_f1 = 0.0
                patience_counter = 0
            phase_name = scheduler_curriculum.get_phase_name(epoch)
            loss_weights = scheduler_curriculum.get_loss_weights(
                epoch,
                base_weights=(cfg.coarse_weight, cfg.medium_weight, cfg.fine_weight, cfg.consistency_weight)
            )

            # Get LR multiplier for phase transitions
            lr_mult = scheduler_curriculum.get_lr_multiplier(epoch)
            current_lr = lr_scheduler.get_last_lr()[0] * lr_mult

            # Apply LR multiplier to optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Handle backbone freezing during phase transitions
            should_freeze = scheduler_curriculum.should_freeze_backbone(epoch)
            if should_freeze:
                model.freeze_backbone()
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} - Phase: {phase_name} (backbone frozen, LR={current_lr:.2e})")
            else:
                model.unfreeze_backbone()
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} - Phase: {phase_name} (LR={current_lr:.2e})")

            # Update loss weights
            loss_fn.coarse_weight = loss_weights["coarse_weight"]
            loss_fn.medium_weight = loss_weights["medium_weight"]
            loss_fn.fine_weight = loss_weights["fine_weight"]

            # Train one epoch
            train_loss = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scaler,
                device, active_heads, cfg.use_amp
            )

            # Validate
            val_loss, val_metrics = self._validate_epoch(
                model, val_loader, loss_fn, device, active_heads, hierarchy_config
            )

            # Update LR
            lr_scheduler.step()

            # Log metrics
            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["val_f1_coarse"].append(val_metrics.get("f1_coarse", 0))
            history["val_f1_medium"].append(val_metrics.get("f1_medium", 0))
            history["val_f1_fine"].append(val_metrics.get("f1_fine", 0))
            history["curriculum_phase"].append(phase_name)

            # Use appropriate F1 based on phase
            if "fine" in active_heads:
                current_f1 = val_metrics.get("f1_fine", 0)
            elif "medium" in active_heads:
                current_f1 = val_metrics.get("f1_medium", 0)
            else:
                current_f1 = val_metrics.get("f1_coarse", 0)

            logger.info(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_f1={current_f1:.4f}"
            )

            # Save best model (per phase)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
                # Save phase-specific best model
                model.save_checkpoint(str(output_dir / f"best_model_phase{current_phase}.pt"))
                # Also save as overall best (will be latest phase's best)
                model.save_checkpoint(str(output_dir / "best_model.pt"))
                logger.info(f"New best model (Phase {current_phase}): F1={best_val_f1:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= cfg.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            # W&B logging
            if cfg.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1_coarse": val_metrics.get("f1_coarse", 0),
                    "val_f1_medium": val_metrics.get("f1_medium", 0),
                    "val_f1_fine": val_metrics.get("f1_fine", 0),
                    "phase": phase_name,
                    "lr": lr_scheduler.get_last_lr()[0],
                })

            # ClearML logging
            if clearml_logger is not None:
                clearml_logger.report_scalar("loss", "train", train_loss, epoch + 1)
                clearml_logger.report_scalar("loss", "validation", val_loss, epoch + 1)
                clearml_logger.report_scalar("f1", "coarse", val_metrics.get("f1_coarse", 0), epoch + 1)
                clearml_logger.report_scalar("f1", "medium", val_metrics.get("f1_medium", 0), epoch + 1)
                clearml_logger.report_scalar("f1", "fine", val_metrics.get("f1_fine", 0), epoch + 1)
                clearml_logger.report_scalar("lr", "learning_rate", lr_scheduler.get_last_lr()[0], epoch + 1)
                clearml_logger.report_text(f"Phase: {phase_name}", iteration=epoch + 1)

        # Save final model
        model.save_checkpoint(str(output_dir / "final_model.pt"))

        # Record final phase best
        phase_best_f1[current_phase] = best_val_f1

        # Log per-phase best metrics
        logger.info("Training complete. Per-phase best F1:")
        for phase, f1 in sorted(phase_best_f1.items()):
            phase_names = {1: "Coarse Only", 2: "Coarse + Medium", 3: "All Heads"}
            logger.info(f"  Phase {phase} ({phase_names.get(phase, 'Unknown')}): F1={f1:.4f}")

        # Save training history
        history["best_val_f1"] = best_val_f1
        history["phase_best_f1"] = phase_best_f1
        with open(output_dir / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)

        # Test assessment
        test_metrics, tissue_metrics = self._run_test_assessment(
            model, test_loader, device, hierarchy_config, train_dataset
        )

        # Save hierarchy config
        if hierarchy_config:
            with open(output_dir / "hierarchy_config.json", "w") as f:
                json.dump({
                    "num_coarse": hierarchy_config.num_coarse,
                    "num_medium": hierarchy_config.num_medium,
                    "num_fine": hierarchy_config.num_fine,
                    "coarse_names": hierarchy_config.coarse_names,
                    "medium_names": hierarchy_config.medium_names,
                    "fine_names": hierarchy_config.fine_names,
                    "fine_to_coarse": hierarchy_config.fine_to_coarse,
                    "medium_to_coarse": hierarchy_config.medium_to_coarse,
                }, f, indent=2)

        if cfg.use_wandb:
            import wandb
            wandb.finish()

        # ClearML: Upload models and log final metrics
        if clearml_task is not None:
            try:
                # Upload model artifacts
                clearml_task.upload_artifact(
                    "best_model",
                    str(output_dir / "best_model.pt"),
                    delete_after_upload=False,
                )
                clearml_task.upload_artifact(
                    "final_model",
                    str(output_dir / "final_model.pt"),
                    delete_after_upload=False,
                )
                clearml_task.upload_artifact(
                    "training_log",
                    str(output_dir / "training_log.json"),
                    delete_after_upload=False,
                )
                if hierarchy_config:
                    clearml_task.upload_artifact(
                        "hierarchy_config",
                        str(output_dir / "hierarchy_config.json"),
                        delete_after_upload=False,
                    )

                # Log final test metrics as summary scalars
                clearml_logger.report_single_value("test/f1_coarse", test_metrics.get("f1_coarse", 0))
                clearml_logger.report_single_value("test/f1_medium", test_metrics.get("f1_medium", 0))
                clearml_logger.report_single_value("test/f1_fine", test_metrics.get("f1_fine", 0))
                clearml_logger.report_single_value("test/accuracy_coarse", test_metrics.get("accuracy_coarse", 0))
                clearml_logger.report_single_value("best_val_f1", best_val_f1)

                # Log per-tissue metrics
                if tissue_metrics:
                    for tissue_key, metrics in tissue_metrics.items():
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                clearml_logger.report_single_value(f"tissue/{tissue_key}/{metric_name}", value)

                logger.info("ClearML artifacts uploaded successfully")
            except Exception as e:
                logger.warning(f"ClearML artifact upload failed: {e}")

        return test_metrics, tissue_metrics

    def _train_epoch(self, model, loader, loss_fn, optimizer, scaler,
                     device, active_heads, use_amp):
        """Train one epoch."""
        import torch

        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            # Extract targets - pass None for inactive heads (curriculum learning)
            coarse_targets = labels["coarse"].to(device)
            medium_targets = labels["medium"].to(device) if "medium" in active_heads else None
            fine_targets = labels["fine"].to(device) if "fine" in active_heads else None

            optimizer.zero_grad()

            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss, _ = loss_fn(outputs, coarse_targets, medium_targets, fine_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss, _ = loss_fn(outputs, coarse_targets, medium_targets, fine_targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(self, model, loader, loss_fn, device, active_heads, hierarchy_config):
        """Validate one epoch."""
        import torch
        from sklearn.metrics import f1_score

        model.train(False)
        total_loss = 0.0
        all_preds = {"coarse": [], "medium": [], "fine": []}
        all_labels = {"coarse": [], "medium": [], "fine": []}

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                # Extract targets - pass None for inactive heads (curriculum learning)
                coarse_targets = labels["coarse"].to(device)
                medium_targets = labels["medium"].to(device) if "medium" in active_heads else None
                fine_targets = labels["fine"].to(device) if "fine" in active_heads else None

                outputs = model(images)
                loss, _ = loss_fn(outputs, coarse_targets, medium_targets, fine_targets)
                total_loss += loss.item()

                # Get predictions at each level from logits
                coarse_preds = outputs.coarse_logits.argmax(dim=-1)
                all_preds["coarse"].extend(coarse_preds.cpu().numpy())
                all_labels["coarse"].extend(coarse_targets.cpu().numpy())

                if "medium" in active_heads and outputs.medium_logits is not None:
                    medium_preds = outputs.medium_logits.argmax(dim=-1)
                    all_preds["medium"].extend(medium_preds.cpu().numpy())
                    all_labels["medium"].extend(medium_targets.cpu().numpy())

                if "fine" in active_heads and outputs.fine_logits is not None:
                    fine_preds = outputs.fine_logits.argmax(dim=-1)
                    all_preds["fine"].extend(fine_preds.cpu().numpy())
                    all_labels["fine"].extend(fine_targets.cpu().numpy())

        # Compute F1 scores
        metrics = {}
        # Coarse is always active
        metrics["f1_coarse"] = f1_score(
            all_labels["coarse"], all_preds["coarse"], average="macro", zero_division=0
        )
        if "medium" in active_heads and all_labels["medium"]:
            metrics["f1_medium"] = f1_score(
                all_labels["medium"], all_preds["medium"], average="macro", zero_division=0
            )
        if "fine" in active_heads and all_labels["fine"]:
            metrics["f1_fine"] = f1_score(
                all_labels["fine"], all_preds["fine"], average="macro", zero_division=0
            )

        return total_loss / len(loader), metrics

    def _run_test_assessment(self, model, test_loader, device, hierarchy_config, train_dataset):
        """Run test set assessment with per-tissue breakdown."""
        import torch
        from sklearn.metrics import f1_score, accuracy_score, classification_report
        import numpy as np

        model.train(False)
        all_preds = {"coarse": [], "medium": [], "fine": []}
        all_labels = {"coarse": [], "medium": [], "fine": []}
        all_tissue_indices = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)

                outputs = model(images)

                # Get predictions at each level from logits
                coarse_preds = outputs.coarse_logits.argmax(dim=-1)
                all_preds["coarse"].extend(coarse_preds.cpu().numpy())
                all_labels["coarse"].extend(labels["coarse"].numpy())

                if outputs.medium_logits is not None:
                    medium_preds = outputs.medium_logits.argmax(dim=-1)
                    all_preds["medium"].extend(medium_preds.cpu().numpy())
                    all_labels["medium"].extend(labels["medium"].numpy())

                if outputs.fine_logits is not None:
                    fine_preds = outputs.fine_logits.argmax(dim=-1)
                    all_preds["fine"].extend(fine_preds.cpu().numpy())
                    all_labels["fine"].extend(labels["fine"].numpy())

                all_tissue_indices.extend(labels["tissue_idx"].numpy())

        all_tissue_indices = np.array(all_tissue_indices)

        # Overall metrics
        test_metrics = {}
        for level in ["coarse", "medium", "fine"]:
            if all_labels[level]:  # Only compute if we have predictions
                test_metrics[f"f1_{level}"] = f1_score(
                    all_labels[level], all_preds[level], average="macro", zero_division=0
                )
                test_metrics[f"accuracy_{level}"] = accuracy_score(
                    all_labels[level], all_preds[level]
                )
            else:
                test_metrics[f"f1_{level}"] = 0.0
                test_metrics[f"accuracy_{level}"] = 0.0

        logger.info(f"Test F1 coarse: {test_metrics['f1_coarse']:.4f}")
        if all_labels["medium"]:
            logger.info(f"Test F1 medium: {test_metrics['f1_medium']:.4f}")
        if all_labels["fine"]:
            logger.info(f"Test F1 fine: {test_metrics['f1_fine']:.4f}")

        # Per-tissue metrics
        tissue_metrics = {}
        for i, ds_config in enumerate(train_dataset.config.datasets):
            mask = all_tissue_indices == i
            if mask.sum() > 0:
                tissue_key = f"{ds_config.tissue}/{ds_config.platform}"
                tissue_metrics[tissue_key] = {}
                for level in ["coarse", "medium", "fine"]:
                    if all_labels[level]:  # Only if we have predictions for this level
                        preds_t = np.array(all_preds[level])[mask]
                        labels_t = np.array(all_labels[level])[mask]
                        tissue_metrics[tissue_key][f"f1_{level}"] = f1_score(
                            labels_t, preds_t, average="macro", zero_division=0
                        )
                tissue_metrics[tissue_key]["n_samples"] = int(mask.sum())

                # Log the finest available level
                if "f1_fine" in tissue_metrics[tissue_key]:
                    logger.info(f"  {tissue_key}: F1_fine={tissue_metrics[tissue_key]['f1_fine']:.4f} "
                               f"(n={tissue_metrics[tissue_key]['n_samples']})")
                elif "f1_medium" in tissue_metrics[tissue_key]:
                    logger.info(f"  {tissue_key}: F1_medium={tissue_metrics[tissue_key]['f1_medium']:.4f} "
                               f"(n={tissue_metrics[tissue_key]['n_samples']})")
                else:
                    logger.info(f"  {tissue_key}: F1_coarse={tissue_metrics[tissue_key].get('f1_coarse', 0):.4f} "
                               f"(n={tissue_metrics[tissue_key]['n_samples']})")

        return test_metrics, tissue_metrics

    def _upload_models_to_s3(
        self,
        output_dir: Path,
        cfg: UniversalTrainingConfig,
        test_metrics: dict,
    ) -> dict[str, str]:
        """Upload trained models to S3."""
        import os
        import subprocess
        from datetime import datetime

        # Generate experiment name
        exp_name = output_dir.parent.name if output_dir.name == "universal_training" else output_dir.name
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_prefix = f"s3://{cfg.s3_bucket}/{cfg.s3_models_prefix}/{exp_name}-{timestamp}"

        logger.info(f"Uploading universal models to S3: {s3_prefix}")

        # ALWAYS set correct credentials - don't rely on environment which may be corrupted
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = "evkizOGyflbhx5uSi4oV"
        env["AWS_SECRET_ACCESS_KEY"] = "zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9"

        s3_urls = {}
        files_to_upload = [
            "best_model.pt",
            "final_model.pt",
            "training_log.json",
            "hierarchy_config.json",
            "training_config.json",
        ]

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

        return s3_urls

    def create_clearml_task(
        self,
        project: str = "DAPIDL/universal",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "clearml_step_runner_universal_training.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.training,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,
            packages=["-e ."],
        )

        # Store all configuration parameters
        params = {
            "step_name": self.name,
            "backbone": self.config.backbone,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "sampling_strategy": self.config.sampling_strategy,
            "tier1_weight": self.config.tier1_weight,
            "tier2_weight": self.config.tier2_weight,
            "tier3_weight": self.config.tier3_weight,
            "standardize_labels": self.config.standardize_labels,
            "coarse_only_epochs": self.config.coarse_only_epochs,
            "coarse_medium_epochs": self.config.coarse_medium_epochs,
            "patience": self.config.patience,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        # Add dataset configs
        for i, ds in enumerate(self.config.datasets):
            self._task.set_parameter(f"datasets/{i}/path", ds.path)
            self._task.set_parameter(f"datasets/{i}/tissue", ds.tissue)
            self._task.set_parameter(f"datasets/{i}/platform", ds.platform)
            self._task.set_parameter(f"datasets/{i}/tier", ds.confidence_tier)

        return self._task
