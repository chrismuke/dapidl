"""Hierarchical Training Pipeline Step.

Step for training multi-head hierarchical classifiers with:
- Three classification heads (coarse/medium/fine)
- Curriculum learning (progressive head activation)
- Hierarchical loss with consistency penalty
- Cell Ontology integration for label mapping
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class HierarchicalTrainingConfig:
    """Configuration for hierarchical training step."""

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

    # Data loading
    num_workers: int = 8
    use_amp: bool = True

    # Early stopping
    patience: int = 15

    # Logging
    wandb_project: str = "dapidl-hierarchical"
    use_wandb: bool = True

    # Output
    output_dir: str | None = None

    # S3 upload (iDrive e2 compatible)
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"
    s3_region: str = "eu-central-2"
    s3_models_prefix: str = "models/hierarchical"


class HierarchicalTrainingStep(PipelineStep):
    """Train hierarchical multi-head classifier.

    Uses HierarchicalClassifier with:
    - Three classification heads at different granularities
    - Curriculum learning (coarse → medium → fine)
    - Hierarchical loss with consistency penalty
    - Cell Ontology integration for label hierarchy

    Queue: gpu (requires GPU for training)
    """

    name = "hierarchical_training"
    queue = "gpu"

    def __init__(self, config: HierarchicalTrainingConfig | None = None):
        """Initialize hierarchical training step.

        Args:
            config: Training configuration
        """
        self.config = config or HierarchicalTrainingConfig()
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
                "warmup_epochs": {
                    "type": "integer",
                    "default": 5,
                    "description": "Epochs to warm up new head weights",
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
                # Options
                "use_focal": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use focal loss instead of cross-entropy",
                },
                "label_smoothing": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Label smoothing factor",
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

        Requires:
        - patches_path: Path to LMDB dataset with fine-grained labels
        - class_mapping: Dict mapping class names to indices
        """
        outputs = artifacts.outputs
        return "patches_path" in outputs and "class_mapping" in outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute hierarchical training step.

        Args:
            artifacts: Input artifacts from patch extraction

        Returns:
            Output artifacts containing:
            - model_path: Path to best model checkpoint
            - test_metrics: Dict with test set metrics at all levels
            - training_history: Training log with curriculum phases
        """
        import json

        from dapidl.training.hierarchical_trainer import HierarchicalTrainer

        cfg = self.config
        inputs = artifacts.outputs

        # Resolve artifact URLs to local paths
        patches_path = resolve_artifact_path(inputs["patches_path"], "patches_path")
        if patches_path is None:
            raise ValueError("patches_path artifact is required")

        # Load class mapping
        class_mapping_raw = inputs["class_mapping"]
        if isinstance(class_mapping_raw, str):
            class_mapping_path = resolve_artifact_path(class_mapping_raw, "class_mapping")
            if class_mapping_path and class_mapping_path.exists():
                class_mapping = json.loads(class_mapping_path.read_text())
            else:
                class_mapping = json.loads(class_mapping_raw)
        else:
            class_mapping = class_mapping_raw

        # Determine output directory
        if cfg.output_dir:
            output_dir = Path(cfg.output_dir)
        else:
            output_dir = patches_path.parent.parent / "hierarchical_training"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Hierarchical training with {len(class_mapping)} fine-grained classes")
        logger.info(f"Backbone: {cfg.backbone}")
        logger.info(f"Curriculum: coarse_only={cfg.coarse_only_epochs}, "
                   f"coarse_medium={cfg.coarse_medium_epochs}")
        logger.info(f"Output: {output_dir}")

        # Create and run trainer
        trainer = HierarchicalTrainer(
            data_path=patches_path,
            output_path=output_dir,
            # Model params
            backbone_name=cfg.backbone,
            pretrained=cfg.pretrained,
            dropout=cfg.dropout,
            use_shared_projection=cfg.use_shared_projection,
            projection_dim=cfg.projection_dim,
            # Training params
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            # Curriculum params
            coarse_only_epochs=cfg.coarse_only_epochs,
            coarse_medium_epochs=cfg.coarse_medium_epochs,
            warmup_epochs=cfg.warmup_epochs,
            # Loss params
            coarse_weight=cfg.coarse_weight,
            medium_weight=cfg.medium_weight,
            fine_weight=cfg.fine_weight,
            consistency_weight=cfg.consistency_weight,
            label_smoothing=cfg.label_smoothing,
            use_focal=cfg.use_focal,
            focal_gamma=cfg.focal_gamma,
            # Class balancing
            max_weight_ratio=cfg.max_weight_ratio,
            min_samples_per_class=cfg.min_samples_per_class,
            # Other
            use_wandb=cfg.use_wandb,
            project_name=cfg.wandb_project,
            num_workers=cfg.num_workers,
            early_stopping_patience=cfg.patience,
            use_amp=cfg.use_amp,
        )

        # Run training
        test_metrics = trainer.train()

        # Save class mapping for reference
        class_mapping_path = output_dir / "class_mapping.json"
        with open(class_mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=2)

        # Save hierarchy config
        if trainer.hierarchy_config:
            hierarchy_info = {
                "num_coarse": trainer.hierarchy_config.num_coarse,
                "num_medium": trainer.hierarchy_config.num_medium,
                "num_fine": trainer.hierarchy_config.num_fine,
                "coarse_names": trainer.hierarchy_config.coarse_names,
                "medium_names": trainer.hierarchy_config.medium_names,
                "fine_to_coarse": trainer.hierarchy_config.fine_to_coarse,
                "medium_to_coarse": trainer.hierarchy_config.medium_to_coarse,
            }
            hierarchy_path = output_dir / "hierarchy_config.json"
            with open(hierarchy_path, "w") as f:
                json.dump(hierarchy_info, f, indent=2)

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
                "s3_urls": s3_urls,
            },
        )

    def _upload_models_to_s3(
        self,
        output_dir: Path,
        cfg: HierarchicalTrainingConfig,
        test_metrics: dict,
    ) -> dict[str, str]:
        """Upload trained models to S3.

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

        # Generate experiment name from output directory
        exp_name = output_dir.parent.name if output_dir.name == "hierarchical_training" else output_dir.name
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_prefix = f"s3://{cfg.s3_bucket}/{cfg.s3_models_prefix}/{exp_name}-{timestamp}"

        logger.info(f"Uploading hierarchical models to S3: {s3_prefix}")

        # Set up AWS credentials
        env = os.environ.copy()
        if "AWS_ACCESS_KEY_ID" not in env:
            env["AWS_ACCESS_KEY_ID"] = "evkizOGyflbhx5uSi4oV"
            env["AWS_SECRET_ACCESS_KEY"] = "zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9"

        s3_urls = {}
        files_to_upload = [
            "best_model.pt",
            "final_model.pt",
            "training_log.json",
            "hierarchy_config.json",
            "class_mapping.json",
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
        project: str = "DAPIDL/hierarchical",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "clearml_step_runner.py"

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
            "coarse_only_epochs": self.config.coarse_only_epochs,
            "coarse_medium_epochs": self.config.coarse_medium_epochs,
            "warmup_epochs": self.config.warmup_epochs,
            "coarse_weight": self.config.coarse_weight,
            "medium_weight": self.config.medium_weight,
            "fine_weight": self.config.fine_weight,
            "consistency_weight": self.config.consistency_weight,
            "use_focal": self.config.use_focal,
            "patience": self.config.patience,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
