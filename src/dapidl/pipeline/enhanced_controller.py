"""Enhanced ClearML Pipeline Controller with GUI parameter groups.

This module provides a comprehensive pipeline controller that:
- Exposes all parameters to ClearML GUI for easy configuration
- Supports dataset lineage to avoid redundant uploads
- Implements smart step skipping when outputs exist
- Handles multi-patch-size training
- Integrates ensemble annotation with consensus voting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.gui_pipeline_config import GUIPipelineConfig


@dataclass
class EnhancedPipelineResult:
    """Result from pipeline execution."""

    success: bool
    model_path: str | None = None
    model_dataset_id: str | None = None
    annotated_dataset_id: str | None = None
    lmdb_dataset_ids: dict[int, str] = field(default_factory=dict)  # patch_size -> id
    training_metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class EnhancedDAPIDLPipelineController:
    """Enhanced ClearML Pipeline Controller with GUI parameter groups.

    Features:
    - GUI-friendly parameter organization
    - Dataset lineage tracking
    - Smart step skipping
    - Multi-patch-size support
    - Ensemble annotation

    Usage:
        config = GUIPipelineConfig(
            raw_dataset_id="abc123",
            celltypist_models=["Cells_Adult_Breast.pkl"],
            patch_sizes=[128, 256],
            epochs=100,
        )
        controller = EnhancedDAPIDLPipelineController(config)

        # Remote execution
        controller.run(wait=True)

        # Or local execution
        result = controller.run_locally()
    """

    def __init__(self, config: GUIPipelineConfig | None = None):
        """Initialize enhanced pipeline controller.

        Args:
            config: Pipeline configuration with GUI parameters
        """
        self.config = config or GUIPipelineConfig()
        self._pipeline = None

    def create_pipeline(self) -> Any:
        """Create ClearML PipelineController with GUI parameter groups.

        Returns:
            PipelineController instance
        """
        from clearml import PipelineController

        cfg = self.config

        self._pipeline = PipelineController(
            name=f"dapidl-enhanced-{cfg.platform}",
            project="DAPIDL/enhanced-pipelines",
            version="2.0.0",
            add_pipeline_tags=True,
        )

        # Add all parameters organized by groups
        self._add_input_parameters()
        self._add_annotation_parameters()
        self._add_ontology_parameters()
        self._add_lmdb_parameters()
        self._add_training_parameters()
        self._add_output_parameters()

        # Add pipeline steps
        self._add_pipeline_steps()

        return self._pipeline

    def _add_input_parameters(self):
        """Add input selection parameters."""
        cfg = self.config
        p = self._pipeline

        p.add_parameter(
            name="input/raw_dataset_id",
            default=cfg.raw_dataset_id or "",
            description="ClearML Dataset ID for raw spatial data",
        )
        p.add_parameter(
            name="input/raw_dataset_name",
            default=cfg.raw_dataset_name or "",
            description="Search for dataset by name (alternative to ID)",
        )
        p.add_parameter(
            name="input/s3_data_uri",
            default=cfg.s3_data_uri or "",
            description="S3 URI for raw data (e.g., s3://dapidl/raw-data/xenium-breast/)",
        )
        p.add_parameter(
            name="input/platform",
            default=cfg.platform,
            description="Platform: auto, xenium, or merscope",
        )

    def _add_annotation_parameters(self):
        """Add annotation configuration parameters."""
        cfg = self.config
        p = self._pipeline

        p.add_parameter(
            name="annotation/strategy",
            default=cfg.annotation_strategy,
            description="Strategy: ensemble, single, or consensus",
        )
        p.add_parameter(
            name="annotation/celltypist_models",
            default=",".join(cfg.celltypist_models),
            description="CellTypist models (comma-separated)",
        )
        p.add_parameter(
            name="annotation/include_singler",
            default=str(cfg.include_singler),
            description="Include SingleR annotation",
        )
        p.add_parameter(
            name="annotation/singler_reference",
            default=cfg.singler_reference,
            description="SingleR reference: blueprint, hpca, monaco",
        )
        p.add_parameter(
            name="annotation/min_agreement",
            default=str(cfg.min_agreement),
            description="Minimum annotators that must agree",
        )
        p.add_parameter(
            name="annotation/confidence_threshold",
            default=str(cfg.confidence_threshold),
            description="Minimum consensus confidence (0-1)",
        )
        p.add_parameter(
            name="annotation/fine_grained",
            default=str(cfg.fine_grained),
            description="Use detailed cell types vs 3 broad categories",
        )

    def _add_lmdb_parameters(self):
        """Add LMDB creation parameters."""
        cfg = self.config
        p = self._pipeline

        p.add_parameter(
            name="lmdb/patch_sizes",
            default=",".join(str(s) for s in cfg.patch_sizes),
            description="Patch sizes to generate (comma-separated, e.g., 32,64,128,256)",
        )
        p.add_parameter(
            name="lmdb/normalization",
            default=cfg.normalization,
            description="Normalization: adaptive, percentile, minmax",
        )
        p.add_parameter(
            name="lmdb/normalize_physical_size",
            default=str(cfg.normalize_physical_size),
            description="Normalize to consistent physical size (cross-platform)",
        )
        p.add_parameter(
            name="lmdb/skip_if_exists",
            default=str(cfg.skip_if_lmdb_exists),
            description="Skip if matching LMDB dataset exists",
        )

    def _add_training_parameters(self):
        """Add training configuration parameters."""
        cfg = self.config
        p = self._pipeline

        p.add_parameter(
            name="training/backbone",
            default=cfg.backbone,
            description="CNN backbone: efficientnetv2_rw_s, resnet50, convnext_tiny",
        )
        p.add_parameter(
            name="training/training_mode",
            default=cfg.training_mode,
            description="Mode: hierarchical (with curriculum) or flat",
        )
        p.add_parameter(
            name="training/epochs",
            default=str(cfg.epochs),
            description="Training epochs",
        )
        p.add_parameter(
            name="training/batch_size",
            default=str(cfg.batch_size),
            description="Batch size (reduce if OOM)",
        )
        p.add_parameter(
            name="training/learning_rate",
            default=str(cfg.learning_rate),
            description="Learning rate",
        )
        p.add_parameter(
            name="training/coarse_only_epochs",
            default=str(cfg.coarse_only_epochs),
            description="Phase 1 epochs (coarse only)",
        )
        p.add_parameter(
            name="training/coarse_medium_epochs",
            default=str(cfg.coarse_medium_epochs),
            description="Phase 2 epochs (coarse + medium)",
        )
        p.add_parameter(
            name="training/combine_patch_sizes",
            default=str(cfg.combine_patch_sizes),
            description="Train on all patch sizes together",
        )

    def _add_ontology_parameters(self):
        """Add Cell Ontology standardization parameters."""
        cfg = self.config
        p = self._pipeline

        p.add_parameter(
            name="ontology/use_cell_ontology",
            default=str(cfg.use_cell_ontology),
            description="Enable Cell Ontology label standardization",
        )
        p.add_parameter(
            name="ontology/target_level",
            default=cfg.cl_target_level,
            description="Hierarchy level: broad, coarse, or fine",
        )
        p.add_parameter(
            name="ontology/min_confidence",
            default=str(cfg.cl_min_confidence),
            description="Minimum mapping confidence (0-1)",
        )
        p.add_parameter(
            name="ontology/include_unmapped",
            default=str(cfg.cl_include_unmapped),
            description="Include cells with unmapped labels",
        )
        p.add_parameter(
            name="ontology/fuzzy_threshold",
            default=str(cfg.cl_fuzzy_threshold),
            description="Fuzzy matching threshold for label mapping",
        )

    def _add_output_parameters(self):
        """Add output configuration parameters."""
        cfg = self.config
        p = self._pipeline

        p.add_parameter(
            name="output/upload_to_s3",
            default=str(cfg.upload_to_s3),
            description="Upload datasets and models to S3",
        )
        p.add_parameter(
            name="output/s3_bucket",
            default=cfg.s3_bucket,
            description="S3 bucket name",
        )
        p.add_parameter(
            name="output/register_datasets",
            default=str(cfg.register_datasets),
            description="Register datasets with ClearML",
        )
        p.add_parameter(
            name="output/register_models",
            default=str(cfg.register_models),
            description="Register trained models with ClearML",
        )
        p.add_parameter(
            name="output/link_to_parent",
            default=str(cfg.link_to_parent),
            description="Use dataset lineage (saves storage space)",
        )

    def _add_pipeline_steps(self):
        """Add pipeline steps with proper dependencies."""
        cfg = self.config
        p = self._pipeline

        # Step 1: Data Loader
        p.add_step(
            name="data_loader",
            base_task_project="DAPIDL/pipelines",
            base_task_name="step-data_loader",
            parameter_override={
                "step_config/dataset_id": "${pipeline.input/raw_dataset_id}",
                "step_config/platform": "${pipeline.input/platform}",
                "step_config/s3_uri": "${pipeline.input/s3_data_uri}",
            },
            cache_executed_step=True,
        )

        # Step 2: Ensemble Annotation
        p.add_step(
            name="ensemble_annotation",
            parents=["data_loader"],
            base_task_project="DAPIDL/pipelines",
            base_task_name="step-ensemble_annotation",
            parameter_override={
                "step_config/celltypist_models": "${pipeline.annotation/celltypist_models}",
                "step_config/include_singler": "${pipeline.annotation/include_singler}",
                "step_config/singler_reference": "${pipeline.annotation/singler_reference}",
                "step_config/min_agreement": "${pipeline.annotation/min_agreement}",
                "step_config/confidence_threshold": "${pipeline.annotation/confidence_threshold}",
                "step_config/fine_grained": "${pipeline.annotation/fine_grained}",
                "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                "step_config/expression_path": "${data_loader.artifacts.expression_path.url}",
                "step_config/parent_dataset_id": "${data_loader.artifacts.raw_dataset_id.url}",
            },
            cache_executed_step=True,
        )

        # Step 3: Cell Ontology Standardization (optional)
        # This step normalizes all labels to Cell Ontology IDs
        p.add_step(
            name="cl_standardization",
            parents=["ensemble_annotation"],
            base_task_project="DAPIDL/pipelines",
            base_task_name="step-cl_standardization",
            parameter_override={
                "step_config/use_cell_ontology": "${pipeline.ontology/use_cell_ontology}",
                "step_config/target_level": "${pipeline.ontology/target_level}",
                "step_config/min_confidence": "${pipeline.ontology/min_confidence}",
                "step_config/include_unmapped": "${pipeline.ontology/include_unmapped}",
                "step_config/fuzzy_threshold": "${pipeline.ontology/fuzzy_threshold}",
                "step_config/annotations_path": "${ensemble_annotation.artifacts.annotations_parquet.url}",
            },
            cache_executed_step=True,
        )

        # Step 4: LMDB Creation (one per patch size)
        lmdb_step_names = []
        for patch_size in cfg.patch_sizes:
            step_name = f"lmdb_p{patch_size}"
            lmdb_step_names.append(step_name)

            p.add_step(
                name=step_name,
                parents=["cl_standardization"],  # Changed from ensemble_annotation
                base_task_project="DAPIDL/pipelines",
                base_task_name="step-lmdb_creation",
                parameter_override={
                    "step_config/patch_size": str(patch_size),
                    "step_config/normalization_method": "${pipeline.lmdb/normalization}",
                    "step_config/normalize_physical_size": "${pipeline.lmdb/normalize_physical_size}",
                    "step_config/skip_if_exists": "${pipeline.lmdb/skip_if_exists}",
                    "step_config/parent_dataset_id": "${ensemble_annotation.artifacts.annotated_dataset_id.url}",
                    "step_config/annotations_path": "${cl_standardization.artifacts.standardized_annotations.url}",
                    "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                    "step_config/upload_to_s3": "${pipeline.output/upload_to_s3}",
                    "step_config/s3_bucket": "${pipeline.output/s3_bucket}",
                },
                cache_executed_step=True,
            )

        # Step 4: Training
        # Use primary patch size or first one
        primary_patch_size = cfg.primary_patch_size or cfg.patch_sizes[0]
        primary_step = f"lmdb_p{primary_patch_size}"

        training_step_name = "hierarchical_training" if cfg.training_mode == "hierarchical" else "training"

        p.add_step(
            name="training",
            parents=[primary_step],
            base_task_project="DAPIDL/pipelines",
            base_task_name=f"step-{training_step_name}",
            parameter_override={
                "step_config/backbone": "${pipeline.training/backbone}",
                "step_config/epochs": "${pipeline.training/epochs}",
                "step_config/batch_size": "${pipeline.training/batch_size}",
                "step_config/learning_rate": "${pipeline.training/learning_rate}",
                "step_config/coarse_only_epochs": "${pipeline.training/coarse_only_epochs}",
                "step_config/coarse_medium_epochs": "${pipeline.training/coarse_medium_epochs}",
                "step_config/lmdb_path": f"${{{primary_step}.artifacts.lmdb_path.url}}",
                "step_config/class_mapping": "${ensemble_annotation.artifacts.class_mapping.url}",
                "step_config/upload_to_s3": "${pipeline.output/upload_to_s3}",
                "step_config/register_model": "${pipeline.output/register_models}",
            },
            execution_queue="gpu",
            cache_executed_step=False,  # Training should run fresh
        )

        return lmdb_step_names

    def run(self, wait: bool = True) -> str:
        """Run pipeline remotely on ClearML.

        Args:
            wait: If True, wait for completion

        Returns:
            Pipeline run ID
        """
        if self._pipeline is None:
            self.create_pipeline()

        return self._pipeline.start(
            wait=wait,
            queue="services",  # Use services queue for controller
        )

    def run_locally(self) -> EnhancedPipelineResult:
        """Run pipeline locally without ClearML agent.

        Returns:
            Pipeline result with model path and metrics
        """
        from dapidl.pipeline.steps.data_loader import DataLoaderStep, DataLoaderConfig
        from dapidl.pipeline.steps.ensemble_annotation import (
            EnsembleAnnotationStep,
            EnsembleAnnotationConfig,
        )
        from dapidl.pipeline.steps.lmdb_creation import (
            LMDBCreationStep,
            LMDBCreationConfig,
        )
        from dapidl.pipeline.base import StepArtifacts

        cfg = self.config

        try:
            # Step 1: Data Loader
            logger.info("Step 1: Loading data...")
            data_config = DataLoaderConfig(
                dataset_id=cfg.raw_dataset_id,
                s3_uri=cfg.s3_data_uri,
                platform=cfg.platform,
            )
            data_step = DataLoaderStep(data_config)
            data_artifacts = data_step.execute(StepArtifacts())

            # Step 2: Ensemble Annotation
            logger.info("Step 2: Running ensemble annotation...")
            annot_config = EnsembleAnnotationConfig(
                celltypist_models=cfg.celltypist_models,
                include_singler=cfg.include_singler,
                singler_reference=cfg.singler_reference,
                min_agreement=cfg.min_agreement,
                confidence_threshold=cfg.confidence_threshold,
                fine_grained=cfg.fine_grained,
                parent_dataset_id=cfg.raw_dataset_id,
                upload_to_s3=cfg.upload_to_s3,
                s3_bucket=cfg.s3_bucket,
            )
            annot_step = EnsembleAnnotationStep(annot_config)
            annot_artifacts = annot_step.execute(data_artifacts)

            # Step 3: Cell Ontology Standardization (if enabled)
            if cfg.use_cell_ontology:
                logger.info("Step 3: Standardizing labels with Cell Ontology...")
                from dapidl.pipeline.steps.cl_standardization import (
                    CLStandardizationStep,
                    CLStandardizationConfig,
                )

                cl_config = CLStandardizationConfig(
                    target_level=cfg.cl_target_level,
                    min_confidence=cfg.cl_min_confidence,
                    include_unmapped=cfg.cl_include_unmapped,
                    fuzzy_threshold=cfg.cl_fuzzy_threshold,
                )
                cl_step = CLStandardizationStep(cl_config)
                cl_artifacts = cl_step.execute(annot_artifacts)
                # Use CL-standardized annotations for LMDB
                lmdb_input_artifacts = cl_artifacts
            else:
                logger.info("Step 3: Skipping Cell Ontology standardization...")
                lmdb_input_artifacts = annot_artifacts

            # Step 4: LMDB Creation (for each patch size)
            logger.info("Step 4: Creating LMDB datasets...")
            lmdb_results = {}
            for patch_size in cfg.patch_sizes:
                logger.info(f"  Creating LMDB for patch size {patch_size}...")
                lmdb_config = LMDBCreationConfig(
                    patch_size=patch_size,
                    normalization_method=cfg.normalization,
                    normalize_physical_size=cfg.normalize_physical_size,
                    skip_if_exists=cfg.skip_if_lmdb_exists,
                    parent_dataset_id=annot_artifacts.outputs.get("annotated_dataset_id"),
                    upload_to_s3=cfg.upload_to_s3,
                    s3_bucket=cfg.s3_bucket,
                )
                lmdb_step = LMDBCreationStep(lmdb_config)
                lmdb_artifacts = lmdb_step.execute(lmdb_input_artifacts)
                lmdb_results[patch_size] = lmdb_artifacts

            # Step 5: Training
            logger.info("Step 5: Training model...")
            primary_patch_size = cfg.primary_patch_size or cfg.patch_sizes[0]
            primary_lmdb = lmdb_results[primary_patch_size]

            if cfg.training_mode == "hierarchical":
                from dapidl.pipeline.steps.universal_training import (
                    UniversalDAPITrainingStep,
                    UniversalTrainingConfig,
                    TissueDatasetSpec,
                )

                train_config = UniversalTrainingConfig(
                    backbone=cfg.backbone,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    learning_rate=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    patience=cfg.patience,
                    output_dir=cfg.output_dir,
                    coarse_only_epochs=cfg.coarse_only_epochs,
                    coarse_medium_epochs=cfg.coarse_medium_epochs,
                    warmup_epochs=cfg.warmup_epochs,
                    datasets=[
                        TissueDatasetSpec(
                            path=primary_lmdb.outputs["lmdb_path"],
                            tissue=data_artifacts.outputs.get("tissue", "breast"),
                            platform=cfg.platform,
                            confidence_tier=1,
                        )
                    ],
                )
                train_step = UniversalDAPITrainingStep(train_config)
                train_artifacts = train_step.execute(primary_lmdb)
                test_metrics = train_artifacts.outputs.get("test_metrics", {})
                tissue_metrics = train_artifacts.outputs.get("tissue_metrics", {})
            else:
                from dapidl.pipeline.steps.training import TrainingStep, TrainingConfig

                train_config = TrainingConfig(
                    backbone=cfg.backbone,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    learning_rate=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                    patience=cfg.patience,
                    output_dir=cfg.output_dir,
                )
                train_step = TrainingStep(train_config)
                train_artifacts = train_step.execute(primary_lmdb)
                test_metrics = train_artifacts.outputs.get("test_metrics", {})

            model_path = Path(cfg.output_dir) / "final_model.pt"
            if not model_path.exists():
                model_path = Path(cfg.output_dir) / "best_model.pt"

            return EnhancedPipelineResult(
                success=True,
                model_path=str(model_path) if model_path.exists() else None,
                annotated_dataset_id=annot_artifacts.outputs.get("annotated_dataset_id"),
                lmdb_dataset_ids={
                    ps: r.outputs.get("lmdb_dataset_id")
                    for ps, r in lmdb_results.items()
                },
                training_metrics=test_metrics if isinstance(test_metrics, dict) else {},
            )

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return EnhancedPipelineResult(
                success=False,
                error=str(e),
            )

    def get_pipeline_dag(self) -> dict:
        """Return pipeline DAG structure for visualization.

        Returns:
            Dict describing the pipeline structure
        """
        cfg = self.config

        dag = {
            "steps": [
                {
                    "name": "data_loader",
                    "type": "DataLoaderStep",
                    "queue": "default",
                    "cache": True,
                    "parents": [],
                },
                {
                    "name": "ensemble_annotation",
                    "type": "EnsembleAnnotationStep",
                    "queue": "default",
                    "cache": True,
                    "parents": ["data_loader"],
                },
            ],
            "parameters": self.config.to_parameter_dict(),
        }

        # Add LMDB steps
        for patch_size in cfg.patch_sizes:
            dag["steps"].append({
                "name": f"lmdb_p{patch_size}",
                "type": "LMDBCreationStep",
                "queue": "default",
                "cache": True,
                "parents": ["ensemble_annotation"],
                "config": {"patch_size": patch_size},
            })

        # Add training step
        primary_patch_size = cfg.primary_patch_size or cfg.patch_sizes[0]
        dag["steps"].append({
            "name": "training",
            "type": "HierarchicalTrainingStep" if cfg.training_mode == "hierarchical" else "TrainingStep",
            "queue": "gpu",
            "cache": False,
            "parents": [f"lmdb_p{primary_patch_size}"],
        })

        return dag


def create_step_base_tasks(project: str = "DAPIDL/pipelines"):
    """Create base tasks for each pipeline step.

    These tasks serve as templates for the PipelineController.
    They should be created once and then cloned for each pipeline run.

    Args:
        project: ClearML project name for step tasks
    """
    from dapidl.pipeline.steps.data_loader import DataLoaderStep
    from dapidl.pipeline.steps.ensemble_annotation import EnsembleAnnotationStep
    from dapidl.pipeline.steps.lmdb_creation import LMDBCreationStep
    from dapidl.pipeline.steps.training import TrainingStep
    from dapidl.pipeline.steps.hierarchical_training import HierarchicalTrainingStep

    steps = [
        DataLoaderStep(),
        EnsembleAnnotationStep(),
        LMDBCreationStep(),
        TrainingStep(),
        HierarchicalTrainingStep(),
    ]

    created_tasks = []
    for step in steps:
        try:
            task = step.create_clearml_task(project=project)
            created_tasks.append({
                "name": f"step-{step.name}",
                "task_id": task.id,
            })
            logger.info(f"Created base task: step-{step.name} ({task.id})")
        except Exception as e:
            logger.warning(f"Failed to create task for {step.name}: {e}")

    return created_tasks
