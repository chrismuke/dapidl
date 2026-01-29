"""Unified Pipeline Controller for DAPIDL.

This is a proof-of-concept controller that uses the unified configuration
from unified_config.py. It demonstrates how to:

1. Use the consolidated DAPIDLPipelineConfig
2. Create ClearML pipelines with proper parameter groups
3. Execute pipelines both remotely and locally
4. Support both single-dataset and multi-tissue modes

This controller will eventually replace:
- controller.py (DAPIDLPipelineController)
- enhanced_controller.py (EnhancedDAPIDLPipelineController)
- universal_controller.py (UniversalDAPIPipelineController)

Usage:
    from dapidl.pipeline.unified_config import (
        DAPIDLPipelineConfig, InputConfig, TrainingConfig, Platform
    )
    from dapidl.pipeline.unified_controller import UnifiedPipelineController

    # Simple single-dataset pipeline
    config = DAPIDLPipelineConfig(
        input=InputConfig(dataset_id="abc123", platform=Platform.XENIUM),
        training=TrainingConfig(epochs=100),
    )
    controller = UnifiedPipelineController(config)
    controller.run()

    # Multi-tissue universal training
    config = DAPIDLPipelineConfig()
    config.input.add_tissue("breast", dataset_id="abc123", confidence_tier=1)
    config.input.add_tissue("lung", dataset_id="def456", confidence_tier=2)
    controller = UnifiedPipelineController(config)
    result = controller.run_locally()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.unified_config import (
    AnnotationStrategy,
    DAPIDLPipelineConfig,
    Platform,
    TrainingMode,
)


@dataclass
class PipelineResult:
    """Result from unified pipeline execution."""

    success: bool
    model_path: str | None = None
    model_dataset_id: str | None = None
    annotated_dataset_id: str | None = None
    lmdb_dataset_ids: dict[int, str] = field(default_factory=dict)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class UnifiedPipelineController:
    """Unified Pipeline Controller using consolidated configuration.

    This controller supports both single-dataset and multi-tissue pipelines
    through a single, consistent interface.

    Modes:
        1. Single Dataset: Standard pipeline with one input dataset
        2. Multi-Tissue: Universal training with multiple tissue datasets

    The mode is automatically determined from the configuration:
    - If config.input.tissues is populated -> Multi-tissue mode
    - Otherwise -> Single dataset mode

    Example:
        ```python
        # Single dataset mode
        config = DAPIDLPipelineConfig(
            input=InputConfig(dataset_id="abc123"),
            training=TrainingConfig(epochs=100),
        )
        controller = UnifiedPipelineController(config)
        controller.run()

        # Multi-tissue mode
        config = DAPIDLPipelineConfig()
        config.input.add_tissue("breast", dataset_id="abc")
        config.input.add_tissue("lung", dataset_id="def")
        controller = UnifiedPipelineController(config)
        controller.run()
        ```
    """

    def __init__(self, config: DAPIDLPipelineConfig | None = None):
        """Initialize the unified pipeline controller.

        Args:
            config: Unified pipeline configuration
        """
        self.config = config or DAPIDLPipelineConfig()
        self._pipeline = None
        self._step_tasks = {}

    @property
    def is_multi_tissue(self) -> bool:
        """Check if this is a multi-tissue pipeline."""
        return len(self.config.input.tissues) > 0

    def create_pipeline(self) -> Any:
        """Create ClearML PipelineController with unified parameter groups.

        Returns:
            PipelineController instance
        """
        from clearml import PipelineController

        cfg = self.config

        # Determine pipeline name based on mode
        if self.is_multi_tissue:
            pipeline_name = f"{cfg.name}-universal"
        else:
            pipeline_name = f"{cfg.name}-{cfg.input.platform.value}"

        self._pipeline = PipelineController(
            name=pipeline_name,
            project=cfg.project,
            version=cfg.version,
            add_pipeline_tags=True,
        )

        # Set default execution queue
        if cfg.execution.execute_remotely:
            self._pipeline.set_default_execution_queue(cfg.execution.default_queue)

        # Add all parameters organized by GUI groups
        self._add_pipeline_parameters()

        # Add pipeline steps based on mode
        if self.is_multi_tissue:
            self._add_multi_tissue_steps()
        else:
            self._add_single_dataset_steps()

        logger.info(f"Created unified pipeline: {pipeline_name} v{cfg.version}")
        return self._pipeline

    def _add_pipeline_parameters(self):
        """Add parameters to ClearML pipeline organized by GUI groups."""
        params = self.config.to_clearml_parameters()

        # Add each parameter with proper grouping
        for key, value in params.items():
            # Derive description from the key name
            param_name = key.split("/")[-1]
            description = param_name.replace("_", " ").title()

            self._pipeline.add_parameter(
                name=key,
                default=value,
                description=description,
            )

    def _add_single_dataset_steps(self):
        """Add pipeline steps for single-dataset mode."""
        cfg = self.config
        p = self._pipeline

        # Step 1: Data Loader
        p.add_step(
            name="data_loader",
            base_task_project=cfg.project,
            base_task_name="step-data_loader",
            parameter_override={
                "step_config/dataset_id": "${pipeline.input/dataset_id}",
                "step_config/platform": "${pipeline.input/platform}",
                "step_config/local_path": "${pipeline.input/local_path}",
                "step_config/s3_uri": "${pipeline.input/s3_uri}",
            },
            execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_data_steps,
        )

        # Step 2: Ensemble Annotation
        p.add_step(
            name="ensemble_annotation",
            parents=["data_loader"],
            base_task_project=cfg.project,
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
            },
            execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_data_steps,
        )

        # Step 3: Cell Ontology Standardization (if enabled)
        if cfg.ontology.enabled:
            p.add_step(
                name="cl_standardization",
                parents=["ensemble_annotation"],
                base_task_project=cfg.project,
                base_task_name="step-cl_standardization",
                parameter_override={
                    "step_config/target_level": "${pipeline.ontology/target_level}",
                    "step_config/min_confidence": "${pipeline.ontology/min_confidence}",
                    "step_config/include_unmapped": "${pipeline.ontology/include_unmapped}",
                    "step_config/fuzzy_threshold": "${pipeline.ontology/fuzzy_threshold}",
                    "step_config/annotations_parquet": "${ensemble_annotation.artifacts.annotations_parquet.url}",
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )
            annotation_parent = "cl_standardization"
        else:
            annotation_parent = "ensemble_annotation"

        # Step 4: LMDB Creation (one per patch size)
        lmdb_step_names = []
        for patch_size in cfg.lmdb.patch_sizes:
            step_name = f"lmdb_p{patch_size}"
            lmdb_step_names.append(step_name)

            p.add_step(
                name=step_name,
                parents=[annotation_parent],
                base_task_project=cfg.project,
                base_task_name="step-lmdb_creation",
                parameter_override={
                    "step_config/patch_size": str(patch_size),
                    "step_config/normalization_method": "${pipeline.lmdb/normalization}",
                    "step_config/normalize_physical_size": "${pipeline.lmdb/normalize_physical_size}",
                    "step_config/skip_if_exists": "${pipeline.lmdb/skip_if_exists}",
                    "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                    "step_config/upload_to_s3": "${pipeline.output/upload_to_s3}",
                    "step_config/s3_bucket": "${pipeline.output/s3_bucket}",
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )

        # Step 5: Training
        primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
        primary_lmdb_step = f"lmdb_p{primary_patch_size}"

        training_step_name = "hierarchical_training" if cfg.training.mode == TrainingMode.HIERARCHICAL else "training"

        p.add_step(
            name="training",
            parents=[primary_lmdb_step],
            base_task_project=cfg.project,
            base_task_name=f"step-{training_step_name}",
            parameter_override={
                "step_config/backbone": "${pipeline.training/backbone}",
                "step_config/epochs": "${pipeline.training/epochs}",
                "step_config/batch_size": "${pipeline.training/batch_size}",
                "step_config/learning_rate": "${pipeline.training/learning_rate}",
                "step_config/weight_decay": "${pipeline.training/weight_decay}",
                "step_config/patience": "${pipeline.training/patience}",
                "step_config/coarse_only_epochs": "${pipeline.training/coarse_only_epochs}",
                "step_config/coarse_medium_epochs": "${pipeline.training/coarse_medium_epochs}",
                "step_config/lmdb_path": f"${{{primary_lmdb_step}.artifacts.lmdb_path.url}}",
                "step_config/upload_to_s3": "${pipeline.output/upload_to_s3}",
                "step_config/register_model": "${pipeline.output/register_models}",
            },
            execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_training,
        )

        # Optional: Cross-modal Validation
        if cfg.validation.enabled:
            p.add_step(
                name="cross_validation",
                parents=["training"],
                base_task_project=cfg.project,
                base_task_name="step-cross_validation",
                parameter_override={
                    "step_config/run_leiden_check": "${pipeline.validation/run_leiden_check}",
                    "step_config/run_dapi_check": "${pipeline.validation/run_dapi_check}",
                    "step_config/min_ari_threshold": "${pipeline.validation/min_ari_threshold}",
                    "step_config/model_path": "${training.artifacts.model_path.url}",
                },
                execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=True,
            )

        # Optional: Cross-platform Transfer
        if cfg.transfer.enabled:
            validation_parent = "cross_validation" if cfg.validation.enabled else "training"
            p.add_step(
                name="cross_platform_transfer",
                parents=[validation_parent],
                base_task_project=cfg.project,
                base_task_name="step-cross_platform_transfer",
                parameter_override={
                    "step_config/target_dataset_id": "${pipeline.transfer/target_dataset_id}",
                    "step_config/target_local_path": "${pipeline.transfer/target_local_path}",
                    "step_config/target_platform": "${pipeline.transfer/target_platform}",
                    "step_config/model_path": "${training.artifacts.model_path.url}",
                },
                execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=True,
            )

        # Optional: Documentation
        if cfg.documentation.enabled:
            # Find last step
            if cfg.transfer.enabled:
                doc_parent = "cross_platform_transfer"
            elif cfg.validation.enabled:
                doc_parent = "cross_validation"
            else:
                doc_parent = "training"

            p.add_step(
                name="documentation",
                parents=[doc_parent],
                base_task_project=cfg.project,
                base_task_name="step-documentation",
                parameter_override={
                    "step_config/obsidian_vault_path": "${pipeline.documentation/obsidian_vault_path}",
                    "step_config/obsidian_folder": "${pipeline.documentation/obsidian_folder}",
                    "step_config/template": "${pipeline.documentation/template}",
                    "step_config/experiment_name": cfg.name,
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=True,
            )

        return lmdb_step_names

    def _add_multi_tissue_steps(self):
        """Add pipeline steps for multi-tissue universal training mode."""
        cfg = self.config
        p = self._pipeline

        patch_extraction_steps = []

        # Create per-tissue processing steps
        for i, tissue_cfg in enumerate(cfg.input.tissues):
            tissue_name = tissue_cfg.tissue
            step_prefix = f"{tissue_name}_{i}"

            # Data Loader for this tissue
            p.add_step(
                name=f"data_loader_{step_prefix}",
                base_task_project=cfg.project,
                base_task_name="step-data_loader",
                parameter_override={
                    "step_config/dataset_id": tissue_cfg.dataset_id or "",
                    "step_config/platform": tissue_cfg.platform.value,
                    "step_config/local_path": tissue_cfg.local_path or "",
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )

            # Segmentation for this tissue
            p.add_step(
                name=f"segmentation_{step_prefix}",
                parents=[f"data_loader_{step_prefix}"],
                base_task_project=cfg.project,
                base_task_name="step-segmentation",
                parameter_override={
                    "step_config/segmenter": cfg.segmentation.segmenter.value,
                    "step_config/diameter": str(cfg.segmentation.diameter),
                    "step_config/flow_threshold": str(cfg.segmentation.flow_threshold),
                    "step_config/data_path": f"${{data_loader_{step_prefix}.artifacts.data_path.url}}",
                },
                execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )

            # Annotation for this tissue
            p.add_step(
                name=f"annotation_{step_prefix}",
                parents=[f"data_loader_{step_prefix}"],
                base_task_project=cfg.project,
                base_task_name="step-annotation",
                parameter_override={
                    "step_config/annotator": "celltypist",
                    "step_config/strategy": "consensus",
                    "step_config/model_names": ",".join(cfg.annotation.celltypist_models),
                    "step_config/fine_grained": str(cfg.annotation.fine_grained),
                    "step_config/data_path": f"${{data_loader_{step_prefix}.artifacts.data_path.url}}",
                    "step_config/platform": f"${{data_loader_{step_prefix}.artifacts.platform.url}}",
                    "step_config/cells_parquet": f"${{data_loader_{step_prefix}.artifacts.cells_parquet.url}}",
                    "step_config/expression_path": f"${{data_loader_{step_prefix}.artifacts.expression_path.url}}",
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )

            # LMDB Creation for this tissue
            patch_step_name = f"lmdb_{step_prefix}"
            primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]

            p.add_step(
                name=patch_step_name,
                parents=[f"segmentation_{step_prefix}", f"annotation_{step_prefix}"],
                base_task_project=cfg.project,
                base_task_name="step-lmdb_creation",
                parameter_override={
                    "step_config/patch_size": str(primary_patch_size),
                    "step_config/normalization_method": cfg.lmdb.normalization.value,
                    "step_config/data_path": f"${{data_loader_{step_prefix}.artifacts.data_path.url}}",
                    "step_config/annotations_parquet": f"${{annotation_{step_prefix}.artifacts.annotations_parquet.url}}",
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )
            patch_extraction_steps.append(patch_step_name)

            logger.info(f"Added processing steps for tissue: {tissue_name}")

        # Universal Training (depends on all tissue LMDB steps)
        p.add_step(
            name="universal_training",
            parents=patch_extraction_steps,
            base_task_project=cfg.project,
            base_task_name="step-universal_training",
            parameter_override={
                "step_config/backbone": cfg.training.backbone.value,
                "step_config/epochs": str(cfg.training.epochs),
                "step_config/batch_size": str(cfg.training.batch_size),
                "step_config/learning_rate": str(cfg.training.learning_rate),
                "step_config/sampling_strategy": cfg.training.sampling_strategy.value,
                "step_config/tier1_weight": str(cfg.training.tier1_weight),
                "step_config/tier2_weight": str(cfg.training.tier2_weight),
                "step_config/tier3_weight": str(cfg.training.tier3_weight),
                "step_config/coarse_only_epochs": str(cfg.training.coarse_only_epochs),
                "step_config/coarse_medium_epochs": str(cfg.training.coarse_medium_epochs),
                # Pass all LMDB paths
                **{
                    f"step_config/lmdb_path_{i}": f"${{{step}.artifacts.lmdb_path.url}}"
                    for i, step in enumerate(patch_extraction_steps)
                },
            },
            execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_training,
        )

        return patch_extraction_steps

    def run(self) -> str:
        """Run pipeline: controller runs locally, steps execute on ClearML agents.

        Uses start_locally(run_pipeline_steps_locally=False) which keeps the
        controller process alive locally while dispatching individual steps
        to remote ClearML agent queues (default/gpu).

        This blocks until the pipeline completes.

        Returns:
            Pipeline run ID
        """
        if self._pipeline is None:
            self.create_pipeline()

        mode = "universal" if self.is_multi_tissue else "single-dataset"
        logger.info(f"Starting unified pipeline execution ({mode} mode)...")
        logger.info("Controller runs locally; steps dispatch to ClearML agents.")

        # Controller runs in this process, steps are enqueued to agent queues.
        # This blocks until all steps complete.
        self._pipeline.start_locally(run_pipeline_steps_locally=False)

        logger.info("Pipeline completed")
        return self._pipeline.id

    def run_locally(self) -> PipelineResult:
        """Run pipeline locally without ClearML agents.

        Useful for debugging or when ClearML agents aren't available.

        Returns:
            PipelineResult with outputs from all steps
        """
        if self.is_multi_tissue:
            return self._run_locally_multi_tissue()
        else:
            return self._run_locally_single_dataset()

    def _run_locally_single_dataset(self) -> PipelineResult:
        """Run single-dataset pipeline locally."""
        from dapidl.pipeline.base import StepArtifacts
        from dapidl.pipeline.steps.data_loader import DataLoaderConfig, DataLoaderStep
        from dapidl.pipeline.steps.ensemble_annotation import (
            EnsembleAnnotationConfig,
            EnsembleAnnotationStep,
        )
        from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep

        cfg = self.config

        try:
            # Step 1: Data Loader
            logger.info("Step 1: Loading data...")
            data_config = DataLoaderConfig(
                dataset_id=cfg.input.dataset_id,
                dataset_name=cfg.input.dataset_name,
                dataset_project=cfg.input.dataset_project,
                local_path=cfg.input.local_path,
                s3_uri=cfg.input.s3_uri,
                platform=cfg.input.platform.value,
            )
            data_step = DataLoaderStep(data_config)
            data_artifacts = data_step.execute(StepArtifacts())

            # Step 2: Ensemble Annotation
            logger.info("Step 2: Running ensemble annotation...")
            annot_config = EnsembleAnnotationConfig(
                celltypist_models=cfg.annotation.celltypist_models,
                include_singler=cfg.annotation.include_singler,
                singler_reference=cfg.annotation.singler_reference,
                min_agreement=cfg.annotation.min_agreement,
                confidence_threshold=cfg.annotation.confidence_threshold,
                fine_grained=cfg.annotation.fine_grained,
                upload_to_s3=cfg.output.upload_to_s3,
                s3_bucket=cfg.output.s3_bucket,
            )
            annot_step = EnsembleAnnotationStep(annot_config)
            annot_artifacts = annot_step.execute(data_artifacts)

            # Step 3: Cell Ontology Standardization (if enabled)
            if cfg.ontology.enabled:
                logger.info("Step 3: Standardizing labels with Cell Ontology...")
                from dapidl.pipeline.steps.cl_standardization import (
                    CLStandardizationConfig,
                    CLStandardizationStep,
                )

                cl_config = CLStandardizationConfig(
                    target_level=cfg.ontology.target_level.value,
                    min_confidence=cfg.ontology.min_confidence,
                    include_unmapped=cfg.ontology.include_unmapped,
                    fuzzy_threshold=cfg.ontology.fuzzy_threshold,
                )
                cl_step = CLStandardizationStep(cl_config)
                lmdb_input_artifacts = cl_step.execute(annot_artifacts)
            else:
                logger.info("Step 3: Skipping Cell Ontology standardization...")
                lmdb_input_artifacts = annot_artifacts

            # Step 4: LMDB Creation (for each patch size)
            logger.info("Step 4: Creating LMDB datasets...")
            lmdb_results = {}
            for patch_size in cfg.lmdb.patch_sizes:
                logger.info(f"  Creating LMDB for patch size {patch_size}...")
                lmdb_config = LMDBCreationConfig(
                    patch_size=patch_size,
                    normalization_method=cfg.lmdb.normalization.value,
                    normalize_physical_size=cfg.lmdb.normalize_physical_size,
                    skip_if_exists=cfg.lmdb.skip_if_exists,
                    upload_to_s3=cfg.output.upload_to_s3,
                    s3_bucket=cfg.output.s3_bucket,
                )
                lmdb_step = LMDBCreationStep(lmdb_config)
                lmdb_artifacts = lmdb_step.execute(lmdb_input_artifacts)
                lmdb_results[patch_size] = lmdb_artifacts

            # Step 5: Training
            logger.info("Step 5: Training model...")
            primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
            primary_lmdb = lmdb_results[primary_patch_size]

            if cfg.training.mode == TrainingMode.HIERARCHICAL:
                from dapidl.pipeline.steps.hierarchical_training import (
                    HierarchicalTrainingConfig,
                    HierarchicalTrainingStep,
                )

                train_config = HierarchicalTrainingConfig(
                    backbone=cfg.training.backbone.value,
                    epochs=cfg.training.epochs,
                    batch_size=cfg.training.batch_size,
                    learning_rate=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                    patience=cfg.training.patience,
                    coarse_only_epochs=cfg.training.coarse_only_epochs,
                    coarse_medium_epochs=cfg.training.coarse_medium_epochs,
                    output_dir=cfg.output.output_dir,
                )
                train_step = HierarchicalTrainingStep(train_config)
            else:
                from dapidl.pipeline.steps.training import TrainingStep, TrainingStepConfig

                train_config = TrainingStepConfig(
                    backbone=cfg.training.backbone.value,
                    epochs=cfg.training.epochs,
                    batch_size=cfg.training.batch_size,
                    learning_rate=cfg.training.learning_rate,
                    weight_decay=cfg.training.weight_decay,
                    patience=cfg.training.patience,
                    output_dir=cfg.output.output_dir,
                    upload_to_s3=cfg.output.upload_to_s3,
                    s3_bucket=cfg.output.s3_bucket,
                )
                train_step = TrainingStep(train_config)

            train_artifacts = train_step.execute(primary_lmdb)
            test_metrics = train_artifacts.outputs.get("test_metrics", {})

            model_path = Path(cfg.output.output_dir) / "final_model.pt"
            if not model_path.exists():
                model_path = Path(cfg.output.output_dir) / "best_model.pt"

            return PipelineResult(
                success=True,
                model_path=str(model_path) if model_path.exists() else None,
                annotated_dataset_id=annot_artifacts.outputs.get("annotated_dataset_id"),
                lmdb_dataset_ids={
                    ps: r.outputs.get("lmdb_dataset_id") for ps, r in lmdb_results.items()
                },
                training_metrics=test_metrics if isinstance(test_metrics, dict) else {},
            )

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return PipelineResult(success=False, error=str(e))

    def _run_locally_multi_tissue(self) -> PipelineResult:
        """Run multi-tissue pipeline locally."""
        from dapidl.pipeline.base import StepArtifacts
        from dapidl.pipeline.steps import (
            AnnotationStep,
            DataLoaderStep,
            SegmentationStep,
        )
        from dapidl.pipeline.steps.annotation import AnnotationStepConfig
        from dapidl.pipeline.steps.data_loader import DataLoaderConfig
        from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep
        from dapidl.pipeline.steps.segmentation import SegmentationStepConfig
        from dapidl.pipeline.steps.universal_training import (
            TissueDatasetSpec,
            UniversalDAPITrainingStep,
            UniversalTrainingConfig,
        )

        cfg = self.config

        try:
            results = {}
            dataset_configs = []
            primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]

            # Process each tissue
            for i, tissue_cfg in enumerate(cfg.input.tissues):
                tissue_name = tissue_cfg.tissue
                logger.info("=" * 60)
                logger.info(f"Processing tissue {i+1}/{len(cfg.input.tissues)}: {tissue_name}")
                logger.info("=" * 60)

                # Step 1: Data Loader
                logger.info(f"  Step 1: Data Loader ({tissue_name})")
                data_config = DataLoaderConfig(
                    dataset_id=tissue_cfg.dataset_id,
                    platform=tissue_cfg.platform.value,
                    local_path=tissue_cfg.local_path,
                )
                data_loader = DataLoaderStep(data_config)
                artifacts = data_loader.execute(StepArtifacts())
                results[f"data_loader_{tissue_name}"] = artifacts.outputs

                # Step 2: Segmentation
                logger.info(f"  Step 2: Segmentation ({tissue_name})")
                seg_config = SegmentationStepConfig(
                    segmenter=cfg.segmentation.segmenter.value,
                    diameter=cfg.segmentation.diameter,
                    flow_threshold=cfg.segmentation.flow_threshold,
                    platform=artifacts.outputs.get("platform", tissue_cfg.platform.value),
                )
                segmentation = SegmentationStep(seg_config)
                seg_artifacts = segmentation.execute(artifacts)
                results[f"segmentation_{tissue_name}"] = seg_artifacts.outputs

                # Step 3: Annotation
                logger.info(f"  Step 3: Annotation ({tissue_name})")
                annot_config = AnnotationStepConfig(
                    annotator="celltypist",
                    strategy="consensus",
                    model_names=cfg.annotation.celltypist_models,
                    fine_grained=cfg.annotation.fine_grained,
                )
                annotation = AnnotationStep(annot_config)
                annot_artifacts = annotation.execute(artifacts)
                results[f"annotation_{tissue_name}"] = annot_artifacts.outputs

                # Merge outputs
                merged_outputs = {**seg_artifacts.outputs, **annot_artifacts.outputs}
                artifacts = StepArtifacts(inputs={}, outputs=merged_outputs)

                # Step 4: LMDB Creation
                logger.info(f"  Step 4: LMDB Creation ({tissue_name})")
                lmdb_config = LMDBCreationConfig(
                    patch_size=primary_patch_size,
                    normalization_method=cfg.lmdb.normalization.value,
                    create_clearml_dataset=False,
                )
                lmdb_step = LMDBCreationStep(lmdb_config)
                lmdb_artifacts = lmdb_step.execute(artifacts)
                results[f"lmdb_{tissue_name}"] = lmdb_artifacts.outputs

                # Collect dataset config for universal training
                dataset_path = lmdb_artifacts.outputs.get("lmdb_path")
                if dataset_path:
                    dataset_configs.append(
                        TissueDatasetSpec(
                            path=dataset_path,
                            tissue=tissue_name,
                            platform=tissue_cfg.platform.value,
                            confidence_tier=tissue_cfg.confidence_tier,
                            weight_multiplier=tissue_cfg.weight_multiplier,
                        )
                    )

            # Universal Training
            logger.info("=" * 60)
            logger.info("Universal Training")
            logger.info("=" * 60)

            train_config = UniversalTrainingConfig(
                backbone=cfg.training.backbone.value,
                epochs=cfg.training.epochs,
                batch_size=cfg.training.batch_size,
                learning_rate=cfg.training.learning_rate,
                sampling_strategy=cfg.training.sampling_strategy.value,
                tier1_weight=cfg.training.tier1_weight,
                tier2_weight=cfg.training.tier2_weight,
                tier3_weight=cfg.training.tier3_weight,
                coarse_only_epochs=cfg.training.coarse_only_epochs,
                coarse_medium_epochs=cfg.training.coarse_medium_epochs,
                output_dir=cfg.output.output_dir,
                datasets=dataset_configs,
            )

            training_artifacts = StepArtifacts(inputs={}, outputs={"dataset_configs": dataset_configs})
            training_step = UniversalDAPITrainingStep(train_config)
            training_artifacts = training_step.execute(training_artifacts)
            results["universal_training"] = training_artifacts.outputs

            model_path = training_artifacts.outputs.get("model_path")
            test_metrics = training_artifacts.outputs.get("test_metrics", {})

            logger.info("=" * 60)
            logger.info("Universal Pipeline completed!")
            logger.info(f"  Model: {model_path}")
            logger.info(f"  Test F1: {test_metrics.get('f1_fine', 'N/A')}")
            logger.info("=" * 60)

            return PipelineResult(
                success=True,
                model_path=model_path,
                training_metrics=test_metrics,
            )

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return PipelineResult(success=False, error=str(e))

    def create_base_tasks(self):
        """Create base tasks for each step (required before running pipeline remotely)."""
        from dapidl.pipeline.steps import (
            AnnotationStep,
            DataLoaderStep,
            PatchExtractionStep,
            SegmentationStep,
            TrainingStep,
        )
        from dapidl.pipeline.steps.ensemble_annotation import EnsembleAnnotationStep
        from dapidl.pipeline.steps.hierarchical_training import HierarchicalTrainingStep
        from dapidl.pipeline.steps.lmdb_creation import LMDBCreationStep

        cfg = self.config
        steps = [
            DataLoaderStep(),
            SegmentationStep(),
            AnnotationStep(),
            EnsembleAnnotationStep(),
            PatchExtractionStep(),
            LMDBCreationStep(),
            TrainingStep(),
            HierarchicalTrainingStep(),
        ]

        # Try to add universal training step if available
        try:
            from dapidl.pipeline.steps.universal_training import UniversalDAPITrainingStep

            steps.append(UniversalDAPITrainingStep())
        except ImportError:
            pass

        for step in steps:
            try:
                task = step.create_clearml_task(
                    project=cfg.project,
                    task_name=f"step-{step.name}",
                )
                task.close()
                logger.info(f"Created base task: step-{step.name}")
            except Exception as e:
                logger.warning(f"Failed to create task for {step.name}: {e}")

        logger.info("All base tasks created")

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        if self._pipeline is None:
            return {"status": "not_created"}

        return {
            "pipeline_id": self._pipeline.id,
            "status": self._pipeline.get_status(),
            "mode": "multi-tissue" if self.is_multi_tissue else "single-dataset",
            "steps": {step.name: step.get_status() for step in self._pipeline.get_steps()},
        }

    def get_pipeline_dag(self) -> dict:
        """Return pipeline DAG structure for visualization."""
        cfg = self.config

        dag = {
            "name": cfg.name,
            "version": cfg.version,
            "mode": "multi-tissue" if self.is_multi_tissue else "single-dataset",
            "steps": [],
            "parameters": cfg.to_clearml_parameters(),
        }

        if self.is_multi_tissue:
            # Add per-tissue steps
            for i, tissue_cfg in enumerate(cfg.input.tissues):
                tissue_name = tissue_cfg.tissue
                dag["steps"].extend(
                    [
                        {"name": f"data_loader_{tissue_name}_{i}", "parents": [], "queue": "default"},
                        {
                            "name": f"segmentation_{tissue_name}_{i}",
                            "parents": [f"data_loader_{tissue_name}_{i}"],
                            "queue": "gpu",
                        },
                        {
                            "name": f"annotation_{tissue_name}_{i}",
                            "parents": [f"data_loader_{tissue_name}_{i}"],
                            "queue": "default",
                        },
                        {
                            "name": f"lmdb_{tissue_name}_{i}",
                            "parents": [f"segmentation_{tissue_name}_{i}", f"annotation_{tissue_name}_{i}"],
                            "queue": "default",
                        },
                    ]
                )
            # Universal training depends on all LMDB steps
            lmdb_steps = [f"lmdb_{t.tissue}_{i}" for i, t in enumerate(cfg.input.tissues)]
            dag["steps"].append({"name": "universal_training", "parents": lmdb_steps, "queue": "gpu"})
        else:
            # Single dataset steps
            dag["steps"] = [
                {"name": "data_loader", "parents": [], "queue": "default"},
                {"name": "ensemble_annotation", "parents": ["data_loader"], "queue": "default"},
            ]

            # Add LMDB steps
            for patch_size in cfg.lmdb.patch_sizes:
                dag["steps"].append(
                    {
                        "name": f"lmdb_p{patch_size}",
                        "parents": ["ensemble_annotation"],
                        "queue": "default",
                    }
                )

            # Training
            primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
            dag["steps"].append(
                {"name": "training", "parents": [f"lmdb_p{primary_patch_size}"], "queue": "gpu"}
            )

        return dag


def create_unified_pipeline(config: DAPIDLPipelineConfig) -> UnifiedPipelineController:
    """Factory function to create a configured unified pipeline.

    Args:
        config: Unified pipeline configuration

    Returns:
        Configured pipeline controller
    """
    return UnifiedPipelineController(config)
