"""State-of-the-Art Pipeline Controller for DAPIDL.

This controller uses the unified configuration system and integrates
best practices from comprehensive benchmarking:

State-of-the-Art Settings (Jan 2025 benchmarks):
- Annotation: PopV Ensemble with UNWEIGHTED voting (F1=0.844)
  - 5 CellTypist + SingleR (HPCA + Blueprint) - NOT confidence-weighted
  - Blueprint is CRITICAL for Stromal (+117% F1)
- Training: EfficientNetV2-S, 256px patches, max_weight_ratio=10.0 (F1=0.8481)
- Class imbalance: WeightedRandomSampler + FocalLoss, max_ratio=10.0
- Normalization: Adaptive percentile (essential for cross-platform)

Key Features:
- Full web UI configurability via ClearML parameters
- Each step works standalone
- Proper parameter validation
- State-of-the-art defaults baked in
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.unified_config import (
    AnnotationConfig,
    AnnotationStrategy,
    AugmentationLevel,
    BackboneType,
    CLTargetLevel,
    DAPIDLPipelineConfig,
    ExecutionConfig,
    InputConfig,
    LMDBConfig,
    NormalizationMethod,
    OntologyConfig,
    OutputConfig,
    Platform,
    SegmentationConfig,
    SegmenterType,
    TrainingConfig,
    TrainingMode,
    ValidationConfig,
)


def create_sota_config(
    dataset_id: str | None = None,
    local_path: str | None = None,
    s3_uri: str | None = None,
    platform: str = "auto",
    fine_grained: bool = False,
    epochs: int = 100,
) -> DAPIDLPipelineConfig:
    """Create state-of-the-art configuration based on benchmarking.

    These defaults are derived from comprehensive benchmarking (Jan 2025):
    - HPO Trial 13 achieved F1=0.8481 with these settings
    - PopV Ensemble with unweighted voting: F1=0.844
    - Blueprint reference: +117% Stromal F1

    Args:
        dataset_id: ClearML dataset ID
        local_path: Local path to data
        s3_uri: S3 URI for data
        platform: Platform (auto, xenium, merscope)
        fine_grained: Use fine-grained classification
        epochs: Training epochs

    Returns:
        Optimized DAPIDLPipelineConfig
    """
    return DAPIDLPipelineConfig(
        name="dapidl-sota-pipeline",
        project="DAPIDL/pipelines",
        version="3.0.0",
        # Input configuration
        input=InputConfig(
            dataset_id=dataset_id,
            local_path=local_path,
            s3_uri=s3_uri,
            platform=Platform(platform),
        ),
        # Segmentation: Cellpose with optimized parameters
        segmentation=SegmentationConfig(
            segmenter=SegmenterType.CELLPOSE,
            diameter=40,
            flow_threshold=0.4,
            match_threshold_um=5.0,
        ),
        # Annotation: PopV Ensemble with state-of-the-art settings
        # CRITICAL: Unweighted voting beats confidence-weighted by 15-22%
        # CRITICAL: Blueprint reference adds +117% Stromal F1
        annotation=AnnotationConfig(
            strategy=AnnotationStrategy.ENSEMBLE,
            # Best performing CellTypist models for breast
            celltypist_models=[
                "Cells_Adult_Breast.pkl",
                "Immune_All_High.pkl",
                "Immune_All_Low.pkl",  # Marginal benefit but helps
            ],
            include_singler=True,  # ESSENTIAL
            singler_reference="blueprint",  # CRITICAL: +117% Stromal F1
            include_sctype=False,  # Not yet integrated
            include_popv=False,  # Using CellTypist ensemble instead
            min_agreement=2,  # >=2 methods must agree
            confidence_threshold=0.5,
            use_confidence_weighting=False,  # CRITICAL: Unweighted is better!
            fine_grained=fine_grained,
            extended_consensus=False,
        ),
        # Cell Ontology: Enable for standardization
        ontology=OntologyConfig(
            enabled=True,
            target_level=CLTargetLevel.COARSE if not fine_grained else CLTargetLevel.FINE,
            min_confidence=0.5,
            include_unmapped=False,
            fuzzy_threshold=0.85,
        ),
        # LMDB: Optimized patch extraction
        # HPO Trial 13: 256px patches achieve best F1
        lmdb=LMDBConfig(
            patch_sizes=[128, 256],  # Generate both for flexibility
            primary_patch_size=256,  # Best performance
            normalization=NormalizationMethod.ADAPTIVE,  # Essential for cross-platform
            normalize_physical_size=True,  # Cross-platform compatibility
            target_pixel_size_um=0.2125,  # Xenium standard
            min_confidence=0.0,  # Filter later
            exclude_edge_cells=True,
            edge_margin_px=64,
            skip_if_exists=True,
        ),
        # Training: State-of-the-art settings from HPO
        training=TrainingConfig(
            # Architecture: EfficientNetV2-S wins
            backbone=BackboneType.EFFICIENTNETV2_S,
            pretrained=True,
            dropout=0.2,  # HPO Trial 13 optimal
            # Training mode: Hierarchical with curriculum
            mode=TrainingMode.HIERARCHICAL,
            # Hyperparameters from HPO Trial 13
            epochs=epochs,
            batch_size=64,  # HPO optimal
            learning_rate=5e-4,  # HPO Trial 13: 0.000512
            weight_decay=1e-5,
            # Class balancing: CRITICAL for preventing mode collapse
            use_weighted_loss=True,
            use_weighted_sampler=True,
            max_weight_ratio=10.0,  # CRITICAL: Prevents mode collapse
            # Data splits
            val_split=0.15,
            test_split=0.15,
            stratified=True,
            # Data loading
            num_workers=4,
            use_dali=False,  # Standard PyTorch dataloader
            # Augmentation
            augmentation=AugmentationLevel.STANDARD,
            cross_platform=False,
            # Early stopping
            patience=15,
            min_delta=0.001,
            # Curriculum learning phases
            coarse_only_epochs=20,
            coarse_medium_epochs=50,
            warmup_epochs=5,
        ),
        # Output: S3 and ClearML registration
        output=OutputConfig(
            output_dir="./pipeline_output",
            save_best=True,
            save_final=True,
            upload_to_s3=True,
            s3_bucket="dapidl",
            s3_endpoint="https://s3.eu-central-2.idrivee2.com",
            s3_region="eu-central-2",
            s3_models_prefix="models",
            register_datasets=True,
            register_models=True,
            link_to_parent=True,
        ),
        # Validation: Enable for quality assurance
        validation=ValidationConfig(
            enabled=True,
            run_leiden_check=True,
            run_dapi_check=True,
            run_consensus_check=True,
            min_ari_threshold=0.5,
            min_agreement_threshold=0.5,
        ),
        # Execution
        execution=ExecutionConfig(
            execute_remotely=True,
            default_queue="default",
            gpu_queue="gpu",
            cache_data_steps=True,
            cache_training=False,  # Re-run training for different seeds
        ),
    )


class SOTAPipelineController:
    """State-of-the-Art Pipeline Controller for DAPIDL.

    This controller:
    1. Uses the unified configuration system
    2. Exposes ALL parameters to ClearML web UI with proper grouping
    3. Supports standalone step execution
    4. Has state-of-the-art defaults baked in

    Example:
        ```python
        # Create with defaults
        controller = SOTAPipelineController()

        # Or with custom config
        config = create_sota_config(
            dataset_id="abc123",
            platform="xenium",
            epochs=100,
        )
        controller = SOTAPipelineController(config)

        # Run pipeline
        controller.run()

        # Or run locally
        results = controller.run_locally()
        ```
    """

    def __init__(self, config: DAPIDLPipelineConfig | None = None):
        """Initialize with unified configuration.

        Args:
            config: Pipeline configuration. Uses SOTA defaults if None.
        """
        self.config = config or create_sota_config()
        self._pipeline = None

    def create_pipeline(self):
        """Create ClearML PipelineController with full web UI support.

        All parameters are exposed to the ClearML web interface
        using the unified configuration's to_clearml_parameters() method.
        """
        from clearml import PipelineController

        cfg = self.config

        self._pipeline = PipelineController(
            name=cfg.name,
            project=cfg.project,
            version=cfg.version,
            add_pipeline_tags=True,
        )

        # Set default execution queue
        if cfg.execution.execute_remotely:
            self._pipeline.set_default_execution_queue(cfg.execution.default_queue)

        # Add ALL parameters to ClearML web UI using unified config
        # This is the key integration - parameters are organized by group
        clearml_params = cfg.to_clearml_parameters()
        for param_name, param_value in clearml_params.items():
            # Extract group and field from "group/field" format
            parts = param_name.split("/", 1)
            if len(parts) == 2:
                group, field = parts
                description = self._get_parameter_description(group, field)
            else:
                description = ""

            self._pipeline.add_parameter(
                name=param_name,
                default=param_value,
                description=description,
            )

        # Add pipeline steps
        self._add_data_loader_step()
        self._add_segmentation_step()
        self._add_annotation_step()
        self._add_lmdb_creation_step()
        self._add_training_step()

        # Optional steps
        if cfg.validation.enabled:
            self._add_validation_step()

        logger.info(f"Created SOTA pipeline: {cfg.name} v{cfg.version}")
        logger.info(f"  Parameters: {len(clearml_params)} configurable via web UI")
        return self._pipeline

    def _get_parameter_description(self, group: str, field: str) -> str:
        """Get description for a parameter from the unified config schema."""
        config_class_map = {
            "input": InputConfig,
            "segmentation": SegmentationConfig,
            "annotation": AnnotationConfig,
            "ontology": OntologyConfig,
            "lmdb": LMDBConfig,
            "training": TrainingConfig,
            "output": OutputConfig,
            "validation": ValidationConfig,
            "execution": ExecutionConfig,
        }

        config_class = config_class_map.get(group)
        if config_class and hasattr(config_class, "model_fields"):
            field_info = config_class.model_fields.get(field)
            if field_info and field_info.description:
                return field_info.description
        return ""

    def _add_data_loader_step(self):
        """Add data loader step."""
        cfg = self.config

        self._pipeline.add_step(
            name="data_loader",
            base_task_project=cfg.project,
            base_task_name="sota-step-data_loader",
            parameter_override={
                "step_config/dataset_id": cfg.input.dataset_id or "",
                "step_config/dataset_name": cfg.input.dataset_name or "",
                "step_config/dataset_project": cfg.input.dataset_project,
                "step_config/local_path": cfg.input.local_path or "",
                "step_config/s3_uri": cfg.input.s3_uri or "",
                "step_config/platform": cfg.input.platform.value,
            },
            execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_data_steps,
        )

    def _add_segmentation_step(self):
        """Add nucleus segmentation step."""
        cfg = self.config

        self._pipeline.add_step(
            name="segmentation",
            parents=["data_loader"],
            base_task_project=cfg.project,
            base_task_name="sota-step-segmentation",
            parameter_override={
                "step_config/segmenter": cfg.segmentation.segmenter.value,
                "step_config/diameter": cfg.segmentation.diameter,
                "step_config/flow_threshold": cfg.segmentation.flow_threshold,
                "step_config/match_threshold_um": cfg.segmentation.match_threshold_um,
                "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                "step_config/platform": "${data_loader.artifacts.platform.url}",
            },
            execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_data_steps,
        )

    def _add_annotation_step(self):
        """Add cell type annotation step with SOTA settings."""
        cfg = self.config

        self._pipeline.add_step(
            name="ensemble_annotation",
            parents=["data_loader"],
            base_task_project=cfg.project,
            base_task_name="sota-step-ensemble_annotation",
            parameter_override={
                # Annotation strategy
                "step_config/strategy": cfg.annotation.strategy.value,
                "step_config/celltypist_models": ",".join(cfg.annotation.celltypist_models),
                "step_config/include_singler": str(cfg.annotation.include_singler),
                "step_config/singler_reference": cfg.annotation.singler_reference,
                "step_config/include_sctype": str(cfg.annotation.include_sctype),
                # Consensus settings - CRITICAL: use_confidence_weighting=False
                "step_config/min_agreement": cfg.annotation.min_agreement,
                "step_config/confidence_threshold": cfg.annotation.confidence_threshold,
                "step_config/use_confidence_weighting": str(cfg.annotation.use_confidence_weighting),
                "step_config/fine_grained": str(cfg.annotation.fine_grained),
                # Ground truth (if applicable)
                "step_config/ground_truth_file": cfg.annotation.ground_truth_file or "",
                # Cell Ontology standardization
                "step_config/use_cell_ontology": str(cfg.ontology.enabled),
                "step_config/cl_target_level": cfg.ontology.target_level.value,
                "step_config/cl_min_confidence": cfg.ontology.min_confidence,
                # Data from parent
                "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                "step_config/platform": "${data_loader.artifacts.platform.url}",
                "step_config/cells_parquet": "${data_loader.artifacts.cells_parquet.url}",
                "step_config/expression_path": "${data_loader.artifacts.expression_path.url}",
            },
            execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_data_steps,
        )

    def _add_lmdb_creation_step(self):
        """Add LMDB dataset creation step."""
        cfg = self.config

        # Create one step per patch size for parallel execution
        patch_sizes = cfg.lmdb.patch_sizes
        for patch_size in patch_sizes:
            step_name = f"lmdb_creation_p{patch_size}"

            self._pipeline.add_step(
                name=step_name,
                parents=["segmentation", "ensemble_annotation"],
                base_task_project=cfg.project,
                base_task_name="sota-step-lmdb_creation",
                parameter_override={
                    "step_config/patch_size": patch_size,
                    "step_config/normalization_method": cfg.lmdb.normalization.value,
                    "step_config/normalize_physical_size": str(cfg.lmdb.normalize_physical_size),
                    "step_config/target_pixel_size_um": cfg.lmdb.target_pixel_size_um,
                    "step_config/exclude_edge_cells": str(cfg.lmdb.exclude_edge_cells),
                    "step_config/edge_margin_px": cfg.lmdb.edge_margin_px,
                    "step_config/skip_if_exists": str(cfg.lmdb.skip_if_exists),
                    # Data from parents
                    "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                    "step_config/platform": "${data_loader.artifacts.platform.url}",
                    "step_config/centroids_parquet": "${segmentation.artifacts.centroids_parquet.url}",
                    "step_config/annotations_parquet": "${ensemble_annotation.artifacts.annotations_parquet.url}",
                },
                execution_queue=cfg.execution.default_queue if cfg.execution.execute_remotely else None,
                cache_executed_step=cfg.execution.cache_data_steps,
            )

    def _add_training_step(self):
        """Add model training step with SOTA settings."""
        cfg = self.config

        # Determine primary LMDB step
        primary_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
        primary_lmdb_step = f"lmdb_creation_p{primary_size}"

        self._pipeline.add_step(
            name="training",
            parents=[primary_lmdb_step],
            base_task_project=cfg.project,
            base_task_name="sota-step-training",
            parameter_override={
                # Architecture
                "step_config/backbone": cfg.training.backbone.value,
                "step_config/pretrained": str(cfg.training.pretrained),
                "step_config/dropout": cfg.training.dropout,
                # Training mode
                "step_config/mode": cfg.training.mode.value,
                # Hyperparameters (from HPO Trial 13)
                "step_config/epochs": cfg.training.epochs,
                "step_config/batch_size": cfg.training.batch_size,
                "step_config/learning_rate": cfg.training.learning_rate,
                "step_config/weight_decay": cfg.training.weight_decay,
                # Class balancing - CRITICAL
                "step_config/use_weighted_loss": str(cfg.training.use_weighted_loss),
                "step_config/use_weighted_sampler": str(cfg.training.use_weighted_sampler),
                "step_config/max_weight_ratio": cfg.training.max_weight_ratio,
                # Data splits
                "step_config/val_split": cfg.training.val_split,
                "step_config/test_split": cfg.training.test_split,
                "step_config/stratified": str(cfg.training.stratified),
                # Data loading
                "step_config/num_workers": cfg.training.num_workers,
                # Augmentation
                "step_config/augmentation": cfg.training.augmentation.value,
                # Early stopping
                "step_config/patience": cfg.training.patience,
                "step_config/min_delta": cfg.training.min_delta,
                # Curriculum learning
                "step_config/coarse_only_epochs": cfg.training.coarse_only_epochs,
                "step_config/coarse_medium_epochs": cfg.training.coarse_medium_epochs,
                "step_config/warmup_epochs": cfg.training.warmup_epochs,
                # Dataset from LMDB step
                "step_config/dataset_path": f"${{{primary_lmdb_step}.artifacts.lmdb_path.url}}",
                "step_config/num_classes": f"${{{primary_lmdb_step}.artifacts.num_classes.url}}",
                "step_config/class_names": f"${{{primary_lmdb_step}.artifacts.class_names.url}}",
            },
            execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=cfg.execution.cache_training,
        )

    def _add_validation_step(self):
        """Add cross-modal validation step."""
        cfg = self.config

        self._pipeline.add_step(
            name="cross_validation",
            parents=["training"],
            base_task_project=cfg.project,
            base_task_name="sota-step-cross_validation",
            parameter_override={
                "step_config/run_leiden_check": str(cfg.validation.run_leiden_check),
                "step_config/run_dapi_check": str(cfg.validation.run_dapi_check),
                "step_config/run_consensus_check": str(cfg.validation.run_consensus_check),
                "step_config/min_ari_threshold": cfg.validation.min_ari_threshold,
                "step_config/min_agreement_threshold": cfg.validation.min_agreement_threshold,
                # Data from parent steps
                "step_config/model_path": "${training.artifacts.model_path.url}",
            },
            execution_queue=cfg.execution.gpu_queue if cfg.execution.execute_remotely else None,
            cache_executed_step=True,
        )

    def run(self, wait: bool = True, run_locally: bool = False) -> str:
        """Run the pipeline on ClearML.

        Args:
            wait: Wait for completion
            run_locally: If True, run controller locally but queue steps to agents.
                        If False, queue the entire pipeline to an agent (default).

        Returns:
            Pipeline run ID
        """
        if self._pipeline is None:
            self.create_pipeline()

        logger.info("Starting SOTA pipeline execution...")

        if run_locally:
            # Run controller locally, queue steps to agents
            self._pipeline.start_locally(run_pipeline_steps_locally=False)
        else:
            # Queue entire pipeline to agent (recommended for fully remote execution)
            self._pipeline.start(queue="default")

        if wait:
            self._pipeline.wait()
            logger.info("Pipeline completed")

        return self._pipeline.id

    def run_locally(self) -> dict[str, Any]:
        """Run pipeline steps locally without ClearML agents.

        Each step is executed in sequence, passing outputs between steps.

        Returns:
            Dict containing outputs from all steps
        """
        from dapidl.pipeline.base import StepArtifacts

        cfg = self.config
        results = {}

        logger.info("=" * 60)
        logger.info("SOTA Pipeline - Local Execution")
        logger.info("=" * 60)
        logger.info(f"Configuration: {cfg.name} v{cfg.version}")

        # Import steps
        from dapidl.pipeline.steps import (
            DataLoaderStep,
            SegmentationStep,
        )
        from dapidl.pipeline.steps.data_loader import DataLoaderConfig
        from dapidl.pipeline.steps.segmentation import SegmentationStepConfig
        from dapidl.pipeline.steps.ensemble_annotation import (
            EnsembleAnnotationConfig,
            EnsembleAnnotationStep,
        )
        from dapidl.pipeline.steps.lmdb_creation import (
            LMDBCreationConfig,
            LMDBCreationStep,
        )
        from dapidl.pipeline.steps.training import (
            TrainingStep,
            TrainingStepConfig,
        )

        # Step 1: Data Loader
        logger.info("\n" + "=" * 50)
        logger.info("Step 1: Data Loader")
        logger.info("=" * 50)

        data_config = DataLoaderConfig(
            dataset_id=cfg.input.dataset_id,
            dataset_name=cfg.input.dataset_name,
            dataset_project=cfg.input.dataset_project,
            local_path=cfg.input.local_path,
            platform=cfg.input.platform.value,
        )
        data_loader = DataLoaderStep(data_config)
        artifacts = data_loader.execute(StepArtifacts(inputs={}, outputs={}))
        results["data_loader"] = artifacts.outputs
        logger.info(f"  Outputs: {list(artifacts.outputs.keys())}")

        # Step 2: Segmentation
        logger.info("\n" + "=" * 50)
        logger.info("Step 2: Segmentation")
        logger.info("=" * 50)

        seg_config = SegmentationStepConfig(
            segmenter=cfg.segmentation.segmenter.value,
            diameter=cfg.segmentation.diameter,
            flow_threshold=cfg.segmentation.flow_threshold,
            match_threshold_um=cfg.segmentation.match_threshold_um,
            platform=artifacts.outputs.get("platform", cfg.input.platform.value),
        )
        segmentation = SegmentationStep(seg_config)
        seg_artifacts = segmentation.execute(artifacts)
        results["segmentation"] = seg_artifacts.outputs
        logger.info(f"  Outputs: {list(seg_artifacts.outputs.keys())}")

        # Step 3: Ensemble Annotation (SOTA settings)
        logger.info("\n" + "=" * 50)
        logger.info("Step 3: Ensemble Annotation (SOTA)")
        logger.info("=" * 50)
        logger.info(f"  Models: {cfg.annotation.celltypist_models}")
        logger.info(f"  SingleR: {cfg.annotation.include_singler} ({cfg.annotation.singler_reference})")
        logger.info(f"  Confidence weighting: {cfg.annotation.use_confidence_weighting}")

        annot_config = EnsembleAnnotationConfig(
            celltypist_models=cfg.annotation.celltypist_models,
            include_singler=cfg.annotation.include_singler,
            singler_reference=cfg.annotation.singler_reference,
            include_sctype=cfg.annotation.include_sctype,
            min_agreement=cfg.annotation.min_agreement,
            confidence_threshold=cfg.annotation.confidence_threshold,
            use_confidence_weighting=cfg.annotation.use_confidence_weighting,
            fine_grained=cfg.annotation.fine_grained,
            skip_if_exists=True,
            create_derived_dataset=False,  # Local mode
        )

        # Use data_loader outputs
        data_artifacts = StepArtifacts(inputs={}, outputs=results["data_loader"])
        annotation = EnsembleAnnotationStep(annot_config)
        annot_artifacts = annotation.execute(data_artifacts)
        results["ensemble_annotation"] = annot_artifacts.outputs
        logger.info(f"  Outputs: {list(annot_artifacts.outputs.keys())}")

        # Merge outputs
        merged_outputs = {**seg_artifacts.outputs, **annot_artifacts.outputs}
        artifacts = StepArtifacts(inputs={}, outputs=merged_outputs)

        # Step 4: LMDB Creation
        primary_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
        logger.info("\n" + "=" * 50)
        logger.info(f"Step 4: LMDB Creation (patch_size={primary_size})")
        logger.info("=" * 50)

        lmdb_config = LMDBCreationConfig(
            patch_size=primary_size,
            normalization_method=cfg.lmdb.normalization.value,
            normalize_physical_size=cfg.lmdb.normalize_physical_size,
            exclude_edge_cells=cfg.lmdb.exclude_edge_cells,
            edge_margin_px=cfg.lmdb.edge_margin_px,
            skip_if_exists=cfg.lmdb.skip_if_exists,
            create_clearml_dataset=False,  # Local mode
        )
        lmdb_step = LMDBCreationStep(lmdb_config)
        lmdb_artifacts = lmdb_step.execute(artifacts)
        results["lmdb_creation"] = lmdb_artifacts.outputs
        logger.info(f"  Outputs: {list(lmdb_artifacts.outputs.keys())}")

        # Step 5: Training (SOTA settings)
        logger.info("\n" + "=" * 50)
        logger.info("Step 5: Training (SOTA)")
        logger.info("=" * 50)
        logger.info(f"  Backbone: {cfg.training.backbone.value}")
        logger.info(f"  Epochs: {cfg.training.epochs}")
        logger.info(f"  max_weight_ratio: {cfg.training.max_weight_ratio}")

        train_config = TrainingStepConfig(
            backbone=cfg.training.backbone.value,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            use_weighted_loss=cfg.training.use_weighted_loss,
            use_weighted_sampler=cfg.training.use_weighted_sampler,
            max_weight_ratio=cfg.training.max_weight_ratio,
            patience=cfg.training.patience,
        )
        training = TrainingStep(train_config)
        train_artifacts = training.execute(lmdb_artifacts)
        results["training"] = train_artifacts.outputs
        logger.info(f"  Outputs: {list(train_artifacts.outputs.keys())}")

        # Step 6: Validation (optional)
        if cfg.validation.enabled:
            logger.info("\n" + "=" * 50)
            logger.info("Step 6: Cross-Modal Validation")
            logger.info("=" * 50)

            from dapidl.pipeline.steps import CrossValidationStep
            from dapidl.pipeline.steps.cross_validation import CrossValidationConfig

            val_outputs = {
                **results["data_loader"],
                **results["lmdb_creation"],
                **results["ensemble_annotation"],
                **results["training"],
            }
            val_artifacts = StepArtifacts(inputs={}, outputs=val_outputs)

            val_config = CrossValidationConfig(
                run_leiden_check=cfg.validation.run_leiden_check,
                run_dapi_check=cfg.validation.run_dapi_check,
                run_consensus_check=cfg.validation.run_consensus_check,
                min_ari_threshold=cfg.validation.min_ari_threshold,
                min_agreement_threshold=cfg.validation.min_agreement_threshold,
            )
            validation = CrossValidationStep(val_config)
            val_result = validation.execute(val_artifacts)
            results["cross_validation"] = val_result.outputs
            logger.info(f"  Outputs: {list(val_result.outputs.keys())}")

        logger.info("\n" + "=" * 60)
        logger.info("SOTA Pipeline completed successfully!")
        logger.info("=" * 60)

        return results

    def create_base_tasks(self):
        """Create base ClearML tasks for each step.

        Must be called once before running pipeline remotely.
        """
        from dapidl.pipeline.steps import (
            DataLoaderStep,
            SegmentationStep,
            CrossValidationStep,
            TrainingStep,
        )
        from dapidl.pipeline.steps.ensemble_annotation import EnsembleAnnotationStep
        from dapidl.pipeline.steps.lmdb_creation import LMDBCreationStep

        cfg = self.config

        # Define steps with their SOTA-specific names
        steps = [
            ("sota-step-data_loader", DataLoaderStep()),
            ("sota-step-segmentation", SegmentationStep()),
            ("sota-step-ensemble_annotation", EnsembleAnnotationStep()),
            ("sota-step-lmdb_creation", LMDBCreationStep()),
            ("sota-step-training", TrainingStep()),
            ("sota-step-cross_validation", CrossValidationStep()),
        ]

        for task_name, step in steps:
            task = step.create_clearml_task(
                project=cfg.project,
                task_name=task_name,
            )
            task.close()
            logger.info(f"Created base task: {task_name}")

        logger.info("All SOTA base tasks created")


def create_sota_pipeline(
    dataset_id: str | None = None,
    local_path: str | None = None,
    s3_uri: str | None = None,
    platform: str = "auto",
    fine_grained: bool = False,
    epochs: int = 100,
) -> SOTAPipelineController:
    """Factory function to create a state-of-the-art pipeline.

    Args:
        dataset_id: ClearML dataset ID
        local_path: Local path to data
        s3_uri: S3 URI
        platform: Platform (auto, xenium, merscope)
        fine_grained: Use fine-grained classification
        epochs: Training epochs

    Returns:
        Configured SOTAPipelineController
    """
    config = create_sota_config(
        dataset_id=dataset_id,
        local_path=local_path,
        s3_uri=s3_uri,
        platform=platform,
        fine_grained=fine_grained,
        epochs=epochs,
    )
    return SOTAPipelineController(config)
