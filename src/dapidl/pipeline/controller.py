"""Pipeline Controller for DAPIDL.

Orchestrates the full spatial transcriptomics processing pipeline using
ClearML PipelineController. Supports both local execution and remote
execution on ClearML agents.

Pipeline Flow:
    DataLoader → Segmentation ─┐
         │                      │
         └─→ Annotation ────────┼→ PatchExtraction → Training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class PipelineConfig:
    """Configuration for the full DAPIDL pipeline."""

    # Pipeline metadata
    name: str = "dapidl-pipeline"
    project: str = "DAPIDL/pipelines"
    version: str = "1.0.0"

    # Data source
    dataset_id: str | None = None
    dataset_name: str | None = None
    dataset_project: str = "DAPIDL/datasets"
    local_path: str | None = None

    # Platform
    platform: str = "auto"  # "auto", "xenium", "merscope"

    # Segmentation
    segmenter: str = "cellpose"  # "cellpose", "native"
    diameter: int = 40
    flow_threshold: float = 0.4
    match_threshold_um: float = 5.0

    # Annotation
    annotator: str = "celltypist"  # "celltypist", "ground_truth", "popv"
    annotation_strategy: str = "consensus"
    model_names: list[str] = field(
        default_factory=lambda: ["Immune_All_High.pkl", "Cells_Adult_Breast.pkl"]
    )
    confidence_threshold: float = 0.5
    ground_truth_file: str | None = None
    extended_consensus: bool = False  # Use 6 CellTypist models instead of 2

    # Patch extraction
    patch_size: int = 128
    output_format: str = "lmdb"
    normalization: str = "adaptive"

    # Training
    backbone: str = "efficientnetv2_rw_s"
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 3e-4
    fine_grained: bool = False

    # Cross-modal validation (optional step after training)
    run_validation: bool = False  # Enable cross-modal validation step
    validation_leiden: bool = True
    validation_dapi: bool = True
    validation_consensus: bool = True
    min_ari_threshold: float = 0.5
    min_agreement_threshold: float = 0.5

    # Ground truth comparison (optional, requires known cell type labels)
    run_ground_truth_comparison: bool = False  # Compare CellTypist vs ground truth
    gt_comparison_file: str | None = None  # Path to ground truth file for comparison
    gt_comparison_sheet: str | None = None  # Excel sheet name (optional)
    gt_comparison_cell_id_col: str = "Barcode"  # Cell ID column in ground truth
    gt_comparison_label_col: str = "Cluster"  # Label column in ground truth

    # Cross-platform transfer testing (optional step after training)
    run_transfer_test: bool = False  # Enable cross-platform transfer testing
    transfer_target_dataset_id: str | None = None  # Target platform dataset
    transfer_target_local_path: str | None = None  # Or local path
    transfer_target_platform: str = "auto"  # "xenium", "merscope", or "auto"

    # Documentation (optional step at end)
    run_documentation: bool = False  # Enable documentation generation
    obsidian_vault_path: str | None = None  # Path to Obsidian vault
    obsidian_folder: str = "DAPIDL"  # Subfolder within vault
    doc_template: str = "default"  # "default", "minimal", "detailed"

    # Execution
    execute_remotely: bool = True
    default_queue: str = "default"
    gpu_queue: str = "gpu"

    # Caching - ClearML will skip steps with unchanged inputs/params
    cache_training: bool = False  # Disabled by default (random seed variance)


class DAPIDLPipelineController:
    """Orchestrates the DAPIDL pipeline using ClearML.

    Creates a multi-step pipeline that processes spatial transcriptomics
    data through segmentation, annotation, patch extraction, and training.

    Example:
        ```python
        config = PipelineConfig(
            dataset_id="abc123",
            segmenter="cellpose",
            annotator="celltypist",
            epochs=50,
        )
        controller = DAPIDLPipelineController(config)
        controller.run()
        ```
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline controller.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._pipeline = None
        self._step_tasks = {}

    def create_pipeline(self):
        """Create ClearML PipelineController with all steps."""
        from clearml import PipelineController

        cfg = self.config

        self._pipeline = PipelineController(
            name=cfg.name,
            project=cfg.project,
            version=cfg.version,
            add_pipeline_tags=True,
        )

        # Set default execution queue for all steps
        if cfg.execute_remotely:
            self._pipeline.set_default_execution_queue(cfg.default_queue)

        # Add pipeline-level parameters
        self._pipeline.add_parameter(
            name="dataset_id",
            default=cfg.dataset_id or "",
            description="ClearML Dataset ID for input data",
        )
        self._pipeline.add_parameter(
            name="platform",
            default=cfg.platform,
            description="Platform: auto, xenium, or merscope",
        )
        self._pipeline.add_parameter(
            name="segmenter",
            default=cfg.segmenter,
            description="Segmentation method: cellpose or native",
        )
        self._pipeline.add_parameter(
            name="annotator",
            default=cfg.annotator,
            description="Annotation method: celltypist, ground_truth, or popv",
        )
        self._pipeline.add_parameter(
            name="patch_size",
            default=cfg.patch_size,
            description="Patch size in pixels: 32, 64, 128, or 256",
        )
        self._pipeline.add_parameter(
            name="epochs",
            default=cfg.epochs,
            description="Number of training epochs",
        )

        # Step 1: Data Loader
        # Note: Use direct values instead of ${pipeline.xxx} substitution
        # because start_locally() doesn't properly resolve pipeline parameters
        self._pipeline.add_step(
            name="data_loader",
            base_task_project=cfg.project,
            base_task_name="step-data_loader",
            parameter_override={
                "step_config/dataset_id": cfg.dataset_id or "",
                "step_config/platform": cfg.platform,
                "step_config/local_path": cfg.local_path or "",
            },
            execution_queue=cfg.default_queue if cfg.execute_remotely else None,
            cache_executed_step=True,  # Skip if inputs/params unchanged
        )

        # Step 2: Segmentation (depends on data_loader)
        # Pass parent outputs via parameter_override using ${step.artifacts.name.url} syntax
        self._pipeline.add_step(
            name="segmentation",
            parents=["data_loader"],
            base_task_project=cfg.project,
            base_task_name="step-segmentation",
            parameter_override={
                "step_config/segmenter": cfg.segmenter,
                "step_config/diameter": cfg.diameter,
                "step_config/flow_threshold": cfg.flow_threshold,
                "step_config/match_threshold_um": cfg.match_threshold_um,
                # Pass outputs from data_loader
                "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                "step_config/platform": "${data_loader.artifacts.platform.url}",
            },
            execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
            cache_executed_step=True,  # Skip if inputs/params unchanged
        )

        # Step 3: Annotation (depends on data_loader)
        self._pipeline.add_step(
            name="annotation",
            parents=["data_loader"],
            base_task_project=cfg.project,
            base_task_name="step-annotation",
            parameter_override={
                "step_config/annotator": cfg.annotator,
                "step_config/strategy": cfg.annotation_strategy,
                "step_config/model_names": ",".join(cfg.model_names),
                "step_config/confidence_threshold": cfg.confidence_threshold,
                "step_config/ground_truth_file": cfg.ground_truth_file or "",
                "step_config/fine_grained": cfg.fine_grained,
                # Pass outputs from data_loader
                "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                "step_config/platform": "${data_loader.artifacts.platform.url}",
                "step_config/cells_parquet": "${data_loader.artifacts.cells_parquet.url}",
                "step_config/expression_path": "${data_loader.artifacts.expression_path.url}",
            },
            execution_queue=cfg.default_queue if cfg.execute_remotely else None,
            cache_executed_step=True,  # Skip if inputs/params unchanged
        )

        # Step 4: Patch Extraction (depends on segmentation + annotation)
        self._pipeline.add_step(
            name="patch_extraction",
            parents=["segmentation", "annotation"],
            base_task_project=cfg.project,
            base_task_name="step-patch_extraction",
            parameter_override={
                "step_config/patch_size": cfg.patch_size,
                "step_config/output_format": cfg.output_format,
                "step_config/normalization_method": cfg.normalization,
                # Pass outputs from parent steps
                "step_config/data_path": "${data_loader.artifacts.data_path.url}",
                "step_config/platform": "${data_loader.artifacts.platform.url}",
                "step_config/centroids_parquet": "${segmentation.artifacts.centroids_parquet.url}",
                "step_config/annotations_parquet": "${annotation.artifacts.annotations_parquet.url}",
            },
            execution_queue=cfg.default_queue if cfg.execute_remotely else None,
            cache_executed_step=True,  # Skip if inputs/params unchanged
        )

        # Step 5: Training (depends on patch_extraction)
        # Note: Training caching disabled by default - users often want to
        # re-run training with same params but different random seeds
        self._pipeline.add_step(
            name="training",
            parents=["patch_extraction"],
            base_task_project=cfg.project,
            base_task_name="step-training",
            parameter_override={
                "step_config/backbone": cfg.backbone,
                "step_config/epochs": cfg.epochs,
                "step_config/batch_size": cfg.batch_size,
                "step_config/learning_rate": cfg.learning_rate,
                # Pass outputs from patch_extraction
                # Note: patches_path is the LMDB dataset, num_classes/class_names/index_to_class from annotations
                "step_config/dataset_path": "${patch_extraction.artifacts.patches_path.url}",
                "step_config/num_classes": "${patch_extraction.artifacts.num_classes.url}",
                "step_config/class_names": "${patch_extraction.artifacts.class_names.url}",
                "step_config/index_to_class": "${patch_extraction.artifacts.index_to_class.url}",
            },
            execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
            cache_executed_step=cfg.cache_training,  # Configurable
        )

        # Step 6 (Optional): Cross-modal Validation (depends on training)
        if cfg.run_validation:
            self._pipeline.add_step(
                name="cross_validation",
                parents=["training"],
                base_task_project=cfg.project,
                base_task_name="step-cross_validation",
                parameter_override={
                    "step_config/run_leiden_check": cfg.validation_leiden,
                    "step_config/run_dapi_check": cfg.validation_dapi,
                    "step_config/run_consensus_check": cfg.validation_consensus,
                    "step_config/min_ari_threshold": cfg.min_ari_threshold,
                    "step_config/min_agreement_threshold": cfg.min_agreement_threshold,
                    # Pass outputs from parent steps
                    "step_config/model_path": "${training.artifacts.model_path.url}",
                    "step_config/patches_path": "${patch_extraction.artifacts.dataset_path.url}",
                    "step_config/annotations_parquet": "${annotation.artifacts.annotations_parquet.url}",
                    "step_config/class_mapping": "${patch_extraction.artifacts.class_names.url}",
                    "step_config/expression_path": "${data_loader.artifacts.expression_path.url}",
                },
                execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )
            logger.info("Added cross-modal validation step")

        # Step 7 (Optional): Cross-Platform Transfer Testing
        if cfg.run_transfer_test:
            # Determine parent step
            transfer_parent = "cross_validation" if cfg.run_validation else "training"
            self._pipeline.add_step(
                name="cross_platform_transfer",
                parents=[transfer_parent],
                base_task_project=cfg.project,
                base_task_name="step-cross_platform_transfer",
                parameter_override={
                    "step_config/target_dataset_id": cfg.transfer_target_dataset_id or "",
                    "step_config/target_local_path": cfg.transfer_target_local_path or "",
                    "step_config/target_platform": cfg.transfer_target_platform,
                    "step_config/source_platform": cfg.platform,
                    "step_config/patch_size": cfg.patch_size,
                    "step_config/normalize_physical_size": True,
                    # Pass outputs from training
                    "step_config/model_path": "${training.artifacts.model_path.url}",
                },
                execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )
            logger.info("Added cross-platform transfer step")

        # Step 8 (Optional): Documentation Generation
        if cfg.run_documentation:
            # Determine parent step (last step in pipeline)
            if cfg.run_transfer_test:
                doc_parent = "cross_platform_transfer"
            elif cfg.run_validation:
                doc_parent = "cross_validation"
            else:
                doc_parent = "training"

            self._pipeline.add_step(
                name="documentation",
                parents=[doc_parent],
                base_task_project=cfg.project,
                base_task_name="step-documentation",
                parameter_override={
                    "step_config/obsidian_vault_path": cfg.obsidian_vault_path or "",
                    "step_config/obsidian_folder": cfg.obsidian_folder,
                    "step_config/template": cfg.doc_template,
                    "step_config/experiment_name": cfg.name,
                },
                execution_queue=cfg.default_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )
            logger.info("Added documentation step")

        logger.info(f"Created pipeline: {cfg.name} v{cfg.version}")
        return self._pipeline

    def run(self, wait: bool = True) -> str:
        """Run the pipeline.

        The pipeline controller runs locally while step tasks are executed
        on ClearML agents. This avoids entry point issues with uv-managed
        CLI tools that don't exist in the cloned repository.

        Args:
            wait: If True, wait for pipeline completion

        Returns:
            Pipeline run ID
        """
        if self._pipeline is None:
            self.create_pipeline()

        logger.info("Starting pipeline execution...")
        # Use start_locally() to run controller here, but steps go to agents
        # This avoids the .venv/bin/dapidl entry point issue on agents
        self._pipeline.start_locally(run_pipeline_steps_locally=False)

        if wait:
            self._pipeline.wait()
            logger.info("Pipeline completed")

        return self._pipeline.id

    def run_locally(self) -> dict[str, Any]:
        """Run pipeline steps locally (without ClearML agents).

        Useful for debugging or when ClearML agents aren't available.

        Returns:
            Dict containing outputs from all steps
        """
        from dapidl.pipeline.base import StepArtifacts
        from dapidl.pipeline.steps import (
            AnnotationStep,
            DataLoaderStep,
            PatchExtractionStep,
            SegmentationStep,
            TrainingStep,
        )
        from dapidl.pipeline.steps.annotation import AnnotationStepConfig
        from dapidl.pipeline.steps.data_loader import DataLoaderConfig
        from dapidl.pipeline.steps.patch_extraction import PatchExtractionConfig
        from dapidl.pipeline.steps.segmentation import SegmentationStepConfig
        from dapidl.pipeline.steps.training import TrainingStepConfig

        cfg = self.config
        results = {}

        # Step 1: Data Loader
        logger.info("=" * 50)
        logger.info("Step 1: Data Loader")
        logger.info("=" * 50)

        data_loader_config = DataLoaderConfig(
            dataset_id=cfg.dataset_id,
            dataset_name=cfg.dataset_name,
            dataset_project=cfg.dataset_project,
            platform=cfg.platform,
            local_path=cfg.local_path,
        )
        data_loader = DataLoaderStep(data_loader_config)
        artifacts = data_loader.execute(StepArtifacts(inputs={}, outputs={}))
        results["data_loader"] = artifacts.outputs
        logger.info(f"Data loader outputs: {list(artifacts.outputs.keys())}")

        # Step 2: Segmentation
        logger.info("=" * 50)
        logger.info("Step 2: Segmentation")
        logger.info("=" * 50)

        seg_config = SegmentationStepConfig(
            segmenter=cfg.segmenter,
            diameter=cfg.diameter,
            flow_threshold=cfg.flow_threshold,
            match_threshold_um=cfg.match_threshold_um,
            platform=artifacts.outputs.get("platform", cfg.platform),
        )
        segmentation = SegmentationStep(seg_config)
        artifacts = segmentation.execute(artifacts)
        results["segmentation"] = artifacts.outputs
        logger.info(f"Segmentation outputs: {list(artifacts.outputs.keys())}")

        # Step 3: Annotation (runs in parallel with segmentation in ClearML)
        logger.info("=" * 50)
        logger.info("Step 3: Annotation")
        logger.info("=" * 50)

        annot_config = AnnotationStepConfig(
            annotator=cfg.annotator,
            strategy=cfg.annotation_strategy,
            model_names=cfg.model_names,
            confidence_threshold=cfg.confidence_threshold,
            ground_truth_file=cfg.ground_truth_file,
            fine_grained=cfg.fine_grained,
            extended_consensus=cfg.extended_consensus,
        )
        annotation = AnnotationStep(annot_config)

        # Use data_loader outputs for annotation (not segmentation outputs)
        data_loader_artifacts = StepArtifacts(
            inputs={}, outputs=results["data_loader"]
        )
        annot_artifacts = annotation.execute(data_loader_artifacts)
        results["annotation"] = annot_artifacts.outputs
        logger.info(f"Annotation outputs: {list(annot_artifacts.outputs.keys())}")

        # Merge segmentation and annotation outputs
        merged_outputs = {**artifacts.outputs, **annot_artifacts.outputs}
        artifacts = StepArtifacts(inputs={}, outputs=merged_outputs)

        # Step 4: Patch Extraction
        logger.info("=" * 50)
        logger.info("Step 4: Patch Extraction")
        logger.info("=" * 50)

        patch_config = PatchExtractionConfig(
            patch_size=cfg.patch_size,
            output_format=cfg.output_format,
            normalization_method=cfg.normalization,
            create_dataset=False,  # Don't create ClearML dataset in local mode
        )
        patch_extraction = PatchExtractionStep(patch_config)
        artifacts = patch_extraction.execute(artifacts)
        results["patch_extraction"] = artifacts.outputs
        logger.info(f"Patch extraction outputs: {list(artifacts.outputs.keys())}")

        # Step 5: Training
        logger.info("=" * 50)
        logger.info("Step 5: Training")
        logger.info("=" * 50)

        train_config = TrainingStepConfig(
            backbone=cfg.backbone,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
        )
        training = TrainingStep(train_config)
        artifacts = training.execute(artifacts)
        results["training"] = artifacts.outputs
        logger.info(f"Training outputs: {list(artifacts.outputs.keys())}")

        # Step 6 (Optional): Cross-modal Validation
        if cfg.run_validation:
            from dapidl.pipeline.steps import CrossValidationStep
            from dapidl.pipeline.steps.cross_validation import CrossValidationConfig

            logger.info("=" * 50)
            logger.info("Step 6: Cross-modal Validation")
            logger.info("=" * 50)

            # Merge all outputs for validation
            validation_outputs = {
                **results["data_loader"],
                **results["patch_extraction"],
                **results["annotation"],
                **results["training"],
            }
            validation_artifacts = StepArtifacts(inputs={}, outputs=validation_outputs)

            validation_config = CrossValidationConfig(
                run_leiden_check=cfg.validation_leiden,
                run_dapi_check=cfg.validation_dapi,
                run_consensus_check=cfg.validation_consensus,
                min_ari_threshold=cfg.min_ari_threshold,
                min_agreement_threshold=cfg.min_agreement_threshold,
                run_ground_truth_comparison=cfg.run_ground_truth_comparison,
                ground_truth_file=cfg.gt_comparison_file,
                ground_truth_sheet=cfg.gt_comparison_sheet,
                ground_truth_cell_id_col=cfg.gt_comparison_cell_id_col,
                ground_truth_label_col=cfg.gt_comparison_label_col,
            )
            validation_step = CrossValidationStep(validation_config)
            validation_artifacts = validation_step.execute(validation_artifacts)
            results["cross_validation"] = validation_artifacts.outputs
            logger.info(f"Validation outputs: {list(validation_artifacts.outputs.keys())}")

        # Step 7 (Optional): Cross-Platform Transfer Testing
        if cfg.run_transfer_test:
            from dapidl.pipeline.steps import CrossPlatformTransferStep
            from dapidl.pipeline.steps.cross_platform_transfer import CrossPlatformTransferConfig

            logger.info("=" * 50)
            logger.info("Step 7: Cross-Platform Transfer Testing")
            logger.info("=" * 50)

            # Merge outputs for transfer testing
            transfer_outputs = {
                **results["training"],
                "platform": results["data_loader"].get("platform", cfg.platform),
            }
            transfer_artifacts = StepArtifacts(inputs=transfer_outputs, outputs={})

            transfer_config = CrossPlatformTransferConfig(
                model_path=results["training"].get("model_path", ""),
                target_dataset_id=cfg.transfer_target_dataset_id or "",
                target_local_path=cfg.transfer_target_local_path or "",
                target_platform=cfg.transfer_target_platform,
                source_platform=results["data_loader"].get("platform", cfg.platform),
                patch_size=cfg.patch_size,
                normalize_physical_size=True,
            )
            transfer_step = CrossPlatformTransferStep(transfer_config)
            transfer_artifacts = transfer_step.execute(transfer_artifacts)
            results["cross_platform_transfer"] = transfer_artifacts.outputs
            logger.info(f"Transfer outputs: {list(transfer_artifacts.outputs.keys())}")

        # Step 8 (Optional): Documentation Generation
        if cfg.run_documentation:
            from dapidl.pipeline.steps import DocumentationStep
            from dapidl.pipeline.steps.documentation import DocumentationConfig

            logger.info("=" * 50)
            logger.info("Step 8: Documentation Generation")
            logger.info("=" * 50)

            # Merge all outputs for documentation
            doc_outputs = {
                **results.get("data_loader", {}),
                **results.get("training", {}),
            }
            if "cross_validation" in results:
                doc_outputs["cross_modal_validation"] = results["cross_validation"]
            if "cross_platform_transfer" in results:
                doc_outputs["transfer_metrics"] = results["cross_platform_transfer"].get("transfer_metrics")

            doc_artifacts = StepArtifacts(inputs=doc_outputs, outputs={})

            doc_config = DocumentationConfig(
                obsidian_vault_path=cfg.obsidian_vault_path or "",
                obsidian_folder=cfg.obsidian_folder,
                template=cfg.doc_template,
                experiment_name=cfg.name,
            )
            doc_step = DocumentationStep(doc_config)
            doc_artifacts = doc_step.execute(doc_artifacts)
            results["documentation"] = doc_artifacts.outputs
            logger.info(f"Documentation outputs: {list(doc_artifacts.outputs.keys())}")

        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 50)

        return results

    def create_base_tasks(self):
        """Create base tasks for each step (required before running pipeline).

        This registers each step as a ClearML Task so the PipelineController
        can clone and parameterize them.
        """
        from dapidl.pipeline.steps import (
            AnnotationStep,
            CrossPlatformTransferStep,
            CrossValidationStep,
            DataLoaderStep,
            DocumentationStep,
            PatchExtractionStep,
            SegmentationStep,
            TrainingStep,
        )

        cfg = self.config
        steps = [
            DataLoaderStep(),
            SegmentationStep(),
            AnnotationStep(),
            PatchExtractionStep(),
            TrainingStep(),
            CrossValidationStep(),
            CrossPlatformTransferStep(),
            DocumentationStep(),
        ]

        for step in steps:
            task = step.create_clearml_task(
                project=cfg.project,
                task_name=f"step-{step.name}",
            )
            task.close()
            logger.info(f"Created base task: step-{step.name}")

        logger.info("All base tasks created")

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Dict with pipeline and step statuses
        """
        if self._pipeline is None:
            return {"status": "not_created"}

        return {
            "pipeline_id": self._pipeline.id,
            "status": self._pipeline.get_status(),
            "steps": {
                step.name: step.get_status()
                for step in self._pipeline.get_steps()
            },
        }


def create_pipeline(config: PipelineConfig) -> DAPIDLPipelineController:
    """Factory function to create a configured pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Configured pipeline controller
    """
    return DAPIDLPipelineController(config)
