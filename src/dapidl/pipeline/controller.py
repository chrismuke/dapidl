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
        self._pipeline.add_step(
            name="data_loader",
            base_task_project=cfg.project,
            base_task_name="step-data_loader",
            parameter_override={
                "step_config/dataset_id": "${pipeline.dataset_id}",
                "step_config/platform": "${pipeline.platform}",
                "step_config/local_path": cfg.local_path or "",
            },
            execution_queue=cfg.default_queue if cfg.execute_remotely else None,
            cache_executed_step=True,  # Skip if inputs/params unchanged
        )

        # Step 2: Segmentation (depends on data_loader)
        self._pipeline.add_step(
            name="segmentation",
            parents=["data_loader"],
            base_task_project=cfg.project,
            base_task_name="step-segmentation",
            parameter_override={
                "step_config/segmenter": "${pipeline.segmenter}",
                "step_config/diameter": cfg.diameter,
                "step_config/flow_threshold": cfg.flow_threshold,
                "step_config/match_threshold_um": cfg.match_threshold_um,
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
                "step_config/annotator": "${pipeline.annotator}",
                "step_config/strategy": cfg.annotation_strategy,
                "step_config/model_names": ",".join(cfg.model_names),
                "step_config/confidence_threshold": cfg.confidence_threshold,
                "step_config/ground_truth_file": cfg.ground_truth_file or "",
                "step_config/fine_grained": cfg.fine_grained,
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
                "step_config/patch_size": "${pipeline.patch_size}",
                "step_config/output_format": cfg.output_format,
                "step_config/normalization_method": cfg.normalization,
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
                "step_config/epochs": "${pipeline.epochs}",
                "step_config/batch_size": cfg.batch_size,
                "step_config/learning_rate": cfg.learning_rate,
            },
            execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
            cache_executed_step=cfg.cache_training,  # Configurable
        )

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
            DataLoaderStep,
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
