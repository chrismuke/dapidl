"""Universal Pipeline Controller for cross-tissue DAPIDL training.

Orchestrates multi-tissue training pipelines that:
1. Process multiple datasets from different tissues in parallel
2. Combine them using Cell Ontology standardized labels
3. Train a universal classifier with tissue-balanced sampling

Pipeline Flow (per tissue):
    DataLoader → Segmentation ─┐
         │                      │
         └─→ Annotation ────────┼→ PatchExtraction
                                         │
                                         ▼ (all tissues)
                              UniversalTraining

Supports both ClearML remote execution and local debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TissueConfig:
    """Configuration for a single tissue dataset in the universal pipeline."""

    # Data source
    dataset_id: str | None = None
    dataset_name: str | None = None
    local_path: str | None = None

    # Tissue metadata
    tissue: str = "unknown"
    platform: str = "xenium"
    confidence_tier: int = 2  # 1=ground truth, 2=consensus, 3=predicted
    weight_multiplier: float = 1.0

    # Annotation
    annotator: str = "celltypist"
    model_names: list[str] = field(
        default_factory=lambda: ["Immune_All_High.pkl", "Cells_Adult_Breast.pkl"]
    )
    ground_truth_file: str | None = None


@dataclass
class UniversalPipelineConfig:
    """Configuration for the universal cross-tissue pipeline."""

    # Pipeline metadata
    name: str = "dapidl-universal"
    project: str = "DAPIDL/universal"
    version: str = "1.0.0"

    # Tissue datasets
    tissues: list[TissueConfig] = field(default_factory=list)

    # Sampling strategy
    sampling_strategy: str = "sqrt"  # "equal", "proportional", "sqrt"

    # Confidence tier weights
    tier1_weight: float = 1.0
    tier2_weight: float = 0.8
    tier3_weight: float = 0.5

    # Segmentation (shared across all)
    segmenter: str = "cellpose"
    diameter: int = 40
    flow_threshold: float = 0.4

    # Patch extraction
    patch_size: int = 128
    output_format: str = "lmdb"

    # Training
    backbone: str = "efficientnetv2_rw_s"
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4

    # Curriculum learning
    coarse_only_epochs: int = 20
    coarse_medium_epochs: int = 50

    # Cell Ontology standardization
    standardize_labels: bool = True

    # Execution
    execute_remotely: bool = True
    default_queue: str = "default"
    gpu_queue: str = "gpu"

    # Output
    output_dir: str | None = None

    def add_tissue(
        self,
        tissue: str,
        dataset_id: str | None = None,
        local_path: str | None = None,
        platform: str = "xenium",
        confidence_tier: int = 2,
        annotator: str = "celltypist",
        ground_truth_file: str | None = None,
        weight_multiplier: float = 1.0,
    ) -> "UniversalPipelineConfig":
        """Add a tissue dataset to the pipeline.

        Args:
            tissue: Tissue type (breast, lung, liver, etc.)
            dataset_id: ClearML Dataset ID
            local_path: Local path (alternative to dataset_id)
            platform: Platform (xenium, merscope)
            confidence_tier: Label confidence tier
            annotator: Annotation method
            ground_truth_file: Path to ground truth labels
            weight_multiplier: Weight multiplier for this tissue

        Returns:
            Self for chaining
        """
        self.tissues.append(TissueConfig(
            tissue=tissue,
            dataset_id=dataset_id,
            local_path=local_path,
            platform=platform,
            confidence_tier=confidence_tier,
            annotator=annotator,
            ground_truth_file=ground_truth_file,
            weight_multiplier=weight_multiplier,
        ))
        return self


class UniversalDAPIPipelineController:
    """Orchestrates multi-tissue universal DAPIDL training.

    Creates a ClearML pipeline that processes multiple tissue datasets
    in parallel, then combines them for universal training.

    Example:
        ```python
        config = UniversalPipelineConfig(name="universal-v1")
        config.add_tissue("breast", dataset_id="abc123", confidence_tier=1)
        config.add_tissue("lung", dataset_id="def456", confidence_tier=2)
        config.add_tissue("liver", local_path="/data/liver", confidence_tier=2)

        controller = UniversalDAPIPipelineController(config)
        controller.run()
        ```
    """

    def __init__(self, config: UniversalPipelineConfig | None = None):
        """Initialize the universal pipeline controller.

        Args:
            config: Pipeline configuration
        """
        self.config = config or UniversalPipelineConfig()
        self._pipeline = None
        self._step_tasks = {}

    def create_pipeline(self):
        """Create ClearML PipelineController with multi-tissue steps."""
        from clearml import PipelineController

        cfg = self.config

        if len(cfg.tissues) == 0:
            raise ValueError("No tissue datasets configured. Use config.add_tissue() to add datasets.")

        self._pipeline = PipelineController(
            name=cfg.name,
            project=cfg.project,
            version=cfg.version,
            add_pipeline_tags=True,
        )

        if cfg.execute_remotely:
            self._pipeline.set_default_execution_queue(cfg.default_queue)

        # Add pipeline-level parameters
        self._pipeline.add_parameter(
            name="sampling_strategy",
            default=cfg.sampling_strategy,
            description="Tissue sampling strategy: equal, proportional, sqrt",
        )
        self._pipeline.add_parameter(
            name="epochs",
            default=cfg.epochs,
            description="Number of training epochs",
        )
        self._pipeline.add_parameter(
            name="backbone",
            default=cfg.backbone,
            description="CNN backbone architecture",
        )

        # Create per-tissue processing steps
        patch_extraction_steps = []

        for i, tissue_cfg in enumerate(cfg.tissues):
            tissue_name = tissue_cfg.tissue
            step_prefix = f"{tissue_name}_{i}"

            # Data Loader for this tissue
            self._pipeline.add_step(
                name=f"data_loader_{step_prefix}",
                base_task_project=cfg.project,
                base_task_name="step-data_loader",
                parameter_override={
                    "step_config/dataset_id": tissue_cfg.dataset_id or "",
                    "step_config/platform": tissue_cfg.platform,
                    "step_config/local_path": tissue_cfg.local_path or "",
                },
                execution_queue=cfg.default_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )

            # Segmentation for this tissue
            self._pipeline.add_step(
                name=f"segmentation_{step_prefix}",
                parents=[f"data_loader_{step_prefix}"],
                base_task_project=cfg.project,
                base_task_name="step-segmentation",
                parameter_override={
                    "step_config/segmenter": cfg.segmenter,
                    "step_config/diameter": cfg.diameter,
                    "step_config/flow_threshold": cfg.flow_threshold,
                    "step_config/data_path": f"${{data_loader_{step_prefix}.artifacts.data_path.url}}",
                    "step_config/platform": f"${{data_loader_{step_prefix}.artifacts.platform.url}}",
                },
                execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )

            # Annotation for this tissue
            self._pipeline.add_step(
                name=f"annotation_{step_prefix}",
                parents=[f"data_loader_{step_prefix}"],
                base_task_project=cfg.project,
                base_task_name="step-annotation",
                parameter_override={
                    "step_config/annotator": tissue_cfg.annotator,
                    "step_config/strategy": "consensus",
                    "step_config/model_names": ",".join(tissue_cfg.model_names),
                    "step_config/ground_truth_file": tissue_cfg.ground_truth_file or "",
                    "step_config/fine_grained": True,
                    "step_config/data_path": f"${{data_loader_{step_prefix}.artifacts.data_path.url}}",
                    "step_config/platform": f"${{data_loader_{step_prefix}.artifacts.platform.url}}",
                    "step_config/cells_parquet": f"${{data_loader_{step_prefix}.artifacts.cells_parquet.url}}",
                    "step_config/expression_path": f"${{data_loader_{step_prefix}.artifacts.expression_path.url}}",
                },
                execution_queue=cfg.default_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )

            # Patch Extraction for this tissue
            patch_step_name = f"patch_extraction_{step_prefix}"
            self._pipeline.add_step(
                name=patch_step_name,
                parents=[f"segmentation_{step_prefix}", f"annotation_{step_prefix}"],
                base_task_project=cfg.project,
                base_task_name="step-patch_extraction",
                parameter_override={
                    "step_config/patch_size": cfg.patch_size,
                    "step_config/output_format": cfg.output_format,
                    "step_config/data_path": f"${{data_loader_{step_prefix}.artifacts.data_path.url}}",
                    "step_config/platform": f"${{data_loader_{step_prefix}.artifacts.platform.url}}",
                    "step_config/centroids_parquet": f"${{segmentation_{step_prefix}.artifacts.centroids_parquet.url}}",
                    "step_config/annotations_parquet": f"${{annotation_{step_prefix}.artifacts.annotations_parquet.url}}",
                },
                execution_queue=cfg.default_queue if cfg.execute_remotely else None,
                cache_executed_step=True,
            )
            patch_extraction_steps.append(patch_step_name)

            logger.info(f"Added processing steps for tissue: {tissue_name}")

        # Universal Training (depends on all patch extraction steps)
        self._pipeline.add_step(
            name="universal_training",
            parents=patch_extraction_steps,
            base_task_project=cfg.project,
            base_task_name="step-universal_training",
            parameter_override={
                "step_config/backbone": cfg.backbone,
                "step_config/epochs": cfg.epochs,
                "step_config/batch_size": cfg.batch_size,
                "step_config/learning_rate": cfg.learning_rate,
                "step_config/sampling_strategy": cfg.sampling_strategy,
                "step_config/tier1_weight": cfg.tier1_weight,
                "step_config/tier2_weight": cfg.tier2_weight,
                "step_config/tier3_weight": cfg.tier3_weight,
                "step_config/standardize_labels": cfg.standardize_labels,
                "step_config/coarse_only_epochs": cfg.coarse_only_epochs,
                "step_config/coarse_medium_epochs": cfg.coarse_medium_epochs,
                # Pass all patch paths
                **{
                    f"step_config/patches_path_{i}": f"${{{step}.artifacts.dataset_path.url}}"
                    for i, step in enumerate(patch_extraction_steps)
                },
            },
            execution_queue=cfg.gpu_queue if cfg.execute_remotely else None,
            cache_executed_step=False,  # Training usually needs fresh runs
        )

        logger.info(f"Created universal pipeline: {cfg.name} v{cfg.version}")
        logger.info(f"  Tissues: {len(cfg.tissues)}")
        logger.info(f"  Sampling: {cfg.sampling_strategy}")
        return self._pipeline

    def run(self, wait: bool = True) -> str:
        """Run the pipeline on ClearML.

        Args:
            wait: If True, wait for pipeline completion

        Returns:
            Pipeline run ID
        """
        if self._pipeline is None:
            self.create_pipeline()

        logger.info("Starting universal pipeline execution...")
        self._pipeline.start_locally(run_pipeline_steps_locally=False)

        if wait:
            self._pipeline.wait()
            logger.info("Universal pipeline completed")

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
            UniversalDAPITrainingStep,
        )
        from dapidl.pipeline.steps.annotation import AnnotationStepConfig
        from dapidl.pipeline.steps.data_loader import DataLoaderConfig
        from dapidl.pipeline.steps.patch_extraction import PatchExtractionConfig
        from dapidl.pipeline.steps.segmentation import SegmentationStepConfig
        from dapidl.pipeline.steps.universal_training import (
            UniversalTrainingConfig,
            TissueDatasetSpec,
        )

        cfg = self.config
        results = {}
        dataset_configs = []

        # Process each tissue in sequence (parallel in ClearML)
        for i, tissue_cfg in enumerate(cfg.tissues):
            tissue_name = tissue_cfg.tissue
            logger.info("=" * 60)
            logger.info(f"Processing tissue {i+1}/{len(cfg.tissues)}: {tissue_name}")
            logger.info("=" * 60)

            # Step 1: Data Loader
            logger.info(f"  Step 1: Data Loader ({tissue_name})")
            data_loader_config = DataLoaderConfig(
                dataset_id=tissue_cfg.dataset_id,
                platform=tissue_cfg.platform,
                local_path=tissue_cfg.local_path,
            )
            data_loader = DataLoaderStep(data_loader_config)
            artifacts = data_loader.execute(StepArtifacts(inputs={}, outputs={}))
            results[f"data_loader_{tissue_name}"] = artifacts.outputs

            # Step 2: Segmentation
            logger.info(f"  Step 2: Segmentation ({tissue_name})")
            seg_config = SegmentationStepConfig(
                segmenter=cfg.segmenter,
                diameter=cfg.diameter,
                flow_threshold=cfg.flow_threshold,
                platform=artifacts.outputs.get("platform", tissue_cfg.platform),
            )
            segmentation = SegmentationStep(seg_config)
            seg_artifacts = segmentation.execute(artifacts)
            results[f"segmentation_{tissue_name}"] = seg_artifacts.outputs

            # Step 3: Annotation
            logger.info(f"  Step 3: Annotation ({tissue_name})")
            annot_config = AnnotationStepConfig(
                annotator=tissue_cfg.annotator,
                strategy="consensus",
                model_names=tissue_cfg.model_names,
                ground_truth_file=tissue_cfg.ground_truth_file,
                fine_grained=True,
            )
            annotation = AnnotationStep(annot_config)
            annot_artifacts = annotation.execute(artifacts)
            results[f"annotation_{tissue_name}"] = annot_artifacts.outputs

            # Merge segmentation and annotation outputs
            merged_outputs = {**seg_artifacts.outputs, **annot_artifacts.outputs}
            artifacts = StepArtifacts(inputs={}, outputs=merged_outputs)

            # Step 4: Patch Extraction
            logger.info(f"  Step 4: Patch Extraction ({tissue_name})")
            patch_config = PatchExtractionConfig(
                patch_size=cfg.patch_size,
                output_format=cfg.output_format,
                create_dataset=False,
            )
            patch_extraction = PatchExtractionStep(patch_config)
            patch_artifacts = patch_extraction.execute(artifacts)
            results[f"patch_extraction_{tissue_name}"] = patch_artifacts.outputs

            # Collect dataset config for universal training
            dataset_path = patch_artifacts.outputs.get("dataset_path")
            if dataset_path:
                dataset_configs.append({
                    "path": dataset_path,
                    "tissue": tissue_name,
                    "platform": tissue_cfg.platform,
                    "confidence_tier": tissue_cfg.confidence_tier,
                    "weight_multiplier": tissue_cfg.weight_multiplier,
                })

        # Universal Training
        logger.info("=" * 60)
        logger.info("Universal Training")
        logger.info("=" * 60)

        # Build training config
        train_config = UniversalTrainingConfig(
            backbone=cfg.backbone,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            sampling_strategy=cfg.sampling_strategy,
            tier1_weight=cfg.tier1_weight,
            tier2_weight=cfg.tier2_weight,
            tier3_weight=cfg.tier3_weight,
            standardize_labels=cfg.standardize_labels,
            coarse_only_epochs=cfg.coarse_only_epochs,
            coarse_medium_epochs=cfg.coarse_medium_epochs,
            output_dir=cfg.output_dir,
        )

        # Add datasets from collected configs
        for ds in dataset_configs:
            train_config.add_dataset(
                path=ds["path"],
                tissue=ds["tissue"],
                platform=ds["platform"],
                confidence_tier=ds["confidence_tier"],
                weight_multiplier=ds["weight_multiplier"],
            )

        # Create training step
        training_artifacts = StepArtifacts(
            inputs={},
            outputs={"dataset_configs": dataset_configs},
        )
        training_step = UniversalDAPITrainingStep(train_config)
        training_artifacts = training_step.execute(training_artifacts)
        results["universal_training"] = training_artifacts.outputs

        logger.info("=" * 60)
        logger.info("Universal Pipeline completed!")
        logger.info(f"  Model: {training_artifacts.outputs.get('model_path')}")
        logger.info(f"  Test F1: {training_artifacts.outputs.get('test_metrics', {}).get('f1_fine', 'N/A')}")
        logger.info("=" * 60)

        return results

    def create_base_tasks(self):
        """Create base tasks for each step (required before running pipeline)."""
        from dapidl.pipeline.steps import (
            AnnotationStep,
            DataLoaderStep,
            PatchExtractionStep,
            SegmentationStep,
            UniversalDAPITrainingStep,
        )

        cfg = self.config
        steps = [
            DataLoaderStep(),
            SegmentationStep(),
            AnnotationStep(),
            PatchExtractionStep(),
            UniversalDAPITrainingStep(),
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
        """Get current pipeline status."""
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


def create_universal_pipeline(config: UniversalPipelineConfig) -> UniversalDAPIPipelineController:
    """Factory function to create a configured universal pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Configured pipeline controller
    """
    return UniversalDAPIPipelineController(config)
