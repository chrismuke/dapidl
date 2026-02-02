"""Task-based Pipeline Orchestrator for DAPIDL.

Unlike UnifiedPipelineController (which builds a static ClearML DAG),
this orchestrator composes pipelines from independent ClearML tasks:

- Each dataset can use a different recipe (step sequence)
- Pre-built LMDB datasets can be reused directly
- Steps are dispatched via Task.clone() + Task.enqueue() for remote execution
- For local execution, steps run in-process (same as unified_controller)

Recipes define step sequences per dataset:
    "default"       → data_loader → ensemble_annotation → cl_standardization → lmdb_creation
    "gt"            → data_loader → gt_annotation → lmdb_creation  (ground truth)
    "no_cl"         → data_loader → ensemble_annotation → lmdb_creation
    "annotate_only" → data_loader → ensemble_annotation → cl_standardization

The "lmdb:" prefix in datasets/spec skips all processing and feeds an
existing LMDB dataset directly to training.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from dapidl.pipeline.unified_config import (
    DAPIDLPipelineConfig,
    TissueDatasetConfig,
)

# Named step sequences. Each recipe is an ordered list of step names.
RECIPES: dict[str, list[str]] = {
    "default": ["data_loader", "ensemble_annotation", "cl_standardization", "lmdb_creation"],
    "gt": ["data_loader", "gt_annotation", "lmdb_creation"],
    "no_cl": ["data_loader", "ensemble_annotation", "lmdb_creation"],
    "annotate_only": ["data_loader", "ensemble_annotation", "cl_standardization"],
}

# Steps that need a GPU queue
GPU_STEPS = {"training", "universal_training", "hierarchical_training", "segmentation"}


@dataclass
class PipelineResult:
    """Result from orchestrator execution."""

    success: bool
    model_path: str | None = None
    lmdb_paths: list[str] = field(default_factory=list)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class PipelineOrchestrator:
    """Composes pipelines from independent ClearML tasks.

    Unlike PipelineController (static DAG), this orchestrator:
    - Runs steps sequentially per dataset, passing artifacts between them
    - Supports different step sequences (recipes) per dataset
    - Can reuse existing LMDB datasets directly
    - Collects all LMDB paths, then runs a single training step
    """

    def __init__(self, config: DAPIDLPipelineConfig):
        self.config = config
        self._run_id = uuid.uuid4().hex[:8]

    def run(self) -> PipelineResult:
        """Execute the pipeline.

        Dispatches to local or remote execution based on config.
        """
        if self.config.execution.execute_remotely:
            return self._run_remote()
        else:
            return self._run_local()

    # =====================================================================
    # Local execution — steps run in-process
    # =====================================================================

    def _run_local(self) -> PipelineResult:
        """Execute pipeline locally, running each step in-process."""
        cfg = self.config

        try:
            # Phase 1: Process each dataset according to its recipe
            all_lmdb_paths: list[str] = []
            all_weights: list[float] = []
            all_tissues: list[TissueDatasetConfig] = []

            for i, tissue_cfg in enumerate(cfg.input.tissues):
                label = f"{tissue_cfg.tissue} ({i + 1}/{len(cfg.input.tissues)})"
                logger.info(f"{'=' * 60}")
                logger.info(f"Processing dataset: {label}")
                logger.info(f"  Recipe: {tissue_cfg.recipe}")
                logger.info(f"  LMDB direct: {tissue_cfg.is_lmdb}")
                logger.info(f"{'=' * 60}")

                result = self._process_dataset_local(tissue_cfg)
                lmdb_path = result.get("lmdb_path")

                if lmdb_path:
                    all_lmdb_paths.append(lmdb_path)
                    all_weights.append(tissue_cfg.weight_multiplier)
                    all_tissues.append(tissue_cfg)
                    logger.info(f"  LMDB: {lmdb_path}")
                else:
                    logger.warning(f"  No LMDB produced for {label}")

            if not all_lmdb_paths:
                logger.warning("No LMDB datasets produced. Nothing to train on.")
                return PipelineResult(success=True, lmdb_paths=[])

            # Phase 2: Training
            if cfg.execution.skip_training:
                logger.info("Training SKIPPED (prepare-only mode)")
                return PipelineResult(success=True, lmdb_paths=all_lmdb_paths)

            logger.info(f"{'=' * 60}")
            logger.info(f"Training on {len(all_lmdb_paths)} LMDB dataset(s)")
            logger.info(f"{'=' * 60}")

            train_result = self._run_training_local(all_lmdb_paths, all_tissues)

            return PipelineResult(
                success=True,
                lmdb_paths=all_lmdb_paths,
                model_path=train_result.get("model_path"),
                training_metrics=train_result.get("test_metrics", {}),
            )

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return PipelineResult(success=False, error=str(e))

    def _process_dataset_local(self, tissue_cfg: TissueDatasetConfig) -> dict[str, Any]:
        """Run recipe steps for one dataset locally."""
        from dapidl.pipeline.base import StepArtifacts

        # Pre-built LMDB: resolve and return immediately
        if tissue_cfg.is_lmdb:
            lmdb_path = self._resolve_lmdb(tissue_cfg)
            return {"lmdb_path": lmdb_path}

        recipe_name = tissue_cfg.recipe
        if recipe_name not in RECIPES:
            raise ValueError(f"Unknown recipe '{recipe_name}'. Available: {list(RECIPES.keys())}")

        recipe_steps = RECIPES[recipe_name]

        # Run steps in sequence, passing artifacts forward
        artifacts = StepArtifacts()

        for step_name in recipe_steps:
            logger.info(f"  Running step: {step_name}")
            artifacts = self._execute_step_local(step_name, tissue_cfg, artifacts)

        # Extract LMDB path from final artifacts
        lmdb_path = artifacts.outputs.get("lmdb_path")
        return {"lmdb_path": lmdb_path, "artifacts": artifacts}

    def _execute_step_local(
        self,
        step_name: str,
        tissue_cfg: TissueDatasetConfig,
        input_artifacts: Any,
    ) -> Any:
        """Execute a single step locally, returning its output artifacts."""
        from dapidl.pipeline.base import StepArtifacts

        cfg = self.config

        if step_name == "data_loader":
            from dapidl.pipeline.steps.data_loader import DataLoaderConfig, DataLoaderStep

            step_config = DataLoaderConfig(
                dataset_id=tissue_cfg.dataset_id,
                platform=tissue_cfg.platform.value,
                local_path=tissue_cfg.local_path,
            )
            step = DataLoaderStep(step_config)
            return step.execute(StepArtifacts())

        elif step_name == "ensemble_annotation":
            from dapidl.pipeline.steps.ensemble_annotation import (
                EnsembleAnnotationConfig,
                EnsembleAnnotationStep,
            )

            step_config = EnsembleAnnotationConfig(
                celltypist_models=cfg.annotation.celltypist_models,
                include_singler=cfg.annotation.include_singler,
                singler_reference=cfg.annotation.singler_reference,
                min_agreement=cfg.annotation.min_agreement,
                confidence_threshold=cfg.annotation.confidence_threshold,
                fine_grained=cfg.annotation.fine_grained,
                upload_to_s3=cfg.output.upload_to_s3,
                s3_bucket=cfg.output.s3_bucket,
            )
            step = EnsembleAnnotationStep(step_config)
            return step.execute(input_artifacts)

        elif step_name == "gt_annotation":
            from dapidl.pipeline.steps.annotation import AnnotationStep, AnnotationStepConfig

            step_config = AnnotationStepConfig(
                annotator="ground_truth",
                ground_truth_file=cfg.annotation.ground_truth_file,
                ground_truth_sheet=cfg.annotation.ground_truth_sheet,
                cell_id_column=cfg.annotation.ground_truth_cell_id_col,
                celltype_column=cfg.annotation.ground_truth_label_col,
                fine_grained=cfg.annotation.fine_grained,
            )
            step = AnnotationStep(step_config)
            return step.execute(input_artifacts)

        elif step_name == "cl_standardization":
            from dapidl.pipeline.steps.cl_standardization import (
                CLStandardizationConfig,
                CLStandardizationStep,
            )

            step_config = CLStandardizationConfig(
                target_level=cfg.ontology.target_level.value,
                min_confidence=cfg.ontology.min_confidence,
                include_unmapped=cfg.ontology.include_unmapped,
                fuzzy_threshold=cfg.ontology.fuzzy_threshold,
            )
            step = CLStandardizationStep(step_config)
            return step.execute(input_artifacts)

        elif step_name == "lmdb_creation":
            from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep

            primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
            step_config = LMDBCreationConfig(
                patch_size=primary_patch_size,
                normalization_method=cfg.lmdb.normalization.value,
                normalize_physical_size=cfg.lmdb.normalize_physical_size,
                skip_if_exists=cfg.lmdb.skip_if_exists,
                upload_to_s3=cfg.output.upload_to_s3,
                s3_bucket=cfg.output.s3_bucket,
            )
            step = LMDBCreationStep(step_config)
            return step.execute(input_artifacts)

        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _run_training_local(
        self,
        lmdb_paths: list[str],
        tissues: list[TissueDatasetConfig],
    ) -> dict[str, Any]:
        """Run training step locally on collected LMDB paths."""
        from dapidl.pipeline.base import StepArtifacts
        from dapidl.pipeline.steps.universal_training import (
            TissueDatasetSpec,
            UniversalDAPITrainingStep,
            UniversalTrainingConfig,
        )

        cfg = self.config

        # Build dataset specs for universal training
        dataset_specs = [
            TissueDatasetSpec(
                path=path,
                tissue=tc.tissue,
                platform=tc.platform.value,
                confidence_tier=tc.confidence_tier,
                weight_multiplier=tc.weight_multiplier,
            )
            for path, tc in zip(lmdb_paths, tissues, strict=True)
        ]

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
            datasets=dataset_specs,
        )

        input_artifacts = StepArtifacts(
            inputs={},
            outputs={"dataset_configs": dataset_specs},
        )
        training_step = UniversalDAPITrainingStep(train_config)
        result_artifacts = training_step.execute(input_artifacts)

        return {
            "model_path": result_artifacts.outputs.get("model_path"),
            "test_metrics": result_artifacts.outputs.get("test_metrics", {}),
        }

    # =====================================================================
    # Remote execution — steps dispatched via ClearML Task.clone()
    # =====================================================================

    def _run_remote(self) -> PipelineResult:
        """Execute pipeline remotely via ClearML task cloning."""
        cfg = self.config

        try:
            # Phase 1: Process each dataset
            all_lmdb_paths: list[str] = []
            all_weights: list[float] = []
            all_tissues: list[TissueDatasetConfig] = []

            for i, tissue_cfg in enumerate(cfg.input.tissues):
                label = f"{tissue_cfg.tissue} ({i + 1}/{len(cfg.input.tissues)})"
                logger.info(f"Processing dataset: {label} [recipe={tissue_cfg.recipe}]")

                result = self._process_dataset_remote(tissue_cfg)
                lmdb_path = result.get("lmdb_path")

                if lmdb_path:
                    all_lmdb_paths.append(lmdb_path)
                    all_weights.append(tissue_cfg.weight_multiplier)
                    all_tissues.append(tissue_cfg)
                    logger.info(f"  LMDB: {lmdb_path}")

            if not all_lmdb_paths:
                return PipelineResult(success=True, lmdb_paths=[])

            # Phase 2: Training
            if cfg.execution.skip_training:
                logger.info("Training SKIPPED (prepare-only mode)")
                return PipelineResult(success=True, lmdb_paths=all_lmdb_paths)

            logger.info(f"Training on {len(all_lmdb_paths)} LMDB dataset(s)")
            train_task = self._run_training_remote(all_lmdb_paths, all_tissues)

            model_path = None
            if train_task and train_task.artifacts:
                model_artifact = train_task.artifacts.get("model_path")
                if model_artifact:
                    model_path = model_artifact.url

            return PipelineResult(
                success=True,
                lmdb_paths=all_lmdb_paths,
                model_path=model_path,
            )

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return PipelineResult(success=False, error=str(e))

    def _process_dataset_remote(self, tissue_cfg: TissueDatasetConfig) -> dict[str, Any]:
        """Run recipe steps for one dataset via ClearML task cloning."""
        if tissue_cfg.is_lmdb:
            lmdb_path = self._resolve_lmdb(tissue_cfg)
            return {"lmdb_path": lmdb_path}

        recipe_name = tissue_cfg.recipe
        if recipe_name not in RECIPES:
            raise ValueError(f"Unknown recipe '{recipe_name}'. Available: {list(RECIPES.keys())}")

        recipe_steps = RECIPES[recipe_name]
        cfg = self.config

        # Build parameters for each step, chaining artifact outputs
        artifacts: dict[str, str] = {}

        for step_name in recipe_steps:
            params = self._build_remote_step_params(step_name, tissue_cfg, artifacts)
            queue = cfg.execution.gpu_queue if step_name in GPU_STEPS else cfg.execution.default_queue
            task = self._clone_and_run_step(step_name, params, queue)

            # Extract artifacts from completed task
            step_artifacts = self._extract_task_artifacts(task)
            artifacts.update(step_artifacts)

        return {"lmdb_path": artifacts.get("lmdb_path")}

    def _build_remote_step_params(
        self,
        step_name: str,
        tissue_cfg: TissueDatasetConfig,
        prev_artifacts: dict[str, str],
    ) -> dict[str, str]:
        """Build parameter dict for a remote step task."""
        cfg = self.config
        params: dict[str, str] = {}

        if step_name == "data_loader":
            params["step_config/dataset_id"] = tissue_cfg.dataset_id or ""
            params["step_config/platform"] = tissue_cfg.platform.value
            params["step_config/local_path"] = tissue_cfg.local_path or ""

        elif step_name == "gt_annotation":
            params["step_config/annotator"] = "ground_truth"
            params["step_config/ground_truth_file"] = cfg.annotation.ground_truth_file or ""
            params["step_config/ground_truth_sheet"] = cfg.annotation.ground_truth_sheet or ""
            params["step_config/fine_grained"] = str(cfg.annotation.fine_grained)
            if "data_path" in prev_artifacts:
                params["step_config/data_path"] = prev_artifacts["data_path"]
            if "cells_parquet" in prev_artifacts:
                params["step_config/cells_parquet"] = prev_artifacts["cells_parquet"]

        elif step_name == "ensemble_annotation":
            params["step_config/celltypist_models"] = ",".join(cfg.annotation.celltypist_models)
            params["step_config/include_singler"] = str(cfg.annotation.include_singler)
            params["step_config/singler_reference"] = cfg.annotation.singler_reference
            params["step_config/min_agreement"] = str(cfg.annotation.min_agreement)
            params["step_config/confidence_threshold"] = str(cfg.annotation.confidence_threshold)
            params["step_config/fine_grained"] = str(cfg.annotation.fine_grained)
            # Chain from data_loader
            if "data_path" in prev_artifacts:
                params["step_config/data_path"] = prev_artifacts["data_path"]
            if "expression_path" in prev_artifacts:
                params["step_config/expression_path"] = prev_artifacts["expression_path"]

        elif step_name == "cl_standardization":
            params["step_config/target_level"] = cfg.ontology.target_level.value
            params["step_config/min_confidence"] = str(cfg.ontology.min_confidence)
            params["step_config/include_unmapped"] = str(cfg.ontology.include_unmapped)
            params["step_config/fuzzy_threshold"] = str(cfg.ontology.fuzzy_threshold)
            if "annotations_parquet" in prev_artifacts:
                params["step_config/annotations_parquet"] = prev_artifacts["annotations_parquet"]
            if "data_path" in prev_artifacts:
                params["step_config/data_path"] = prev_artifacts["data_path"]

        elif step_name == "lmdb_creation":
            primary_patch_size = cfg.lmdb.primary_patch_size or cfg.lmdb.patch_sizes[0]
            params["step_config/patch_size"] = str(primary_patch_size)
            params["step_config/normalization_method"] = cfg.lmdb.normalization.value
            params["step_config/normalize_physical_size"] = str(cfg.lmdb.normalize_physical_size)
            params["step_config/skip_if_exists"] = str(cfg.lmdb.skip_if_exists)
            params["step_config/upload_to_s3"] = str(cfg.output.upload_to_s3)
            params["step_config/s3_bucket"] = cfg.output.s3_bucket
            if "data_path" in prev_artifacts:
                params["step_config/data_path"] = prev_artifacts["data_path"]
            # Forward annotation artifacts — cl_annotations_parquet preferred over annotations_parquet
            if "cl_annotations_parquet" in prev_artifacts:
                params["step_config/cl_annotations_parquet"] = prev_artifacts["cl_annotations_parquet"]
            if "annotations_parquet" in prev_artifacts:
                params["step_config/annotations_parquet"] = prev_artifacts["annotations_parquet"]

        return params

    def _clone_and_run_step(self, step_name: str, params: dict[str, str], queue: str) -> Any:
        """Clone a base task, configure it, enqueue, and wait for completion."""
        from clearml import Task

        logger.info(f"  Cloning step-{step_name} → queue '{queue}'")

        base_task = Task.get_task(
            project_name=self.config.project,
            task_name=f"step-{step_name}",
        )
        cloned = Task.clone(
            source_task=base_task,
            name=f"{step_name}-{self._run_id}",
        )

        # Apply parameter overrides (set each individually to merge with existing)
        for key, value in params.items():
            cloned.set_parameter(key, value)

        # Enqueue and wait
        Task.enqueue(cloned, queue_name=queue)
        logger.info(f"  Enqueued {step_name} (task {cloned.id[:8]}), waiting...")

        cloned.wait_for_status(
            status=("completed", "failed", "stopped"),
            check_interval_sec=10,
        )

        if cloned.status != "completed":
            raise RuntimeError(
                f"Step {step_name} failed with status '{cloned.status}' "
                f"(task {cloned.id[:8]})"
            )

        logger.info(f"  Step {step_name} completed")
        return cloned

    def _extract_task_artifacts(self, task: Any) -> dict[str, str]:
        """Extract artifact URLs from a completed ClearML task."""
        result: dict[str, str] = {}
        if not task.artifacts:
            return result

        for name, artifact in task.artifacts.items():
            if hasattr(artifact, "url") and artifact.url:
                result[name] = artifact.url

        return result

    def _run_training_remote(
        self,
        lmdb_paths: list[str],
        tissues: list[TissueDatasetConfig],
    ) -> Any:
        """Run training step remotely."""
        cfg = self.config

        params: dict[str, str] = {
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
        }

        # Pass LMDB paths as numbered parameters
        for i, path in enumerate(lmdb_paths):
            params[f"step_config/lmdb_path_{i}"] = path

        step_name = "universal_training"
        queue = cfg.execution.gpu_queue
        return self._clone_and_run_step(step_name, params, queue)

    # =====================================================================
    # Helpers
    # =====================================================================

    def _resolve_lmdb(self, tissue_cfg: TissueDatasetConfig) -> str:
        """Resolve an LMDB dataset reference to a usable path.

        Search order:
        1. Explicit local_path on the config
        2. Local derived dataset directories (DAPIDL_DERIVED_DATA or defaults)
        3. Absolute path (if name looks like one)
        4. ClearML Dataset resolution
        """
        import os
        from pathlib import Path

        name = tissue_cfg.dataset_id or tissue_cfg.local_path
        if not name:
            raise ValueError("LMDB dataset has no dataset_id or local_path")

        # 1. Explicit local path
        if tissue_cfg.local_path:
            return tissue_cfg.local_path

        # 2. Search local derived directories
        derived_dirs = self._get_derived_data_dirs()
        for d in derived_dirs:
            candidate = d / name
            if candidate.is_dir() and (candidate / "patches.lmdb").exists():
                logger.info(f"Resolved LMDB '{name}' → {candidate} (local derived)")
                return str(candidate)

        # 3. If it looks like an absolute path, use directly
        if os.path.isabs(name) and Path(name).is_dir():
            return name

        # 4. Try ClearML dataset resolution
        try:
            from clearml import Dataset

            ds = Dataset.get(
                dataset_project="DAPIDL/datasets",
                dataset_name=name,
                only_completed=True,
            )
            local_path = ds.get_local_copy()
            logger.info(f"Resolved LMDB '{name}' → {local_path}")
            return local_path
        except Exception:
            logger.warning(f"Could not resolve LMDB '{name}' via ClearML, using as path")
            return name

    @staticmethod
    def _get_derived_data_dirs() -> list[Any]:
        """Get directories to search for pre-built LMDB datasets.

        Uses DAPIDL_DERIVED_DATA env var (colon-separated) or defaults to
        /mnt/work/datasets/derived and ~/datasets/derived.
        """
        import os
        from pathlib import Path

        env = os.environ.get("DAPIDL_DERIVED_DATA")
        if env:
            return [Path(p) for p in env.split(":") if p]

        defaults = [
            Path("/mnt/work/datasets/derived"),
            Path.home() / "datasets" / "derived",
        ]
        return [d for d in defaults if d.is_dir()]
