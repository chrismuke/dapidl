"""Annotation Pipeline Step.

Step 3: Cell type annotation using configurable method.

This step:
1. Loads expression data or ground truth file
2. Runs cell type annotation (CellTypist, popV, or ground truth)
3. Maps detailed types to broad categories
4. Outputs annotation DataFrame with cell types and confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import (
    AnnotationConfig,
    PipelineStep,
    StepArtifacts,
    resolve_artifact_path,
)
from dapidl.pipeline.registry import get_annotator


@dataclass
class AnnotationStepConfig:
    """Configuration for annotation step."""

    # Method selection
    annotator: str = "celltypist"  # "celltypist", "ground_truth", "popv"

    # CellTypist parameters
    strategy: str = "consensus"  # "single", "consensus", "hierarchical"
    model_names: list[str] = field(
        default_factory=lambda: ["Immune_All_High.pkl", "Cells_Adult_Breast.pkl"]
    )
    majority_voting: bool = True
    confidence_threshold: float = 0.5
    extended_consensus: bool = False  # Use 6 CellTypist models for better coverage

    # Ground truth parameters
    ground_truth_file: str | None = None
    ground_truth_sheet: str | None = None
    cell_id_column: str | None = None
    celltype_column: str | None = None

    # Output options
    fine_grained: bool = False
    filter_category: str | None = None  # Filter to specific category


class AnnotationStep(PipelineStep):
    """Cell type annotation step.

    Annotates cells using CellTypist (expression-based), popV (HuggingFace models),
    or loads from ground truth files (Excel/CSV/Parquet).

    Queue: default (CPU - CellTypist is CPU-bound)
    """

    name = "annotation"
    queue = "default"  # CPU queue

    def __init__(self, config: AnnotationStepConfig | None = None):
        """Initialize annotation step.

        Args:
            config: Annotation configuration
        """
        self.config = config or AnnotationStepConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "annotator": {
                    "type": "string",
                    "enum": ["celltypist", "ground_truth", "popv"],
                    "default": "celltypist",
                    "description": "Annotation method",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["single", "consensus", "hierarchical"],
                    "default": "consensus",
                    "description": "CellTypist voting strategy",
                },
                "model_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["Immune_All_High.pkl", "Cells_Adult_Breast.pkl"],
                    "description": "CellTypist model names",
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum annotation confidence",
                },
                "ground_truth_file": {
                    "type": "string",
                    "description": "Path to ground truth file (Excel/CSV)",
                },
                "fine_grained": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use detailed cell types (vs 3 broad categories)",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - For CellTypist/popV: expression_path from data loader
        - For ground_truth: ground_truth_file in config
        """
        cfg = self.config

        if cfg.annotator == "ground_truth":
            return bool(cfg.ground_truth_file)

        # CellTypist/popV need expression data
        return "expression_path" in artifacts.outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute annotation step.

        Args:
            artifacts: Input artifacts from DataLoaderStep

        Returns:
            Output artifacts containing:
            - annotations_parquet: Path to annotation DataFrame
            - class_mapping: Dict mapping class names to indices
            - annotation_stats: Dict with annotation statistics
        """
        cfg = self.config
        inputs = artifacts.outputs

        # Resolve artifact URLs to local paths
        data_path = resolve_artifact_path(inputs["data_path"], "data_path")
        if data_path is None:
            raise ValueError("data_path artifact is required")

        # Platform can be a URL to a text file or a direct value
        platform_value = inputs.get("platform", "xenium")
        platform_path = resolve_artifact_path(platform_value, "platform")
        if platform_path and platform_path.exists() and platform_path.is_file():
            platform = platform_path.read_text().strip()
            logger.info(f"Read platform from artifact: {platform}")
        else:
            platform = str(platform_value)

        # Create annotation config
        annot_config = AnnotationConfig(
            method=cfg.annotator,
            strategy=cfg.strategy,
            model_names=cfg.model_names,
            confidence_threshold=cfg.confidence_threshold,
            majority_voting=cfg.majority_voting,
            fine_grained=cfg.fine_grained,
            ground_truth_file=cfg.ground_truth_file,
            ground_truth_sheet=cfg.ground_truth_sheet,
            cell_id_column=cfg.cell_id_column,
            celltype_column=cfg.celltype_column,
            extended_consensus=cfg.extended_consensus,
        )

        # Get annotator
        annotator = get_annotator(cfg.annotator, annot_config)

        # Prepare inputs based on method
        if cfg.annotator == "ground_truth":
            # Ground truth: load from file
            cells_parquet_path = resolve_artifact_path(
                inputs.get("cells_parquet"), "cells_parquet"
            )
            cells_df = self._load_cells_df(str(cells_parquet_path) if cells_parquet_path else None)
            result = annotator.annotate(
                config=annot_config,
                cells_df=cells_df,
            )
        else:
            # CellTypist/popV: need expression data
            expression_path_raw = inputs.get("expression_path")
            if not expression_path_raw:
                raise ValueError(
                    f"{cfg.annotator} requires expression data but none provided"
                )

            expression_path = resolve_artifact_path(expression_path_raw, "expression_path")
            adata = self._load_expression(expression_path, data_path, platform)

            # Ensure cell_id is in obs (required by CellTypeAnnotator)
            if "cell_id" not in adata.obs.columns:
                adata.obs["cell_id"] = adata.obs.index.astype(str)

            result = annotator.annotate(
                config=annot_config,
                adata=adata,
            )

        logger.info(f"Annotation complete: {result.stats}")

        # Filter by confidence if needed
        annotations_df = result.annotations_df
        if cfg.confidence_threshold > 0 and "confidence" in annotations_df.columns:
            before_count = annotations_df.height
            annotations_df = annotations_df.filter(
                pl.col("confidence") >= cfg.confidence_threshold
            )
            logger.info(
                f"Filtered by confidence >= {cfg.confidence_threshold}: "
                f"{before_count} -> {annotations_df.height}"
            )

        # Filter by category if specified
        if cfg.filter_category and "broad_category" in annotations_df.columns:
            annotations_df = annotations_df.filter(
                pl.col("broad_category") == cfg.filter_category
            )
            logger.info(f"Filtered to {cfg.filter_category}: {annotations_df.height}")

        # Save outputs
        output_dir = data_path / "pipeline_outputs" / "annotation"
        output_dir.mkdir(parents=True, exist_ok=True)

        annotations_path = output_dir / "annotations.parquet"
        annotations_df.write_parquet(annotations_path)
        logger.info(f"Saved annotations to {annotations_path}")

        # Save class mapping
        import json

        mapping_path = output_dir / "class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(
                {
                    "class_mapping": result.class_mapping,
                    "index_to_class": result.index_to_class,
                },
                f,
                indent=2,
            )

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,  # Pass through
                "annotations_parquet": str(annotations_path),
                "class_mapping": result.class_mapping,
                "index_to_class": result.index_to_class,
                "annotation_stats": result.stats,
                "annotator_used": cfg.annotator,
            },
        )

    def _load_cells_df(self, cells_path: str | None) -> pl.DataFrame | None:
        """Load cells DataFrame if available."""
        if not cells_path:
            return None

        path = Path(cells_path)
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        else:
            return pl.read_csv(path)

    def _load_expression(
        self,
        expression_path: Path,
        data_path: Path,
        platform: str,
    ):
        """Load expression data as AnnData."""
        import anndata as ad
        import scanpy as sc

        # Try direct load first based on file type
        if expression_path.suffix == ".h5ad":
            return ad.read_h5ad(expression_path)
        elif expression_path.suffix == ".h5":
            # 10x Genomics H5 format (Xenium, Visium)
            return sc.read_10x_h5(expression_path)
        elif expression_path.suffix == ".zarr" or expression_path.is_dir():
            if expression_path.suffix == ".zarr":
                return ad.read_zarr(expression_path)
            else:
                # Likely cell_feature_matrix directory (10x format)
                return self._load_10x_matrix(expression_path)

        # For CSV files (MERSCOPE), create AnnData
        if expression_path.suffix == ".csv":
            return self._load_csv_expression(expression_path, data_path, platform)

        raise ValueError(f"Unsupported expression format: {expression_path}")

    def _load_10x_matrix(self, matrix_dir: Path):
        """Load 10x Genomics format expression matrix."""
        import scanpy as sc

        return sc.read_10x_mtx(matrix_dir)

    def _load_csv_expression(
        self,
        expression_path: Path,
        data_path: Path,
        platform: str,
    ):
        """Load expression from CSV (MERSCOPE format)."""
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        # Read expression CSV
        expr_df = pd.read_csv(expression_path, index_col=0)

        # Create sparse matrix
        X = sp.csr_matrix(expr_df.values)

        # Create AnnData
        adata = ad.AnnData(X=X)
        adata.obs_names = expr_df.index.astype(str)
        adata.var_names = expr_df.columns

        # Add cell_id to obs
        adata.obs["cell_id"] = adata.obs_names

        return adata

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
        runner_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "clearml_step_runner.py"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,
            # Install dapidl from the cloned repo
            packages=["-e ."],
        )

        # Connect parameters for UI editing
        # step_name is used by clearml_step_runner.py to identify which step to run
        params = {
            "step_name": self.name,
            "annotator": self.config.annotator,
            "strategy": self.config.strategy,
            "model_names": ",".join(self.config.model_names),
            "confidence_threshold": self.config.confidence_threshold,
            "majority_voting": self.config.majority_voting,
            "fine_grained": self.config.fine_grained,
            "ground_truth_file": self.config.ground_truth_file or "",
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
