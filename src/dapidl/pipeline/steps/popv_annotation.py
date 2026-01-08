"""PopV Annotation Pipeline Step.

Step for universal tissue-agnostic cell type annotation using popV ensemble.

This step:
1. Runs full popV pipeline with 8+ annotation methods
2. Uses Tabula Sapiens reference with auto tissue detection
3. Applies ontology-based voting for consensus predictions
4. Filters by consensus score for high-confidence labels
5. Outputs Cell Ontology standardized annotations

PopV combines predictions from multiple methods:
- Random Forest, SVM, XGBoost (classical ML)
- CellTypist, OnClass (specialized cell type tools)
- scVI, scANVI, BBKNN, Scanorama, Harmony (integration-based)

References:
- Ergen et al. (2024) Nature Genetics
- https://github.com/YosefLab/popV
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import (
    PipelineStep,
    StepArtifacts,
    resolve_artifact_path,
)


@dataclass
class PopVAnnotationConfig:
    """Configuration for popV annotation step.

    Attributes:
        reference: Reference dataset ('tabula_sapiens' or path to custom)
        reference_organ: Organ for reference ('auto' to detect from markers)
        mode: Annotation mode ('fast', 'inference', 'retrain')
        min_consensus_score: Minimum methods agreeing (0-8, default 6 = ~90% accuracy)
        methods: Specific methods to use (None = all available)
        use_ontology_voting: Use Cell Ontology hierarchy for voting
        n_top_genes: Number of HVGs for processing
        fine_grained: Use detailed cell types vs broad categories
        include_method_predictions: Include per-method columns in output
        upload_to_s3: Upload results to S3
        s3_bucket: S3 bucket for uploads
    """

    # Reference selection
    reference: str = "tabula_sapiens"
    reference_organ: str = "auto"  # auto, Mammary, Lung, Liver, Heart, etc.
    custom_reference_path: str | None = None

    # Mode selection
    mode: str = "fast"  # fast (5min), inference (30min), retrain (1hr)

    # Consensus settings
    min_consensus_score: int = 6  # 6/8 = ~90% accuracy
    use_ontology_voting: bool = True

    # Method selection
    methods: list[str] | None = None  # None = all available

    # Processing settings
    n_top_genes: int = 2000
    batch_key: str | None = None

    # Output settings
    fine_grained: bool = True
    include_method_predictions: bool = True
    include_ontology_parent: bool = True

    # S3 settings
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"

    # ClearML/WandB settings
    use_wandb: bool = True
    use_clearml: bool = True
    project_name: str = "dapidl-popv"


class PopVAnnotationStep(PipelineStep):
    """Step for universal cell type annotation using popV ensemble.

    Runs the full popV pipeline with 8+ annotation methods and ontology-based
    voting for consensus predictions. Uses Tabula Sapiens as universal reference
    with automatic tissue detection.

    This is the recommended annotation method for generating universal training
    datasets because:
    - Uses unified reference (Tabula Sapiens) for all tissues
    - Consensus score provides confidence metric (7+ = 95% accuracy)
    - Ontology voting resolves disagreements using Cell Ontology hierarchy
    - Supports 20+ human organs without manual tuning

    Queue: gpu (scVI/scANVI benefit from GPU acceleration)
    """

    name = "popv_annotation"
    description = "Universal cell type annotation with popV ensemble"

    def __init__(self, config: PopVAnnotationConfig | None = None):
        """Initialize popV annotation step.

        Args:
            config: PopV annotation configuration
        """
        self.config = config or PopVAnnotationConfig()
        self._task = None
        self._wandb_run = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "string",
                    "default": "tabula_sapiens",
                    "description": "Reference dataset (tabula_sapiens or custom path)",
                },
                "reference_organ": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "Mammary",
                        "Lung",
                        "Liver",
                        "Heart",
                        "Kidney",
                        "Brain",
                        "Pancreas",
                        "Spleen",
                        "Small_Intestine",
                        "Large_Intestine",
                        "Bladder",
                        "Eye",
                        "Skin",
                        "Fat",
                        "Prostate",
                        "Uterus",
                        "Muscle",
                        "Bone_Marrow",
                        "Blood",
                        "Thymus",
                    ],
                    "default": "auto",
                    "description": "Organ-specific reference (auto-detected by default)",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fast", "inference", "retrain"],
                    "default": "fast",
                    "description": "Annotation mode: fast (5min), inference (30min), retrain (1hr)",
                },
                "min_consensus_score": {
                    "type": "integer",
                    "default": 6,
                    "minimum": 0,
                    "maximum": 8,
                    "description": "Minimum methods agreeing (6/8 = 90% accuracy, 7/8 = 95%)",
                },
                "use_ontology_voting": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use Cell Ontology hierarchy for voting",
                },
                "n_top_genes": {
                    "type": "integer",
                    "default": 2000,
                    "minimum": 500,
                    "maximum": 5000,
                    "description": "Number of highly variable genes",
                },
                "fine_grained": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use detailed cell types (vs broad categories)",
                },
                "include_method_predictions": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include per-method prediction columns",
                },
                "use_wandb": {
                    "type": "boolean",
                    "default": True,
                    "description": "Log metrics to Weights & Biases",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - expression_path: Path to expression data (H5, H5AD, CSV)
        - data_path: Path to spatial data directory
        """
        return "expression_path" in artifacts.outputs and "data_path" in artifacts.outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute popV annotation.

        Args:
            artifacts: Input artifacts from DataLoaderStep

        Returns:
            Output artifacts containing:
            - annotations_parquet: Path to annotation DataFrame
            - class_mapping: Dict mapping class names to indices
            - popv_stats: Dict with annotation statistics
            - consensus_distribution: Histogram of consensus scores
        """
        # Check popV availability
        try:
            from dapidl.pipeline.components.annotators.popv import (
                PopVAnnotator,
                PopVConfig,
                is_popv_available,
            )
        except ImportError:
            raise ImportError(
                "PopV annotator not available. Install with: pip install popv"
            )

        if not is_popv_available():
            raise ImportError(
                "popV package not installed. Install with: pip install popv"
            )

        cfg = self.config
        inputs = artifacts.outputs
        start_time = time.time()

        # Initialize tracking
        self._init_tracking(cfg)

        # Resolve artifact URLs to local paths
        data_path = resolve_artifact_path(inputs["data_path"], "data_path")
        expression_path = resolve_artifact_path(
            inputs.get("expression_path"), "expression_path"
        )

        if data_path is None:
            raise ValueError("data_path artifact is required")
        if expression_path is None:
            raise ValueError("expression_path artifact is required")

        # Get platform
        platform = self._resolve_platform(inputs)
        logger.info(f"Platform: {platform}")

        # Load expression data
        adata = self._load_expression(expression_path, data_path, platform)
        logger.info(f"Loaded expression data: {adata.n_obs} cells, {adata.n_vars} genes")

        # Log input stats
        self._log_input_stats(adata, cfg)

        # Create PopVConfig from step config
        popv_config = PopVConfig(
            reference=cfg.reference,
            reference_organ=cfg.reference_organ,
            custom_reference_path=cfg.custom_reference_path,
            methods=cfg.methods,
            mode=cfg.mode,
            min_consensus_score=cfg.min_consensus_score,
            use_ontology_voting=cfg.use_ontology_voting,
            n_top_genes=cfg.n_top_genes,
            batch_key=cfg.batch_key,
            include_method_predictions=cfg.include_method_predictions,
            include_ontology_parent=cfg.include_ontology_parent,
            fine_grained=cfg.fine_grained,
        )

        # Run popV annotation
        logger.info(f"Running popV annotation (mode={cfg.mode}, organ={cfg.reference_organ})")
        annotator = PopVAnnotator()
        result = annotator.annotate(config=popv_config, adata=adata)

        elapsed = time.time() - start_time
        logger.info(f"popV annotation complete in {elapsed:.1f}s: {result.stats}")

        # Filter by consensus score
        annotations_df = result.annotations_df
        if cfg.min_consensus_score > 0 and "consensus_score" in annotations_df.columns:
            before_count = annotations_df.height
            annotations_df = annotations_df.filter(
                pl.col("consensus_score") >= cfg.min_consensus_score
            )
            logger.info(
                f"Filtered by consensus >= {cfg.min_consensus_score}: "
                f"{before_count} -> {annotations_df.height}"
            )

        # Compute class mapping
        class_mapping = self._compute_class_mapping(annotations_df, cfg.fine_grained)
        index_to_class = {v: k for k, v in class_mapping.items()}

        # Save outputs
        output_dir = data_path / "pipeline_outputs" / "popv_annotation"
        output_dir.mkdir(parents=True, exist_ok=True)

        annotations_path = output_dir / "annotations.parquet"
        annotations_df.write_parquet(annotations_path)
        logger.info(f"Saved annotations to {annotations_path}")

        # Save class mapping
        mapping_path = output_dir / "class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(
                {
                    "class_mapping": class_mapping,
                    "index_to_class": index_to_class,
                },
                f,
                indent=2,
            )

        # Compute and save consensus distribution
        consensus_dist = None
        if "consensus_score" in annotations_df.columns:
            consensus_dist = (
                annotations_df.group_by("consensus_score")
                .agg(pl.len().alias("count"))
                .sort("consensus_score")
                .to_dict(as_series=False)
            )
            with open(output_dir / "consensus_distribution.json", "w") as f:
                json.dump(consensus_dist, f, indent=2)

        # Log output stats
        stats = {
            **result.stats,
            "elapsed_seconds": elapsed,
            "filtered_cells": annotations_df.height,
            "n_classes": len(class_mapping),
            "consensus_distribution": consensus_dist,
        }
        self._log_output_stats(stats, annotations_df, cfg)

        # Finalize tracking
        self._finalize_tracking()

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,  # Pass through
                "annotations_parquet": str(annotations_path),
                "class_mapping": class_mapping,
                "index_to_class": index_to_class,
                "popv_stats": stats,
                "annotator_used": "popv",
            },
        )

    def _resolve_platform(self, inputs: dict) -> str:
        """Resolve platform from inputs."""
        platform_value = inputs.get("platform", "xenium")
        platform_path = resolve_artifact_path(platform_value, "platform")
        if platform_path and platform_path.exists() and platform_path.is_file():
            return platform_path.read_text().strip()
        return str(platform_value)

    def _load_expression(
        self,
        expression_path: Path,
        data_path: Path,
        platform: str,
    ):
        """Load expression data as AnnData."""
        import anndata as ad
        import scanpy as sc

        if expression_path.suffix == ".h5ad":
            adata = ad.read_h5ad(expression_path)
        elif expression_path.suffix == ".h5":
            adata = sc.read_10x_h5(expression_path)
        elif expression_path.is_dir():
            adata = sc.read_10x_mtx(expression_path)
        elif expression_path.suffix == ".csv":
            adata = self._load_csv_expression(expression_path)
        else:
            raise ValueError(f"Unsupported expression format: {expression_path}")

        # Ensure cell_id in obs
        if "cell_id" not in adata.obs.columns:
            adata.obs["cell_id"] = adata.obs.index.astype(str)

        return adata

    def _load_csv_expression(self, expression_path: Path):
        """Load expression from CSV (MERSCOPE format)."""
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        expr_df = pd.read_csv(expression_path, index_col=0)
        X = sp.csr_matrix(expr_df.values)

        adata = ad.AnnData(X=X)
        adata.obs_names = expr_df.index.astype(str)
        adata.var_names = expr_df.columns
        adata.obs["cell_id"] = adata.obs_names

        return adata

    def _compute_class_mapping(
        self, annotations_df: pl.DataFrame, fine_grained: bool
    ) -> dict[str, int]:
        """Compute class mapping from annotations DataFrame."""
        if fine_grained:
            col = "predicted_type"
        else:
            col = "broad_category"

        if col not in annotations_df.columns:
            # Fallback
            col = "predicted_type" if "predicted_type" in annotations_df.columns else "broad_category"

        cell_types = sorted(annotations_df[col].unique().to_list())
        # Filter out Unknown if present, add at end
        if "Unknown" in cell_types:
            cell_types.remove("Unknown")
            cell_types.append("Unknown")

        return {ct: i for i, ct in enumerate(cell_types)}

    def _init_tracking(self, cfg: PopVAnnotationConfig) -> None:
        """Initialize ClearML and WandB tracking."""
        # ClearML
        if cfg.use_clearml:
            try:
                from clearml import Task

                task = Task.current_task()
                if task is None:
                    task = Task.init(
                        project_name=cfg.project_name,
                        task_name="popv_annotation",
                        task_type=Task.TaskTypes.data_processing,
                    )
                self._task = task

                # Log config
                task.connect(
                    {
                        "reference": cfg.reference,
                        "reference_organ": cfg.reference_organ,
                        "mode": cfg.mode,
                        "min_consensus_score": cfg.min_consensus_score,
                        "use_ontology_voting": cfg.use_ontology_voting,
                        "n_top_genes": cfg.n_top_genes,
                        "fine_grained": cfg.fine_grained,
                    },
                    name="popv_config",
                )
            except ImportError:
                logger.debug("ClearML not available")
            except Exception as e:
                logger.warning(f"ClearML init failed: {e}")

        # WandB
        if cfg.use_wandb:
            try:
                import wandb

                if wandb.run is None:
                    wandb.init(
                        project=cfg.project_name,
                        name="popv_annotation",
                        config={
                            "reference": cfg.reference,
                            "reference_organ": cfg.reference_organ,
                            "mode": cfg.mode,
                            "min_consensus_score": cfg.min_consensus_score,
                            "use_ontology_voting": cfg.use_ontology_voting,
                            "n_top_genes": cfg.n_top_genes,
                            "fine_grained": cfg.fine_grained,
                        },
                    )
                self._wandb_run = wandb.run
            except ImportError:
                logger.debug("WandB not available")
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")

    def _log_input_stats(self, adata, cfg: PopVAnnotationConfig) -> None:
        """Log input statistics."""
        stats = {
            "input_cells": adata.n_obs,
            "input_genes": adata.n_vars,
        }

        if self._task:
            try:
                self._task.get_logger().report_single_value("input_cells", adata.n_obs)
                self._task.get_logger().report_single_value("input_genes", adata.n_vars)
            except Exception:
                pass

        if self._wandb_run:
            try:
                import wandb

                wandb.log(stats)
            except Exception:
                pass

    def _log_output_stats(
        self,
        stats: dict,
        annotations_df: pl.DataFrame,
        cfg: PopVAnnotationConfig,
    ) -> None:
        """Log output statistics."""
        metrics = {
            "popv/n_cells": stats.get("n_cells", 0),
            "popv/filtered_cells": stats.get("filtered_cells", 0),
            "popv/n_classes": stats.get("n_classes", 0),
            "popv/elapsed_seconds": stats.get("elapsed_seconds", 0),
        }

        # Add consensus score stats if available
        if "consensus_score" in annotations_df.columns:
            consensus_scores = annotations_df["consensus_score"].to_list()
            if consensus_scores:
                import numpy as np

                metrics["popv/consensus_mean"] = float(np.mean(consensus_scores))
                metrics["popv/consensus_std"] = float(np.std(consensus_scores))
                metrics["popv/high_confidence_rate"] = sum(
                    1 for s in consensus_scores if s >= 7
                ) / len(consensus_scores)

        # Log to ClearML
        if self._task:
            try:
                logger_obj = self._task.get_logger()
                for key, value in metrics.items():
                    logger_obj.report_single_value(key.replace("/", "_"), value)

                # Log consensus distribution histogram
                if "consensus_score" in annotations_df.columns:
                    dist = (
                        annotations_df.group_by("consensus_score")
                        .agg(pl.len().alias("count"))
                        .sort("consensus_score")
                    )
                    logger_obj.report_histogram(
                        "Consensus Score Distribution",
                        "consensus_score",
                        iteration=0,
                        values=dist["consensus_score"].to_list(),
                        xlabels=[str(x) for x in dist["consensus_score"].to_list()],
                    )
            except Exception as e:
                logger.debug(f"ClearML logging failed: {e}")

        # Log to WandB
        if self._wandb_run:
            try:
                import wandb

                wandb.log(metrics)

                # Log consensus histogram
                if "consensus_score" in annotations_df.columns:
                    consensus_scores = annotations_df["consensus_score"].to_list()
                    wandb.log(
                        {
                            "popv/consensus_histogram": wandb.Histogram(consensus_scores),
                        }
                    )

                # Log class distribution
                if "predicted_type" in annotations_df.columns:
                    class_counts = (
                        annotations_df.group_by("predicted_type")
                        .agg(pl.len().alias("count"))
                        .sort("count", descending=True)
                    )
                    wandb.log(
                        {
                            "popv/class_distribution": wandb.Table(
                                dataframe=class_counts.to_pandas()
                            ),
                        }
                    )
            except Exception as e:
                logger.debug(f"WandB logging failed: {e}")

    def _finalize_tracking(self) -> None:
        """Finalize tracking (close runs if we created them)."""
        # Don't close runs - let the pipeline controller handle that
        pass

    def get_queue(self) -> str:
        """Return queue name for this step.

        Returns 'gpu' because scVI/scANVI methods benefit from GPU acceleration.
        """
        return "gpu"

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from pathlib import Path

        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        # Use the runner script for remote execution
        runner_script = (
            Path(__file__).parent.parent.parent.parent.parent
            / "scripts"
            / "clearml_step_runner.py"
        )

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
            script=str(runner_script),
            argparse_args=[f"--step={self.name}"],
            add_task_init_call=False,
            packages=["-e ."],
        )

        # Connect parameters for UI editing
        params = {
            "step_name": self.name,
            "reference": self.config.reference,
            "reference_organ": self.config.reference_organ,
            "mode": self.config.mode,
            "min_consensus_score": self.config.min_consensus_score,
            "use_ontology_voting": self.config.use_ontology_voting,
            "n_top_genes": self.config.n_top_genes,
            "fine_grained": self.config.fine_grained,
            "include_method_predictions": self.config.include_method_predictions,
        }
        self._task.set_parameters(params, __parameters_prefix="popv_config")

        return self._task
