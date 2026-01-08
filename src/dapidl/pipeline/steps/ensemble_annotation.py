"""Ensemble Annotation Pipeline Step.

Step 2: Cell type annotation using ensemble of multiple methods.

This step:
1. Runs multiple CellTypist models
2. Optionally runs SingleR (R-based reference mapping)
3. Builds consensus across all methods
4. Creates derived annotated dataset with ClearML lineage
5. Uploads to S3 if configured

The ensemble approach improves annotation quality by:
- Reducing model-specific biases
- Providing confidence through agreement
- Leveraging different reference datasets
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.base import (
    AnnotationConfig,
    PipelineStep,
    StepArtifacts,
    resolve_artifact_path,
)


@dataclass
class EnsembleAnnotationConfig:
    """Configuration for ensemble annotation step."""

    # CellTypist models (multi-select in GUI)
    celltypist_models: list[str] = field(
        default_factory=lambda: [
            "Cells_Adult_Breast.pkl",
            "Immune_All_High.pkl",
        ]
    )

    # Additional annotators
    include_singler: bool = True
    singler_reference: str = "blueprint"  # "blueprint", "hpca", "monaco"
    include_sctype: bool = False

    # Consensus settings
    min_agreement: int = 2  # Minimum annotators agreeing
    confidence_threshold: float = 0.5
    use_confidence_weighting: bool = True

    # Output settings
    fine_grained: bool = True  # Use detailed cell types
    create_derived_dataset: bool = True
    parent_dataset_id: str | None = None  # For lineage

    # S3 settings
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"


class EnsembleAnnotationStep(PipelineStep):
    """Step 2: Cell Type Prediction with Ensemble Annotation.

    Runs multiple annotation methods and creates consensus:
    1. CellTypist (multiple models)
    2. SingleR (R-based reference mapping)
    3. scType (marker-based, optional)

    Creates a derived annotated dataset and registers it with ClearML.
    Uses parent_datasets for lineage to avoid re-uploading raw data.

    Queue: default (CPU-bound - CellTypist and R are CPU-only)
    """

    name = "ensemble_annotation"
    description = "Ensemble cell type annotation with multiple methods"

    def __init__(self, config: EnsembleAnnotationConfig | None = None):
        """Initialize ensemble annotation step.

        Args:
            config: Ensemble annotation configuration
        """
        self.config = config or EnsembleAnnotationConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "celltypist_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
                    "description": "CellTypist model names (comma-separated)",
                },
                "include_singler": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include SingleR annotation (requires R)",
                },
                "singler_reference": {
                    "type": "string",
                    "enum": ["blueprint", "hpca", "monaco"],
                    "default": "blueprint",
                    "description": "SingleR reference dataset",
                },
                "min_agreement": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 1,
                    "description": "Minimum annotators that must agree",
                },
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum consensus confidence",
                },
                "fine_grained": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use detailed cell types (vs 3 broad categories)",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate step inputs.

        Requires:
        - expression_path: Path to expression data (H5, H5AD, CSV)
        - data_path: Path to Xenium/MERSCOPE output directory
        """
        return "expression_path" in artifacts.outputs and "data_path" in artifacts.outputs

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Execute ensemble annotation.

        Args:
            artifacts: Input artifacts from DataLoaderStep

        Returns:
            Output artifacts containing:
            - annotations_parquet: Path to consensus annotation DataFrame
            - class_mapping: Dict mapping class names to indices
            - annotated_dataset_id: ClearML Dataset ID (if registered)
            - annotation_stats: Dict with annotation statistics
        """
        cfg = self.config
        inputs = artifacts.outputs

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

        # Load expression data
        adata = self._load_expression(expression_path, data_path, platform)
        logger.info(f"Loaded expression data: {adata.n_obs} cells, {adata.n_vars} genes")

        # Run annotation methods
        all_predictions = []

        # 1. Run CellTypist models
        for model_name in cfg.celltypist_models:
            try:
                result = self._run_celltypist(adata, model_name)
                all_predictions.append(result)
                logger.info(f"CellTypist {model_name}: {len(result['predictions'])} cells")
            except Exception as e:
                logger.warning(f"CellTypist {model_name} failed: {e}")

        # 2. Run SingleR if enabled
        if cfg.include_singler:
            try:
                result = self._run_singler(adata, cfg.singler_reference, data_path)
                all_predictions.append(result)
                logger.info(f"SingleR {cfg.singler_reference}: {len(result['predictions'])} cells")
            except Exception as e:
                logger.warning(f"SingleR failed: {e}")

        # 3. Run scType if enabled
        if cfg.include_sctype:
            try:
                result = self._run_sctype(adata)
                all_predictions.append(result)
                logger.info(f"scType: {len(result['predictions'])} cells")
            except Exception as e:
                logger.warning(f"scType failed: {e}")

        if not all_predictions:
            raise ValueError("No annotation methods succeeded")

        # Build consensus
        consensus_df, stats = self._build_consensus(
            all_predictions,
            adata.obs.index.tolist(),
            cfg,
        )
        logger.info(f"Consensus built: {len(consensus_df)} cells with agreement")

        # Compute class mapping
        class_mapping = self._compute_class_mapping(consensus_df, cfg.fine_grained)
        index_to_class = {v: k for k, v in class_mapping.items()}

        # Save outputs
        output_dir = data_path / "pipeline_outputs" / "ensemble_annotation"
        output_dir.mkdir(parents=True, exist_ok=True)

        annotations_path = output_dir / "annotations.parquet"
        consensus_df.write_parquet(annotations_path)
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

        # Create derived dataset with lineage
        dataset_id = None
        if cfg.create_derived_dataset:
            try:
                dataset_id = self._create_annotated_dataset(
                    output_dir, inputs, consensus_df, stats, cfg
                )
                logger.info(f"Created annotated dataset: {dataset_id}")
            except Exception as e:
                logger.warning(f"Failed to create ClearML dataset: {e}")

        return StepArtifacts(
            inputs=inputs,
            outputs={
                **inputs,  # Pass through
                "annotations_parquet": str(annotations_path),
                "class_mapping": class_mapping,
                "index_to_class": index_to_class,
                "annotated_dataset_id": dataset_id,
                "annotation_stats": stats,
                "annotation_methods": [p["source"] for p in all_predictions],
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

    def _run_celltypist(self, adata, model_name: str) -> dict:
        """Run CellTypist annotation."""
        import celltypist
        from celltypist import models

        # Download model if needed
        models.download_models(force_update=False, model=model_name)
        model = models.Model.load(model=model_name)

        # Normalize data (CellTypist expects normalized)
        import scanpy as sc

        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Predict
        predictions = celltypist.annotate(
            adata_norm,
            model=model,
            majority_voting=True,
        )

        # Extract results
        labels = predictions.predicted_labels.predicted_labels.tolist()
        confidence = predictions.probability_matrix.max(axis=1).tolist()

        return {
            "source": f"celltypist_{model_name.replace('.pkl', '')}",
            "predictions": labels,
            "confidence": confidence,
            "cell_ids": adata.obs.index.tolist(),
        }

    def _run_singler(self, adata, reference: str, data_path: Path) -> dict:
        """Run SingleR annotation (requires R).

        This uses rpy2 to call SingleR from Python.
        """
        # Check if R is available
        import shutil

        if not shutil.which("Rscript"):
            raise RuntimeError("R is not installed or not in PATH")

        # Save expression to temp file for R
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        expr_path = temp_dir / "expression.h5ad"
        adata.write_h5ad(expr_path)

        # Run SingleR via R script
        r_script = self._get_singler_script()
        r_script_path = temp_dir / "run_singler.R"
        with open(r_script_path, "w") as f:
            f.write(r_script)

        import subprocess

        result = subprocess.run(
            ["Rscript", str(r_script_path), str(expr_path), reference, str(temp_dir)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SingleR failed: {result.stderr}")

        # Load results
        results_path = temp_dir / "singler_results.csv"
        results_df = pl.read_csv(results_path)

        # Clean up
        import shutil as sh

        sh.rmtree(temp_dir)

        # Convert cell_ids to strings to match CellTypist format
        # (pandas reads numeric strings from CSV as integers)
        cell_ids = [str(cid) for cid in results_df["cell_id"].to_list()]

        return {
            "source": f"singler_{reference}",
            "predictions": results_df["labels"].to_list(),
            "confidence": results_df["delta.next"].to_list(),
            "cell_ids": cell_ids,
        }

    def _get_singler_script(self) -> str:
        """Return R script for SingleR annotation."""
        return '''
library(SingleR)
library(celldex)
library(anndata)

args <- commandArgs(trailingOnly = TRUE)
expr_path <- args[1]
reference_name <- args[2]
output_dir <- args[3]

# Load expression data
adata <- read_h5ad(expr_path)
counts <- t(adata$X)
colnames(counts) <- rownames(adata$obs)

# Load reference
ref <- switch(reference_name,
    "blueprint" = BlueprintEncodeData(),
    "hpca" = HumanPrimaryCellAtlasData(),
    "monaco" = MonacoImmuneData(),
    stop("Unknown reference")
)

# Run SingleR
results <- SingleR(
    test = counts,
    ref = ref,
    labels = ref$label.main,
    de.method = "wilcox"
)

# Save results
output <- data.frame(
    cell_id = rownames(results),
    labels = results$labels,
    delta.next = results$delta.next
)
write.csv(output, file.path(output_dir, "singler_results.csv"), row.names = FALSE)
'''

    def _run_sctype(self, adata) -> dict:
        """Run scType annotation (marker-based)."""
        # scType implementation would go here
        # For now, raise not implemented
        raise NotImplementedError("scType annotation not yet implemented")

    def _build_consensus(
        self,
        all_predictions: list[dict],
        cell_ids: list[str],
        cfg: EnsembleAnnotationConfig,
    ) -> tuple[pl.DataFrame, dict]:
        """Build consensus from multiple annotation methods.

        Uses confidence-weighted voting when available.
        For single-annotator cases, uses predictions directly.
        """
        from collections import defaultdict

        # Adjust min_agreement if we have fewer annotators than required
        num_annotators = len(all_predictions)
        effective_min_agreement = min(cfg.min_agreement, num_annotators)
        if num_annotators == 1:
            logger.info("Single annotator mode: using predictions directly")
            effective_min_agreement = 1

        # Create cell_id to prediction mapping
        cell_votes = defaultdict(list)

        for pred in all_predictions:
            pred_dict = dict(zip(pred["cell_ids"], pred["predictions"]))
            conf_dict = dict(zip(pred["cell_ids"], pred.get("confidence", [1.0] * len(pred["cell_ids"]))))

            for cell_id in cell_ids:
                if cell_id in pred_dict:
                    cell_votes[cell_id].append({
                        "source": pred["source"],
                        "label": pred_dict[cell_id],
                        "confidence": conf_dict[cell_id],
                    })

        # Build consensus
        consensus_records = []
        method_agreements = defaultdict(int)
        n_unanimous = 0
        n_majority = 0
        n_insufficient = 0

        for cell_id, votes in cell_votes.items():
            if len(votes) < effective_min_agreement:
                n_insufficient += 1
                continue

            # Standardize labels to broad categories for voting
            vote_labels = [self._standardize_label(v["label"]) for v in votes]
            vote_confidences = [v["confidence"] for v in votes]

            # Weighted voting
            label_scores = defaultdict(float)
            for label, conf in zip(vote_labels, vote_confidences):
                if cfg.use_confidence_weighting:
                    label_scores[label] += conf
                else:
                    label_scores[label] += 1.0

            # Get winner
            if not label_scores:
                continue

            sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])
            winner_label = sorted_labels[0][0]
            winner_score = sorted_labels[0][1]
            total_score = sum(label_scores.values())

            # Calculate consensus confidence
            consensus_confidence = winner_score / total_score if total_score > 0 else 0

            if consensus_confidence < cfg.confidence_threshold:
                continue

            # Count agreement type
            unique_labels = set(vote_labels)
            if len(unique_labels) == 1:
                n_unanimous += 1
            else:
                n_majority += 1

            # Track which methods agreed
            for vote in votes:
                if self._standardize_label(vote["label"]) == winner_label:
                    method_agreements[vote["source"]] += 1

            # Get original fine-grained label from winner method
            winner_vote = next(
                (v for v in votes if self._standardize_label(v["label"]) == winner_label),
                votes[0]
            )

            consensus_records.append({
                "cell_id": cell_id,
                "predicted_type": winner_vote["label"],  # Original fine-grained
                "broad_category": winner_label,  # Standardized
                "confidence": consensus_confidence,
                "n_votes": len(votes),
                "n_agreement": sum(1 for l in vote_labels if l == winner_label),
            })

        # Create DataFrame
        consensus_df = pl.DataFrame(consensus_records)

        stats = {
            "total_cells": len(cell_ids),
            "annotated_cells": len(consensus_records),
            "unanimous_agreement": n_unanimous,
            "majority_agreement": n_majority,
            "insufficient_votes": n_insufficient,
            "methods_used": [p["source"] for p in all_predictions],
            "method_agreements": dict(method_agreements),
        }

        return consensus_df, stats

    def _standardize_label(self, label: str) -> str:
        """Standardize cell type label to broad category.

        Maps fine-grained cell types to: Epithelial, Immune, Stromal, Endothelial, Other
        """
        label_lower = label.lower()

        # Epithelial markers
        if any(marker in label_lower for marker in [
            "epithelial", "luminal", "basal", "keratinocyte", "secretory",
            "ductal", "acinar", "alveolar", "squamous", "glandular",
            "mammary", "breast", "tumor"
        ]):
            return "Epithelial"

        # Immune markers
        if any(marker in label_lower for marker in [
            "t cell", "t_cell", "cd4", "cd8", "nk", "natural killer",
            "b cell", "b_cell", "plasma", "myeloid", "monocyte", "macrophage",
            "dendritic", "dc", "neutrophil", "granulocyte", "mast",
            "lymphocyte", "immune", "leukocyte"
        ]):
            return "Immune"

        # Stromal markers
        if any(marker in label_lower for marker in [
            "fibroblast", "stromal", "mesenchymal", "adipocyte", "fat",
            "myofibroblast", "pericyte", "smooth muscle", "caf"
        ]):
            return "Stromal"

        # Endothelial markers
        if any(marker in label_lower for marker in [
            "endothelial", "vascular", "blood vessel", "lymphatic"
        ]):
            return "Endothelial"

        return "Other"

    def _compute_class_mapping(
        self, annotations_df: pl.DataFrame, fine_grained: bool
    ) -> dict[str, int]:
        """Compute class mapping from annotations DataFrame."""
        if fine_grained:
            col = "predicted_type"
        else:
            col = "broad_category"

        cell_types = sorted(annotations_df[col].unique().to_list())
        # Filter out Unknown if present, add at end
        if "Unknown" in cell_types:
            cell_types.remove("Unknown")
            cell_types.append("Unknown")

        return {ct: i for i, ct in enumerate(cell_types)}

    def _create_annotated_dataset(
        self,
        output_dir: Path,
        inputs: dict,
        consensus_df: pl.DataFrame,
        stats: dict,
        cfg: EnsembleAnnotationConfig,
    ) -> str | None:
        """Create ClearML Dataset with lineage to raw data.

        Uses parent_datasets to avoid re-uploading raw data.
        """
        try:
            from clearml import Dataset
        except ImportError:
            logger.warning("ClearML not available, skipping dataset creation")
            return None

        # Determine parent dataset
        parent_ids = []
        if cfg.parent_dataset_id:
            parent_ids.append(cfg.parent_dataset_id)
        elif inputs.get("raw_dataset_id"):
            parent_ids.append(inputs["raw_dataset_id"])

        # Determine output URI
        output_uri = None
        if cfg.upload_to_s3:
            output_uri = f"s3://{cfg.s3_bucket}/datasets/annotated/"

        # Create dataset with lineage
        platform = inputs.get("platform", "unknown")
        dataset_name = f"annotated-{platform}-ensemble-{len(stats['methods_used'])}m"

        try:
            dataset = Dataset.create(
                dataset_project="DAPIDL/annotated",
                dataset_name=dataset_name,
                parent_datasets=parent_ids if parent_ids else None,
                output_uri=output_uri,
            )

            # Add only new annotation files (parent has raw data)
            dataset.add_files(str(output_dir / "annotations.parquet"))
            dataset.add_files(str(output_dir / "class_mapping.json"))

            # Metadata
            dataset.set_metadata({
                "annotation_methods": stats["methods_used"],
                "n_cells": len(consensus_df),
                "n_classes": len(consensus_df["predicted_type"].unique()),
                "parent_dataset": cfg.parent_dataset_id,
                "platform": platform,
                "unanimous_agreement_pct": stats["unanimous_agreement"] / stats["annotated_cells"]
                if stats["annotated_cells"] > 0
                else 0,
            })

            dataset.finalize()
            dataset.upload()

            logger.info(f"Created ClearML dataset: {dataset.id}")
            return dataset.id

        except Exception as e:
            logger.warning(f"Failed to create ClearML dataset: {e}")
            return None

    def get_queue(self) -> str:
        """Return queue name for this step."""
        return "default"  # CPU queue for CellTypist/SingleR

    def create_clearml_task(
        self,
        project: str = "DAPIDL/pipeline",
        task_name: str | None = None,
    ):
        """Create ClearML Task for this step."""
        from clearml import Task

        task_name = task_name or f"step-{self.name}"

        self._task = Task.create(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
        )

        # Connect parameters
        params = {
            "step_name": self.name,
            "celltypist_models": ",".join(self.config.celltypist_models),
            "include_singler": self.config.include_singler,
            "singler_reference": self.config.singler_reference,
            "min_agreement": self.config.min_agreement,
            "confidence_threshold": self.config.confidence_threshold,
            "fine_grained": self.config.fine_grained,
        }
        self._task.set_parameters(params, __parameters_prefix="step_config")

        return self._task
