"""PopV-style Ensemble Annotator with Cell Ontology-based Voting.

This module implements an ensemble annotation approach inspired by popV
(popularity voting) that combines multiple annotation methods:

1. **Unweighted voting**: Each method gets exactly 1 vote (like popV)
2. **Cell Ontology hierarchical voting**: Methods don't need exact match to agree
3. **Depth-based tie-breaking**: More specific (deeper) predictions preferred
4. **Three granularity levels**: Coarse (3), Medium (~10), Fine (50+)

References:
    - popV: https://github.com/YosefLab/popV
    - Cell Ontology: https://obofoundry.org/ontology/cl.html
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.components.annotators.cell_ontology_mapping import (
    ANNOTATOR_TO_CL,
)
from dapidl.pipeline.components.annotators.mapping import (
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


class GranularityLevel(str, Enum):
    """Annotation granularity levels."""

    COARSE = "coarse"  # 3-4 classes: Epithelial, Immune, Stromal, (Endothelial)
    MEDIUM = "medium"  # ~10 classes: Major subtypes
    FINE = "fine"  # 50+ classes: Full resolution


class VotingStrategy(str, Enum):
    """Voting strategies for ensemble."""

    UNWEIGHTED = "unweighted"  # Each method = 1 vote (popV style)
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence
    ONTOLOGY_HIERARCHICAL = "ontology_hierarchical"  # Cell Ontology-based


# Medium granularity classes (~10 types for breast tissue)
MEDIUM_CLASS_NAMES = [
    "Epithelial_Luminal",
    "Epithelial_Basal",
    "Epithelial_Tumor",
    "T_Cell",
    "B_Cell",
    "Myeloid",
    "NK_Cell",
    "Stromal_Fibroblast",
    "Stromal_Pericyte",
    "Endothelial",
]

# Mapping from fine-grained to medium-grained
FINE_TO_MEDIUM_MAPPING = {
    # Epithelial subtypes
    "LummHR": "Epithelial_Luminal",
    "LummHR-SCGB": "Epithelial_Luminal",
    "LummHR-active": "Epithelial_Luminal",
    "LummHR-major": "Epithelial_Luminal",
    "Lumsec": "Epithelial_Luminal",
    "Lumsec-HLA": "Epithelial_Luminal",
    "Lumsec-KIT": "Epithelial_Luminal",
    "Lumsec-basal": "Epithelial_Basal",
    "Lumsec-lac": "Epithelial_Luminal",
    "Lumsec-major": "Epithelial_Luminal",
    "Lumsec-myo": "Epithelial_Basal",
    "Lumsec-prol": "Epithelial_Tumor",
    "basal": "Epithelial_Basal",
    "luminal epithelial cell of mammary gland": "Epithelial_Luminal",
    "myoepithelial cell": "Epithelial_Basal",
    "epithelial cell": "Epithelial_Luminal",
    # Ground truth epithelial
    "DCIS_1": "Epithelial_Tumor",
    "DCIS_2": "Epithelial_Tumor",
    "Invasive_Tumor": "Epithelial_Tumor",
    "Prolif_Invasive_Tumor": "Epithelial_Tumor",
    "Myoepi_ACTA2+": "Epithelial_Basal",
    "Myoepi_KRT15+": "Epithelial_Basal",
    # T cells
    "CD4": "T_Cell",
    "CD8": "T_Cell",
    "T_prol": "T_Cell",
    "GD": "T_Cell",
    "NKT": "T_Cell",
    "Treg": "T_Cell",
    "CD4+_T_Cells": "T_Cell",
    "CD8+_T_Cells": "T_Cell",
    "T cell": "T_Cell",
    "CD4-positive, alpha-beta T cell": "T_Cell",
    "CD8-positive, alpha-beta T cell": "T_Cell",
    "regulatory T cell": "T_Cell",
    # B cells
    "b_naive": "B_Cell",
    "bmem_switched": "B_Cell",
    "bmem_unswitched": "B_Cell",
    "plasma": "B_Cell",
    "plasma_IgA": "B_Cell",
    "plasma_IgG": "B_Cell",
    "B_Cells": "B_Cell",
    "B cell": "B_Cell",
    "plasma cell": "B_Cell",
    # Myeloid
    "Macro": "Myeloid",
    "Mono": "Myeloid",
    "cDC": "Myeloid",
    "mDC": "Myeloid",
    "pDC": "Myeloid",
    "mye-prol": "Myeloid",
    "Macrophages_1": "Myeloid",
    "Macrophages_2": "Myeloid",
    "IRF7+_DCs": "Myeloid",
    "LAMP3+_DCs": "Myeloid",
    "Mast_Cells": "Myeloid",
    "Mast": "Myeloid",
    "Neutrophil": "Myeloid",
    "macrophage": "Myeloid",
    "monocyte": "Myeloid",
    "dendritic cell": "Myeloid",
    "mast cell": "Myeloid",
    # NK cells
    "NK": "NK_Cell",
    "NK-ILCs": "NK_Cell",
    "natural killer cell": "NK_Cell",
    # Stromal
    "Fibro": "Stromal_Fibroblast",
    "Fibro-SFRP4": "Stromal_Fibroblast",
    "Fibro-major": "Stromal_Fibroblast",
    "Fibro-matrix": "Stromal_Fibroblast",
    "Fibro-prematrix": "Stromal_Fibroblast",
    "Fibroblast": "Stromal_Fibroblast",
    "Stromal": "Stromal_Fibroblast",
    "fibroblast": "Stromal_Fibroblast",
    "pericytes": "Stromal_Pericyte",
    "Pericyte": "Stromal_Pericyte",
    "pericyte": "Stromal_Pericyte",
    "Perivascular-Like": "Stromal_Pericyte",
    "vsmc": "Stromal_Pericyte",
    "smooth muscle cell": "Stromal_Pericyte",
    # Endothelial
    "Vas": "Endothelial",
    "Vas-arterial": "Endothelial",
    "Vas-capillary": "Endothelial",
    "Vas-venous": "Endothelial",
    "Endothelial": "Endothelial",
    "Lymph": "Endothelial",
    "Lymph-immune": "Endothelial",
    "Lymph-major": "Endothelial",
    "endothelial cell": "Endothelial",
}


# Cell Ontology hierarchy for tie-breaking (depth from root)
# Higher depth = more specific = preferred in ties
CL_DEPTH = {
    # Root level (depth 1)
    "CL:0000000": 1,  # cell
    # Major categories (depth 2)
    "CL:0000066": 2,  # epithelial cell
    "CL:0000084": 2,  # T cell
    "CL:0000236": 2,  # B cell
    "CL:0000235": 2,  # macrophage
    "CL:0000057": 2,  # fibroblast
    "CL:0000115": 2,  # endothelial cell
    # Subcategories (depth 3)
    "CL:0002327": 3,  # mammary gland epithelial cell
    "CL:0000624": 3,  # CD4+ T cell
    "CL:0000625": 3,  # CD8+ T cell
    "CL:0000786": 3,  # plasma cell
    "CL:0000451": 3,  # dendritic cell
    "CL:0000576": 3,  # monocyte
    "CL:0000669": 3,  # pericyte
    "CL:0002138": 3,  # lymphatic endothelial
    # Fine-grained (depth 4)
    "CL:0002325": 4,  # mammary luminal epithelial
    "CL:0000815": 4,  # regulatory T cell
    "CL:0000097": 3,  # mast cell
    "CL:0000623": 3,  # NK cell
    "CL:0000646": 3,  # basal cell
    "CL:0000185": 3,  # myoepithelial cell
    "CL:0000136": 3,  # adipocyte
    "CL:0000192": 3,  # smooth muscle cell
    "CL:0000186": 3,  # myofibroblast
}


@dataclass
class PopVEnsembleConfig:
    """Configuration for PopV-style ensemble annotation."""

    # Methods to include
    celltypist_models: list[str] = field(
        default_factory=lambda: [
            "Cells_Adult_Breast.pkl",
            "Immune_All_High.pkl",
        ]
    )
    include_singler_hpca: bool = True
    include_singler_blueprint: bool = True
    include_sctype: bool = False

    # Voting configuration
    voting_strategy: VotingStrategy = VotingStrategy.ONTOLOGY_HIERARCHICAL
    min_agreement: int = 2  # Minimum methods that must agree
    confidence_threshold: float = 0.0  # For filtering low-confidence predictions

    # Granularity
    granularity: GranularityLevel = GranularityLevel.COARSE

    # Output settings
    include_per_method_predictions: bool = True
    include_voting_details: bool = True

    # ClearML/WandB tracking
    log_to_clearml: bool = True
    log_to_wandb: bool = True
    experiment_name: str = "popv_ensemble"


@dataclass
class VotingResult:
    """Result of voting for a single cell."""

    cell_id: str
    winner: str  # Winning prediction
    winner_cl_id: str | None  # Cell Ontology ID if mapped
    broad_category: str  # Coarse category
    medium_category: str  # Medium granularity
    n_votes: int  # Total votes cast
    n_agreement: int  # Votes for winner
    agreement_ratio: float  # n_agreement / n_votes
    is_unanimous: bool
    all_votes: list[str]  # All method predictions
    confidence_scores: list[float]  # Per-method confidences
    voting_strategy: str
    ontology_depth: int  # Winner's CL depth
    per_method_predictions: dict[str, str] = field(default_factory=dict)  # source -> prediction
    per_method_confidences: dict[str, float] = field(default_factory=dict)  # source -> confidence


@dataclass
class EnsembleResult:
    """Complete ensemble annotation result."""

    annotations_df: pl.DataFrame
    voting_stats: dict[str, Any]
    per_method_metrics: dict[str, dict[str, float]]
    class_distribution: dict[str, int]
    agreement_matrix: np.ndarray | None = None


@register_annotator
class PopVStyleEnsembleAnnotator:
    """Ensemble annotator using popV-style voting mechanisms.

    Key features:
    1. Unweighted voting: Each method = 1 vote (default)
    2. Cell Ontology hierarchical voting: Methods agree if predictions
       share a common ancestor in the Cell Ontology
    3. Depth-based tie-breaking: More specific predictions preferred
    4. Three granularity levels: coarse, medium, fine
    5. Per-method predictions: Individual annotator results exposed

    Example:
        >>> config = PopVEnsembleConfig(
        ...     celltypist_models=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        ...     voting_strategy=VotingStrategy.ONTOLOGY_HIERARCHICAL,
        ...     granularity=GranularityLevel.MEDIUM,
        ... )
        >>> annotator = PopVStyleEnsembleAnnotator(config)
        >>> result = annotator.annotate(adata)
        >>> # Access per-method predictions
        >>> result.annotations_df["pred_celltypist_Cells_Adult_Breast"]
    """

    name = "popv_ensemble"

    def __init__(self, config: PopVEnsembleConfig | None = None):
        """Initialize ensemble annotator.

        Args:
            config: Configuration for the ensemble
        """
        self.config = config or PopVEnsembleConfig()
        self._predictions_cache: dict[str, dict] = {}
        self._method_sources: list[str] = []  # Track which methods were used

    def annotate(self, adata) -> EnsembleResult:
        """Run ensemble annotation on AnnData object.

        Args:
            adata: AnnData with expression data

        Returns:
            EnsembleResult with consensus annotations and statistics
        """
        cfg = self.config
        logger.info(
            f"PopV-style ensemble annotation: {len(cfg.celltypist_models)} CellTypist models, "
            f"HPCA={cfg.include_singler_hpca}, Blueprint={cfg.include_singler_blueprint}"
        )

        # Collect predictions from all methods
        all_predictions = []

        # Run CellTypist models
        for model_name in cfg.celltypist_models:
            try:
                result = self._run_celltypist(adata, model_name)
                all_predictions.append(result)
                logger.info(f"CellTypist {model_name}: {len(result['predictions'])} predictions")
            except Exception as e:
                logger.warning(f"CellTypist {model_name} failed: {e}")

        # Run SingleR HPCA
        if cfg.include_singler_hpca:
            try:
                result = self._load_or_run_singler(adata, "hpca")
                if result:
                    all_predictions.append(result)
                    logger.info(f"SingleR-HPCA: {len(result['predictions'])} predictions")
            except Exception as e:
                logger.warning(f"SingleR-HPCA failed: {e}")

        # Run SingleR Blueprint
        if cfg.include_singler_blueprint:
            try:
                result = self._load_or_run_singler(adata, "blueprint")
                if result:
                    all_predictions.append(result)
                    logger.info(f"SingleR-Blueprint: {len(result['predictions'])} predictions")
            except Exception as e:
                logger.warning(f"SingleR-Blueprint failed: {e}")

        if not all_predictions:
            raise ValueError("No annotation methods produced results")

        # Track which methods were used (for column generation)
        self._method_sources = [pred["source"] for pred in all_predictions]
        logger.info(f"Methods used: {self._method_sources}")

        # Build consensus using popV-style voting
        voting_results = self._build_consensus_popv_style(
            all_predictions,
            adata.obs.index.tolist(),
        )

        # Convert to DataFrame (includes per-method predictions if configured)
        annotations_df = self._voting_results_to_dataframe(
            voting_results,
            all_predictions,
            include_per_method=cfg.include_per_method_predictions,
        )

        # Compute statistics
        voting_stats = self._compute_voting_stats(voting_results, all_predictions)

        # Compute per-method metrics (if ground truth available)
        per_method_metrics = {}

        # Class distribution at requested granularity
        class_col = self._get_class_column(cfg.granularity)
        class_distribution = dict(annotations_df[class_col].value_counts().iter_rows())

        return EnsembleResult(
            annotations_df=annotations_df,
            voting_stats=voting_stats,
            per_method_metrics=per_method_metrics,
            class_distribution=class_distribution,
        )

    def _run_celltypist(self, adata, model_name: str) -> dict:
        """Run CellTypist annotation."""
        import celltypist
        import scanpy as sc
        from celltypist import models

        # Download and load model
        models.download_models(force_update=False, model=model_name)
        model = models.Model.load(model=model_name)

        # Normalize (CellTypist expects normalized data)
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Predict with majority voting
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
            "method_type": "celltypist",
            "predictions": labels,
            "confidence": confidence,
            "cell_ids": [str(cid) for cid in adata.obs.index.tolist()],
        }

    def _load_or_run_singler(self, adata, reference: str) -> dict | None:
        """Load pre-computed SingleR results or run if not available.

        SingleR requires R, so we check for pre-computed results first.
        Matches files by cell count to ensure correct dataset alignment.
        """
        # Check cache
        cache_key = f"singler_{reference}"
        if cache_key in self._predictions_cache:
            return self._predictions_cache[cache_key]

        n_cells = adata.n_obs  # Use cell count for matching

        # All possible SingleR files with format info
        singler_files = [
            # Format: (path, format_type, expected_n_cells or None)
            # Prefer dual_reference format (has both HPCA and Blueprint)
            (Path("experiment_all_methods/breast_rep1/singler_predictions.csv"), "dual_reference", 167780),
            (Path("experiment_all_methods/breast_rep2/singler_predictions.csv"), "dual_reference", 118752),
            # Fallback to hpca_only format
            (Path("experiment_singler_comparison/rep1/singler_annotations.csv"), "hpca_only", 167780),
            (Path("experiment_singler_comparison/rep2/singler_annotations.csv"), "hpca_only", 118752),
        ]

        # Try to find matching file by cell count (within 5% tolerance)
        for path, fmt, expected_cells in singler_files:
            if not path.exists():
                continue

            df = pl.read_csv(path)
            file_cells = len(df)

            # Check if cell count matches (with tolerance for filtered cells)
            if expected_cells and abs(file_cells - n_cells) > n_cells * 0.1:
                continue  # Skip if cell count doesn't match

            logger.info(f"Loading SingleR from {path} ({file_cells} cells, format={fmt})")

            if fmt == "hpca_only":
                # Rep1 format: predicted_type column (HPCA only)
                if reference == "blueprint":
                    continue  # No Blueprint in this format

                if "predicted_type" in df.columns:
                    result = {
                        "source": f"singler_{reference}",
                        "method_type": "singler",
                        "predictions": df["predicted_type"].to_list(),
                        "confidence": df["delta"].fill_null(0.0).to_list() if "delta" in df.columns else [1.0] * len(df),
                        "cell_ids": [str(cid) for cid in df["cell_id"].to_list()],
                    }
                    self._predictions_cache[cache_key] = result
                    return result
            else:
                # Dual reference format: separate HPCA and Blueprint columns
                if reference == "hpca":
                    label_col = "singler_hpca_label"
                    conf_col = "singler_hpca_score"
                else:  # blueprint
                    label_col = "singler_bp_label"
                    conf_col = "singler_bp_score"

                if label_col in df.columns:
                    result = {
                        "source": f"singler_{reference}",
                        "method_type": "singler",
                        "predictions": df[label_col].to_list(),
                        "confidence": df[conf_col].to_list() if conf_col in df.columns else [1.0] * len(df),
                        "cell_ids": [str(cid) for cid in df["cell_id"].to_list()],
                    }
                    self._predictions_cache[cache_key] = result
                    return result

        # Format 2: Single-reference files (legacy)
        possible_paths = [
            Path(f"singler_{reference}_predictions.csv"),
            Path(f"singler_results_{reference}.csv"),
            Path(f"experiment_singler_comparison/singler_{reference}.csv"),
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading pre-computed SingleR results from {path}")
                df = pl.read_csv(path)

                # Handle different column names
                cell_id_col = "cell_id" if "cell_id" in df.columns else df.columns[0]
                label_col = (
                    "labels"
                    if "labels" in df.columns
                    else "predicted_labels" if "predicted_labels" in df.columns
                    else "predicted_type" if "predicted_type" in df.columns
                    else df.columns[1]
                )
                conf_col = "delta.next" if "delta.next" in df.columns else "delta" if "delta" in df.columns else None

                result = {
                    "source": f"singler_{reference}",
                    "method_type": "singler",
                    "predictions": df[label_col].to_list(),
                    "confidence": df[conf_col].to_list() if conf_col else [1.0] * len(df),
                    "cell_ids": [str(cid) for cid in df[cell_id_col].to_list()],
                }

                self._predictions_cache[cache_key] = result
                return result

        # If no pre-computed results, try to run SingleR
        try:
            return self._run_singler_r(adata, reference)
        except Exception as e:
            logger.warning(f"Could not run SingleR-{reference}: {e}")
            return None

    def _run_singler_r(self, adata, reference: str) -> dict:
        """Run SingleR via R (requires R installation)."""
        import shutil
        import subprocess
        import tempfile

        if not shutil.which("Rscript"):
            raise RuntimeError("R is not installed")

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Save expression data
            expr_path = temp_dir / "expression.h5ad"
            adata.write_h5ad(expr_path)

            # Write R script
            r_script = self._get_singler_script()
            r_script_path = temp_dir / "run_singler.R"
            r_script_path.write_text(r_script)

            # Run R
            result = subprocess.run(
                ["Rscript", str(r_script_path), str(expr_path), reference, str(temp_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(f"SingleR failed: {result.stderr}")

            # Load results
            results_path = temp_dir / "singler_results.csv"
            results_df = pl.read_csv(results_path)

            return {
                "source": f"singler_{reference}",
                "method_type": "singler",
                "predictions": results_df["labels"].to_list(),
                "confidence": results_df["delta.next"].to_list(),
                "cell_ids": [str(cid) for cid in results_df["cell_id"].to_list()],
            }

        finally:
            # Cleanup
            import shutil as sh

            sh.rmtree(temp_dir, ignore_errors=True)

    def _get_singler_script(self) -> str:
        """Return R script for SingleR."""
        return '''
library(SingleR)
library(celldex)
library(anndata)

args <- commandArgs(trailingOnly = TRUE)
expr_path <- args[1]
reference_name <- args[2]
output_dir <- args[3]

adata <- read_h5ad(expr_path)
counts <- t(adata$X)
colnames(counts) <- rownames(adata$obs)

ref <- switch(reference_name,
    "blueprint" = BlueprintEncodeData(),
    "hpca" = HumanPrimaryCellAtlasData(),
    "monaco" = MonacoImmuneData(),
    stop("Unknown reference")
)

results <- SingleR(
    test = counts,
    ref = ref,
    labels = ref$label.main,
    de.method = "wilcox"
)

output <- data.frame(
    cell_id = rownames(results),
    labels = results$labels,
    delta.next = results$delta.next
)
write.csv(output, file.path(output_dir, "singler_results.csv"), row.names = FALSE)
'''

    def _build_consensus_popv_style(
        self,
        all_predictions: list[dict],
        cell_ids: list[str],
    ) -> list[VotingResult]:
        """Build consensus using popV-style voting.

        Key mechanisms:
        1. Unweighted voting (each method = 1 vote)
        2. Cell Ontology-based hierarchical agreement
        3. Depth-based tie-breaking (prefer more specific)

        Args:
            all_predictions: List of prediction dicts from each method
            cell_ids: List of cell IDs to annotate

        Returns:
            List of VotingResult for each cell
        """
        cfg = self.config
        voting_results = []

        # Build cell_id -> predictions mapping
        cell_predictions = defaultdict(list)
        for pred in all_predictions:
            pred_dict = dict(zip(pred["cell_ids"], pred["predictions"]))
            conf_dict = dict(zip(pred["cell_ids"], pred["confidence"]))

            for cell_id in cell_ids:
                str_cell_id = str(cell_id)
                if str_cell_id in pred_dict:
                    cell_predictions[str_cell_id].append(
                        {
                            "source": pred["source"],
                            "label": pred_dict[str_cell_id],
                            "confidence": conf_dict[str_cell_id],
                        }
                    )

        # Process each cell
        for cell_id, votes in cell_predictions.items():
            if len(votes) < cfg.min_agreement:
                continue

            if cfg.voting_strategy == VotingStrategy.UNWEIGHTED:
                result = self._unweighted_vote(cell_id, votes)
            elif cfg.voting_strategy == VotingStrategy.CONFIDENCE_WEIGHTED:
                result = self._confidence_weighted_vote(cell_id, votes)
            elif cfg.voting_strategy == VotingStrategy.ONTOLOGY_HIERARCHICAL:
                result = self._ontology_hierarchical_vote(cell_id, votes)
            else:
                result = self._unweighted_vote(cell_id, votes)

            if result:
                voting_results.append(result)

        return voting_results

    def _extract_per_method_predictions(
        self, votes: list[dict]
    ) -> tuple[dict[str, str], dict[str, float]]:
        """Extract per-method predictions and confidences from votes.

        Args:
            votes: List of vote dicts with source, label, confidence

        Returns:
            Tuple of (source -> prediction, source -> confidence)
        """
        per_method_preds = {}
        per_method_confs = {}
        for v in votes:
            source = v["source"]
            per_method_preds[source] = v["label"]
            per_method_confs[source] = v["confidence"]
        return per_method_preds, per_method_confs

    def _unweighted_vote(
        self, cell_id: str, votes: list[dict]
    ) -> VotingResult | None:
        """Simple majority voting (popV default).

        Each method gets exactly 1 vote. Winner is most common prediction.
        Ties broken by alphabetical order (deterministic).
        """
        labels = [v["label"] for v in votes]
        confidences = [v["confidence"] for v in votes]

        # Extract per-method predictions
        per_method_preds, per_method_confs = self._extract_per_method_predictions(votes)

        # Count votes per label
        vote_counts = Counter(labels)

        # Get winner (most votes, alphabetical tie-break)
        max_count = max(vote_counts.values())
        winners = sorted([l for l, c in vote_counts.items() if c == max_count])
        winner = winners[0]

        # Map to categories
        broad_cat = map_to_broad_category(winner)
        medium_cat = self._map_to_medium_category(winner)

        # Get CL ID and depth
        cl_id = ANNOTATOR_TO_CL.get(winner)
        depth = CL_DEPTH.get(cl_id, 1) if cl_id else 1

        return VotingResult(
            cell_id=cell_id,
            winner=winner,
            winner_cl_id=cl_id,
            broad_category=broad_cat,
            medium_category=medium_cat,
            n_votes=len(votes),
            n_agreement=vote_counts[winner],
            agreement_ratio=vote_counts[winner] / len(votes),
            is_unanimous=len(vote_counts) == 1,
            all_votes=labels,
            confidence_scores=confidences,
            voting_strategy="unweighted",
            ontology_depth=depth,
            per_method_predictions=per_method_preds,
            per_method_confidences=per_method_confs,
        )

    def _confidence_weighted_vote(
        self, cell_id: str, votes: list[dict]
    ) -> VotingResult | None:
        """Confidence-weighted voting.

        Each vote weighted by method's confidence score.
        """
        labels = [v["label"] for v in votes]
        confidences = [v["confidence"] for v in votes]

        # Extract per-method predictions
        per_method_preds, per_method_confs = self._extract_per_method_predictions(votes)

        # Aggregate confidence per label
        label_scores = defaultdict(float)
        for label, conf in zip(labels, confidences):
            label_scores[label] += conf

        # Get winner
        winner = max(label_scores.items(), key=lambda x: x[1])[0]
        total_score = sum(label_scores.values())

        # Map to categories
        broad_cat = map_to_broad_category(winner)
        medium_cat = self._map_to_medium_category(winner)

        # Get CL ID and depth
        cl_id = ANNOTATOR_TO_CL.get(winner)
        depth = CL_DEPTH.get(cl_id, 1) if cl_id else 1

        # Count actual votes for winner
        n_agreement = sum(1 for l in labels if l == winner)

        return VotingResult(
            cell_id=cell_id,
            winner=winner,
            winner_cl_id=cl_id,
            broad_category=broad_cat,
            medium_category=medium_cat,
            n_votes=len(votes),
            n_agreement=n_agreement,
            agreement_ratio=label_scores[winner] / total_score if total_score > 0 else 0,
            is_unanimous=len(set(labels)) == 1,
            all_votes=labels,
            confidence_scores=confidences,
            voting_strategy="confidence_weighted",
            ontology_depth=depth,
            per_method_predictions=per_method_preds,
            per_method_confidences=per_method_confs,
        )

    def _ontology_hierarchical_vote(
        self, cell_id: str, votes: list[dict]
    ) -> VotingResult | None:
        """Cell Ontology-based hierarchical voting (popV style).

        Key innovation: Methods agree if their predictions share a common
        ancestor in the Cell Ontology. Tie-breaking prefers more specific
        (deeper) predictions.

        For example, "CD4+ T cell" and "regulatory T cell" both agree at
        the "T cell" level.
        """
        labels = [v["label"] for v in votes]
        confidences = [v["confidence"] for v in votes]

        # Extract per-method predictions
        per_method_preds, per_method_confs = self._extract_per_method_predictions(votes)

        # Map labels to Cell Ontology IDs
        label_to_cl = {}
        for label in set(labels):
            cl_id = ANNOTATOR_TO_CL.get(label)
            if cl_id:
                label_to_cl[label] = cl_id
            else:
                # Fall back to broad category matching
                broad = map_to_broad_category(label)
                label_to_cl[label] = f"BROAD:{broad}"

        # Group by broad category first (first-stage voting like popV)
        broad_votes = [map_to_broad_category(l) for l in labels]
        broad_counts = Counter(broad_votes)
        winning_broad = max(broad_counts.items(), key=lambda x: x[1])[0]

        # Filter to methods that voted for winning broad category
        matching_labels = [
            l for l in labels if map_to_broad_category(l) == winning_broad
        ]

        if not matching_labels:
            # Fallback to simple majority
            return self._unweighted_vote(cell_id, votes)

        # Second-stage: Among matching broad category, prefer deepest (most specific)
        label_depths = {}
        for label in set(matching_labels):
            cl_id = label_to_cl.get(label, "")
            if cl_id.startswith("BROAD:"):
                label_depths[label] = 1
            else:
                label_depths[label] = CL_DEPTH.get(cl_id, 2)

        # Get deepest label(s)
        max_depth = max(label_depths.values())
        deepest_labels = [l for l, d in label_depths.items() if d == max_depth]

        # Among deepest, pick most common
        deepest_counts = Counter(l for l in matching_labels if l in deepest_labels)

        if deepest_counts:
            winner = max(deepest_counts.items(), key=lambda x: (x[1], -len(x[0])))[0]
        else:
            # Fallback
            winner = matching_labels[0]

        # Map to categories
        broad_cat = map_to_broad_category(winner)
        medium_cat = self._map_to_medium_category(winner)

        # Get CL ID
        cl_id = ANNOTATOR_TO_CL.get(winner)
        depth = label_depths.get(winner, 1)

        # Count agreement
        n_agreement = sum(1 for l in labels if l == winner)

        return VotingResult(
            cell_id=cell_id,
            winner=winner,
            winner_cl_id=cl_id,
            broad_category=broad_cat,
            medium_category=medium_cat,
            n_votes=len(votes),
            n_agreement=n_agreement,
            agreement_ratio=n_agreement / len(votes),
            is_unanimous=len(set(labels)) == 1,
            all_votes=labels,
            confidence_scores=confidences,
            voting_strategy="ontology_hierarchical",
            ontology_depth=depth,
            per_method_predictions=per_method_preds,
            per_method_confidences=per_method_confs,
        )

    def _map_to_medium_category(self, label: str) -> str:
        """Map fine-grained label to medium granularity."""
        # Direct mapping
        if label in FINE_TO_MEDIUM_MAPPING:
            return FINE_TO_MEDIUM_MAPPING[label]

        # Fall back to keyword matching
        label_lower = label.lower()

        # T cells
        if any(x in label_lower for x in ["t cell", "t_cell", "cd4", "cd8", "treg"]):
            return "T_Cell"
        # B cells
        if any(x in label_lower for x in ["b cell", "b_cell", "plasma"]):
            return "B_Cell"
        # NK cells
        if "nk" in label_lower or "natural killer" in label_lower:
            return "NK_Cell"
        # Myeloid
        if any(
            x in label_lower
            for x in ["macro", "mono", "dendrit", "mast", "neutro", "myeloid"]
        ):
            return "Myeloid"
        # Stromal
        if any(x in label_lower for x in ["fibro", "stromal", "mesench"]):
            return "Stromal_Fibroblast"
        if any(x in label_lower for x in ["pericyte", "smooth muscle", "perivascular"]):
            return "Stromal_Pericyte"
        # Endothelial
        if "endothelial" in label_lower or "vascular" in label_lower:
            return "Endothelial"
        # Epithelial
        if any(x in label_lower for x in ["tumor", "dcis", "invasive", "prolif"]):
            return "Epithelial_Tumor"
        if any(x in label_lower for x in ["basal", "myoepi"]):
            return "Epithelial_Basal"
        if any(x in label_lower for x in ["luminal", "epithelial"]):
            return "Epithelial_Luminal"

        # Default
        return "Unknown"

    def _get_class_column(self, granularity: GranularityLevel) -> str:
        """Get the column name for the requested granularity."""
        if granularity == GranularityLevel.COARSE:
            return "predicted_type_coarse"
        elif granularity == GranularityLevel.MEDIUM:
            return "predicted_type_medium"
        else:
            return "predicted_type_fine"

    def _source_to_column_name(self, source: str) -> str:
        """Convert a method source name to a valid column name.

        Examples:
            'celltypist_Cells_Adult_Breast' -> 'pred_celltypist_Cells_Adult_Breast'
            'singler_hpca' -> 'pred_singler_hpca'
            'singler_blueprint' -> 'pred_singler_blueprint'
        """
        return f"pred_{source}"

    def _voting_results_to_dataframe(
        self,
        voting_results: list[VotingResult],
        all_predictions: list[dict] | None = None,
        include_per_method: bool = True,
    ) -> pl.DataFrame:
        """Convert voting results to Polars DataFrame.

        Args:
            voting_results: List of VotingResult objects
            all_predictions: Original predictions from all methods (for column ordering)
            include_per_method: Whether to include per-method prediction columns

        Returns:
            DataFrame with consensus predictions and optionally per-method columns
        """
        # Collect all unique method sources for consistent column ordering
        all_sources = []
        if all_predictions:
            all_sources = [pred["source"] for pred in all_predictions]
        elif voting_results:
            # Fallback: collect from voting results
            all_sources = list(voting_results[0].per_method_predictions.keys())

        records = []
        for vr in voting_results:
            record = {
                "cell_id": vr.cell_id,
                "predicted_type_fine": vr.winner,
                "predicted_type_medium": vr.medium_category,
                "predicted_type_coarse": vr.broad_category,
                "confidence": np.mean(vr.confidence_scores) if vr.confidence_scores else 0.0,
                "cl_id": vr.winner_cl_id,
                "n_votes": vr.n_votes,
                "n_agreement": vr.n_agreement,
                "agreement_ratio": vr.agreement_ratio,
                "is_unanimous": vr.is_unanimous,
                "voting_strategy": vr.voting_strategy,
                "ontology_depth": vr.ontology_depth,
            }

            # Add per-method predictions if enabled
            if include_per_method and vr.per_method_predictions:
                for source in all_sources:
                    pred_col = self._source_to_column_name(source)
                    conf_col = f"conf_{source}"

                    # Add prediction column (None if method didn't vote for this cell)
                    record[pred_col] = vr.per_method_predictions.get(source)

                    # Add confidence column
                    record[conf_col] = vr.per_method_confidences.get(source)

            records.append(record)

        return pl.DataFrame(records)

    def _compute_voting_stats(
        self, voting_results: list[VotingResult], all_predictions: list[dict]
    ) -> dict[str, Any]:
        """Compute voting statistics."""
        n_cells = len(voting_results)
        n_unanimous = sum(1 for vr in voting_results if vr.is_unanimous)
        n_majority = n_cells - n_unanimous

        # Agreement distribution
        agreement_ratios = [vr.agreement_ratio for vr in voting_results]

        # Ontology depth distribution
        depths = [vr.ontology_depth for vr in voting_results]

        return {
            "n_cells": n_cells,
            "n_unanimous": n_unanimous,
            "n_majority": n_majority,
            "pct_unanimous": n_unanimous / n_cells if n_cells > 0 else 0,
            "avg_agreement_ratio": np.mean(agreement_ratios) if agreement_ratios else 0,
            "avg_ontology_depth": np.mean(depths) if depths else 0,
            "methods_used": [p["source"] for p in all_predictions],
            "n_methods": len(all_predictions),
            "voting_strategy": self.config.voting_strategy.value,
            "granularity": self.config.granularity.value,
        }

    def get_method_sources(self) -> list[str]:
        """Get the list of method sources used in the last annotation.

        Returns:
            List of method source names (e.g., ['celltypist_Cells_Adult_Breast', 'singler_hpca'])
        """
        return self._method_sources.copy()

    def get_prediction_columns(self) -> list[str]:
        """Get the per-method prediction column names.

        Returns:
            List of prediction column names (e.g., ['pred_celltypist_Cells_Adult_Breast', 'pred_singler_hpca'])
        """
        return [self._source_to_column_name(src) for src in self._method_sources]

    def get_confidence_columns(self) -> list[str]:
        """Get the per-method confidence column names.

        Returns:
            List of confidence column names (e.g., ['conf_celltypist_Cells_Adult_Breast', 'conf_singler_hpca'])
        """
        return [f"conf_{src}" for src in self._method_sources]


# Convenience function for standalone use
def run_popv_ensemble(
    adata,
    celltypist_models: list[str] | None = None,
    include_singler: bool = True,
    voting_strategy: str = "ontology_hierarchical",
    granularity: str = "coarse",
    include_per_method_predictions: bool = True,
) -> pl.DataFrame:
    """Run popV-style ensemble annotation.

    Args:
        adata: AnnData with expression data
        celltypist_models: List of CellTypist model names
        include_singler: Whether to include SingleR (HPCA + Blueprint)
        voting_strategy: One of "unweighted", "confidence_weighted", "ontology_hierarchical"
        granularity: One of "coarse", "medium", "fine"
        include_per_method_predictions: Whether to include individual annotator predictions

    Returns:
        Polars DataFrame with annotations. Columns include:
        - cell_id: Cell identifier
        - predicted_type_fine: Fine-grained consensus prediction
        - predicted_type_medium: Medium-granularity prediction
        - predicted_type_coarse: Coarse prediction (Epithelial/Immune/Stromal)
        - confidence: Average confidence across methods
        - pred_celltypist_*: Per-CellTypist-model predictions (if enabled)
        - pred_singler_hpca: SingleR HPCA prediction (if enabled)
        - pred_singler_blueprint: SingleR Blueprint prediction (if enabled)
        - conf_*: Confidence scores for each method
    """
    config = PopVEnsembleConfig(
        celltypist_models=celltypist_models
        or ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        include_singler_hpca=include_singler,
        include_singler_blueprint=include_singler,
        voting_strategy=VotingStrategy(voting_strategy),
        granularity=GranularityLevel(granularity),
        include_per_method_predictions=include_per_method_predictions,
    )

    annotator = PopVStyleEnsembleAnnotator(config)
    result = annotator.annotate(adata)

    return result.annotations_df
