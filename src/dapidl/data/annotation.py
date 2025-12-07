"""Cell type annotation using CellTypist with multiple annotation strategies.

Supported strategies:
1. popV - Ensemble prediction using popV (if installed)
2. hierarchical - Tissue-specific model + specialized refinement
3. consensus - Consensus voting across multiple CellTypist models (default)
"""

from enum import Enum
from pathlib import Path
from typing import Any

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import polars as pl
from celltypist import models
from loguru import logger

from dapidl.data.xenium import XeniumDataReader


# Default model for breast tissue
DEFAULT_MODEL = "Cells_Adult_Breast.pkl"


class AnnotationStrategy(str, Enum):
    """Available annotation strategies."""

    SINGLE = "single"  # Single model annotation
    CONSENSUS = "consensus"  # Consensus voting across multiple models (default)
    HIERARCHICAL = "hierarchical"  # Tissue-specific + specialized refinement
    POPV = "popv"  # popV ensemble prediction


# Default strategy
DEFAULT_STRATEGY = AnnotationStrategy.CONSENSUS


def list_available_models(force_update: bool = False) -> pd.DataFrame:
    """List all available CellTypist models.

    Args:
        force_update: Whether to fetch the latest model list from the server

    Returns:
        DataFrame with model names and descriptions
    """
    return models.models_description(on_the_fly=False)


def get_downloaded_models() -> list[str]:
    """Get list of locally downloaded CellTypist models.

    Returns:
        List of model filenames that are available locally
    """
    return models.get_all_models()


def download_model(model_name: str, force_update: bool = False) -> None:
    """Download a specific CellTypist model.

    Args:
        model_name: Name of the model to download (e.g., 'Cells_Adult_Breast.pkl')
        force_update: Whether to re-download even if exists
    """
    models.download_models(model=model_name, force_update=force_update)


def is_popv_available() -> bool:
    """Check if popV is installed and available."""
    try:
        import popv
        return True
    except ImportError:
        return False


# Mapping from CellTypist cell types to broad categories
# Based on Cells_Adult_Breast.pkl model cell types
CELL_TYPE_HIERARCHY = {
    "Epithelial": [
        # Luminal HR+ cells
        "LummHR",
        "LummHR-SCGB",
        "LummHR-active",
        "LummHR-major",
        # Luminal secretory cells
        "Lumsec",
        "Lumsec-HLA",
        "Lumsec-KIT",
        "Lumsec-basal",
        "Lumsec-lac",
        "Lumsec-major",
        "Lumsec-myo",
        "Lumsec-prol",
        # Basal cells
        "basal",
    ],
    "Immune": [
        # T cells
        "CD4",
        "CD8",
        "T_prol",
        "GD",  # gamma-delta T cells
        "NKT",
        "Treg",
        # B cells
        "b_naive",
        "bmem_switched",
        "bmem_unswitched",
        "plasma",
        "plasma_IgA",
        "plasma_IgG",
        # Myeloid
        "Macro",
        "Mono",
        "cDC",
        "mDC",
        "pDC",
        "mye-prol",
        # Others
        "NK",
        "NK-ILCs",
        "Mast",
        "Neutrophil",
        # Additional immune cell types from Immune_All_High model
        "B cell",
        "B cells",
        "T cell",
        "T cells",
        "Macrophage",
        "Macrophages",
        "Dendritic",
        "DC",
        "Monocyte",
        "Monocytes",
    ],
    "Stromal": [
        # Fibroblasts
        "Fibro",
        "Fibro-SFRP4",
        "Fibro-major",
        "Fibro-matrix",
        "Fibro-prematrix",
        "Fibroblast",
        "Fibroblasts",
        # Smooth muscle and pericytes
        "pericytes",
        "Pericyte",
        "Pericytes",
        "vsmc",
        "Smooth muscle",
    ],
    "Endothelial": [
        # Vascular endothelial
        "Vas",
        "Vas-arterial",
        "Vas-capillary",
        "Vas-venous",
        "Endothelial",
        # Lymphatic endothelial
        "Lymph",
        "Lymph-immune",
        "Lymph-major",
        "Lymph-valve1",
        "Lymph-valve2",
    ],
}


def map_to_broad_category(cell_type: str) -> str:
    """Map a detailed cell type to a broad category.

    Uses prefix matching - checks if the cell type starts with any keyword.

    Args:
        cell_type: Detailed cell type string from CellTypist

    Returns:
        Broad category name or 'Unknown' if no match
    """
    # First try exact match, then prefix match
    cell_type_lower = cell_type.lower()
    for broad_cat, keywords in CELL_TYPE_HIERARCHY.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if cell_type_lower == keyword_lower or cell_type_lower.startswith(keyword_lower):
                return broad_cat
    return "Unknown"


class CellTypeAnnotator:
    """Annotate cells using CellTypist models with multiple annotation strategies.

    Supported strategies:
    - single: Use a single CellTypist model
    - consensus: Combine predictions from multiple models using voting (default)
    - hierarchical: Use tissue-specific model + specialized refinement
    - popv: Use popV ensemble prediction (requires popv package)
    """

    def __init__(
        self,
        model_names: str | list[str] = DEFAULT_MODEL,
        confidence_threshold: float = 0.5,
        majority_voting: bool = True,
        strategy: AnnotationStrategy | str = DEFAULT_STRATEGY,
    ) -> None:
        """Initialize annotator.

        Args:
            model_names: Name(s) of CellTypist model(s) to use. Can be a single
                string or a list of model names for multi-model annotation.
            confidence_threshold: Minimum confidence score to accept prediction
            majority_voting: Whether to use majority voting for predictions
            strategy: Annotation strategy to use (single, consensus, hierarchical, popv)
        """
        # Normalize to list
        if isinstance(model_names, str):
            self.model_names = [model_names]
        else:
            self.model_names = list(model_names)

        self.confidence_threshold = confidence_threshold
        self.majority_voting = majority_voting

        # Parse strategy
        if isinstance(strategy, str):
            strategy = AnnotationStrategy(strategy.lower())
        self.strategy = strategy

        # Validate strategy
        if self.strategy == AnnotationStrategy.POPV and not is_popv_available():
            logger.warning("popV not installed. Falling back to consensus strategy.")
            logger.warning("Install popV with: pip install popv")
            self.strategy = AnnotationStrategy.CONSENSUS

        if self.strategy == AnnotationStrategy.CONSENSUS and len(self.model_names) < 2:
            logger.warning("Consensus strategy requires at least 2 models. Adding Immune_All_High.pkl")
            if "Immune_All_High.pkl" not in self.model_names:
                self.model_names.append("Immune_All_High.pkl")

        self._models: dict[str, Any] = {}
        logger.info(f"Annotation strategy: {self.strategy.value}")

    def _load_model(self, model_name: str) -> Any:
        """Download and load a CellTypist model.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded CellTypist model
        """
        if model_name not in self._models:
            logger.info(f"Loading CellTypist model: {model_name}")
            models.download_models(model=model_name, force_update=False)
            self._models[model_name] = models.Model.load(model=model_name)
            logger.info(f"Model loaded with {len(self._models[model_name].cell_types)} cell types")
        return self._models[model_name]

    def _load_all_models(self) -> dict[str, Any]:
        """Load all configured models.

        Returns:
            Dictionary mapping model names to loaded models
        """
        for model_name in self.model_names:
            self._load_model(model_name)
        return self._models

    def create_anndata(self, reader: XeniumDataReader) -> ad.AnnData:
        """Create AnnData object from Xenium data.

        Args:
            reader: XeniumDataReader with loaded expression data

        Returns:
            AnnData object suitable for CellTypist
        """
        expr, genes, cell_ids = reader.load_expression_matrix()

        # Create AnnData
        adata = ad.AnnData(X=expr)
        adata.var_names = genes
        adata.obs_names = [str(cid) for cid in cell_ids]
        adata.obs["cell_id"] = cell_ids

        # Add spatial coordinates
        centroids_um = reader.get_centroids_microns()
        adata.obsm["spatial"] = centroids_um

        logger.info(f"Created AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
        return adata

    def _run_single_model(
        self, adata_norm: ad.AnnData, model_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run annotation with a single model.

        Args:
            adata_norm: Normalized AnnData object
            model_name: Name of the model to use

        Returns:
            Tuple of (cell_types, confidence_scores, probability_matrix)
        """
        model = self._load_model(model_name)

        logger.info(f"Running CellTypist prediction with {model_name}...")
        predictions = celltypist.annotate(
            adata_norm,
            model=model,
            majority_voting=self.majority_voting,
        )

        # Extract results
        pred_labels = predictions.predicted_labels
        label_col = "majority_voting" if self.majority_voting and "majority_voting" in pred_labels else "predicted_labels"
        cell_types = pred_labels[label_col].astype(str).values
        prob_matrix = predictions.probability_matrix.values
        conf_scores = prob_matrix.max(axis=1)

        return cell_types, conf_scores, prob_matrix

    def _annotate_popv(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run annotation using popV ensemble prediction.

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        try:
            import popv
        except ImportError:
            raise ImportError("popV not installed. Install with: pip install popv")

        logger.info("Running popV ensemble prediction...")

        # popV requires a reference dataset - use pretrained models if available
        # For now, we'll run popV's prediction pipeline
        # Note: This is a simplified integration - full popV requires reference data

        import scanpy as sc
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run popV annotation
        # popV.annotation.annotate_data returns annotated AnnData
        try:
            popv.annotation.annotate_data(
                adata_norm,
                methods=["celltypist"],  # Start with celltypist only for compatibility
                model_name=self.model_names[0] if self.model_names else DEFAULT_MODEL,
            )

            # Extract results
            cell_types = adata_norm.obs.get("popv_prediction", adata_norm.obs.get("celltypist_prediction", np.array(["Unknown"] * len(adata_norm))))
            confidence = adata_norm.obs.get("popv_confidence", np.ones(len(adata_norm)) * 0.5)

        except Exception as e:
            logger.warning(f"popV annotation failed: {e}. Falling back to CellTypist.")
            return self._annotate_consensus(adata)

        # Map to broad categories
        if isinstance(cell_types, pd.Series):
            cell_types = cell_types.values
        broad_categories = [map_to_broad_category(ct) for ct in cell_types]

        results = pl.DataFrame({
            "cell_id": adata.obs["cell_id"].values,
            "predicted_type": cell_types,
            "broad_category": broad_categories,
            "confidence": confidence if isinstance(confidence, np.ndarray) else confidence.values,
        })

        logger.info(f"popV annotation complete for {len(results)} cells")
        return results

    def _annotate_hierarchical(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run hierarchical annotation: tissue-specific + specialized refinement.

        Strategy:
        1. Run tissue-specific model (first model) for initial classification
        2. For cells classified as specific types (e.g., Immune), run specialized model
        3. Merge results with refined predictions

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        import scanpy as sc

        logger.info("Running hierarchical annotation...")

        # Normalize data
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Step 1: Run primary (tissue-specific) model
        primary_model = self.model_names[0]
        cell_types_primary, conf_primary, _ = self._run_single_model(adata_norm, primary_model)
        broad_primary = np.array([map_to_broad_category(ct) for ct in cell_types_primary])

        # Initialize final results with primary predictions
        final_cell_types = cell_types_primary.copy()
        final_confidence = conf_primary.copy()
        final_broad = broad_primary.copy()
        refined_by = np.array([""] * len(cell_types_primary))

        # Step 2: Run specialized models for refinement
        for i, specialized_model in enumerate(self.model_names[1:], start=2):
            logger.info(f"Running specialized model {specialized_model} for refinement...")

            cell_types_spec, conf_spec, _ = self._run_single_model(adata_norm, specialized_model)
            broad_spec = np.array([map_to_broad_category(ct) for ct in cell_types_spec])

            # Determine which cells to refine based on model type
            # If specialized model is immune-focused, refine immune cells
            model_lower = specialized_model.lower()

            if "immune" in model_lower:
                # Refine cells that are either Immune or Unknown from primary
                # but have high confidence immune prediction from specialized model
                refine_mask = (
                    ((broad_primary == "Immune") | (broad_primary == "Unknown")) &
                    (broad_spec == "Immune") &
                    (conf_spec > self.confidence_threshold)
                )
            else:
                # For other specialized models, refine Unknown cells or low-confidence predictions
                refine_mask = (
                    ((broad_primary == "Unknown") | (conf_primary < self.confidence_threshold)) &
                    (broad_spec != "Unknown") &
                    (conf_spec > conf_primary)
                )

            # Apply refinement
            final_cell_types[refine_mask] = cell_types_spec[refine_mask]
            final_confidence[refine_mask] = conf_spec[refine_mask]
            final_broad[refine_mask] = broad_spec[refine_mask]
            refined_by[refine_mask] = specialized_model

            n_refined = refine_mask.sum()
            logger.info(f"  Refined {n_refined} cells using {specialized_model}")

        results = pl.DataFrame({
            "cell_id": adata.obs["cell_id"].values,
            "predicted_type": final_cell_types,
            "broad_category": final_broad,
            "confidence": final_confidence,
            "refined_by": refined_by,
            "primary_model": primary_model,
        })

        logger.info(f"Hierarchical annotation complete for {len(results)} cells")
        self._log_category_distribution(results, "broad_category")

        return results

    def _annotate_consensus(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run consensus annotation: voting across multiple CellTypist models.

        Strategy:
        1. Run all models independently
        2. For each cell, aggregate predictions using voting
        3. Final prediction is the consensus with confidence based on agreement

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        import scanpy as sc
        from collections import Counter

        logger.info("Running consensus annotation with voting...")

        # Normalize data
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        n_cells = len(adata)
        n_models = len(self.model_names)

        # Collect predictions from all models
        all_cell_types = []
        all_broad_categories = []
        all_confidences = []

        for model_name in self.model_names:
            cell_types, conf_scores, _ = self._run_single_model(adata_norm, model_name)
            broad_categories = [map_to_broad_category(ct) for ct in cell_types]

            all_cell_types.append(cell_types)
            all_broad_categories.append(broad_categories)
            all_confidences.append(conf_scores)

        # Convert to arrays for easier manipulation
        all_cell_types = np.array(all_cell_types)  # (n_models, n_cells)
        all_broad_categories = np.array(all_broad_categories)
        all_confidences = np.array(all_confidences)

        # Compute consensus for each cell
        final_cell_types = []
        final_broad = []
        final_confidence = []
        consensus_scores = []

        for cell_idx in range(n_cells):
            # Get predictions for this cell from all models
            cell_broads = all_broad_categories[:, cell_idx]
            cell_types = all_cell_types[:, cell_idx]
            cell_confs = all_confidences[:, cell_idx]

            # Vote on broad category (excluding Unknown if possible)
            valid_broads = [b for b in cell_broads if b != "Unknown"]
            if valid_broads:
                broad_counter = Counter(valid_broads)
            else:
                broad_counter = Counter(cell_broads)

            consensus_broad, consensus_count = broad_counter.most_common(1)[0]
            consensus_score = consensus_count / n_models

            # Find the best cell type prediction among models that agree on broad category
            best_type = None
            best_conf = 0.0
            for model_idx in range(n_models):
                if all_broad_categories[model_idx, cell_idx] == consensus_broad:
                    if all_confidences[model_idx, cell_idx] > best_conf:
                        best_conf = all_confidences[model_idx, cell_idx]
                        best_type = all_cell_types[model_idx, cell_idx]

            # If no model matches consensus broad (shouldn't happen), take highest confidence
            if best_type is None:
                best_idx = np.argmax(cell_confs)
                best_type = cell_types[best_idx]
                best_conf = cell_confs[best_idx]

            # Adjust confidence based on consensus score
            # High agreement (all models agree) = full confidence
            # Low agreement = reduced confidence
            adjusted_confidence = best_conf * consensus_score

            final_cell_types.append(best_type)
            final_broad.append(consensus_broad)
            final_confidence.append(adjusted_confidence)
            consensus_scores.append(consensus_score)

        # Build results with individual model predictions for reference
        results_data = {
            "cell_id": adata.obs["cell_id"].values,
            "predicted_type": final_cell_types,
            "broad_category": final_broad,
            "confidence": final_confidence,
            "consensus_score": consensus_scores,
        }

        # Add individual model predictions
        for i, model_name in enumerate(self.model_names):
            suffix = f"_{i+1}"
            results_data[f"predicted_type{suffix}"] = all_cell_types[i]
            results_data[f"broad_category{suffix}"] = all_broad_categories[i]
            results_data[f"confidence{suffix}"] = all_confidences[i]
            results_data[f"model{suffix}"] = model_name

        results = pl.DataFrame(results_data)

        # Log statistics
        logger.info(f"Consensus annotation complete for {len(results)} cells")
        logger.info(f"Mean consensus score: {np.mean(consensus_scores):.3f}")
        high_consensus = sum(1 for s in consensus_scores if s >= 0.5)
        logger.info(f"High consensus (≥50% agreement): {high_consensus} cells ({100*high_consensus/n_cells:.1f}%)")

        self._log_category_distribution(results, "broad_category")

        return results

    def _log_category_distribution(self, results: pl.DataFrame, col: str) -> None:
        """Log the distribution of categories."""
        logger.info(f"Broad category distribution:")
        for cat in results[col].unique().sort():
            count = (results[col] == cat).sum()
            pct = count / len(results) * 100
            logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    def annotate(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run cell type annotation using the configured strategy.

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with cell_id and prediction columns.
        """
        if self.strategy == AnnotationStrategy.POPV:
            return self._annotate_popv(adata)
        elif self.strategy == AnnotationStrategy.HIERARCHICAL:
            return self._annotate_hierarchical(adata)
        elif self.strategy == AnnotationStrategy.CONSENSUS:
            return self._annotate_consensus(adata)
        else:  # SINGLE
            return self._annotate_single(adata)

    def _annotate_single(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run single-model annotation (original behavior).

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        import scanpy as sc

        # Normalize data (CellTypist expects log1p normalized data)
        logger.info("Normalizing expression data...")
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run predictions for each model
        results_data: dict[str, Any] = {"cell_id": adata.obs["cell_id"].values}

        for i, model_name in enumerate(self.model_names, start=1):
            cell_types, conf_scores, _ = self._run_single_model(adata_norm, model_name)

            # Map to broad categories
            broad_categories = [map_to_broad_category(ct) for ct in cell_types]

            # Add columns with suffix if multiple models
            if len(self.model_names) == 1:
                results_data["predicted_type"] = cell_types
                results_data["broad_category"] = broad_categories
                results_data["confidence"] = conf_scores
            else:
                results_data[f"predicted_type_{i}"] = cell_types
                results_data[f"broad_category_{i}"] = broad_categories
                results_data[f"confidence_{i}"] = conf_scores
                results_data[f"model_{i}"] = model_name

        # Create results DataFrame
        results = pl.DataFrame(results_data)

        # Log statistics
        logger.info(f"Annotation complete for {len(results)} cells with {len(self.model_names)} model(s)")

        # Log per-model stats
        for i, model_name in enumerate(self.model_names, start=1):
            suffix = "" if len(self.model_names) == 1 else f"_{i}"
            conf_col = f"confidence{suffix}"
            broad_col = f"broad_category{suffix}"

            high_conf = (results[conf_col] > self.confidence_threshold).sum()
            logger.info(
                f"[{model_name}] Confidence > {self.confidence_threshold}: {high_conf} cells"
            )
            logger.info(f"[{model_name}] Broad category distribution:")
            for cat in results[broad_col].unique().sort():
                count = (results[broad_col] == cat).sum()
                pct = count / len(results) * 100
                logger.info(f"  {cat}: {count} ({pct:.1f}%)")

        return results

    def annotate_from_reader(self, reader: XeniumDataReader) -> pl.DataFrame:
        """Convenience method to annotate directly from Xenium reader.

        Args:
            reader: XeniumDataReader instance

        Returns:
            DataFrame with annotations
        """
        adata = self.create_anndata(reader)
        return self.annotate(adata)

    def filter_by_confidence(self, annotations: pl.DataFrame) -> pl.DataFrame:
        """Filter annotations by confidence threshold.

        Args:
            annotations: DataFrame from annotate()

        Returns:
            Filtered DataFrame with only high-confidence predictions
        """
        # Determine confidence column name (handles all strategies)
        if "confidence" in annotations.columns:
            conf_col = "confidence"
        elif "confidence_1" in annotations.columns:
            conf_col = "confidence_1"  # Use first model's confidence for filtering
        else:
            raise ValueError("No confidence column found in annotations")

        filtered = annotations.filter(pl.col(conf_col) >= self.confidence_threshold)
        logger.info(
            f"Filtered from {len(annotations)} to {len(filtered)} cells "
            f"(threshold={self.confidence_threshold}, column={conf_col})"
        )
        return filtered

    def get_class_mapping(self, annotations: pl.DataFrame) -> dict[str, int]:
        """Get mapping from broad category names to integer labels.

        Args:
            annotations: DataFrame with broad_category column

        Returns:
            Dictionary mapping category name to integer label
        """
        # Determine broad_category column name (handles all strategies)
        if "broad_category" in annotations.columns:
            broad_col = "broad_category"
        elif "broad_category_1" in annotations.columns:
            broad_col = "broad_category_1"  # Use first model for class mapping
        else:
            raise ValueError("No broad_category column found in annotations")

        categories = sorted(annotations[broad_col].unique().to_list())
        # Filter out Unknown if present, add at end
        if "Unknown" in categories:
            categories.remove("Unknown")
            categories.append("Unknown")
        return {cat: i for i, cat in enumerate(categories)}
