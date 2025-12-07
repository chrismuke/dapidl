"""Cell type annotation using CellTypist."""

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
    ],
    "Stromal": [
        # Fibroblasts
        "Fibro",
        "Fibro-SFRP4",
        "Fibro-major",
        "Fibro-matrix",
        "Fibro-prematrix",
        # Smooth muscle and pericytes
        "pericytes",
        "vsmc",
    ],
    "Endothelial": [
        # Vascular endothelial
        "Vas",
        "Vas-arterial",
        "Vas-capillary",
        "Vas-venous",
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
    for broad_cat, keywords in CELL_TYPE_HIERARCHY.items():
        for keyword in keywords:
            if cell_type == keyword or cell_type.startswith(keyword):
                return broad_cat
    return "Unknown"


class CellTypeAnnotator:
    """Annotate cells using CellTypist models.

    Uses CellTypist to predict cell types from gene expression data,
    then maps to broad categories for classification.

    Supports running multiple models for ensemble annotation.
    """

    def __init__(
        self,
        model_names: str | list[str] = DEFAULT_MODEL,
        confidence_threshold: float = 0.5,
        majority_voting: bool = True,
    ) -> None:
        """Initialize annotator.

        Args:
            model_names: Name(s) of CellTypist model(s) to use. Can be a single
                string or a list of model names for multi-model annotation.
            confidence_threshold: Minimum confidence score to accept prediction
            majority_voting: Whether to use majority voting for predictions
        """
        # Normalize to list
        if isinstance(model_names, str):
            self.model_names = [model_names]
        else:
            self.model_names = list(model_names)

        self.confidence_threshold = confidence_threshold
        self.majority_voting = majority_voting
        self._models: dict[str, Any] = {}

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run annotation with a single model.

        Args:
            adata_norm: Normalized AnnData object
            model_name: Name of the model to use

        Returns:
            Tuple of (cell_types, confidence_scores)
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
        conf_scores = predictions.probability_matrix.max(axis=1).values

        return cell_types, conf_scores

    def annotate(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run cell type annotation with all configured models.

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with cell_id and prediction columns for each model.
            For single model: predicted_type, broad_category, confidence
            For multiple models: predicted_type_1, confidence_1, predicted_type_2, ...
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
            cell_types, conf_scores = self._run_single_model(adata_norm, model_name)

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
        filtered = annotations.filter(pl.col("confidence") >= self.confidence_threshold)
        logger.info(
            f"Filtered from {len(annotations)} to {len(filtered)} cells "
            f"(threshold={self.confidence_threshold})"
        )
        return filtered

    def get_class_mapping(self, annotations: pl.DataFrame) -> dict[str, int]:
        """Get mapping from broad category names to integer labels.

        Args:
            annotations: DataFrame with broad_category column

        Returns:
            Dictionary mapping category name to integer label
        """
        categories = sorted(annotations["broad_category"].unique().to_list())
        # Filter out Unknown if present, add at end
        if "Unknown" in categories:
            categories.remove("Unknown")
            categories.append("Unknown")
        return {cat: i for i, cat in enumerate(categories)}
