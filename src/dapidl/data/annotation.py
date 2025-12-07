"""Cell type annotation using CellTypist."""

from pathlib import Path
from typing import Any

import anndata as ad
import celltypist
import numpy as np
import polars as pl
from celltypist import models
from loguru import logger

from dapidl.data.xenium import XeniumDataReader


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
    """

    def __init__(
        self,
        model_name: str = "Cells_Adult_Breast.pkl",
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize annotator.

        Args:
            model_name: Name of CellTypist model to use
            confidence_threshold: Minimum confidence score to accept prediction
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model: Any = None

    def _load_model(self) -> Any:
        """Download and load CellTypist model."""
        if self._model is None:
            logger.info(f"Loading CellTypist model: {self.model_name}")
            models.download_models(model=self.model_name, force_update=False)
            self._model = models.Model.load(model=self.model_name)
            logger.info(f"Model loaded with {len(self._model.cell_types)} cell types")
        return self._model

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

    def annotate(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run cell type annotation.

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with cell_id, predicted_type, broad_category, confidence
        """
        import scanpy as sc

        model = self._load_model()

        # Normalize data (CellTypist expects log1p normalized data)
        logger.info("Normalizing expression data...")
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run CellTypist prediction
        logger.info("Running CellTypist prediction...")
        predictions = celltypist.annotate(
            adata_norm,
            model=model,
            majority_voting=True,
        )

        # Extract results
        pred_labels = predictions.predicted_labels
        cell_types = pred_labels["majority_voting"].astype(str).values
        conf_scores = predictions.probability_matrix.max(axis=1).values

        # Map to broad categories
        broad_categories = [map_to_broad_category(ct) for ct in cell_types]

        # Create results DataFrame
        results = pl.DataFrame(
            {
                "cell_id": adata.obs["cell_id"].values,
                "predicted_type": cell_types,
                "broad_category": broad_categories,
                "confidence": conf_scores,
            }
        )

        # Log statistics
        logger.info(f"Annotation complete for {len(results)} cells")
        logger.info(
            f"Confidence > {self.confidence_threshold}: "
            f"{(results['confidence'] > self.confidence_threshold).sum()} cells"
        )
        logger.info("Broad category distribution:")
        for cat in results["broad_category"].unique().sort():
            count = (results["broad_category"] == cat).sum()
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
