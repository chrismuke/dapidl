"""scType-based cell type annotation component.

scType is a marker-based cell type annotation method that uses predefined
marker gene sets to classify cells without requiring a reference dataset.

Reference: Ianevski et al. Nature Communications 2022
https://github.com/IasanevtchikVladimir/sctype
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import issparse

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


# Default marker genes for common cell types (can be extended)
DEFAULT_MARKERS = {
    # Epithelial
    "Epithelial": {
        "positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1"],
        "negative": ["PTPRC", "VIM", "PECAM1"],
    },
    # Immune - T cells
    "T cells": {
        "positive": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
        "negative": ["CD19", "CD14", "NCAM1"],
    },
    "CD4+ T cells": {
        "positive": ["CD3D", "CD3E", "CD4", "IL7R"],
        "negative": ["CD8A", "CD8B"],
    },
    "CD8+ T cells": {
        "positive": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB"],
        "negative": ["CD4"],
    },
    "Regulatory T cells": {
        "positive": ["CD3D", "CD4", "FOXP3", "IL2RA", "CTLA4"],
        "negative": ["CD8A"],
    },
    # Immune - B cells
    "B cells": {
        "positive": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5"],
        "negative": ["CD3D", "CD14"],
    },
    "Plasma cells": {
        "positive": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "XBP1"],
        "negative": ["MS4A1", "CD19"],
    },
    # Immune - Myeloid
    "Macrophages": {
        "positive": ["CD68", "CD163", "CSF1R", "MARCO", "MSR1"],
        "negative": ["CD3D", "CD19"],
    },
    "Monocytes": {
        "positive": ["CD14", "FCGR3A", "CSF1R", "LYZ", "S100A8"],
        "negative": ["CD3D", "CD19", "CD68"],
    },
    "Dendritic cells": {
        "positive": ["ITGAX", "CD1C", "CLEC9A", "FLT3", "HLA-DRA"],
        "negative": ["CD3D", "CD14", "CD19"],
    },
    "Mast cells": {
        "positive": ["KIT", "TPSAB1", "CPA3", "MS4A2", "FCER1A"],
        "negative": ["CD3D", "CD14"],
    },
    # Immune - NK
    "NK cells": {
        "positive": ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRF1"],
        "negative": ["CD3D", "CD19"],
    },
    # Stromal
    "Fibroblasts": {
        "positive": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FAP"],
        "negative": ["PTPRC", "EPCAM", "PECAM1"],
    },
    "Myofibroblasts": {
        "positive": ["ACTA2", "TAGLN", "MYL9", "COL1A1"],
        "negative": ["PTPRC", "EPCAM"],
    },
    "Pericytes": {
        "positive": ["PDGFRB", "RGS5", "CSPG4", "ACTA2"],
        "negative": ["PECAM1", "PTPRC"],
    },
    "Adipocytes": {
        "positive": ["ADIPOQ", "LEP", "PLIN1", "FABP4"],
        "negative": ["PTPRC", "EPCAM"],
    },
    # Endothelial
    "Endothelial cells": {
        "positive": ["PECAM1", "VWF", "CDH5", "KDR", "CLDN5"],
        "negative": ["PTPRC", "EPCAM", "ACTA2"],
    },
    "Lymphatic endothelial": {
        "positive": ["PROX1", "LYVE1", "PDPN", "FLT4"],
        "negative": ["PTPRC", "EPCAM"],
    },
}

# Mapping scType labels to broad categories
SCTYPE_TO_BROAD = {
    "Epithelial": "Epithelial",
    "T cells": "Immune",
    "CD4+ T cells": "Immune",
    "CD8+ T cells": "Immune",
    "Regulatory T cells": "Immune",
    "B cells": "Immune",
    "Plasma cells": "Immune",
    "Macrophages": "Immune",
    "Monocytes": "Immune",
    "Dendritic cells": "Immune",
    "Mast cells": "Immune",
    "NK cells": "Immune",
    "Fibroblasts": "Stromal",
    "Myofibroblasts": "Stromal",
    "Pericytes": "Stromal",
    "Adipocytes": "Stromal",
    "Endothelial cells": "Endothelial",
    "Lymphatic endothelial": "Endothelial",
}


def calculate_sctype_scores(
    expr_matrix: np.ndarray,
    gene_names: list[str],
    markers: dict[str, dict[str, list[str]]],
) -> tuple[np.ndarray, list[str]]:
    """
    Calculate scType scores for each cell and cell type.

    Args:
        expr_matrix: Expression matrix (cells x genes), normalized log-transformed
        gene_names: List of gene names
        markers: Dictionary of cell type markers {type: {positive: [...], negative: [...]}}

    Returns:
        Tuple of (scores matrix [cells x types], cell type names)
    """
    n_cells = expr_matrix.shape[0]
    cell_types = list(markers.keys())
    n_types = len(cell_types)

    # Create gene name to index mapping
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}

    # Convert sparse to dense if needed
    if issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()

    scores = np.zeros((n_cells, n_types))

    for type_idx, cell_type in enumerate(cell_types):
        marker_info = markers[cell_type]
        positive_markers = marker_info.get("positive", [])
        negative_markers = marker_info.get("negative", [])

        # Get indices for available markers
        pos_indices = [gene_to_idx[g.upper()] for g in positive_markers if g.upper() in gene_to_idx]
        neg_indices = [gene_to_idx[g.upper()] for g in negative_markers if g.upper() in gene_to_idx]

        if not pos_indices:
            # No positive markers available for this type
            scores[:, type_idx] = -np.inf
            continue

        # Calculate positive score (mean expression of positive markers)
        pos_expr = expr_matrix[:, pos_indices]
        pos_score = np.mean(pos_expr, axis=1)

        # Calculate negative score (mean expression of negative markers)
        if neg_indices:
            neg_expr = expr_matrix[:, neg_indices]
            neg_score = np.mean(neg_expr, axis=1)
        else:
            neg_score = 0

        # Final score: positive - negative
        scores[:, type_idx] = pos_score - neg_score * 0.5

    return scores, cell_types


def assign_cell_types(
    scores: np.ndarray,
    cell_types: list[str],
    min_score: float = 0.0,
) -> tuple[list[str], list[float]]:
    """
    Assign cell types based on scType scores.

    Args:
        scores: Score matrix (cells x types)
        cell_types: List of cell type names
        min_score: Minimum score threshold for assignment

    Returns:
        Tuple of (assigned types, confidence scores)
    """
    # Get best type for each cell
    best_indices = np.argmax(scores, axis=1)
    best_scores = np.max(scores, axis=1)

    # Calculate confidence as relative score
    # (how much better than second best)
    sorted_scores = np.sort(scores, axis=1)
    second_best = sorted_scores[:, -2] if scores.shape[1] > 1 else np.zeros(len(best_scores))
    confidence = (best_scores - second_best) / (np.abs(best_scores) + 1e-10)
    confidence = np.clip(confidence, 0, 1)

    assigned_types = []
    for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
        if score >= min_score:
            assigned_types.append(cell_types[idx])
        else:
            assigned_types.append("Unknown")

    return assigned_types, confidence.tolist()


@register_annotator
class ScTypeAnnotator:
    """Cell type annotation using scType marker-based method.

    scType assigns cell types based on the expression of positive and
    negative marker genes without requiring a reference dataset.
    """

    name = "sctype"

    def __init__(
        self,
        config: AnnotationConfig | None = None,
        custom_markers: dict | None = None,
    ):
        """Initialize the scType annotator.

        Args:
            config: Annotation configuration
            custom_markers: Custom marker definitions (overrides defaults)
        """
        self.config = config or AnnotationConfig()
        self.markers = custom_markers or DEFAULT_MARKERS

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels using scType.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data (required)
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used)

        Returns:
            Annotation results with cell types and confidence
        """
        import scanpy as sc

        cfg = config or self.config

        # Need either adata or expression_path
        if adata is None and expression_path is None:
            raise ValueError("ScTypeAnnotator requires either adata or expression_path")

        # Load adata if needed
        if adata is None and expression_path is not None:
            adata = self._load_expression(expression_path)

        logger.info("Running scType marker-based annotation...")

        # Normalize if not already
        adata_norm = adata.copy()
        if "log1p" not in adata_norm.uns:
            sc.pp.normalize_total(adata_norm, target_sum=1e4)
            sc.pp.log1p(adata_norm)

        # Get expression matrix and gene names
        expr_matrix = adata_norm.X
        gene_names = list(adata_norm.var_names)

        # Check marker coverage
        available_genes = {g.upper() for g in gene_names}
        logger.info(f"Checking marker coverage in {len(available_genes)} genes...")

        for cell_type, marker_info in self.markers.items():
            pos_markers = marker_info.get("positive", [])
            pos_available = [m for m in pos_markers if m.upper() in available_genes]
            logger.debug(f"  {cell_type}: {len(pos_available)}/{len(pos_markers)} positive markers")

        # Calculate scores
        logger.info("Calculating scType scores...")
        scores, cell_types = calculate_sctype_scores(expr_matrix, gene_names, self.markers)

        # Assign cell types
        logger.info("Assigning cell types...")
        assigned_types, confidence = assign_cell_types(scores, cell_types, min_score=0.0)

        # Get cell IDs
        cell_ids = (
            adata.obs["cell_id"].values
            if "cell_id" in adata.obs
            else adata.obs_names.tolist()
        )

        # Build annotations DataFrame
        annotations_data = []
        for i, (cid, pred, conf) in enumerate(zip(cell_ids, assigned_types, confidence)):
            broad_cat = SCTYPE_TO_BROAD.get(pred, map_to_broad_category(pred))
            annotations_data.append({
                "cell_id": str(cid),
                "predicted_type": pred,
                "broad_category": broad_cat,
                "confidence": conf,
            })

        annotations_df = pl.DataFrame(annotations_data)

        # Filter out Unknown if configured (default: True = keep Unknown)
        include_unknown = getattr(cfg, "include_unknown", True)
        if not include_unknown:
            annotations_df = annotations_df.filter(pl.col("broad_category") != "Unknown")

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        # Calculate statistics
        n_annotated = annotations_df.height
        class_dist = dict(
            annotations_df.group_by("broad_category")
            .agg(pl.len().alias("count"))
            .iter_rows()
        )

        # Type distribution (top 10)
        type_dist = dict(
            annotations_df.group_by("predicted_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(10)
            .iter_rows()
        )

        logger.info(f"scType annotation complete: {n_annotated} cells annotated")
        logger.info(f"  Broad category distribution: {class_dist}")
        logger.info(f"  Top cell types: {type_dist}")

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "type_distribution": type_dist,
                "n_markers_used": len(self.markers),
            },
        )

    def _load_expression(self, path: Path) -> Any:
        """Load expression data from file."""
        import anndata as ad

        path = Path(path)

        if path.suffix in [".h5", ".h5ad"]:
            return ad.read_h5ad(path)
        elif path.suffix == ".zarr" or path.is_dir():
            return ad.read_zarr(path)
        else:
            raise ValueError(f"Unsupported expression format: {path}")


def run_sctype_standalone(
    data_path: Path,
    output_path: Path | None = None,
    custom_markers: dict | None = None,
) -> pl.DataFrame:
    """Run scType annotation as standalone script.

    Args:
        data_path: Path to expression data (h5ad, zarr, or 10x h5)
        output_path: Where to save results
        custom_markers: Custom marker definitions

    Returns:
        DataFrame with annotations
    """
    import anndata as ad
    import h5py

    # Load data
    data_path = Path(data_path)

    if data_path.suffix == ".h5" and not data_path.suffix == ".h5ad":
        # 10x format
        logger.info(f"Loading 10x H5 format from {data_path}")
        with h5py.File(data_path, 'r') as f:
            data = f['matrix/data'][:]
            indices = f['matrix/indices'][:]
            indptr = f['matrix/indptr'][:]
            shape = f['matrix/shape'][:]
            barcodes = [b.decode() for b in f['matrix/barcodes'][:]]
            genes = [g.decode() for g in f['matrix/features/name'][:]]

        from scipy.sparse import csc_matrix
        X = csc_matrix((data, indices, indptr), shape=shape).T
        adata = ad.AnnData(X=X)
        adata.obs_names = barcodes
        adata.var_names = genes
    else:
        adata = ad.read_h5ad(data_path)

    # Run annotation
    annotator = ScTypeAnnotator(custom_markers=custom_markers)
    result = annotator.annotate(adata=adata)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.annotations_df.write_parquet(output_path)
        logger.info(f"Saved annotations to {output_path}")

    return result.annotations_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="scType cell type annotation")
    parser.add_argument("data_path", type=Path, help="Path to expression data")
    parser.add_argument("--output", "-o", type=Path, help="Output parquet path")

    args = parser.parse_args()
    run_sctype_standalone(args.data_path, args.output)
