"""Marker Gene Validation for Cell Type Annotations.

Validates cell type predictions by checking marker gene expression.
Uses canonical markers from CellMarker database and literature.

This is a GT-FREE validation method - no ground truth required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    import anndata as ad


# Canonical marker genes for breast tissue cell types
# Sources: CellMarker database, PanglaoDB, literature
BREAST_MARKERS = {
    # Epithelial cells
    "Epithelial": {
        "positive": ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "ELF5"],
        "negative": ["PTPRC", "VIM"],  # Should NOT express these
    },
    "Epithelial_Luminal": {
        "positive": ["KRT8", "KRT18", "KRT19", "GATA3", "ESR1", "PGR", "FOXA1"],
        "negative": ["KRT5", "KRT14", "TP63"],
    },
    "Epithelial_Basal": {
        "positive": ["KRT5", "KRT14", "KRT17", "TP63", "EGFR"],
        "negative": ["ESR1", "PGR"],
    },
    "Tumor": {
        "positive": ["EPCAM", "KRT8", "KRT18", "MKI67"],  # High proliferation
        "negative": [],
    },
    # Immune cells
    "Immune": {
        "positive": ["PTPRC", "CD45"],  # PTPRC = CD45
        "negative": ["EPCAM", "COL1A1"],
    },
    "T_Cell": {
        "positive": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
        "negative": ["CD19", "CD14"],
    },
    "CD4_T_Cell": {
        "positive": ["CD3D", "CD4", "IL7R", "CCR7"],
        "negative": ["CD8A", "CD8B"],
    },
    "CD8_T_Cell": {
        "positive": ["CD3D", "CD8A", "CD8B", "GZMB", "PRF1"],
        "negative": ["CD4"],
    },
    "B_Cell": {
        "positive": ["MS4A1", "CD19", "CD79A", "CD79B", "PAX5"],
        "negative": ["CD3D", "CD14"],
    },
    "Plasma_Cell": {
        "positive": ["SDC1", "MZB1", "JCHAIN", "IGHG1"],
        "negative": ["MS4A1"],
    },
    "NK_Cell": {
        "positive": ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRB1"],
        "negative": ["CD3D"],
    },
    "Macrophage": {
        "positive": ["CD68", "CD163", "CSF1R", "MARCO", "MRC1"],
        "negative": ["CD3D", "MS4A1"],
    },
    "Monocyte": {
        "positive": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN"],
        "negative": ["CD3D", "FCGR3A"],
    },
    "Dendritic_Cell": {
        "positive": ["ITGAX", "HLA-DRA", "FCER1A", "CD1C"],
        "negative": ["CD14", "CD3D"],
    },
    "Mast_Cell": {
        "positive": ["KIT", "TPSAB1", "TPSB2", "CPA3", "MS4A2"],
        "negative": ["CD3D", "CD14"],
    },
    "Neutrophil": {
        "positive": ["FCGR3B", "CSF3R", "S100A8", "S100A9"],
        "negative": ["CD3D", "CD14"],
    },
    # Stromal cells
    "Stromal": {
        "positive": ["VIM", "COL1A1", "COL1A2"],
        "negative": ["EPCAM", "PTPRC"],
    },
    "Fibroblast": {
        "positive": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FAP"],
        "negative": ["EPCAM", "PTPRC", "PECAM1"],
    },
    "CAF": {  # Cancer-associated fibroblast
        "positive": ["FAP", "ACTA2", "PDPN", "COL1A1", "POSTN"],
        "negative": ["EPCAM"],
    },
    "Pericyte": {
        "positive": ["RGS5", "PDGFRB", "CSPG4", "ACTA2", "MCAM"],
        "negative": ["PECAM1", "CDH5"],
    },
    "Smooth_Muscle": {
        "positive": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES"],
        "negative": ["PECAM1"],
    },
    "Adipocyte": {
        "positive": ["ADIPOQ", "LEP", "FABP4", "PLIN1", "LPL"],
        "negative": ["COL1A1"],
    },
    # Endothelial cells
    "Endothelial": {
        "positive": ["PECAM1", "VWF", "CDH5", "CLDN5", "KDR", "ERG"],
        "negative": ["EPCAM", "PTPRC", "ACTA2"],
    },
    "Lymphatic_Endothelial": {
        "positive": ["PROX1", "LYVE1", "PDPN", "FLT4"],
        "negative": ["PECAM1"],  # Low or absent
    },
}


@dataclass
class MarkerValidationResult:
    """Results from marker gene validation."""

    cell_type: str
    n_cells: int
    markers_found: list[str]
    markers_missing: list[str]

    # Positive markers
    positive_enrichment: float  # Mean expression vs background
    positive_fraction: float    # Fraction of cells expressing markers

    # Negative markers (should NOT be expressed)
    negative_leakage: float     # Fraction expressing "wrong" markers

    # Combined score
    marker_score: float  # 0-1, higher = better

    details: dict = field(default_factory=dict)


def compute_marker_scores(
    adata: "ad.AnnData",
    predictions: np.ndarray,
    markers_db: dict[str, dict[str, list[str]]] = BREAST_MARKERS,
    min_cells: int = 50,
) -> dict[str, MarkerValidationResult]:
    """Compute marker validation scores for each predicted cell type.

    Args:
        adata: AnnData with expression data (log-normalized)
        predictions: Cell type predictions (N,)
        markers_db: Marker gene database
        min_cells: Minimum cells per type to evaluate

    Returns:
        Dictionary mapping cell type to validation result
    """
    results = {}

    # Get available genes
    available_genes = set(adata.var_names)

    # Get unique predicted types
    unique_types = np.unique(predictions)

    for cell_type in unique_types:
        # Skip unknown/unlabeled
        if cell_type.lower() in ["unknown", "unlabeled", "unassigned"]:
            continue

        # Find matching marker set
        markers = _find_matching_markers(cell_type, markers_db)
        if markers is None:
            logger.debug(f"No markers found for {cell_type}")
            continue

        # Get cells of this type
        mask = predictions == cell_type
        n_cells = mask.sum()

        if n_cells < min_cells:
            logger.debug(f"Skipping {cell_type}: only {n_cells} cells (min={min_cells})")
            continue

        # Find available positive markers
        positive_markers = [m for m in markers.get("positive", []) if m in available_genes]
        negative_markers = [m for m in markers.get("negative", []) if m in available_genes]

        if not positive_markers:
            logger.debug(f"No positive markers available for {cell_type}")
            continue

        # Compute positive marker enrichment
        pos_enrichment, pos_fraction = _compute_positive_enrichment(
            adata, mask, positive_markers
        )

        # Compute negative marker leakage
        neg_leakage = _compute_negative_leakage(
            adata, mask, negative_markers
        ) if negative_markers else 0.0

        # Combined score: high positive enrichment, low negative leakage
        # Score = positive_enrichment * (1 - negative_leakage)
        marker_score = pos_enrichment * pos_fraction * (1 - neg_leakage)
        marker_score = np.clip(marker_score, 0, 1)

        results[cell_type] = MarkerValidationResult(
            cell_type=cell_type,
            n_cells=int(n_cells),
            markers_found=positive_markers,
            markers_missing=[m for m in markers.get("positive", []) if m not in available_genes],
            positive_enrichment=float(pos_enrichment),
            positive_fraction=float(pos_fraction),
            negative_leakage=float(neg_leakage),
            marker_score=float(marker_score),
            details={
                "negative_markers_checked": negative_markers,
            }
        )

        logger.info(
            f"  {cell_type}: score={marker_score:.3f} "
            f"(enrich={pos_enrichment:.2f}, frac={pos_fraction:.2f}, leak={neg_leakage:.2f})"
        )

    return results


def _find_matching_markers(
    cell_type: str,
    markers_db: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]] | None:
    """Find matching marker set for a cell type."""
    cell_type_lower = cell_type.lower().replace("_", " ").replace("-", " ")

    # Try exact match first
    for db_type, markers in markers_db.items():
        if db_type.lower() == cell_type_lower:
            return markers

    # Try partial match
    for db_type, markers in markers_db.items():
        db_type_lower = db_type.lower().replace("_", " ")
        if db_type_lower in cell_type_lower or cell_type_lower in db_type_lower:
            return markers

    # Try keyword matching
    keywords = {
        "epithelial": "Epithelial",
        "immune": "Immune",
        "stromal": "Stromal",
        "fibroblast": "Fibroblast",
        "endothelial": "Endothelial",
        "macrophage": "Macrophage",
        "t cell": "T_Cell",
        "b cell": "B_Cell",
        "nk": "NK_Cell",
        "dendritic": "Dendritic_Cell",
        "mast": "Mast_Cell",
        "plasma": "Plasma_Cell",
        "tumor": "Tumor",
        "luminal": "Epithelial_Luminal",
        "basal": "Epithelial_Basal",
        "pericyte": "Pericyte",
        "adipocyte": "Adipocyte",
    }

    for keyword, db_type in keywords.items():
        if keyword in cell_type_lower:
            return markers_db.get(db_type)

    return None


def _compute_positive_enrichment(
    adata: "ad.AnnData",
    mask: np.ndarray,
    markers: list[str],
) -> tuple[float, float]:
    """Compute positive marker enrichment and expressing fraction."""
    # Get expression for markers
    X = adata[:, markers].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Mean expression in target vs background
    target_expr = X[mask].mean(axis=0)
    background_expr = X[~mask].mean(axis=0)

    # Enrichment ratio (avoid division by zero)
    enrichment = np.mean(target_expr / (background_expr + 0.01))
    enrichment = np.clip(enrichment / 5.0, 0, 1)  # Normalize: 5x enrichment = score 1.0

    # Fraction of cells expressing at least one marker
    expressing = (X[mask] > 0).any(axis=1).mean()

    return float(enrichment), float(expressing)


def _compute_negative_leakage(
    adata: "ad.AnnData",
    mask: np.ndarray,
    markers: list[str],
) -> float:
    """Compute fraction of cells expressing negative markers."""
    if not markers:
        return 0.0

    # Get expression for negative markers
    X = adata[:, markers].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Fraction expressing any negative marker (above threshold)
    threshold = 0.5  # log-normalized expression threshold
    leakage = (X[mask] > threshold).any(axis=1).mean()

    return float(leakage)


def validate_with_markers(
    adata: "ad.AnnData",
    predictions: np.ndarray,
    tissue_type: str = "breast",
) -> dict:
    """High-level marker validation function.

    Args:
        adata: AnnData with expression data
        predictions: Cell type predictions
        tissue_type: Tissue type for marker selection

    Returns:
        Validation results dictionary
    """
    logger.info(f"Running marker gene validation for {tissue_type} tissue...")

    # Select marker database based on tissue
    if tissue_type.lower() == "breast":
        markers_db = BREAST_MARKERS
    else:
        logger.warning(f"No specific markers for {tissue_type}, using breast markers")
        markers_db = BREAST_MARKERS

    # Compute scores
    results = compute_marker_scores(adata, predictions, markers_db)

    # Aggregate metrics
    if results:
        scores = [r.marker_score for r in results.values()]
        overall_score = np.mean(scores)

        # Weighted by cell count
        total_cells = sum(r.n_cells for r in results.values())
        weighted_score = sum(r.marker_score * r.n_cells for r in results.values()) / total_cells
    else:
        overall_score = 0.0
        weighted_score = 0.0

    return {
        "overall_marker_score": float(overall_score),
        "weighted_marker_score": float(weighted_score),
        "n_types_validated": len(results),
        "per_type_results": {k: vars(v) for k, v in results.items()},
    }
