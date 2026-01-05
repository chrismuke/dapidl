"""SCINA-based cell type annotation component.

SCINA (Semi-supervised Category Identification and Assignment) uses marker genes
and an EM algorithm to identify cell types without requiring reference data.

Reference: Zhang et al. Genes 2019
https://github.com/jcao89757/SCINA
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from scipy.special import softmax
from scipy.stats import zscore

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


# Default marker genes for cell types (similar to scType but for SCINA format)
DEFAULT_SIGNATURES = {
    # Epithelial
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1"],
    # Immune - T cells
    "T_cells": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
    "CD4_T_cells": ["CD3D", "CD3E", "CD4", "IL7R"],
    "CD8_T_cells": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB"],
    "Tregs": ["CD3D", "CD4", "FOXP3", "IL2RA", "CTLA4"],
    # Immune - B cells
    "B_cells": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5"],
    "Plasma_cells": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "XBP1"],
    # Immune - Myeloid
    "Macrophages": ["CD68", "CD163", "CSF1R", "MARCO", "MSR1"],
    "Monocytes": ["CD14", "FCGR3A", "CSF1R", "LYZ", "S100A8"],
    "Dendritic_cells": ["ITGAX", "CD1C", "CLEC9A", "FLT3", "HLA-DRA"],
    "Mast_cells": ["KIT", "TPSAB1", "CPA3", "MS4A2", "FCER1A"],
    # Immune - NK
    "NK_cells": ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRF1"],
    # Stromal
    "Fibroblasts": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA", "FAP"],
    "Myofibroblasts": ["ACTA2", "TAGLN", "MYL9", "COL1A1"],
    "Pericytes": ["PDGFRB", "RGS5", "CSPG4", "ACTA2"],
    "Adipocytes": ["ADIPOQ", "LEP", "PLIN1", "FABP4"],
    # Endothelial
    "Endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "CLDN5"],
    "Lymphatic_endothelial": ["PROX1", "LYVE1", "PDPN", "FLT4"],
}

# Mapping SCINA labels to broad categories
SCINA_TO_BROAD = {
    "Epithelial": "Epithelial",
    "T_cells": "Immune",
    "CD4_T_cells": "Immune",
    "CD8_T_cells": "Immune",
    "Tregs": "Immune",
    "B_cells": "Immune",
    "Plasma_cells": "Immune",
    "Macrophages": "Immune",
    "Monocytes": "Immune",
    "Dendritic_cells": "Immune",
    "Mast_cells": "Immune",
    "NK_cells": "Immune",
    "Fibroblasts": "Stromal",
    "Myofibroblasts": "Stromal",
    "Pericytes": "Stromal",
    "Adipocytes": "Stromal",
    "Endothelial": "Endothelial",
    "Lymphatic_endothelial": "Endothelial",
}


def run_scina(
    expr_matrix: np.ndarray,
    gene_names: list[str],
    signatures: dict[str, list[str]],
    max_iter: int = 100,
    convergence_threshold: float = 1e-4,
    rm_overlap: bool = True,
) -> tuple[list[str], list[float]]:
    """Run SCINA algorithm for cell type assignment.

    SCINA uses an EM algorithm to assign cells to types based on marker expression.

    Args:
        expr_matrix: Normalized log-expression matrix (cells x genes)
        gene_names: List of gene names
        signatures: Dict of cell_type -> [marker genes]
        max_iter: Maximum EM iterations
        convergence_threshold: Convergence threshold for EM
        rm_overlap: Remove overlapping markers between cell types

    Returns:
        Tuple of (predicted labels, confidence scores)
    """
    n_cells, n_genes = expr_matrix.shape
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}

    # Filter signatures to available genes
    filtered_sigs = {}
    for cell_type, markers in signatures.items():
        available = [m for m in markers if m.upper() in gene_to_idx]
        if len(available) >= 2:  # Need at least 2 markers
            filtered_sigs[cell_type] = available
        else:
            logger.debug(f"Skipping {cell_type}: only {len(available)} markers available")

    if not filtered_sigs:
        logger.warning("No cell types have sufficient markers")
        return ["Unknown"] * n_cells, [0.0] * n_cells

    # Remove overlapping markers if requested
    if rm_overlap:
        all_markers = {}
        for ct, markers in filtered_sigs.items():
            for m in markers:
                if m not in all_markers:
                    all_markers[m] = []
                all_markers[m].append(ct)

        # Keep marker only for cell type with fewest total markers
        for marker, cell_types in all_markers.items():
            if len(cell_types) > 1:
                # Keep in cell type with fewest markers
                keep_ct = min(cell_types, key=lambda x: len(filtered_sigs[x]))
                for ct in cell_types:
                    if ct != keep_ct and marker in filtered_sigs[ct]:
                        filtered_sigs[ct].remove(marker)

        # Re-filter
        filtered_sigs = {ct: m for ct, m in filtered_sigs.items() if len(m) >= 2}

    logger.info(f"SCINA using {len(filtered_sigs)} cell types with filtered markers")

    cell_types = list(filtered_sigs.keys())
    n_types = len(cell_types)

    # Initialize parameters
    # Prior probabilities (uniform)
    pi = np.ones(n_types) / n_types

    # Get marker expression matrices
    marker_expr = {}
    for ct, markers in filtered_sigs.items():
        indices = [gene_to_idx[m.upper()] for m in markers]
        marker_expr[ct] = expr_matrix[:, indices]

    # Z-score normalize marker expression
    for ct in marker_expr:
        marker_expr[ct] = zscore(marker_expr[ct], axis=0)
        marker_expr[ct] = np.nan_to_num(marker_expr[ct], nan=0.0)

    # Initialize responsibilities (random)
    gamma = np.random.dirichlet(np.ones(n_types), n_cells)

    # EM algorithm
    prev_log_lik = -np.inf

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        log_prob = np.zeros((n_cells, n_types))

        for k, ct in enumerate(cell_types):
            # Score based on mean marker expression
            scores = marker_expr[ct].mean(axis=1)
            log_prob[:, k] = np.log(pi[k] + 1e-10) + scores

        # Normalize
        log_prob = log_prob - log_prob.max(axis=1, keepdims=True)
        gamma = softmax(log_prob, axis=1)

        # M-step: update pi
        pi = gamma.mean(axis=0)
        pi = pi / pi.sum()

        # Check convergence
        log_lik = np.sum(np.log(gamma.max(axis=1) + 1e-10))
        if abs(log_lik - prev_log_lik) < convergence_threshold:
            logger.info(f"SCINA converged at iteration {iteration}")
            break
        prev_log_lik = log_lik

    # Assign labels based on maximum responsibility
    assignments = gamma.argmax(axis=1)
    confidence = gamma.max(axis=1)

    # Low confidence -> Unknown
    min_confidence = 0.3
    labels = []
    final_confidence = []

    for i in range(n_cells):
        if confidence[i] >= min_confidence:
            labels.append(cell_types[assignments[i]])
            final_confidence.append(confidence[i])
        else:
            labels.append("Unknown")
            final_confidence.append(confidence[i])

    return labels, final_confidence


@register_annotator
class SCINAAnnotator:
    """Cell type annotation using SCINA marker-based EM algorithm.

    SCINA uses an expectation-maximization approach to assign cells
    to types based on marker gene signatures.
    """

    name = "scina"

    def __init__(
        self,
        config: AnnotationConfig | None = None,
        custom_signatures: dict | None = None,
        max_iter: int = 100,
        rm_overlap: bool = True,
    ):
        """Initialize the SCINA annotator.

        Args:
            config: Annotation configuration
            custom_signatures: Custom marker signatures (overrides defaults)
            max_iter: Maximum EM iterations
            rm_overlap: Remove overlapping markers between types
        """
        self.config = config or AnnotationConfig()
        self.signatures = custom_signatures or DEFAULT_SIGNATURES
        self.max_iter = max_iter
        self.rm_overlap = rm_overlap

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels using SCINA.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data (required)
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used)

        Returns:
            Annotation results with cell types and confidence
        """
        import scanpy as sc
        from scipy.sparse import issparse

        cfg = config or self.config

        # Need either adata or expression_path
        if adata is None and expression_path is None:
            raise ValueError("SCINAAnnotator requires either adata or expression_path")

        # Load adata if needed
        if adata is None and expression_path is not None:
            adata = self._load_expression(expression_path)

        logger.info("Running SCINA marker-based annotation...")

        # Normalize if not already
        adata_norm = adata.copy()
        if "log1p" not in adata_norm.uns:
            sc.pp.normalize_total(adata_norm, target_sum=1e4)
            sc.pp.log1p(adata_norm)

        # Get expression matrix
        expr_matrix = adata_norm.X
        if issparse(expr_matrix):
            expr_matrix = expr_matrix.toarray()

        gene_names = list(adata_norm.var_names)

        # Check marker coverage
        available_genes = {g.upper() for g in gene_names}
        logger.info(f"Checking marker coverage in {len(available_genes)} genes...")

        for cell_type, markers in self.signatures.items():
            available = [m for m in markers if m.upper() in available_genes]
            logger.debug(f"  {cell_type}: {len(available)}/{len(markers)} markers")

        # Run SCINA
        logger.info("Running SCINA EM algorithm...")
        assigned_types, confidence = run_scina(
            expr_matrix,
            gene_names,
            self.signatures,
            max_iter=self.max_iter,
            rm_overlap=self.rm_overlap,
        )

        # Get cell IDs
        cell_ids = (
            adata.obs["cell_id"].values
            if "cell_id" in adata.obs
            else adata.obs_names.tolist()
        )

        # Build annotations DataFrame
        annotations_data = []
        for cid, pred, conf in zip(cell_ids, assigned_types, confidence):
            broad_cat = SCINA_TO_BROAD.get(pred, map_to_broad_category(pred))
            annotations_data.append({
                "cell_id": str(cid),
                "predicted_type": pred,
                "broad_category": broad_cat,
                "confidence": conf,
            })

        annotations_df = pl.DataFrame(annotations_data)

        # Filter out Unknown if configured
        include_unknown = getattr(cfg, "include_unknown", True)
        if not include_unknown:
            annotations_df = annotations_df.filter(pl.col("broad_category") != "Unknown")

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        # Calculate statistics
        n_annotated = annotations_df.height
        class_dist = (
            annotations_df.group_by("broad_category")
            .count()
            .to_pandas()
            .set_index("broad_category")["count"]
            .to_dict()
        )

        type_dist = (
            annotations_df.group_by("predicted_type")
            .count()
            .sort("count", descending=True)
            .head(10)
            .to_pandas()
            .set_index("predicted_type")["count"]
            .to_dict()
        )

        logger.info(f"SCINA annotation complete: {n_annotated} cells annotated")
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
                "n_signatures": len(self.signatures),
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCINA cell type annotation")
    parser.add_argument("data_path", type=Path, help="Path to expression data")
    parser.add_argument("--output", "-o", type=Path, help="Output parquet path")

    args = parser.parse_args()

    annotator = SCINAAnnotator()

    import anndata as ad
    adata = ad.read_h5ad(args.data_path) if args.data_path.suffix == ".h5ad" else None

    if adata is None:
        # Try loading as Xenium
        import h5py
        from scipy.sparse import csc_matrix

        with h5py.File(args.data_path, 'r') as f:
            data = f['matrix/data'][:]
            indices = f['matrix/indices'][:]
            indptr = f['matrix/indptr'][:]
            shape = f['matrix/shape'][:]
            barcodes = [b.decode() for b in f['matrix/barcodes'][:]]
            genes = [g.decode() for g in f['matrix/features/name'][:]]

        X = csc_matrix((data, indices, indptr), shape=shape).T
        adata = ad.AnnData(X=X)
        adata.obs_names = barcodes
        adata.var_names = genes

    result = annotator.annotate(adata=adata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.annotations_df.write_parquet(args.output)
        logger.info(f"Saved annotations to {args.output}")

    print(result.annotations_df.head(10))
