"""SingleR-based cell type annotation component.

This module wraps SingleR (R-based reference annotation) via rpy2,
providing a pipeline-compatible interface with standard AnnotationResult output.

SingleR is an R package that uses correlation-based assignment to annotate
cells against reference datasets. Benchmark results show:
- BlueprintEncodeData: 92.0% accuracy, 0.907 F1 (best)
- HumanPrimaryCellAtlas: 89.2% accuracy, 0.877 F1

Requires R with SingleR and celldex packages installed:
    R -e 'BiocManager::install(c("SingleR", "celldex"))'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


# Available SingleR reference datasets
SINGLER_REFERENCES = {
    "blueprint": "BlueprintEncodeData",  # Best for general use (92% accuracy)
    "hpca": "HumanPrimaryCellAtlasData",  # General human cell atlas
    "monaco": "MonacoImmuneData",  # Detailed immune subtypes
    "novershtern": "NovershternHematopoieticData",  # Hematopoietic lineages
}


def _fix_libstdcxx() -> None:
    """Fix libstdc++ version conflict between conda and system libraries.

    When miniconda is installed, its older libstdc++.so.6 may lack GLIBCXX_3.4.30
    required by system libraries like libicuuc.so.74. This function uses ctypes
    to preload the system libstdc++ before rpy2 tries to load R.

    If you encounter errors like:
        GLIBCXX_3.4.30 not found (required by /lib/x86_64-linux-gnu/libicuuc.so.74)

    Fix by updating miniconda's libstdcxx-ng:
        conda install -c conda-forge libstdcxx-ng
    """
    import ctypes

    system_libstdcxx = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if system_libstdcxx.exists():
        try:
            # Preload system libstdc++ with RTLD_GLOBAL to make symbols available
            ctypes.CDLL(str(system_libstdcxx), mode=ctypes.RTLD_GLOBAL)
            logger.debug(f"Preloaded system libstdc++ from {system_libstdcxx}")
        except OSError as e:
            logger.debug(f"Could not preload system libstdc++: {e}")


def is_singler_available() -> bool:
    """Check if rpy2 and SingleR R packages are available."""
    # Fix libstdc++ before importing rpy2
    _fix_libstdcxx()

    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        # Try to load the R packages
        importr("SingleR")
        importr("celldex")
        return True
    except Exception as e:
        logger.debug(f"SingleR not available: {e}")
        return False


@register_annotator
class SingleRAnnotator:
    """Cell type annotation using SingleR via rpy2.

    SingleR is an R-based reference-based annotation method that uses
    correlation to assign cell types. Supports multiple reference datasets
    from celldex package.

    Benchmark results on Xenium breast data:
    - BlueprintEncodeData: 92.0% accuracy, 0.907 macro F1
    - HumanPrimaryCellAtlas: 89.2% accuracy, 0.877 macro F1
    """

    name = "singler"

    def __init__(self, config: AnnotationConfig | None = None):
        """Initialize the SingleR annotator.

        Args:
            config: Annotation configuration. If None, uses defaults.
        """
        self.config = config or AnnotationConfig()
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if SingleR is available, warn if not."""
        if not is_singler_available():
            logger.warning(
                "SingleR not available. Requires R with SingleR and celldex packages. "
                "Install with: R -e 'BiocManager::install(c(\"SingleR\", \"celldex\"))'"
            )

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels using SingleR.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used for SingleR)

        Returns:
            Annotation results with cell types and confidence
        """
        cfg = config or self.config

        # Need either adata or expression_path
        if adata is None and expression_path is None:
            raise ValueError(
                "SingleRAnnotator requires either adata or expression_path"
            )

        # Load adata if needed
        if adata is None and expression_path is not None:
            adata = self._load_expression(expression_path)

        # Get reference (default to blueprint)
        reference = getattr(cfg, "singler_reference", "blueprint")
        if reference not in SINGLER_REFERENCES:
            logger.warning(f"Unknown reference {reference}, using blueprint")
            reference = "blueprint"

        # Run annotation via rpy2
        annotations_df = self._run_singler(adata, reference)

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        # Calculate statistics
        n_annotated = annotations_df.height
        class_dist = (
            annotations_df.group_by("broad_category")
            .len()
            .to_pandas()
            .set_index("broad_category")["len"]
            .to_dict()
        )

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "method": "singler",
                "reference": reference,
            },
        )

    def _run_singler(self, adata: Any, reference: str) -> pl.DataFrame:
        """Run SingleR annotation via rpy2.

        Args:
            adata: AnnData with expression data
            reference: Reference dataset name

        Returns:
            Polars DataFrame with annotations
        """
        import numpy as np
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        logger.info(f"Running SingleR annotation with {reference} reference...")

        # Import R packages
        singler = importr("SingleR")
        celldex = importr("celldex")

        # Get reference data
        ref_func_name = SINGLER_REFERENCES[reference]
        ref_func = getattr(celldex, ref_func_name)
        logger.info(f"Loading reference: {ref_func_name}")
        ref_data = ref_func()

        # Get expression matrix - use raw counts if available
        if adata.raw is not None:
            expr = adata.raw.X
            genes = list(adata.raw.var_names)
        else:
            expr = adata.X
            genes = list(adata.var_names)

        # Convert sparse to dense if needed
        if hasattr(expr, "toarray"):
            expr = expr.toarray()

        # Log-normalize expression (SingleR expects log-normalized)
        # Formula: log1p(counts / sum * 10000)
        lib_sizes = expr.sum(axis=1, keepdims=True)
        lib_sizes[lib_sizes == 0] = 1  # Avoid division by zero
        expr_norm = np.log1p(expr / lib_sizes * 10000)

        # Get reference gene names and find intersection
        ref_genes = list(ro.r.rownames(ref_data))
        common_genes = list(set(genes) & set(ref_genes))
        logger.info(
            f"Common genes: {len(common_genes)} / {len(genes)} "
            f"({100*len(common_genes)/len(genes):.1f}%)"
        )

        if len(common_genes) < 50:
            logger.warning(
                f"Low gene overlap ({len(common_genes)} genes). Results may be unreliable."
            )

        # Subset to common genes
        gene_indices = [genes.index(g) for g in common_genes]

        # Create matrix with dimnames using list (genes as rows, cells as columns)
        cell_names = [str(cid) for cid in adata.obs_names]
        dimnames = ro.r.list(ro.StrVector(common_genes), ro.StrVector(cell_names))
        expr_subset = ro.r.matrix(
            ro.FloatVector(expr_norm[:, gene_indices].T.flatten()),
            nrow=len(common_genes),
            ncol=expr_norm.shape[0],
            dimnames=dimnames,
        )

        ref_subset = ro.r["["](ref_data, ro.StrVector(common_genes), True)

        # Get labels from reference
        labels = ro.r("function(x) x$label.main")(ref_data)

        # Run SingleR with classic DE method (avoids scrapper dependency)
        logger.info("Running SingleR prediction...")
        results = singler.SingleR(
            test=expr_subset,
            ref=ref_subset,
            labels=labels,
            de_method="classic",
        )

        # Extract predictions
        pred_labels = list(ro.r("function(x) x$labels")(results))
        # Get max score as confidence (scores are correlation values)
        scores = np.array(ro.r("function(x) x$scores")(results))
        if len(scores.shape) == 1:
            confidences = scores
        else:
            confidences = scores.max(axis=1)
        # Normalize scores to 0-1 range (correlation can be negative)
        confidences = (confidences + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Map to broad categories
        broad_categories = [map_to_broad_category(label) for label in pred_labels]

        # Create output DataFrame
        result_df = pl.DataFrame(
            {
                "cell_id": list(adata.obs_names),
                "predicted_type": pred_labels,
                "broad_category": broad_categories,
                "confidence": confidences.tolist(),
                "singler_reference": [reference] * len(pred_labels),
            }
        )

        # Log distribution
        logger.info(f"SingleR annotation complete: {len(result_df)} cells")
        for cat, count in result_df.group_by("broad_category").len().iter_rows():
            logger.info(f"  {cat}: {count} ({100*count/len(result_df):.1f}%)")

        return result_df

    def _load_expression(self, path: Path) -> Any:
        """Load expression data from file.

        Supports h5, h5ad, and zarr formats.
        """
        import anndata as ad

        path = Path(path)

        if path.suffix in [".h5", ".h5ad"]:
            return ad.read_h5ad(path)
        elif path.suffix == ".zarr" or path.is_dir():
            return ad.read_zarr(path)
        else:
            raise ValueError(f"Unsupported expression format: {path}")

    def annotate_from_reader(
        self,
        reader: Any,  # XeniumDataReader or MerscopeDataReader
        config: AnnotationConfig | None = None,
    ) -> AnnotationResult:
        """Annotate using a data reader (convenience method).

        Args:
            reader: Data reader with load_expression_matrix() method
            config: Override config for this call

        Returns:
            Annotation results
        """
        import anndata as ad

        cfg = config or self.config

        # Create AnnData from reader
        expr, genes, cell_ids = reader.load_expression_matrix()
        adata = ad.AnnData(X=expr)
        adata.var_names = genes
        adata.obs_names = [str(cid) for cid in cell_ids]
        adata.obs["cell_id"] = cell_ids

        return self.annotate(config=cfg, adata=adata)
