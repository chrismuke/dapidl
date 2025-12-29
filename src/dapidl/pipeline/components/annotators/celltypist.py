"""CellTypist-based cell type annotation component.

This module wraps the existing CellTypeAnnotator from dapidl.data.annotation,
providing a pipeline-compatible interface with standard AnnotationResult output.

Supports multiple CellTypist strategies:
- single: Use a single model
- consensus: Vote across multiple models
- hierarchical: Tissue-specific + specialized refinement
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    BROAD_CATEGORY_MAPPING,
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


@register_annotator
class CellTypistAnnotator:
    """Cell type annotation using CellTypist models.

    Wraps the existing CellTypeAnnotator to provide a pipeline-compatible
    interface. Supports single, consensus, and hierarchical strategies.
    """

    name = "celltypist"

    def __init__(self, config: AnnotationConfig | None = None):
        """Initialize the CellTypist annotator.

        Args:
            config: Annotation configuration. If None, uses defaults.
        """
        self.config = config or AnnotationConfig()
        self._annotator = None

    @property
    def annotator(self):
        """Lazy-load the underlying CellTypeAnnotator."""
        if self._annotator is None:
            from dapidl.data.annotation import CellTypeAnnotator

            self._annotator = CellTypeAnnotator(
                model_names=self.config.model_names,
                confidence_threshold=self.config.confidence_threshold,
                majority_voting=self.config.majority_voting,
                strategy=self.config.strategy,
                fine_grained=self.config.fine_grained,
                extended_consensus=self.config.extended_consensus,
            )
        return self._annotator

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells with type labels using CellTypist.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used for CellTypist)

        Returns:
            Annotation results with cell types and confidence
        """
        cfg = config or self.config

        # Need either adata or expression_path
        if adata is None and expression_path is None:
            raise ValueError(
                "CellTypistAnnotator requires either adata or expression_path"
            )

        # Load adata if needed
        if adata is None and expression_path is not None:
            adata = self._load_expression(expression_path)

        # Update annotator config if different
        if cfg != self.config:
            self._annotator = None
            self.config = cfg

        # Run annotation
        annotations_df = self.annotator.annotate(adata)

        # Convert pandas to polars if needed
        if hasattr(annotations_df, "to_pandas"):
            annotations_df = pl.from_pandas(annotations_df.to_pandas())
        elif not isinstance(annotations_df, pl.DataFrame):
            annotations_df = pl.from_pandas(annotations_df)

        # Add broad category if not present
        if "broad_category" not in annotations_df.columns:
            predicted_col = (
                "predicted_type"
                if "predicted_type" in annotations_df.columns
                else annotations_df.columns[1]  # First non-cell_id column
            )
            annotations_df = annotations_df.with_columns(
                pl.col(predicted_col)
                .map_elements(map_to_broad_category, return_dtype=pl.Utf8)
                .alias("broad_category")
            )

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

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "strategy": cfg.strategy,
                "model_names": cfg.model_names,
            },
        )

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
        cfg = config or self.config

        # Create AnnData from reader
        adata = self.annotator.create_anndata(reader)

        return self.annotate(config=cfg, adata=adata)
