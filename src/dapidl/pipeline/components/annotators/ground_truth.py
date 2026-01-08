"""Ground truth annotation component.

This module loads cell type annotations from curated ground truth files
(Excel, CSV, or Parquet). Used for supervised training datasets like
the Janesick breast cancer dataset.

Supports:
- Excel files with specified sheet names
- CSV files
- Parquet files
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    GROUND_TRUTH_MAPPING,
    get_class_names,
)
from dapidl.pipeline.registry import register_annotator


@register_annotator
class GroundTruthAnnotator:
    """Load cell type annotations from curated ground truth files.

    Supports Excel, CSV, and Parquet formats. Maps fine-grained cell
    types to broad categories using GROUND_TRUTH_MAPPING.
    """

    name = "ground_truth"

    def __init__(self, config: AnnotationConfig | None = None):
        """Initialize the ground truth annotator.

        Args:
            config: Annotation configuration with ground_truth_file path
        """
        self.config = config or AnnotationConfig()

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Load annotations from ground truth file.

        Args:
            config: Override config for this call
            adata: Not used for ground truth (ignored)
            expression_path: Not used for ground truth (ignored)
            cells_df: Optional cell metadata to filter/join

        Returns:
            Annotation results with cell types from ground truth
        """
        cfg = config or self.config

        if not cfg.ground_truth_file:
            raise ValueError(
                "GroundTruthAnnotator requires ground_truth_file in config"
            )

        gt_path = Path(cfg.ground_truth_file)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        # Load annotations based on file type
        annotations_df = self._load_file(gt_path, cfg)

        # Identify cell_id and cell_type columns
        cell_id_col = self._find_column(
            annotations_df,
            cfg.cell_id_column,
            ["cell_id", "Cell_Barcode", "barcode", "Barcode", "cell_ID", "CellID"],
        )
        celltype_col = self._find_column(
            annotations_df,
            cfg.celltype_column,
            ["cell_type", "Cell_Type", "celltype", "type", "cluster", "Cluster", "annotation", "Annotation"],
        )

        # Rename columns to standard names
        annotations_df = annotations_df.rename(
            {cell_id_col: "cell_id", celltype_col: "predicted_type"}
        )

        # Add broad category
        annotations_df = annotations_df.with_columns(
            pl.col("predicted_type")
            .map_elements(self._map_to_broad, return_dtype=pl.Utf8)
            .alias("broad_category")
        )

        # Filter out excluded categories
        excluded = ["Hybrid", "Unlabeled", "Unknown"]
        initial_count = annotations_df.height
        annotations_df = annotations_df.filter(
            ~pl.col("broad_category").is_in(excluded)
        )
        filtered_count = initial_count - annotations_df.height

        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} cells with excluded categories")

        # Filter by category if specified
        if cfg.fine_grained and hasattr(cfg, "filter_category") and cfg.filter_category:
            annotations_df = annotations_df.filter(
                pl.col("broad_category") == cfg.filter_category
            )
            logger.info(
                f"Filtered to {cfg.filter_category}: {annotations_df.height} cells"
            )

        # Join with cells_df if provided (to get spatial coordinates, etc.)
        if cells_df is not None:
            # Ensure cell_id types match
            cells_df = cells_df.with_columns(
                pl.col("cell_id").cast(annotations_df["cell_id"].dtype)
            )
            annotations_df = annotations_df.join(
                cells_df.select(["cell_id"]),
                on="cell_id",
                how="inner",
            )
            logger.info(f"Joined with cells: {annotations_df.height} matched")

        # Add confidence column (1.0 for ground truth)
        if "confidence" not in annotations_df.columns:
            annotations_df = annotations_df.with_columns(
                pl.lit(1.0).alias("confidence")
            )

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

        fine_dist = {}
        if cfg.fine_grained:
            fine_dist = dict(
                annotations_df.group_by("predicted_type")
                .agg(pl.len().alias("count"))
                .iter_rows()
            )

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "fine_grained_distribution": fine_dist,
                "source_file": str(gt_path),
                "source_sheet": cfg.ground_truth_sheet,
            },
        )

    def _load_file(
        self, path: Path, cfg: AnnotationConfig
    ) -> pl.DataFrame:
        """Load annotations from file based on extension."""
        if path.suffix in [".xlsx", ".xls"]:
            return self._load_excel(path, cfg.ground_truth_sheet)
        elif path.suffix == ".csv":
            return pl.read_csv(path)
        elif path.suffix == ".parquet":
            return pl.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_excel(
        self, path: Path, sheet_name: str | None = None
    ) -> pl.DataFrame:
        """Load annotations from Excel file."""
        import pandas as pd

        # Default sheet names for Xenium breast datasets
        default_sheets = [
            "Xenium R1 Fig1-5 (supervised)",
            "Xenium R2 Fig1-5 (supervised)",
            "Sheet1",
        ]

        sheets_to_try = [sheet_name] if sheet_name else default_sheets

        for sheet in sheets_to_try:
            try:
                df = pd.read_excel(path, sheet_name=sheet)
                logger.info(f"Loaded sheet '{sheet}' from {path.name}")
                return pl.from_pandas(df)
            except Exception:
                continue

        raise ValueError(
            f"Could not load any sheet from {path}. "
            f"Tried: {sheets_to_try}"
        )

    def _find_column(
        self,
        df: pl.DataFrame,
        preferred: str | None,
        candidates: list[str],
    ) -> str:
        """Find a column by preferred name or candidates (case-insensitive).

        Matching priority:
        1. Exact match for preferred name
        2. Case-insensitive match for preferred name
        3. Case-insensitive match for candidates (in order)

        Args:
            df: DataFrame to search
            preferred: User-specified column name (highest priority)
            candidates: List of common column name variations

        Returns:
            Actual column name from DataFrame

        Raises:
            ValueError: If no matching column found
        """
        # Build case-insensitive lookup: lowercase -> actual column name
        col_lookup = {col.lower(): col for col in df.columns}

        # 1. Check preferred name (exact match for backwards compatibility)
        if preferred and preferred in df.columns:
            return preferred

        # 2. Check preferred name (case-insensitive)
        if preferred and preferred.lower() in col_lookup:
            return col_lookup[preferred.lower()]

        # 3. Check candidates (case-insensitive, preserves priority order)
        for candidate in candidates:
            if candidate.lower() in col_lookup:
                return col_lookup[candidate.lower()]

        raise ValueError(
            f"Could not find column. "
            f"Preferred: {preferred}, Candidates: {candidates}, "
            f"Available: {list(df.columns)}"
        )

    def _map_to_broad(self, cell_type: str) -> str:
        """Map cell type to broad category using ground truth mapping."""
        if cell_type in GROUND_TRUTH_MAPPING:
            return GROUND_TRUTH_MAPPING[cell_type]

        # Fall back to checking if it's already a broad category
        if cell_type in ["Epithelial", "Immune", "Stromal"]:
            return cell_type

        return "Unknown"
