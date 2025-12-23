"""Native (pass-through) segmentation component.

This module implements a pass-through segmenter that uses existing
platform boundaries (Xenium or MERSCOPE built-in segmentation).

Use this when you want to use the original platform segmentation
without applying Cellpose or other custom methods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from dapidl.pipeline.base import SegmentationConfig, SegmentationResult
from dapidl.pipeline.registry import register_segmenter


@register_segmenter
class NativeSegmenter:
    """Pass-through segmenter using existing platform boundaries.

    Loads nucleus boundaries from the platform's output files:
    - Xenium: nucleus_boundaries.parquet
    - MERSCOPE: cell_boundaries with nucleus_boundary column

    No segmentation is performed - just converts existing boundaries
    to the standard SegmentationResult format.
    """

    name = "native"

    def __init__(self, config: SegmentationConfig | None = None):
        """Initialize the native segmenter.

        Args:
            config: Segmentation configuration. If None, uses defaults.
        """
        self.config = config or SegmentationConfig()

    def segment(
        self,
        dapi_image: np.ndarray,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Segment nuclei from DAPI image.

        Note: This method doesn't actually use the DAPI image.
        For native segmentation, use load_from_platform() instead.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16 - not used
            config: Override config for this call

        Returns:
            Empty result - use load_from_platform() instead
        """
        return SegmentationResult(
            centroids_df=pl.DataFrame(),
            boundaries_df=pl.DataFrame(),
            masks=None,
            matching_stats={"note": "Use load_from_platform() for native segmentation"},
        )

    def segment_and_match(
        self,
        dapi_image: np.ndarray,
        cells_df: pl.DataFrame,
        config: SegmentationConfig | None = None,
    ) -> SegmentationResult:
        """Segment nuclei and match to existing cell data.

        For native segmentation, cells are already matched by cell_id.
        This is a pass-through operation.

        Args:
            dapi_image: 2D DAPI image (H, W), uint16 - not used
            cells_df: Cell metadata DataFrame
            config: Override config for this call

        Returns:
            Result with all cells marked as matched
        """
        cfg = config or self.config

        # Build centroids from cells_df
        x_col = "x_centroid" if "x_centroid" in cells_df.columns else "centroid_x"
        y_col = "y_centroid" if "y_centroid" in cells_df.columns else "centroid_y"

        centroids_df = cells_df.select(
            [
                pl.col("cell_id"),
                pl.col(x_col).alias("centroid_x_um"),
                pl.col(y_col).alias("centroid_y_um"),
                pl.lit(True).alias("matched"),
            ]
        )

        n_cells = cells_df.height

        return SegmentationResult(
            centroids_df=centroids_df,
            boundaries_df=None,  # Load separately if needed
            masks=None,
            matching_stats={
                "n_cells": n_cells,
                "n_matched": n_cells,
                "match_rate": 1.0,
                "method": "native",
            },
        )

    def load_from_platform(
        self,
        data_path: Path,
        platform: str = "xenium",
    ) -> SegmentationResult:
        """Load segmentation from platform output files.

        Args:
            data_path: Path to platform output directory (e.g., xenium/outs/)
            platform: Platform type ('xenium' or 'merscope')

        Returns:
            Segmentation results loaded from platform files
        """
        data_path = Path(data_path)

        if platform.lower() == "xenium":
            return self._load_xenium(data_path)
        elif platform.lower() == "merscope":
            return self._load_merscope(data_path)
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def _load_xenium(self, data_path: Path) -> SegmentationResult:
        """Load Xenium nucleus boundaries."""
        # Load cell data for centroids
        cells_path = data_path / "cells.parquet"
        if not cells_path.exists():
            raise FileNotFoundError(f"Cells file not found: {cells_path}")

        cells_df = pl.read_parquet(cells_path)

        # Load nucleus boundaries
        boundaries_path = data_path / "nucleus_boundaries.parquet"
        boundaries_df = None
        if boundaries_path.exists():
            boundaries_df = pl.read_parquet(boundaries_path)

        # Build centroids DataFrame
        centroids_df = cells_df.select(
            [
                pl.col("cell_id"),
                pl.col("x_centroid").alias("centroid_x_um"),
                pl.col("y_centroid").alias("centroid_y_um"),
                pl.lit(True).alias("matched"),
            ]
        )

        return SegmentationResult(
            centroids_df=centroids_df,
            boundaries_df=boundaries_df,
            masks=None,
            matching_stats={
                "n_cells": cells_df.height,
                "n_matched": cells_df.height,
                "match_rate": 1.0,
                "platform": "xenium",
                "method": "native",
            },
        )

    def _load_merscope(self, data_path: Path) -> SegmentationResult:
        """Load MERSCOPE cell boundaries.

        Note: MERSCOPE doesn't always have separate nucleus boundaries.
        """
        # Find cell metadata file
        cell_meta_files = list(data_path.glob("**/cell_metadata*.csv"))
        if not cell_meta_files:
            raise FileNotFoundError(f"Cell metadata not found in {data_path}")

        cells_df = pl.read_csv(cell_meta_files[0])

        # Handle MERSCOPE column naming
        centroid_x = (
            "center_x"
            if "center_x" in cells_df.columns
            else cells_df.columns[1]  # Usually second column
        )
        centroid_y = (
            "center_y"
            if "center_y" in cells_df.columns
            else cells_df.columns[2]  # Usually third column
        )

        # Use EntityID as cell_id if available
        cell_id_col = (
            "EntityID"
            if "EntityID" in cells_df.columns
            else cells_df.columns[0]
        )

        centroids_df = cells_df.select(
            [
                pl.col(cell_id_col).alias("cell_id"),
                pl.col(centroid_x).alias("centroid_x_um"),
                pl.col(centroid_y).alias("centroid_y_um"),
                pl.lit(True).alias("matched"),
            ]
        )

        # Try to load cell boundaries
        boundaries_df = None
        boundary_files = list(data_path.glob("**/cell_boundaries*.parquet"))
        if boundary_files:
            boundaries_df = pl.read_parquet(boundary_files[0])

        return SegmentationResult(
            centroids_df=centroids_df,
            boundaries_df=boundaries_df,
            masks=None,
            matching_stats={
                "n_cells": cells_df.height,
                "n_matched": cells_df.height,
                "match_rate": 1.0,
                "platform": "merscope",
                "method": "native",
            },
        )


def get_platform_pixel_size(platform: str) -> float:
    """Get pixel size in microns for a platform."""
    pixel_sizes = {
        "xenium": 0.2125,  # ~4.7 pixels/um
        "merscope": 0.108,  # ~9.3 pixels/um
    }
    return pixel_sizes.get(platform.lower(), 0.2125)
