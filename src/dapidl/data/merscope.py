"""MERSCOPE/Vizgen data reader for loading DAPI images and cell metadata."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tifffile
from loguru import logger


class MerscopeDataReader:
    """Reader for Vizgen MERSCOPE spatial transcriptomics data.

    Loads DAPI morphology images, cell centroids, and gene expression data
    from MERSCOPE output directories.

    Attributes:
        merscope_path: Path to MERSCOPE output directory
        z_index: Z-slice index to use for DAPI image (default: 3)
    """

    def __init__(self, merscope_path: str | Path, z_index: int = 3) -> None:
        """Initialize MERSCOPE data reader.

        Args:
            merscope_path: Path to MERSCOPE output directory
            z_index: Z-slice index for DAPI image (default 3, typically best focus)
        """
        self.merscope_path = Path(merscope_path)
        self.z_index = z_index
        self._validate_paths()
        self._image: np.ndarray | None = None
        self._cells_df: pl.DataFrame | None = None
        self._expression_matrix: np.ndarray | None = None
        self._gene_names: list[str] | None = None
        self._cell_ids: np.ndarray | None = None
        self._transform_matrix: np.ndarray | None = None

    def _validate_paths(self) -> None:
        """Validate that required files exist."""
        required_files = [
            "cell_metadata.csv",
            "cell_by_gene.csv",
        ]

        for f in required_files:
            if not (self.merscope_path / f).exists():
                raise FileNotFoundError(
                    f"Required file not found: {self.merscope_path / f}"
                )

        # Check for images directory and DAPI image
        images_dir = self.merscope_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        dapi_pattern = f"mosaic_DAPI_z{self.z_index}.tif"
        if not (images_dir / dapi_pattern).exists():
            # Try to find any DAPI z-slice
            dapi_files = list(images_dir.glob("mosaic_DAPI_z*.tif"))
            if not dapi_files:
                raise FileNotFoundError(
                    f"No DAPI images found in {images_dir}"
                )
            # Use the first available z-slice
            self.z_index = int(dapi_files[0].stem.split("_z")[-1])
            logger.warning(f"Z-index {self.z_index} not found, using z{self.z_index}")

        transform_path = images_dir / "micron_to_mosaic_pixel_transform.csv"
        if not transform_path.exists():
            raise FileNotFoundError(f"Transform file not found: {transform_path}")

        logger.info(f"MERSCOPE data validated at {self.merscope_path}")

    def _load_transform(self) -> np.ndarray:
        """Load the micron-to-pixel affine transform matrix."""
        transform_path = self.merscope_path / "images" / "micron_to_mosaic_pixel_transform.csv"

        # Read space-separated values as 3x3 matrix
        transform = np.loadtxt(transform_path)

        if transform.shape != (3, 3):
            raise ValueError(f"Expected 3x3 transform matrix, got {transform.shape}")

        logger.info(f"Loaded transform: scale_x={transform[0,0]:.4f}, scale_y={transform[1,1]:.4f}")
        return transform

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get the micron-to-pixel affine transform matrix."""
        if self._transform_matrix is None:
            self._transform_matrix = self._load_transform()
        return self._transform_matrix

    @property
    def image(self) -> np.ndarray:
        """Load and return the DAPI morphology image.

        Returns:
            2D numpy array (H, W) of uint16 DAPI intensities
        """
        if self._image is None:
            self._image = self._load_image()
        return self._image

    @property
    def cells_df(self) -> pl.DataFrame:
        """Load and return cell metadata.

        Returns:
            Polars DataFrame with cell_id, centroids, and metadata
        """
        if self._cells_df is None:
            self._cells_df = self._load_cells()
        return self._cells_df

    @property
    def num_cells(self) -> int:
        """Return number of cells."""
        return len(self.cells_df)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (height, width) of DAPI image."""
        return self.image.shape

    def _load_image(self) -> np.ndarray:
        """Load DAPI morphology image from TIFF."""
        image_path = self.merscope_path / "images" / f"mosaic_DAPI_z{self.z_index}.tif"
        logger.info(f"Loading DAPI image from {image_path}")

        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()

        logger.info(f"Loaded image with shape {image.shape}, dtype {image.dtype}")
        return image

    def _load_cells(self) -> pl.DataFrame:
        """Load cell metadata from CSV."""
        cells_path = self.merscope_path / "cell_metadata.csv"
        logger.info(f"Loading cell data from {cells_path}")

        df = pl.read_csv(cells_path)

        # The first column is unnamed and contains cell IDs
        # Rename it to 'cell_id' for consistency
        first_col = df.columns[0]
        if first_col == "" or first_col.startswith("Unnamed"):
            df = df.rename({first_col: "cell_id"})
        elif first_col != "cell_id":
            # Assume first column is cell_id regardless of name
            df = df.rename({first_col: "cell_id"})

        # Ensure we have the expected coordinate columns
        if "center_x" not in df.columns or "center_y" not in df.columns:
            raise ValueError(
                f"Expected 'center_x' and 'center_y' columns, got: {df.columns}"
            )

        # Add pixel coordinates using the transform matrix
        transform = self.transform_matrix
        scale_x, tx = transform[0, 0], transform[0, 2]
        scale_y, ty = transform[1, 1], transform[1, 2]

        df = df.with_columns(
            (pl.col("center_x") * scale_x + tx).alias("x_centroid_px"),
            (pl.col("center_y") * scale_y + ty).alias("y_centroid_px"),
            # Add aliases for Xenium-compatible column names
            pl.col("center_x").alias("x_centroid"),
            pl.col("center_y").alias("y_centroid"),
        )

        logger.info(f"Loaded {len(df)} cells")
        return df

    def get_centroids_pixels(self) -> np.ndarray:
        """Get cell centroids in pixel coordinates.

        Returns:
            Array of shape (N, 2) with (x, y) pixel coordinates
        """
        df = self.cells_df
        return np.column_stack(
            [df["x_centroid_px"].to_numpy(), df["y_centroid_px"].to_numpy()]
        )

    def get_centroids_microns(self) -> np.ndarray:
        """Get cell centroids in micron coordinates.

        Returns:
            Array of shape (N, 2) with (x, y) micron coordinates
        """
        df = self.cells_df
        return np.column_stack(
            [df["center_x"].to_numpy(), df["center_y"].to_numpy()]
        )

    def get_cell_ids(self) -> np.ndarray:
        """Get array of cell IDs."""
        return self.cells_df["cell_id"].to_numpy()

    def load_expression_matrix(self) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Load gene expression matrix from CSV file.

        Returns:
            Tuple of (expression_matrix, gene_names, cell_ids)
            - expression_matrix: shape (n_cells, n_genes), dense array
            - gene_names: list of gene names
            - cell_ids: array of cell IDs matching matrix rows
        """
        csv_path = self.merscope_path / "cell_by_gene.csv"
        logger.info(f"Loading expression matrix from {csv_path}")

        # Read the CSV - first column is cell ID, rest are genes
        df = pl.read_csv(csv_path)

        # First column is cell ID
        first_col = df.columns[0]
        cell_ids = df[first_col].to_numpy()

        # All other columns are gene counts
        gene_names = df.columns[1:]
        expression_matrix = df.select(gene_names).to_numpy().astype(np.float32)

        logger.info(
            f"Loaded expression matrix: {expression_matrix.shape[0]} cells x {expression_matrix.shape[1]} genes"
        )

        self._expression_matrix = expression_matrix
        self._gene_names = list(gene_names)
        self._cell_ids = cell_ids

        return expression_matrix, list(gene_names), cell_ids

    def get_experiment_metadata(self) -> dict[str, Any]:
        """Load experiment metadata if available.

        MERSCOPE doesn't have a standard metadata file like Xenium,
        so we return basic info about the dataset.
        """
        metadata = {
            "platform": "MERSCOPE",
            "path": str(self.merscope_path),
            "z_index": self.z_index,
            "num_cells": self.num_cells if self._cells_df is not None else None,
        }

        # Try to get image dimensions without loading full image
        if self._image is not None:
            metadata["image_shape"] = self._image.shape

        return metadata

    def __repr__(self) -> str:
        cells = self.num_cells if self._cells_df is not None else "not loaded"
        img_shape = self._image.shape if self._image is not None else "not loaded"
        return (
            f"MerscopeDataReader(path={self.merscope_path}, "
            f"z_index={self.z_index}, "
            f"cells={cells}, "
            f"image_shape={img_shape})"
        )


def detect_platform(path: str | Path) -> str:
    """Detect whether a path contains Xenium or MERSCOPE data.

    Args:
        path: Path to spatial transcriptomics data directory

    Returns:
        'xenium' or 'merscope'

    Raises:
        ValueError: If platform cannot be determined
    """
    path = Path(path)

    # Check for Xenium-specific files
    xenium_markers = [
        "morphology_focus.ome.tif",
        "cells.parquet",
        "cell_feature_matrix.h5",
    ]

    # Check both direct path and outs subdirectory
    for check_path in [path, path / "outs"]:
        if check_path.exists():
            xenium_count = sum(
                1 for f in xenium_markers if (check_path / f).exists()
            )
            if xenium_count >= 2:
                return "xenium"

    # Check for MERSCOPE-specific files
    merscope_markers = [
        "cell_metadata.csv",
        "cell_by_gene.csv",
    ]
    merscope_count = sum(1 for f in merscope_markers if (path / f).exists())

    # Also check for images directory with DAPI
    images_dir = path / "images"
    if images_dir.exists() and list(images_dir.glob("mosaic_DAPI_z*.tif")):
        merscope_count += 1

    if merscope_count >= 2:
        return "merscope"

    raise ValueError(
        f"Cannot determine platform for {path}. "
        "Expected Xenium files (morphology_focus.ome.tif, cells.parquet, cell_feature_matrix.h5) "
        "or MERSCOPE files (cell_metadata.csv, cell_by_gene.csv, images/mosaic_DAPI_z*.tif)"
    )


def create_reader(path: str | Path, **kwargs) -> "XeniumDataReader | MerscopeDataReader":
    """Create appropriate data reader based on detected platform.

    Args:
        path: Path to spatial transcriptomics data directory
        **kwargs: Additional arguments passed to the reader

    Returns:
        XeniumDataReader or MerscopeDataReader instance
    """
    from dapidl.data.xenium import XeniumDataReader

    platform = detect_platform(path)

    if platform == "xenium":
        return XeniumDataReader(path)
    else:
        return MerscopeDataReader(path, **kwargs)
