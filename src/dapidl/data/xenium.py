"""Xenium data reader for loading DAPI images and cell metadata."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import tifffile
from loguru import logger


class XeniumDataReader:
    """Reader for 10x Xenium spatial transcriptomics data.

    Loads DAPI morphology images, cell centroids, and gene expression data
    from Xenium output directories.

    Attributes:
        xenium_path: Path to Xenium output directory
        pixel_size: Microns per pixel (default 0.2125 for Xenium)
    """

    PIXEL_SIZE = 0.2125  # µm/pixel for Xenium

    def __init__(self, xenium_path: str | Path) -> None:
        """Initialize Xenium data reader.

        Args:
            xenium_path: Path to Xenium output directory (containing 'outs' folder)
        """
        self.xenium_path = Path(xenium_path)
        self._validate_paths()
        self._image: np.ndarray | None = None
        self._cells_df: pl.DataFrame | None = None
        self._expression_matrix: np.ndarray | None = None
        self._gene_names: list[str] | None = None
        self._cell_ids: np.ndarray | None = None

    def _validate_paths(self) -> None:
        """Validate that required files exist."""
        outs_path = self._get_outs_path()

        required_files = [
            "morphology_focus.ome.tif",
            "cells.parquet",
            "cell_feature_matrix.h5",
        ]

        for f in required_files:
            if not (outs_path / f).exists():
                raise FileNotFoundError(f"Required file not found: {outs_path / f}")

        logger.info(f"Xenium data validated at {self.xenium_path}")

    def _get_outs_path(self) -> Path:
        """Get path to outs directory."""
        # Handle both direct outs path and parent path
        if (self.xenium_path / "outs").exists():
            return self.xenium_path / "outs"
        elif self.xenium_path.name == "outs":
            return self.xenium_path
        else:
            # Assume xenium_path is the outs directory
            return self.xenium_path

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
        """Load DAPI morphology image from OME-TIFF."""
        image_path = self._get_outs_path() / "morphology_focus.ome.tif"
        logger.info(f"Loading DAPI image from {image_path}")

        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()

        logger.info(f"Loaded image with shape {image.shape}, dtype {image.dtype}")
        return image

    def _load_cells(self) -> pl.DataFrame:
        """Load cell metadata from parquet."""
        cells_path = self._get_outs_path() / "cells.parquet"
        logger.info(f"Loading cell data from {cells_path}")

        df = pl.read_parquet(cells_path)

        # Add pixel coordinates (convert from microns)
        df = df.with_columns(
            (pl.col("x_centroid") / self.PIXEL_SIZE).alias("x_centroid_px"),
            (pl.col("y_centroid") / self.PIXEL_SIZE).alias("y_centroid_px"),
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
            [df["x_centroid"].to_numpy(), df["y_centroid"].to_numpy()]
        )

    def get_cell_ids(self) -> np.ndarray:
        """Get array of cell IDs."""
        return self.cells_df["cell_id"].to_numpy()

    def load_expression_matrix(self) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Load gene expression matrix from H5 file.

        Returns:
            Tuple of (expression_matrix, gene_names, cell_ids)
            - expression_matrix: shape (n_cells, n_genes), sparse converted to dense
            - gene_names: list of gene names
            - cell_ids: array of cell IDs matching matrix rows
        """
        import h5py
        from scipy import sparse

        h5_path = self._get_outs_path() / "cell_feature_matrix.h5"
        logger.info(f"Loading expression matrix from {h5_path}")

        with h5py.File(h5_path, "r") as f:
            # Get matrix data
            # 10x stores in CSC format: shape is (genes, cells), indptr has (n_cells+1) entries
            matrix_group = f["matrix"]
            data = matrix_group["data"][:]
            indices = matrix_group["indices"][:]
            indptr = matrix_group["indptr"][:]
            shape = matrix_group["shape"][:]

            # Create CSC matrix (genes x cells) and transpose to get (cells x genes)
            sparse_matrix = sparse.csc_matrix(
                (data, indices, indptr), shape=(shape[0], shape[1])
            )
            expression_matrix = sparse_matrix.T.toarray()  # (cells x genes)

            # Get gene names
            features = f["matrix/features"]
            gene_names = [name.decode() for name in features["name"][:]]

            # Get cell barcodes/IDs
            barcodes = f["matrix/barcodes"][:]
            # Barcodes are stored as strings like "1", "2", etc.
            cell_ids = np.array([int(b.decode()) for b in barcodes])

        logger.info(
            f"Loaded expression matrix: {expression_matrix.shape[0]} cells x {expression_matrix.shape[1]} genes"
        )

        self._expression_matrix = expression_matrix
        self._gene_names = gene_names
        self._cell_ids = cell_ids

        return expression_matrix, gene_names, cell_ids

    def get_experiment_metadata(self) -> dict[str, Any]:
        """Load experiment metadata from JSON file."""
        import json

        metadata_path = self._get_outs_path() / "experiment.xenium"
        if not metadata_path.exists():
            return {}

        with open(metadata_path) as f:
            return json.load(f)

    def __repr__(self) -> str:
        cells = self.num_cells if self._cells_df is not None else "not loaded"
        img_shape = self._image.shape if self._image is not None else "not loaded"
        return (
            f"XeniumDataReader(path={self.xenium_path}, "
            f"cells={cells}, "
            f"image_shape={img_shape})"
        )
