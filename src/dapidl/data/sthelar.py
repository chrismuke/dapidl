"""STHELAR dataset reader for loading DAPI images and cell metadata from zarr.

Reads SpatialData zarr objects from the STHELAR dataset
(Giraud-Sauveur et al., 2025) directly via zarr — no spatialdata dependency.

STHELAR provides co-registered DAPI + H&E + spatial transcriptomics
for 31 Xenium FFPE sections across 16 tissue types.

Zarr structure (nested):
    sdata_breast_s0.zarr/sdata_breast_s0.zarr/
        images/morpho/{0..4}    — DAPI pyramid, (1, H, W) uint16
        images/he/{0..7}        — H&E pyramid, (3, H, W) uint8
        tables/table_nuclei/    — Nucleus-level annotations (DAPI-based)
            obs/                — label1, label2, label3, final_label, ct_tangram, PanNuke_label
            obsm/spatial        — (N, 2) centroid coordinates in microns
        tables/table_cells/     — Cell-level annotations (CellViT H&E-based)
            obs/                — label1, label2, label3, final_label, PanNuke_label
            obsm/spatial        — (N, 2) centroid coordinates in microns
        shapes/nucleus_boundaries/
        shapes/cell_boundaries/
        points/st/              — Spatial transcripts
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zarr
from loguru import logger

# STHELAR group label mapping (from cells_final_label_group)
# WARNING: These group labels are noisy — Leiden refinement misassigns cells.
# Prefer TANGRAM_TO_COARSE for direct fine-grained mapping.
STHELAR_TO_DAPIDL_COARSE = {
    "Epithelial": "Epithelial",
    "Blood_vessel": "Endothelial",
    "Endothelial": "Endothelial",
    "Fibroblast_Myofibroblast": "Stromal",
    "Fibroblast": "Stromal",
    "CAF": "Stromal",
    "Myeloid": "Immune",
    "Monocyte/Macrophage": "Immune",
    "T_NK": "Immune",
    "T": "Immune",
    "B_Plasma": "Immune",
    "B": "Immune",
    "NK": "Immune",
    "Specialized": "Unknown",
    "Melanocyte": "Unknown",
    "Glioblastoma": "Unknown",
    "Adipocyte": "Unknown",
    "Mast": "Immune",
    "Other": "Unknown",
    "Less10": "Unknown",
    "less10": "Unknown",
    "Unknown": "Unknown",
}

# Biologically correct mapping from raw Tangram fine-grained labels.
# These are more accurate than STHELAR's Leiden-refined group labels.
TANGRAM_TO_COARSE = {
    # Epithelial (mammary)
    "CXCL14 mammary basal cell": "Epithelial",
    "SFN mammary luminal progenitor": "Epithelial",
    "KRT6B mammary basal cell": "Epithelial",
    "Secretoglobin mammary luminal progenitor": "Epithelial",
    "Cycling mammary luminal progenitor": "Epithelial",
    "CCSER1 mammary basal cell": "Epithelial",
    "SCGB3A1 mammary luminal progenitor": "Epithelial",
    "PIP mammary luminal cell": "Epithelial",
    "SAA2 mammary luminal progenitor": "Epithelial",
    "KRT17 mammary luminal cell": "Epithelial",
    "Secretoglobin mammary luminal cell": "Epithelial",
    "Lactocyte": "Epithelial",
    # Endothelial
    "Venous EC": "Endothelial",
    "Capillary EC": "Endothelial",
    "Arterial EC": "Endothelial",
    "Lymphatic EC": "Endothelial",
    # Stromal
    "CXCL+ fibroblast": "Stromal",
    "IGFBP6+APOD+ fibroblast": "Stromal",
    "IGFBP6+SFRP4+ fibroblast": "Stromal",
    "CCL19/21 pericyte": "Stromal",
    "Pericyte": "Stromal",
    "CXCL+ pericyte": "Stromal",
    "CREB+MT1A+ vascular smooth muscle cell": "Stromal",
    "Vascular smooth muscle cell": "Stromal",
    # Immune
    "Monocyte": "Immune",
    "Dendritic cell": "Immune",
    "pDC": "Immune",
    "Macrophage": "Immune",
    "M1 macrophage": "Immune",
    "LYVE1 macrophage": "Immune",
    "CD4 T cell": "Immune",
    "GZMK CD8 T cell": "Immune",
    "GZMB CD8 T cell": "Immune",
    "NK cell": "Immune",
    "B cell": "Immune",
    "Plasma cell": "Immune",
    "Treg cell": "Immune",
    "ILC": "Immune",
    "Mast cell": "Immune",
}

XENIUM_PIXEL_SIZE_UM = 0.2125


class SthelarDataReader:
    """Reader for STHELAR SpatialData zarr objects.

    Loads DAPI images, cell/nucleus annotations, and centroids directly
    from zarr — no spatialdata dependency required.

    Attributes:
        zarr_path: Path to the inner zarr directory
        dapi_level: Pyramid level (0=full res ~54K x 48K, 4=smallest)
    """

    PIXEL_SIZE = XENIUM_PIXEL_SIZE_UM  # µm/pixel (same as Xenium)

    def __init__(
        self,
        zarr_path: str | Path,
        dapi_level: int = 0,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        self.dapi_level = dapi_level

        # Resolve nested zarr: outer/inner or direct path
        self._root_path = self._resolve_zarr_root()
        self._validate_paths()

        self._store: zarr.Group | None = None
        self._dapi: np.ndarray | None = None
        self._nucleus_df: pl.DataFrame | None = None
        self._cell_df: pl.DataFrame | None = None

    def _resolve_zarr_root(self) -> Path:
        """Resolve the nested zarr layout (outer.zarr/inner.zarr/)."""
        # Direct path to inner zarr
        if (self.zarr_path / "images").exists():
            return self.zarr_path

        # Nested: outer.zarr/inner.zarr/
        stem = self.zarr_path.stem
        inner = self.zarr_path / f"{stem}.zarr"
        if inner.exists() and (inner / "images").exists():
            return inner

        # Also try the name without .zarr extension
        inner2 = self.zarr_path / stem
        if inner2.exists() and (inner2 / "images").exists():
            return inner2

        return self.zarr_path

    def _validate_paths(self) -> None:
        """Validate that required zarr groups exist."""
        if not self._root_path.exists():
            raise FileNotFoundError(f"Zarr root not found: {self._root_path}")

        required = ["images/morpho", "tables/table_nuclei"]
        for group_path in required:
            if not (self._root_path / group_path).exists():
                raise FileNotFoundError(
                    f"Required zarr group not found: {self._root_path / group_path}"
                )

        logger.info(f"STHELAR data validated at {self._root_path}")

    @property
    def store(self) -> zarr.Group:
        """Open zarr store (lazy)."""
        if self._store is None:
            self._store = zarr.open(str(self._root_path), mode="r")
        return self._store

    @property
    def name(self) -> str:
        """Slide name (e.g., 'sdata_breast_s0')."""
        return self.zarr_path.stem

    @property
    def image(self) -> np.ndarray:
        """Load and return the DAPI morphology image.

        Returns:
            2D numpy array (H, W) of uint16 DAPI intensities
        """
        if self._dapi is None:
            self._dapi = self._load_dapi()
        return self._dapi

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return (height, width) of DAPI image without loading it fully."""
        arr = self.store["images"]["morpho"][str(self.dapi_level)]
        # Shape is (1, H, W)
        return (arr.shape[1], arr.shape[2])

    @property
    def nucleus_df(self) -> pl.DataFrame:
        """Nucleus-level annotations from table_nuclei."""
        if self._nucleus_df is None:
            self._nucleus_df = self._load_table("table_nuclei")
        return self._nucleus_df

    @property
    def cells_df(self) -> pl.DataFrame:
        """Cell-level annotations from table_cells (CellViT H&E-based).

        Falls back to nucleus_df if table_cells doesn't exist.
        """
        if self._cell_df is None:
            if "table_cells" in self.store["tables"]:
                self._cell_df = self._load_table("table_cells")
            else:
                logger.warning("table_cells not found, using table_nuclei")
                self._cell_df = self.nucleus_df
        return self._cell_df

    @property
    def transcripts_df(self) -> pl.DataFrame:
        """Lazy-load spatial transcripts from zarr points/st.

        Returns:
            Polars DataFrame with x (pixels), y (pixels), gene columns
        """
        if not hasattr(self, "_transcripts_df") or self._transcripts_df is None:
            from starpose.io.transcripts import load_sthelar_transcripts

            self._transcripts_df = load_sthelar_transcripts(
                self._root_path, pixel_size=self.PIXEL_SIZE,
            )
            logger.info(f"Loaded {self._transcripts_df.height:,} transcripts")
        return self._transcripts_df

    @property
    def num_cells(self) -> int:
        return len(self.nucleus_df)

    def _load_dapi(self) -> np.ndarray:
        """Load DAPI image from morpho pyramid at specified level."""
        morpho = self.store["images"]["morpho"]
        level_key = str(self.dapi_level)

        if level_key not in morpho:
            available = sorted(morpho.keys(), key=int)
            raise ValueError(f"Pyramid level {self.dapi_level} not found. Available: {available}")

        arr = morpho[level_key]
        logger.info(f"Loading DAPI level {self.dapi_level}: shape={arr.shape}, dtype={arr.dtype}")

        # Shape is (1, H, W) — squeeze the channel dimension
        dapi = arr[0]  # triggers zarr read into memory
        logger.info(
            f"DAPI loaded: {dapi.shape[0]}x{dapi.shape[1]}, range=[{dapi.min()}, {dapi.max()}]"
        )
        return dapi

    def load_dapi_region(self, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
        """Load a rectangular region of the DAPI image (tiled access).

        Args:
            y_start, y_end: Row slice (height)
            x_start, x_end: Column slice (width)

        Returns:
            2D numpy array (h, w) of uint16
        """
        arr = self.store["images"]["morpho"][str(self.dapi_level)]
        return arr[0, y_start:y_end, x_start:x_end]

    def load_he(self, level: int | None = None) -> np.ndarray:
        """Load H&E image at specified pyramid level.

        Returns:
            3D numpy array (3, H, W) of uint8
        """
        level = level if level is not None else self.dapi_level
        he = self.store["images"]["he"]
        level_key = str(level)

        if level_key not in he:
            available = sorted(he.keys(), key=int)
            raise ValueError(f"H&E pyramid level {level} not found. Available: {available}")

        arr = he[level_key]
        logger.info(f"Loading H&E level {level}: shape={arr.shape}")
        return arr[:]

    def load_he_region(
        self,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        level: int | None = None,
    ) -> np.ndarray:
        """Load a rectangular region of the H&E image.

        Returns:
            3D numpy array (3, h, w) of uint8
        """
        level = level if level is not None else self.dapi_level
        arr = self.store["images"]["he"][str(level)]
        return arr[:, y_start:y_end, x_start:x_end]

    def _read_categorical(self, group: zarr.Group) -> np.ndarray:
        """Read a categorical zarr group (categories + codes) into string array."""
        categories = group["categories"][:]
        codes = group["codes"][:]
        # Negative codes indicate NaN/missing
        result = np.array(
            [categories[c] if c >= 0 else "Unknown" for c in codes],
            dtype=object,
        )
        return result

    def _load_table(self, table_name: str) -> pl.DataFrame:
        """Load a STHELAR annotation table as polars DataFrame.

        Reads obs columns and obsm/spatial coordinates, converts
        micron coords to pixel coords at the current pyramid level.
        """
        table = self.store["tables"][table_name]
        obs = table["obs"]

        logger.info(f"Loading {table_name} ({obs['_index'].shape[0]} entries)")

        # Start with cell IDs
        data: dict[str, Any] = {
            "cell_id": obs["cell_id"][:],
        }

        # Read spatial coordinates (microns)
        spatial = table["obsm"]["spatial"][:]  # (N, 2)
        data["x_centroid"] = spatial[:, 0]
        data["y_centroid"] = spatial[:, 1]

        # Convert to pixel coordinates at requested pyramid level
        scale_factor = 2**self.dapi_level
        pixel_size = self.PIXEL_SIZE * scale_factor
        data["x_centroid_px"] = spatial[:, 0] / pixel_size
        data["y_centroid_px"] = spatial[:, 1] / pixel_size

        # Read annotation columns — handle both plain arrays and categoricals
        label_cols = ["label1", "label2", "label3", "final_label", "PanNuke_label"]
        for col in label_cols:
            if col in obs:
                arr = obs[col]
                if isinstance(arr, zarr.Group):
                    # Categorical encoding
                    data[col] = self._read_categorical(arr)
                else:
                    data[col] = arr[:]

        # Read ct_tangram if present (categorical in nucleus table)
        if "ct_tangram" in obs:
            arr = obs["ct_tangram"]
            if isinstance(arr, zarr.Group):
                data["ct_tangram"] = self._read_categorical(arr)
            else:
                data["ct_tangram"] = arr[:]

        # Read numeric columns
        numeric_cols = {
            "area": "area",
            "cell_area": "cell_area",
            "nucleus_area": "nucleus_area",
            "PanNuke_proba": "PanNuke_proba",
            "transcript_counts": "transcript_counts",
        }
        for col, alias in numeric_cols.items():
            if col in obs:
                arr = obs[col]
                if not isinstance(arr, zarr.Group):
                    data[alias] = arr[:]

        df = pl.DataFrame(data)
        logger.info(f"Loaded {len(df)} entries from {table_name}")
        return df

    def get_centroids_pixels(self) -> np.ndarray:
        """Get nucleus centroids in pixel coordinates.

        Returns:
            Array of shape (N, 2) with (x, y) pixel coordinates
        """
        df = self.nucleus_df
        return np.column_stack([df["x_centroid_px"].to_numpy(), df["y_centroid_px"].to_numpy()])

    def get_centroids_microns(self) -> np.ndarray:
        """Get nucleus centroids in micron coordinates.

        Returns:
            Array of shape (N, 2) with (x, y) micron coordinates
        """
        df = self.nucleus_df
        return np.column_stack([df["x_centroid"].to_numpy(), df["y_centroid"].to_numpy()])

    def get_cell_ids(self) -> np.ndarray:
        """Get array of cell/nucleus IDs."""
        return self.nucleus_df["cell_id"].to_numpy()

    def get_labels_coarse(self) -> list[str]:
        """Get coarse DAPIDL labels (Epithelial/Immune/Stromal/Endothelial/Unknown).

        Uses Tangram labels (ct_tangram) with TANGRAM_TO_COARSE mapping when
        available, falling back to STHELAR group labels via STHELAR_TO_DAPIDL_COARSE.
        """
        df = self.nucleus_df

        # Prefer ct_tangram (biologically correct) over STHELAR Leiden labels
        if "ct_tangram" in df.columns:
            tangram = df["ct_tangram"].to_list()
            coarse = [TANGRAM_TO_COARSE.get(lab, "Unknown") for lab in tangram]

            # Fallback to group labels where tangram gives Unknown
            if "final_label" in df.columns:
                final = df["final_label"].to_list()
                for i, lab in enumerate(coarse):
                    if lab == "Unknown" and final[i] != "Unknown":
                        coarse[i] = STHELAR_TO_DAPIDL_COARSE.get(final[i], "Unknown")
            return coarse

        # No tangram — use final_label with group mapping
        if "final_label" in df.columns:
            final = df["final_label"].to_list()
            return [STHELAR_TO_DAPIDL_COARSE.get(lab, "Unknown") for lab in final]

        return ["Unknown"] * len(df)

    def get_available_pyramid_levels(self) -> dict[str, list[int]]:
        """List available pyramid levels for morpho and H&E images."""
        result = {}
        for img_name in ["morpho", "he"]:
            if img_name in self.store["images"]:
                levels = sorted(int(k) for k in self.store["images"][img_name])
                result[img_name] = levels
        return result

    def get_experiment_metadata(self) -> dict[str, Any]:
        """Return basic metadata about this slide."""
        h, w = self.image_shape
        return {
            "platform": "STHELAR",
            "name": self.name,
            "path": str(self.zarr_path),
            "dapi_level": self.dapi_level,
            "image_height": h,
            "image_width": w,
            "num_nuclei": self.store["tables"]["table_nuclei"]["obs"]["_index"].shape[0],
            "pyramid_levels": self.get_available_pyramid_levels(),
        }

    def __repr__(self) -> str:
        n_cells = self.num_cells if self._nucleus_df is not None else "not loaded"
        shape = self._dapi.shape if self._dapi is not None else "not loaded"
        return (
            f"SthelarDataReader(name={self.name}, "
            f"level={self.dapi_level}, "
            f"cells={n_cells}, "
            f"image_shape={shape})"
        )
