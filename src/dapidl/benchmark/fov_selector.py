"""FOV selector for the segmentation benchmark.

Selects 5 representative Fields of View from a MERSCOPE / Xenium dataset by
picking tiles that cover different biological and imaging regimes:

    dense   — highest-density FOV (p95 density)
    sparse  — lowest-density FOV (p5 density)
    mixed   — median-density FOV (p50 density)
    edge    — FOV at the maximum pixel coordinate (tissue boundary)
    immune  — FOV with smallest mean cell volume (small nuclei → immune cells)

Coordinates are converted between microns and pixels using an affine transform
read from a MERSCOPE micron_to_mosaic_pixel_transform.csv file (3×3 matrix,
assumed to be a uniform-scale + translation mapping with no rotation).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default affine transform constants (MERSCOPE breast dataset)
# ---------------------------------------------------------------------------
DEFAULT_SCALE: float = 9.259259  # pixels per micron
DEFAULT_OFFSET_X: float = 357.2  # pixel x-offset
DEFAULT_OFFSET_Y: float = 2007.97  # pixel y-offset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FOVTile:
    """A representative tile selected from a spatial transcriptomics dataset.

    Attributes:
        fov_id: Integer FOV identifier from the dataset.
        label: Selection criterion used — one of
            ``dense``, ``sparse``, ``mixed``, ``edge``, ``immune``.
        n_cells: Number of cells in the FOV.
        density: Cell density in cells per 1000 µm².
        mean_volume: Mean cell volume within the FOV (same units as the input).
        pixel_bbox: Bounding box in pixel coordinates as
            ``(y_min, y_max, x_min, x_max)``.
        micron_bbox: Bounding box in micron coordinates as
            ``(x_min, x_max, y_min, y_max)``.
    """

    fov_id: int
    label: str
    n_cells: int
    density: float
    mean_volume: float
    pixel_bbox: tuple[int, int, int, int]
    micron_bbox: tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Affine transform helpers
# ---------------------------------------------------------------------------


def load_transform(path: str | Path) -> tuple[float, float, float]:
    """Parse a MERSCOPE 3×3 affine CSV and return (scale, offset_x, offset_y).

    The transform is assumed to be::

        | scale   0       offset_x |
        | 0       scale   offset_y |
        | 0       0       1        |

    Args:
        path: Path to ``micron_to_mosaic_pixel_transform.csv``.

    Returns:
        ``(scale, offset_x, offset_y)`` floats extracted from the matrix.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    matrix = [[float(v) for v in row] for row in rows if row]
    scale = matrix[0][0]
    offset_x = matrix[0][2]
    offset_y = matrix[1][2]
    return scale, offset_x, offset_y


def _micron_to_pixel(
    x_um: float,
    y_um: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> tuple[int, int]:
    """Convert a single (x, y) point from microns to pixel coordinates."""
    px = int(round(x_um * scale + offset_x))
    py = int(round(y_um * scale + offset_y))
    return px, py


def _micron_bbox_to_pixel_bbox(
    x_min_um: float,
    x_max_um: float,
    y_min_um: float,
    y_max_um: float,
    scale: float,
    offset_x: float,
    offset_y: float,
    padding_um: float = 10.0,
) -> tuple[int, int, int, int]:
    """Convert a micron bounding box to pixel (y_min, y_max, x_min, x_max)."""
    x_min_um -= padding_um
    x_max_um += padding_um
    y_min_um -= padding_um
    y_max_um += padding_um

    px_x_min, px_y_min = _micron_to_pixel(x_min_um, y_min_um, scale, offset_x, offset_y)
    px_x_max, px_y_max = _micron_to_pixel(x_max_um, y_max_um, scale, offset_x, offset_y)

    y_min = min(px_y_min, px_y_max)
    y_max = max(px_y_min, px_y_max)
    x_min = min(px_x_min, px_x_max)
    x_max = max(px_x_min, px_x_max)

    return (y_min, y_max, x_min, x_max)


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------


def select_fovs(
    cell_metadata: pl.DataFrame,
    n_fovs: int = 5,
    padding_um: float = 10.0,
    scale: float = DEFAULT_SCALE,
    offset_x: float = DEFAULT_OFFSET_X,
    offset_y: float = DEFAULT_OFFSET_Y,
) -> list[FOVTile]:
    """Select *n_fovs* representative FOVs from cell metadata.

    Selection strategy (in priority order; duplicates skipped):

    1. **dense**  — FOV at the 95th-percentile of cell density
    2. **sparse** — FOV at the  5th-percentile of cell density
    3. **mixed**  — FOV at the 50th-percentile (median) of cell density
    4. **edge**   — FOV with the largest pixel coordinate (tissue edge)
    5. **immune** — FOV with the smallest mean cell volume

    Args:
        cell_metadata: Polars DataFrame with columns
            ``fov, volume, center_x, center_y, min_x, max_x, min_y, max_y``.
            Spatial coordinates are in microns.
        n_fovs: Number of tiles to return (default 5).
        padding_um: Extra micron padding added to each side of the bounding box
            before conversion to pixels.
        scale: Pixels per micron (from affine transform).
        offset_x: Pixel x-offset (from affine transform).
        offset_y: Pixel y-offset (from affine transform).

    Returns:
        List of :class:`FOVTile` instances, up to *n_fovs* long.
    """
    # ------------------------------------------------------------------
    # Step 1 — compute per-FOV statistics
    # ------------------------------------------------------------------
    fov_stats = (
        cell_metadata.group_by("fov")
        .agg(
            pl.len().alias("n_cells"),
            pl.col("volume").mean().alias("mean_volume"),
            pl.col("min_x").min().alias("x_min"),
            pl.col("max_x").max().alias("x_max"),
            pl.col("min_y").min().alias("y_min"),
            pl.col("max_y").max().alias("y_max"),
        )
        .with_columns(
            (
                (pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))
            ).alias("area_um2")
        )
        .with_columns(
            (pl.col("n_cells") / pl.col("area_um2") * 1000.0).alias("density")
        )
        .sort("fov")
    )

    # Convert to Python-side for selection logic (small dataframe — fine)
    records = fov_stats.to_dicts()

    # ------------------------------------------------------------------
    # Step 2 — rank by density for percentile-based selection
    # ------------------------------------------------------------------
    records_by_density = sorted(records, key=lambda r: r["density"])

    # ------------------------------------------------------------------
    # Step 3 — build ordered candidate lists per label
    # ------------------------------------------------------------------
    # For density-based labels we build a ranked list so we can fall back to
    # the next-best candidate when a duplicate is encountered.
    records_by_density_desc = list(reversed(records_by_density))  # high → low

    def _ranked_density(high_to_low: bool) -> list[dict]:
        return records_by_density_desc if high_to_low else records_by_density

    # Sorted by max coordinate descending (edge) and mean_volume ascending (immune)
    records_by_edge = sorted(records, key=lambda r: max(r["x_max"], r["y_max"]), reverse=True)
    records_by_volume = sorted(records, key=lambda r: r["mean_volume"])

    def _median_ranked() -> list[dict]:
        """Return records sorted by distance from median density."""
        densities = [r["density"] for r in records_by_density]
        median = densities[len(densities) // 2]
        return sorted(records, key=lambda r: abs(r["density"] - median))

    # Each entry is (label, ordered_candidate_list)
    label_candidates: list[tuple[str, list[dict]]] = [
        ("dense", _ranked_density(high_to_low=True)),
        ("sparse", _ranked_density(high_to_low=False)),
        ("mixed", _median_ranked()),
        ("edge", records_by_edge),
        ("immune", records_by_volume),
    ]

    # ------------------------------------------------------------------
    # Step 4 — deduplicate (first-seen label wins, fallback through list)
    # ------------------------------------------------------------------
    tiles: list[FOVTile] = []
    used_fov_ids: set[int] = set()

    for label, ranked_list in label_candidates:
        if len(tiles) >= n_fovs:
            break
        # Find first candidate not already used
        rec = next((r for r in ranked_list if int(r["fov"]) not in used_fov_ids), None)
        if rec is None:
            continue
        fov_id = int(rec["fov"])
        used_fov_ids.add(fov_id)

        x_min_um: float = float(rec["x_min"])
        x_max_um: float = float(rec["x_max"])
        y_min_um: float = float(rec["y_min"])
        y_max_um: float = float(rec["y_max"])

        pixel_bbox = _micron_bbox_to_pixel_bbox(
            x_min_um, x_max_um, y_min_um, y_max_um,
            scale, offset_x, offset_y, padding_um,
        )
        micron_bbox = (x_min_um, x_max_um, y_min_um, y_max_um)

        tiles.append(
            FOVTile(
                fov_id=fov_id,
                label=label,
                n_cells=int(rec["n_cells"]),
                density=float(rec["density"]),
                mean_volume=float(rec["mean_volume"]),
                pixel_bbox=pixel_bbox,
                micron_bbox=micron_bbox,
            )
        )

    return tiles


# ---------------------------------------------------------------------------
# TIFF extraction helper
# ---------------------------------------------------------------------------


def load_dapi_mosaic(dapi_path: str | Path) -> np.ndarray:
    """Load the full DAPI mosaic image once into memory.

    Args:
        dapi_path: Path to the DAPI TIFF mosaic.

    Returns:
        2-D NumPy array (H, W), typically uint16.
    """
    import tifffile

    logger.info("Loading DAPI mosaic from %s (this may take a moment)...", dapi_path)
    with tifffile.TiffFile(dapi_path) as tif:
        image = tif.pages[0].asarray()
    logger.info("DAPI mosaic loaded: shape=%s, dtype=%s", image.shape, image.dtype)
    return image


def extract_fov_tile(
    dapi_image: np.ndarray,
    fov: FOVTile,
) -> np.ndarray:
    """Extract a FOV tile from a pre-loaded DAPI mosaic array.

    Args:
        dapi_image: Full mosaic array (H, W), already loaded into memory.
        fov: :class:`FOVTile` whose ``pixel_bbox`` defines the region to extract.

    Returns:
        2-D NumPy array (H, W) cropped to the FOV bounding box.
    """
    y_min, y_max, x_min, x_max = fov.pixel_bbox
    h, w = dapi_image.shape[:2]

    y_min = max(0, y_min)
    y_max = min(h, y_max)
    x_min = max(0, x_min)
    x_max = min(w, x_max)

    return dapi_image[y_min:y_max, x_min:x_max].copy()
