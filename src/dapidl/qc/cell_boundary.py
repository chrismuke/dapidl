# src/dapidl/qc/cell_boundary.py
"""Per-source cell-mask resolver for the p64 QC re-score: native polygon (Xenium
cell_boundaries.parquet; STHELAR zarr shape if present) rasterized into the patch
frame, else a Voronoi expansion of the central-nucleus mask bounded by neighbours."""
from __future__ import annotations

import numpy as np
from skimage.draw import polygon as sk_polygon


def rasterize_polygon_to_patch(poly_um, x0: float, y0: float, pixel_size: float,
                               patch_size: int) -> np.ndarray:
    """Transform a cell polygon (full-res micron coords, shape (K,2) [x,y]) into the
    patch frame: um -> px (/pixel_size) -> minus crop origin (x0,y0) -> rasterize.
    Off-frame polygons yield an all-False mask of (patch_size, patch_size)."""
    mask = np.zeros((patch_size, patch_size), dtype=bool)
    p = np.asarray(poly_um, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return mask
    xs = p[:, 0] / pixel_size - x0
    ys = p[:, 1] / pixel_size - y0
    rr, cc = sk_polygon(ys, xs, shape=(patch_size, patch_size))  # row=y, col=x
    mask[rr, cc] = True
    return mask


def voronoi_cell_mask(nucleus_mask, max_radius_px: int | None = None) -> np.ndarray:
    """Fallback: grow the central-nucleus mask outward (distance-to-nucleus expansion)
    into the patch, capped so it stays a plausible single-cell territory. Contains the
    nucleus; bounded by the patch (and max_radius_px if given)."""
    from scipy import ndimage
    nuc = np.asarray(nucleus_mask, dtype=bool)
    if not nuc.any():
        return nuc.copy()
    dist = ndimage.distance_transform_edt(~nuc)
    if max_radius_px is None:
        max_radius_px = nuc.shape[0] // 2
    return (dist <= max_radius_px) | nuc
