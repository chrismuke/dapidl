"""Native-p64 QC collage renderer: DAPI grayscale + 1px cyan central-nucleus outline +
1px magenta resolved-cell outline. render_tile is pure (unit-tested); the grouping
driver (main) is added in a later task."""
from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


def _stretch_to_uint8(patch: np.ndarray) -> np.ndarray:
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def render_tile(patch, nucleus_mask, cell_mask) -> np.ndarray:
    """uint16 DAPI patch + bool nucleus/cell masks -> (H,W,3) uint8 RGB with 1px inner
    outlines: cyan nucleus, magenta cell. Empty masks paint nothing."""
    g = _stretch_to_uint8(np.asarray(patch))
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    cell = np.asarray(cell_mask, dtype=bool)
    nuc = np.asarray(nucleus_mask, dtype=bool)
    if cell.any():
        rgb[find_boundaries(cell, mode="inner")] = MAGENTA
    if nuc.any():
        rgb[find_boundaries(nuc, mode="inner")] = CYAN   # nucleus drawn last (on top)
    return rgb
