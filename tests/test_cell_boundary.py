# tests/test_cell_boundary.py
import numpy as np
from dapidl.qc.cell_boundary import rasterize_polygon_to_patch, voronoi_cell_mask


def test_polygon_um_to_patch_frame_raster():
    poly_um = np.array([[10.0, 10.0], [14.0, 10.0], [14.0, 14.0], [10.0, 14.0]])
    mask = rasterize_polygon_to_patch(poly_um, x0=40, y0=40, pixel_size=0.2125, patch_size=64)
    assert mask.dtype == bool and mask.shape == (64, 64)
    assert mask.sum() > 0
    assert mask[16, 16]  # interior point


def test_polygon_offframe_returns_empty():
    poly_um = np.array([[1000.0, 1000.0], [1004.0, 1000.0], [1004.0, 1004.0], [1000.0, 1004.0]])
    mask = rasterize_polygon_to_patch(poly_um, x0=40, y0=40, pixel_size=0.2125, patch_size=64)
    assert mask.shape == (64, 64) and mask.sum() == 0


def test_voronoi_fallback_contains_nucleus_and_is_bounded():
    nuc = np.zeros((64, 64), bool)
    nuc[28:36, 28:36] = True
    cell = voronoi_cell_mask(nuc)
    assert cell.shape == (64, 64) and cell.dtype == bool
    assert (cell & nuc).sum() == nuc.sum()        # contains the nucleus
    assert cell.sum() >= nuc.sum() and cell.sum() <= 64 * 64
