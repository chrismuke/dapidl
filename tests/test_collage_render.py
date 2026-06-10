# tests/test_collage_render.py
import numpy as np
import sys; sys.path.insert(0, "scripts")
from pilot_qc_collages_v3 import render_tile, CYAN, MAGENTA


def test_render_tile_paints_1px_dual_outlines():
    patch = (np.ones((64, 64)) * 1000).astype(np.uint16)
    nuc = np.zeros((64, 64), bool); nuc[24:40, 24:40] = True
    cell = np.zeros((64, 64), bool); cell[16:48, 16:48] = True
    rgb = render_tile(patch, nuc, cell)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert np.any(np.all(rgb == np.array(CYAN, np.uint8), axis=-1))
    assert np.any(np.all(rgb == np.array(MAGENTA, np.uint8), axis=-1))
    cyan_n = int(np.all(rgb == np.array(CYAN, np.uint8), axis=-1).sum())
    assert 0 < cyan_n < int(nuc.sum())


def test_render_tile_handles_empty_cell_mask():
    patch = (np.ones((64, 64)) * 1000).astype(np.uint16)
    nuc = np.zeros((64, 64), bool); nuc[24:40, 24:40] = True
    rgb = render_tile(patch, nuc, np.zeros((64, 64), bool))
    assert rgb.shape == (64, 64, 3)
    assert not np.any(np.all(rgb == np.array(MAGENTA, np.uint8), axis=-1))
