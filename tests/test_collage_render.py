# tests/test_collage_render.py
import math
import numpy as np
import sys; sys.path.insert(0, "scripts")
from pilot_qc_collages_v3 import render_tile, build_montage, assign_group, CYAN, MAGENTA


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


def test_build_montage_3tiles_gives_2x2_grid():
    """3 tiles should produce a 2-col x 2-row grid (ceil(sqrt(3))=2, rows=2)."""
    p = 64
    tiles = [np.full((p, p, 3), i * 80, dtype=np.uint8) for i in range(3)]
    grid = build_montage(tiles, patch_size=p)
    cols = math.ceil(math.sqrt(3))   # 2
    rows = math.ceil(3 / cols)       # 2
    assert grid.shape == (rows * p, cols * p, 3), (
        f"expected ({rows * p}, {cols * p}, 3), got {grid.shape}"
    )
    assert grid.dtype == np.uint8
    # Fourth slot (padding) must be black
    black = grid[p : 2 * p, p : 2 * p, :]
    assert black.max() == 0, "padding tile should be all-black"


def test_build_montage_empty_returns_single_black_tile():
    grid = build_montage([], patch_size=64)
    assert grid.shape == (64, 64, 3)
    assert grid.max() == 0


def test_assign_group_broken_geom():
    assert assign_group(True, "off_center", "broken") == "Broken-geom"
    assert assign_group(True, "cut_at_edge", "broken") == "Broken-geom"


def test_assign_group_broken_quality():
    assert assign_group(True, "no_nucleus", "broken") == "Broken-quality"
    assert assign_group(True, "false_detection", "broken") == "Broken-quality"


def test_assign_group_passing_uses_grade():
    assert assign_group(False, "ok", "Excellent") == "Excellent"
    assert assign_group(False, "ok", "Good") == "Good"
    assert assign_group(False, "ok", "Weak-passing") == "Weak-passing"
