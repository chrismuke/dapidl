"""Tests for XeniumDataReader.get_nucleus_centroids_pixels."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dapidl.data.xenium import XeniumDataReader


def _mk_xenium(outs: Path, cells: pl.DataFrame, nuc: pl.DataFrame) -> None:
    """Materialise a minimal Xenium 'outs' tree for the reader's path validator."""
    outs.mkdir(parents=True, exist_ok=True)
    (outs / "morphology_focus.ome.tif").write_bytes(b"")
    (outs / "cell_feature_matrix.h5").write_bytes(b"")
    cells.write_parquet(outs / "cells.parquet")
    nuc.write_parquet(outs / "nucleus_boundaries.parquet")


@pytest.fixture
def synthetic_outs(tmp_path: Path) -> Path:
    """Three cells; cell A has a centered square nucleus, B an offset one, C no nucleus."""
    outs = tmp_path / "outs"
    cells = pl.DataFrame({
        "cell_id": ["A", "B", "C"],
        "x_centroid": [10.0, 100.0, 50.0],  # microns
        "y_centroid": [20.0, 200.0, 60.0],
    })
    nuc = pl.DataFrame({
        # nuclei intentionally out of cells_df order to verify join preserves order
        "cell_id": ["B", "B", "B", "B", "A", "A", "A", "A"],
        # B: square with vertices (98, 198), (102, 198), (102, 202), (98, 202) -> centroid (100, 200)
        "vertex_x": [98.0, 102.0, 102.0, 98.0, 8.0, 12.0, 12.0, 8.0],
        # A: square with vertices (8, 18), (12, 18), (12, 22), (8, 22) -> centroid (10, 20)
        "vertex_y": [198.0, 198.0, 202.0, 202.0, 18.0, 18.0, 22.0, 22.0],
    })
    _mk_xenium(outs, cells, nuc)
    return outs


def test_polygon_centroid_matches_mean_of_vertices(synthetic_outs):
    """Centroid = mean(vertices) per cell_id, converted µm -> px via PIXEL_SIZE."""
    reader = XeniumDataReader(synthetic_outs)
    centroids = reader.get_nucleus_centroids_pixels()
    px = XeniumDataReader.PIXEL_SIZE
    # A's nucleus centroid is (10, 20) µm (matches its cell centroid exactly here)
    np.testing.assert_allclose(centroids[0], [10.0 / px, 20.0 / px])
    # B's nucleus centroid is (100, 200) µm
    np.testing.assert_allclose(centroids[1], [100.0 / px, 200.0 / px])


def test_cell_id_order_preserved(synthetic_outs):
    """Output rows must match cells_df cell_id order (A, B, C) regardless of nucleus_boundaries order."""
    reader = XeniumDataReader(synthetic_outs)
    centroids = reader.get_nucleus_centroids_pixels()
    cell_ids = reader.cells_df["cell_id"].to_list()
    assert cell_ids == ["A", "B", "C"], f"cells_df order shifted: {cell_ids}"
    assert centroids.shape == (3, 2)


def test_missing_polygon_falls_back_to_cell_centroid(synthetic_outs):
    """Cell C has no nucleus polygon; centroid must fall back to cell centroid (50, 60) µm."""
    reader = XeniumDataReader(synthetic_outs)
    centroids = reader.get_nucleus_centroids_pixels()
    px = XeniumDataReader.PIXEL_SIZE
    np.testing.assert_allclose(centroids[2], [50.0 / px, 60.0 / px])


def test_shape_matches_cells_df(synthetic_outs):
    """Returned (N, 2) where N = len(cells_df)."""
    reader = XeniumDataReader(synthetic_outs)
    centroids = reader.get_nucleus_centroids_pixels()
    assert centroids.shape == (reader.cells_df.height, 2)


# --- Pure-helper tests (review 2026-05-29 B7): dtype guard + fallback count ---

def _cells_df(ids, xs, ys):
    return pl.DataFrame({"cell_id": ids, "x_centroid": xs, "y_centroid": ys})


def _nb_df(rows):
    return pl.DataFrame({"cell_id": [r[0] for r in rows],
                         "vertex_x": [r[1] for r in rows],
                         "vertex_y": [r[2] for r in rows]})


def test_helper_reports_fallback_count():
    px = XeniumDataReader.PIXEL_SIZE
    cells = _cells_df(["A", "B"], [7.0, 99.0], [7.0, 99.0])
    nb = _nb_df([("A", 10.0, 10.0), ("A", 20.0, 20.0)])   # B absent -> fallback
    arr, n_fb = XeniumDataReader._nucleus_centroids_from_boundaries(cells, nb, px)
    assert n_fb == 1
    np.testing.assert_allclose(arr[0], [15.0 / px, 15.0 / px])   # A from nucleus
    np.testing.assert_allclose(arr[1], [99.0 / px, 99.0 / px])   # B fell back to cell


def test_helper_dtype_mismatch_no_silent_total_fallback():
    """B7: nucleus_boundaries.cell_id Categorical vs cells.cell_id Utf8 must NOT
    produce an all-null join (100% fallback -> nuc-LMDB == cell-LMDB, a silent
    no-op that looks like the experiment ran)."""
    px = XeniumDataReader.PIXEL_SIZE
    cells = _cells_df(["A", "B"], [0.0, 0.0], [0.0, 0.0])
    nb = _nb_df([("A", 10.0, 10.0), ("B", 30.0, 30.0)]).with_columns(
        pl.col("cell_id").cast(pl.Categorical))
    arr, n_fb = XeniumDataReader._nucleus_centroids_from_boundaries(cells, nb, px)
    assert n_fb == 0, "dtype mismatch caused a silent total fallback (B7)"
    np.testing.assert_allclose(arr[0], [10.0 / px, 10.0 / px])
    np.testing.assert_allclose(arr[1], [30.0 / px, 30.0 / px])
