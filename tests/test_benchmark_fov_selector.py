"""Tests for the FOV selector benchmark utility.

Uses synthetic cell_metadata with 10 FOVs of varying density to verify:
- Correct number of FOVTiles returned
- Unique FOV selection
- Required fields on FOVTile
- Dense FOV has higher density than sparse
"""

import numpy as np
import polars as pl
import pytest

from dapidl.benchmark.fov_selector import select_fovs


@pytest.fixture
def cell_metadata() -> pl.DataFrame:
    """Synthetic cell_metadata with 10 FOVs of varying cell density.

    FOV 0  → very dense  (~300 cells in small area)
    FOV 9  → very sparse (~10 cells in large area)
    FOVs 1-8 → intermediate densities
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []

    # Assign cell counts and approximate extents per FOV so density varies
    fov_specs = [
        # (fov_id, n_cells, x_size_um, y_size_um)
        (0, 300, 200.0, 200.0),   # dense  (~7.5 cells/1000 um2)
        (1, 200, 250.0, 250.0),
        (2, 150, 300.0, 300.0),
        (3, 120, 350.0, 350.0),
        (4, 100, 400.0, 400.0),   # near median
        (5, 80, 400.0, 400.0),
        (6, 60, 450.0, 450.0),
        (7, 40, 450.0, 450.0),
        (8, 25, 500.0, 500.0),
        (9, 10, 600.0, 600.0),    # sparse (~0.028 cells/1000 um2)
    ]

    for fov_id, n_cells, x_size, y_size in fov_specs:
        # FOV origin: stagger so they don't overlap
        x_origin = fov_id * 700.0
        y_origin = 0.0

        for _ in range(n_cells):
            cx = x_origin + rng.uniform(0, x_size)
            cy = y_origin + rng.uniform(0, y_size)
            w = rng.uniform(10.0, 30.0)
            h = rng.uniform(10.0, 30.0)
            rows.append(
                {
                    "fov": fov_id,
                    "volume": float(rng.uniform(200.0, 800.0)),
                    "center_x": cx,
                    "center_y": cy,
                    "min_x": cx - w / 2,
                    "max_x": cx + w / 2,
                    "min_y": cy - h / 2,
                    "max_y": cy + h / 2,
                }
            )

    return pl.DataFrame(rows)


def test_select_fovs_returns_5(cell_metadata: pl.DataFrame) -> None:
    """select_fovs must return exactly 5 FOVTiles by default."""
    tiles = select_fovs(cell_metadata)
    assert len(tiles) == 5


def test_select_fovs_all_unique(cell_metadata: pl.DataFrame) -> None:
    """All returned FOVTiles must have distinct fov_ids."""
    tiles = select_fovs(cell_metadata)
    fov_ids = [t.fov_id for t in tiles]
    assert len(fov_ids) == len(set(fov_ids)), "Duplicate fov_ids in selection"


def test_fov_tile_has_required_fields(cell_metadata: pl.DataFrame) -> None:
    """Each FOVTile must have valid label and non-None pixel_bbox."""
    valid_labels = {"dense", "sparse", "mixed", "edge", "immune"}
    tiles = select_fovs(cell_metadata)
    for tile in tiles:
        assert tile.label in valid_labels, f"Unexpected label: {tile.label!r}"
        assert tile.pixel_bbox is not None, "pixel_bbox must not be None"
        assert len(tile.pixel_bbox) == 4, "pixel_bbox must be a 4-tuple"
        assert len(tile.micron_bbox) == 4, "micron_bbox must be a 4-tuple"


def test_dense_fov_has_highest_density(cell_metadata: pl.DataFrame) -> None:
    """The dense tile must have strictly higher density than the sparse tile."""
    tiles = select_fovs(cell_metadata)
    tile_by_label = {t.label: t for t in tiles}

    assert "dense" in tile_by_label, "No dense tile selected"
    assert "sparse" in tile_by_label, "No sparse tile selected"

    assert tile_by_label["dense"].density > tile_by_label["sparse"].density


def test_select_fovs_n_fovs_param(cell_metadata: pl.DataFrame) -> None:
    """n_fovs parameter controls the number of returned tiles (capped by available FOVs)."""
    tiles = select_fovs(cell_metadata, n_fovs=3)
    assert len(tiles) == 3


def test_fov_tile_n_cells_positive(cell_metadata: pl.DataFrame) -> None:
    """Every FOVTile must have at least 1 cell."""
    tiles = select_fovs(cell_metadata)
    for tile in tiles:
        assert tile.n_cells > 0


def test_fov_tile_density_positive(cell_metadata: pl.DataFrame) -> None:
    """Every FOVTile must have a positive density value."""
    tiles = select_fovs(cell_metadata)
    for tile in tiles:
        assert tile.density > 0.0
