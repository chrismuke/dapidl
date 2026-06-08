"""Polygon → uint16 instance-mask rasterization for STHELAR breast slides.

Used by Phase A tile-cache build and Phase D slide-level eval.

Design notes:
- uint16 supports up to 65535 instance IDs per tile. Largest observed STHELAR
  breast tile (1024² at stride 1024) has < 1500 cells; even 512² with 50%
  overlap has < 600. uint16 is comfortable.
- Background is 0; instance IDs start at 1.
- `make_valid` repair already applied upstream by
  `load_nucleus_geometry_with_labels`. This module assumes valid Polygon input;
  callers should drop GeometryCollection/MultiPolygon/empty before calling.
"""

import numpy as np
import rasterio.features
from shapely.geometry.base import BaseGeometry


def rasterize_instances(
    geoms: list[BaseGeometry],
    instance_ids: np.ndarray,
    *,
    bbox_px: tuple[float, float, float, float],
    out_size: tuple[int, int],
    dtype: np.dtype = np.dtype(np.uint16),
) -> np.ndarray:
    """Rasterize a list of polygons into a single instance-id array.

    Args:
        geoms: list of shapely Polygon geometries in pixel coordinates.
        instance_ids: (N,) integer IDs to burn (must be > 0).
        bbox_px: (minx, miny, maxx, maxy) of the output canvas in pixels.
            Polygons outside this bbox are not drawn.
        out_size: (H, W) output array shape.
        dtype: integer dtype; defaults to uint16 (max 65535).

    Returns:
        (H, W) array of dtype with 0 background and `instance_ids[i]` painted
        wherever `geoms[i]` covers a pixel.
    """
    if len(geoms) != len(instance_ids):
        raise ValueError(
            f"geoms ({len(geoms)}) and instance_ids ({len(instance_ids)}) "
            "must match"
        )
    if len(geoms) == 0:
        return np.zeros(out_size, dtype=dtype)

    minx, miny, maxx, maxy = bbox_px
    h, w = out_size
    sx = w / (maxx - minx)
    sy = h / (maxy - miny)
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
    # Sanity: sx ≈ sy ≈ 1 when out_size matches bbox extent at level-0 px.
    assert abs(sx - sy) < 1e-3, f"non-square scale sx={sx}, sy={sy}"

    # Convert IDs to int (rasterize wants Python ints/numpy)
    shapes = list(zip(geoms, [int(i) for i in instance_ids]))

    out = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=out_size,
        transform=transform,
        fill=0,
        dtype=dtype,
        all_touched=False,  # fill only pixels whose center is inside (avoids over-merging)
    )
    return out


def compute_centroids_px(gdf) -> np.ndarray:
    """Compute (N, 2) centroids in pixel coords once per slide.

    Pass the result to `rasterize_tile` to avoid re-iterating geometries on
    every tile. Hot loop in Phase A cache build.
    """
    return np.array(
        [[g.centroid.x, g.centroid.y] for g in gdf.geometry], dtype=np.float64
    )


def rasterize_tile(
    gdf,
    instance_id_col: str,
    *,
    x0: int,
    y0: int,
    tile_size: int,
    pad: int = 0,
    dtype: np.dtype = np.dtype(np.uint16),
    centroids_px: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize one tile from a GeoDataFrame.

    Filters polygons whose centroid falls inside `[x0, x0+tile_size) ×
    [y0, y0+tile_size)`, then rasterizes. Returns the instance map, a
    boolean array marking border-touching instances (centroid within `pad`
    pixels of any border), and the indices of the rows that landed in the tile.

    Args:
        gdf: GeoDataFrame with column `instance_id_col` (uint16-compatible).
        instance_id_col: column holding the per-tile instance ID.
        x0, y0: tile origin in pixels.
        tile_size: output tile size (square).
        pad: border-pad zone in pixels for `is_border` flag.
        dtype: output dtype.
        centroids_px: optional precomputed (N, 2) array; pass via
            `compute_centroids_px(gdf)` to avoid recomputing per call.

    Returns:
        (instance_map, is_border, in_tile_idx) — instance_map shape
        (tile_size, tile_size); is_border length n_in_tile; in_tile_idx is
        the integer row positions in `gdf` that landed in this tile.
    """
    if centroids_px is None:
        centroids_px = compute_centroids_px(gdf)
    cents = centroids_px
    in_tile_mask = (
        (cents[:, 0] >= x0)
        & (cents[:, 0] < x0 + tile_size)
        & (cents[:, 1] >= y0)
        & (cents[:, 1] < y0 + tile_size)
    )
    in_tile_idx = np.where(in_tile_mask)[0]
    if len(in_tile_idx) == 0:
        return (
            np.zeros((tile_size, tile_size), dtype=dtype),
            np.zeros(0, dtype=bool),
            in_tile_idx,
        )
    sub = gdf.iloc[in_tile_idx]
    sub_cents = cents[in_tile_mask]
    is_border = (
        (sub_cents[:, 0] - x0 < pad)
        | (x0 + tile_size - sub_cents[:, 0] < pad)
        | (sub_cents[:, 1] - y0 < pad)
        | (y0 + tile_size - sub_cents[:, 1] < pad)
    )

    geoms = [g for g in sub.geometry]
    instance_map = rasterize_instances(
        geoms,
        sub[instance_id_col].to_numpy(),
        bbox_px=(x0, y0, x0 + tile_size, y0 + tile_size),
        out_size=(tile_size, tile_size),
        dtype=dtype,
    )
    return instance_map, is_border, in_tile_idx
