"""Load source (built-in) segmentation masks + DAPI + transcripts per source.

Two platforms: Xenium (OME-TIFF DAPI + parquet polygons in microns) and
STHELAR (SpatialData zarr). All returned masks/centroids are in PIXEL
coordinates of the DAPI raster.
"""
from pathlib import Path

import numpy as np
import polars as pl
from skimage.draw import polygon as sk_polygon

XENIUM_PX = 0.2125  # microns per pixel (Xenium morphology)


def rasterize_polygons(df, id_col: str, x_col: str, y_col: str,
                       bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Rasterize per-instance polygons (pixel coords) into a label mask for a bbox.

    df: long-format polars DF, one row per polygon vertex, grouped by id_col.
    bbox: (y0, x0, y1, x1) in pixels. Returns (y1-y0, x1-x0) int32 label mask.
    """
    y0, x0, y1, x1 = bbox
    h, w = y1 - y0, x1 - x0
    mask = np.zeros((h, w), dtype=np.int32)
    label = 0
    for _id, grp in df.group_by(id_col, maintain_order=True):
        xs = grp[x_col].to_numpy() - x0
        ys = grp[y_col].to_numpy() - y0
        if xs.max() < 0 or ys.max() < 0 or xs.min() >= w or ys.min() >= h:
            continue
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if rr.size == 0:
            continue
        label += 1
        mask[rr, cc] = label
    return mask


def load_xenium(root: str | Path):
    """Return dict: dapi (callable->2D array), pixel_size, nucleus_polys/cell_polys
    (callables -> polars long-format with px/py columns), transcripts (polars x,y px
    + gene), centroids (Nx2 [y,x] px)."""
    root = Path(root)
    outs = root / "outs"
    import tifffile

    def _dapi():
        return tifffile.imread(str(root / "morphology_focus.ome.tif"))

    def _polys(name):
        df = pl.read_parquet(outs / f"{name}.parquet")
        return df.with_columns(
            (pl.col("vertex_x") / XENIUM_PX).alias("px"),
            (pl.col("vertex_y") / XENIUM_PX).alias("py"),
        )

    cells = pl.read_parquet(outs / "cells.parquet")
    centroids = np.stack([
        cells["y_centroid"].to_numpy() / XENIUM_PX,
        cells["x_centroid"].to_numpy() / XENIUM_PX,
    ], axis=1)
    tx = pl.read_parquet(outs / "transcripts.parquet").select(
        (pl.col("x_location") / XENIUM_PX).alias("x"),
        (pl.col("y_location") / XENIUM_PX).alias("y"),
        pl.col("feature_name").alias("gene"),
    )
    return {
        "dapi": _dapi,
        "pixel_size": XENIUM_PX,
        "nucleus_polys": lambda: _polys("nucleus_boundaries"),
        "cell_polys": lambda: _polys("cell_boundaries"),
        "transcripts": tx,
        "centroids": centroids,
    }
