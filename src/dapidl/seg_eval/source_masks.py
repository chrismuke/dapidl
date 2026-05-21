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


STHELAR_PX = 0.2125  # microns per pixel (same chemistry as Xenium)


def load_sthelar(zarr_path: str | Path):
    """STHELAR SpatialData loader.  Same dict contract as load_xenium.

    sdata layout:
      images['morpho']           – multiscale DataTree; scale0 has (c, y, x) uint16 DAPI
      shapes['nucleus_boundaries'] / ['cell_boundaries']  – GeoDataFrames in µm coords
      points['st']               – Dask DataFrame of transcripts in µm coords

    All pixel outputs are in the DAPI raster coordinate frame (divide µm by STHELAR_PX).
    """
    import spatialdata as sd

    sdata = sd.read_zarr(str(zarr_path))

    # ------------------------------------------------------------------ DAPI
    morpho = sdata.images["morpho"]
    # morpho is a multiscale DataTree; highest resolution lives at scale0
    scale0_ds = morpho["scale0"].ds
    img_da = scale0_ds["image"]          # DataArray (c, y, x)
    # Load eagerly only when called; keep the DataArray reference here
    _img_da = img_da

    def _dapi():
        arr = np.asarray(_img_da)        # (1, H, W) uint16
        return arr[0]                    # → (H, W)

    # ----------------------------------------------------------- polygons
    def _polys_from_shapes(key: str) -> pl.DataFrame:
        gdf = sdata.shapes[key]
        cell_ids: list[str] = []
        pxs: list[float] = []
        pys: list[float] = []
        for i, geom in enumerate(gdf.geometry):
            xs, ys = geom.exterior.coords.xy
            n = len(xs)
            cell_ids.extend([str(i)] * n)
            # coords are in µm — convert to pixels
            pxs.extend([float(v) / STHELAR_PX for v in xs])
            pys.extend([float(v) / STHELAR_PX for v in ys])
        return pl.DataFrame({"cell_id": cell_ids, "px": pxs, "py": pys})

    # ----------------------------------------------------------- centroids
    nuc_gdf = sdata.shapes["nucleus_boundaries"]
    # GeoDataFrame centroid x/y are in µm
    centroids = np.stack([
        nuc_gdf.geometry.centroid.y.to_numpy() / STHELAR_PX,   # row = y
        nuc_gdf.geometry.centroid.x.to_numpy() / STHELAR_PX,   # col = x
    ], axis=1)

    # ----------------------------------------------------------- transcripts
    pts_dask = sdata.points["st"]
    pts_pd = pts_dask.compute()          # pandas DataFrame
    tx = pl.from_pandas(pts_pd.reset_index(drop=True)).select(
        (pl.col("x") / STHELAR_PX).alias("x"),
        (pl.col("y") / STHELAR_PX).alias("y"),
        pl.col("feature_name").alias("gene"),
    )

    return {
        "dapi": _dapi,
        "pixel_size": STHELAR_PX,
        "nucleus_polys": lambda: _polys_from_shapes("nucleus_boundaries"),
        "cell_polys": lambda: _polys_from_shapes("cell_boundaries"),
        "transcripts": tx,
        "centroids": centroids,
    }


SOURCES = {
    "xenium_rep1": {"kind": "xenium", "root": "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1"},
    "xenium_rep2": {"kind": "xenium", "root": "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2"},
    "sthelar_breast_s0": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/sdata_breast_s0.zarr"},
    "sthelar_breast_s1": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s1.zarr/sdata_breast_s1.zarr"},
    "sthelar_breast_s3": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s3.zarr/sdata_breast_s3.zarr"},
    "sthelar_breast_s6": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s6.zarr/sdata_breast_s6.zarr"},
}
