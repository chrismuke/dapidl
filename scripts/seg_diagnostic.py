"""Phase 1 segmentation diagnostic: starpose vs source on representative FOVs."""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dapidl.seg_eval.source_masks import (SOURCES, load_sthelar, load_xenium,
                                          rasterize_polygons)
from dapidl.seg_eval.compare import detection_metrics, morphometrics
from dapidl.seg_eval.qc_compare import score_centroid_patches

from starpose.benchmark.fov_selector import select_fovs, extract_tile
from starpose.core import segment_multimodal
from starpose.types import ModalityBundle

OUT = Path("pipeline_output/seg_diagnostic_2026_05")


def _segment_fov(dapi_tile, transcripts_tile, pixel_size, expander,
                 nucleus_method="cellpose_nuclei"):
    mb = ModalityBundle(dapi=dapi_tile, transcripts=transcripts_tile,
                        pixel_size=pixel_size, platform="diagnostic")
    return segment_multimodal(mb, gpu=True, nucleus_method=nucleus_method,
                              expansion_method=expander)


def _crop_polys(polys, bbox):
    """Keep only WHOLE cells all of whose vertices fall inside the bbox."""
    y0, x0, y1, x1 = bbox
    inside = polys.filter((pl.col("px") >= x0) & (pl.col("px") < x1)
                          & (pl.col("py") >= y0) & (pl.col("py") < y1))
    full = (polys.group_by("cell_id").agg(pl.len().alias("total"))
            .join(inside.group_by("cell_id").agg(pl.len().alias("kept")), on="cell_id")
            .filter(pl.col("kept") == pl.col("total"))["cell_id"])
    return inside.filter(pl.col("cell_id").is_in(full))


def _local_centroids(centroids, bbox):
    y0, x0, y1, x1 = bbox
    m = ((centroids[:, 0] >= y0) & (centroids[:, 0] < y1)
         & (centroids[:, 1] >= x0) & (centroids[:, 1] < x1))
    c = centroids[m].copy()
    c[:, 0] -= y0; c[:, 1] -= x0
    return c


def diagnose_fov(source, fov_label, dapi_tile, src_nuc, src_cell, src_centroids_local,
                 transcripts, pixel_size, expander="watershed",
                 nucleus_method="cellpose_nuclei"):
    """One FOV: segment, compare nucleus+cell masks, compare QC. Returns a row dict."""
    res = _segment_fov(dapi_tile, transcripts, pixel_size, expander,
                       nucleus_method=nucleus_method)
    sp_nuc = res.nucleus_masks
    sp_cell = res.cell_masks if res.cell_masks is not None else res.nucleus_masks
    sp_cen_raw = res.nucleus_centroids if res.nucleus_centroids is not None else res.cell_centroids
    sp_cen = np.asarray(sp_cen_raw) if sp_cen_raw is not None else np.zeros((0, 2))

    nuc = detection_metrics(sp_nuc, src_nuc)
    cell = detection_metrics(sp_cell, src_cell)
    sp_morph = morphometrics(sp_nuc, pixel_size)
    src_morph = morphometrics(src_nuc, pixel_size)
    qc_own = score_centroid_patches(dapi_tile, sp_cen)
    qc_src = score_centroid_patches(dapi_tile, src_centroids_local)

    return {
        "source": source, "fov": fov_label,
        "n_starpose_nuc": nuc["n_pred"], "n_source_nuc": nuc["n_true"],
        "nuc_precision": nuc["precision"], "nuc_recall": nuc["recall"],
        "nuc_f1": nuc["f1"], "nuc_median_iou": nuc["median_iou"],
        "nuc_count_ratio": nuc["count_ratio"],
        "cell_f1": cell["f1"], "cell_count_ratio": cell["count_ratio"],
        "sp_mean_area_um2": sp_morph["mean_area_um2"],
        "src_mean_area_um2": src_morph["mean_area_um2"],
        "qc_own_mean": float(qc_own["qc_score"].mean()) if qc_own.height else float("nan"),
        "qc_src_mean": float(qc_src["qc_score"].mean()) if qc_src.height else float("nan"),
        "qc_own_n": qc_own.height, "qc_src_n": qc_src.height,
    }


def run_source(name, n_fovs, tile, expander, nucleus_method="cellpose_nuclei"):
    cfg = SOURCES[name]
    src = load_xenium(cfg["root"]) if cfg["kind"] == "xenium" else load_sthelar(cfg["zarr"])
    dapi = src["dapi"]()
    px = src["pixel_size"]
    fovs = select_fovs(src["centroids"], dapi.shape, pixel_size=px,
                       n_fovs=n_fovs, tile_size_px=tile)
    nuc_polys, cell_polys = src["nucleus_polys"](), src["cell_polys"]()
    rows = []
    for fov in fovs:
        y0, x0, y1, x1 = fov.bbox
        tile_img = extract_tile(dapi, fov)
        src_nuc = rasterize_polygons(_crop_polys(nuc_polys, fov.bbox), "cell_id", "px", "py", fov.bbox)
        src_cell = rasterize_polygons(_crop_polys(cell_polys, fov.bbox), "cell_id", "px", "py", fov.bbox)
        src_cen = _local_centroids(src["centroids"], fov.bbox)
        tx_tile = src["transcripts_for_bbox"](fov.bbox).with_columns(
            (pl.col("x") - x0).alias("x"),
            (pl.col("y") - y0).alias("y"),
        )
        try:
            rows.append(diagnose_fov(name, fov.label, tile_img, src_nuc, src_cell,
                                     src_cen, tx_tile, px, expander,
                                     nucleus_method=nucleus_method))
        except Exception as e:
            logger.warning(f"{name}/{fov.label} failed: {e}")
    logger.info(f"{name}: {len(rows)} FOVs done")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--n-fovs", type=int, default=8)
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--expander", default="watershed", choices=["proseg", "watershed"])
    ap.add_argument("--nucleus-method", default="cellpose_nuclei",
                    choices=["adaptive", "cellpose_nuclei", "stardist"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name in args.sources.split(","):
        all_rows.extend(run_source(name, args.n_fovs, args.tile, args.expander,
                                   nucleus_method=args.nucleus_method))
    df = pl.DataFrame(all_rows)
    df.write_parquet(OUT / "results.parquet")
    logger.info(f"wrote {OUT/'results.parquet'} ({df.height} rows)")
    if df.height == 0:
        logger.warning("no FOVs produced rows — check segmentation backend availability")
        return
    print(df.group_by("source").agg(
        pl.col("nuc_f1").mean().round(3),
        pl.col("nuc_count_ratio").mean().round(3),
        pl.col("qc_own_mean").mean().round(3),
        pl.col("qc_src_mean").mean().round(3),
    ).sort("source"))


if __name__ == "__main__":
    main()
