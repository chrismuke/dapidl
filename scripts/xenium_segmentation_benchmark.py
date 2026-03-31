#!/usr/bin/env python3
"""Benchmark adaptive consensus segmentation on Xenium breast rep1.

Compares: Cellpose default, Cellpose aggressive, StarDist,
Adaptive consensus vs 10x native segmentation.

Usage:
    uv run python scripts/xenium_segmentation_benchmark.py
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from skimage.measure import regionprops

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

XENIUM_DIR = Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1")
DAPI_PATH = XENIUM_DIR / "morphology_focus.ome.tif"
CELLS_PATH = XENIUM_DIR / "cells.parquet"
OUTPUT_DIR = Path("pipeline_output/xenium_segmentation_benchmark")
PIXEL_SIZE = 0.2125  # um/px for Xenium


def select_xenium_fovs(cells_px, img_shape, tile_size_px=2000):
    """Select 5 representative tiles from Xenium image."""
    h, w = img_shape

    cells_px = cells_px.with_columns(
        (pl.col("cx_px") // tile_size_px).alias("gx"),
        (pl.col("cy_px") // tile_size_px).alias("gy"),
    )
    grid_counts = cells_px.group_by(["gx", "gy"]).agg(
        pl.len().alias("n_cells"),
        pl.col("nucleus_area").mean().alias("mean_nuc_area"),
    ).sort("n_cells", descending=True)

    tiles = []
    used = set()

    def add(label, gx, gy, n):
        y0 = max(0, int(gy * tile_size_px))
        x0 = max(0, int(gx * tile_size_px))
        y1 = min(h, y0 + tile_size_px)
        x1 = min(w, x0 + tile_size_px)
        tiles.append({"label": label, "bbox": (y0, y1, x0, x1), "n_native": n})
        used.add((gx, gy))

    # Dense
    r = grid_counts.row(0, named=True)
    add("dense", r["gx"], r["gy"], r["n_cells"])

    # Sparse
    for r in grid_counts.reverse().iter_rows(named=True):
        if (r["gx"], r["gy"]) not in used and r["n_cells"] > 5:
            add("sparse", r["gx"], r["gy"], r["n_cells"])
            break

    # Mixed (median)
    for r in grid_counts.slice(len(grid_counts) // 2).iter_rows(named=True):
        if (r["gx"], r["gy"]) not in used:
            add("mixed", r["gx"], r["gy"], r["n_cells"])
            break

    # Edge
    for r in grid_counts.iter_rows(named=True):
        gx, gy = r["gx"], r["gy"]
        if (gx, gy) not in used:
            if gx == 0 or gy == 0 or (gx + 1) * tile_size_px >= w or (gy + 1) * tile_size_px >= h:
                add("edge", gx, gy, r["n_cells"])
                break

    # Small nuclei (immune)
    for r in grid_counts.sort("mean_nuc_area").iter_rows(named=True):
        if (r["gx"], r["gy"]) not in used:
            add("immune", r["gx"], r["gy"], r["n_cells"])
            break

    return tiles


def get_native_centroids_in_bbox(cells_px, bbox):
    """Get native centroids in tile-relative pixel coords."""
    y0, y1, x0, x1 = bbox
    fov = cells_px.filter(
        (pl.col("cx_px") >= x0) & (pl.col("cx_px") < x1) &
        (pl.col("cy_px") >= y0) & (pl.col("cy_px") < y1)
    )
    if fov.is_empty():
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([
        fov["cy_px"].to_numpy() - y0,
        fov["cx_px"].to_numpy() - x0,
    ]).astype(np.float64)


def run_cellpose(model, tile, ft=0.4, cp=0.0):
    t0 = time.perf_counter()
    # Cellpose 4 eval method
    masks_result = model.eval(  # noqa: S307 — cellpose inference
        tile, channels=[0, 0], diameter=None,
        flow_threshold=ft, cellprob_threshold=cp,
    )
    return masks_result[0].astype(np.int32), time.perf_counter() - t0


def run_stardist(model, tile):
    img = tile.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99.8))
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1), 0, 1)
    t0 = time.perf_counter()
    masks, _ = model.predict_instances(img)
    return masks.astype(np.int32), time.perf_counter() - t0


def adaptive_consensus(masks_cp_def, masks_cp_agg, masks_sd):
    """StarDist/Cellpose ratio > 2 → SD + CP_agg fill; else CP default."""
    n_cp = int(masks_cp_def.max())
    n_sd = int(masks_sd.max())
    ratio = n_sd / max(n_cp, 1)

    if ratio > 2.0:
        consensus = masks_sd.copy()
        next_label = n_sd + 1
        min_area = 20.0 / (PIXEL_SIZE ** 2)

        for cl in np.unique(masks_cp_agg):
            if cl == 0:
                continue
            cp_mask = masks_cp_agg == cl
            if np.sum(cp_mask) < min_area:
                continue
            if np.sum(masks_sd[cp_mask] > 0) / np.sum(cp_mask) >= 0.2:
                continue
            consensus[cp_mask & (consensus == 0)] = next_label
            next_label += 1
        return consensus, "SD+CP_agg"
    else:
        return masks_cp_def.copy(), "CP_default"


def compute_recovery(masks, centroids):
    n = len(centroids)
    if n == 0:
        return 0.0, 0
    h, w = masks.shape
    recovered = sum(
        1 for cy, cx in centroids
        if 0 <= int(round(cy)) < h and 0 <= int(round(cx)) < w
        and masks[int(round(cy)), int(round(cx))] > 0
    )
    return recovered / n, recovered


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Xenium data...")
    cells = pl.read_parquet(str(CELLS_PATH))
    dapi = tifffile.imread(str(DAPI_PATH))
    logger.info("DAPI: %s (%s), %d cells", dapi.shape, dapi.dtype, len(cells))

    p1, p50, p99 = np.percentile(dapi, (1, 50, 99))
    logger.info("DAPI intensity: p1=%.0f, p50=%.0f, p99=%.0f, max=%d", p1, p50, p99, dapi.max())

    # Pixel-space centroids
    cells_px = cells.with_columns(
        (pl.col("x_centroid") / PIXEL_SIZE).cast(pl.Int32).alias("cx_px"),
        (pl.col("y_centroid") / PIXEL_SIZE).cast(pl.Int32).alias("cy_px"),
    )

    tiles_info = select_xenium_fovs(cells_px, dapi.shape)
    logger.info("Selected %d tiles", len(tiles_info))

    logger.info("Loading Cellpose...")
    from cellpose import models as cp_models
    cp_model = cp_models.CellposeModel(gpu=True)

    logger.info("Loading StarDist...")
    from stardist.models import StarDist2D
    sd_model = StarDist2D.from_pretrained("2D_versatile_fluo")

    results = {}
    for ti in tiles_info:
        label = ti["label"]
        y0, y1, x0, x1 = ti["bbox"]
        tile = dapi[y0:y1, x0:x1]
        native_centroids = get_native_centroids_in_bbox(cells_px, ti["bbox"])
        n_native = len(native_centroids)

        logger.info("=== %s: %dx%d, %d native cells ===", label, tile.shape[0], tile.shape[1], n_native)

        tp1, tp99 = np.percentile(tile, (1, 99))
        logger.info("  Tile DAPI: p1=%.0f, p99=%.0f, max=%d", tp1, tp99, tile.max())

        masks_cp_def, t_cp = run_cellpose(cp_model, tile)
        masks_cp_agg, t_ca = run_cellpose(cp_model, tile, ft=0.8, cp=-3.0)
        masks_sd, t_sd = run_stardist(sd_model, tile)
        masks_adaptive, strategy = adaptive_consensus(masks_cp_def, masks_cp_agg, masks_sd)

        tile_results = {}
        for name, masks, rt in [
            ("cellpose_default", masks_cp_def, t_cp),
            ("cellpose_aggressive", masks_cp_agg, t_ca),
            ("stardist", masks_sd, t_sd),
            ("adaptive_consensus", masks_adaptive, 0),
        ]:
            n_det = int(masks.max())
            rec, n_rec = compute_recovery(masks, native_centroids)
            props = regionprops(masks)
            areas = np.array([p.area for p in props]) * PIXEL_SIZE ** 2 if props else np.array([])
            sol = np.mean([p.solidity for p in props]) if props else 0

            tile_results[name] = {
                "n_detected": n_det, "n_native": n_native,
                "recovery": round(rec, 3), "n_recovered": n_rec,
                "mean_area_um2": round(float(np.mean(areas)), 1) if len(areas) else 0,
                "solidity": round(sol, 3), "runtime": round(rt, 2),
            }
            extra = f" [{strategy}]" if name == "adaptive_consensus" else ""
            logger.info("  %s: %d cells, recovery=%.1f%%, solidity=%.3f%s",
                       name, n_det, rec * 100, sol, extra)

        results[label] = tile_results

    del dapi

    # Summary
    print("\n" + "=" * 110)
    print("XENIUM BREAST REP1 — SEGMENTATION BENCHMARK")
    print("=" * 110)
    hdr = f"{'FOV':8s} {'Native':>7s} | {'CP_def':>7s} {'CP_agg':>7s} {'StarDist':>8s} {'Adaptive':>8s} | {'CP_rec':>7s} {'SD_rec':>7s} {'AD_rec':>7s} {'strategy':>12s}"
    print(f"\n{hdr}")
    print("-" * 110)

    for label, tr in sorted(results.items()):
        n = tr["cellpose_default"]["n_native"]
        cp = tr["cellpose_default"]["n_detected"]
        ca = tr["cellpose_aggressive"]["n_detected"]
        sd = tr["stardist"]["n_detected"]
        ad = tr["adaptive_consensus"]["n_detected"]
        cr = tr["cellpose_default"]["recovery"]
        sr = tr["stardist"]["recovery"]
        ar = tr["adaptive_consensus"]["recovery"]

        # Determine strategy
        strat = "SD+CP_agg" if sd / max(cp, 1) > 2.0 else "CP_default"
        print(f"{label:8s} {n:>7d} | {cp:>7d} {ca:>7d} {sd:>8d} {ad:>8d} | {cr:>7.1%} {sr:>7.1%} {ar:>7.1%} {strat:>12s}")

    # Averages
    print("-" * 110)
    for method in ["cellpose_default", "stardist", "adaptive_consensus"]:
        recs = [tr[method]["recovery"] for tr in results.values()]
        avg = sum(recs) / len(recs)
        print(f"{'AVG ' + method:40s} recovery={avg:.1%}")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", OUTPUT_DIR / "results.json")


if __name__ == "__main__":
    main()
