#!/usr/bin/env python
"""
Compare starpose (StarDist + Cellpose) vs CellViT segmentation on STHELAR breast_s0.

Runs starpose on 5 representative DAPI FOVs and compares detected nuclei
against CellViT ground truth centroids.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, "/home/chrism/git/starpose/src")

STHELAR_ZARR = Path(
    "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
    "sdata_breast_s0.zarr"
)
OUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/segmentation_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE = 0.2125  # um/px for Xenium


def load_dapi_and_centroids():
    """Load DAPI image and CellViT centroids."""
    import anndata as ad

    print("Loading DAPI image (zarr)...")
    dapi = zarr.open_array(str(STHELAR_ZARR / "images" / "morpho" / "0"))
    print(f"  DAPI shape: {dapi.shape}, dtype: {dapi.dtype}")

    print("Loading CellViT centroids...")
    adata = ad.read_zarr(str(STHELAR_ZARR / "tables" / "table_cells"))
    coords_um = adata.obsm["spatial"]  # (N, 2) in microns [x, y]
    # Convert to pixel coordinates
    coords_px = coords_um / PIXEL_SIZE  # [x_px, y_px]
    print(f"  CellViT: {len(coords_px):,} cells")
    print(f"  Coords (px): x=[{coords_px[:,0].min():.0f}, {coords_px[:,0].max():.0f}], "
          f"y=[{coords_px[:,1].min():.0f}, {coords_px[:,1].max():.0f}]")

    return dapi, coords_px, adata


def select_fovs(coords_px, img_shape, n_fovs=5, tile_size=4096):
    """Select representative FOVs by density."""
    from scipy.spatial import cKDTree

    h, w = img_shape[1], img_shape[2]

    # Grid of possible FOV centers
    step = tile_size // 2
    centers = []
    for y in range(tile_size // 2, h - tile_size // 2, step):
        for x in range(tile_size // 2, w - tile_size // 2, step):
            y0, x0 = y - tile_size // 2, x - tile_size // 2
            y1, x1 = y0 + tile_size, x0 + tile_size
            # Count CellViT cells in this FOV
            in_fov = ((coords_px[:, 0] >= x0) & (coords_px[:, 0] < x1) &
                      (coords_px[:, 1] >= y0) & (coords_px[:, 1] < y1))
            n = in_fov.sum()
            if n > 10:
                centers.append((y, x, n))

    if not centers:
        raise ValueError("No valid FOV centers found")

    # Sort by density and pick diverse FOVs
    centers.sort(key=lambda c: c[2])
    n_total = len(centers)

    # Pick: densest, sparsest, median, 25th percentile, 75th percentile
    indices = [0, n_total // 4, n_total // 2, 3 * n_total // 4, n_total - 1]
    labels = ["sparse", "low", "medium", "high", "dense"]

    fovs = []
    for idx, label in zip(indices[:n_fovs], labels[:n_fovs]):
        y, x, n = centers[idx]
        y0, x0 = y - tile_size // 2, x - tile_size // 2
        fovs.append({
            "label": label,
            "y0": y0, "x0": x0,
            "y1": y0 + tile_size, "x1": x0 + tile_size,
            "cellvit_count": int(n),
        })
        print(f"  FOV {label}: [{y0}:{y0+tile_size}, {x0}:{x0+tile_size}] — {n} CellViT cells")

    return fovs


def run_starpose_on_fov(dapi, fov, method="adaptive"):
    """Run starpose segmentation on a single FOV."""
    from starpose.methods.stardist import StarDist2DAdapter
    from starpose.methods.cellpose import CellposeSAM
    from starpose.consensus.adaptive import AdaptiveConsensus

    tile = np.array(dapi[0, fov["y0"]:fov["y1"], fov["x0"]:fov["x1"]])

    results = {}

    # Run StarDist
    sd = StarDist2DAdapter(gpu=True)
    t0 = time.time()
    sd_result = sd.segment(tile, pixel_size=PIXEL_SIZE)
    results["stardist"] = {
        "n_cells": sd_result.n_cells,
        "runtime": round(time.time() - t0, 2),
    }

    # Run Cellpose
    cp = CellposeSAM(gpu=True)
    t0 = time.time()
    cp_result = cp.segment(tile, pixel_size=PIXEL_SIZE)
    results["cellpose"] = {
        "n_cells": cp_result.n_cells,
        "runtime": round(time.time() - t0, 2),
    }

    # Run Adaptive Consensus
    ac = AdaptiveConsensus(gpu=True)
    t0 = time.time()
    ac_result = ac.segment(tile, pixel_size=PIXEL_SIZE)
    results["adaptive"] = {
        "n_cells": ac_result.n_cells,
        "runtime": round(time.time() - t0, 2),
        "strategy": ac_result.metadata.get("strategy", ""),
        "density_ratio": round(ac_result.metadata.get("density_ratio", 0), 2),
    }

    return results, {
        "stardist": sd_result,
        "cellpose": cp_result,
        "adaptive": ac_result,
    }


def compute_recovery(result, cellvit_px, fov, search_radius_px=20):
    """Compute centroid recovery rate: fraction of CellViT centroids
    matched to a starpose detection within search_radius."""
    from scipy.spatial import cKDTree

    # CellViT centroids in this FOV (in FOV-local pixel coords)
    in_fov = ((cellvit_px[:, 0] >= fov["x0"]) & (cellvit_px[:, 0] < fov["x1"]) &
              (cellvit_px[:, 1] >= fov["y0"]) & (cellvit_px[:, 1] < fov["y1"]))
    cv_local = cellvit_px[in_fov].copy()
    cv_local[:, 0] -= fov["x0"]  # x
    cv_local[:, 1] -= fov["y0"]  # y
    # CellViT is [x, y], starpose centroids are [row, col] = [y, x]
    cv_yx = cv_local[:, ::-1]  # → [y, x]

    if len(cv_yx) == 0 or result.n_cells == 0:
        return {"recovery": 0, "n_cellvit": len(cv_yx), "n_starpose": result.n_cells}

    sp_centroids = result.centroids  # (N, 2) [y, x]

    # KDTree on starpose centroids, query CellViT centroids
    tree = cKDTree(sp_centroids)
    distances, _ = tree.query(cv_yx, k=1)

    recovered = (distances <= search_radius_px).sum()
    recovery_rate = recovered / len(cv_yx)

    # Also check reverse: starpose cells matched to CellViT
    tree_cv = cKDTree(cv_yx)
    dist_rev, _ = tree_cv.query(sp_centroids, k=1)
    precision = (dist_rev <= search_radius_px).sum() / len(sp_centroids) if len(sp_centroids) > 0 else 0

    return {
        "recovery": round(recovery_rate, 4),
        "precision": round(precision, 4),
        "n_cellvit": int(len(cv_yx)),
        "n_starpose": int(result.n_cells),
        "ratio": round(result.n_cells / max(len(cv_yx), 1), 3),
        "mean_distance": round(float(distances.mean()), 2),
    }


def main():
    t_total = time.time()

    dapi, cellvit_px, adata = load_dapi_and_centroids()

    print("\nSelecting 5 FOVs by density...")
    fovs = select_fovs(cellvit_px, dapi.shape)

    all_results = {}
    for fov in fovs:
        label = fov["label"]
        print(f"\n{'='*60}")
        print(f"FOV: {label} ({fov['cellvit_count']} CellViT cells)")
        print(f"{'='*60}")

        # Run starpose
        print("  Running starpose (StarDist + Cellpose + Adaptive)...")
        method_results, seg_results = run_starpose_on_fov(dapi, fov)

        # Compare with CellViT
        print("  Computing recovery metrics...")
        fov_summary = {"fov": fov, "methods": {}}

        for method_name, seg_result in seg_results.items():
            recovery = compute_recovery(seg_result, cellvit_px, fov)
            method_results[method_name].update(recovery)
            fov_summary["methods"][method_name] = method_results[method_name]

            print(f"    {method_name}: {seg_result.n_cells} cells, "
                  f"recovery={recovery['recovery']:.1%}, "
                  f"precision={recovery['precision']:.1%}, "
                  f"ratio={recovery['ratio']:.2f}, "
                  f"time={method_results[method_name]['runtime']}s")

        all_results[label] = fov_summary

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Starpose vs CellViT Segmentation")
    print(f"{'='*60}")
    print(f"\n{'FOV':<10} {'Method':<12} {'CellViT':<10} {'Starpose':<10} {'Recovery':<10} {'Precision':<10} {'Ratio':<8} {'Time':<8}")
    print("-" * 78)

    for label, data in all_results.items():
        for method_name, m in data["methods"].items():
            print(f"{label:<10} {method_name:<12} {m['n_cellvit']:<10} {m['n_starpose']:<10} "
                  f"{m['recovery']:.1%}      {m['precision']:.1%}       {m['ratio']:<8.2f} {m['runtime']:<8.1f}")

    # Averages
    print(f"\n{'Averages':}")
    for method_name in ["stardist", "cellpose", "adaptive"]:
        recoveries = [d["methods"][method_name]["recovery"] for d in all_results.values()]
        precisions = [d["methods"][method_name]["precision"] for d in all_results.values()]
        ratios = [d["methods"][method_name]["ratio"] for d in all_results.values()]
        runtimes = [d["methods"][method_name]["runtime"] for d in all_results.values()]
        print(f"  {method_name:<12} recovery={np.mean(recoveries):.1%}  precision={np.mean(precisions):.1%}  "
              f"ratio={np.mean(ratios):.2f}  time={np.mean(runtimes):.1f}s")

    # Save
    out_path = OUT_DIR / "starpose_vs_cellvit_breast_s0.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
