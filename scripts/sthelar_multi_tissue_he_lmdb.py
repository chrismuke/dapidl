#!/usr/bin/env python3
"""Build H&E LMDB from STHELAR slides at the exact same cell positions as the
existing DAPI multi-tissue LMDB. Output:
    /mnt/work/datasets/derived/sthelar-multitissue-p128-he/

Each entry: int64 label (8 bytes, native endian) + uint8 H&E patch (3, 128, 128).
This mirrors the DAPI LMDB layout but with 3-channel uint8 instead of 1-channel
uint16. Train/val/test splits will be identical because we use the same labels.npy.

The cell positions are derived deterministically from the slide ordering and the
50k-patches-per-slide cap, exactly matching `sthelar_multi_tissue_lmdb.py`. We
use seed=42 for sub-sampling so the chosen cells match.

Usage:
    uv run python scripts/sthelar_multi_tissue_he_lmdb.py
"""
from __future__ import annotations

import gc
import json
import logging
import struct
import time
from pathlib import Path

import lmdb
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

STHELAR_DIR = Path("/mnt/work/datasets/STHELAR/sdata_slides")
OUTPUT_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128-he")
DAPI_LMDB_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128")
PATCH_SIZE = 128
HALF_PATCH = PATCH_SIZE // 2
MAX_PER_SLIDE = 50_000
MAP_SIZE = 80 * 1024**3  # 80 GB cap

# Same mapping as DAPI build
STHELAR_TO_CL = {
    "T": "T cell",
    "B": "B cell",
    "Perivascular": "pericyte",
    "Monocyte/Macrophage": "macrophage",
    "Mast": "mast cell",
    "Epithelial": "epithelial cell",
    "Fibroblast": "fibroblast",
    "Endothelial": "endothelial cell",
    "Adipocyte": "adipocyte",
}


def get_he_to_global_affine(reader) -> np.ndarray | None:
    """Read the H&E → global 2D affine matrix from the zarr's multiscales attrs.

    Returns a 3x3 numpy array mapping H&E (y_he, x_he, 1) → global (x_g, y_g, 1).
    NOTE the axis swap in SpatialData: H&E is in (c, y, x) order, global is (c, x, y).
    Returns None if no affine transform found (treat as identity).
    """
    multiscales = reader.store["images"]["he"].attrs.get("multiscales", [])
    for ms in multiscales:
        for ct in ms.get("coordinateTransformations", []):
            if ct.get("output", {}).get("name") == "global" and ct.get("type") == "affine":
                A = np.array(ct["affine"])  # 4x4 in (c, y_or_x, x_or_y) order
                # Take the spatial 2D portion (rows 1,2 and cols 1,2 + translation col 3)
                M = np.eye(3)
                M[0, 0] = A[1, 1]; M[0, 1] = A[1, 2]; M[0, 2] = A[1, 3]
                M[1, 0] = A[2, 1]; M[1, 1] = A[2, 2]; M[1, 2] = A[2, 3]
                return M
    return None


def process_slide(
    zarr_path: Path,
    env: lmdb.Environment,
    dapi_patch_idx: int,
    class_mapping: dict[str, int],
    slide_stats: dict,
) -> int:
    """Iterate the SAME cells in the SAME order as the DAPI build, advancing the
    DAPI patch_idx counter for every cell that DAPI's boundary check accepts.
    For each such cell, attempt H&E extraction (with the affine inverse) and,
    if HE bounds are also OK, write to the HE LMDB at key = dapi_patch_idx.

    Result: HE LMDB keys are a strict subset of DAPI LMDB keys, so the same
    patch_idx points to the same biological cell across both LMDBs.
    """
    from dapidl.data.sthelar import SthelarDataReader

    slide_name = zarr_path.name
    tissue = slide_name.replace("sdata_", "").split("_s")[0]

    log.info(f"Processing {slide_name} (tissue={tissue})")
    t0 = time.time()

    try:
        reader = SthelarDataReader(zarr_path)
    except Exception as e:
        log.error(f"  Failed to load {slide_name}: {e}")
        return dapi_patch_idx

    nuc_df = reader.nucleus_df
    if "label1" not in nuc_df.columns:
        log.warning(f"  No label1 column in {slide_name}, skipping")
        return dapi_patch_idx

    nuc_df = nuc_df.with_columns(
        pl.col("label1").replace_strict(STHELAR_TO_CL, default=None).alias("cl_name")
    )
    nuc_df = nuc_df.filter(pl.col("cl_name").is_not_null())
    n_cells = len(nuc_df)
    if n_cells == 0:
        log.warning(f"  No mapped cells in {slide_name}")
        return dapi_patch_idx

    if n_cells > MAX_PER_SLIDE:
        nuc_df = nuc_df.sample(n=MAX_PER_SLIDE, seed=42)
        log.info(f"  Sampled {MAX_PER_SLIDE} from {n_cells} cells")
    else:
        log.info(f"  Using all {n_cells} cells")

    for cl_name in nuc_df["cl_name"].unique().to_list():
        if cl_name not in class_mapping:
            class_mapping[cl_name] = len(class_mapping)

    # DAPI shape (no need to load actual data — just for boundary check)
    dapi_shape = reader.store["images"]["morpho"]["0"].shape  # (1, H, W)
    _, dapi_h, dapi_w = dapi_shape

    # Load full H&E (level 0). Shape (3, H, W), uint8.
    he = reader.load_he(level=0)
    _, he_h, he_w = he.shape

    # Compute global → H&E pixel affine (inverse of H&E → global)
    M_he2global = get_he_to_global_affine(reader)
    if M_he2global is None:
        M_global2he = np.eye(3)
        log.info(f"  H&E identity transform; HE shape={he.shape}")
    else:
        M_global2he = np.linalg.inv(M_he2global)
        log.info(f"  HE shape={he.shape}, scale_y={M_he2global[0,0]:.3f}, scale_x={M_he2global[1,1]:.3f}")

    centroids = reader.get_centroids_pixels()
    all_cell_ids = reader.get_cell_ids()
    centroid_map = {cid: (centroids[i, 0], centroids[i, 1]) for i, cid in enumerate(all_cell_ids)}

    n_dapi_pass = 0
    n_he_written = 0
    with env.begin(write=True) as txn:
        for row in nuc_df.iter_rows(named=True):
            cell_id = row["cell_id"]
            cl_name = row["cl_name"]
            label_idx = class_mapping[cl_name]

            if cell_id not in centroid_map:
                continue

            cx_d_raw, cy_d_raw = centroid_map[cell_id]
            cx_d = int(round(cx_d_raw))
            cy_d = int(round(cy_d_raw))

            # ---- Replicate DAPI boundary check ----
            d_y0, d_y1 = cy_d - HALF_PATCH, cy_d + HALF_PATCH
            d_x0, d_x1 = cx_d - HALF_PATCH, cx_d + HALF_PATCH
            if d_y0 < 0 or d_x0 < 0 or d_y1 > dapi_h or d_x1 > dapi_w:
                continue  # DAPI build also skipped this cell — no patch_idx given

            # This cell PASSED DAPI bounds; advance the counter (matches DAPI patch_idx)
            current_dapi_idx = dapi_patch_idx
            dapi_patch_idx += 1
            n_dapi_pass += 1

            # ---- Now attempt H&E extraction at the same cell ----
            global_xy = np.array([cx_d_raw, cy_d_raw, 1.0])
            yhe_xhe = M_global2he @ global_xy
            cy_h = int(round(yhe_xhe[0]))
            cx_h = int(round(yhe_xhe[1]))

            h_y0, h_y1 = cy_h - HALF_PATCH, cy_h + HALF_PATCH
            h_x0, h_x1 = cx_h - HALF_PATCH, cx_h + HALF_PATCH
            if h_y0 < 0 or h_x0 < 0 or h_y1 > he_h or h_x1 > he_w:
                continue  # HE bounds fail — leave HE key absent

            patch = he[:, h_y0:h_y1, h_x0:h_x1]
            if patch.shape != (3, PATCH_SIZE, PATCH_SIZE):
                continue

            key = struct.pack(">Q", current_dapi_idx)
            label_bytes = np.array([label_idx], dtype=np.int64).tobytes()
            value = label_bytes + np.ascontiguousarray(patch).tobytes()
            txn.put(key, value)
            n_he_written += 1

    elapsed = time.time() - t0
    slide_stats[slide_name] = {
        "tissue": tissue, "cells_total": n_cells,
        "dapi_passes": n_dapi_pass, "patches_written": n_he_written,
        "time_s": round(elapsed, 1),
    }
    log.info(f"  DAPI bounds OK on {n_dapi_pass}, HE LMDB written {n_he_written} (drop {n_dapi_pass - n_he_written}) in {elapsed:.1f}s")

    del he, reader, nuc_df
    gc.collect()
    return dapi_patch_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lmdb_path = OUTPUT_DIR / "patches.lmdb"

    slides = sorted(STHELAR_DIR.glob("*.zarr"))
    slides = [s for s in slides if s.is_dir()]
    log.info(f"Found {len(slides)} STHELAR slides")

    env = lmdb.open(str(lmdb_path), map_size=MAP_SIZE)
    class_mapping: dict[str, int] = {}
    slide_stats: dict = {}
    patch_idx = 0

    for i, slide_path in enumerate(slides):
        log.info(f"\n{'='*60}")
        log.info(f"Slide {i+1}/{len(slides)}: {slide_path.name}")
        log.info(f"{'='*60}")
        patch_idx = process_slide(slide_path, env, patch_idx, class_mapping, slide_stats)
        log.info(f"  Running total: {patch_idx:,} HE patches, {len(class_mapping)} classes")

    env.close()

    # Reuse the existing DAPI LMDB's labels.npy and class_mapping
    # (cells were sub-sampled with the same seed → identical assignment).
    # We still write our own to verify.
    with open(OUTPUT_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    with open(OUTPUT_DIR / "slide_stats.json", "w") as f:
        json.dump(slide_stats, f, indent=2)

    metadata = {
        "n_samples": patch_idx, "n_classes": len(class_mapping),
        "n_tissues": len({v["tissue"] for v in slide_stats.values()}),
        "n_slides": len(slides), "patch_size": PATCH_SIZE,
        "patch_shape": [3, PATCH_SIZE, PATCH_SIZE],
        "dtype": "uint8", "format": "lmdb",
        "platform": "sthelar", "modality": "he",
        "max_per_slide": MAX_PER_SLIDE,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(lmdb_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE: {patch_idx:,} HE patches from {len(slides)} slides")
    log.info(f"Classes: {len(class_mapping)}")
    log.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
