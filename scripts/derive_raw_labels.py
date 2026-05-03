"""Derive raw_labels.npy aligned to the LMDB by mirroring the LMDB build's filter.

The LMDB build (breast_dapi_lmdb.py) drops cells based on:
  - Xenium: missing barcode in Janesick GT, or 17-class -> coarse returns None
  - STHELAR: label1 not in STHELAR_LABEL1_TO_COARSE
  - Both: OOB patch (centroid too close to image edge)

To produce labels_medium.npy / labels_fine.npy aligned to labels.npy, we need
to walk the SAME cells in the SAME order. This script runs the iteration
without writing patches, just collecting the original raw label string for
each kept cell.

Output: raw_labels.npy (object array, len == labels.npy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl
from loguru import logger

# Reuse exact filter logic from the LMDB build script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from breast_dapi_lmdb import (
    STHELAR_LABEL1_TO_COARSE, COARSE_TO_IDX,
    _xenium_fine_to_coarse, _load_xenium_supervised_gt,
    XENIUM_BASE, STHELAR_BASE,
)
from dapidl.data.sthelar import SthelarDataReader
from dapidl.data.xenium import XeniumDataReader

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
PATCH_SIZE = 128


def _walk_xenium(rep_name: str):
    """Yield raw fine-class string for each cell the LMDB would have kept."""
    raw_dir = XENIUM_BASE / f"xenium-breast-tumor-{rep_name}"
    reader = XenniumDataReader(raw_dir / "outs") if False else XeniumDataReader(raw_dir / "outs")
    dapi = reader.image
    h, w = dapi.shape
    half = PATCH_SIZE // 2
    gt = _load_xenium_supervised_gt(rep_name)
    centroids = reader.get_centroids_pixels()
    cell_ids = reader.get_cell_ids()

    candidates = []
    for i, cid in enumerate(cell_ids):
        fine = gt.get(str(cid))
        if fine is None:
            continue
        coarse = _xenium_fine_to_coarse(fine)
        if coarse is None:
            continue
        candidates.append((i, fine))

    out = []
    for i, fine in candidates:
        cx, cy = int(round(centroids[i, 0])), int(round(centroids[i, 1]))
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        out.append(fine)
    logger.info(f"  xenium_{rep_name}: produced {len(out)} raw labels")
    return out


def _walk_sthelar(slide_zarr: Path):
    slide_name = slide_zarr.name.replace("sdata_", "").replace(".zarr", "")
    reader = SthelarDataReader(slide_zarr)
    nuc_df = reader.nucleus_df
    if "label1" not in nuc_df.columns:
        return []
    # Same filter as LMDB build:
    nuc_df = nuc_df.with_columns(
        pl.col("label1").replace_strict(STHELAR_LABEL1_TO_COARSE,
                                        default=None).alias("coarse_filter")
    ).filter(pl.col("coarse_filter").is_not_null())
    if len(nuc_df) == 0:
        return []
    # We need image bounds for OOB check
    dapi = reader.image
    h, w = dapi.shape
    half = PATCH_SIZE // 2

    # Pick the raw label column to keep — prefer ct_tangram, fall back to label1
    raw_col = "ct_tangram" if "ct_tangram" in nuc_df.columns else "label1"
    out = []
    n_oob = 0
    for row in nuc_df.iter_rows(named=True):
        cx = int(round(row["x_centroid_px"]))
        cy = int(round(row["y_centroid_px"]))
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            n_oob += 1
            continue
        out.append(str(row[raw_col]))
    logger.info(f"  sthelar_{slide_name}: produced {len(out)} raw labels (skipped {n_oob} OOB)")
    return out


def main():
    stats = json.loads((LMDB_DIR / "slide_stats.json").read_text())
    expected_per_source = {s["source"]: s["n_written"] for s in stats}
    logger.info(f"Expected per-source counts (from LMDB slide_stats):")
    for k, v in expected_per_source.items():
        logger.info(f"  {k:30s} {v:>10,d}")

    raw_labels = []
    for s in stats:
        src = s["source"]
        n_expected = s["n_written"]
        if src.startswith("xenium_rep"):
            rep_name = src.split("xenium_")[1]
            chunk = _walk_xenium(rep_name)
        elif src.startswith("sthelar_breast_"):
            slide = src.split("sthelar_breast_")[1]
            slide_zarr = STHELAR_BASE / f"sdata_breast_{slide}.zarr"
            chunk = _walk_sthelar(slide_zarr)
        else:
            logger.warning(f"unknown source {src}")
            continue
        if len(chunk) != n_expected:
            logger.error(f"  {src}: produced {len(chunk)} labels but LMDB wrote {n_expected}")
            # fail loud
            raise SystemExit(f"alignment mismatch for {src}")
        raw_labels.extend(chunk)

    raw_arr = np.array(raw_labels, dtype=object)
    out_path = LMDB_DIR / "raw_labels.npy"
    np.save(out_path, raw_arr)
    logger.info(f"wrote {out_path}: {len(raw_arr):,} raw labels")
    logger.info(f"unique raw labels: {len(set(raw_arr.tolist()))}")


if __name__ == "__main__":
    main()
