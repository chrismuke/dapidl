#!/usr/bin/env python3
"""Build the 6-source nucleus-centered pilot LMDB (v3) with cell_id tracking.

Same nuc-centered extraction as v2 but:
- Re-extracts ALL 6 breast sources (rep1, rep2, s0, s1, s3, s6) from raw — no
  byte-copy, because v2's byte-copy lost the cell_id needed to look up
  medium labels later (combined_final_label for STHELAR).
- Persists a `patch_registry.parquet` so each row carries
  (row_idx, slide, cell_id, coarse_idx) and downstream tools can both look up
  medium labels by (slide, cell_id) AND fetch a specific patch by row_idx —
  this is the "unique patch ID" the user asked for.

LMDB / labels.npy / slide_stats.json format is unchanged so quality_control_seg
runs without changes.

Usage:
    uv run python scripts/build_pilot_lmdb_v3.py --per-source 5000
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
from loguru import logger

from dapidl.data.lazy_mosaic import LazyMosaic, normalize_crop, open_xenium_mosaic
from dapidl.data.sthelar import SthelarDataReader
from dapidl.data.xenium import XeniumDataReader
from dapidl.qc.io import FORMAT_KEY, FORMAT_U64KEY_SQUARE

DERIVED = Path("/mnt/work/datasets/derived")
LMDB_TXN_CHUNK = 5000  # commit write txn every N patches (review B8)
XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")
GT_XLSX = XENIUM_BASE / "xenium-breast-tumor-rep1" / "Cell_Barcode_Type_Matrices.xlsx"

COARSE_CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_CLASSES)}

JANESICK17_TO_COARSE = {
    "Stromal": "Stromal", "Invasive_Tumor": "Epithelial", "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial", "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial", "Myoepi_KRT15+": "Epithelial",
    "Endothelial": "Endothelial", "Macrophages_1": "Immune",
    "Macrophages_2": "Immune", "B_Cells": "Immune", "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune", "Mast_Cells": "Immune", "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune", "Perivascular-Like": "Stromal",
    "Unlabeled": None, "Stromal_&_T_Cell_Hybrid": None, "T_Cell_&_Tumor_Hybrid": None,
}

# STHELAR `label1` (9-class CL-style) -> coarse 4.
STHELAR_LABEL1_TO_COARSE = {
    "Endothelial": "Endothelial", "Perivascular": "Stromal",
    "Epithelial": "Epithelial", "T": "Immune", "B": "Immune",
    "Monocyte/Macrophage": "Immune", "Mast": "Immune",
    "Fibroblast": "Stromal", "Adipocyte": "Stromal",
}

REP_SHEETS = {
    "rep1": "Xenium R1 Fig1-5 (supervised)",
    "rep2": "Xenium R2 Fig1-5 (supervised)",
}


def _load_supervised_gt(rep_name: str) -> dict[str, str]:
    """Janesick supervised barcode -> fine class for rep1 or rep2."""
    import pandas as pd
    sheet = REP_SHEETS[rep_name]
    df = pd.read_excel(GT_XLSX, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    label_col = "Cluster" if "Cluster" in df.columns else "Annotation"
    return dict(zip(df["Barcode"].astype(str), df[label_col].astype(str)))


def _normalize(image: np.ndarray) -> tuple[np.ndarray, dict]:
    p_low = float(np.percentile(image, 1))
    p_high = float(np.percentile(image, 99.5))
    if p_high <= p_low:
        p_high = p_low + 1
    norm = np.clip(image.astype(np.float32), p_low, p_high)
    norm = (norm - p_low) / (p_high - p_low)
    return norm, {"p_low": p_low, "p_high": p_high}


def _stratified_pick(
    candidates: list[tuple[int, int, str]], cap: int, rng: np.random.Generator
) -> list[tuple[int, int, str]]:
    """Equal-per-class cap. Candidate = (source_idx, coarse_idx, cell_id_str)."""
    by_cls: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for src_idx, cls, cid in candidates:
        by_cls[cls].append((src_idx, cid))
    per = max(1, cap // max(1, len(by_cls)))
    picked: list[tuple[int, int, str]] = []
    for cls, items in by_cls.items():
        n = min(per, len(items))
        sel = rng.choice(len(items), size=n, replace=False)
        picked.extend([(items[i][0], cls, items[i][1]) for i in sel])
    rng.shuffle(picked)
    return picked


def _write_patch(txn, row_idx: int, patch_u16: np.ndarray, coarse_idx: int) -> None:
    """LMDB value = int64 label + uint16 patch bytes (matches existing format)."""
    label_bytes = np.array([coarse_idx], dtype=np.int64).tobytes()
    txn.put(struct.pack(">Q", row_idx), label_bytes + patch_u16.tobytes())


def extract_xenium(
    env: lmdb.Environment, rep_name: str, idx_start: int, patch_size: int,
    per_source: int, all_labels: list[int], registry: list[dict],
    rng: np.random.Generator,
) -> tuple[int, dict]:
    half = patch_size // 2
    raw_dir = XENIUM_BASE / f"xenium-breast-tumor-{rep_name}"
    reader = XeniumDataReader(raw_dir / "outs")
    gt = _load_supervised_gt(rep_name)
    centroids = reader.get_nucleus_centroids_pixels()
    cell_ids = reader.get_cell_ids()

    candidates: list[tuple[int, int, str]] = []
    for i, cid in enumerate(cell_ids):
        fine = gt.get(str(cid))
        if fine is None:
            continue
        coarse = JANESICK17_TO_COARSE.get(fine)
        if coarse is None:
            continue
        candidates.append((i, COARSE_TO_IDX[coarse], str(cid)))
    logger.info(f"{rep_name}: {len(candidates)} labelled cells")
    picked = _stratified_pick(candidates, per_source, rng) if per_source > 0 else candidates

    slide = f"xenium_{rep_name}_nuc"
    n_written, n_oob = 0, 0
    idx = idx_start
    with open_xenium_mosaic(reader.image_path) as mosaic:   # lazy crops, no full load (B8)
        h, w = mosaic.shape
        p_low, p_high = mosaic.subsample_percentiles(1.0, 99.5)
        norm = {"p_low": p_low, "p_high": p_high}
        txn = env.begin(write=True)
        try:
            for src_i, coarse_idx, cid in picked:
                cx, cy = int(round(centroids[src_i, 0])), int(round(centroids[src_i, 1]))
                y0, y1 = cy - half, cy + half
                x0, x1 = cx - half, cx + half
                if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                    n_oob += 1
                    continue
                crop = mosaic.read(y0, y1, x0, x1)
                if crop.shape != (patch_size, patch_size):
                    n_oob += 1
                    continue
                _write_patch(txn, idx, normalize_crop(crop, p_low, p_high), coarse_idx)
                all_labels.append(coarse_idx)
                registry.append({"row_idx": idx, "slide": slide, "cell_id": cid,
                                 "coarse_idx": coarse_idx})
                idx += 1
                n_written += 1
                if n_written % LMDB_TXN_CHUNK == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
    logger.info(f"{slide}: wrote {n_written}, skipped {n_oob} OOB")
    return idx, {"source": slide, "n_written": n_written,
                 "n_candidates": len(picked), "norm": norm}


def extract_sthelar(
    env: lmdb.Environment, slide_zarr: Path, idx_start: int, patch_size: int,
    per_source: int, all_labels: list[int], registry: list[dict],
    rng: np.random.Generator,
) -> tuple[int, dict]:
    half = patch_size // 2
    slide_name = slide_zarr.name.replace("sdata_", "").replace(".zarr", "")
    slide = f"sthelar_{slide_name}"
    reader = SthelarDataReader(slide_zarr)
    nuc = reader.nucleus_df
    if "label1" not in nuc.columns:
        logger.warning(f"{slide}: no label1, skipping")
        return idx_start, {}

    nuc = nuc.with_columns(
        pl.col("label1").replace_strict(STHELAR_LABEL1_TO_COARSE, default=None).alias("coarse")
    ).filter(pl.col("coarse").is_not_null())

    candidates: list[tuple[int, int, str]] = []
    for row in nuc.iter_rows(named=True):
        candidates.append((-1, COARSE_TO_IDX[row["coarse"]], str(row["cell_id"])))
    logger.info(f"{slide}: {len(candidates)} labelled cells")
    picked = _stratified_pick(candidates, per_source, rng) if per_source > 0 else candidates

    # centroid_map: cell_id -> (x_px, y_px)
    all_cids = reader.get_cell_ids()
    centroids = reader.get_centroids_pixels()
    cmap = {str(cid): (centroids[i, 0], centroids[i, 1]) for i, cid in enumerate(all_cids)}

    mosaic = LazyMosaic(reader.dapi_lazy)        # lazy (1,H,W) crops, no full load (B8)
    h, w = mosaic.shape
    p_low, p_high = mosaic.subsample_percentiles(1.0, 99.5)
    norm = {"p_low": p_low, "p_high": p_high}

    n_written, n_oob, n_miss = 0, 0, 0
    idx = idx_start
    txn = env.begin(write=True)
    try:
        for _, coarse_idx, cid in picked:
            if cid not in cmap:
                n_miss += 1
                continue
            cx, cy = cmap[cid]
            cx, cy = int(round(cx)), int(round(cy))
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                n_oob += 1
                continue
            crop = mosaic.read(y0, y1, x0, x1)
            if crop.shape != (patch_size, patch_size):
                n_oob += 1
                continue
            _write_patch(txn, idx, normalize_crop(crop, p_low, p_high), coarse_idx)
            all_labels.append(coarse_idx)
            registry.append({"row_idx": idx, "slide": slide, "cell_id": cid,
                             "coarse_idx": coarse_idx})
            idx += 1
            n_written += 1
            if n_written % LMDB_TXN_CHUNK == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except BaseException:
        txn.abort()
        raise
    logger.info(f"{slide}: wrote {n_written} (OOB {n_oob}, miss {n_miss})")
    return idx, {"source": slide, "n_written": n_written,
                 "n_candidates": len(picked), "norm": norm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=5000)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--output", type=str, default="breast-pilot-6source-dapi-p128-nuc-v3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--map-size-gb", type=int, default=8)
    args = ap.parse_args()

    out_dir = DERIVED / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "patches.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=args.map_size_gb * 1024**3)

    rng = np.random.default_rng(args.seed)
    all_labels: list[int] = []
    registry: list[dict] = []
    slide_stats: list[dict] = []
    idx = 0

    for rep_name in ["rep1", "rep2"]:
        idx, st = extract_xenium(env, rep_name, idx, args.patch_size,
                                  args.per_source, all_labels, registry, rng)
        if st:
            slide_stats.append(st)

    for slide_zarr in sorted(STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        if not slide_zarr.is_dir():
            continue
        idx, st = extract_sthelar(env, slide_zarr, idx, args.patch_size,
                                   args.per_source, all_labels, registry, rng)
        if st:
            slide_stats.append(st)

    # Stamp the self-describing format tag (B10) so the QC reader never has to
    # guess the layout from key bytes (Format B: >Q keys, int64 label + square).
    with env.begin(write=True) as txn:
        txn.put(FORMAT_KEY, FORMAT_U64KEY_SQUARE)
    env.close()

    labels_arr = np.array(all_labels, dtype=np.int64)
    np.save(out_dir / "labels.npy", labels_arr)
    (out_dir / "class_mapping.json").write_text(json.dumps(COARSE_TO_IDX, indent=2))

    reg_df = pl.DataFrame(registry)
    reg_df.write_parquet(out_dir / "patch_registry.parquet")

    class_counts = dict(Counter(all_labels))
    metadata = {
        "n_samples": int(idx), "n_classes": 4,
        "patch_size": args.patch_size, "patch_shape": [args.patch_size, args.patch_size],
        "dtype": "uint16", "format": "lmdb", "normalization": "adaptive_per_slide",
        "platform": "xenium+sthelar",
        "scope": "pilot v3 — 6 sources nuc-centered, cell_id tracked",
        "class_names": COARSE_CLASSES,
        "class_counts": {COARSE_CLASSES[k]: v for k, v in class_counts.items()},
        "source_counts": {s["source"]: s["n_written"] for s in slide_stats},
        "n_sources": len(slide_stats),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (lmdb_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "slide_stats.json").write_text(json.dumps(slide_stats, indent=2))

    logger.info(f"\nPILOT v3 COMPLETE: {idx:,} patches; registry size {len(registry):,}")
    for k, v in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / idx if idx else 0.0
        logger.info(f"  {COARSE_CLASSES[k]:<14s}: {v:>6,} ({pct:5.1f}%)")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
