#!/usr/bin/env python3
"""Build the 3-source nucleus-centered pilot LMDB for QC validation.

- xenium_rep1: re-extract DAPI patches centered on nucleus polygon centroids
  (XeniumDataReader.get_nucleus_centroids_pixels), normalized per-slide.
- sthelar_breast_s0, sthelar_breast_s6: byte-copy patches (and existing
  label-int64 prefix) directly from the shipped breast-6source-dapi-p128 LMDB
  — those rows are already nucleus-centered (offset 0.000 µm verified) so a
  re-extract would reproduce identical bytes.

Output layout matches the existing breast-multisource builder so
quality_control_seg, read_patches, _slide_groups, etc. work unchanged:
    breast-pilot-3source-dapi-p128-nuc/
        patches.lmdb/
        labels.npy
        class_mapping.json
        metadata.json
        slide_stats.json

Per-source cap is stratified by the existing coarse label so all 4 classes are
represented (else Stromal/Endothelial would be too thin to fill the collages).

Usage:
    uv run python scripts/build_pilot_lmdb.py --per-source 10000
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
from loguru import logger

from dapidl.data.xenium import XeniumDataReader

DERIVED = Path("/mnt/work/datasets/derived")
XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
SOURCE_LMDB = DERIVED / "breast-6source-dapi-p128"
GT_XLSX = XENIUM_BASE / "xenium-breast-tumor-rep1" / "Cell_Barcode_Type_Matrices.xlsx"

COARSE_CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_CLASSES)}

# Janesick 17-class supervised -> coarse 4 (mirror of breast_dapi_lmdb.py).
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

REP_TO_SHEET = {"rep1": "Xenium R1 Fig1-5 (supervised)"}


def _load_supervised_gt() -> dict[str, str]:
    import pandas as pd
    sheet = REP_TO_SHEET["rep1"]
    df = pd.read_excel(GT_XLSX, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    label_col = "Cluster" if "Cluster" in df.columns else "Annotation"
    return dict(zip(df["Barcode"].astype(str), df[label_col].astype(str)))


def _normalize_dapi(image: np.ndarray) -> tuple[np.ndarray, dict]:
    p_low = float(np.percentile(image, 1))
    p_high = float(np.percentile(image, 99.5))
    if p_high <= p_low:
        p_high = p_low + 1
    norm = np.clip(image.astype(np.float32), p_low, p_high)
    norm = (norm - p_low) / (p_high - p_low)
    return norm, {"p_low": p_low, "p_high": p_high}


def _stratified_pick(
    candidates: list[tuple[int, int]], cap: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Equal-per-class cap; falls back to fewer per class if a class is small."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, c in candidates:
        by_class[c].append(i)
    per_class = max(1, cap // max(1, len(by_class)))
    picked: list[tuple[int, int]] = []
    for c, idx_list in by_class.items():
        n_take = min(per_class, len(idx_list))
        sampled = rng.choice(idx_list, size=n_take, replace=False)
        picked.extend([(int(i), c) for i in sampled])
    rng.shuffle(picked)
    return picked


def reextract_rep1_nuc(
    env: lmdb.Environment, idx_start: int, patch_size: int, per_source: int,
    all_labels: list[int], rng: np.random.Generator,
) -> tuple[int, dict]:
    """Re-extract rep1 patches on nucleus polygon centroids."""
    half = patch_size // 2
    raw_dir = XENIUM_BASE / "xenium-breast-tumor-rep1"
    reader = XeniumDataReader(raw_dir / "outs")
    dapi = reader.image
    h, w = dapi.shape
    dapi_norm, norm = _normalize_dapi(dapi)

    gt = _load_supervised_gt()
    nuc_centroids = reader.get_nucleus_centroids_pixels()
    cell_ids = reader.get_cell_ids()

    candidates: list[tuple[int, int]] = []
    for i, cid in enumerate(cell_ids):
        fine = gt.get(str(cid))
        if fine is None:
            continue
        coarse = JANESICK17_TO_COARSE.get(fine)
        if coarse is None:
            continue
        candidates.append((i, COARSE_TO_IDX[coarse]))

    logger.info(f"rep1: {len(candidates)} cells with valid supervised labels")
    picked = _stratified_pick(candidates, per_source, rng) if per_source > 0 else candidates

    n_written, n_oob = 0, 0
    with env.begin(write=True) as txn:
        idx = idx_start
        for i, coarse_idx in picked:
            cx, cy = int(round(nuc_centroids[i, 0])), int(round(nuc_centroids[i, 1]))
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                n_oob += 1
                continue
            patch = dapi_norm[y0:y1, x0:x1]
            if patch.shape != (patch_size, patch_size):
                n_oob += 1
                continue
            patch_uint16 = (patch * 65535).clip(0, 65535).astype(np.uint16)
            label_bytes = np.array([coarse_idx], dtype=np.int64).tobytes()
            txn.put(struct.pack(">Q", idx), label_bytes + patch_uint16.tobytes())
            all_labels.append(coarse_idx)
            idx += 1
            n_written += 1

    logger.info(f"rep1 (nuc-centered): wrote {n_written}, skipped {n_oob} OOB")
    return idx, {"source": "xenium_rep1_nuc", "n_written": n_written,
                 "n_candidates": len(candidates), "norm": norm}


def bytecopy_source(
    src_env: lmdb.Environment, dst_env: lmdb.Environment,
    src_offset: int, src_n: int, idx_start: int, source: str,
    per_source: int, src_labels: np.ndarray, all_labels: list[int],
    rng: np.random.Generator,
) -> tuple[int, dict]:
    """Byte-copy a contiguous source range from src_env into dst_env, stratified.

    Filters labels < 0 (the source LMDB writes -1 for cells with no usable
    coarse label; they would otherwise pollute the collage stratification).
    """
    candidates = [(int(src_offset + j), int(src_labels[src_offset + j]))
                  for j in range(src_n) if src_labels[src_offset + j] >= 0]
    picked = _stratified_pick(candidates, per_source, rng) if per_source > 0 else candidates

    n_written = 0
    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        idx = idx_start
        for src_idx, lbl in picked:
            value = src_txn.get(struct.pack(">Q", src_idx))
            if value is None:
                continue
            dst_txn.put(struct.pack(">Q", idx), value)
            all_labels.append(lbl)
            idx += 1
            n_written += 1

    logger.info(f"{source}: byte-copied {n_written} patches "
                f"(picked {len(picked)} from {src_n} candidates)")
    return idx, {"source": source, "n_written": n_written,
                 "n_candidates": len(picked), "source_range": [src_offset, src_offset + src_n]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=10000,
                    help="Stratified cap per source (0 = no cap).")
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--output", type=str, default="breast-pilot-3source-dapi-p128-nuc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--map-size-gb", type=int, default=8)
    args = ap.parse_args()

    out_dir = DERIVED / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_lmdb_path = out_dir / "patches.lmdb"
    src_lmdb_path = SOURCE_LMDB / "patches.lmdb"

    if not src_lmdb_path.exists():
        raise SystemExit(f"Source LMDB not found: {src_lmdb_path}")

    src_stats = json.loads((SOURCE_LMDB / "slide_stats.json").read_text())
    src_labels = np.load(SOURCE_LMDB / "labels.npy")

    # Build offset table from source slide_stats.
    offsets: dict[str, tuple[int, int]] = {}
    off = 0
    for s in src_stats:
        n = int(s["n_written"])
        offsets[s["source"]] = (off, n)
        off += n

    logger.info(f"Source LMDB: {SOURCE_LMDB}")
    for k, (o, n) in offsets.items():
        logger.info(f"  {k}: offset={o:,}  n={n:,}")

    rng = np.random.default_rng(args.seed)
    src_env = lmdb.open(str(src_lmdb_path), readonly=True, lock=False, max_readers=1)
    dst_env = lmdb.open(str(dst_lmdb_path), map_size=args.map_size_gb * 1024**3)

    all_labels: list[int] = []
    slide_stats: list[dict] = []
    idx = 0

    # 1. rep1 re-extract on nucleus centroids
    idx, st = reextract_rep1_nuc(dst_env, idx, args.patch_size, args.per_source,
                                  all_labels, rng)
    slide_stats.append(st)

    # 2. s0 byte-copy
    s0_off, s0_n = offsets["sthelar_breast_s0"]
    idx, st = bytecopy_source(src_env, dst_env, s0_off, s0_n, idx,
                              "sthelar_breast_s0", args.per_source,
                              src_labels, all_labels, rng)
    slide_stats.append(st)

    # 3. s6 byte-copy
    s6_off, s6_n = offsets["sthelar_breast_s6"]
    idx, st = bytecopy_source(src_env, dst_env, s6_off, s6_n, idx,
                              "sthelar_breast_s6", args.per_source,
                              src_labels, all_labels, rng)
    slide_stats.append(st)

    src_env.close()
    dst_env.close()

    # Persist labels + class_mapping + metadata + slide_stats.
    labels_arr = np.array(all_labels, dtype=np.int64)
    np.save(out_dir / "labels.npy", labels_arr)
    (out_dir / "class_mapping.json").write_text(json.dumps(COARSE_TO_IDX, indent=2))

    class_counts = dict(Counter(all_labels))
    metadata = {
        "n_samples": int(idx),
        "n_classes": 4,
        "patch_size": args.patch_size,
        "patch_shape": [args.patch_size, args.patch_size],
        "dtype": "uint16",
        "format": "lmdb",
        "normalization": "adaptive_per_slide",
        "platform": "xenium+sthelar",
        "scope": "pilot — rep1 (nuc-centered) + s0 + s6 (= xenium-prime, byte-copy)",
        "class_names": COARSE_CLASSES,
        "class_counts": {COARSE_CLASSES[k]: v for k, v in class_counts.items()},
        "source_counts": {s["source"]: s["n_written"] for s in slide_stats},
        "n_sources": len(slide_stats),
        "parent_lmdb": str(SOURCE_LMDB),
        "centroid_source_per_slide": {
            "xenium_rep1_nuc": "nucleus_boundaries.parquet polygon centroid",
            "sthelar_breast_s0": "table_nuclei (already nucleus-centered, byte-copy)",
            "sthelar_breast_s6": "table_nuclei (already nucleus-centered, byte-copy)",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (dst_lmdb_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "slide_stats.json").write_text(json.dumps(slide_stats, indent=2))

    logger.info(f"\nPILOT LMDB COMPLETE: {idx:,} patches at p{args.patch_size}")
    for k, v in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / idx if idx else 0.0
        logger.info(f"  {COARSE_CLASSES[k]:<14s}: {v:>6,} ({pct:5.1f}%)")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
