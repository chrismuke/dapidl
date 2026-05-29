#!/usr/bin/env python3
"""Build a breast-multisource DAPI LMDB at a configurable patch size.

Sources combined into one LMDB:
- Xenium rep1 (raw DAPI + Janesick supervised GT)
- Xenium rep2 (raw DAPI + same supervised GT)
- STHELAR breast s0/s1/s3/s6 (DAPI + label1 9-class CL)

Labels are mapped to coarse 4-class {Endothelial, Epithelial, Immune, Stromal}
via the existing `cell_ontology_mapping` infrastructure so all sources share
a common label space.

Output structure (matches existing LMDBs):
    /mnt/work/datasets/derived/<output>/
        patches.lmdb/                # uint16 patch + int64 label
        labels.npy                   # (N,) int64
        class_mapping.json           # {coarse_class: idx}
        metadata.json                # n_samples, patch_size, dtype, sources
        normalization_stats.json     # adaptive percentile per slide

Usage:
    uv run python scripts/breast_dapi_lmdb.py --patch-size 32 --output breast-multisource-dapi-p32
    uv run python scripts/breast_dapi_lmdb.py --patch-size 64
    uv run python scripts/breast_dapi_lmdb.py --patch-size 256
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
from loguru import logger

# Import existing data readers
from dapidl.data.sthelar import SthelarDataReader
from dapidl.data.xenium import XeniumDataReader

DERIVED = Path("/mnt/work/datasets/derived")
XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")

COARSE_CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_CLASSES)}

# STHELAR label1 -> coarse 4
STHELAR_LABEL1_TO_COARSE = {
    "Endothelial": "Endothelial",
    "Perivascular": "Stromal",
    "Epithelial": "Epithelial",
    "T": "Immune",
    "B": "Immune",
    "Monocyte/Macrophage": "Immune",
    "Mast": "Immune",
    "Fibroblast": "Stromal",
    "Adipocyte": "Stromal",
}

# Janesick supervised 17-class -> coarse 4 (from Cell_Barcode_Type_Matrices.xlsx)
JANESICK17_TO_COARSE = {
    "Stromal": "Stromal",
    "Invasive_Tumor": "Epithelial",
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "Endothelial": "Endothelial",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Mast_Cells": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Perivascular-Like": "Stromal",
    # Drop: Unlabeled, hybrids
    "Unlabeled": None,
    "Stromal_&_T_Cell_Hybrid": None,
    "T_Cell_&_Tumor_Hybrid": None,
}


def _xenium_fine_to_coarse(name: str) -> str | None:
    """Map Janesick supervised 17-class to coarse 4-class."""
    return JANESICK17_TO_COARSE.get(name)


def normalize_dapi(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Adaptive percentile normalization to [0, 1] float32."""
    p_low = float(np.percentile(image, 1))
    p_high = float(np.percentile(image, 99.5))
    if p_high <= p_low:
        p_high = p_low + 1
    normalized = np.clip(image.astype(np.float32), p_low, p_high)
    normalized = (normalized - p_low) / (p_high - p_low)
    return normalized, {"p_low": p_low, "p_high": p_high}


REP_TO_SHEET = {
    "rep1": "Xenium R1 Fig1-5 (supervised)",
    "rep2": "Xenium R2 Fig1-5 (supervised)",
}

# rep1 has the xlsx with both rep1 and rep2 sheets; rep2 directory does not.
GT_XLSX_PATH = (
    Path("/mnt/work/datasets/raw/xenium")
    / "xenium-breast-tumor-rep1"
    / "Cell_Barcode_Type_Matrices.xlsx"
)


def _load_xenium_supervised_gt(rep_name: str) -> dict[str, str]:
    """Load Janesick supervised barcode → 17-class label for rep1 or rep2."""
    if not GT_XLSX_PATH.exists():
        return {}
    import pandas as pd
    sheet = REP_TO_SHEET.get(rep_name)
    if sheet is None:
        return {}
    df = pd.read_excel(GT_XLSX_PATH, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    label_col = "Cluster" if "Cluster" in df.columns else "Annotation"
    return dict(zip(df["Barcode"].astype(str), df[label_col].astype(str)))


def extract_xenium_breast(
    rep_name: str,
    raw_dir: Path,
    patch_size: int,
    env: lmdb.Environment,
    patch_idx_start: int,
    all_labels: list[int],
    all_sources: list[str],
    all_cell_ids: list[str],
    max_cells: int = 0,
    nucleus_centered: bool = False,
) -> tuple[int, dict]:
    """Extract DAPI patches from Xenium breast rep1/rep2.

    nucleus_centered=True crops on the nucleus polygon centroid instead of the
    cell centroid (review B2 / spec 2026-05-25). STHELAR is already
    nucleus-centered, so only the Xenium path has this switch.
    """
    half = patch_size // 2
    logger.info(f"=== Xenium {rep_name}: {raw_dir} "
                f"({'nucleus' if nucleus_centered else 'cell'}-centered) ===")

    reader = XeniumDataReader(raw_dir / "outs")
    dapi = reader.image  # uint16 (H, W)
    h, w = dapi.shape
    dapi_norm, norm_stats = normalize_dapi(dapi)

    gt_lookup = _load_xenium_supervised_gt(rep_name)
    if not gt_lookup:
        logger.warning(f"  No GT for {rep_name}, skipping")
        return patch_idx_start, {}
    logger.info(f"  Loaded {len(gt_lookup)} barcode→annotation entries from {REP_TO_SHEET[rep_name]}")

    centroids = (reader.get_nucleus_centroids_pixels() if nucleus_centered
                 else reader.get_centroids_pixels())
    cell_ids = reader.get_cell_ids()

    # Build candidate list (passes label filter + bounds), then optionally subsample
    candidates: list[tuple[int, int]] = []  # (cell_idx, coarse_idx)
    for i, cid in enumerate(cell_ids):
        fine = gt_lookup.get(str(cid))
        if fine is None:
            continue
        coarse = _xenium_fine_to_coarse(fine)
        if coarse is None:
            continue
        candidates.append((i, COARSE_TO_IDX[coarse]))

    if max_cells > 0 and len(candidates) > max_cells:
        rng = np.random.default_rng(seed=42)
        # Stratified subsample by class
        from collections import defaultdict
        by_class: dict[int, list[int]] = defaultdict(list)
        for i, c in candidates:
            by_class[c].append(i)
        per_class = max(1, max_cells // len(by_class))
        picked: list[tuple[int, int]] = []
        for c, idx_list in by_class.items():
            n_take = min(per_class, len(idx_list))
            sampled = rng.choice(idx_list, size=n_take, replace=False)
            picked.extend([(int(i), c) for i in sampled])
        candidates = picked
        rng.shuffle(candidates)
        logger.info(f"  Subsampled to {len(candidates)} cells (cap={max_cells})")

    n_written = 0
    n_skipped_oob = 0
    with env.begin(write=True) as txn:
        idx = patch_idx_start
        for i, coarse_idx in candidates:
            cx, cy = int(round(centroids[i, 0])), int(round(centroids[i, 1]))
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                n_skipped_oob += 1
                continue
            patch = dapi_norm[y0:y1, x0:x1]
            if patch.shape != (patch_size, patch_size):
                n_skipped_oob += 1
                continue
            patch_uint16 = (patch * 65535).clip(0, 65535).astype(np.uint16)
            label_bytes = np.array([coarse_idx], dtype=np.int64).tobytes()
            value = label_bytes + patch_uint16.tobytes()
            txn.put(struct.pack(">Q", idx), value)
            all_labels.append(coarse_idx)
            all_sources.append(f"xenium_{rep_name}")
            all_cell_ids.append(str(cell_ids[i]))
            idx += 1
            n_written += 1

    logger.info(
        f"  {rep_name}: wrote {n_written} cells "
        f"(candidates {len(candidates)}, skipped {n_skipped_oob} OOB)"
    )
    return idx, {
        "source": f"xenium_{rep_name}",
        "n_written": n_written,
        "n_total": len(cell_ids),
        "centroid_source": "nucleus" if nucleus_centered else "cell",
        "norm": norm_stats,
    }


def extract_sthelar_breast(
    slide_zarr: Path,
    patch_size: int,
    env: lmdb.Environment,
    patch_idx_start: int,
    all_labels: list[int],
    all_sources: list[str],
    all_cell_ids: list[str],
    max_cells: int = 0,
) -> tuple[int, dict]:
    """Extract DAPI patches from one STHELAR breast slide (already nucleus-centered)."""
    half = patch_size // 2
    slide_name = slide_zarr.name.replace("sdata_", "").replace(".zarr", "")
    logger.info(f"=== STHELAR {slide_name}: {slide_zarr} ===")

    reader = SthelarDataReader(slide_zarr)
    nuc_df = reader.nucleus_df
    if "label1" not in nuc_df.columns:
        logger.warning(f"  No label1 column in {slide_zarr.name}, skipping")
        return patch_idx_start, {}

    nuc_df = nuc_df.with_columns(
        pl.col("label1").replace_strict(STHELAR_LABEL1_TO_COARSE,
                                        default=None).alias("coarse")
    ).filter(pl.col("coarse").is_not_null())

    if len(nuc_df) == 0:
        logger.warning(f"  No mapped cells in {slide_zarr.name}")
        return patch_idx_start, {}

    # Cap per slide (stratified by class for class balance)
    if max_cells > 0 and len(nuc_df) > max_cells:
        per_class = max(1, max_cells // 4)
        sampled = []
        for cls in nuc_df["coarse"].unique().to_list():
            sub = nuc_df.filter(pl.col("coarse") == cls)
            n_take = min(per_class, len(sub))
            sampled.append(sub.sample(n=n_take, seed=42))
        nuc_df = pl.concat(sampled).sample(fraction=1.0, seed=42)
        logger.info(f"  Capped to {len(nuc_df)} cells (max_cells={max_cells})")

    dapi = reader.image
    h, w = dapi.shape
    dapi_norm, norm_stats = normalize_dapi(dapi)

    centroids = reader.get_centroids_pixels()
    reader_cell_ids = reader.get_cell_ids()  # renamed: avoid shadowing the all_cell_ids registry param
    centroid_map = {cid: (centroids[i, 0], centroids[i, 1]) for i, cid in enumerate(reader_cell_ids)}

    n_written = 0
    n_skipped_oob = 0
    with env.begin(write=True) as txn:
        idx = patch_idx_start
        for row in nuc_df.iter_rows(named=True):
            cid = row["cell_id"]
            coarse = row["coarse"]
            if cid not in centroid_map:
                continue
            cx, cy = centroid_map[cid]
            cx, cy = int(round(cx)), int(round(cy))
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                n_skipped_oob += 1
                continue
            patch = dapi_norm[y0:y1, x0:x1]
            if patch.shape != (patch_size, patch_size):
                n_skipped_oob += 1
                continue
            patch_uint16 = (patch * 65535).clip(0, 65535).astype(np.uint16)
            label_bytes = np.array([COARSE_TO_IDX[coarse]], dtype=np.int64).tobytes()
            value = label_bytes + patch_uint16.tobytes()
            txn.put(struct.pack(">Q", idx), value)
            all_labels.append(COARSE_TO_IDX[coarse])
            all_sources.append(f"sthelar_{slide_name}")
            all_cell_ids.append(str(cid))
            idx += 1
            n_written += 1

    logger.info(
        f"  {slide_name}: wrote {n_written} cells (skipped {n_skipped_oob} OOB)"
    )
    return idx, {
        "source": f"sthelar_{slide_name}",
        "n_written": n_written,
        "n_total": len(nuc_df),
        "centroid_source": "nucleus",
        "norm": norm_stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", type=int, required=True,
                    choices=[32, 64, 128, 256])
    ap.add_argument("--output", default=None,
                    help="Output dir name under /mnt/work/datasets/derived/")
    ap.add_argument("--nucleus-centered", action="store_true",
                    help="Crop Xenium on the nucleus polygon centroid (review B2 / spec "
                         "2026-05-25). STHELAR is already nucleus-centered. Persists a "
                         "patch_registry.parquet + cell_ids.npy so cross-LMDB A/B can pair by cell_id.")
    ap.add_argument("--no-xenium", action="store_true", help="Skip rep1+rep2")
    ap.add_argument("--no-sthelar", action="store_true", help="Skip STHELAR breast slides")
    ap.add_argument("--map-size-gb", type=int, default=80,
                    help="LMDB max map size in GB")
    ap.add_argument("--max-cells-per-source", type=int, default=0,
                    help="If > 0, cap each Xenium rep / STHELAR slide to this many "
                         "cells (stratified by class). Use to keep disk usage manageable "
                         "and balance source sizes for fair cross-source comparison.")
    args = ap.parse_args()

    default_name = f"breast-multisource-dapi-p{args.patch_size}"
    if args.nucleus_centered:
        default_name += "-nuc"
    output = args.output or default_name
    out_dir = DERIVED / output
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "patches.lmdb"

    env = lmdb.open(str(lmdb_path), map_size=args.map_size_gb * 1024**3)
    all_labels: list[int] = []
    all_sources: list[str] = []
    all_cell_ids: list[str] = []
    slide_stats: list[dict] = []
    idx = 0

    if not args.no_xenium:
        for rep_name in ["rep1", "rep2"]:
            raw_dir = XENIUM_BASE / f"xenium-breast-tumor-{rep_name}"
            if not raw_dir.exists():
                logger.warning(f"Skipping xenium {rep_name}: {raw_dir} not found")
                continue
            idx, st = extract_xenium_breast(
                rep_name, raw_dir, args.patch_size, env, idx,
                all_labels, all_sources, all_cell_ids,
                max_cells=args.max_cells_per_source,
                nucleus_centered=args.nucleus_centered,
            )
            if st:
                slide_stats.append(st)

    if not args.no_sthelar:
        breast_zarrs = sorted(STHELAR_BASE.glob("sdata_breast_s*.zarr"))
        breast_zarrs = [z for z in breast_zarrs if z.is_dir()]
        for slide_zarr in breast_zarrs:
            idx, st = extract_sthelar_breast(
                slide_zarr, args.patch_size, env, idx,
                all_labels, all_sources, all_cell_ids,
                max_cells=args.max_cells_per_source,
            )
            if st:
                slide_stats.append(st)

    env.close()

    np.save(out_dir / "labels.npy", np.array(all_labels, dtype=np.int64))
    (out_dir / "class_mapping.json").write_text(
        json.dumps(COARSE_TO_IDX, indent=2)
    )

    # cell_id provenance (review B2): persist a registry so cross-LMDB A/B
    # (M_cell vs M_nuc_full) can pair by (source, cell_id) and splits can be
    # derived from hash(cell_id) instead of a fragile positional index.
    assert len(all_cell_ids) == len(all_labels) == idx, \
        f"registry desync: cell_ids={len(all_cell_ids)} labels={len(all_labels)} idx={idx}"
    np.save(out_dir / "cell_ids.npy", np.array(all_cell_ids, dtype=object))
    pl.DataFrame({
        "row_idx": np.arange(idx, dtype=np.int64),
        "source": all_sources,
        "cell_id": all_cell_ids,
        "coarse_idx": np.array(all_labels, dtype=np.int64),
    }).write_parquet(out_dir / "patch_registry.parquet")

    from collections import Counter
    class_counts = dict(Counter(all_labels))
    src_counts = dict(Counter(all_sources))
    metadata = {
        "n_samples": idx,
        "n_classes": 4,
        "patch_size": args.patch_size,
        "patch_shape": [args.patch_size, args.patch_size],
        "dtype": "uint16",
        "format": "lmdb",
        "normalization": "adaptive_per_slide",
        "platform": "xenium+sthelar",
        "scope": "breast_tumor + breast normal",
        "class_names": COARSE_CLASSES,
        "class_counts": {COARSE_CLASSES[k]: v for k, v in class_counts.items()},
        "source_counts": src_counts,
        "n_sources": len(slide_stats),
        "nucleus_centered": bool(args.nucleus_centered),
        "centroid_source": {s["source"]: s.get("centroid_source", "?") for s in slide_stats},
        "has_registry": True,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (lmdb_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "slide_stats.json").write_text(json.dumps(slide_stats, indent=2))

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {idx:,} patches at p{args.patch_size}")
    logger.info(f"Sources: {len(slide_stats)}")
    logger.info(f"Class distribution:")
    for k, v in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / idx if idx else 0
        logger.info(f"  {COARSE_CLASSES[k]:<14s}: {v:>9,} ({pct:5.1f}%)")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
