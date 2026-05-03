"""Derive coarse + medium labels from STHELAR cells_label2 (and Janesick GT).

Replaces derive_raw_labels.py + derive_tier_labels.py for the breast-6source
LMDB. Walks the SAME cells in the SAME order as breast_dapi_lmdb.py so the
output arrays align with the existing patches.lmdb (no rebuild required).

For STHELAR slides, joins the per-slide parquet
(/mnt/work/datasets/STHELAR/summary_all_labels_per_slide/<slide>_cell_metadata.parquet)
by cell_id to fetch `cells_label2` (11-class subcell) and confidence.

Outputs (overwrites existing):
    /mnt/work/datasets/derived/breast-6source-dapi-p128/
        raw_labels.npy            object array of source-string labels
        labels.npy                int64 coarse 4 (Endo/Epi/Imm/Stromal); -1 = drop
        class_mapping.json        coarse name -> idx
        labels_medium.npy         int64 medium 12; -1 = drop
        class_mapping_medium.json medium name -> idx
        confidence.npy            float32 confidence (1.0 for Janesick, parquet val for STHELAR)

The `-1` cells (less10 / Unknown / Plasma without map / etc.) are dropped at
training time in the cross-source training script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from loguru import logger

from breast_dapi_lmdb import (
    STHELAR_LABEL1_TO_COARSE,
    JANESICK17_TO_COARSE,
    XENIUM_BASE,
    STHELAR_BASE,
    _xenium_fine_to_coarse,
    _load_xenium_supervised_gt,
)
from dapidl.data.sthelar import SthelarDataReader
from dapidl.data.xenium import XeniumDataReader

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
PARQUET_DIR = Path("/mnt/work/datasets/STHELAR/summary_all_labels_per_slide")
PATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Canonical tier vocabularies (must match training_tiers.py)
# ---------------------------------------------------------------------------
COARSE_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal"]  # breast = 4
MEDIUM_NAMES = [
    "Epithelial_Luminal", "Epithelial_Basal",
    "T_Cell", "B_Cell", "Macrophage", "Dendritic_Cell", "Mast_Cell",
    "Fibroblast", "Pericyte", "Adipocyte",
    "Endothelial",
    "Neural",
]
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_NAMES)}
MEDIUM_TO_IDX = {c: i for i, c in enumerate(MEDIUM_NAMES)}

# ---------------------------------------------------------------------------
# STHELAR cells_label2 -> canonical tiers (breast-specific values)
# ---------------------------------------------------------------------------
STHELAR_LABEL2_TO_COARSE = {
    "Mammary_luminal_cell": "Epithelial",
    "Mammary_basal_cell_(=myoepithelial)": "Epithelial",
    "CAF": "Stromal",
    "Monocyte/Macrophage": "Immune",
    "T": "Immune",
    "B": "Immune",
    "Plasma": "Immune",
    "Mast": "Immune",
    "Adipocyte": "Stromal",
    "Endothelial_Pericyte_Smooth_muscle": "Endothelial",
    "less10": None,
    None: None,
}

STHELAR_LABEL2_TO_MEDIUM = {
    "Mammary_luminal_cell": "Epithelial_Luminal",
    "Mammary_basal_cell_(=myoepithelial)": "Epithelial_Basal",
    "CAF": "Fibroblast",
    "Monocyte/Macrophage": "Macrophage",
    "T": "T_Cell",
    "B": "B_Cell",
    "Plasma": "B_Cell",
    "Mast": "Mast_Cell",
    "Adipocyte": "Adipocyte",
    "Endothelial_Pericyte_Smooth_muscle": "Endothelial",
    "less10": None,
    None: None,
}

# ---------------------------------------------------------------------------
# Janesick 17-class -> medium 12 (coarse already in breast_dapi_lmdb.py)
# ---------------------------------------------------------------------------
JANESICK17_TO_MEDIUM = {
    "Stromal":              "Fibroblast",
    "Invasive_Tumor":       "Epithelial_Luminal",
    "DCIS_1":               "Epithelial_Luminal",
    "DCIS_2":               "Epithelial_Luminal",
    "Prolif_Invasive_Tumor": "Epithelial_Luminal",
    "Myoepi_ACTA2+":        "Epithelial_Basal",
    "Myoepi_KRT15+":        "Epithelial_Basal",
    "Endothelial":          "Endothelial",
    "Macrophages_1":        "Macrophage",
    "Macrophages_2":        "Macrophage",
    "B_Cells":              "B_Cell",
    "CD4+_T_Cells":         "T_Cell",
    "CD8+_T_Cells":         "T_Cell",
    "Mast_Cells":           "Mast_Cell",
    "IRF7+_DCs":            "Dendritic_Cell",
    "LAMP3+_DCs":           "Dendritic_Cell",
    "Perivascular-Like":    "Pericyte",
    "Unlabeled":            None,
    "Stromal_&_T_Cell_Hybrid": None,
    "T_Cell_&_Tumor_Hybrid":   None,
}


def _walk_xenium(rep_name: str, n_expected: int) -> tuple[list[str], list[int], list[int], list[float]]:
    """Walk Xenium rep cells in same order as breast_dapi_lmdb.py."""
    raw_dir = XENIUM_BASE / f"xenium-breast-tumor-{rep_name}"
    reader = XeniumDataReader(raw_dir / "outs")
    dapi = reader.image
    h, w = dapi.shape
    half = PATCH_SIZE // 2
    gt = _load_xenium_supervised_gt(rep_name)
    centroids = reader.get_centroids_pixels()
    cell_ids = reader.get_cell_ids()

    candidates: list[tuple[int, str]] = []
    for i, cid in enumerate(cell_ids):
        fine = gt.get(str(cid))
        if fine is None:
            continue
        coarse = _xenium_fine_to_coarse(fine)
        if coarse is None:
            continue
        candidates.append((i, fine))

    raw, c_idx, m_idx, conf = [], [], [], []
    for i, fine in candidates:
        cx, cy = int(round(centroids[i, 0])), int(round(centroids[i, 1]))
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        coarse_name = JANESICK17_TO_COARSE.get(fine)
        medium_name = JANESICK17_TO_MEDIUM.get(fine)
        raw.append(fine)
        c_idx.append(COARSE_TO_IDX.get(coarse_name, -1) if coarse_name else -1)
        m_idx.append(MEDIUM_TO_IDX.get(medium_name, -1) if medium_name else -1)
        conf.append(1.0)

    if len(raw) != n_expected:
        raise SystemExit(
            f"alignment mismatch: xenium_{rep_name} produced {len(raw)} != {n_expected}")
    logger.info(f"  xenium_{rep_name}: {len(raw):,} cells (kept), all conf=1.0")
    return raw, c_idx, m_idx, conf


def _walk_sthelar(slide_zarr: Path, n_expected: int) -> tuple[list[str], list[int], list[int], list[float]]:
    """Walk STHELAR slide cells in same order as breast_dapi_lmdb.py.

    Joins the per-slide parquet to fetch cells_label2 + confidence.
    """
    slide_name = slide_zarr.name.replace("sdata_", "").replace(".zarr", "")
    parquet_path = PARQUET_DIR / f"{slide_name}_cell_metadata.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"missing parquet: {parquet_path}")

    pq = pl.read_parquet(parquet_path).select(
        ["cell_id", "cells_label2", "cells_final_label_confidence"]
    )
    label2_lookup = dict(zip(pq["cell_id"].to_list(), pq["cells_label2"].to_list()))
    conf_lookup = dict(zip(pq["cell_id"].to_list(),
                           [float(c) if c is not None else 0.0
                            for c in pq["cells_final_label_confidence"].to_list()]))

    reader = SthelarDataReader(slide_zarr)
    nuc_df = reader.nucleus_df
    if "label1" not in nuc_df.columns:
        raise SystemExit(f"no label1 in {slide_zarr.name}")

    # Same coarse filter as LMDB build
    nuc_df = nuc_df.with_columns(
        pl.col("label1").replace_strict(STHELAR_LABEL1_TO_COARSE,
                                        default=None).alias("coarse_filter")
    ).filter(pl.col("coarse_filter").is_not_null())
    if len(nuc_df) == 0:
        raise SystemExit(f"no cells survive label1 filter in {slide_name}")

    dapi = reader.image
    h, w = dapi.shape
    half = PATCH_SIZE // 2

    centroids = reader.get_centroids_pixels()
    all_cell_ids = reader.get_cell_ids()
    centroid_map = {cid: (centroids[i, 0], centroids[i, 1])
                    for i, cid in enumerate(all_cell_ids)}

    raw, c_idx, m_idx, conf = [], [], [], []
    n_unmatched = 0
    for row in nuc_df.iter_rows(named=True):
        cid = row["cell_id"]
        if cid not in centroid_map:
            continue
        cx, cy = centroid_map[cid]
        cx, cy = int(round(cx)), int(round(cy))
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        l2 = label2_lookup.get(cid)
        if l2 not in label2_lookup.values() and l2 is None:
            n_unmatched += 1
        coarse_name = STHELAR_LABEL2_TO_COARSE.get(l2)
        medium_name = STHELAR_LABEL2_TO_MEDIUM.get(l2)
        raw.append(str(l2) if l2 is not None else "Unknown")
        c_idx.append(COARSE_TO_IDX.get(coarse_name, -1) if coarse_name else -1)
        m_idx.append(MEDIUM_TO_IDX.get(medium_name, -1) if medium_name else -1)
        conf.append(conf_lookup.get(cid, 0.0))

    if len(raw) != n_expected:
        raise SystemExit(
            f"alignment mismatch: {slide_name} produced {len(raw)} != {n_expected}")
    logger.info(
        f"  sthelar_{slide_name}: {len(raw):,} cells "
        f"(parquet missing for {n_unmatched})")
    return raw, c_idx, m_idx, conf


def main() -> None:
    stats = json.loads((LMDB_DIR / "slide_stats.json").read_text())
    logger.info("Expected per-source counts (from LMDB slide_stats):")
    for s in stats:
        logger.info(f"  {s['source']:30s} {s['n_written']:>10,d}")

    raw_all: list[str] = []
    coarse_all: list[int] = []
    medium_all: list[int] = []
    conf_all: list[float] = []

    for s in stats:
        src = s["source"]
        n_expected = s["n_written"]
        if src.startswith("xenium_rep"):
            rep = src.split("xenium_")[1]
            r, c, m, cf = _walk_xenium(rep, n_expected)
        elif src.startswith("sthelar_breast_"):
            slide = src.split("sthelar_breast_")[1]
            slide_zarr = STHELAR_BASE / f"sdata_breast_{slide}.zarr"
            r, c, m, cf = _walk_sthelar(slide_zarr, n_expected)
        else:
            raise SystemExit(f"unknown source: {src}")
        raw_all.extend(r); coarse_all.extend(c)
        medium_all.extend(m); conf_all.extend(cf)

    raw_arr = np.array(raw_all, dtype=object)
    coarse_arr = np.array(coarse_all, dtype=np.int64)
    medium_arr = np.array(medium_all, dtype=np.int64)
    conf_arr = np.array(conf_all, dtype=np.float32)

    np.save(LMDB_DIR / "raw_labels.npy", raw_arr)
    np.save(LMDB_DIR / "labels.npy", coarse_arr)
    np.save(LMDB_DIR / "labels_medium.npy", medium_arr)
    np.save(LMDB_DIR / "confidence.npy", conf_arr)

    (LMDB_DIR / "class_mapping.json").write_text(
        json.dumps(COARSE_TO_IDX, indent=2))
    (LMDB_DIR / "class_mapping_medium.json").write_text(
        json.dumps(MEDIUM_TO_IDX, indent=2))

    logger.info(f"wrote {len(raw_arr):,} cells")
    logger.info("=== COARSE distribution ===")
    n_drop_c = int((coarse_arr == -1).sum())
    logger.info(f"  -1 (drop)              {n_drop_c:>10,d} ({100*n_drop_c/len(coarse_arr):.1f}%)")
    for name, i in COARSE_TO_IDX.items():
        n = int((coarse_arr == i).sum())
        logger.info(f"  {name:22s} {n:>10,d} ({100*n/len(coarse_arr):.1f}%)")
    logger.info("=== MEDIUM distribution ===")
    n_drop_m = int((medium_arr == -1).sum())
    logger.info(f"  -1 (drop)              {n_drop_m:>10,d} ({100*n_drop_m/len(medium_arr):.1f}%)")
    for name, i in MEDIUM_TO_IDX.items():
        n = int((medium_arr == i).sum())
        logger.info(f"  {name:22s} {n:>10,d} ({100*n/len(medium_arr):.1f}%)")
    logger.info(f"=== CONFIDENCE: median={np.median(conf_arr):.3f}, "
                f"q05={np.quantile(conf_arr, 0.05):.3f} ===")


if __name__ == "__main__":
    main()
