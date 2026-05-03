#!/usr/bin/env python3
"""Build a skin-multisource DAPI LMDB at 128px patches.

Sources: STHELAR skin slides s1, s2, s3, s4 (zarr label2 used directly —
no parquet metadata exists for non-breast tissues).

Two outputs simultaneously:
- labels.npy            int64 COARSE-5 (Endo/Epi/Imm/Stromal/Neural); -1 = drop
- labels_medium.npy     int64 MEDIUM-12; -1 = drop
- raw_labels.npy        object  source label2 string
- sources.npy           object  per-cell source tag (sthelar_skin_s1, etc.)
- patches.lmdb/         uint16 patch payload (DAPI, normalized & scaled)

Usage:
    uv run python scripts/skin_dapi_lmdb.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lmdb
import numpy as np
import polars as pl
from loguru import logger

from dapidl.data.sthelar import SthelarDataReader

DERIVED = Path("/mnt/work/datasets/derived")
STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")
OUT_DIR = DERIVED / "skin-4source-dapi-p128"
PATCH_SIZE = 128
HALF = PATCH_SIZE // 2
MAP_SIZE_BYTES = 30 * 1024 ** 3  # 30 GB

# COARSE 5 (with Neural for melanocyte)
COARSE_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal", "Neural"]
COARSE_TO_IDX = {c: i for i, c in enumerate(COARSE_NAMES)}

# MEDIUM 12 (canonical, matches breast)
MEDIUM_NAMES = [
    "Epithelial_Luminal", "Epithelial_Basal",
    "T_Cell", "B_Cell", "Macrophage", "Dendritic_Cell", "Mast_Cell",
    "Fibroblast", "Pericyte", "Adipocyte",
    "Endothelial", "Neural",
]
MEDIUM_TO_IDX = {c: i for i, c in enumerate(MEDIUM_NAMES)}

# Per Codex two-head design: skin label2 -> COARSE-5
SKIN_LABEL2_TO_COARSE = {
    "Keratinocyte":              "Epithelial",
    "Basal_keatinocyte":         "Epithelial",  # typo in source
    "Basal_keratinocyte":        "Epithelial",
    "Proliferating_basal_keratinocyte": "Epithelial",
    "Endothelial":               "Endothelial",
    "Fibroblast":                "Stromal",
    "CAF":                       "Stromal",
    "Macrophage/DC":             "Immune",
    "Monocyte/Macrophage/DC":    "Immune",
    "DC":                        "Immune",
    "Mast":                      "Immune",
    "T":                         "Immune",
    "T/NK":                      "Immune",
    "T_CD8":                     "Immune",
    "Treg":                      "Immune",
    "Proliferating_T_NK":        "Immune",
    "B":                         "Immune",
    "Plasma":                    "Immune",
    "Smooth_muscle":             "Stromal",
    "Smooth_muscle_cell_and_pericyte": None,  # ambiguous → drop for medium AND coarse
    "Smooth_muscle_and_pericyte": None,
    "Melanocyte":                "Neural",
    "High_proliferating_melanocyte": "Neural",
    # Cancer states — drop
    "Melanoma":                  None,
    "High_proliferating_melanoma": None,
    "Cancer_stem_like_cells":    None,
    "less10":                    None,
    "Less10":                    None,
}

# Skin label2 -> MEDIUM-12 (cleaner mapping per codex)
SKIN_LABEL2_TO_MEDIUM = {
    "Keratinocyte":              "Epithelial_Luminal",  # spinous/superficial
    "Basal_keatinocyte":         "Epithelial_Basal",
    "Basal_keratinocyte":        "Epithelial_Basal",
    "Proliferating_basal_keratinocyte": "Epithelial_Basal",
    "Endothelial":               "Endothelial",
    "Fibroblast":                "Fibroblast",
    "CAF":                       "Fibroblast",
    "Macrophage/DC":             "Macrophage",  # pooled label, codex would prefer drop
    "Monocyte/Macrophage/DC":    "Macrophage",
    "DC":                        "Dendritic_Cell",
    "Mast":                      "Mast_Cell",
    "T":                         "T_Cell",
    "T/NK":                      "T_Cell",
    "T_CD8":                     "T_Cell",
    "Treg":                      "T_Cell",
    "Proliferating_T_NK":        "T_Cell",
    "B":                         "B_Cell",
    "Plasma":                    "B_Cell",  # roll up to B parent
    "Smooth_muscle":             "Pericyte",
    "Smooth_muscle_cell_and_pericyte": None,
    "Smooth_muscle_and_pericyte": None,
    "Melanocyte":                "Neural",
    "High_proliferating_melanocyte": "Neural",
    "Melanoma":                  None,
    "High_proliferating_melanoma": None,
    "Cancer_stem_like_cells":    None,
    "less10":                    None,
    "Less10":                    None,
}

SKIN_SLIDES = ["skin_s1", "skin_s2", "skin_s3", "skin_s4"]


def normalize_dapi(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Adaptive percentile normalization to [0, 1] float32."""
    p_low = float(np.percentile(image, 1))
    p_high = float(np.percentile(image, 99.5))
    if p_high <= p_low:
        p_high = p_low + 1
    norm = np.clip(image.astype(np.float32), p_low, p_high)
    norm = (norm - p_low) / (p_high - p_low)
    return norm, {"p_low": p_low, "p_high": p_high}


def extract_skin_slide(
    slide_name: str,
    env: lmdb.Environment,
    patch_idx_start: int,
    raw_labels: list[str],
    coarse_labels: list[int],
    medium_labels: list[int],
    sources: list[str],
) -> tuple[int, dict]:
    """Extract DAPI patches for one skin slide."""
    slide_zarr = STHELAR_BASE / f"sdata_{slide_name}.zarr"
    logger.info(f"=== {slide_name}: {slide_zarr} ===")
    reader = SthelarDataReader(slide_zarr)
    nuc_df = reader.nucleus_df
    if "label2" not in nuc_df.columns:
        logger.warning(f"  {slide_name}: no label2 column, skipping")
        return patch_idx_start, {}

    dapi = reader.image
    h, w = dapi.shape
    dapi_norm, norm_stats = normalize_dapi(dapi)

    centroids = reader.get_centroids_pixels()
    all_cell_ids = reader.get_cell_ids()
    centroid_map = {cid: (centroids[i, 0], centroids[i, 1])
                    for i, cid in enumerate(all_cell_ids)}

    n_written = 0
    n_oob = 0
    n_no_centroid = 0
    n_unmapped = 0
    with env.begin(write=True) as txn:
        idx = patch_idx_start
        for row in nuc_df.iter_rows(named=True):
            l2 = row["label2"]
            cid = row["cell_id"]
            if cid not in centroid_map:
                n_no_centroid += 1
                continue
            cx, cy = centroid_map[cid]
            cx, cy = int(round(cx)), int(round(cy))
            y0, y1 = cy - HALF, cy + HALF
            x0, x1 = cx - HALF, cx + HALF
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                n_oob += 1
                continue
            patch = dapi_norm[y0:y1, x0:x1]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                n_oob += 1
                continue
            patch_uint16 = (patch * 65535).clip(0, 65535).astype(np.uint16)
            coarse_name = SKIN_LABEL2_TO_COARSE.get(l2)
            medium_name = SKIN_LABEL2_TO_MEDIUM.get(l2)
            c_idx = COARSE_TO_IDX[coarse_name] if coarse_name else -1
            m_idx = MEDIUM_TO_IDX[medium_name] if medium_name else -1
            if coarse_name is None and l2 not in SKIN_LABEL2_TO_COARSE:
                n_unmapped += 1
            label_bytes = np.array([c_idx], dtype=np.int64).tobytes()
            value = label_bytes + patch_uint16.tobytes()
            txn.put(struct.pack(">Q", idx), value)
            raw_labels.append(str(l2))
            coarse_labels.append(c_idx)
            medium_labels.append(m_idx)
            sources.append(f"sthelar_{slide_name}")
            idx += 1
            n_written += 1

    logger.info(
        f"  {slide_name}: wrote {n_written} cells "
        f"(oob {n_oob}, no-centroid {n_no_centroid}, unmapped-l2 {n_unmapped})"
    )
    return idx, {
        "source": f"sthelar_{slide_name}",
        "n_written": n_written,
        "norm": norm_stats,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {OUT_DIR}")

    env = lmdb.open(str(OUT_DIR / "patches.lmdb"),
                    map_size=MAP_SIZE_BYTES, subdir=True,
                    max_readers=64, meminit=False, sync=False)

    raw_labels: list[str] = []
    coarse_labels: list[int] = []
    medium_labels: list[int] = []
    sources: list[str] = []
    stats = []

    idx = 0
    for slide in SKIN_SLIDES:
        idx, s = extract_skin_slide(
            slide, env, idx, raw_labels, coarse_labels, medium_labels, sources)
        if s:
            stats.append(s)
    env.sync()
    env.close()

    raw_arr = np.array(raw_labels, dtype=object)
    coarse_arr = np.array(coarse_labels, dtype=np.int64)
    medium_arr = np.array(medium_labels, dtype=np.int64)
    sources_arr = np.array(sources, dtype=object)

    np.save(OUT_DIR / "raw_labels.npy", raw_arr)
    np.save(OUT_DIR / "labels.npy", coarse_arr)
    np.save(OUT_DIR / "labels_medium.npy", medium_arr)
    np.save(OUT_DIR / "sources.npy", sources_arr)

    (OUT_DIR / "class_mapping.json").write_text(
        json.dumps(COARSE_TO_IDX, indent=2))
    (OUT_DIR / "class_mapping_medium.json").write_text(
        json.dumps(MEDIUM_TO_IDX, indent=2))
    (OUT_DIR / "slide_stats.json").write_text(json.dumps(stats, indent=2))
    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "n_samples": int(len(coarse_arr)),
        "patch_size": PATCH_SIZE,
        "dtype": "uint16",
        "sources": SKIN_SLIDES,
        "tier_coarse_classes": COARSE_NAMES,
        "tier_medium_classes": MEDIUM_NAMES,
    }, indent=2))

    logger.info(f"=== TOTAL: {len(coarse_arr):,} cells across {len(stats)} slides ===")
    n_drop_c = int((coarse_arr == -1).sum())
    n_drop_m = int((medium_arr == -1).sum())
    logger.info(f"COARSE drop: {n_drop_c:,} ({100*n_drop_c/len(coarse_arr):.1f}%)")
    for n, i in COARSE_TO_IDX.items():
        c = int((coarse_arr == i).sum())
        logger.info(f"  {n:14s} {c:>10,d}")
    logger.info(f"MEDIUM drop: {n_drop_m:,} ({100*n_drop_m/len(medium_arr):.1f}%)")
    for n, i in MEDIUM_TO_IDX.items():
        c = int((medium_arr == i).sum())
        logger.info(f"  {n:20s} {c:>10,d}")


if __name__ == "__main__":
    main()
