#!/usr/bin/env python3
"""Create a single multi-tissue LMDB from all 31 STHELAR slides.

Processes each slide sequentially to avoid memory/disk issues:
1. Load DAPI + labels via SthelarDataReader
2. Sample up to MAX_PER_SLIDE cells (sqrt sampling for balance)
3. Extract 128x128 patches, normalize adaptively
4. Write to a single combined LMDB

Usage:
    uv run python scripts/sthelar_multi_tissue_lmdb.py
"""

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

# Configuration
STHELAR_DIR = Path("/mnt/work/datasets/STHELAR/sdata_slides")
OUTPUT_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128")
PATCH_SIZE = 128
HALF_PATCH = PATCH_SIZE // 2
MAX_PER_SLIDE = 50_000  # Cap per slide for disk/balance
MAP_SIZE = 120 * 1024**3  # 120 GB max LMDB size

# STHELAR label -> CL name mapping (from annotator_mappings.py GT_STHELAR)
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


def normalize_dapi(image: np.ndarray) -> np.ndarray:
    """Adaptive percentile normalization to [0, 1] float32."""
    p_low = np.percentile(image, 1)
    p_high = np.percentile(image, 99.5)
    if p_high <= p_low:
        p_high = p_low + 1
    normalized = np.clip(image.astype(np.float32), p_low, p_high)
    normalized = (normalized - p_low) / (p_high - p_low)
    return normalized


def process_slide(
    zarr_path: Path,
    env: lmdb.Environment,
    patch_idx: int,
    class_mapping: dict[str, int],
    all_labels: list[int],
    all_tissues: list[str],
    slide_stats: dict,
) -> int:
    """Process one STHELAR slide and write patches to LMDB.

    Returns: updated patch_idx
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
        return patch_idx

    # Get labels
    nuc_df = reader.nucleus_df
    if "label1" not in nuc_df.columns:
        log.warning(f"  No label1 column in {slide_name}, skipping")
        return patch_idx

    # Map labels to CL names
    nuc_df = nuc_df.with_columns(
        pl.col("label1").replace_strict(STHELAR_TO_CL, default=None).alias("cl_name")
    )
    # Drop unmapped (less10, etc.)
    nuc_df = nuc_df.filter(pl.col("cl_name").is_not_null())

    n_cells = len(nuc_df)
    if n_cells == 0:
        log.warning(f"  No mapped cells in {slide_name}")
        return patch_idx

    # Sample if too many cells
    if n_cells > MAX_PER_SLIDE:
        nuc_df = nuc_df.sample(n=MAX_PER_SLIDE, seed=42)
        log.info(f"  Sampled {MAX_PER_SLIDE} from {n_cells} cells")
    else:
        log.info(f"  Using all {n_cells} cells")

    # Add new classes to mapping
    for cl_name in nuc_df["cl_name"].unique().to_list():
        if cl_name not in class_mapping:
            class_mapping[cl_name] = len(class_mapping)

    # Load DAPI image
    dapi = reader.image  # uint16, (H, W)
    h, w = dapi.shape
    log.info(f"  DAPI: {h}x{w}, dtype={dapi.dtype}")

    # Normalize
    dapi_norm = normalize_dapi(dapi)

    # Get centroids in pixels
    centroids = reader.get_centroids_pixels()  # (N, 2) = (x, y)

    # Build cell_id -> centroid lookup
    all_cell_ids = reader.get_cell_ids()
    centroid_map = {cid: (centroids[i, 0], centroids[i, 1]) for i, cid in enumerate(all_cell_ids)}

    # Extract patches
    n_written = 0
    with env.begin(write=True) as txn:
        for row in nuc_df.iter_rows(named=True):
            cell_id = row["cell_id"]
            cl_name = row["cl_name"]
            label_idx = class_mapping[cl_name]

            if cell_id not in centroid_map:
                continue

            cx, cy = centroid_map[cell_id]
            cx, cy = int(round(cx)), int(round(cy))

            # Boundary check
            y0, y1 = cy - HALF_PATCH, cy + HALF_PATCH
            x0, x1 = cx - HALF_PATCH, cx + HALF_PATCH
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                continue

            patch = dapi_norm[y0:y1, x0:x1]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                continue

            # Convert to uint16 for storage
            patch_uint16 = (patch * 65535).clip(0, 65535).astype(np.uint16)

            # Pack: label (native int64) + patch bytes
            key = struct.pack(">Q", patch_idx)
            label_bytes = np.array([label_idx], dtype=np.int64).tobytes()
            value = label_bytes + patch_uint16.tobytes()
            txn.put(key, value)

            all_labels.append(label_idx)
            all_tissues.append(tissue)
            patch_idx += 1
            n_written += 1

    elapsed = time.time() - t0
    slide_stats[slide_name] = {"tissue": tissue, "cells_total": n_cells, "patches_written": n_written, "time_s": round(elapsed, 1)}
    log.info(f"  Written {n_written} patches in {elapsed:.1f}s")

    # Free memory
    del dapi, dapi_norm, reader, nuc_df
    gc.collect()

    return patch_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lmdb_path = OUTPUT_DIR / "patches.lmdb"

    # Get all slides
    slides = sorted(STHELAR_DIR.glob("*.zarr"))
    # Filter out zip files that happen to match
    slides = [s for s in slides if s.is_dir()]
    log.info(f"Found {len(slides)} STHELAR slides")

    # Create LMDB
    env = lmdb.open(str(lmdb_path), map_size=MAP_SIZE)

    class_mapping: dict[str, int] = {}
    all_labels: list[int] = []
    all_tissues: list[str] = []
    slide_stats: dict = {}
    patch_idx = 0

    for i, slide_path in enumerate(slides):
        log.info(f"\n{'='*60}")
        log.info(f"Slide {i+1}/{len(slides)}: {slide_path.name}")
        log.info(f"{'='*60}")
        patch_idx = process_slide(
            slide_path, env, patch_idx, class_mapping, all_labels, all_tissues, slide_stats,
        )
        log.info(f"  Running total: {patch_idx:,} patches, {len(class_mapping)} classes")

    env.close()

    # Save metadata files
    labels_array = np.array(all_labels, dtype=np.int64)
    np.save(OUTPUT_DIR / "labels.npy", labels_array)

    with open(OUTPUT_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    # Count per-class
    from collections import Counter
    class_counts = dict(Counter(all_labels))
    idx_to_name = {v: k for k, v in class_mapping.items()}
    named_counts = {idx_to_name.get(k, str(k)): v for k, v in class_counts.items()}

    # Tissue distribution
    tissue_counts = dict(Counter(all_tissues))

    metadata = {
        "n_samples": patch_idx,
        "n_classes": len(class_mapping),
        "n_tissues": len(tissue_counts),
        "n_slides": len(slides),
        "patch_size": PATCH_SIZE,
        "patch_shape": [PATCH_SIZE, PATCH_SIZE],
        "dtype": "uint16",
        "format": "lmdb",
        "normalization": "adaptive",
        "platform": "sthelar",
        "max_per_slide": MAX_PER_SLIDE,
        "class_counts": named_counts,
        "tissue_counts": tissue_counts,
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    # Also in patches.lmdb for DALI
    with open(lmdb_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save per-slide stats
    with open(OUTPUT_DIR / "slide_stats.json", "w") as f:
        json.dump(slide_stats, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE: {patch_idx:,} patches from {len(slides)} slides")
    log.info(f"Classes: {len(class_mapping)}")
    log.info(f"Tissues: {len(tissue_counts)}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"Class distribution:")
    for name, count in sorted(named_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / patch_idx
        log.info(f"  {name:<20s}: {count:>8,} ({pct:5.1f}%)")
    log.info(f"Tissue distribution:")
    for tissue, count in sorted(tissue_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / patch_idx
        log.info(f"  {tissue:<20s}: {count:>8,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
