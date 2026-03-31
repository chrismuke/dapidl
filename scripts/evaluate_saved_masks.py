#!/usr/bin/env python3
"""Evaluate saved segmentation masks from benchmark runs.

Loads .npy masks from disk, computes all metrics, generates report.
Handles InstanSeg masks from separate env runs too.

Usage:
    uv run python scripts/evaluate_saved_masks.py \
        --masks-dir pipeline_output/segmentation_benchmark_v3/masks \
        --tiles-dir pipeline_output/segmentation_benchmark_v3/tiles \
        --extra-masks pipeline_output/segmentation_benchmark_v2/instanseg_masks \
        --output-dir pipeline_output/segmentation_benchmark_final
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MERSCOPE_DIR = Path("/mnt/work/datasets/raw/merscope/merscope-breast")
CELL_METADATA_PATH = MERSCOPE_DIR / "cell_metadata.csv"

# Affine transform constants
SCALE = 9.259316
OFFSET_X = 357.162
OFFSET_Y = 2007.972


def parse_mask_filename(filename: str) -> tuple[str, str, str]:
    """Parse 'fov_{id}_{label}_{method}.npy' into (fov_id, label, method)."""
    stem = filename.replace(".npy", "")
    parts = stem.split("_")
    # fov_{id}_{label}_{method parts...}
    fov_id = parts[1]
    fov_label = parts[2]
    method = "_".join(parts[3:])
    return fov_id, fov_label, method


def load_all_masks(masks_dir: Path, extra_dirs: list[Path] | None = None) -> dict:
    """Load masks grouped by FOV label and method.

    Returns: {fov_label: {method: masks_array}}
    """
    all_masks = {}

    for d in [masks_dir] + (extra_dirs or []):
        if not d.exists():
            logger.warning("Masks dir not found: %s", d)
            continue
        for f in sorted(d.glob("*.npy")):
            # Handle different naming: fov_{id}_{label}_{method}.npy or fov_{label}_masks.npy
            stem = f.stem
            if stem.endswith("_masks"):
                # InstanSeg style: fov_{id}_{label}_masks.npy
                parts = stem.replace("_masks", "").split("_")
                fov_label = parts[2] if len(parts) >= 3 else parts[1]
                method = d.name.replace("_masks", "")  # e.g. "instanseg"
            else:
                _, fov_label, method = parse_mask_filename(f.name)

            masks = np.load(f)
            # InstanSeg outputs (2, H, W) — nuclei + cells; take nuclei channel
            if masks.ndim == 3 and masks.shape[0] == 2:
                masks = masks[0]
            all_masks.setdefault(fov_label, {})[method] = masks
            n = int(masks.max())
            logger.info("  Loaded %s/%s: %d cells, shape %s", fov_label, method, n, masks.shape)

    return all_masks


def get_fov_native_centroids(cell_metadata: pl.DataFrame, fov_id: int, tile_shape: tuple, fov_pixel_bbox: tuple) -> np.ndarray:
    """Get native centroids in tile-relative pixel coordinates."""
    fov_cells = cell_metadata.filter(pl.col("fov") == fov_id)
    if fov_cells.is_empty():
        return np.empty((0, 2), dtype=np.float64)

    y_origin, x_origin = fov_pixel_bbox[0], fov_pixel_bbox[2]
    centroids = np.column_stack([
        fov_cells["center_y"].to_numpy() * SCALE + OFFSET_Y - y_origin,
        fov_cells["center_x"].to_numpy() * SCALE + OFFSET_X - x_origin,
    ])
    return centroids.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved segmentation masks")
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--extra-masks", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-metadata", type=Path, default=CELL_METADATA_PATH)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load cell metadata
    logger.info("Loading cell metadata...")
    cell_metadata = pl.read_csv(str(args.cell_metadata))

    # Parse FOV info from tile filenames
    fov_info = {}  # label -> (fov_id, tile_shape, pixel_bbox approximation)
    import tifffile
    for tf in sorted(args.tiles_dir.glob("fov_*.tif")):
        parts = tf.stem.split("_")
        fov_id = int(parts[1])
        fov_label = parts[2]
        tile = tifffile.imread(str(tf))
        # Approximate pixel bbox from cell metadata
        fov_cells = cell_metadata.filter(pl.col("fov") == fov_id)
        if not fov_cells.is_empty():
            x_min = fov_cells["min_x"].min() - 10
            y_min = fov_cells["min_y"].min() - 10
            py_min = int(y_min * SCALE + OFFSET_Y)
            px_min = int(x_min * SCALE + OFFSET_X)
            fov_info[fov_label] = (fov_id, tile.shape, (py_min, py_min + tile.shape[0], px_min, px_min + tile.shape[1]))
        del tile  # free memory

    # Load all masks
    logger.info("Loading masks...")
    all_masks = load_all_masks(args.masks_dir, args.extra_masks)

    # Import evaluation functions
    from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics
    from dapidl.benchmark.evaluation.biological import compute_biological_metrics

    # Evaluate each method on each FOV
    all_metrics = {}
    for fov_label, methods in sorted(all_masks.items()):
        if fov_label not in fov_info:
            logger.warning("No tile info for FOV %s, skipping bio metrics", fov_label)
            continue

        fov_id, tile_shape, pixel_bbox = fov_info[fov_label]

        # Get native centroids once per FOV
        native_centroids = get_fov_native_centroids(cell_metadata, fov_id, tile_shape, pixel_bbox)

        for method, masks in sorted(methods.items()):
            logger.info("Evaluating %s / %s...", fov_label, method)

            # Morphometric
            morph = compute_morphometric_metrics(masks, pixel_size_um=0.108)

            # Biological
            bio = compute_biological_metrics(masks, native_centroids, pixel_size_um=0.108)

            all_metrics.setdefault(method, {})[fov_label] = {
                "morphometric": morph,
                "biological": bio,
                "practical": {"n_detected": int(masks.max()), "runtime_seconds": 0.0, "peak_memory_mb": 0.0},
            }

            # Free mask memory
            del masks

    # Cross-method evaluation (one FOV at a time to save memory)
    logger.info("Computing cross-method agreement...")
    from dapidl.benchmark.consensus.instance_matching import match_instances_iou

    cross_method = {}
    for fov_label in sorted(all_masks.keys()):
        methods = all_masks[fov_label]
        method_names = sorted(m for m in methods if not m.startswith("consensus"))
        n = len(method_names)
        pairwise_iou = np.zeros((n, n))
        pairwise_match = np.zeros((n, n))

        for i in range(n):
            pairwise_iou[i, i] = 1.0
            pairwise_match[i, i] = 1.0
            for j in range(i + 1, n):
                masks_a = methods[method_names[i]]
                masks_b = methods[method_names[j]]
                matches, ious = match_instances_iou(masks_a, masks_b, iou_threshold=0.3)
                n_max = max(int(masks_a.max()), int(masks_b.max()), 1)
                mean_iou = float(np.mean(ious)) if ious else 0.0
                match_rate = len(matches) / n_max
                pairwise_iou[i, j] = pairwise_iou[j, i] = mean_iou
                pairwise_match[i, j] = pairwise_match[j, i] = match_rate

        cross_method[fov_label] = {
            "methods": method_names,
            "pairwise_iou": pairwise_iou.tolist(),
            "pairwise_match_rate": pairwise_match.tolist(),
        }
        logger.info("  FOV %s: %d methods compared", fov_label, n)

    # Save cross-method
    eval_dir = args.output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / "cross_method.json", "w") as f:
        json.dump(cross_method, f, indent=2)

    # Generate report
    logger.info("Generating report...")
    from dapidl.benchmark.reporting import generate_report
    report_path = generate_report(all_metrics, args.output_dir)

    logger.info("Report: %s", report_path)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
