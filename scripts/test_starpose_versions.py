#!/usr/bin/env python3
"""Smoke-test starpose on representative Xenium and MERSCOPE versions.

For each dataset, lazily read a small (1024x1024) DAPI crop from a region
with tissue signal, run starpose with a lightweight method on CPU, and
report n_cells + runtime. Designed to coexist with running GPU jobs by
keeping starpose on CPU and operating on tiny images.

Usage:
    uv run python scripts/test_starpose_versions.py
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

import starpose

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("test_starpose")

CROP = 1024  # px per side
MIN_MEAN = 200  # uint16 mean threshold to consider a region "tissue"


@dataclass
class Case:
    name: str
    version_label: str
    image_path: Path
    pixel_size: float = 0.2125  # Xenium default; overridden for MERSCOPE
    series: int = 0  # for OME-TIFF series selection
    level: int = 0  # for pyramidal OME-TIFF


def _pick_tissue_crop(arr_or_store, shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Try a few crop offsets, pick the first with mean intensity > MIN_MEAN."""
    H, W = shape[:2]
    candidates = [
        (H // 2, W // 2),
        (H // 3, W // 3),
        (2 * H // 3, 2 * W // 3),
        (H // 4, W // 2),
        (H // 2, W // 4),
        (3 * H // 4, W // 2),
        (H // 2, 3 * W // 4),
    ]
    best = None
    best_mean = 0
    for cy, cx in candidates:
        y0 = max(0, cy - CROP // 2)
        x0 = max(0, cx - CROP // 2)
        y1 = min(H, y0 + CROP)
        x1 = min(W, x0 + CROP)
        crop = arr_or_store[y0:y1, x0:x1]
        if hasattr(crop, "compute"):
            crop = crop.compute()
        crop = np.asarray(crop)
        m = float(crop.mean())
        if m > best_mean:
            best_mean = m
            best = (crop, (y0, x0))
        if m > MIN_MEAN * 4:  # plenty of signal — stop early
            break
    return best  # type: ignore[return-value]


def load_crop(case: Case) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Lazily open OME-TIFF and extract a 1024x1024 crop from a tissue region.

    Uses tifffile -> zarr store for true lazy slicing (no full-file mmap).
    """
    import zarr

    log.info(f"opening {case.name} ({case.image_path})")
    with tifffile.TiffFile(case.image_path) as tf:
        # Pyramidal OME-TIFF: pick the right level (0 = full resolution)
        series = tf.series[case.series]
        levels = series.levels if (hasattr(series, "levels") and series.levels) else [series]
        level = levels[min(case.level, len(levels) - 1)]
        log.info(f"  full shape={level.shape}, dtype={level.dtype}, levels={len(levels)}")

        # Open as zarr store for lazy slicing — does NOT read pixel data.
        store = level.aszarr()
        try:
            z = zarr.open(store, mode="r")
            # Some pyramidal OME-TIFFs return a Group at top level instead of an Array.
            if hasattr(z, "array_keys"):  # duck-type Group (zarr v2 + v3 compatible)
                key = sorted(z.array_keys())[0]
                z = z[key]
            # Reduce to 2D by selecting the middle Z-plane (best focus by convention).
            arr = z
            while arr.ndim > 2:
                mid = arr.shape[0] // 2
                arr = arr[mid]
            shape = (int(arr.shape[0]), int(arr.shape[1]))
            crop, (y0, x0) = _pick_tissue_crop(arr, shape)
            return crop, (y0, x0), shape
        finally:
            try:
                store.close()
            except Exception:
                pass


def run_case(case: Case, method: str = "cellpose_nuclei", gpu: bool = True) -> dict:
    """Load a crop and run starpose. Return a metrics dict."""
    t0 = time.time()
    try:
        crop, offset, full_shape = load_crop(case)
    except Exception as e:
        return {"name": case.name, "version": case.version_label, "error": f"load: {e}"}

    log.info(
        f"  crop shape={crop.shape}, dtype={crop.dtype}, "
        f"offset=(y={offset[0]}, x={offset[1]}), "
        f"mean={crop.mean():.0f}, max={crop.max()}"
    )

    if crop.dtype != np.uint16:
        crop = crop.astype(np.uint16)

    # Run starpose with a small CPU-friendly method
    try:
        t1 = time.time()
        result = starpose.segment(
            crop,
            method=method,
            gpu=gpu,
            pixel_size=case.pixel_size,
            tile_size=CROP + 1,  # disable internal tiling for this small crop
            overlap=0,
        )
        seg_time = time.time() - t1
    except Exception as e:
        return {
            "name": case.name,
            "version": case.version_label,
            "method": method,
            "n_cells": 0,
            "load_time": time.time() - t0,
            "error": f"segment: {type(e).__name__}: {e}",
        }

    return {
        "name": case.name,
        "version": case.version_label,
        "method": method,
        "n_cells": int(result.n_cells),
        "load_time": round(time.time() - t0 - seg_time, 1),
        "seg_time": round(seg_time, 1),
        "crop_mean": int(crop.mean()),
        "full_shape": full_shape,
        "offset": offset,
    }


def main():
    cases = [
        Case(
            name="xenium-lung-2fov",
            version_label="Xenium v1 (lung 2fov, small)",
            image_path=Path("/mnt/work/datasets/raw/xenium/xenium-lung-2fov/morphology.ome.tif"),
            pixel_size=0.2125,
        ),
        Case(
            name="xenium-breast-tumor-rep2",
            version_label="Xenium v1 (focus + new, both present)",
            image_path=Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/morphology_focus.ome.tif"),
            pixel_size=0.2125,
        ),
        Case(
            name="xenium-breast-tumor-rep2-newformat",
            version_label="Xenium v2 (new format, same dataset)",
            image_path=Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2/morphology.ome.tif"),
            pixel_size=0.2125,
        ),
        Case(
            name="xenium-skin-prime-ffpe",
            version_label="Xenium Prime (5K panel)",
            image_path=Path("/mnt/work/datasets/raw/xenium/xenium-skin-prime-ffpe/morphology.ome.tif"),
            pixel_size=0.2125,
        ),
        Case(
            name="merscope-breast",
            version_label="MERSCOPE FFPE",
            image_path=Path("/mnt/work/datasets/raw/merscope/merscope-breast/images/mosaic_DAPI_z3.tif"),
            pixel_size=0.108,
        ),
        Case(
            name="merscope-ovarian-cancer-3",
            version_label="MERSCOPE smallest",
            image_path=Path("/mnt/work/datasets/raw/merscope/merscope-ovarian-cancer-3/images/mosaic_DAPI_z3.tif"),
            pixel_size=0.108,
        ),
    ]

    results = []
    for case in cases:
        if not case.image_path.exists():
            log.warning(f"SKIP {case.name}: image not found at {case.image_path}")
            continue
        log.info(f"=== {case.name} | {case.version_label} ===")
        r = run_case(case, method="cellpose_nuclei", gpu=True)
        results.append(r)
        log.info(f"  result: {r}\n")

    log.info("=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)
    print(f"{'name':<40} {'version':<35} {'n_cells':>8} {'sec':>6}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<40} {r['version']:<35} {'ERROR':>8} {r.get('error','')[:40]}")
        else:
            print(f"{r['name']:<40} {r['version']:<35} {r['n_cells']:>8} {r['seg_time']:>6}")


if __name__ == "__main__":
    main()
