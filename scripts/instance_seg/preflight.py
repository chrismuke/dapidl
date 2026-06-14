"""Phase 0 preflight: validate STHELAR breast slides before tile-cache build.

Checks every assumption from the design doc:
  1. DAPI channel selectable via OMERO label (s6 is multi-channel)
  2. Geometry parquet readable via PyArrow + GeoPandas (Polars panics)
  3. cell_id join works (s1 has row-position misalignment)
  4. Polygon validity (s6 has 1.6% invalid)
  5. ct_tangram → fine → medium → broad mapping covers all labels
  6. Free disk vs estimated cache size at the chosen stride
  7. Free RAM vs largest geometry load
  8. Per-slide class distributions at all three tiers; warns on biological outliers
     (e.g. s6 has 29% Mast cells in ct_tangram)

Exits non-zero if any blocker fails.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import psutil
from loguru import logger

from dapidl.data.sthelar import (
    DapiChannelError,
    STHELAR_SCALE_FACTOR,
    TANGRAM_TO_BROAD,
    TANGRAM_TO_MEDIUM,
    load_nucleus_geometry_with_labels,
    load_omero_attrs,
    select_dapi_channel,
)
from dapidl.pipeline.components.annotators.mapping import COARSE_CLASS_NAMES
from dapidl.pipeline.components.annotators.popv_ensemble import MEDIUM_CLASS_NAMES

DEFAULT_ROOT = Path("/mnt/work/datasets/STHELAR/sdata_slides")
DEFAULT_OUT = Path("/mnt/work/datasets/derived/sthelar_breast_tiles")
TILE_SIZE = 1024
DTYPE_BYTES = 2  # uint16


def slide_path(root: Path, slide: str) -> Path:
    return root / f"sdata_{slide}.zarr" / f"sdata_{slide}.zarr"


def estimate_grid_tiles(slide_root: Path, stride: int) -> int:
    import zarr

    grp = zarr.open(str(slide_root), mode="r")
    shape = grp["images/morpho/0"].shape  # (C, H, W) or (H, W)
    h, w = (shape[1], shape[2]) if len(shape) == 3 else shape
    return ((h + stride - 1) // stride) * ((w + stride - 1) // stride)


def map_fine(label: str) -> tuple[str, str, str]:
    """Return (fine, medium, broad) for a `ct_tangram` value."""
    return (
        label,
        TANGRAM_TO_MEDIUM.get(label, "Unknown"),
        TANGRAM_TO_BROAD.get(label, "Unknown"),
    )


def check_slide(
    slide: str,
    slide_root: Path,
    rare_warn_min: int = 10,
    outlier_frac: float = 0.25,
) -> dict:
    """Returns dict with `pass: bool`, `errors: list[str]`, `warnings: list[str]`,
    plus computed metrics (cell counts, invalid drops, class distributions).
    """
    rec: dict = {"slide": slide, "errors": [], "warnings": [], "metrics": {}}

    # 1. DAPI channel
    try:
        dapi_ch = select_dapi_channel(slide_root)
        rec["metrics"]["dapi_channel"] = dapi_ch
        omero = load_omero_attrs(slide_root).get("omero", {}).get("channels", [])
        rec["metrics"]["n_channels"] = len(omero)
    except DapiChannelError as e:
        rec["errors"].append(f"DAPI selection: {e}")
        return {"pass": False, **rec}

    # 2-4. Geometry load + join + validity
    try:
        gdf = load_nucleus_geometry_with_labels(
            slide_root, ["ct_tangram", "label1", "label2"]
        )
    except Exception as e:
        rec["errors"].append(f"Geometry load: {type(e).__name__}: {e}")
        return {"pass": False, **rec}

    rec["metrics"]["n_cells"] = len(gdf)
    rec["metrics"]["n_invalid_dropped"] = gdf.attrs.get("n_invalid_dropped", 0)
    invalid_frac = rec["metrics"]["n_invalid_dropped"] / max(
        rec["metrics"]["n_cells"] + rec["metrics"]["n_invalid_dropped"], 1
    )
    rec["metrics"]["invalid_polygon_frac"] = round(invalid_frac, 4)
    if invalid_frac > 0.05:
        rec["warnings"].append(
            f"{invalid_frac*100:.1f}% invalid polygons (>5% threshold)"
        )

    # geometry bounds in pixels — sanity vs image size
    bx0, by0, bx1, by1 = gdf.total_bounds
    import zarr

    grp = zarr.open(str(slide_root), mode="r")
    img_shape = grp["images/morpho/0"].shape
    h, w = (img_shape[1], img_shape[2]) if len(img_shape) == 3 else img_shape
    rec["metrics"]["image_h_w"] = (int(h), int(w))
    rec["metrics"]["geom_bounds_px"] = [
        round(float(b), 1) for b in (bx0, by0, bx1, by1)
    ]
    if bx1 > w * 1.01 or by1 > h * 1.01:
        rec["errors"].append(
            f"Geometry bounds {(bx1, by1)} exceed image {(w, h)} — wrong scale?"
        )

    # 5. ct_tangram → 3-tier mapping coverage (uses STHELAR-specific maps)
    fine_counts: dict[str, int] = {}
    medium_counts: dict[str, int] = {}
    broad_counts: dict[str, int] = {}
    unmapped_to_medium: set[str] = set()
    unmapped_to_broad: set[str] = set()
    for ct, n in gdf["ct_tangram"].value_counts().items():
        n = int(n)
        fine_counts[ct] = n
        med = TANGRAM_TO_MEDIUM.get(ct, "__UNMAPPED__")
        if med == "__UNMAPPED__":
            unmapped_to_medium.add(ct)
            med = "Unknown"
        medium_counts[med] = medium_counts.get(med, 0) + n
        broad = TANGRAM_TO_BROAD.get(ct, "__UNMAPPED__")
        if broad == "__UNMAPPED__":
            unmapped_to_broad.add(ct)
            broad = "Unknown"
        broad_counts[broad] = broad_counts.get(broad, 0) + n

    rec["metrics"]["fine_counts"] = fine_counts
    rec["metrics"]["medium_counts"] = medium_counts
    rec["metrics"]["broad_counts"] = broad_counts

    if unmapped_to_medium:
        rec["errors"].append(
            f"{len(unmapped_to_medium)} ct_tangram labels missing from "
            f"TANGRAM_TO_MEDIUM in sthelar.py: {sorted(unmapped_to_medium)}"
        )
    if unmapped_to_broad:
        rec["errors"].append(
            f"{len(unmapped_to_broad)} ct_tangram labels missing from "
            f"TANGRAM_TO_BROAD in sthelar.py: {sorted(unmapped_to_broad)}"
        )

    # 5a. Required broad classes present
    expected_broad = {"Epithelial", "Immune", "Stromal", "Endothelial"}
    missing_broad = expected_broad - set(broad_counts)
    if missing_broad:
        rec["errors"].append(f"Missing broad classes: {missing_broad}")

    # 5b. Required medium classes present at minimum count
    rare_medium = {
        c: broad_counts.get(c, 0)
        for c in MEDIUM_CLASS_NAMES
        if medium_counts.get(c, 0) < rare_warn_min
    }
    if rare_medium:
        rec["warnings"].append(
            f"Medium classes below {rare_warn_min} cells: "
            f"{ {c: medium_counts.get(c, 0) for c in rare_medium} }"
        )

    # 6. Biological outlier check (s6 issue)
    biologically_meaningful = {
        c: n
        for c, n in fine_counts.items()
        if c.lower() not in {"less10", "unknown"}
    }
    bio_total = sum(biologically_meaningful.values()) or 1
    for c, n in biologically_meaningful.items():
        if n / bio_total > outlier_frac:
            rec["warnings"].append(
                f"Biological outlier: ct_tangram={c} is {n/bio_total*100:.1f}% "
                f"of non-Unknown cells (>{outlier_frac*100:.0f}% threshold)"
            )

    rec["pass"] = not rec["errors"]
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--slides", nargs="+", default=["breast_s0", "breast_s1", "breast_s3", "breast_s6"]
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("pipeline_output/instance_seg/preflight_report.json"),
    )
    ap.add_argument(
        "--free-disk-min-gib",
        type=float,
        default=80.0,
        help="Fail if /mnt/work free disk falls below this after cache estimate",
    )
    ap.add_argument(
        "--free-ram-min-gib",
        type=float,
        default=8.0,
        help="Fail if available RAM is below this (geometry load alone can use ~1 GB per slide)",
    )
    args = ap.parse_args()

    logger.info(f"Preflight on {len(args.slides)} slide(s) at stride={args.stride}")
    logger.info(f"  STHELAR_SCALE_FACTOR = {STHELAR_SCALE_FACTOR}")
    logger.info(f"  COARSE = {COARSE_CLASS_NAMES} + Endothelial (4-class)")
    logger.info(f"  MEDIUM ({len(MEDIUM_CLASS_NAMES)}) = {MEDIUM_CLASS_NAMES}")

    # System resources
    free_ram_gib = psutil.virtual_memory().available / 1024**3
    free_disk_gib = shutil.disk_usage(args.root).free / 1024**3
    logger.info(f"  free RAM = {free_ram_gib:.1f} GiB")
    logger.info(f"  free disk on {args.root} = {free_disk_gib:.1f} GiB")
    sys_errors: list[str] = []
    if free_ram_gib < args.free_ram_min_gib:
        sys_errors.append(
            f"Available RAM {free_ram_gib:.1f} GiB < required {args.free_ram_min_gib}"
        )

    # Per-slide
    per_slide = []
    total_tiles = 0
    for slide in args.slides:
        sroot = slide_path(args.root, slide)
        if not sroot.exists():
            logger.error(f"  {slide}: NOT FOUND at {sroot}")
            sys_errors.append(f"{slide} missing at {sroot}")
            continue
        logger.info(f"  → {slide}")
        rec = check_slide(slide, sroot)
        per_slide.append(rec)
        total_tiles += estimate_grid_tiles(sroot, args.stride)

    # Cache size estimate (raw, uncompressed, uint16)
    raw_gib = total_tiles * TILE_SIZE * TILE_SIZE * DTYPE_BYTES / 1024**3
    image_raw_gib = total_tiles * TILE_SIZE * TILE_SIZE * 2 / 1024**3  # uint16 image
    raw_total_gib = raw_gib + image_raw_gib
    expected_after_compression_gib = raw_total_gib * 0.20  # rough zstd ratio for sparse
    logger.info(
        f"  total tiles at stride {args.stride} = {total_tiles}; raw uint16 = "
        f"{raw_total_gib:.1f} GiB (≈{expected_after_compression_gib:.1f} GiB compressed)"
    )
    if free_disk_gib - raw_total_gib < args.free_disk_min_gib:
        sys_errors.append(
            f"After raw cache build, free disk would be "
            f"{free_disk_gib - raw_total_gib:.1f} GiB < required {args.free_disk_min_gib}"
        )

    # Report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "stride": args.stride,
        "tile_size": TILE_SIZE,
        "free_ram_gib": round(free_ram_gib, 1),
        "free_disk_gib": round(free_disk_gib, 1),
        "total_grid_tiles": total_tiles,
        "raw_cache_gib": round(raw_total_gib, 1),
        "expected_compressed_gib": round(expected_after_compression_gib, 1),
        "system_errors": sys_errors,
        "slides": per_slide,
    }
    args.report.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Report written: {args.report}")

    # Stdout summary
    print()
    print("=" * 80)
    print(f"{'slide':<14} {'cells':>10} {'inv%':>5} {'ch':>3} "
          f"{'Epi':>8} {'Imm':>8} {'Str':>8} {'End':>8}")
    print("-" * 80)
    for r in per_slide:
        m = r["metrics"]
        b = m.get("broad_counts", {})
        print(f"{r['slide']:<14} {m.get('n_cells', 0):>10} "
              f"{m.get('invalid_polygon_frac', 0)*100:>4.1f}% "
              f"{m.get('dapi_channel', '?'):>3} "
              f"{b.get('Epithelial', 0):>8} {b.get('Immune', 0):>8} "
              f"{b.get('Stromal', 0):>8} {b.get('Endothelial', 0):>8}")
    print("=" * 80)
    print()

    # Failures
    n_errors = len(sys_errors) + sum(len(r["errors"]) for r in per_slide)
    n_warnings = sum(len(r["warnings"]) for r in per_slide)

    if sys_errors:
        for e in sys_errors:
            logger.error(f"  SYSTEM: {e}")
    for r in per_slide:
        for e in r["errors"]:
            logger.error(f"  {r['slide']}: {e}")
        for w in r["warnings"]:
            logger.warning(f"  {r['slide']}: {w}")

    if n_errors:
        logger.error(f"PREFLIGHT FAILED: {n_errors} blocker(s), {n_warnings} warning(s)")
        sys.exit(1)
    logger.success(f"PREFLIGHT PASSED: 0 blockers, {n_warnings} warning(s)")


if __name__ == "__main__":
    main()
