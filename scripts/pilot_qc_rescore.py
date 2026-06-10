"""2-pass enriched QC re-score for the pilot LMDB.

Pass 1 (GPU): StarDist segmentation + metrics + masks → seg_scores_raw.parquet
Pass 2 (cheap): calibrate → axes → grade → seg_scores.parquet

Usage:
    # Both passes (default)
    uv run python scripts/pilot_qc_rescore.py

    # Smoke against p128 LMDB (12 patches, no registry required)
    uv run python scripts/pilot_qc_rescore.py \\
        --phase pass1 \\
        --lmdb /mnt/work/datasets/derived/breast-pilot-6source-dapi-p128-nuc-v3 \\
        --patch-size 128 --limit 12

    # Full run after the p64 LMDB is complete
    uv run python scripts/pilot_qc_rescore.py --phase both
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import polars as pl
import scipy.ndimage

from dapidl.qc.cell_boundary import rasterize_polygon_to_patch, voronoi_cell_mask
from dapidl.qc.quality_model import SAT_LEVEL, Calibration, axes, calibrate, grade
from starpose.qc.segmentation_grounded import (
    SegQCConfig,
    SegmentationGroundedScorer,
    decide_broken,
    score_from_segmentation,
    select_center_nucleus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LMDB_DIR = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1")
XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
CHUNK = 2000  # mask flush granularity

# All keys emitted by score_from_segmentation (has_nucleus + full set)
_FULL_METRIC_KEYS = [
    "area_um2", "brenner", "centeredness", "completeness", "dominant_central",
    "eccentricity", "edge_cut", "glcm_asm", "glcm_entropy", "has_nucleus",
    "intensity_ok", "intensity_ratio", "interior_cov", "morph_ok",
    "solidity", "stardist_prob", "structure_raw",
]


def _fill_metrics(partial: dict) -> dict:
    """Ensure all metric keys are present (fill missing with NaN for no-nucleus rows)."""
    out: dict = {}
    for k in _FULL_METRIC_KEYS:
        out[k] = float(partial.get(k, float("nan")))
    return out


# ---------------------------------------------------------------------------
# Xenium cell-boundary loader (slide-level cache)
# ---------------------------------------------------------------------------
class XeniumBoundaryCache:
    """Lazy per-slide dict[cell_id -> np.ndarray shape (K,2) in microns (x,y)]."""

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}

    def _slide_to_rep(self, slide: str) -> Optional[str]:
        """Map slide name e.g. 'xenium_rep1_nuc' → 'xenium-breast-tumor-rep1'."""
        # strip leading 'xenium_' and trailing '_nuc' suffix
        inner = slide.removeprefix("xenium_")
        inner = inner.removesuffix("_nuc")
        # rep1 / rep2 → xenium-breast-tumor-rep1 / xenium-breast-tumor-rep2
        if inner.startswith("rep"):
            return f"xenium-breast-tumor-{inner}"
        return None

    def load(self, slide: str) -> Optional[dict]:
        if slide in self._cache:
            return self._cache[slide]
        rep_dir = self._slide_to_rep(slide)
        if rep_dir is None:
            self._cache[slide] = {}
            return {}
        parquet = XENIUM_BASE / rep_dir / "outs" / "cell_boundaries.parquet"
        if not parquet.exists():
            print(f"[warn] cell_boundaries not found: {parquet}", file=sys.stderr)
            self._cache[slide] = {}
            return {}
        df = pl.read_parquet(parquet)
        # Build dict[cell_id -> (K,2) array (x,y) in microns]
        poly_map: dict = {}
        for cell_id, group in df.group_by("cell_id"):
            pts = group.select(["vertex_x", "vertex_y"]).to_numpy()
            poly_map[int(cell_id)] = pts.astype(np.float64)
        self._cache[slide] = poly_map
        print(f"[bound] loaded {len(poly_map):,} cells for {slide}", flush=True)
        return poly_map


# ---------------------------------------------------------------------------
# resolve_cell
# ---------------------------------------------------------------------------
def resolve_cell(
    row: dict,
    nuc_mask: np.ndarray,
    patch_size: int,
    boundary_cache: XeniumBoundaryCache,
) -> tuple[np.ndarray, str]:
    """Return (cell_mask bool HxW, provenance_str).

    Priority:
      1. Native Xenium polygon (if slide starts with 'xenium_' AND polygon found AND nonzero)
      2. Voronoi expansion of nuc_mask
      3. All-False mask if nuc_mask is empty and no polygon
    """
    slide: str = row.get("slide", "")
    has_coords = "x0" in row and "y0" in row and "pixel_size" in row

    if slide.startswith("xenium_") and has_coords:
        poly_map = boundary_cache.load(slide)
        try:
            cell_id = int(row["cell_id"])
        except (KeyError, TypeError, ValueError):
            cell_id = None
        if poly_map and cell_id is not None and cell_id in poly_map:
            poly_um = poly_map[cell_id]
            mask = rasterize_polygon_to_patch(
                poly_um, float(row["x0"]), float(row["y0"]),
                float(row["pixel_size"]), patch_size,
            )
            if mask.any():
                return mask, "native_xenium"
            # polygon off-frame → fall through

    if not nuc_mask.any():
        return np.zeros((patch_size, patch_size), dtype=bool), "none"

    return voronoi_cell_mask(nuc_mask), "voronoi"


# ---------------------------------------------------------------------------
# Pass 1: GPU segmentation + raw scoring
# ---------------------------------------------------------------------------
def _read_patch(txn, idx: int, patch_size: int) -> np.ndarray:
    value = txn.get(struct.pack(">Q", int(idx)))
    if value is None:
        return np.zeros((patch_size, patch_size), dtype=np.uint16)
    return np.frombuffer(value[8:], dtype=np.uint16).reshape(patch_size, patch_size)


def pass1(
    lmdb_dir: Path,
    out_dir: Path,
    patch_size: int,
    limit: int = 0,
) -> None:
    """GPU pass: segment every patch, compute metrics, write masks + seg_scores_raw.parquet."""
    t_start = time.time()
    print("[pass1] starting GPU segmentation pass", flush=True)

    # -----------------------------------------------------------------------
    # Load registry (or synthesise a stub if absent — for smoke runs)
    # -----------------------------------------------------------------------
    registry_path = lmdb_dir / "patch_registry.parquet"
    if registry_path.exists():
        registry = pl.read_parquet(registry_path)
        has_coords = "x0" in registry.columns
        print(f"[pass1] registry: {registry.shape} rows, has_coords={has_coords}", flush=True)
    else:
        print("[pass1] no patch_registry.parquet found — synthesising stub from LMDB keys",
              flush=True)
        env_tmp = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False)
        with env_tmp.begin() as txn:
            n_total = env_tmp.stat()["entries"]
        env_tmp.close()
        registry = pl.DataFrame({
            "row_idx": list(range(n_total)),
            "slide": ["unknown"] * n_total,
            "cell_id": ["0"] * n_total,
            "coarse_idx": [0] * n_total,
        })
        has_coords = False

    total = len(registry)
    if limit > 0:
        registry = registry.head(limit)
        print(f"[pass1] --limit {limit}: processing {len(registry)}/{total} rows", flush=True)

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    cfg = SegQCConfig(erode_px=1)
    scorer = SegmentationGroundedScorer(cfg, gpu=True, pixel_size=0.2125)
    boundary_cache = XeniumBoundaryCache()

    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False,
                    readahead=False, meminit=False)

    # -----------------------------------------------------------------------
    # Per-slide reference fitting (fit_reference on up to 200 patches/slide)
    # -----------------------------------------------------------------------
    slide_refs: dict[str, float] = {}
    slides = registry["slide"].unique().to_list()
    with env.begin() as txn:
        for slide in slides:
            slide_rows = registry.filter(pl.col("slide") == slide)["row_idx"].to_list()
            rng = np.random.default_rng(0)
            sample_idxs = rng.choice(slide_rows, size=min(200, len(slide_rows)),
                                     replace=False).tolist()
            sample_patches = np.stack([
                _read_patch(txn, idx, patch_size) for idx in sample_idxs
            ])
            ref = scorer.fit_reference(sample_patches)
            slide_refs[slide] = ref.varlap_p90
            print(f"[pass1] slide={slide}  ref_p90={ref.varlap_p90:.4f}", flush=True)

    # -----------------------------------------------------------------------
    # Main scoring loop
    # -----------------------------------------------------------------------
    rows_out: list[dict] = []
    nuc_buf: list[np.ndarray] = []
    cell_buf: list[np.ndarray] = []
    chunk_row_idxs: list[int] = []
    chunk_start = 0
    n_done = n_broken = 0

    def _flush_masks(force: bool = False) -> None:
        nonlocal chunk_start, nuc_buf, cell_buf, chunk_row_idxs
        if not nuc_buf:
            return
        if not force and len(nuc_buf) < CHUNK:
            return
        flat = len(nuc_buf)
        packed_nuc = np.packbits(
            np.stack(nuc_buf).reshape(flat, -1).astype(np.uint8), axis=1
        )
        packed_cell = np.packbits(
            np.stack(cell_buf).reshape(flat, -1).astype(np.uint8), axis=1
        )
        np.savez_compressed(
            masks_dir / f"nuc_chunk_{chunk_start:09d}.npz",
            packed=packed_nuc,
            row_idx=np.array(chunk_row_idxs, dtype=np.int64),
        )
        np.savez_compressed(
            masks_dir / f"cell_chunk_{chunk_start:09d}.npz",
            packed=packed_cell,
            row_idx=np.array(chunk_row_idxs, dtype=np.int64),
        )
        chunk_start = chunk_row_idxs[-1] + 1
        nuc_buf, cell_buf, chunk_row_idxs = [], [], []

    with env.begin() as txn:
        for row_pl in registry.iter_rows(named=True):
            row_idx = int(row_pl["row_idx"])
            slide = str(row_pl["slide"])
            ref_p90 = slide_refs.get(slide, 1.0)

            patch = _read_patch(txn, row_idx, patch_size)
            masks, probs = scorer._segment(patch)
            cn = select_center_nucleus(masks, probs, cfg)
            qs = score_from_segmentation(patch, masks, probs, ref_p90, 0.2125, cfg)
            broken, reason = decide_broken(qs, cfg)

            # Nucleus mask
            nuc_mask = cn.mask.astype(bool) if cn is not None else np.zeros(
                (patch_size, patch_size), dtype=bool
            )

            # Saturation penalty (eroded nucleus interior)
            if nuc_mask.any():
                interior = scipy.ndimage.binary_erosion(nuc_mask, iterations=1)
                if interior.any():
                    sat_penalty = float(
                        (patch[interior] >= SAT_LEVEL * 65535).mean()
                    )
                else:
                    sat_penalty = 0.0
            else:
                sat_penalty = 0.0

            # Cell mask
            cell_mask, cell_provenance = resolve_cell(
                row_pl, nuc_mask, patch_size, boundary_cache
            )

            # Fill metric dict with all expected keys
            metrics = _fill_metrics(qs.metrics)

            # Build output row
            out_row: dict = {
                "row_idx": row_idx,
                "slide": slide,
                "cell_id": str(row_pl.get("cell_id", "")),
                "coarse_idx": int(row_pl.get("coarse_idx", -1)),
                "broken": bool(broken),
                "broken_reason": str(reason),
                "sat_penalty": sat_penalty,
                "cell_provenance": cell_provenance,
                **{k: metrics[k] for k in _FULL_METRIC_KEYS},
            }
            rows_out.append(out_row)

            # Accumulate masks
            nuc_buf.append(nuc_mask)
            cell_buf.append(cell_mask)
            chunk_row_idxs.append(row_idx)
            if len(nuc_buf) >= CHUNK:
                _flush_masks()

            n_done += 1
            if broken:
                n_broken += 1
            if n_done % 500 == 0:
                rate = n_done / (time.time() - t_start)
                print(f"[pass1] {n_done}/{len(registry)}  "
                      f"broken={n_broken}  {rate:.1f} patches/s", flush=True)

    _flush_masks(force=True)

    # Write raw parquet
    raw_path = out_dir / "seg_scores_raw.parquet"
    pl.DataFrame(rows_out).write_parquet(raw_path)
    dt = time.time() - t_start
    print(
        f"[pass1] DONE  {n_done} patches  broken={n_broken} ({100*n_broken/max(n_done,1):.1f}%)"
        f"  {dt:.1f}s  → {raw_path}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Pass 2: calibrate → axes → grade
# ---------------------------------------------------------------------------
def pass2(out_dir: Path) -> None:
    """Cheap CPU pass: load raw scores, calibrate, compute axes + grade."""
    raw_path = out_dir / "seg_scores_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Pass 1 output not found: {raw_path}. Run pass1 first.")

    raw = pl.read_parquet(raw_path)
    print(f"[pass2] loaded {raw.shape} from {raw_path}", flush=True)

    # Calibrate on non-broken patches only
    cal: Calibration = calibrate(raw)
    cal_dict = {
        "brenner_lo": cal.brenner_lo, "brenner_hi": cal.brenner_hi,
        "ent_lo": cal.ent_lo, "ent_hi": cal.ent_hi,
        "cov_lo": cal.cov_lo, "cov_hi": cal.cov_hi,
        "ir_lo": cal.ir_lo, "ir_hi": cal.ir_hi,
    }
    cal_path = out_dir / "quality_calibration.json"
    cal_path.write_text(json.dumps(cal_dict, indent=2))
    print(f"[pass2] calibration written → {cal_path}", flush=True)

    # Compute axes + grade per row
    detections, focuses, textures, brightnesses, quality_mins, grades = (
        [], [], [], [], [], []
    )
    for row in raw.iter_rows(named=True):
        metric = {k: row[k] for k in _FULL_METRIC_KEYS}
        metric["sat_penalty"] = row["sat_penalty"]
        if row["broken"]:
            ax_vals = {"detection": float("nan"), "focus": float("nan"),
                       "texture": float("nan"), "brightness": float("nan")}
            qmin = float("nan")
            g = "broken"
        else:
            ax_vals = axes(metric, cal)
            qmin = float(min(ax_vals.values()))
            g = grade(qmin)
        detections.append(ax_vals["detection"])
        focuses.append(ax_vals["focus"])
        textures.append(ax_vals["texture"])
        brightnesses.append(ax_vals["brightness"])
        quality_mins.append(qmin)
        grades.append(g)

    scored = raw.with_columns([
        pl.Series("detection", detections),
        pl.Series("focus", focuses),
        pl.Series("texture", textures),
        pl.Series("brightness", brightnesses),
        pl.Series("quality_min", quality_mins),
        pl.Series("grade", grades),
    ])
    out_path = out_dir / "seg_scores.parquet"
    scored.write_parquet(out_path)
    # Summary
    grade_counts = scored["grade"].value_counts().sort("count", descending=True)
    print(f"[pass2] grade distribution:\n{grade_counts}", flush=True)
    print(f"[pass2] DONE  → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="2-pass enriched QC re-score (Pass1=GPU seg, Pass2=axes+grade)"
    )
    ap.add_argument("--lmdb", type=Path, default=LMDB_DIR,
                    help=f"LMDB dataset directory (default: {LMDB_DIR})")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory (default: <lmdb>/qc)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N patches (0 = all); for smoke runs")
    ap.add_argument("--patch-size", type=int, default=64,
                    help="Patch side length in pixels (default 64; use 128 for p128 smoke)")
    ap.add_argument("--phase", choices=["pass1", "pass2", "both"], default="both",
                    help="Which phase to run")
    args = ap.parse_args()

    lmdb_dir: Path = args.lmdb
    out_dir: Path = args.out if args.out is not None else lmdb_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (lmdb_dir / "patches.lmdb").exists():
        print(f"[error] LMDB not found: {lmdb_dir / 'patches.lmdb'}", file=sys.stderr)
        sys.exit(1)

    print(f"[main] lmdb_dir  = {lmdb_dir}", flush=True)
    print(f"[main] out_dir   = {out_dir}", flush=True)
    print(f"[main] patch_size= {args.patch_size}", flush=True)
    print(f"[main] limit     = {args.limit if args.limit > 0 else 'all'}", flush=True)
    print(f"[main] phase     = {args.phase}", flush=True)

    if args.phase in ("pass1", "both"):
        pass1(lmdb_dir, out_dir, args.patch_size, limit=args.limit)

    if args.phase in ("pass2", "both"):
        pass2(out_dir)


if __name__ == "__main__":
    main()
