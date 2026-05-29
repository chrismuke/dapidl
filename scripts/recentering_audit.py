"""Re-centering audit — the spec §5 fail-fast gate for nucleus re-centering.

Verifies that switching Xenium patches from the CELL centroid to the NUCLEUS
polygon centroid moves cells by a sane, bounded amount, and that the fallback
rate (cells with no nucleus polygon -> reuse cell centroid) is small. STHELAR is
already nucleus-centered in both the cell- and nuc-centered builds, so its
displacement is 0 by construction (the experiment's placebo arm).

This reads ONLY centroids (cells.parquet + nucleus_boundaries.parquet) via the
readers — no DAPI image, no LMDB — so it is fast and RAM-safe.

Gates (per spec 2026-05-25 §5):
    - Xenium median Δ ∈ [0.5, 5.0] µm
    - Xenium fallback rate < 10 %
    - STHELAR Δ = 0.000 µm (by construction)

Outputs:
    pipeline_output/recentering_audit/
        audit_summary.parquet   one row per source
        audit_summary.md        markdown rendering for the readout

Usage:
    uv run python scripts/recentering_audit.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from dapidl.data.xenium import XeniumDataReader

XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")
DEFAULT_OUT = Path("/mnt/work/git/dapidl/pipeline_output/recentering_audit")

MED_LO, MED_HI = 0.5, 5.0          # acceptable median displacement (µm)
FALLBACK_MAX = 0.10                # max fraction of cells reusing the cell centroid


def audit_xenium(rep_name: str) -> dict:
    """Per-cell |nucleus_centroid - cell_centroid| in µm, from the readers."""
    outs = XENIUM_BASE / f"xenium-breast-tumor-{rep_name}" / "outs"
    reader = XeniumDataReader(outs)
    px = reader.PIXEL_SIZE
    cell_c = reader.get_centroids_pixels()                       # (N, 2) px, cells_df order
    nb = pl.read_parquet(outs / "nucleus_boundaries.parquet")
    nuc_c, n_fallback = XeniumDataReader._nucleus_centroids_from_boundaries(
        reader.cells_df, nb, px
    )
    n = len(cell_c)
    d_um = np.hypot(nuc_c[:, 0] - cell_c[:, 0], nuc_c[:, 1] - cell_c[:, 1]) * px
    return {
        "source": f"xenium_{rep_name}",
        "kind": "xenium",
        "n_cells": int(n),
        "fallback_frac": float(n_fallback / max(n, 1)),
        "median_um": float(np.median(d_um)),
        "p25_um": float(np.percentile(d_um, 25)),
        "p75_um": float(np.percentile(d_um, 75)),
        "p90_um": float(np.percentile(d_um, 90)),
        "max_um": float(d_um.max()),
        "frac_gt_2_7um": float((d_um > 2.7).mean()),
        "frac_gt_4_8um": float((d_um > 4.8).mean()),
    }


def sthelar_placebo(slide_zarr: Path) -> dict:
    """STHELAR is nucleus-centered in BOTH builds -> displacement 0 by construction."""
    slide = "sthelar_" + slide_zarr.name.replace("sdata_", "").replace(".zarr", "")
    return {
        "source": slide, "kind": "sthelar_placebo", "n_cells": 0,
        "fallback_frac": 0.0, "median_um": 0.0, "p25_um": 0.0, "p75_um": 0.0,
        "p90_um": 0.0, "max_um": 0.0, "frac_gt_2_7um": 0.0, "frac_gt_4_8um": 0.0,
    }


def _gate(r: dict) -> bool:
    if r["kind"] == "sthelar_placebo":
        return r["median_um"] == 0.0
    return (MED_LO <= r["median_um"] <= MED_HI) and (r["fallback_frac"] < FALLBACK_MAX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for rep in ("rep1", "rep2"):
        if (XENIUM_BASE / f"xenium-breast-tumor-{rep}" / "outs").exists():
            rows.append(audit_xenium(rep))
        else:
            logger.warning(f"skipping {rep}: outs not found")
    for slide_zarr in sorted(STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        if slide_zarr.is_dir():
            rows.append(sthelar_placebo(slide_zarr))

    df = pl.DataFrame(rows)
    df.write_parquet(args.out / "audit_summary.parquet")

    md = ["# Re-centering audit — breast pool (spec §5)", "",
          f"Xenium gates: median Δ ∈ [{MED_LO}, {MED_HI}] µm AND fallback < "
          f"{FALLBACK_MAX:.0%}. STHELAR: Δ = 0 (placebo, not re-centered).", "",
          "| source | n_cells | median Δµm | p75 | p90 | max | >2.7µm | >4.8µm | fallback | gate |",
          "|" + "---|" * 10]
    pass_all = True
    for r in df.iter_rows(named=True):
        ok = _gate(r)
        pass_all &= ok
        md.append(
            f"| {r['source']} | {r['n_cells']:,} | {r['median_um']:.3f} | "
            f"{r['p75_um']:.2f} | {r['p90_um']:.2f} | {r['max_um']:.2f} | "
            f"{r['frac_gt_2_7um']:.1%} | {r['frac_gt_4_8um']:.1%} | "
            f"{r['fallback_frac']:.1%} | {'✅' if ok else '❌'} |")
    md += ["", f"**Overall**: {'PASS' if pass_all else 'FAIL'}"]
    (args.out / "audit_summary.md").write_text("\n".join(md) + "\n")
    logger.info(f"wrote audit_summary.{{parquet,md}} to {args.out}  ->  "
                f"{'PASS' if pass_all else 'FAIL'}")
    for r in df.iter_rows(named=True):
        logger.info(f"  {r['source']}: median={r['median_um']:.3f}µm  "
                    f"fallback={r['fallback_frac']:.1%}  gate={'PASS' if _gate(r) else 'FAIL'}")
    if not pass_all:
        raise SystemExit("re-centering audit FAILED — fix before building/training")


if __name__ == "__main__":
    main()
