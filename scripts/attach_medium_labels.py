#!/usr/bin/env python3
"""Augment a pilot LMDB's patch_registry with medium-granularity labels.

Per source:
- STHELAR slides (sthelar_breast_s*) -> read `combined_final_label` (10-class
  expert-curated consensus) from /mnt/work/datasets/STHELAR/summary_all_labels_
  per_slide/breast_s*_cell_metadata.parquet, joined by cell_id.
- Xenium reps (xenium_rep*_nuc) -> use Janesick17 supervised label directly
  (the same xlsx the coarse label was derived from).

Rows whose cell_id has no medium label get medium_label = "Unknown".

Output: same patch_registry.parquet, augmented with a `medium_label` column.

Usage:
    uv run python scripts/attach_medium_labels.py \
        --dataset ~/datasets/derived/breast-pilot-6source-dapi-p128-nuc-v3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
STHELAR_LABELS_DIR = Path("/mnt/work/datasets/STHELAR/summary_all_labels_per_slide")
GT_XLSX = XENIUM_BASE / "xenium-breast-tumor-rep1" / "Cell_Barcode_Type_Matrices.xlsx"

REP_SHEETS = {
    "xenium_rep1_nuc": "Xenium R1 Fig1-5 (supervised)",
    "xenium_rep2_nuc": "Xenium R2 Fig1-5 (supervised)",
}


def _load_xenium_medium(slide: str) -> dict[str, str]:
    """Janesick17 fine label per barcode (= our 'medium' for Xenium)."""
    import pandas as pd
    sheet = REP_SHEETS[slide]
    df = pd.read_excel(GT_XLSX, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    label_col = "Cluster" if "Cluster" in df.columns else "Annotation"
    return dict(zip(df["Barcode"].astype(str), df[label_col].astype(str)))


def _load_sthelar_medium(slide: str) -> dict[str, str]:
    """combined_final_label per cell_id from STHELAR summary metadata."""
    short = slide.replace("sthelar_", "")  # 'breast_s0' etc.
    p = STHELAR_LABELS_DIR / f"{short}_cell_metadata.parquet"
    df = pl.read_parquet(p).select(["cell_id", "combined_final_label"])
    return dict(zip(df["cell_id"].cast(pl.Utf8).to_list(),
                    df["combined_final_label"].cast(pl.Utf8).to_list()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()

    reg_path = args.dataset / "patch_registry.parquet"
    reg = pl.read_parquet(reg_path)
    logger.info(f"registry n={reg.height:,}, slides={sorted(set(reg['slide'].to_list()))}")

    medium_labels: list[str] = []
    for slide in sorted(set(reg["slide"].to_list())):
        if slide.startswith("xenium_"):
            lookup = _load_xenium_medium(slide)
        elif slide.startswith("sthelar_"):
            lookup = _load_sthelar_medium(slide)
        else:
            raise ValueError(f"Unknown slide prefix: {slide}")
        logger.info(f"{slide}: medium-label lookup has {len(lookup):,} cell_ids")
        sub = reg.filter(pl.col("slide") == slide)
        for cid in sub["cell_id"].to_list():
            medium_labels.append(lookup.get(str(cid), "Unknown"))

    # Rebuild in slide-iteration order: re-sort to match registry order.
    # (We iterate slides in sorted set order above; do the same now.)
    augmented_rows = []
    pointer = 0
    for slide in sorted(set(reg["slide"].to_list())):
        sub = reg.filter(pl.col("slide") == slide)
        for r in sub.iter_rows(named=True):
            r = dict(r)
            r["medium_label"] = medium_labels[pointer]
            augmented_rows.append(r)
            pointer += 1
    out = pl.DataFrame(augmented_rows).sort("row_idx")
    out.write_parquet(reg_path)
    logger.info(f"wrote {reg_path} (rows={out.height}, medium label counts:)")
    print(out.group_by("medium_label").agg(pl.len().alias("n"))
              .sort("n", descending=True))


if __name__ == "__main__":
    main()
