"""Audit per-slide cell-type distributions across the canonical 3-tier ontology.

For every STHELAR slide and every Janesick replicate, count cells per
(coarse, medium, fine) tier class. Saves a master parquet:

    pipeline_output/granularity_audit/per_slide_counts.parquet
        columns: source, tissue, slide, donor, tier, class_name, count

Then a dependency-free aggregation step builds scenario tables.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl
import zarr
from loguru import logger

from dapidl.ontology.cl_mapper import get_mapper
from dapidl.ontology.training_tiers import derive_tier_label

STHELAR_ROOT = Path("/mnt/work/datasets/STHELAR/sdata_slides")
JANESICK_ROOT = Path("/mnt/work/datasets/raw/xenium")
OUT = Path("/mnt/work/git/dapidl/pipeline_output/granularity_audit")


def _resolve_zarr_root(slide_path: Path) -> Path:
    """STHELAR uses both nested (sdata_X.zarr/sdata_X.zarr) and flat layouts."""
    if (slide_path / "images").exists():
        return slide_path
    inner = slide_path / slide_path.stem
    if inner.exists() and (inner / "images").exists():
        return inner
    inner_zarr = slide_path / f"{slide_path.stem}.zarr"
    if inner_zarr.exists() and (inner_zarr / "images").exists():
        return inner_zarr
    return slide_path


def _read_categorical(node: zarr.Group) -> list[str]:
    cats = node["categories"][:]
    codes = node["codes"][:]
    return [str(cats[c]) if 0 <= c < len(cats) else "Unknown" for c in codes]


def load_sthelar_labels(slide_path: Path) -> list[str]:
    """Pull ct_tangram (preferred) or final_label fallback for one slide."""
    root = _resolve_zarr_root(slide_path)
    z = zarr.open(str(root), mode="r")
    obs = z["tables/table_nuclei/obs"]
    if "ct_tangram" in obs:
        node = obs["ct_tangram"]
        return _read_categorical(node) if isinstance(node, zarr.Group) else [str(v) for v in node[:]]
    if "final_label" in obs:
        node = obs["final_label"]
        return _read_categorical(node) if isinstance(node, zarr.Group) else [str(v) for v in node[:]]
    return []


def load_janesick_labels(rep: int) -> list[str]:
    """Read Janesick GT excel — Cluster column = supervised labels."""
    import pandas as pd
    fname = f"celltypes_ground_truth_rep{rep}_supervised.xlsx"
    path = JANESICK_ROOT / f"xenium-breast-tumor-rep{rep}" / fname
    df = pd.read_excel(path)
    return df["Cluster"].astype(str).tolist()


def aggregate_slide(labels: list[str], mapper) -> dict[str, Counter]:
    """Map each label to all 3 tiers, return tier → Counter(class → count)."""
    tier_counts: dict[str, Counter] = {"coarse": Counter(), "medium": Counter(), "fine": Counter()}
    for raw in labels:
        for tier in tier_counts:
            tier_counts[tier][derive_tier_label(raw, tier, mapper)] += 1
    return tier_counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = get_mapper()

    rows: list[dict] = []

    # --- STHELAR (all 31 slides) ---
    sthelar_dirs = sorted(STHELAR_ROOT.glob("sdata_*.zarr"))
    logger.info(f"Found {len(sthelar_dirs)} STHELAR slides")
    for sd in sthelar_dirs:
        # parse "sdata_breast_s0" → tissue=breast, slide=s0
        stem = sd.stem.replace("sdata_", "")
        parts = stem.rsplit("_", 1)
        tissue = parts[0]
        slide = parts[1] if len(parts) == 2 else "s0"
        try:
            labels = load_sthelar_labels(sd)
        except Exception as e:
            logger.error(f"  {stem}: failed to load — {e}")
            continue
        if not labels:
            logger.warning(f"  {stem}: no labels found")
            continue
        n = len(labels)
        logger.info(f"  {stem}: {n:,} cells, aggregating …")
        tier_counts = aggregate_slide(labels, mapper)
        for tier, ctr in tier_counts.items():
            for cls, cnt in ctr.items():
                rows.append(dict(
                    source="STHELAR", tissue=tissue, slide=slide,
                    donor=f"sthelar_{tissue}_{slide}",
                    tier=tier, class_name=cls, count=cnt,
                ))

    # --- Janesick rep1 + rep2 (same donor) ---
    for rep in (1, 2):
        try:
            labels = load_janesick_labels(rep)
        except Exception as e:
            logger.error(f"Janesick rep{rep}: {e}")
            continue
        n = len(labels)
        logger.info(f"Janesick rep{rep}: {n:,} cells, aggregating …")
        tier_counts = aggregate_slide(labels, mapper)
        for tier, ctr in tier_counts.items():
            for cls, cnt in ctr.items():
                rows.append(dict(
                    source="Janesick", tissue="breast", slide=f"rep{rep}",
                    donor="janesick_patientA",  # rep1+rep2 = same donor
                    tier=tier, class_name=cls, count=cnt,
                ))

    df = pl.DataFrame(rows)
    out = OUT / "per_slide_counts.parquet"
    df.write_parquet(out)
    logger.info(f"Wrote {out}  ({df.height:,} rows)")
    logger.info(f"Total cells across all slides: {df.filter(pl.col('tier')=='coarse')['count'].sum():,}")


if __name__ == "__main__":
    main()
