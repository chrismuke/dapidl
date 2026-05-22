"""Segmentation-grounded QC pass over a built DAPIDL dataset.

Mirrors quality_control.py: per-slide reference, chunked scoring, sidecar write.
Adds a stratified broken-rate audit (source x cell_type x size bin) as the
anti-censoring guardrail. metadata.parquet is never modified.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.steps.quality_control import (
    _atomic_write_parquet,
    _load_patch_labels,
    _slide_groups,
)
from dapidl.qc.io import read_patches
from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer, decide_broken

REFERENCE_SAMPLE = 2000


def stratified_audit(df: pl.DataFrame, n_size_bins: int = 4) -> pl.DataFrame:
    """Broken-rate by source x cell_type x size bin (the censoring guardrail).

    qcut WITHOUT explicit labels avoids a label/bin-count mismatch when area is
    near-constant or has few unique values; bins are cast to string for grouping.
    """
    df = df.with_columns(
        pl.col("area_um2").qcut(n_size_bins, allow_duplicates=True)
        .cast(pl.Utf8).alias("size_bin")
    )
    return (df.group_by(["source", "cell_type", "size_bin"])
              .agg(pl.len().alias("n"),
                   pl.col("broken").cast(pl.Float64).mean().alias("broken_rate"))
              .sort(["source", "cell_type", "size_bin"]))


def run_quality_control_seg(dataset_path, use_structure_cut: bool = False,
                            seed: int = 42) -> Path:
    dataset_path = Path(dataset_path)
    n, cell_ids, class_names = _load_patch_labels(dataset_path)
    sources = _slide_groups(dataset_path, n)
    scorer = SegmentationGroundedScorer()
    rng = np.random.default_rng(seed)

    cols = {k: np.zeros(n) for k in (
        "structure_score", "centeredness", "dominant_central", "completeness",
        "area_um2", "stardist_prob", "eccentricity", "solidity", "intensity_ratio")}
    broken = np.zeros(n, dtype=bool)
    reason = np.empty(n, dtype=object)

    for slide in sorted(set(sources.tolist())):
        idx = np.where(sources == slide)[0]
        sample = idx if len(idx) <= REFERENCE_SAMPLE else rng.choice(idx, REFERENCE_SAMPLE, replace=False)
        ref = scorer.fit_reference(read_patches(dataset_path, sample))
        for start in range(0, len(idx), 1000):
            chunk = idx[start:start + 1000]
            scores = scorer.score_batch(read_patches(dataset_path, chunk), ref=ref)
            for j, gi in enumerate(chunk):
                s = scores[j]
                cols["structure_score"][gi] = s.focus_score
                cols["stardist_prob"][gi] = s.metrics.get("stardist_prob", 0.0)
                for k in ("centeredness", "dominant_central", "completeness",
                          "area_um2", "eccentricity", "solidity", "intensity_ratio"):
                    cols[k][gi] = s.metrics.get(k, 0.0)
                b, r = decide_broken(s, scorer.cfg, use_structure_cut=use_structure_cut)
                broken[gi] = b
                reason[gi] = r
        logger.info(f"seg-QC scored slide {slide}: {len(idx)} patches")

    out_dir = dataset_path / "qc"
    out_dir.mkdir(exist_ok=True)
    df = pl.DataFrame({"cell_id": cell_ids, "source": sources,
                       "cell_type": class_names, **cols,
                       "broken": broken, "broken_reason": list(reason)})
    _atomic_write_parquet(df, out_dir / "seg_scores.parquet")
    audit = stratified_audit(df)
    _atomic_write_parquet(audit, out_dir / "seg_broken_audit.parquet")
    (out_dir / "seg_scores.meta.json").write_text(json.dumps({
        "scorer": scorer.name, "cfg": scorer.cfg.__dict__,
        "use_structure_cut": use_structure_cut,
        "broken_rate": float(broken.mean()), "date": date.today().isoformat()}, indent=2))
    logger.info(f"wrote {out_dir/'seg_scores.parquet'} (broken {broken.mean():.1%})")
    return out_dir
