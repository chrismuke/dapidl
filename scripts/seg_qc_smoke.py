"""Smoke for the segmentation-grounded QC scorer: score a stratified per-source
sample, write a sidecar, print per-source/per-reason/audit stats, and build
score-ladders (structure + objectness) + per-reason montages for visual review.

Maps band-selected rows back through cell_id so read_patches reads the correct
LMDB rows (the sample is NOT 1:1 with the full dataset index).

    uv run python scripts/seg_qc_smoke.py --per-source 4000
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402

from dapidl.pipeline.steps.quality_control import _load_patch_labels, _slide_groups  # noqa: E402
from dapidl.pipeline.steps.quality_control_seg import stratified_audit  # noqa: E402
from dapidl.qc.io import read_patches  # noqa: E402
from dapidl.qc.montage import build_reason_montage  # noqa: E402
from starpose.qc import (  # noqa: E402
    SegmentationGroundedScorer,
    decide_broken,
)

sys.path.insert(0, str(Path(__file__).parent))
from qc_validation_montage import render_ladder  # noqa: E402

DEFAULT_DATASET = Path.home() / "datasets/derived/breast-6source-dapi-p128"


def smoke_ladder(dataset, df, axis, src, out_path, bands=8, per_band=10, seed=42):
    """Score-ladder for `axis` over rows of `src` (or ALL); reads patches by cell_id."""
    sub = df if src == "ALL" else df.filter(pl.col("source") == src)
    vals = sub[axis].to_numpy()
    cids = sub["cell_id"].to_numpy()
    if len(vals) < bands:
        return
    edges = np.quantile(vals, np.linspace(0, 1, bands + 1))
    rng = np.random.default_rng(seed)
    bands_data = []
    for b in range(bands):
        lo, hi = edges[b], edges[b + 1]
        m = (vals >= lo) & (vals <= hi) if b == bands - 1 else (vals >= lo) & (vals < hi)
        bidx = np.where(m)[0]
        if len(bidx) == 0:
            continue
        pick = rng.choice(bidx, size=min(per_band, len(bidx)), replace=False)
        bands_data.append((f"{lo:.2f}–{hi:.2f}\n(n={len(bidx):,})",
                           cids[pick].astype(int), vals[pick]))
    if bands_data:
        render_ladder(dataset, bands_data, f"{axis} — {src} (n={len(vals):,})",
                      out_path, per_band)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--per-source", type=int, default=4000)
    ap.add_argument("--ref-n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n, cell_ids, class_names = _load_patch_labels(args.dataset)
    cell_ids = np.asarray(cell_ids)
    sources = _slide_groups(args.dataset, n)
    scorer = SegmentationGroundedScorer()
    rng = np.random.default_rng(args.seed)

    rows = []
    for slide in sorted(set(sources.tolist())):
        idx = np.where(sources == slide)[0]
        sel = idx if len(idx) <= args.per_source else rng.choice(idx, args.per_source, replace=False)
        sel = np.sort(sel)
        ref = scorer.fit_reference(read_patches(args.dataset, sel[: args.ref_n]))
        for start in range(0, len(sel), 1000):
            chunk = sel[start : start + 1000]
            scores = scorer.score_batch(read_patches(args.dataset, chunk), ref=ref)
            for j, gi in enumerate(chunk):
                s = scores[j]
                b, r = decide_broken(s, scorer.cfg)
                rows.append(dict(
                    cell_id=int(cell_ids[gi]), source=slide, cell_type=str(class_names[gi]),
                    structure_score=float(s.focus_score), objectness_score=float(s.detection_score),
                    centeredness=float(s.metrics.get("centeredness", 0.0)),
                    area_um2=float(s.metrics.get("area_um2", 0.0)),
                    stardist_prob=float(s.metrics.get("stardist_prob", 0.0)),
                    broken=b, broken_reason=r))
        logger.info(f"{slide}: scored {len(sel)} patches")

    df = pl.DataFrame(rows)
    out_dir = args.dataset / "qc"
    out_dir.mkdir(exist_ok=True)
    df.write_parquet(out_dir / "seg_scores_smoke.parquet")
    logger.info(f"wrote {out_dir/'seg_scores_smoke.parquet'} ({df.height} rows)")

    with pl.Config(tbl_rows=30):
        print("\n=== broken-rate per source ===")
        print(df.group_by("source").agg(pl.len().alias("n"),
              pl.col("broken").mean().round(3).alias("broken_rate")).sort("source"))
        print("\n=== broken_reason counts ===")
        print(df.group_by("broken_reason").agg(pl.len().alias("n")).sort("n", descending=True))
        print("\n=== stratified audit: highest broken-rate (source x class x size) ===")
        print(stratified_audit(df).filter(pl.col("n") >= 20).sort("broken_rate", descending=True).head(15))

    # Ladders for the two key axes (the visual validation the classical scorer failed).
    lad = out_dir / "ladders_seg_smoke"
    lad.mkdir(exist_ok=True)
    for axis in ("structure_score", "objectness_score"):
        smoke_ladder(args.dataset, df, axis, "ALL", lad / f"ladder_{axis}_ALL.png")
        for src in sorted(set(df["source"].to_list())):
            smoke_ladder(args.dataset, df, axis, src, lad / f"ladder_{axis}_{src}.png")

    # Per-reason montages (read up to 64 broken patches per reason by cell_id).
    mon = out_dir / "broken_montages_smoke"
    mon.mkdir(exist_ok=True)
    for reason in sorted(set(df.filter(pl.col("broken"))["broken_reason"].to_list())):
        rsub = df.filter(pl.col("broken_reason") == reason).head(64)
        patches = read_patches(args.dataset, rsub["cell_id"].to_numpy().astype(int))
        reasons = np.array([reason] * patches.shape[0], dtype=object)
        img = build_reason_montage(patches, reasons, reason=reason, top_n=64)
        mpimg.imsave(mon / f"broken_{reason}.png", img)
    logger.info(f"wrote ladders to {lad} and montages to {mon}")


if __name__ == "__main__":
    main()
