"""Score-ladder montages to VALIDATE the QC scorer (not just surface trash).

The existing per-class montages (dapidl.qc.montage.build_class_montage) show the
worst-scoring patches — good for "is there garbage", useless for "does the score
track quality". This builds the complementary view: patches sampled across the
whole score range, laid out worst (top) → best (bottom). If the scorer works you
see a clean visual gradient; if rows look interchangeable, the score is noise.

Produces a global ladder plus one per source (QC is normalized per slide, and
Prime/s6 sits lowest, so per-source is the honest check).

    uv run python scripts/qc_validation_montage.py
    uv run python scripts/qc_validation_montage.py --metric focus_score --bands 10
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402

from dapidl.pipeline.steps.quality_control import _slide_groups  # noqa: E402
from dapidl.qc.io import read_patches  # noqa: E402

DEFAULT_DATASET = Path.home() / "datasets/derived/breast-6source-dapi-p128"


def select_ladder(metric: np.ndarray, subset: np.ndarray, bands: int,
                  per_band: int, rng: np.random.Generator):
    """Quantile-bin `subset` by `metric`, sample `per_band` per band.

    Returns a list of (label, abs_indices, scores) ordered worst → best, where
    abs_indices are absolute patch indices (so read_patches/metric indexing work).
    """
    vals = metric[subset]
    edges = np.quantile(vals, np.linspace(0, 1, bands + 1))
    out = []
    for b in range(bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (vals >= lo) & (vals <= hi) if b == bands - 1 else (vals >= lo) & (vals < hi)
        band_idx = subset[mask]
        if len(band_idx) == 0:
            continue
        sel = rng.choice(band_idx, size=min(per_band, len(band_idx)), replace=False)
        out.append((f"{lo:.2f}–{hi:.2f}\n(n={len(band_idx):,})", sel, metric[sel]))
    return out


def render_ladder(dataset_path: Path, bands_data, title: str, out_path: Path,
                  per_band: int) -> None:
    rows = len(bands_data)
    fig, axes = plt.subplots(rows, per_band, figsize=(per_band * 1.3, rows * 1.5))
    axes = np.atleast_2d(axes)
    for r, (label, sel, sc) in enumerate(bands_data):
        patches = read_patches(dataset_path, sel)
        for c in range(per_band):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c < len(sel):
                p = patches[c].astype(np.float32)
                lo, hi = np.percentile(p, [1, 99])
                ax.imshow(np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
                ax.set_title(f"{sc[c]:.2f}", fontsize=6, pad=1)
            if c == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"{title}\ntop = lowest QC  →  bottom = highest QC", fontsize=11)
    fig.tight_layout(rect=(0.06, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--metric", default="qc_score",
                    choices=["qc_score", "focus_score", "detection_score",
                             "structure_score", "centeredness", "objectness_score"])
    ap.add_argument("--scores", default="qc/qc_scores.parquet",
                    help="sidecar relative to dataset (qc/qc_scores.parquet or qc/seg_scores.parquet)")
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--per-band", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    qc = pl.read_parquet(args.dataset / args.scores).sort("cell_id")
    metric = qc[args.metric].to_numpy()
    n = len(metric)
    sources = _slide_groups(args.dataset, n)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out or (args.dataset / "qc" / "ladders")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global ladder over all patches.
    all_idx = np.arange(n)
    bands_data = select_ladder(metric, all_idx, args.bands, args.per_band, rng)
    render_ladder(args.dataset, bands_data, f"{args.metric} — ALL sources (n={n:,})",
                  out_dir / f"ladder_{args.metric}_ALL.png", args.per_band)

    # Per-source ladders (QC is per-slide; Prime/s6 is the source of interest).
    for src in sorted(set(sources.tolist())):
        subset = np.where(sources == src)[0]
        bands_data = select_ladder(metric, subset, args.bands, args.per_band, rng)
        render_ladder(args.dataset, bands_data,
                      f"{args.metric} — {src} (n={len(subset):,})",
                      out_dir / f"ladder_{args.metric}_{src}.png", args.per_band)
        lo, hi = float(metric[subset].min()), float(metric[subset].max())
        logger.info(f"{src:22s} n={len(subset):>7,}  {args.metric} "
                    f"min={lo:.3f} mean={metric[subset].mean():.3f} max={hi:.3f}")


if __name__ == "__main__":
    main()
