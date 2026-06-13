"""QC embedding-anomaly score driver.

LOSO anomaly scoring for one or more frozen embedders (DINOv2 / NuSPIRe), with
AUROC sanity + a HARD per-class fairness gate + embedder selection, and a
"missed-break" montage (classical-good but high-anomaly crops).

Usage (after embeddings are cached by dapidl.qc.embeddings.compute_embeddings):
    uv run python scripts/qc_anomaly_score.py --models dinov2_vitb14 nuspire --eval
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import lmdb
import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dapidl.qc.anomaly import score_all_slides_loso  # noqa: E402
from dapidl.qc.anomaly_eval import (  # noqa: E402
    auroc,
    fairness_pass,
    per_class_anomaly,
    select_embedder,
)

DSET = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1")
GOOD_NOT_BROKEN = ["Good", "Weak-passing"]


def _pct_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    if m.sum() > 1:
        out[m] = 100.0 * x[m].argsort().argsort() / (m.sum() - 1)
    return out


def _read_patch(txn, idx, ps=64):
    v = txn.get(struct.pack(">Q", int(idx)))
    if v is None:
        return np.zeros((ps, ps), dtype=np.uint16)
    return np.frombuffer(v[8:], dtype=np.uint16).reshape(ps, ps)


def _stretch(p):
    p = p.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    hi = hi if hi > lo else lo + 1.0
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0)


def missed_break_montage(df: pl.DataFrame, lmdb_dir: Path, out_png: Path, top_k: int = 49) -> int:
    cand = (df.filter(pl.col("grade").is_in(GOOD_NOT_BROKEN))
              .sort("anomaly_pct", descending=True, nulls_last=True).head(top_k))
    rows = cand["row_idx"].to_list()
    if not rows:
        return 0
    cols = math.ceil(math.sqrt(len(rows)))
    rws = math.ceil(len(rows) / cols)
    env = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False, readahead=False)
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 1.4, rws * 1.5))
    axes = np.atleast_1d(axes).ravel()
    with env.begin() as txn:
        for ax, ri, g, pct in zip(axes, rows, cand["grade"], cand["anomaly_pct"], strict=False):
            ax.imshow(_stretch(_read_patch(txn, ri)), cmap="gray")
            ax.set_title(f"{ri}·{g[:4]}·{pct:.0f}", fontsize=5)
            ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")
    env.close()
    fig.suptitle("classical-good but high-anomaly (candidate missed breaks)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    cand.select(["row_idx", "slide", "coarse_idx", "grade", "anomaly_score", "anomaly_pct"]).write_csv(
        out_png.with_suffix(".csv"))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", type=Path, default=DSET)
    ap.add_argument("--models", nargs="+", default=["dinov2_vitb14", "nuspire"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--per-class-cap", type=int, default=1500)
    ap.add_argument("--coreset-frac", type=float, default=1.0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    qc = a.lmdb / "qc"
    out = a.out or qc / "seg_scores_anom.parquet"
    df = pl.read_parquet(qc / "seg_scores.parquet")
    rows = df["row_idx"].to_numpy()
    slides = df["slide"].to_numpy()
    coarse = df["coarse_idx"].to_numpy()
    broken = df["broken"].to_numpy()
    grades = df["grade"].to_numpy()
    idx2name = {v: k for k, v in json.loads((a.lmdb / "class_mapping.json").read_text()).items()}
    endo_idx = {v: k for k, v in idx2name.items()}["Endothelial"]

    results = {}
    for m in a.models:
        emb = np.load(qc / f"embeddings_{m}.npy").astype(np.float32)
        erows = np.load(qc / f"embeddings_{m}_rows.npy")
        if not np.array_equal(erows, rows):
            raise SystemExit(f"[{m}] embedding rows do not match seg_scores order")
        rng = np.random.default_rng(0)
        score = score_all_slides_loso(emb, rows, slides, coarse, broken, grades,
                                      k=a.k, per_class_cap=a.per_class_cap, rng=rng,
                                      coreset_frac=a.coreset_frac)
        df = df.with_columns([pl.Series(f"anomaly_score_{m}", score),
                              pl.Series(f"anomaly_pct_{m}", _pct_rank(score))])
        if a.eval:
            au = auroc(score, broken)
            table = per_class_anomaly(score, coarse, broken, endo_idx=endo_idx)
            fp = fairness_pass(table, endo_idx=endo_idx)
            results[m] = {"auroc": au, "fairness_pass": fp}
            print(f"\n=== {m} ===  AUROC(anomaly vs classical broken) = {au:.3f}", flush=True)
            print("  per-class mean anomaly (non-broken):")
            for c, (mean, med) in sorted(table.items()):
                star = "  <-- Endothelial" if c == endo_idx else ""
                print(f"    {idx2name.get(c, c):12s} mean={mean:.4f} median={med:.4f}{star}")
            print(f"  fairness_pass = {fp}", flush=True)

    if a.eval:
        winner = select_embedder(results)
        print(f"\n=== embedder selection ===\n  results: {results}\n  WINNER: {winner}", flush=True)
        if winner is None:
            print("  *** BOTH EMBEDDERS FAILED the fairness gate — no canonical anomaly_score "
                  "written; fall back to classical-only. ***", flush=True)
        else:
            df = df.with_columns([pl.col(f"anomaly_score_{winner}").alias("anomaly_score"),
                                  pl.col(f"anomaly_pct_{winner}").alias("anomaly_pct")])
            n = missed_break_montage(df, a.lmdb, qc / "anomaly_missed_breaks.png")
            print(f"  canonical anomaly_score = {winner}; missed-break montage: {n} crops -> "
                  f"{qc / 'anomaly_missed_breaks.png'}", flush=True)

    df.write_parquet(out)
    print(f"\n[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
