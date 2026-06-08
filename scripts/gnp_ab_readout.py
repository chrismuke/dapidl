"""Paired A/B readout for gnp-v1 (review 2026-05-29, Phase 2).

Consumes per-arm ``preds.parquet`` (columns: source, cell_id, y_true, y_pred —
written by breast_pooled_train.py) and emits a defensible readout instead of an
eyeballed point delta:

- pairs the two arms on the SHARED (source, cell_id) cells (so the comparison is
  truly paired; reports any non-overlap),
- paired McNemar p-value + bootstrap 95% CI on macroF1(B) − macroF1(A), per
  source and pooled over the Xenium sources (the decisional subset),
- per-arm per-class F1 with bootstrap CIs + support (so a rare-class move is read
  with its uncertainty),
- optional multi-seed mean±SD per arm (pass several --preds-* files).

The paired stats use the FIRST preds file of each arm; extra files are treated as
additional seeds for the variance estimate only.

Usage:
    uv run python scripts/gnp_ab_readout.py \
        --name-a M_cell      --preds-a runs/cell/preds.parquet \
        --name-b M_nuc_full  --preds-b runs/nuc/preds.parquet runs/nuc_s1/preds.parquet \
        --class-names Endothelial,Epithelial,Immune,Stromal \
        --xenium-prefix xenium_ --out runs/gnp_centering_readout.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dapidl.evaluation.ab_stats import (  # noqa: E402
    bootstrap_macro_f1_diff,
    macro_f1_fast,
    mcnemar,
    per_class_f1_ci,
)


def _load(paths: list[Path]) -> list[pl.DataFrame]:
    out = []
    for p in paths:
        df = pl.read_parquet(p).with_columns(pl.col("cell_id").cast(pl.Utf8))
        need = {"source", "cell_id", "y_true", "y_pred"}
        if not need.issubset(df.columns):
            raise SystemExit(f"{p} missing columns {need - set(df.columns)}")
        out.append(df)
    return out


def _paired(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Inner-join two arms on (source, cell_id); keep A's y_true as ground truth
    and B's y_true alongside for a label-consistency check (same frame = aligned)."""
    j = a.join(
        b.select("source", "cell_id",
                 pl.col("y_pred").alias("y_pred_b"),
                 pl.col("y_true").alias("y_true_b")),
        on=["source", "cell_id"], how="inner")
    return j.rename({"y_pred": "y_pred_a"})


def _subset_stats(sub: pl.DataFrame, k, n_boot, seed) -> dict:
    yt = sub["y_true"].to_numpy()
    pa = sub["y_pred_a"].to_numpy()
    pb = sub["y_pred_b"].to_numpy()
    mc = mcnemar(yt, pa, pb)
    bs = bootstrap_macro_f1_diff(yt, pa, pb, k, n_boot=n_boot, seed=seed)
    return {
        "n": len(sub),
        "f1_a": macro_f1_fast(yt, pa, k), "f1_b": macro_f1_fast(yt, pb, k),
        "diff": bs["diff"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
        "sig": bs["p_excludes_zero"], "mcnemar_p": mc.p_value,
        "n01_b_wins": mc.n01, "n10_a_wins": mc.n10,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name-a", required=True)
    ap.add_argument("--name-b", required=True)
    ap.add_argument("--preds-a", required=True, nargs="+", type=Path)
    ap.add_argument("--preds-b", required=True, nargs="+", type=Path)
    ap.add_argument("--class-names", default=None,
                    help="comma-separated; default = class indices")
    ap.add_argument("--xenium-prefix", default="xenium_",
                    help="sources with this prefix form the decisional 'pooled-Xenium' subset")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a_files = _load(args.preds_a)
    b_files = _load(args.preds_b)
    a0, b0 = a_files[0], b_files[0]

    k = int(max(int(a0["y_true"].max()), int(a0["y_pred"].max()),
                int(b0["y_pred"].max())) + 1)
    names = args.class_names.split(",") if args.class_names else [str(i) for i in range(k)]
    if len(names) < k:
        names += [str(i) for i in range(len(names), k)]

    paired = _paired(a0, b0)
    n_a, n_b, n_shared = len(a0), len(b0), len(paired)
    # label-consistency check on shared cells (aligned within the joined frame)
    mism = int((paired["y_true"] != paired["y_true_b"]).sum())

    xen = paired.filter(pl.col("source").str.starts_with(args.xenium_prefix))
    subsets = [("POOLED-Xenium", xen)] + [
        (s, paired.filter(pl.col("source") == s))
        for s in sorted(paired["source"].unique().to_list())
    ]

    md = [f"# gnp-v1 A/B readout — {args.name_a} vs {args.name_b}", "",
          f"Arms: **A = {args.name_a}**, **B = {args.name_b}**. Positive Δ = B better. "
          f"`sig` = bootstrap 95% CI of Δ excludes 0. McNemar = paired test on "
          f"discordant cells. K={k} classes; {args.n_boot} bootstrap resamples.", ""]
    md.append(f"Pairing: A has {n_a:,} test cells, B has {n_b:,}; **{n_shared:,} shared** "
              f"(by source,cell_id). " + (f"⚠ {mism:,} shared cells have mismatched y_true "
              "(label drift!)" if mism else "y_true consistent on shared cells. ✓"))
    if len(a_files) > 1 or len(b_files) > 1:
        def _seed_spread(name, files):
            vals = [macro_f1_fast(f.filter(pl.col("source").str.starts_with(args.xenium_prefix))["y_true"].to_numpy(),
                                  f.filter(pl.col("source").str.starts_with(args.xenium_prefix))["y_pred"].to_numpy(), k)
                    for f in files]
            return f"{name}: Xenium macroF1 {np.mean(vals):.4f} ± {np.std(vals):.4f} (n_seeds={len(vals)}; {[round(v,4) for v in vals]})"
        md += ["", "**Multi-seed variance (the real noise floor):**",
               "- " + _seed_spread(args.name_a, a_files),
               "- " + _seed_spread(args.name_b, b_files)]

    md += ["", "## Paired comparison (per source + pooled-Xenium)", "",
           "| subset | n | F1(A) | F1(B) | Δ (B−A) | 95% CI | sig | McNemar p | B wins / A wins |",
           "|" + "---|" * 9]
    for label, sub in subsets:
        if len(sub) == 0:
            continue
        st = _subset_stats(sub, k, args.n_boot, args.seed)
        md.append(
            f"| {label} | {st['n']:,} | {st['f1_a']:.4f} | {st['f1_b']:.4f} | "
            f"{st['diff']:+.4f} | [{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}] | "
            f"{'✅' if st['sig'] else '—'} | {st['mcnemar_p']:.2e} | "
            f"{st['n01_b_wins']:,} / {st['n10_a_wins']:,} |")

    # per-arm per-class F1 CIs on pooled-Xenium
    md += ["", "## Per-class F1 (pooled-Xenium, bootstrap 95% CI)", ""]
    for nm, col, frame in [(args.name_a, "y_pred_a", xen), (args.name_b, "y_pred_b", xen)]:
        if len(frame) == 0:
            continue
        rows = per_class_f1_ci(frame["y_true"].to_numpy(), frame[col].to_numpy(), k,
                               class_names=names, n_boot=args.n_boot, seed=args.seed)
        md.append(f"**{nm}**")
        md.append("| class | F1 | 95% CI | support |")
        md.append("|" + "---|" * 4)
        for r in rows:
            md.append(f"| {r['class_name']} | {r['f1']:.3f} | "
                      f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['support']:,} |")
        md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    print(f"wrote {args.out}  ({n_shared:,} shared cells, K={k})")
    for label, sub in subsets[:1]:
        if len(sub):
            st = _subset_stats(sub, k, args.n_boot, args.seed)
            print(f"  {label}: Δ={st['diff']:+.4f} CI=[{st['ci_lo']:+.4f},{st['ci_hi']:+.4f}] "
                  f"sig={st['sig']} McNemar_p={st['mcnemar_p']:.2e}")


if __name__ == "__main__":
    main()
