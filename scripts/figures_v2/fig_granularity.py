"""Figure 1 · Granularity ladder — what DAPI predicts at each CL tier.

Two panels:
A. Macro F1 vs CL tier depth (COARSE 5 → MEDIUM 12 → FINE 18).
   Each tier label tells the reader exactly which CL hierarchy level it is,
   not an ad-hoc class count.
B. Per-class F1 from the Xenium 4-class breast model (the canonical
   "what works, what doesn't" view).

Tier labels match `dapidl.ontology.training_tiers`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import cell_color, apply_style
from dapidl.ontology.training_tiers import (
    COARSE_NAMES, MEDIUM_NAMES, FINE_NAMES,
)


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig01_granularity.png"


def main() -> None:
    apply_style()
    metrics = pl.read_parquet(ROOT / "model_eval_2026_05" / "master_metrics.parquet")

    # --- Panel A: F1 vs CL tier depth ----------------------------------
    # Each entry: (n_classes, F1, label, source_run)
    runs = []

    # COARSE 4 — Xenium breast 4-class (matches CL Super-Coarse minus Neural)
    p128 = json.loads((ROOT / "breast_dapi_p128" / "analysis" / "summary.json").read_text())
    runs.append((4, p128.get("test_macro_f1", p128.get("macro_f1")),
                 "COARSE\n(CL Super-Coarse)", "Xenium breast"))

    # MEDIUM-ish 7 — STHELAR breast 7-class (between coarse and medium)
    row7 = metrics.filter(pl.col("run") == "sthelar_exp5_7class").row(0, named=True)
    runs.append((7, row7["test_macro_f1"], "MEDIUM partial\n(7-class breast)",
                 "STHELAR breast"))

    # MEDIUM 9 — STHELAR multi-tissue (CL Coarse / data-pruned)
    row9 = metrics.filter(pl.col("run") == "sthelar_modality_dapi").row(0, named=True)
    runs.append((9, row9["test_macro_f1"], "MEDIUM\n(CL Coarse, multi-tissue)",
                 "STHELAR 16 tissues"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5))

    xs = [r[0] for r in runs]
    ys = [r[1] for r in runs]
    tier_labels = [r[2] for r in runs]
    src_labels = [r[3] for r in runs]
    colors = ["#3D5A80", "#7B5EA7", "#A04D9D"]

    ax1.plot(xs, ys, "-", lw=2.0, color="#999", zorder=1)
    for x, y, tl, sl, c in zip(xs, ys, tier_labels, src_labels, colors):
        ax1.scatter(x, y, s=200, color=c, zorder=3, edgecolor="white", lw=1.5,
                    label=f"{x}-class · {sl}")
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=11, fontweight="bold")

    # Tier annotations along the top axis
    for x, tl in zip(xs, tier_labels):
        ax1.annotate(tl, xy=(x, 0.83), xycoords=("data", "axes fraction"),
                     ha="center", va="top", fontsize=8.5, color="#555",
                     style="italic")

    ax1.set_xticks(xs)
    ax1.set_xlabel("number of classes (CL tier depth)")
    ax1.set_ylabel("test macro F1")
    ax1.set_ylim(0.30, 0.85)
    ax1.set_title("A. Granularity → F1: deeper CL tier = lower ceiling",
                  loc="left", fontsize=12)
    ax1.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)

    # Inset note about FINE tier
    ax1.text(0.5, 0.05,
             "FINE (~17 classes, CL Medium L3+L4 + 2 pathology) not yet trained\n"
             "→ next experiment: pooled breast 4-class is the COARSE baseline",
             transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=8.5, color="#888", style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                       edgecolor="#CCC", lw=0.8))

    # --- Panel B: per-class F1 from the Xenium 4-class model -----------
    pcm = json.loads((ROOT / "breast_dapi_p128" / "analysis" /
                      "per_class_metrics.json").read_text())
    per = pcm.get("per_class", pcm)
    items = sorted(per.items(), key=lambda kv: kv[1].get("f1", 0.0), reverse=True)
    names = [k for k, _ in items]
    f1s = [v.get("f1", 0.0) for _, v in items]
    sup = [v.get("support", 0) for _, v in items]
    colors_b = [cell_color(n) for n in names]
    y = np.arange(len(names))
    bars = ax2.barh(y, f1s, color=colors_b, edgecolor="white", lw=1.2)
    for bar, f, s in zip(bars, f1s, sup):
        ax2.text(f + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{f:.2f}  ({s/1000:.0f}k cells)", va="center", fontsize=10)
    ax2.set_yticks(y, names)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("test F1 per class (CL Super-Coarse, 4 classes)")
    ax2.set_title("B. Per-class F1 — Xenium breast COARSE", loc="left", fontsize=12)

    fig.suptitle(
        "Granularity ladder — DAPI predicts CL Super-Coarse well, drops with depth",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(0.99, 0.005,
             f"Tiers from `dapidl.ontology.training_tiers`: COARSE={len(COARSE_NAMES)} · "
             f"MEDIUM={len(MEDIUM_NAMES)} · FINE={len(FINE_NAMES)}.   "
             "All runs use EfficientNetV2-S, 128 px patches.",
             ha="right", va="bottom", fontsize=8.5, color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
