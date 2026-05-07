"""Annotation method comparison panel — DAPIDL deck companion.

Panels:
A. Per-method per-slide F1 at COARSE-4 (grouped bar by method, x = slide)
B. Same at MEDIUM-12 (granularity drop)
C. Top consensus combinations vs best single method (sorted ranking bar)
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import apply_style

ROOT = Path("/mnt/work/git/dapidl/pipeline_output/annotation_run_2026_05")
OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a3_annotation.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

SLIDES = ["rep1", "rep2", "breast_s0", "breast_s1", "breast_s3", "breast_s6"]
SLIDE_LBL = {
    "rep1": "Janesick\nrep1", "rep2": "Janesick\nrep2",
    "breast_s0": "STHELAR\ns0", "breast_s1": "STHELAR\ns1",
    "breast_s3": "STHELAR\ns3", "breast_s6": "STHELAR\ns6 (Prime)",
}

METHOD_SHORT = {
    "celltypist_noMV_Cells_Adult_Breast": "CellTypist\n(gene-expr)",
    "sctype_custom_default": "scType\n(marker DB)",
    "singler_blueprint": "SingleR\n(ref atlas)",
    "banksy_sctype_l0.5_r1.0": "BANKSY+scType\n(spatial-aware)",
}
METHOD_COLOR = {
    "celltypist_noMV_Cells_Adult_Breast": "#3D5A80",  # navy = ML / gene expr
    "sctype_custom_default": "#BC8D5A",                # warm = marker DB
    "singler_blueprint": "#4A9D9C",                    # teal = ref atlas
    "banksy_sctype_l0.5_r1.0": "#D97757",             # coral = spatial-aware
}


def grouped_bar(ax, df, title: str, ymax: float = 0.9):
    methods = list(METHOD_SHORT.keys())
    n_methods = len(methods)
    x = np.arange(len(SLIDES))
    # Total bar-group width = 0.84, divided across N methods with thin gaps.
    width = 0.84 / n_methods
    for k, m in enumerate(methods):
        vals = [
            float(df.filter((pl.col("slide") == s) & (pl.col("method") == m))["f1_macro"][0])
            if not df.filter((pl.col("slide") == s) & (pl.col("method") == m)).is_empty()
            else 0.0
            for s in SLIDES
        ]
        offset = (k - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=METHOD_COLOR[m], edgecolor="white", lw=1.0,
                      label=METHOD_SHORT[m])
        for bar, v in zip(bars, vals):
            if v > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [SLIDE_LBL[s] for s in SLIDES], fontsize=9.5)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("macro F1")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def main() -> None:
    apply_style()
    coarse = pl.read_parquet(ROOT / "coarse_metrics.parquet")
    medium = pl.read_parquet(ROOT / "medium_metrics.parquet")
    cons = pl.read_parquet(ROOT / "consensus_results.parquet")

    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    grouped_bar(ax1, coarse, "A. COARSE 4-class · per-method per-slide", ymax=0.85)
    ax1.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9, ncol=3)

    grouped_bar(ax2, medium, "B. MEDIUM 12-class · per-method per-slide", ymax=0.45)

    # Panel C: consensus ranking
    cons_coarse = (cons.filter(pl.col("tier") == "coarse")
                   .group_by("combo")
                   .agg(pl.col("f1_macro").mean().alias("mean_f1"))
                   .sort("mean_f1", descending=True))
    # Single methods at coarse
    singles = (coarse.group_by("method")
               .agg(pl.col("f1_macro").mean().alias("mean_f1"))
               .sort("mean_f1", descending=True))
    rows = []
    for m, mean in zip(singles["method"].to_list(), singles["mean_f1"].to_list()):
        rows.append({"label": METHOD_SHORT[m].replace("\n", " "), "mean_f1": float(mean), "kind": "single"})
    for combo, mean in zip(cons_coarse["combo"].to_list(), cons_coarse["mean_f1"].to_list()):
        # Shorten combo name
        short = combo.replace("celltypist_noMV_Cells_Adult_Breast", "CT") \
                     .replace("sctype_custom_default", "scType") \
                     .replace("singler_blueprint", "SR") \
                     .replace("banksy_sctype_l0.5_r1.0", "BANKSY")
        n_methods = combo.count("+") + 1
        rows.append({"label": short, "mean_f1": float(mean),
                     "kind": f"consensus ({n_methods}-way)"})
    df = pl.DataFrame(rows).sort("mean_f1", descending=True)

    KIND_COLOR = {
        "single": "#888",
        "consensus (2-way)": "#6F4C9C",
        "consensus (3-way)": "#3A2F6E",
        "consensus (4-way)": "#1F1448",
    }
    bars = ax3.barh(np.arange(len(df)), df["mean_f1"].to_list(),
                    color=[KIND_COLOR.get(k, "#444") for k in df["kind"].to_list()],
                    edgecolor="white", lw=0.8)
    for i, (lab, v, k) in enumerate(zip(df["label"].to_list(),
                                         df["mean_f1"].to_list(),
                                         df["kind"].to_list())):
        ax3.text(v + 0.005, i, f"  {v:.3f}  ({k})",
                 va="center", fontsize=9.5)
    ax3.set_yticks(np.arange(len(df)), df["label"].to_list(), fontsize=9.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 0.75)
    ax3.set_xlabel("mean macro F1 across 6 slides")
    ax3.set_title("C. COARSE-4 ranking — singles vs confidence-weighted consensus",
                  loc="left", fontsize=12, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)
    ax3.set_axisbelow(True)

    fig.suptitle(
        "Annotation method baseline — defends the labels we trained the DAPI model on",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.text(0.99, 0.005,
             "Ground truth: STHELAR cells_label2 → COARSE/MEDIUM (slides s0-s6); "
             "Janesick supervised 17→4 (rep1/rep2). 4 methods, 1 per family "
             "(gene-expr / marker DB / ref atlas / spatial-aware). "
             "BANKSY missing on s1 (cell-count drift) and s6 (Prime panel subsample). "
             "Consensus = per-cell confidence-weighted vote.",
             ha="right", va="bottom", fontsize=8.5, color="#666")

    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
