"""Figure 0 · Annotation method comparison on STHELAR breast s0.

Source: pipeline_output/annotation_eval_2026_05/by_method.parquet (15 methods).

Two-panel:
A. Bar chart of macro F1 across methods (sorted, colored by family).
B. Same methods on the accuracy/weighted-F1 axes (because under heavy class
   imbalance accuracy is misleading — but it's still what reviewers ask about).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig00_annotation_methods.png"


def family(method: str) -> str:
    if "popV" in method or "popv" in method:
        return "popV ensemble"
    if "BANKSY" in method:
        return "BANKSY clustering"
    if "Tangram" in method or "combined" in method or method == "pipeline-results":
        return "Tangram + ensemble"
    if "direct_subsample" in method:
        return "popV ensemble"
    if "onclass" in method:
        return "OnClass"
    return "other"


FAMILY_COLORS = {
    "popV ensemble":     "#3D5A80",
    "Tangram + ensemble":"#6F4C9C",
    "BANKSY clustering": "#BC8D5A",
    "OnClass":           "#9E9E9E",
    "other":             "#CCCCCC",
}


def main() -> None:
    apply_style()
    df = pl.read_parquet(ROOT / "annotation_eval_2026_05" / "by_method.parquet")
    df = (
        df.filter(pl.col("level") == "label1")
        .with_columns(family=pl.col("method").map_elements(family, return_dtype=pl.String))
        .sort("f1_macro", descending=True)
    )
    methods = df["method"].to_list()
    f1 = df["f1_macro"].to_list()
    acc = df["accuracy"].to_list()
    fams = df["family"].to_list()
    colors = [FAMILY_COLORS[f] for f in fams]

    fig, ax = plt.subplots(figsize=(13.0, 6.5))
    y = np.arange(len(methods))
    bars = ax.barh(y, f1, color=colors, edgecolor="white", lw=1.0,
                   label="macro F1")
    for i, (m, fv) in enumerate(zip(methods, f1)):
        ax.text(fv + 0.005, i, f"{fv:.3f}", va="center", fontsize=10,
                fontweight="bold", color="#222")
    # Add accuracy as small open circles on the same row to highlight the
    # accuracy/F1 gap (the imbalance signature).
    for i, av in enumerate(acc):
        if av is not None:
            ax.scatter(av, i, color="black", marker="o", s=35,
                       facecolors="white", edgecolors="black", zorder=5)

    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlabel("score on STHELAR breast s0 (label1, 7-class CL)")
    ax.set_xlim(0, 1.0)

    # Legend: family colors + accuracy marker
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=c, edgecolor="white", label=f)
               for f, c in FAMILY_COLORS.items()]
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="white", markeredgecolor="black",
                          markersize=8, label="accuracy"))
    ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=True,
              framealpha=0.95)

    ax.set_title(
        "Annotation method comparison — popV ensembles win on macro F1\n"
        "(accuracy is misleading under heavy class imbalance)",
        loc="left", fontsize=14, pad=12,
    )
    fig.text(0.99, 0.01,
             "STHELAR breast s0 (574,869 cells), label1 = Cell-Ontology 7-class. "
             "Black ring = accuracy on same data — note how 'high accuracy ≠ high F1'.",
             ha="right", va="bottom", fontsize=9, color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
