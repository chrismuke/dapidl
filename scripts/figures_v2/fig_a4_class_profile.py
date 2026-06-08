"""Per-class performance profile — DAPI model vs annotation methods (COARSE-4).

For each class (Endo/Epi/Imm/Stromal):
  - DAPI model A test-mean F1 (across 4 STHELAR slides)
  - Each annotation method's mean F1 across 6 slides

Shows which biological classes are easy/hard for both DAPI morphology AND
gene-expression-based methods. Strong DAPI signal where bars match across
methods; biological diversity where they diverge.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import apply_style, CELL_COLORS

ROOT_TRAIN = Path("/mnt/work/git/dapidl/pipeline_output/breast_pooled_2026_05")
ROOT_ANN = Path("/mnt/work/git/dapidl/pipeline_output/annotation_run_2026_05")
OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a4_class_profile.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
SLIDES = ["s0", "s1", "s3", "s6"]

GROUP_COLOR = {
    "DAPI model": "#1A4B7A",
    "CellTypist": "#3D5A80",
    "scType": "#BC8D5A",
    "SingleR": "#4A9D9C",
}


def dapi_per_class_mean() -> dict[str, float]:
    """Mean F1 per class across the 4 STHELAR test slides."""
    s = json.loads((ROOT_TRAIN / "A_janesick_to_sthelar" / "summary.json").read_text())
    out = {c: [] for c in CLASSES}
    for slide in SLIDES:
        per = s["per_test"][f"sthelar_breast_{slide}"]["per_class"]
        for c in CLASSES:
            out[c].append(per[c]["f1"])
    return {c: float(np.mean(v)) for c, v in out.items()}


def annotation_per_class_mean() -> dict[str, dict[str, float]]:
    """Mean F1 per (method, class) across 6 slides."""
    df = pl.read_parquet(ROOT_ANN / "coarse_metrics.parquet")
    out = {"CellTypist": {}, "scType": {}, "SingleR": {}}
    method_map = {
        "celltypist_noMV_Cells_Adult_Breast": "CellTypist",
        "sctype_custom_default": "scType",
        "singler_blueprint": "SingleR",
    }
    for raw, short in method_map.items():
        sub = df.filter(pl.col("method") == raw)
        for c in CLASSES:
            col = f"f1_{c}"
            if col in sub.columns:
                out[short][c] = float(sub[col].mean())
    return out


def main() -> None:
    apply_style()
    dapi = dapi_per_class_mean()
    ann = annotation_per_class_mean()

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    x = np.arange(len(CLASSES))
    groups = ["DAPI model", "CellTypist", "scType", "SingleR"]
    width = 0.2

    for k, g in enumerate(groups):
        if g == "DAPI model":
            vals = [dapi[c] for c in CLASSES]
        else:
            vals = [ann[g].get(c, 0.0) for c in CLASSES]
        bars = ax.bar(x + (k - 1.5) * width, vals, width,
                      color=GROUP_COLOR[g], edgecolor="white", lw=1.0,
                      label=g, alpha=0.95 if g == "DAPI model" else 0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold" if g == "DAPI model" else "normal")

    # Class-color tick label backgrounds
    for i, c in enumerate(CLASSES):
        ax.add_patch(plt.Rectangle((i - 0.45, -0.04), 0.9, 0.025,
                                    facecolor=CELL_COLORS.get(c, "#888"),
                                    transform=ax.get_xaxis_transform(),
                                    clip_on=False, edgecolor="none"))

    ax.set_xticks(x, CLASSES, fontsize=11, fontweight="bold")
    ax.set_ylabel("mean macro F1", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Per-class F1 profile — DAPI morphology vs gene-expression baselines (COARSE-4)",
        loc="left", fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Insight callout
    ax.text(0.02, 0.97,
            "DAPI is competitive on Epithelial (densely packed nuclei) and Stromal\n"
            "(elongated nuclei). Endothelial is the hardest class for everyone.",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color="#444", style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC",
                      edgecolor="#B0A060", lw=0.8))

    fig.text(0.99, 0.005,
             "DAPI model: per-class mean over 4 STHELAR test slides (s0/s1/s3/s6). "
             "Annotation methods: per-class mean over slides with that class in GT "
             "(≤6: rep1+rep2 + 4 STHELAR; Endothelial only s0/rep1/rep2). cells_label2 GT.",
             ha="right", va="bottom", fontsize=8.5, color="#666")

    plt.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
