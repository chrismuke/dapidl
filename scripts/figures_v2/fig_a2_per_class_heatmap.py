"""Per-class F1 heatmap for COARSE-4 (Coarse A) + MEDIUM-12 (Medium A).

Rows = cell-type classes (COARSE: 4; MEDIUM: 12).
Cols = STHELAR test slides (s0, s1, s3, s6 — Prime).
Cell value = per-class F1 (0..1).
Annotated with support count (millions of cells per class for visual scale).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from _style import apply_style

ROOT = Path("/mnt/work/git/dapidl/pipeline_output/breast_pooled_2026_05")
OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a2_per_class_heatmap.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

SLIDES = ["s0", "s1", "s3", "s6"]
COARSE = ["Endothelial", "Epithelial", "Immune", "Stromal"]
MEDIUM = [
    "Epithelial_Luminal", "Epithelial_Basal",
    "T_Cell", "B_Cell", "Macrophage", "Dendritic_Cell", "Mast_Cell",
    "Fibroblast", "Pericyte", "Adipocyte", "Endothelial", "Neural",
]


def load_per_class(d: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    s = json.loads((d / "summary.json").read_text())
    f1_mat = np.zeros((len(classes), len(SLIDES)))
    sup_mat = np.zeros((len(classes), len(SLIDES)), dtype=int)
    for j, slide in enumerate(SLIDES):
        per = s["per_test"][f"sthelar_breast_{slide}"]["per_class"]
        for i, c in enumerate(classes):
            if c in per:
                f1_mat[i, j] = per[c]["f1"]
                sup_mat[i, j] = per[c]["support"]
    return f1_mat, sup_mat


def render_panel(ax, f1, sup, classes, title: str, cmap_name: str = "viridis"):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1.0)
    im = ax.imshow(f1, cmap=cmap, norm=norm, aspect="auto")

    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            v = f1[i, j]
            s = sup[i, j]
            if v == 0 and s == 0:
                txt = "—"
            else:
                if s >= 100_000:
                    s_str = f"{s/1000:.0f}k"
                elif s >= 1000:
                    s_str = f"{s/1000:.1f}k"
                else:
                    s_str = str(s)
                txt = f"{v:.2f}\n({s_str})"
            color = "white" if v > 0.55 else "#222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.0,
                    color=color, fontweight="bold" if v > 0.5 else "normal")

    ax.set_xticks(np.arange(len(SLIDES)),
                  ["s0", "s1", "s3", "s6\n(Prime)"], fontsize=10)
    ax.set_yticks(np.arange(len(classes)), classes, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("STHELAR test slide", fontsize=10)
    return im


def main() -> None:
    apply_style()
    f1_c, sup_c = load_per_class(ROOT / "A_janesick_to_sthelar", COARSE)
    f1_m, sup_m = load_per_class(ROOT / "A_janesick_to_sthelar_medium", MEDIUM)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 7.5),
        gridspec_kw={"width_ratios": [1, 1.55]},
    )
    im1 = render_panel(ax1, f1_c, sup_c, COARSE, "A. COARSE 4-class")
    im2 = render_panel(ax2, f1_m, sup_m, MEDIUM, "B. MEDIUM 12-class")

    cb = fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.02)
    cb.set_label("per-class F1", fontsize=10)

    fig.suptitle(
        "Cross-source per-class F1 — Janesick → STHELAR (cell counts in parentheses)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.text(0.99, 0.005,
             "Best class everywhere: Epithelial (dominant majority cell). "
             "Hardest: rare classes (Adipocyte, Pericyte, Dendritic_Cell) drop to 0 in many slides.",
             ha="right", va="bottom", fontsize=9, color="#555", style="italic")

    plt.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
