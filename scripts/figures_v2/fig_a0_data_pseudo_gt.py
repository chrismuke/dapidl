"""Data sources & pseudo-GT methods — visual data card.

Two columns showing:
- Janesick et al. 2023 (Xenium breast tumor): rep1 + rep2 with EXPERT 17-class
  supervised ground truth (gold standard, hand-curated)
- STHELAR (Giraud-Sauveur 2025): 4 breast slides with cells_label2 — STHELAR's
  own multi-method consensus (Tangram on scRNA-seq + nuclei labels +
  confidence filter), median confidence 1.0

For each slide: cell count, label source, gene panel, confidence.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import numpy as np

from _style import apply_style

OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a0_data_pseudo_gt.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Data inventory
JANESICK = [
    {"name": "rep1", "n_total": 167_780, "n_used": 157_231, "panel": "313 genes"},
    {"name": "rep2", "n_total": 118_752, "n_used": 113_726, "panel": "313 genes"},
]

STHELAR = [
    {"name": "breast_s0", "n_total": 576_963, "n_used": 542_818,
     "panel": "Standard Xenium · 573 genes", "high_conf_pct": 88.2},
    {"name": "breast_s1", "n_total": 892_966, "n_used": 757_374,
     "panel": "Standard Xenium · 573 genes", "high_conf_pct": 74.3},
    {"name": "breast_s3", "n_total": 365_604, "n_used": 345_805,
     "panel": "Standard Xenium · 573 genes", "high_conf_pct": 83.1},
    {"name": "breast_s6", "n_total": 692_184, "n_used": 360_923,
     "panel": "Xenium PRIME · 8,232 genes", "high_conf_pct": 53.0},
]

JANESICK_COLOR = "#D4A017"  # gold = expert
STHELAR_COLOR = "#3D5A80"   # navy = automated consensus
HEADER_BG = "#F5F5F0"


def panel(ax, title: str, subtitle: str, slides: list,
          gt_method: str, gt_why: str, color: str,
          badge: str, x_offset: float = 0):
    """Draw one column of the data card."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Header
    ax.add_patch(FancyBboxPatch((0.2, 12.5), 9.6, 1.2,
                                boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=color, edgecolor="none"))
    ax.text(5.0, 13.4, title, ha="center", va="center", fontsize=15,
            fontweight="bold", color="white")
    ax.text(5.0, 12.85, subtitle, ha="center", va="center", fontsize=10,
            color="white", style="italic")

    # Badge (top-right corner)
    ax.add_patch(FancyBboxPatch((7.5, 11.5), 2.3, 0.7,
                                boxstyle="round,pad=0.04,rounding_size=0.1",
                                facecolor="white", edgecolor=color, lw=2))
    ax.text(8.65, 11.85, badge, ha="center", va="center", fontsize=10,
            fontweight="bold", color=color)

    # Slide cards
    n_slides = len(slides)
    block_h = 6.5  # total area for slide cards
    card_h = block_h / n_slides - 0.18
    y_start = 11.0
    for i, sl in enumerate(slides):
        y = y_start - i * (card_h + 0.18) - card_h
        # Slide card
        ax.add_patch(FancyBboxPatch((0.4, y), 9.2, card_h,
                                    boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor="white", edgecolor="#CCC", lw=1.0))
        # Color stripe on left
        ax.add_patch(plt.Rectangle((0.4, y), 0.18, card_h,
                                    facecolor=color, edgecolor="none"))
        # Slide name
        ax.text(0.85, y + card_h - 0.25, sl["name"], fontsize=12,
                fontweight="bold", color="#222", va="top")
        # Panel + cell count
        ax.text(0.85, y + card_h - 0.65, sl["panel"], fontsize=9,
                color="#666", va="top", style="italic")
        # Counts (right-aligned)
        ax.text(9.4, y + card_h - 0.25,
                f"{sl['n_total']:,} cells",
                fontsize=11, fontweight="bold", ha="right", va="top",
                color="#222")
        ax.text(9.4, y + card_h - 0.65,
                f"used: {sl['n_used']:,} ({100*sl['n_used']/sl['n_total']:.0f}%)",
                fontsize=9, ha="right", va="top", color="#666")
        # High-confidence percentage bar (STHELAR only)
        if "high_conf_pct" in sl:
            bar_y = y + 0.15
            ax.add_patch(plt.Rectangle((0.85, bar_y), 5.0, 0.18,
                                        facecolor="#E0E0E0", edgecolor="none"))
            ax.add_patch(plt.Rectangle((0.85, bar_y),
                                        5.0 * sl["high_conf_pct"] / 100,
                                        0.18, facecolor=color, edgecolor="none"))
            ax.text(6.0, bar_y + 0.09,
                    f"{sl['high_conf_pct']:.0f}% conf ≥ 0.99",
                    fontsize=8.5, va="center", ha="left", color="#444")

    # GT method box (bottom)
    ax.add_patch(FancyBboxPatch((0.4, 0.5), 9.2, 3.7,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor=HEADER_BG, edgecolor=color, lw=1.5))
    ax.text(0.85, 3.85, "▶ Pseudo-GT method", fontsize=11, fontweight="bold",
            color=color, va="top")
    ax.text(0.85, 3.35, gt_method, fontsize=10, color="#222", va="top",
            wrap=True)

    ax.text(0.85, 1.85, "▶ Why we trust it", fontsize=10, fontweight="bold",
            color=color, va="top", style="italic")
    ax.text(0.85, 1.45, gt_why, fontsize=9.5, color="#444", va="top",
            wrap=True)


def main() -> None:
    apply_style()

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(1, 2, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    panel(ax1,
          title="Janesick et al. 2023",
          subtitle="Xenium breast tumor (Nature Comm 2023)",
          slides=JANESICK,
          gt_method=("Manual expert annotation by Janesick et al. — "
                     "17 fine-grained breast cell types\n"
                     "(Invasive_Tumor, DCIS_1/2, Myoepi×2, Macrophages×2,\n"
                     "CD4/CD8/Treg, Endo, Stromal, Mast, B, IRF7+/LAMP3+ DCs).\n\n"
                     "Mapped 17 → COARSE 4 + MEDIUM 12 via JANESICK17_TO_*\n"
                     "(pathology subtypes use DAPIDL: pseudo-CL IDs)."),
          gt_why=("Hand-curated by domain experts on Xenium FFPE breast tumor — \n"
                   "the de-facto gold standard for breast Xenium evaluation.\n"
                   "Cited 200+ times. Zero algorithmic uncertainty."),
          color=JANESICK_COLOR,
          badge="★ EXPERT GT")

    panel(ax2,
          title="STHELAR (Giraud-Sauveur 2025)",
          subtitle="16 tissues × 31 slides — 11M cells (breast slice shown)",
          slides=STHELAR,
          gt_method=("STHELAR-published cells_label2 column from per-slide parquet\n"
                     "metadata. Built by STHELAR's pipeline:\n"
                     "  ① Tangram label transfer from matched scRNA-seq atlas\n"
                     "  ② Nuclei-based annotations (independent verification)\n"
                     "  ③ Multi-method consensus + confidence scoring\n"
                     "  ④ Mapped 11 → COARSE 4 + MEDIUM 12 (STHELAR_LABEL2_TO_*)"),
          gt_why=("Multi-method consensus = self-consistency across orthogonal\n"
                   "annotators. Median per-cell confidence = 1.000 (q05 = 0.764).\n"
                   "Drops only `less10` and ambiguous labels — keep ~94%."),
          color=STHELAR_COLOR,
          badge="✓ CONSENSUS GT")

    # Bottom convergence: both feed into DAPI training
    fig.text(0.5, 0.04,
             "↓  Both feed into the DAPIDL training pipeline as pseudo-ground-truth labels  ↓",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="#444",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFAEB",
                       edgecolor="#444", lw=1.2))

    fig.suptitle(
        "DAPIDL pseudo-ground-truth sources — what we trained the DAPI model on",
        fontsize=15, fontweight="bold", y=0.995,
    )

    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.97))
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
