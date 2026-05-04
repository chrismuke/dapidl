"""Pipeline overview schematic with result callouts.

Visual story:
  Janesick rep1+rep2 (expert 17→4 GT) ──┐
  STHELAR cells_label2 (consensus 11→12)─┴→ DAPI patches (128px) ──→
        EfficientNetV2-S backbone ──→  predicted cell type (4 / 12)

With result callouts: in-domain val F1, cross-source test F1 ranges,
annotation method baseline.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _style import apply_style

OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a6_pipeline.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Color contract
JANESICK = "#D4A017"
STHELAR = "#3D5A80"
MODEL = "#1A4B7A"
COARSE = "#1A4B7A"
MEDIUM = "#A33"
ANN_BG = "#F5F5F0"


def box(ax, x, y, w, h, color, label, sub=None, text_color="white",
        rounding=0.12, lw=0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0.04,rounding_size={rounding}",
                                facecolor=color, edgecolor="#222" if lw > 0 else "none",
                                lw=lw))
    if sub:
        ax.text(x, y + 0.06, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color)
        ax.text(x, y - 0.10, sub, ha="center", va="center",
                fontsize=9, color=text_color, alpha=0.9, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color)


def arrow(ax, x0, y0, x1, y1, color="#444", lw=2.0, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                        arrowstyle="->", mutation_scale=18,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)


def callout(ax, x, y, text, color, w=1.5, h=0.45):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.07",
                                facecolor="white", edgecolor=color, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=10,
            fontweight="bold", color=color)


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Row 1: Pseudo-GT sources (top) ──────────────────────────────
    box(ax, 1.6, 6.7, 2.6, 0.95, JANESICK,
        "Janesick rep1+rep2",
        "Expert 17 → 4 GT (270k cells)")
    box(ax, 1.6, 5.5, 2.6, 0.95, STHELAR,
        "STHELAR breast s0/s1/s3/s6",
        "cells_label2 11 → 12 GT (2M cells)")

    # ── Row 2: Patch extraction ─────────────────────────────────────
    box(ax, 5.8, 6.1, 2.6, 1.1, "#444",
        "DAPI patches",
        "128 × 128 px around\nnucleus centroid")

    # Arrows from GT → patches
    arrow(ax, 2.9, 6.7, 4.5, 6.4, color=JANESICK, lw=2.0)
    arrow(ax, 2.9, 5.5, 4.5, 5.85, color=STHELAR, lw=2.0)

    # ── Row 3: Model ────────────────────────────────────────────────
    box(ax, 9.5, 6.1, 3.0, 1.1, MODEL,
        "EfficientNetV2-S",
        "1× → 3× channel adapt\n+ Dropout(0.3) + Linear")

    arrow(ax, 7.1, 6.1, 8.0, 6.1, color=MODEL, lw=2.5)

    # ── Row 4: Two heads ────────────────────────────────────────────
    box(ax, 13.5, 6.7, 2.4, 0.95, COARSE,
        "COARSE 4-class",
        "Endo / Epi / Imm / Stromal")
    box(ax, 13.5, 5.5, 2.4, 0.95, MEDIUM,
        "MEDIUM 12-class",
        "+ Luminal/Basal split, T/B/Macro")

    arrow(ax, 11.0, 6.3, 12.3, 6.7, color=COARSE, lw=2.0)
    arrow(ax, 11.0, 5.9, 12.3, 5.5, color=MEDIUM, lw=2.0)

    # ── Result callouts (mid section) ───────────────────────────────
    ax.text(8.0, 4.5, "Cross-source A pair (Janesick → STHELAR)",
            fontsize=12, fontweight="bold", ha="center", color="#222")

    # Coarse callouts
    callout(ax, 4.0, 3.7, "in-domain val\nF1 = 0.753", COARSE, w=2.2, h=0.85)
    callout(ax, 8.0, 3.7, "STHELAR test mean\nF1 = 0.382", COARSE, w=2.2, h=0.85)
    callout(ax, 12.0, 3.7,
            "best slide  s0\nF1 = 0.468", COARSE, w=2.2, h=0.85)

    ax.text(2.1, 3.7, "COARSE 4", fontsize=12, fontweight="bold",
            ha="center", color=COARSE, rotation=0)

    # Medium callouts
    callout(ax, 4.0, 2.5, "in-domain val\nF1 = 0.497", MEDIUM, w=2.2, h=0.85)
    callout(ax, 8.0, 2.5, "STHELAR test mean\nF1 = 0.144", MEDIUM, w=2.2, h=0.85)
    callout(ax, 12.0, 2.5, "best slide  s0\nF1 = 0.194", MEDIUM, w=2.2, h=0.85)

    ax.text(2.1, 2.5, "MEDIUM 12", fontsize=12, fontweight="bold",
            ha="center", color=MEDIUM, rotation=0)

    # ── Annotation baseline footer ──────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.5, 0.4), 15.0, 1.2,
                                boxstyle="round,pad=0.04,rounding_size=0.1",
                                facecolor=ANN_BG, edgecolor="#666", lw=1.0))
    ax.text(0.95, 1.35,
            "▶ Annotation baseline (gene-expression methods on the same slides):",
            fontsize=11, fontweight="bold", color="#333", va="center")

    ax.text(0.95, 0.85,
            "scType custom_default (best single, COARSE):  mean F1 = 0.558  ·  "
            "CT + scType (2-method consensus):  F1 = 0.538  ·  "
            "Adding SingleR HURTS (drops to 0.218).",
            fontsize=9.5, color="#444", va="center")

    fig.suptitle(
        "DAPIDL pipeline — pseudo-GT in, cell-type predictions out (DAPI nuclear staining only)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
