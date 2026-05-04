"""HEADLINE figure: granularity ladder + cross-source domain gap.

Panel A — Validation (in-domain, Janesick) vs per-slide test (out-of-domain
STHELAR) F1 at COARSE-4 and MEDIUM-12. Visualizes:
  1. Best in-domain F1 we can hit with DAPI alone
  2. The drop when crossing source (Janesick → STHELAR)
  3. Granularity cost (4-class easy, 12-class hard)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style

ROOT = Path("/mnt/work/git/dapidl/pipeline_output/breast_pooled_2026_05")
OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a1_headline.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

SLIDES = ["s0", "s1", "s3", "s6"]
SLIDE_FULL = {
    "s0": "STHELAR\nbreast_s0",
    "s1": "STHELAR\nbreast_s1",
    "s3": "STHELAR\nbreast_s3",
    "s6": "STHELAR Prime\nbreast_s6",
}

COARSE_COL = "#1A4B7A"   # navy
MEDIUM_COL = "#A33"      # crimson


def load_per_test(d: Path) -> tuple[float, dict[str, float]]:
    s = json.loads((d / "summary.json").read_text())
    val = float(s["best_val_macro_f1"])
    per = {k.replace("sthelar_breast_", ""): float(v["macro_f1"])
           for k, v in s["per_test"].items()}
    return val, per


def main() -> None:
    apply_style()
    val_c, per_c = load_per_test(ROOT / "A_janesick_to_sthelar")
    val_m, per_m = load_per_test(ROOT / "A_janesick_to_sthelar_medium")

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(SLIDES) + 1)  # +1 for val
    width = 0.36

    # Coarse bars
    coarse_vals = [val_c] + [per_c[s] for s in SLIDES]
    medium_vals = [val_m] + [per_m[s] for s in SLIDES]

    bars_c = ax.bar(x - width / 2, coarse_vals, width, label="COARSE 4-class",
                    color=COARSE_COL, edgecolor="white", lw=1.4)
    bars_m = ax.bar(x + width / 2, medium_vals, width, label="MEDIUM 12-class",
                    color=MEDIUM_COL, edgecolor="white", lw=1.4)

    # Value labels above bars
    for bars, vals in [(bars_c, coarse_vals), (bars_m, medium_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    # Vertical separator between val and test
    ax.axvline(0.5, color="#888", lw=1.2, ls=":")
    ax.text(0, 0.92, "in-domain\nvalidation", ha="center", va="top",
            fontsize=10, color="#444", style="italic",
            transform=ax.get_xaxis_transform())
    ax.text(2.75, 0.92, "out-of-domain TEST  (cross-source: Janesick → STHELAR)",
            ha="center", va="top", fontsize=10, color="#444", style="italic",
            transform=ax.get_xaxis_transform())

    # Domain-gap arrows from val to mean test
    mean_test_c = np.mean([per_c[s] for s in SLIDES])
    mean_test_m = np.mean([per_m[s] for s in SLIDES])
    ax.annotate("", xy=(2.5, mean_test_c), xytext=(0, val_c),
                arrowprops=dict(arrowstyle="->", color=COARSE_COL,
                                alpha=0.4, lw=2))
    ax.annotate(f"  Δ = -{(val_c - mean_test_c)*100:.0f} pp",
                xy=(0.7, (val_c + mean_test_c) / 2),
                fontsize=10, color=COARSE_COL, fontweight="bold")
    ax.annotate("", xy=(2.5, mean_test_m), xytext=(0, val_m),
                arrowprops=dict(arrowstyle="->", color=MEDIUM_COL,
                                alpha=0.4, lw=2))
    ax.annotate(f"  Δ = -{(val_m - mean_test_m)*100:.0f} pp",
                xy=(0.7, (val_m + mean_test_m) / 2),
                fontsize=10, color=MEDIUM_COL, fontweight="bold")

    ax.set_xticks(x, ["validation\n(rep1+rep2)"] + [SLIDE_FULL[s] for s in SLIDES],
                  fontsize=10)
    ax.set_ylabel("macro F1 score", fontsize=12)
    ax.set_ylim(0, 0.92)
    ax.set_title(
        "DAPI-only cell-type prediction:  granularity ladder × cross-source generalization",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.99, 0.005,
             "Train: Janesick rep1+rep2 (270k cells, 17→4 GT classes).  "
             "Test: STHELAR breast_s0/s1/s3 (Standard Xenium) + s6 (Xenium Prime, 5K-gene panel).  "
             "EfficientNetV2-S, 128px DAPI patches.",
             ha="right", va="bottom", fontsize=8.5, color="#666")

    plt.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
