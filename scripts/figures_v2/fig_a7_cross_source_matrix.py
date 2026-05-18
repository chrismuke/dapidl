"""Cross-source training matrix — A/B/C/D pairs.

Four scenarios on breast tissue:
  A  Janesick rep1+rep2     → test STHELAR s0/s1/s3/s6
  B  STHELAR std (s0/s1/s3) → test Janesick rep1/rep2 + Prime s6
  C  STHELAR Prime (s6)     → test all others
  D  all STHELAR (s0+s1+s3+s6) → test Janesick rep1/rep2

Headline story:
  - C: single-source training (s6 alone) overfits — val 0.88, transfer 0.22
  - D: multi-source pool — val 0.68, transfer 0.62 to Janesick
  - Adding heterogeneous training data turns 0.22 → 0.62 (3x improvement)
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
OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a7_cross_source_matrix.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Column order in the heatmap (matches the deck's narrative left→right):
# in-domain validation first, then Janesick test slides, then STHELAR
COLS = ["val", "rep1", "rep2", "s0", "s1", "s3", "s6"]
COL_LABEL = {
    "val": "in-domain\nVAL",
    "rep1": "Janesick\nrep1",
    "rep2": "Janesick\nrep2",
    "s0": "STHELAR\ns0",
    "s1": "STHELAR\ns1",
    "s3": "STHELAR\ns3",
    "s6": "STHELAR\ns6 (Prime)",
}

# Each row is a scenario; the values dict has F1 by column key.
ROWS = [
    ("A", "Janesick → STHELAR",        "A_janesick_to_sthelar"),
    ("B", "STHELAR std → Janesick+s6", "B_sthelar_std_to_janesick_prime"),
    ("C", "STHELAR-s6 (Prime) → all",  "C_sthelar_prime_to_all"),
    ("D", "all STHELAR → Janesick",    "D_all_sthelar_to_janesick"),
]


def load_scenario(stem: str, suffix: str = "") -> dict[str, float | None]:
    """Return {column: F1 or None} for one (scenario, tier) combination.

    Per_test keys in the summary use the canonical source names (e.g.
    'sthelar_breast_s0', 'xenium_rep1') — strip those prefixes to match COLS.
    """
    p = ROOT / f"{stem}{suffix}" / "summary.json"
    if not p.exists():
        return {c: None for c in COLS}
    s = json.loads(p.read_text())
    out: dict[str, float | None] = {c: None for c in COLS}
    out["val"] = float(s["best_val_macro_f1"])
    for k, v in s.get("per_test", {}).items():
        slide_key = k.replace("sthelar_breast_", "").replace("xenium_", "")
        if slide_key in COLS:
            out[slide_key] = float(v["macro_f1"])
    return out


def draw_heatmap(ax, matrix: list[list[float | None]], title: str,
                 vmax: float, annotate_arrow: bool = False) -> None:
    """Draw a row-by-row colormapped grid with values written in each cell.

    None cells are drawn as a hatched gray to make 'not run' visually obvious.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    cmap = plt.get_cmap("YlGnBu")
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v is None:
                ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                            facecolor="#EEEEEE",
                                            hatch="///", edgecolor="white",
                                            lw=2.0))
                ax.text(j + 0.5, n_rows - 1 - i + 0.5, "—",
                        ha="center", va="center", fontsize=12, color="#888")
                continue
            color = cmap(v / vmax)
            ax.add_patch(plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                        facecolor=color, edgecolor="white",
                                        lw=2.0))
            tcolor = "white" if v / vmax > 0.55 else "#1A1A1A"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, f"{v:.2f}",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=tcolor)

    # Vertical separator after VAL column
    ax.axvline(1.0, color="#444", lw=1.6)
    ax.text(0.5, n_rows + 0.18, "in-domain", ha="center", fontsize=10,
            style="italic", color="#666")
    ax.text((n_cols + 1) / 2, n_rows + 0.18, "out-of-domain TEST per slide",
            ha="center", fontsize=10, style="italic", color="#666")

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([COL_LABEL[c] for c in COLS], fontsize=10)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([f"{label}\n{descr}"
                        for (label, descr, _) in reversed(ROWS)],
                       fontsize=10.5)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 0.4)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=18)
    # Hide spines + ticks (cells already provide visual structure)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.grid(False)

    # The headline annotation — only on COARSE panel
    if annotate_arrow:
        # C → D arrow showing 3× transfer improvement on rep1
        # ROWS[2]=C is at row index 2 (drawn at n_rows-1-2 = 1 in plot coords)
        # ROWS[3]=D is at row index 3 (drawn at n_rows-1-3 = 0)
        # rep1 column = index 1
        ax.annotate(
            "",
            xy=(1.5, 0.5), xytext=(1.5, 1.5),
            arrowprops=dict(arrowstyle="->", color="#A33", lw=2.4,
                            connectionstyle="arc3,rad=-0.3"),
        )
        ax.text(2.3, 1.0,
                "  Multi-source pool\n  transfer: 0.23 → 0.62\n  (≈3× improvement)",
                fontsize=10, color="#A33", fontweight="bold", va="center")


def main() -> None:
    apply_style()

    # Load all 8 cells (4 scenarios × 2 tiers)
    coarse_matrix = [list(load_scenario(stem).values()) for _, _, stem in ROWS]
    medium_matrix = [list(load_scenario(stem, "_medium").values()) for _, _, stem in ROWS]

    fig = plt.figure(figsize=(14.0, 7.5))
    gs = fig.add_gridspec(2, 1, hspace=0.55)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    draw_heatmap(ax1, coarse_matrix,
                 "A. COARSE-4 · macro F1 by training scenario × per-slide test",
                 vmax=0.9, annotate_arrow=True)
    draw_heatmap(ax2, medium_matrix,
                 "B. MEDIUM-12 · macro F1 (same matrix, harder task)",
                 vmax=0.65)

    fig.suptitle(
        "Cross-source training matrix · breast tissue · DAPI-only EfficientNetV2-S",
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.text(0.99, 0.005,
             "STHELAR breast cells_label2 GT (s0/s1/s3/s6); "
             "Janesick supervised 17→4 (rep1/rep2). "
             "VAL = held-out from train pool. "
             "— = test source overlapped train (excluded). "
             "Best results in dark blue.",
             ha="right", va="bottom", fontsize=8.5, color="#666")

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
