"""Build heatmap: rows = scenarios, cols = classes, color = log10(cells).

Shows at-a-glance which classes are populated in which scenarios.
One heatmap per tier (3 figures total).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "figures_v2"))

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from _style import apply_style
from dapidl.ontology.training_tiers import COARSE_NAMES, MEDIUM_NAMES, FINE_NAMES

ROOT = Path("/mnt/work/git/dapidl")
PARQUET = ROOT / "pipeline_output" / "granularity_audit" / "per_slide_counts.parquet"
FIG_DIR = ROOT / "pipeline_output" / "figures_v2"


def define_scenarios(df: pl.DataFrame):
    is_sthelar = pl.col("source") == "STHELAR"
    return [
        ("STHELAR breast s0",                      is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s0")),
        ("STHELAR breast s1",                      is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s1")),
        ("STHELAR breast s3",                      is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s3")),
        ("STHELAR breast s6 (Prime)",              is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s6")),
        ("Janesick rep1",                          (pl.col("source") == "Janesick") & (pl.col("slide") == "rep1")),
        ("Janesick rep2",                          (pl.col("source") == "Janesick") & (pl.col("slide") == "rep2")),
        ("STHELAR breast standard (s0+s1+s3)",     is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide").is_in(["s0", "s1", "s3"]))),
        ("STHELAR breast Prime (s6)",              is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s6")),
        ("Janesick (rep1+rep2)",                    pl.col("source") == "Janesick"),
        ("All STHELAR breast (s0+s1+s3+s6)",       is_sthelar & (pl.col("tissue") == "breast")),
        ("All breast (STHELAR+Janesick)",           pl.col("tissue") == "breast"),
        ("All STHELAR skin",                        is_sthelar & (pl.col("tissue") == "skin")),
        ("All STHELAR (16 tissues)",                is_sthelar),
    ]


def build_matrix(df: pl.DataFrame, tier: str, class_order: list[str]):
    scenarios = define_scenarios(df)
    # Union of classes appearing in this tier across all scenarios
    union = set(class_order)
    for _, flt in scenarios:
        union.update(df.filter(flt & (pl.col("tier") == tier))["class_name"].unique().to_list())
    extras = sorted(c for c in union if c not in class_order)
    classes = list(class_order) + extras

    M = np.zeros((len(scenarios), len(classes)), dtype=np.int64)
    for i, (_label, flt) in enumerate(scenarios):
        sub = (
            df.filter(flt & (pl.col("tier") == tier))
              .group_by("class_name").agg(pl.col("count").sum())
        )
        d = dict(zip(sub["class_name"].to_list(), sub["count"].to_list()))
        for j, c in enumerate(classes):
            M[i, j] = d.get(c, 0)
    return [s[0] for s in scenarios], classes, M


def render_heatmap(scen_labels, class_names, M, tier_name: str):
    apply_style()
    fig, ax = plt.subplots(figsize=(0.5 * len(class_names) + 4.5, 0.45 * len(scen_labels) + 2.5))
    # log10(cell+1) for color, label with raw count
    log_M = np.log10(M + 1)
    vmax = max(np.log10(M.max() + 1), 1)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    im = ax.imshow(log_M, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Annotate each cell with the count
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v == 0:
                txt = "—"
            elif v < 1000:
                txt = f"{v}"
            elif v < 1_000_000:
                txt = f"{v/1000:.0f}k"
            else:
                txt = f"{v/1_000_000:.1f}M"
            color = "white" if log_M[i, j] > 0.55 * vmax else "#222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(scen_labels)), scen_labels, fontsize=9)
    # Separator line between individual and grouped scenarios (after row 5 = first 6 are individual)
    ax.axhline(y=5.5, color="white", lw=2.2)
    ax.axhline(y=5.5, color="#444", lw=0.8, linestyle="--")
    # Annotation for the divider
    ax.text(-0.7, 5.5, "—— grouped ——", ha="right", va="center", fontsize=8,
            color="#444", style="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("log₁₀(cells + 1)", fontsize=9)
    ax.set_title(f"{tier_name.upper()} tier — coverage matrix\n"
                 "(rows = scenarios · columns = classes · cell = cell count, color = log₁₀)",
                 loc="left", fontsize=12, pad=10)
    plt.tight_layout()
    out = FIG_DIR / f"fig_granularity_heatmap_{tier_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    df = pl.read_parquet(PARQUET)
    for tier, names in (("coarse", COARSE_NAMES),
                        ("medium", MEDIUM_NAMES),
                        ("fine",   FINE_NAMES)):
        scen, classes, M = build_matrix(df, tier, names)
        render_heatmap(scen, classes, M, tier)


if __name__ == "__main__":
    main()
