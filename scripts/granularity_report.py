"""Build per-scenario tables + composition figures from per_slide_counts.parquet.

Scenarios:
    Individual : s0, s1, s3, s6 (STHELAR breast), rep1, rep2 (Janesick)
    Grouped    : STHELAR breast standard (s0+s1+s3) · STHELAR breast Prime (s6 alone)
                 · Janesick (rep1+rep2) · all STHELAR breast (s0+s1+s3+s6)
                 · all breast combined (STHELAR breast + Janesick)
                 · all STHELAR skin · all STHELAR (all 16 tissues)

Outputs:
    docs/reports/granularity_audit_2026_05.md   — markdown report (all 13 × 3 tables)
    pipeline_output/figures_v2/fig_granularity_audit_*.png  — composition figures
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "figures_v2"))

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style, cell_color
from dapidl.ontology.training_tiers import COARSE_NAMES, MEDIUM_NAMES, FINE_NAMES

ROOT = Path("/mnt/work/git/dapidl")
PARQUET = ROOT / "pipeline_output" / "granularity_audit" / "per_slide_counts.parquet"
DOC_OUT = ROOT / "docs" / "reports" / "granularity_audit_2026_05.md"
FIG_DIR = ROOT / "pipeline_output" / "figures_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Scenarios — each is (label, filter_predicate)
# ------------------------------------------------------------------
def define_scenarios(df: pl.DataFrame) -> list[tuple[str, str, pl.Expr]]:
    """Return (group, label, polars filter) tuples."""
    is_sthelar = pl.col("source") == "STHELAR"
    return [
        # group "Individual slides"
        ("Individual", "STHELAR breast s0",  is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s0")),
        ("Individual", "STHELAR breast s1",  is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s1")),
        ("Individual", "STHELAR breast s3",  is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s3")),
        ("Individual", "STHELAR breast s6 (Prime)", is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s6")),
        ("Individual", "Janesick rep1",       (pl.col("source") == "Janesick") & (pl.col("slide") == "rep1")),
        ("Individual", "Janesick rep2",       (pl.col("source") == "Janesick") & (pl.col("slide") == "rep2")),
        # group "Grouped"
        ("Grouped", "STHELAR breast standard (s0 + s1 + s3)", is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide").is_in(["s0", "s1", "s3"]))),
        ("Grouped", "STHELAR breast Prime (s6 only)",          is_sthelar & (pl.col("tissue") == "breast") & (pl.col("slide") == "s6")),
        ("Grouped", "Janesick (rep1 + rep2)",                   pl.col("source") == "Janesick"),
        ("Grouped", "All STHELAR breast (s0 + s1 + s3 + s6)",   is_sthelar & (pl.col("tissue") == "breast")),
        ("Grouped", "All breast (STHELAR + Janesick)",          pl.col("tissue") == "breast"),
        ("Grouped", "All STHELAR skin",                         is_sthelar & (pl.col("tissue") == "skin")),
        ("Grouped", "All STHELAR (all 16 tissues)",             is_sthelar),
    ]


# ------------------------------------------------------------------
# Per-scenario tier aggregation
# ------------------------------------------------------------------
def tier_distribution(df: pl.DataFrame, scenario_filter: pl.Expr, tier: str,
                      class_order: list[str]) -> pl.DataFrame:
    sub = df.filter(scenario_filter & (pl.col("tier") == tier))
    if sub.is_empty():
        return pl.DataFrame({"class_name": class_order, "count": [0]*len(class_order),
                             "percent": [0.0]*len(class_order)})
    agg = (
        sub.group_by("class_name").agg(pl.col("count").sum())
           .with_columns(pl.col("count").cast(pl.Int64))
    )
    total = agg["count"].sum()
    agg = agg.with_columns((pl.col("count") / total * 100.0).alias("percent"))
    # Reindex to class_order, fill missing with 0; "Unknown" + any extras append at end
    seen = dict(zip(agg["class_name"].to_list(), zip(agg["count"].to_list(), agg["percent"].to_list())))
    rows = []
    for c in class_order:
        cnt, pct = seen.pop(c, (0, 0.0))
        rows.append((c, cnt, pct))
    for c, (cnt, pct) in seen.items():  # extras (e.g., generic compartment fallbacks, Unknown)
        rows.append((c, cnt, pct))
    return pl.DataFrame(rows, schema=["class_name", "count", "percent"], orient="row")


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------
def fmt_table(dist: pl.DataFrame, title: str, total: int) -> str:
    lines = [f"### {title}", "", f"_Total cells: **{total:,}**_", "",
             "| # | class | count | % |", "|---|-------|------:|----:|"]
    for i, row in enumerate(dist.iter_rows(named=True), start=1):
        cls, cnt, pct = row["class_name"], row["count"], row["percent"]
        lines.append(f"| {i} | {cls} | {cnt:,} | {pct:.2f}% |")
    return "\n".join(lines)


def build_markdown(df: pl.DataFrame) -> None:
    scenarios = define_scenarios(df)
    out = ["# Granularity audit — cell counts by scenario × tier",
           "",
           f"_Generated from `{PARQUET.relative_to(ROOT)}`_",
           "",
           "**Tier definitions** (single source of truth: `dapidl.ontology.training_tiers`):",
           "",
           f"- **COARSE ({len(COARSE_NAMES)})** — CL Super-Coarse (Level 1): " + " · ".join(COARSE_NAMES),
           f"- **MEDIUM ({len(MEDIUM_NAMES)})** — CL Coarse + L3, body-wide: " + " · ".join(MEDIUM_NAMES),
           f"- **FINE ({len(FINE_NAMES)})** — CL Medium L3+L4 + 2 pathology: " + " · ".join(FINE_NAMES),
           "",
           "Cells whose CL ancestry doesn't match a specific class fall back to the parent compartment "
           "name (e.g., `keratinocyte` → MEDIUM:`Epithelial`). `Unknown` = no CL mapping at all.",
           "",
           "---", ""]

    for tier, class_order in (("coarse", COARSE_NAMES),
                              ("medium", MEDIUM_NAMES),
                              ("fine",   FINE_NAMES)):
        out.append(f"## TIER: {tier.upper()}  ({len(class_order)} canonical classes)")
        out.append("")
        last_group = None
        for group, label, flt in scenarios:
            if group != last_group:
                out.append(f"### {group} scenarios")
                out.append("")
                last_group = group
            sub_total = int(df.filter(flt & (pl.col("tier") == tier))["count"].sum() or 0)
            dist = tier_distribution(df, flt, tier, class_order)
            out.append(fmt_table(dist, label, sub_total))
            out.append("")
        out.append("---")
        out.append("")
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text("\n".join(out))
    print(f"wrote {DOC_OUT}")


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def _color_for(cls: str) -> str:
    """Per-class color: prefer cell_color() palette; fallback to greys."""
    if cls in ("Unknown",):
        return "#CCCCCC"
    # Map known DAPIDL names to palette
    aliases = {
        "Epithelial_Luminal": "Epithelial", "Epithelial_Basal": "Epithelial",
        "Mammary_Luminal":    "Epithelial", "Myoepithelial":    "Epithelial",
        "DCIS":               "Neoplastic", "Invasive":         "Neoplastic",
        "T_Cell": "Immune", "B_Cell": "Immune", "NK_Cell": "Immune",
        "CD4_T_Cell": "Immune", "CD8_T_Cell": "Immune", "Treg": "Immune",
        "Plasma_Cell": "Immune", "Macrophage": "Immune",
        "pDC": "Immune", "cDC": "Immune", "Dendritic_Cell": "Immune",
        "Mast_Cell": "Mast", "Adipocyte": "Adipocyte",
        "Fibroblast": "Stromal", "Pericyte": "Stromal",
    }
    key = aliases.get(cls, cls)
    return cell_color(key)


def fig_composition(df: pl.DataFrame) -> None:
    """One figure per tier: stacked horizontal bars, one row per scenario."""
    scenarios = define_scenarios(df)

    for tier, class_order in (("coarse", COARSE_NAMES),
                              ("medium", MEDIUM_NAMES),
                              ("fine",   FINE_NAMES)):
        # Pass 1: collect union of class names across ALL scenarios for this tier
        all_classes_set: set[str] = set(class_order)
        for _grp, _label, flt in scenarios:
            extras = (
                df.filter(flt & (pl.col("tier") == tier))["class_name"].unique().to_list()
            )
            all_classes_set.update(extras)
        # Order: canonical first (in defined order), then extras alphabetically
        extras_ordered = sorted(c for c in all_classes_set if c not in class_order)
        all_classes = list(class_order) + extras_ordered

        # Pass 2: aggregate every scenario against the unified column set
        scen_labels = []
        scen_totals = []
        rows_pct = []
        for _grp, label, flt in scenarios:
            dist = tier_distribution(df, flt, tier, all_classes)
            # tier_distribution may append additional unseen extras after the requested order;
            # truncate to all_classes to keep matrix rectangular.
            dist_dict = dict(zip(dist["class_name"].to_list(), dist["percent"].to_list()))
            scen_labels.append(label)
            scen_totals.append(int(dist["count"].sum()))
            rows_pct.append([dist_dict.get(c, 0.0) for c in all_classes])
        rows_pct = np.array(rows_pct)

        apply_style()
        n_rows = len(scen_labels)
        fig, ax = plt.subplots(figsize=(13.5, 0.45 * n_rows + 2.0))
        y = np.arange(n_rows)
        left = np.zeros(n_rows)
        for j, cls in enumerate(all_classes):
            vals = rows_pct[:, j]
            color = _color_for(cls)
            ax.barh(y, vals, left=left, color=color, edgecolor="white", lw=0.6,
                    label=cls)
            # Annotate inside the bar if segment is wide enough
            for i, v in enumerate(vals):
                if v >= 4.0:  # only label segments ≥4%
                    ax.text(left[i] + v / 2, y[i], f"{v:.0f}%", ha="center",
                            va="center", fontsize=7.5, color="white", fontweight="bold")
            left += vals
        # Right-side total cell counts
        for i, t in enumerate(scen_totals):
            ax.text(101, y[i], f"  {t:,}", va="center", fontsize=9, color="#444")
        ax.set_yticks(y, scen_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of cells (%)")
        ax.set_title(f"{tier.upper()} tier composition — {len(all_classes)} classes  "
                     f"·  totals shown to right of bars",
                     loc="left", fontsize=13, pad=10)
        # Legend below in 2-3 columns
        ncol = min(6, max(3, len(all_classes) // 2))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=ncol, fontsize=8, frameon=False)
        plt.tight_layout()
        out = FIG_DIR / f"fig_granularity_audit_{tier}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def fig_total_cells_per_scenario(df: pl.DataFrame) -> None:
    """One bar per scenario showing total cell count (log y), color-coded by group."""
    scenarios = define_scenarios(df)
    scen_labels, scen_totals, scen_groups = [], [], []
    for grp, label, flt in scenarios:
        # use coarse tier (any tier gives same total)
        t = int(df.filter(flt & (pl.col("tier") == "coarse"))["count"].sum() or 0)
        scen_labels.append(label)
        scen_totals.append(t)
        scen_groups.append(grp)

    apply_style()
    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    x = np.arange(len(scen_labels))
    colors = ["#3D5A80" if g == "Individual" else "#6F4C9C" for g in scen_groups]
    bars = ax.bar(x, scen_totals, color=colors, edgecolor="white", lw=1.0)
    for b, t in zip(bars, scen_totals):
        ax.text(b.get_x() + b.get_width() / 2, t * 1.05, f"{t/1000:.0f}k" if t < 1e6 else f"{t/1e6:.1f}M",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x, scen_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("cells (log)")
    ax.set_title("Cell-count budget per scenario (log scale)",
                 loc="left", fontsize=13, pad=10)
    # Legend by group
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#3D5A80", label="Individual slide"),
               Patch(facecolor="#6F4C9C", label="Pooled / grouped")]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "fig_granularity_audit_totals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    df = pl.read_parquet(PARQUET)
    print(f"loaded {df.height:,} rows from {PARQUET}")
    build_markdown(df)
    fig_composition(df)
    fig_total_cells_per_scenario(df)


if __name__ == "__main__":
    main()
