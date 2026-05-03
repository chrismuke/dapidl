"""DAPIDL canonical 3-tier CL ontology — hierarchical tree.

Visualizes the 5-12-18 class hierarchy with parent-child edges, CL IDs,
and color coding by source compartment. Pathology extensions (DCIS, Invasive)
shown with a distinct outline.

Layout: 3 columns (COARSE, MEDIUM, FINE) with directed edges from each
class to its parent in the prior column.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from _style import apply_style
from dapidl.ontology.training_tiers import COARSE, MEDIUM, FINE

ROOT = Path("/mnt/work/git/dapidl")
OUT = ROOT / "pipeline_output" / "figures_v2" / "fig_ontology_hierarchy.png"

# Compartment colors (DAPIDL convention)
COMP_COLOR = {
    "Epithelial":  "#ED553B",
    "Immune":      "#3D5A80",
    "Stromal":     "#BC8D5A",
    "Endothelial": "#4A9D9C",
    "Neural":      "#9C2A2A",
}


def get_compartment(cl_id: str) -> str:
    """Map any CL ID in our schema to its super-coarse compartment."""
    # Check direct membership
    for c in COARSE:
        if c.cl_id == cl_id:
            return c.name
    # Walk: medium → coarse parent
    for m in MEDIUM:
        if m.cl_id == cl_id:
            for c in COARSE:
                if c.cl_id == m.parent_cl:
                    return c.name
    # Fine → walk parent chain
    for f in FINE:
        if f.cl_id == cl_id:
            # walk up
            parent = f.parent_cl
            while parent:
                for c in COARSE:
                    if c.cl_id == parent:
                        return c.name
                # Try medium parent
                med_match = next((m for m in MEDIUM if m.cl_id == parent), None)
                if med_match:
                    parent = med_match.parent_cl
                    continue
                # Try fine parent
                fine_match = next((ff for ff in FINE if ff.cl_id == parent), None)
                if fine_match:
                    parent = fine_match.parent_cl
                    continue
                break
    return "Epithelial"  # fallback (shouldn't happen for our 18)


def draw():
    apply_style()
    # Group items by compartment first to compute total Y range
    fine_groups = {c.name: [] for c in COARSE}
    medium_groups = {c.name: [] for c in COARSE}
    for f in FINE:
        fine_groups[get_compartment(f.cl_id)].append(f)
    for m in MEDIUM:
        medium_groups[get_compartment(m.cl_id)].append(m)

    # Spacing — fine column drives total height (most items per compartment)
    FINE_SPACING = 0.85
    MED_SPACING = 1.10
    COMP_PAD = 1.6  # extra blank space between compartments

    # Compute compartment Y centers from FINE column requirements
    fine_y = {}
    medium_y = {}
    coarse_y = {}
    cur_y = 1.6
    for c in COARSE:
        n_fine = len(fine_groups[c.name])
        n_med = len(medium_groups[c.name])
        # Compartment height driven by max(fine, med) layout
        h_fine = (n_fine - 1) * FINE_SPACING if n_fine > 1 else 0
        h_med = (n_med - 1) * MED_SPACING if n_med > 1 else 0
        comp_h = max(h_fine, h_med, 0.5)
        center = cur_y + comp_h / 2
        coarse_y[c.name] = center
        # Place mediums centered on compartment
        for j, m in enumerate(medium_groups[c.name]):
            offset = -(n_med - 1) / 2 * MED_SPACING + j * MED_SPACING
            medium_y[m.name] = center + offset
        # Place fines centered on compartment
        for j, f in enumerate(fine_groups[c.name]):
            offset = -(n_fine - 1) / 2 * FINE_SPACING + j * FINE_SPACING
            fine_y[f.name] = center + offset
        cur_y += comp_h + COMP_PAD

    total_h = cur_y + 1.0

    fig, ax = plt.subplots(figsize=(15.5, max(10.5, total_h * 0.55)))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.axis("off")

    # X positions for tier columns
    X_COARSE, X_MEDIUM, X_FINE = 0.5, 5.0, 10.0
    BOX_W = {"coarse": 2.2, "medium": 2.4, "fine": 1.95}

    # Draw column headers
    for x, label in [(X_COARSE, "COARSE\n(5)"), (X_MEDIUM, "MEDIUM\n(12)"), (X_FINE, "FINE\n(18)")]:
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=14, fontweight="bold",
                color="#1A4B7A")
    ax.text(0.5, 1.4, "CL Super-Coarse\nLevel 1", ha="center", va="center",
            fontsize=8, color="#666", style="italic")
    ax.text(5.0, 1.4, "CL Coarse + Medium\nLevels 2-3, body-wide", ha="center", va="center",
            fontsize=8, color="#666", style="italic")
    ax.text(10.0, 1.4, "CL Medium + Fine + pathology\nLevels 3-4 + DAPIDL: ext.",
            ha="center", va="center", fontsize=8, color="#666", style="italic")

    def box(x, y, w, h, color, label, cl_id, dashed=False):
        edge_style = "--" if dashed else "-"
        edge_lw = 2.0 if dashed else 0.8
        edge_color = "#9C2A2A" if dashed else "#444"
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                       boxstyle="round,pad=0.04,rounding_size=0.06",
                                       facecolor=color, edgecolor=edge_color,
                                       lw=edge_lw, linestyle=edge_style, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y - 0.10, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white")
        ax.text(x, y + 0.20, cl_id, ha="center", va="center", fontsize=7.5,
                color="white", style="italic", alpha=0.85)

    # Draw COARSE boxes
    for c in COARSE:
        box(X_COARSE, coarse_y[c.name], BOX_W["coarse"], 0.8,
            COMP_COLOR[c.name], c.name, c.cl_id)

    # Draw MEDIUM boxes + edges to parent COARSE
    for m in MEDIUM:
        comp = get_compartment(m.cl_id)
        color = COMP_COLOR[comp]
        # Edge from coarse to medium (curved)
        ax.plot([X_COARSE + BOX_W["coarse"]/2, X_MEDIUM - BOX_W["medium"]/2],
                [coarse_y[comp], medium_y[m.name]],
                color=color, lw=0.7, alpha=0.55, zorder=0)
        box(X_MEDIUM, medium_y[m.name], BOX_W["medium"], 0.7,
            color, m.name.replace("_", " "), m.cl_id)

    # Draw FINE boxes + edges to parent (medium or coarse)
    for f in FINE:
        comp = get_compartment(f.cl_id)
        color = COMP_COLOR[comp]
        is_pathology = f.cl_id.startswith("DAPIDL:")
        # Find parent y in MEDIUM column
        parent_y = None
        if f.parent_cl:
            # find parent in medium first
            for m in MEDIUM:
                if m.cl_id == f.parent_cl:
                    parent_y = medium_y[m.name]
                    break
            # else find in coarse
            if parent_y is None:
                for c in COARSE:
                    if c.cl_id == f.parent_cl:
                        parent_y = coarse_y[c.name]
                        break
        if parent_y is None:
            parent_y = coarse_y[comp]  # fallback: connect to coarse compartment

        ax.plot([X_MEDIUM + BOX_W["medium"]/2, X_FINE - BOX_W["fine"]/2],
                [parent_y, fine_y[f.name]],
                color=color, lw=0.6, alpha=0.5, zorder=0,
                linestyle="--" if is_pathology else "-")
        box(X_FINE, fine_y[f.name], BOX_W["fine"], 0.62,
            color, f.name.replace("_", " "), f.cl_id, dashed=is_pathology)

    # Title + legend
    ax.text(6.0, -0.5, "DAPIDL canonical 3-tier ontology — anchored to Cell Ontology",
            ha="center", va="bottom", fontsize=15, fontweight="bold", color="#1A4B7A")

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COMP_COLOR[c.name], edgecolor="#444", label=c.name)
        for c in COARSE
    ]
    legend_handles.append(
        Line2D([0], [0], marker="s", linestyle="none", markersize=12,
               markerfacecolor="#ED553B", markeredgecolor="#9C2A2A",
               markeredgewidth=2.0, label="DAPIDL pathology extension")
    )
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=6, fontsize=9, frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    draw()
