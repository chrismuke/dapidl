"""Shared figure style for the v2 deck.

Keep colors, fonts and sizes consistent across all figures so the deck looks
like one piece of work.

Color contract (cell types — same across all figures):
    Epithelial   #ED553B   warm red
    Immune       #3D5A80   navy
    Stromal      #BC8D5A   warm brown
    Endothelial  #4A9D9C   teal
    Neoplastic   #9C2A2A   deep red
    Adipocyte    #9DA17B   olive
    Mast         #C58CD3   muted purple

Modality contract:
    DAPI         #2E86AB   blue
    H&E          #B03A48   brick red
    DAPI+H&E     #6F4C9C   purple
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl


CELL_COLORS: dict[str, str] = {
    "Epithelial": "#ED553B",
    "epithelial cell": "#ED553B",
    "Epithelial_Luminal": "#ED553B",
    "Epithelial_Basal": "#F08775",
    "Epithelial_Tumor": "#9C2A2A",
    "Immune": "#3D5A80",
    "T cell": "#3D5A80",
    "T_Cell": "#3D5A80",
    "B cell": "#5677A1",
    "B_Cell": "#5677A1",
    "macrophage": "#7090BC",
    "Macrophage": "#7090BC",
    "Myeloid": "#7090BC",
    "NK_Cell": "#9DAFD0",
    "Stromal": "#BC8D5A",
    "fibroblast": "#BC8D5A",
    "Fibroblast": "#BC8D5A",
    "Stromal_Fibroblast": "#BC8D5A",
    "Pericyte": "#A07338",
    "Stromal_Pericyte": "#A07338",
    "Endothelial": "#4A9D9C",
    "endothelial cell": "#4A9D9C",
    "Vascular_Endothelial": "#4A9D9C",
    "adipocyte": "#9DA17B",
    "mast cell": "#C58CD3",
    "Mast_Cell": "#C58CD3",
    "Neoplastic": "#9C2A2A",
    "Specialized": "#9DA17B",
}

MODALITY_COLORS: dict[str, str] = {
    "DAPI": "#2E86AB",
    "H&E": "#B03A48",
    "HE": "#B03A48",
    "DAPI+H&E": "#6F4C9C",
    "DAPI+HE": "#6F4C9C",
    "Fusion": "#3F2E69",
}

PLATFORM_COLORS: dict[str, str] = {
    "Xenium": "#1F77B4",
    "STHELAR": "#D62728",
    "MERSCOPE": "#FF7F0E",
}


def apply_style() -> None:
    """One-time matplotlib rcParams setup. Call at the top of every figure script."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })


def cell_color(name: str, default: str = "#9E9E9E") -> str:
    return CELL_COLORS.get(name, default)


def annotate_bars(ax, fmt: str = "{:.2f}", fontsize: int = 10, dx: float = 0.0,
                  dy: float = 0.0) -> None:
    """Annotate every bar in `ax` with its value."""
    for p in ax.patches:
        value = p.get_width() if p.get_y() != 0 or p.get_height() < 1 else p.get_height()
        # Default: horizontal bar — width is the value
        if p.get_height() <= 1.0 and p.get_width() > 0:
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2.0
            ax.text(x + dx, y + dy, fmt.format(p.get_width()),
                    va="center", ha="left", fontsize=fontsize)
        else:
            x = p.get_x() + p.get_width() / 2.0
            y = p.get_height()
            ax.text(x + dx, y + dy, fmt.format(p.get_height()),
                    va="bottom", ha="center", fontsize=fontsize)
