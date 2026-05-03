"""DAPIDL ontology sankey — raw source labels → CL fine class → CL super-coarse.

3-column Sankey-style flow diagram showing how raw cell-type labels from
both Janesick GT and STHELAR ct_tangram resolve through the canonical CL
hierarchy. Flow width is proportional to total cell count.

Top 30 source labels are shown explicitly; the remaining long tail is
grouped into "Other (raw)" per source.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import zarr

from _style import apply_style
from dapidl.ontology.cl_mapper import get_mapper
from dapidl.ontology.training_tiers import (
    COARSE_NAMES, FINE_NAMES, derive_tier_label,
)

ROOT = Path("/mnt/work/git/dapidl")
OUT = ROOT / "pipeline_output" / "figures_v2" / "fig_ontology_sankey.png"

# Compartment palette (matches hierarchy figure)
COMP_COLOR = {
    "Epithelial":  "#ED553B",
    "Immune":      "#3D5A80",
    "Stromal":     "#BC8D5A",
    "Endothelial": "#4A9D9C",
    "Neural":      "#9C2A2A",
    "Unknown":     "#9E9E9E",
}

TOP_N_SOURCE_LABELS = 50


def collect_raw_labels() -> dict[tuple[str, str], int]:
    """Returns (source, raw_label) → cell count, summed across all slides."""
    out: Counter = Counter()
    # STHELAR
    for sd in Path("/mnt/work/datasets/STHELAR/sdata_slides").glob("sdata_*.zarr"):
        inner = sd / f"{sd.stem}.zarr"
        root = inner if (inner / "images").exists() else sd
        z = zarr.open(str(root), mode="r")
        obs = z["tables/table_nuclei/obs"]
        if "ct_tangram" not in obs:
            continue
        node = obs["ct_tangram"]
        if isinstance(node, zarr.Group):
            cats = node["categories"][:]
            codes = node["codes"][:]
            labels = [str(cats[c]) if 0 <= c < len(cats) else "Unknown" for c in codes]
        else:
            labels = [str(v) for v in node[:]]
        for raw, cnt in Counter(labels).items():
            out[("STHELAR", raw)] += cnt
    # Janesick
    import pandas as pd
    for rep in (1, 2):
        path = (Path("/mnt/work/datasets/raw/xenium")
                / f"xenium-breast-tumor-rep{rep}"
                / f"celltypes_ground_truth_rep{rep}_supervised.xlsx")
        df = pd.read_excel(path)
        for raw, cnt in Counter(df["Cluster"].astype(str)).items():
            out[("Janesick", raw)] += cnt
    return out


def build_flows():
    """Return (source_node, fine_node, coarse_node, count) tuples."""
    mapper = get_mapper()
    raw_counts = collect_raw_labels()

    # Per source: pick top N raw labels by count, group rest into Other
    by_source: dict[str, Counter] = defaultdict(Counter)
    for (src, raw), cnt in raw_counts.items():
        by_source[src][raw] += cnt

    flows = []
    for src, ctr in by_source.items():
        top = ctr.most_common(TOP_N_SOURCE_LABELS)
        kept = {k for k, _ in top}
        for raw, cnt in ctr.items():
            label = raw if raw in kept else f"Other ({src})"
            fine = derive_tier_label(raw, "fine", mapper)
            coarse = derive_tier_label(raw, "coarse", mapper)
            flows.append((f"{label} [{src[0]}]", fine, coarse, cnt))
    return flows


def render():
    apply_style()
    flows = build_flows()
    src_to_total = Counter()
    fine_to_total = Counter()
    coarse_to_total = Counter()
    for s, f, c, n in flows:
        src_to_total[s] += n
        fine_to_total[f] += n
        coarse_to_total[c] += n

    # Column orderings — fine + coarse follow canonical order with extras at end
    sources = [s for s, _ in src_to_total.most_common()]
    fines = list(FINE_NAMES) + sorted(set(fine_to_total) - set(FINE_NAMES) - {"Unknown"})
    if "Unknown" in fine_to_total:
        fines.append("Unknown")
    fines = [f for f in fines if f in fine_to_total]
    coarses = list(COARSE_NAMES) + (["Unknown"] if "Unknown" in coarse_to_total else [])
    coarses = [c for c in coarses if c in coarse_to_total]

    # Compute Y positions (top→bottom, height proportional to count)
    total_cells = sum(src_to_total.values())
    H = 100.0   # canvas height
    PAD = 0.4   # blank space between nodes (in canvas units)

    def positions(items, totals, h):
        n = len(items)
        used = sum(totals[i] for i in items) or 1
        # available height after padding
        avail = h - PAD * (n - 1)
        scale = avail / used
        ys = []
        cur = 0.0
        for it in items:
            ht = max(totals[it] * scale, 0.4)  # min visible height
            ys.append((cur, cur + ht))
            cur += ht + PAD
        return ys

    src_pos = positions(sources, src_to_total, H)
    fine_pos = positions(fines, fine_to_total, H)
    coarse_pos = positions(coarses, coarse_to_total, H)

    # Track running offsets for each node so flows stack
    src_offsets = {s: src_pos[i][0] for i, s in enumerate(sources)}
    fine_offsets_in = {f: fine_pos[i][0] for i, f in enumerate(fines)}
    fine_offsets_out = {f: fine_pos[i][0] for i, f in enumerate(fines)}
    coarse_offsets = {c: coarse_pos[i][0] for i, c in enumerate(coarses)}

    # Group flows by (src, fine) and (fine, coarse) for cleaner stacking
    src_fine_flow: dict[tuple[str, str], int] = Counter()
    fine_coarse_flow: dict[tuple[str, str], int] = Counter()
    for s, f, c, n in flows:
        src_fine_flow[(s, f)] += n
        fine_coarse_flow[(f, c)] += n

    fig, ax = plt.subplots(figsize=(20.0, 22.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, H + 3)
    ax.invert_yaxis()
    ax.axis("off")

    # X positions and box widths
    X_SRC = 12
    X_FINE = 50
    X_COARSE = 88
    BOX_W = 1.8

    def draw_node(x, y0, y1, color, label, count, side="left"):
        rect = mpatches.Rectangle((x - BOX_W/2, y0), BOX_W, y1 - y0,
                                   facecolor=color, edgecolor="white", lw=0.8, zorder=3)
        ax.add_patch(rect)
        cnt_str = f"{count/1000:.0f}k" if count < 1e6 else f"{count/1e6:.1f}M"
        if side == "left":
            # Label outside the box on the left, with count appended
            ax.text(x - BOX_W - 0.6, (y0 + y1) / 2,
                    f"{label}  ·  {cnt_str}",
                    ha="right", va="center", fontsize=9.5, color="#222")
        elif side == "right":
            ax.text(x + BOX_W + 0.6, (y0 + y1) / 2,
                    f"{label}   ({cnt_str})",
                    ha="left", va="center", fontsize=12, color="#222", fontweight="bold")
        else:  # middle
            # Inline name only if box is tall enough
            box_h = y1 - y0
            if box_h > 1.5:
                ax.text(x, (y0 + y1) / 2, label, ha="center", va="center",
                        fontsize=9.0, color="white", fontweight="bold")
            else:
                # Tiny box — label outside on right
                ax.text(x + BOX_W + 0.4, (y0 + y1) / 2, label,
                        ha="left", va="center", fontsize=8.0, color="#444")
            # Count above box
            ax.text(x, y0 - 0.25, cnt_str, ha="center", va="bottom",
                    fontsize=7.5, color="#666")

    # Draw source nodes
    for s, (y0, y1) in zip(sources, src_pos):
        # pick color from the dominant compartment of this source
        comp_for_color = "Unknown"
        comp_max = 0
        for (ss, ff, cc, n) in flows:
            if ss == s and n > comp_max:
                comp_max = n
                comp_for_color = cc
        draw_node(X_SRC, y0, y1, COMP_COLOR.get(comp_for_color, "#888"),
                  s, src_to_total[s], side="left")

    # Draw fine nodes (color by compartment ancestry)
    fine_to_compartment = {}
    for s, f, c, _ in flows:
        if f not in fine_to_compartment:
            fine_to_compartment[f] = c
    for f, (y0, y1) in zip(fines, fine_pos):
        comp = fine_to_compartment.get(f, "Unknown")
        draw_node(X_FINE, y0, y1, COMP_COLOR.get(comp, "#888"),
                  f, fine_to_total[f], side="middle")

    # Draw coarse nodes
    for c, (y0, y1) in zip(coarses, coarse_pos):
        draw_node(X_COARSE, y0, y1, COMP_COLOR.get(c, "#888"),
                  c, coarse_to_total[c], side="right")

    # ---- Bezier flows -------------------------------------------------
    def bezier_band(x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.22):
        """Filled cubic-bezier band between (x0,[y0a,y0b]) and (x1,[y1a,y1b])."""
        n = 30
        xs = np.linspace(x0, x1, n)
        # interpolation parameter
        t = (xs - x0) / (x1 - x0)
        # cubic Bezier with control points at horizontal midpoint for smooth S
        s = 3 * t**2 - 2 * t**3  # smoothstep
        ya = y0a + (y1a - y0a) * s
        yb = y0b + (y1b - y0b) * s
        path_x = np.concatenate([xs, xs[::-1]])
        path_y = np.concatenate([ya, yb[::-1]])
        ax.fill(path_x, path_y, color=color, alpha=alpha, edgecolor="none", zorder=1)

    # Compute scaling: total height = H; map count → height
    # Use the same scale for each column based on total cells in that column
    src_scale = (H - PAD * (len(sources) - 1)) / max(sum(src_to_total.values()), 1)
    fine_scale = (H - PAD * (len(fines) - 1)) / max(sum(fine_to_total.values()), 1)
    coarse_scale = (H - PAD * (len(coarses) - 1)) / max(sum(coarse_to_total.values()), 1)

    # Source → Fine flows
    src_running = {s: src_pos[i][0] for i, s in enumerate(sources)}
    fine_running = {f: fine_pos[i][0] for i, f in enumerate(fines)}
    # Sort flows so that destination ordering matches column ordering for cleaner band layout
    sf_ordered = sorted(src_fine_flow.items(), key=lambda kv: (sources.index(kv[0][0]), fines.index(kv[0][1])))
    for (src, fine), cnt in sf_ordered:
        h_src = cnt * src_scale
        h_fine = cnt * fine_scale
        y0a = src_running[src]
        y0b = y0a + h_src
        y1a = fine_running[fine]
        y1b = y1a + h_fine
        comp = fine_to_compartment.get(fine, "Unknown")
        bezier_band(X_SRC + BOX_W/2, y0a, y0b, X_FINE - BOX_W/2, y1a, y1b,
                    COMP_COLOR.get(comp, "#888"))
        src_running[src] += h_src
        fine_running[fine] += h_fine

    # Fine → Coarse flows
    fine_running2 = {f: fine_pos[i][0] for i, f in enumerate(fines)}
    coarse_running = {c: coarse_pos[i][0] for i, c in enumerate(coarses)}
    fc_ordered = sorted(fine_coarse_flow.items(), key=lambda kv: (fines.index(kv[0][0]), coarses.index(kv[0][1])))
    for (fine, coarse), cnt in fc_ordered:
        h_fine = cnt * fine_scale
        h_coarse = cnt * coarse_scale
        y0a = fine_running2[fine]
        y0b = y0a + h_fine
        y1a = coarse_running[coarse]
        y1b = y1a + h_coarse
        bezier_band(X_FINE + BOX_W/2, y0a, y0b, X_COARSE - BOX_W/2, y1a, y1b,
                    COMP_COLOR.get(coarse, "#888"))
        fine_running2[fine] += h_fine
        coarse_running[coarse] += h_coarse

    # Column headers
    ax.text(X_SRC, -1.5, "Source labels\n(Janesick + STHELAR ct_tangram)",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1A4B7A")
    ax.text(X_FINE, -1.5, "Cell Ontology FINE  ({} classes)".format(len(fines)),
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1A4B7A")
    ax.text(X_COARSE, -1.5, "Super-COARSE  ({} compartments)".format(len(coarses)),
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1A4B7A")
    ax.text(50, H + 1.5,
            f"Total cells flowing through ontology: {total_cells/1e6:.1f}M  ·  "
            f"{len(sources)} source labels shown (rest grouped into 'Other')",
            ha="center", va="top", fontsize=10, color="#666", style="italic")

    # Suptitle
    fig.suptitle("DAPIDL ontology mapping — raw source labels → CL fine → CL super-coarse",
                 fontsize=15, fontweight="bold", color="#1A4B7A", y=0.98)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
