from __future__ import annotations

"""Cascade figure rendering utilities for STHELAR/MERSCOPE patch visualisation."""

from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a 2-D float32 or 3-D uint8 array to HxWx3 uint8 RGB."""
    if image.ndim == 2:
        # DAPI greyscale float32 [0,1] → gamma boost → 3-channel uint8
        boosted = np.clip(image.astype(np.float32) ** 0.7, 0.0, 1.0)
        grey = (boosted * 255).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=-1)
    if image.ndim == 3 and image.dtype != np.uint8:
        return np.clip(image, 0, 255).astype(np.uint8)
    return image


def _clip_poly(pts: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    """Clip polygon vertex coordinates to image bounds and return as int tuples."""
    clipped = pts.copy().astype(np.float64)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)
    return [(int(round(x)), int(round(y))) for x, y in clipped]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_annotation_overlay(
    image_2d_or_3d: np.ndarray,
    polygons_in_local_px: list[np.ndarray],
    polygon_colors: list[tuple[int, int, int]],
    polygon_kinds: list[str] | None = None,
    base_alpha: float = 0.5,
    line_width: float = 1.5,
) -> np.ndarray:
    """Render polygon annotations onto a patch image.

    DAPI (2-D float) is gamma-boosted and converted to RGB. Cell polygons receive
    a translucent fill (alpha=0.30) plus a semi-opaque outline; nucleus polygons
    are drawn as solid outlines only. Returns HxWx3 uint8.
    """
    rgb = _to_rgb_uint8(image_2d_or_3d)
    h, w = rgb.shape[:2]

    if not polygons_in_local_px:
        return rgb

    kinds = polygon_kinds if polygon_kinds is not None else ["cell"] * len(polygons_in_local_px)

    base_pil = Image.fromarray(rgb, mode="RGB")
    overlay = Image.new("RGBA", base_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lw = max(1, int(round(line_width)))

    for poly, color, kind in zip(polygons_in_local_px, polygon_colors, kinds):
        if len(poly) < 3:
            continue
        pts = _clip_poly(poly, h, w)
        if len(pts) < 3:
            continue
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        if kind == "nucleus":
            draw.polygon(pts, fill=None, outline=(r, g, b, 255))
        else:
            # cell: translucent fill + semi-opaque outline
            draw.polygon(pts, fill=(r, g, b, int(0.30 * 255)))
            draw.polygon(pts, fill=None, outline=(r, g, b, int(0.60 * 255)))

    result = Image.alpha_composite(base_pil.convert("RGBA"), overlay)
    return np.array(result.convert("RGB"), dtype=np.uint8)


def render_cascade_figure(
    panels: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    title: str,
    legend_items: list[tuple[str, tuple[int, int, int]]],
    dpi: int = 130,
) -> None:
    """Save a 2-row × N-column cascade figure to *output_path*.

    Row 1 shows raw images, row 2 shows annotated overlays at each patch size.
    A colour-square legend is rendered below the grid without overlapping panels.
    """
    sizes = sorted(panels.keys())
    n_cols = len(sizes)

    fig = plt.figure(figsize=(max(12, 2.2 * n_cols), 5.5), dpi=dpi)
    gs = GridSpec(
        3,
        n_cols,
        figure=fig,
        height_ratios=[10, 10, 1],
        hspace=0.35,
        wspace=0.04,
    )

    for col_idx, sz in enumerate(sizes):
        raw, overlay = panels[sz]

        ax_raw = fig.add_subplot(gs[0, col_idx])
        raw_rgb = _to_rgb_uint8(raw)
        ax_raw.imshow(raw_rgb, aspect="auto", interpolation="nearest")
        ax_raw.set_title(f"{sz} px", fontsize=8, pad=3)
        ax_raw.axis("off")

        ax_ann = fig.add_subplot(gs[1, col_idx])
        ax_ann.imshow(overlay, aspect="auto", interpolation="nearest")
        ax_ann.axis("off")

    # Legend row
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    if legend_items:
        patches = [
            mpatches.Patch(
                facecolor=tuple(c / 255 for c in rgb),
                edgecolor="none",
                label=label,
            )
            for label, rgb in legend_items
        ]
        ax_leg.legend(
            handles=patches,
            loc="center",
            ncol=min(len(legend_items), 8),
            frameon=False,
            fontsize=7,
            handlelength=1.2,
            handleheight=1.0,
            columnspacing=1.0,
        )

    fig.suptitle(title, fontsize=10, y=0.98)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
