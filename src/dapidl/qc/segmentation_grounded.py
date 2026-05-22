"""Segmentation-grounded per-nucleus QC: reject obviously broken training patches.

Pure scoring functions (GPU-free) operate on a patch + a StarDist segmentation
(label mask + per-object prob). SegmentationGroundedScorer (later task) supplies
the segmentation. Lives in dapidl; imports the starpose.qc.base ABC read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegQCConfig:
    """Thresholds for the broken-patch rejector. Conservative = high specificity."""

    erode_px: int = 3
    min_interior_px: int = 20
    structure_floor: float = 0.05      # absolute floor on structure_raw
    area_min_um2: float = 8.0
    area_max_um2: float = 400.0        # above => merged/touching blob
    edge_px: int = 1                   # mask within this many px of frame => cut
    center_max_dist_frac: float = 0.35  # centroid dist / half-patch
    dominant_min_frac: float = 0.5     # target's share of the central box
    central_box_frac: float = 0.5      # central box size as fraction of patch
    prob_min: float = 0.40             # StarDist objectness floor
    solidity_min: float = 0.50         # below => debris-like
    eccentricity_max: float = 0.98     # above => line-like
    intensity_ratio_min: float = 1.10  # interior mean / background median
    structure_min: float = 0.15        # for the OPTIONAL structure cut (off by default)


@dataclass(frozen=True)
class CenterNucleus:
    """The chosen target nucleus in a patch."""

    label: int
    mask: np.ndarray   # bool (H, W)
    prob: float
    centroid: tuple[float, float]  # (y, x) px
    area_px: int


def select_center_nucleus(
    masks: np.ndarray, probs: np.ndarray, cfg: SegQCConfig
) -> CenterNucleus | None:
    """Pick the object covering the patch centre; else nearest centroid within
    a small radius; else None (no nucleus at centre)."""
    h, w = masks.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    center_label = int(masks[round(cy), round(cx)])
    if center_label == 0:
        # nearest object centroid within radius = center_max_dist_frac * half-patch
        radius = cfg.center_max_dist_frac * (h / 2.0)
        best, best_d = 0, radius
        for lab in range(1, int(masks.max()) + 1):
            m = masks == lab
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            d = float(np.hypot(ys.mean() - cy, xs.mean() - cx))
            if d < best_d:
                best, best_d = lab, d
        center_label = best
    if center_label == 0:
        return None
    m = masks == center_label
    ys, xs = np.nonzero(m)
    return CenterNucleus(
        label=center_label,
        mask=m,
        prob=float(probs[center_label - 1]),
        centroid=(float(ys.mean()), float(xs.mean())),
        area_px=int(m.sum()),
    )
