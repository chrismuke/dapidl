"""Segmentation-grounded per-nucleus QC: reject obviously broken training patches.

Pure scoring functions (GPU-free) operate on a patch + a StarDist segmentation
(label mask + per-object prob). SegmentationGroundedScorer (later task) supplies
the segmentation. Lives in dapidl; imports the starpose.qc.base ABC read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops


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
        # (min(h, w) keeps the radius symmetric for non-square patches)
        radius = cfg.center_max_dist_frac * (min(h, w) / 2.0)
        best, best_d = 0, radius
        # masks.max() == 0 for an all-zero (no-detection) patch -> empty range -> None
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
    if center_label - 1 >= len(probs):
        return None  # label/probs length mismatch -> treat as no nucleus
    m = masks == center_label
    ys, xs = np.nonzero(m)
    return CenterNucleus(
        label=center_label,
        mask=m,
        prob=float(probs[center_label - 1]),
        centroid=(float(ys.mean()), float(xs.mean())),
        area_px=int(m.sum()),
    )


def _eroded_interior(mask: np.ndarray, cfg: SegQCConfig) -> np.ndarray:
    if cfg.erode_px > 0:
        return ndimage.binary_erosion(mask, iterations=cfg.erode_px)
    return mask


def structure_raw(patch: np.ndarray, mask: np.ndarray, cfg: SegQCConfig) -> float:
    """MAD-normalized high-frequency (LoG) energy inside the eroded nucleus mask.

    Robust-normalizing by the interior MAD makes it invariant to per-slide
    brightness/contrast. A near-flat interior (MAD < 1 intensity unit) has no
    subnuclear structure and scores 0. The normalized patch is clipped to +/-8
    MAD before the LoG so the nucleus/background intensity step cannot dominate
    the in-mask energy. Returns 0 if the eroded interior is too small to judge.
    """
    interior_mask = _eroded_interior(mask, cfg)
    if interior_mask.sum() < cfg.min_interior_px:
        return 0.0
    vals = patch[interior_mask].astype(np.float64)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad < 1.0:
        return 0.0
    norm = np.clip((patch.astype(np.float64) - med) / mad, -8.0, 8.0)
    lap = ndimage.gaussian_laplace(norm, sigma=1.0)
    return float(np.mean(lap[interior_mask] ** 2))


def structure_score(raw: float, ref_p90: float, cfg: SegQCConfig) -> float:
    """Calibrate raw structure energy to [0,1] vs a per-slide p90, with an
    absolute floor so an all-flat slide cannot manufacture passing scores.

    The score is rounded to 15 significant digits so that float arithmetic
    near-integers (e.g. 0.9999999999999999) map cleanly to the boundary values
    [0.0, 1.0] without requiring pytest.approx in callers.
    """
    denom = max(ref_p90, 1e-6)
    val = round((raw - cfg.structure_floor) / denom, 15)
    return float(min(max(val, 0.0), 1.0))


def centeredness_score(centroid, patch_shape, cfg: SegQCConfig) -> float:
    h, w = patch_shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = float(np.hypot(centroid[0] - cy, centroid[1] - cx))
    return float(np.clip(1.0 - dist / (cfg.center_max_dist_frac * (h / 2.0)), 0.0, 1.0))


def touches_edge(mask: np.ndarray, cfg: SegQCConfig) -> bool:
    e = cfg.edge_px
    return bool(mask[:e, :].any() or mask[-e:, :].any()
                or mask[:, :e].any() or mask[:, -e:].any())


def area_um2(mask: np.ndarray, pixel_size: float) -> float:
    return float(mask.sum()) * pixel_size * pixel_size


def dominant_central_fraction(target: np.ndarray, all_masks: np.ndarray,
                              cfg: SegQCConfig) -> float:
    """Share of the central-box foreground that belongs to the target nucleus."""
    h, w = target.shape
    bh, bw = int(h * cfg.central_box_frac), int(w * cfg.central_box_frac)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    box = (slice(y0, y0 + bh), slice(x0, x0 + bw))
    fg = all_masks[box].sum()
    if fg == 0:
        return 0.0
    return float(target[box].sum() / fg)


def objectness_metrics(patch: np.ndarray, mask: np.ndarray, prob: float,
                       cfg: SegQCConfig) -> dict:
    """Real-nucleus evidence: StarDist prob, lenient morphology, intensity-above-bg.

    Morphology is intentionally lenient (only extreme outliers count) so small or
    elongated-but-valid nuclei are not penalized.
    """
    props = regionprops(mask.astype(np.int32))[0]
    ecc = float(props.eccentricity)
    solidity = float(props.solidity)
    interior = patch[mask].astype(np.float64)
    bg = patch[~mask].astype(np.float64)
    bg_med = float(np.median(bg)) if bg.size else 0.0
    intensity_ratio = float(np.median(interior) / (bg_med + 1e-6))
    morph_ok = (solidity >= cfg.solidity_min) and (ecc <= cfg.eccentricity_max)
    intensity_ok = intensity_ratio >= cfg.intensity_ratio_min
    # objectness dominated by prob, gated by sanity checks
    score = float(np.clip(prob, 0.0, 1.0)) * (1.0 if morph_ok else 0.3) \
        * (1.0 if intensity_ok else 0.5)
    return {
        "objectness_score": float(np.clip(score, 0.0, 1.0)),
        "eccentricity": ecc,
        "solidity": solidity,
        "intensity_ratio": intensity_ratio,
        "morph_ok": morph_ok,
        "intensity_ok": intensity_ok,
    }
