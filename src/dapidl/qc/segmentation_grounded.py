"""Segmentation-grounded per-nucleus QC: reject obviously broken training patches.

Pure scoring functions (GPU-free) operate on a patch + a StarDist segmentation
(label mask + per-object prob). SegmentationGroundedScorer (later task) supplies
the segmentation. Lives in dapidl; imports the starpose.qc.base ABC read-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from starpose.qc.base import NormRef, QualityScore, QualityScorer


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
    solidity_min: float = 0.50         # SOFT: below => objectness down-weighted
    eccentricity_max: float = 0.98     # SOFT: above => objectness down-weighted
    intensity_ratio_min: float = 1.10  # SOFT: interior mean / background median
    # HARD morphology gate -- only GENUINELY degenerate shapes are dropped, so a
    # moderately irregular / small / dim but real nucleus is never censored
    # (the spec's #1 risk: class-correlated false drops of faint immune/pyknotic).
    solidity_hard_min: float = 0.30    # below => debris/sliver (hard false_detection)
    eccentricity_hard_max: float = 0.995  # above => degenerate line (hard false_detection)
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
    """Absolute variance of the Laplacian-of-Gaussian inside the eroded nucleus.

    No MAD normalization: that previously rewarded dim/flat nuclei with one
    bright speckle (high *relative* gradient over a near-flat background) and
    flagged bright, in-focus, well-textured nuclei as merely middling (because
    their large absolute gradient was divided by an equally large MAD). Using
    var(LoG) on the raw patch makes both brightness AND sharpness contribute:
    a dim or blurry nucleus has small absolute LoG values -> low variance;
    a bright, sharp, textured nucleus has large absolute LoG values -> high
    variance. The per-slide structure_score calibration (p90 + floor) then
    maps that absolute scale to [0, 1].
    """
    interior_mask = _eroded_interior(mask, cfg)
    if interior_mask.sum() < cfg.min_interior_px:
        return 0.0
    # Flat-interior guard: if the interior intensity itself has near-zero MAD
    # the nucleus has no subnuclear structure (e.g. apical/basal Z-cap), even
    # if the LoG picks up the nucleus/background boundary that leaks through
    # the eroded mask. This zeros it out at the source without touching the
    # absolute scale of the LoG variance.
    vals = patch[interior_mask].astype(np.float64)
    if float(np.median(np.abs(vals - np.median(vals)))) < 1.0:
        return 0.0
    lap = ndimage.gaussian_laplace(patch.astype(np.float64), sigma=1.0)
    return float(np.var(lap[interior_mask]))


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
    if not mask.any():
        # degenerate (empty) selection -- never crash the batch; zero objectness.
        return {
            "objectness_score": 0.0, "eccentricity": 1.0, "solidity": 0.0,
            "intensity_ratio": 0.0, "morph_ok": False, "intensity_ok": False,
        }
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


def score_from_segmentation(patch, masks, probs, ref_p90, pixel_size,
                            cfg: SegQCConfig) -> QualityScore:
    """Compose sub-scores into a QualityScore (no GPU). All raw signals are
    stored in metrics; broken/reason is decided separately by decide_broken."""
    cn = select_center_nucleus(masks, probs, cfg)
    if cn is None:
        return QualityScore(focus_score=0.0, detection_score=0.0, qc_score=0.0,
                            metrics={"has_nucleus": 0.0})
    s_raw = structure_raw(patch, cn.mask, cfg)
    struct = structure_score(s_raw, ref_p90, cfg)
    cent = centeredness_score(cn.centroid, patch.shape, cfg)
    a_um2 = area_um2(cn.mask, pixel_size)
    edge = touches_edge(cn.mask, cfg)
    dom = dominant_central_fraction(cn.mask, masks > 0, cfg)
    obj = objectness_metrics(patch, cn.mask, cn.prob, cfg)
    completeness = float(
        (not edge) and (cfg.area_min_um2 <= a_um2 <= cfg.area_max_um2)
    )
    qc = min(struct, cent, obj["objectness_score"])  # combined headline (reporting)
    return QualityScore(
        focus_score=struct, detection_score=obj["objectness_score"], qc_score=qc,
        metrics={
            "has_nucleus": 1.0, "structure_raw": s_raw, "centeredness": cent,
            "dominant_central": dom, "completeness": completeness,
            "area_um2": a_um2, "edge_cut": float(edge),
            "stardist_prob": cn.prob, "eccentricity": obj["eccentricity"],
            "solidity": obj["solidity"], "intensity_ratio": obj["intensity_ratio"],
            "morph_ok": float(obj["morph_ok"]), "intensity_ok": float(obj["intensity_ok"]),
        },
    )


def decide_broken(qs: QualityScore, cfg: SegQCConfig,
                  use_structure_cut: bool = False) -> tuple[bool, str]:
    """High-specificity broken decision. Order = most severe first. structure is
    never the sole reason unless use_structure_cut is explicitly enabled."""
    m = qs.metrics
    if m.get("has_nucleus", 0.0) < 1.0:
        return True, "no_nucleus"
    if m["edge_cut"] >= 1.0:
        return True, "cut_at_edge"
    # off_center fires ONLY on a real centeredness defect (StarDist centroid >
    # center_max_dist_frac * half-patch off centre). The previous version
    # OR'd in `dominant_central < 0.5`, but that gated on a CROWDING signal
    # (neighbors taking >50% of the inner-box foreground) and wrongly flagged
    # well-centered nuclei in dense epithelial / immune fields. dominant_central
    # is still computed and reported for diagnostics, just not a drop reason.
    if m["centeredness"] <= 0.0:
        return True, "off_center"
    # HARD gate: StarDist confidence + area sanity + GENUINELY degenerate shape.
    # morph_ok / intensity_ok stay SOFT (objectness-score multipliers in
    # objectness_metrics) and are reported, but are NOT drop reasons -- so a dim
    # or moderately irregular but real nucleus is never censored (spec's #1 risk).
    if (m["stardist_prob"] < cfg.prob_min) \
            or (m["solidity"] < cfg.solidity_hard_min) \
            or (m["eccentricity"] > cfg.eccentricity_hard_max) \
            or not (cfg.area_min_um2 <= m["area_um2"] <= cfg.area_max_um2):
        return True, "false_detection"   # low conf / sliver / degenerate / merged-blob
    if use_structure_cut and qs.focus_score < cfg.structure_min:
        return True, "no_structure"
    return False, "ok"


class SegmentationGroundedScorer(QualityScorer):
    """StarDist-grounded broken-patch rejector. v1: StarDist only."""

    def __init__(self, cfg: SegQCConfig | None = None, gpu: bool = True,
                 pixel_size: float = 0.2125):
        self.cfg = cfg or SegQCConfig()
        self.gpu = gpu
        self.pixel_size = pixel_size
        self._model = None

    @property
    def name(self) -> str:
        return "segmentation_grounded"

    def _get_model(self):
        if self._model is None:
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            from stardist.models import StarDist2D
            self._model = StarDist2D.from_pretrained("2D_versatile_fluo")
        return self._model

    def _segment(self, patch: np.ndarray):
        """Return (label mask int32, per-object prob array). Label k -> prob[k-1]."""
        model = self._get_model()
        p_low, p_high = np.percentile(patch, [1, 99.8])
        if p_high - p_low < 1e-6:
            return np.zeros(patch.shape, np.int32), np.array([])
        img = ((patch.astype(np.float32) - p_low) / (p_high - p_low)).clip(0, 1)
        labels, details = model.predict_instances(img)
        return labels.astype(np.int32), np.asarray(details["prob"], dtype=float)

    def fit_reference(self, patches: np.ndarray) -> NormRef:
        """Per-slide structure reference: p90 of structure_raw over a sample."""
        raws = []
        for p in patches:
            masks, probs = self._segment(p)
            cn = select_center_nucleus(masks, probs, self.cfg)
            if cn is not None:
                raws.append(structure_raw(p, cn.mask, self.cfg))
        p90 = float(np.percentile(raws, 90)) if raws else 1.0
        return NormRef(varlap_p90=p90)  # field reused to hold structure-raw p90

    def score_batch(self, patches: np.ndarray, ref: NormRef | None = None):
        if ref is None:
            ref = self.fit_reference(patches)
        out = []
        for p in patches:
            masks, probs = self._segment(p)
            out.append(score_from_segmentation(
                p, masks, probs, ref.varlap_p90, self.pixel_size, self.cfg))
        return out
