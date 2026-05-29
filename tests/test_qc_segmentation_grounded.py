import numpy as np
from dapidl.qc.segmentation_grounded import SegQCConfig, select_center_nucleus
from dapidl.qc.segmentation_grounded import structure_raw, structure_score


def _disk(h, w, cy, cx, r, label):
    yy, xx = np.ogrid[:h, :w]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    out = np.zeros((h, w), np.int32)
    out[m] = label
    return out


def test_select_center_nucleus_prefers_object_covering_center():
    masks = _disk(128, 128, 64, 64, 12, 1) + _disk(128, 128, 20, 20, 8, 2)
    probs = np.array([0.9, 0.8])
    cn = select_center_nucleus(masks, probs, SegQCConfig())
    assert cn is not None and cn.label == 1
    assert cn.prob == 0.9
    assert cn.mask.sum() > 0 and cn.mask[64, 64]


def test_select_center_nucleus_none_when_center_empty_and_far():
    masks = _disk(128, 128, 15, 15, 6, 1)  # only a far corner object
    cn = select_center_nucleus(masks, np.array([0.9]), SegQCConfig())
    assert cn is None


def test_select_center_nucleus_none_on_probs_underrun():
    masks = _disk(128, 128, 64, 64, 12, 3)  # label 3 covers centre
    cn = select_center_nucleus(masks, np.array([0.9]), SegQCConfig())  # only 1 prob
    assert cn is None


def test_structure_raw_textured_beats_flat_same_mean():
    h = w = 128
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 < 18 ** 2
    flat = np.full((h, w), 1000.0)
    textured = flat.copy()
    textured[mask] += ((np.indices((h, w)).sum(0) % 7) * 60.0)[mask]  # high-freq detail
    cfg = SegQCConfig()
    assert structure_raw(textured, mask, cfg) > 5 * structure_raw(flat, mask, cfg)


def test_structure_raw_zero_when_interior_too_small():
    cfg = SegQCConfig()
    tiny = np.zeros((128, 128), bool)
    tiny[64, 64] = True
    assert structure_raw(np.random.rand(128, 128) * 1000, tiny, cfg) == 0.0


def test_structure_score_calibrates_with_floor():
    cfg = SegQCConfig()
    # raw below floor -> 0; raw == p90 -> 1
    assert structure_score(0.0, ref_p90=2.0, cfg=cfg) == 0.0
    assert structure_score(2.0 + cfg.structure_floor, ref_p90=2.0, cfg=cfg) == 1.0


from dapidl.qc.segmentation_grounded import (
    centeredness_score, touches_edge, area_um2, dominant_central_fraction,
)


def test_centeredness_high_at_center_low_at_offset():
    cfg = SegQCConfig()
    assert centeredness_score((64.0, 64.0), (128, 128), cfg) > 0.95
    assert centeredness_score((110.0, 110.0), (128, 128), cfg) < 0.2


def test_touches_edge():
    cfg = SegQCConfig()
    m = np.zeros((128, 128), bool); m[60:70, 0:5] = True   # touches left frame
    assert touches_edge(m, cfg)
    m2 = np.zeros((128, 128), bool); m2[60:70, 60:70] = True
    assert not touches_edge(m2, cfg)


def test_area_um2():
    m = np.zeros((128, 128), bool); m[:10, :10] = True  # 100 px
    assert abs(area_um2(m, pixel_size=0.2125) - 100 * 0.2125 ** 2) < 1e-6


def test_dominant_central_fraction():
    cfg = SegQCConfig()
    big = np.zeros((128, 128), bool); big[48:80, 48:80] = True   # fills central box
    small = np.zeros((128, 128), bool); small[48:52, 48:52] = True
    f = dominant_central_fraction(target=big, all_masks=(big | small), cfg=cfg)
    assert f > 0.9


from dapidl.qc.segmentation_grounded import objectness_metrics


def test_objectness_round_bright_blob_is_object_like():
    cfg = SegQCConfig()
    yy, xx = np.ogrid[:128, :128]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 < 14 ** 2
    patch = np.full((128, 128), 300.0); patch[mask] = 1500.0
    om = objectness_metrics(patch, mask, prob=0.9, cfg=cfg)
    assert om["solidity"] > cfg.solidity_min
    assert om["eccentricity"] < cfg.eccentricity_max
    assert om["intensity_ratio"] > cfg.intensity_ratio_min
    assert om["objectness_score"] > 0.7


def test_objectness_low_prob_scores_low():
    cfg = SegQCConfig()
    yy, xx = np.ogrid[:128, :128]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 < 14 ** 2
    patch = np.full((128, 128), 300.0); patch[mask] = 1500.0
    om = objectness_metrics(patch, mask, prob=0.05, cfg=cfg)
    assert om["objectness_score"] < 0.4


from dapidl.qc.segmentation_grounded import score_from_segmentation, decide_broken


def _one_disk(cy, cx, r, label=1):
    yy, xx = np.ogrid[:128, :128]
    m = np.zeros((128, 128), np.int32); m[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = label
    return m


def test_decide_broken_no_nucleus():
    cfg = SegQCConfig()
    qs = score_from_segmentation(np.zeros((128, 128)), np.zeros((128, 128), np.int32),
                                 np.array([]), ref_p90=2.0, pixel_size=0.2125, cfg=cfg)
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason == "no_nucleus"


def test_decide_broken_cut_at_edge():
    cfg = SegQCConfig()
    masks = _one_disk(64, 64, 70)             # large disk: covers centre AND touches frame
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.9]), 2.0, 0.2125, cfg)
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason == "cut_at_edge"


def test_decide_broken_off_center():
    """Genuine off-center: nucleus bulk far from patch centre with a thin bridge
    over (64, 64) so select_center_nucleus picks it via the center pixel rule,
    but its centroid is > 4.76 µm (22.4 px) off centre -> centeredness <= 0."""
    cfg = SegQCConfig()
    yy, xx = np.ogrid[:128, :128]
    bulk = (yy - 25) ** 2 + (xx - 25) ** 2 < 18 ** 2          # large mass top-left
    bridge = (yy >= 60) & (yy <= 68) & (xx >= 60) & (xx <= 68)  # covers (64, 64)
    masks = np.zeros((128, 128), np.int32)
    masks[bulk | bridge] = 1
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.9]), 2.0, 0.2125, cfg)
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason == "off_center"


def test_dominant_central_no_longer_a_broken_reason():
    """Dense field around a well-centered nucleus must NOT be flagged as
    off_center (the old crowding-gate artifact). The neighbor is bright and big
    enough to dominate the central box, but the target is dead-centered and
    real -> should be 'ok'."""
    cfg = SegQCConfig()
    target = _one_disk(64, 64, 14, 1)
    neighbor = _one_disk(64, 90, 18, 2)
    masks = target + neighbor
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.95, 0.95]), 2.0, 0.2125, cfg)
    assert qs.metrics["dominant_central"] < 0.5  # neighbor dominates central box
    broken, reason = decide_broken(qs, cfg)
    assert not broken, f"unexpected broken={reason}"


def test_good_nucleus_not_broken_even_if_low_structure():
    cfg = SegQCConfig()
    masks = _one_disk(64, 64, 16, 1)
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0   # flat interior
    qs = score_from_segmentation(patch, masks, np.array([0.95]), 2.0, 0.2125, cfg)
    broken, reason = decide_broken(qs, cfg)            # structure cut OFF by default
    assert not broken
    assert qs.focus_score < 0.5                        # structure IS reported as low


from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer


def test_scorer_score_batch_uses_injected_segmentation(monkeypatch):
    sc = SegmentationGroundedScorer()
    # fake _segment: a centered disk + prob, independent of pixels
    yy, xx = np.ogrid[:128, :128]
    masks = np.zeros((128, 128), np.int32); masks[(yy - 64) ** 2 + (xx - 64) ** 2 < 16 ** 2] = 1
    monkeypatch.setattr(sc, "_segment", lambda p: (masks, np.array([0.95])))
    patch = np.full((1, 128, 128), 300.0, np.float64); patch[0][masks > 0] = 1500.0
    from starpose.qc.base import NormRef
    out = sc.score_batch(patch.astype(np.uint16), ref=NormRef(varlap_p90=2.0))
    assert len(out) == 1 and out[0].metrics["has_nucleus"] == 1.0
    assert sc.name == "segmentation_grounded"


def test_objectness_metrics_empty_mask_no_crash():
    """B3: an all-False mask must not crash regionprops()[0] (IndexError).
    Returns a neutral/zero objectness dict instead of detonating the batch."""
    cfg = SegQCConfig()
    patch = np.full((128, 128), 300.0)
    empty = np.zeros((128, 128), bool)
    om = objectness_metrics(patch, empty, prob=0.0, cfg=cfg)
    assert om["objectness_score"] == 0.0


def test_decide_broken_dim_real_nucleus_not_dropped():
    """B4 (censoring guard): a dim but real, well-centered, in-bounds, high-prob
    nucleus must NOT be hard-dropped as false_detection. Intensity is a SOFT
    objectness signal, not a hard gate -- faint immune/pyknotic nuclei are real."""
    cfg = SegQCConfig()
    masks = _one_disk(64, 64, 14, 1)
    patch = np.full((128, 128), 300.0)
    patch[masks > 0] = 312.0   # interior ~1.04x background -> intensity_ok False
    qs = score_from_segmentation(patch, masks, np.array([0.95]), 2.0, 0.2125, cfg)
    assert qs.metrics["intensity_ratio"] < cfg.intensity_ratio_min  # old hard gate would drop
    broken, reason = decide_broken(qs, cfg)
    assert not broken, f"dim real nucleus wrongly dropped as {reason}"


def test_decide_broken_irregular_real_nucleus_not_dropped():
    """B4 (censoring guard): a moderately irregular (solidity in
    [solidity_hard_min, solidity_min)) but high-prob, centered, in-bounds nucleus
    must NOT be hard-dropped. Only TRULY degenerate shapes (< solidity_hard_min)
    gate; moderate irregularity is a soft objectness multiplier only."""
    cfg = SegQCConfig()
    masks = np.zeros((128, 128), np.int32)
    masks[60:69, 35:94] = 1   # horizontal bar (len 59, width 9)
    masks[35:94, 60:69] = 1   # vertical bar -> a thin cross covering centre
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.9]), 2.0, 0.2125, cfg)
    assert qs.metrics["solidity"] < cfg.solidity_min        # moderate irregularity
    assert qs.metrics["solidity"] >= cfg.solidity_hard_min  # but not degenerate
    broken, reason = decide_broken(qs, cfg)
    assert not broken, f"irregular real nucleus wrongly dropped as {reason}"


def test_decide_broken_degenerate_shape_still_dropped():
    """B4: a genuinely degenerate sliver (solidity < solidity_hard_min) with high
    prob IS still dropped -- the extreme-shape hard gate is retained."""
    cfg = SegQCConfig()
    masks = np.zeros((128, 128), np.int32)
    # thin spiral-ish bracket: low solidity, but big enough to be in area bounds
    masks[20:110, 62:67] = 1
    masks[20:25, 62:100] = 1
    masks[105:110, 62:100] = 1
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.9]), 2.0, 0.2125, cfg)
    assert (qs.metrics["solidity"] < cfg.solidity_hard_min) \
        or (qs.metrics["eccentricity"] > cfg.eccentricity_hard_max)  # genuinely degenerate
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason == "false_detection"


from dapidl.qc.montage import build_reason_montage


def test_build_reason_montage_returns_rgb():
    patches = (np.random.rand(20, 128, 128) * 1000).astype(np.uint16)
    reasons = np.array(["off_center"] * 10 + ["cut_at_edge"] * 10, dtype=object)
    img = build_reason_montage(patches, reasons, reason="off_center", top_n=8)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
