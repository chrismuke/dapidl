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
