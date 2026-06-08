import numpy as np
from starpose.qc.segmentation_grounded import SegQCConfig

from dapidl.qc.patch_features import (
    CTX_COLUMNS,
    NUC_COLUMNS,
    haralick_features,
    nucleus_feature_vector,
)


def _disc(textured, size=128, r=20, seed=0):
    patch = np.full((size, size), 1000.0)
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= r ** 2
    if textured:
        rng = np.random.default_rng(seed)
        patch[mask] = rng.integers(500, 4000, size=int(mask.sum()))
    else:
        patch[mask] = 2500.0
    return patch, mask


def test_haralick_contrast_and_entropy_higher_on_textured():
    pt, m = _disc(True)
    pf, _ = _disc(False)
    ht = haralick_features(pt, m)
    hf = haralick_features(pf, m)
    assert ht["contrast"] > hf["contrast"]
    assert ht["entropy"] > hf["entropy"]


def test_haralick_returns_six_keys():
    pt, m = _disc(True)
    h = haralick_features(pt, m)
    assert set(h) == {"contrast", "homogeneity", "energy", "correlation", "asm", "entropy"}


def test_haralick_degenerate_small_mask_is_smooth():
    patch = np.ones((32, 32)) * 1000.0
    tiny = np.zeros((32, 32), bool)
    tiny[0, 0] = True
    h = haralick_features(patch, tiny)
    assert h["entropy"] == 0.0 and h["asm"] == 1.0


def test_feature_vector_textured_has_more_structure_than_flat():
    pt, m = _disc(True)
    pf, _ = _disc(False)
    ft = nucleus_feature_vector(pt, m, 0.9, SegQCConfig(), 0.2125)
    ff = nucleus_feature_vector(pf, m, 0.9, SegQCConfig(), 0.2125)
    assert ft["has_nucleus"] == 1.0
    assert ft["nuc_structure_raw"] > ff["nuc_structure_raw"]
    assert ft["nuc_contrast"] > ff["nuc_contrast"]


def test_feature_vector_area_fraction_exact():
    patch = np.ones((128, 128)) * 1000.0
    m = np.zeros((128, 128), bool)
    m[50:60, 50:70] = True  # 200 px
    f = nucleus_feature_vector(patch, m, 0.5, SegQCConfig(), 0.2125)
    assert abs(f["nuc_area_fraction"] - 200.0 / (128 * 128)) < 1e-9


def test_feature_vector_no_nucleus_is_nan_but_keeps_context():
    patch = np.ones((128, 128)) * 1000.0
    f = nucleus_feature_vector(patch, None, 0.0, SegQCConfig(), 0.2125)
    assert f["has_nucleus"] == 0.0
    assert np.isnan(f["nuc_area_fraction"])
    assert np.isnan(f["nuc_structure_raw"])
    assert "ctx_int_mean" in f and not np.isnan(f["ctx_int_mean"])


def test_feature_vector_stable_columns():
    pt, m = _disc(True)
    with_nuc = nucleus_feature_vector(pt, m, 0.9, SegQCConfig(), 0.2125)
    without = nucleus_feature_vector(pt, None, 0.0, SegQCConfig(), 0.2125)
    assert set(with_nuc) == set(without)
    for c in NUC_COLUMNS + CTX_COLUMNS + ["has_nucleus"]:
        assert c in with_nuc
