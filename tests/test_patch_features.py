import numpy as np

from dapidl.qc.patch_features import haralick_features


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
