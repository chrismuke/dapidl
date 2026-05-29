"""Tests for low-RAM lazy mosaic access (review B8)."""
import numpy as np

from dapidl.data.lazy_mosaic import LazyMosaic, normalize_crop


def test_normalize_crop_matches_legacy_full_normalize_then_crop():
    """Per-crop normalization must be byte-identical to normalizing the whole
    image then cropping (given the same percentiles) — the behavior-preserving
    guarantee for the memmap refactor."""
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300)) * 5000).astype(np.uint16)
    p_low = float(np.percentile(img, 1))
    p_high = float(np.percentile(img, 99.5))
    norm = np.clip(img.astype(np.float32), p_low, p_high)
    norm = (norm - p_low) / (p_high - p_low)
    legacy = (norm[50:80, 60:90] * 65535).clip(0, 65535).astype(np.uint16)
    new = normalize_crop(img[50:80, 60:90], p_low, p_high)
    assert np.array_equal(legacy, new)


def test_lazy_mosaic_read_2d_and_3d_equivalent():
    img = np.arange(100 * 120, dtype=np.uint16).reshape(100, 120)
    m2 = LazyMosaic(img)
    assert m2.shape == (100, 120)
    assert np.array_equal(m2.read(10, 20, 30, 40), img[10:20, 30:40])
    m3 = LazyMosaic(img[None, :, :])           # (1, H, W) like STHELAR
    assert m3.shape == (100, 120)
    assert np.array_equal(m3.read(10, 20, 30, 40), img[10:20, 30:40])


def test_subsample_percentiles_close_to_full():
    rng = np.random.default_rng(1)
    img = (rng.random((1000, 1000)) * 3000).astype(np.uint16)
    m = LazyMosaic(img)
    pl, ph = m.subsample_percentiles(1.0, 99.5, target_px=50_000)
    fl, fh = float(np.percentile(img, 1)), float(np.percentile(img, 99.5))
    assert abs(pl - fl) < 60 and abs(ph - fh) < 120


def test_normalize_crop_degenerate_flat_is_safe():
    flat = np.full((10, 10), 500, np.uint16)
    out = normalize_crop(flat, 500.0, 500.0)   # p_high <= p_low -> guarded
    assert out.dtype == np.uint16 and out.shape == (10, 10)


def test_lazy_mosaic_multichannel_reads_channel_0():
    """A (C,H,W) multi-stain morpho (e.g. STHELAR s6 is 5-channel) -> read DAPI =
    channel 0, matching SthelarDataReader._load_dapi's arr[0]."""
    h, w = 40, 50
    arr = np.zeros((5, h, w), np.uint16)
    arr[0] = np.arange(h * w, dtype=np.uint16).reshape(h, w)   # DAPI channel
    arr[1:] = 9999                                             # other stains -> must be ignored
    m = LazyMosaic(arr)
    assert m.shape == (h, w)
    assert np.array_equal(m.read(5, 15, 5, 25), arr[0, 5:15, 5:25])
    pl, ph = m.subsample_percentiles(1.0, 99.5, target_px=500)
    assert ph < 9000   # came from channel 0, not the 9999 stains


def test_lazy_mosaic_rejects_bad_ndim():
    import pytest
    with pytest.raises(ValueError):
        LazyMosaic(np.zeros((2, 3, 10, 10), np.uint16))   # 4D not supported
