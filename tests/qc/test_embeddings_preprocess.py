import numpy as np

from dapidl.qc.embeddings import preprocess_dinov2, preprocess_nuspire


def test_preprocess_dinov2_shape_range_and_stretch():
    patch = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))
    out = preprocess_dinov2(patch, size=224)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_preprocess_dinov2_constant_patch_no_nan():
    patch = np.full((64, 64), 7, dtype=np.uint16)
    out = preprocess_dinov2(patch)
    assert np.isfinite(out).all()


def test_preprocess_nuspire_shape():
    patch = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    out = preprocess_nuspire(patch, size=112)
    assert out.shape == (1, 112, 112)
    assert np.isfinite(out).all()
