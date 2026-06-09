# tests/test_graph_embed.py
import struct
import numpy as np
from dapidl.graph.embed import decode_record, pca_fit_transform


def test_decode_record_splits_label_and_patch():
    label = np.array([2], dtype=np.int64).tobytes()
    patch = (np.arange(128 * 128, dtype=np.uint16)).tobytes()
    lab, arr = decode_record(label + patch, patch_size=128)
    assert lab == 2
    assert arr.shape == (128, 128) and arr.dtype == np.uint16
    assert arr[0, 1] == 1


def test_pca_reduces_width_and_is_finite():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 64)).astype(np.float32)
    red, model = pca_fit_transform(emb, n_components=8, fit_sample=200, seed=0)
    assert red.shape == (500, 8)
    assert np.all(np.isfinite(red))
    # variance is ordered (first component explains the most)
    assert model.explained_variance_[0] >= model.explained_variance_[-1]
