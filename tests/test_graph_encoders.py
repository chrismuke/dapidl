import numpy as np
import torch
from dapidl.graph.encoders import FrozenFeatureEncoder, CropCNNEncoder


def test_frozen_encoder_lookup_values_and_shape():
    feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    enc = FrozenFeatureEncoder(feats, device="cpu")
    assert enc.out_dim == 4
    out = enc.encode(np.array([0, 2, 4]))
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.tensor(feats[[0, 2, 4]]))


def test_frozen_encoder_projection_changes_out_dim():
    feats = np.random.RandomState(0).randn(6, 8).astype(np.float32)
    enc = FrozenFeatureEncoder(feats, device="cpu", proj_dim=3)
    assert enc.out_dim == 3
    assert enc.encode(np.array([1, 5])).shape == (2, 3)


def test_cropcnn_encoder_shape():
    crops = (np.random.RandomState(0).rand(7, 40, 40) * 65535).astype(np.uint16)
    enc = CropCNNEncoder(crops, device="cpu", out_dim=128)
    assert enc.out_dim == 128
    assert enc.encode(np.array([0, 3, 6])).shape == (3, 128)
