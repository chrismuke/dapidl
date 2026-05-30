"""Tests for the NuSPIRe DAPI-native ViT-MAE backbone wrapper.

Uses a tiny randomly-initialized ViT-MAE encoder injected via `encoder=` so the
contract is pinned without downloading the 400 MB checkpoint or needing a GPU.
The real-weights path is exercised by the GPU smoke (scripts/nuspire_smoke.py).
"""
import pytest
import torch

from dapidl.models.nuspire import NuSPIReBackbone


def _tiny_encoder():
    # hidden_size 32 / 2 layers keeps the unit tests fast; geometry (112/patch 8,
    # 1 channel, mask_ratio 0) matches the real NuSPIRe so the wrapper logic is
    # exercised faithfully.
    from transformers import ViTMAEConfig, ViTMAEModel

    cfg = ViTMAEConfig(
        image_size=112,
        patch_size=8,
        num_channels=1,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        mask_ratio=0.0,
    )
    return ViTMAEModel(cfg)


def test_forward_resizes_128_patch_to_feature_vector():
    bb = NuSPIReBackbone(encoder=_tiny_encoder())
    out = bb(torch.randn(4, 1, 128, 128))  # DAPIDL p128 input -> resized to 112
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_forward_native_112_no_resize():
    bb = NuSPIReBackbone(encoder=_tiny_encoder())
    out = bb(torch.randn(2, 1, 112, 112))
    assert out.shape == (2, 32)


def test_num_features_taken_from_encoder_config():
    bb = NuSPIReBackbone(encoder=_tiny_encoder())
    assert bb.num_features == 32


def test_features_deterministic_in_eval():
    # ViT-MAE shuffles patches even at mask_ratio=0; the wrapper must still yield
    # identical features across forwards (explicit identity noise + order-invariant
    # mean pooling). A flaky encoder would break downstream A/B reproducibility.
    bb = NuSPIReBackbone(encoder=_tiny_encoder()).eval()
    x = torch.randn(3, 1, 112, 112)
    a = bb(x)
    b = bb(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_pool_cls_shape():
    bb = NuSPIReBackbone(encoder=_tiny_encoder(), pool="cls")
    out = bb(torch.randn(2, 1, 112, 112))
    assert out.shape == (2, 32)


def test_invalid_pool_raises():
    with pytest.raises(ValueError):
        NuSPIReBackbone(encoder=_tiny_encoder(), pool="bogus")


def test_gradients_reach_encoder():
    bb = NuSPIReBackbone(encoder=_tiny_encoder())
    out = bb(torch.randn(2, 1, 112, 112))
    out.sum().backward()
    grads = [p.grad is not None for p in bb.encoder.parameters() if p.requires_grad]
    assert any(grads)  # fine-tunable end-to-end


def test_force_no_masking_neutralizes_injected_mask_ratio():
    # Coverage for the _force_no_masking guard (ultracode review P2): every other
    # fixture hardcodes mask_ratio=0.0, so the branch that neutralizes a >0 ratio
    # was never asserted. Inject mask_ratio=0.5 and prove the guard pins it to 0 so
    # ALL patch tokens (1 CLS + _n_patches) reach the mean-pool — not a masked 50%.
    from transformers import ViTMAEConfig, ViTMAEModel

    cfg = ViTMAEConfig(
        image_size=112, patch_size=8, num_channels=1, hidden_size=32,
        num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        mask_ratio=0.5,  # deliberately > 0
    )
    enc = ViTMAEModel(cfg)
    assert enc.config.mask_ratio == 0.5  # precondition: guard has not run yet

    bb = NuSPIReBackbone(encoder=enc)
    assert bb.encoder.config.mask_ratio == 0.0  # guard pinned it back

    # behavioral proof: no patches dropped -> full token set is returned
    n = bb._n_patches
    noise = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    tokens = bb.encoder(pixel_values=torch.randn(1, 1, 112, 112), noise=noise).last_hidden_state
    assert tokens.shape[1] == n + 1  # 1 CLS + every patch (would be ~n/2+1 if masked)

    out = bb(torch.randn(2, 1, 112, 112))
    assert out.shape == (2, 32) and torch.isfinite(out).all()
