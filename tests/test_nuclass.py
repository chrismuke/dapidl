"""Tests for the NuClass two-stream nucleus+context model (dual-scale v1).

Injects tiny stub streams so the fusion/crop/routing logic is pinned without
downloading NuSPIRe or needing a GPU. The real-weights path is covered by the
GPU smoke (scripts/nuclass_smoke.py).
"""
import pytest
import torch
import torch.nn as nn

from dapidl.models.nuclass import GatedFusion, NuClassTwoStream, center_crop


class _RecordStub(nn.Module):
    """Differentiable [B,1,H,W] -> [B,dim] stub that records the H,W it received,
    so a test can assert each stream got the right scale."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(1, dim)
        self.last_hw = None

    def forward(self, x):
        self.last_hw = tuple(x.shape[-2:])
        return self.lin(x.flatten(1).mean(1, keepdim=True))


def _two_stream(nuc_dim=8, ctx_dim=12, fusion_dim=16, crop=64):
    return NuClassTwoStream(
        nucleus_backbone=_RecordStub(nuc_dim),
        context_backbone=_RecordStub(ctx_dim),
        nucleus_dim=nuc_dim, context_dim=ctx_dim,
        nucleus_crop=crop, fusion_dim=fusion_dim,
    )


def test_center_crop_extracts_center():
    x = torch.arange(1 * 1 * 8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    out = center_crop(x, 4)
    assert out.shape == (1, 1, 4, 4)
    assert torch.equal(out, x[..., 2:6, 2:6])


def test_center_crop_noop_when_size_ge_input():
    x = torch.randn(2, 1, 10, 10)
    assert torch.equal(center_crop(x, 10), x)
    assert torch.equal(center_crop(x, 99), x)


def test_forward_shape():
    bb = _two_stream(fusion_dim=16)
    out = bb(torch.randn(4, 1, 128, 128))
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_num_features_is_fusion_dim():
    assert _two_stream(fusion_dim=16).num_features == 16


def test_streams_receive_correct_scales():
    # The dual-scale contract: nucleus stream sees the tight center crop, context
    # stream sees the full patch. This is the whole point of the model.
    bb = _two_stream(crop=64)
    bb(torch.randn(2, 1, 128, 128))
    assert bb.nucleus_backbone.last_hw == (64, 64)
    assert bb.context_backbone.last_hw == (128, 128)


def test_gate_in_unit_interval():
    bb = _two_stream(fusion_dim=16)
    g = bb.gate_values(torch.randn(3, 1, 128, 128))
    assert g.shape == (3, 16)
    assert (g >= 0).all() and (g <= 1).all()


def test_deterministic_in_eval():
    bb = _two_stream().eval()
    x = torch.randn(2, 1, 128, 128)
    assert torch.allclose(bb(x), bb(x), atol=1e-6)


def test_gradients_reach_both_streams_and_fusion():
    bb = _two_stream()
    bb(torch.randn(2, 1, 128, 128)).sum().backward()
    assert bb.nucleus_backbone.lin.weight.grad is not None
    assert bb.context_backbone.lin.weight.grad is not None
    assert bb.fusion.gate.weight.grad is not None


def test_gated_fusion_is_convex_combination():
    # With gate forced to 1, output must equal the nucleus projection exactly.
    fuse = GatedFusion(nucleus_dim=4, context_dim=4, out_dim=4)
    nn.init.constant_(fuse.gate.weight, 0.0)
    nn.init.constant_(fuse.gate.bias, 50.0)  # sigmoid(50) ~= 1 -> all nucleus
    nf = torch.randn(2, 4)
    cf = torch.randn(2, 4)
    out = fuse(nf, cf)
    assert torch.allclose(out, fuse.nuc_proj(nf), atol=1e-4)


def test_invalid_crop_raises():
    with pytest.raises(ValueError):
        _two_stream(crop=0)


def test_nuclass_registered_native_single_channel():
    from dapidl.models.backbone import BACKBONE_PRESETS

    assert "nuclass" in BACKBONE_PRESETS
    assert BACKBONE_PRESETS["nuclass"]["native_channels"] == 1


def test_create_backbone_routes_to_nuclass(monkeypatch):
    import dapidl.models.backbone as bb

    sentinel = _two_stream(fusion_dim=16)
    monkeypatch.setattr(bb, "create_nuclass", lambda **kw: sentinel)
    model, nf = bb.create_backbone("nuclass", pretrained=False, in_channels=1)
    assert model is sentinel
    assert nf == 16
