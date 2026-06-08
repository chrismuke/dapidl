"""NuSPIRe registration in the backbone registry + create_backbone routing.

Kept download-free: the routing test monkeypatches the backbone class so no
checkpoint is fetched and no 85M-param model is built.
"""
import torch
import torch.nn as nn

from dapidl.models.backbone import BACKBONE_PRESETS


def test_nuspire_registered_as_native_single_channel():
    assert "nuspire" in BACKBONE_PRESETS
    # native_channels==1 makes CellTypeClassifier pick input_adapter="none"
    # (the same tested path as microscopy_cnn), feeding 1-channel DAPI straight in.
    assert BACKBONE_PRESETS["nuspire"]["native_channels"] == 1


def test_create_backbone_routes_to_nuspire(monkeypatch):
    import dapidl.models.backbone as bb

    class FakeNuSPIRe(nn.Module):
        def __init__(self, pretrained=True, **kwargs):
            super().__init__()
            self.num_features = 768
            self.received_pretrained = pretrained

        def forward(self, x):
            return torch.zeros(x.shape[0], self.num_features)

    monkeypatch.setattr(bb, "NuSPIReBackbone", FakeNuSPIRe)
    model, nf = bb.create_backbone("nuspire", pretrained=False, in_channels=1)
    assert isinstance(model, FakeNuSPIRe)
    assert nf == 768
    assert model.received_pretrained is False  # pretrained flag threaded through
