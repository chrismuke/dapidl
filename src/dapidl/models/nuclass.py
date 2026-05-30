"""NuClass: a two-stream nucleus+context classifier backbone (dual-scale v1).

Cell type is partly nuclear morphology and partly tissue context (a T-cell looks
like its neighborhood as much as its own nucleus). NuClass fuses two streams:

  * nucleus stream  -- a tight center crop (default 64px ~= the cell) through the
    DAPI-native NuSPIRe ViT-MAE encoder (fine morphology);
  * context stream  -- the full patch (128px ~= 27um, cell + immediate neighbors)
    through a lighter encoder (default the native-1-channel microscopy CNN);

fused by a learnable per-sample gate ``g * nucleus + (1-g) * context``. The gate
is inspectable (``gate_values``), giving a falsifiable readout of whether context
actually mattered for a class.

v1 caveat (review, dual-scale option): both streams crop the SAME existing 128px
patch, so "context" is only ~27um. The faithful wide-tissue version needs a
source-slide extraction pass; the architecture here is unchanged for that step --
only the context stream's input would widen.
"""
from __future__ import annotations

import torch
import torch.nn as nn

NUCLASS_DEFAULT_CROP = 64
NUCLASS_FUSION_DIM = 512


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop a [B, C, H, W] tensor to size x size. No-op if size >= H and W."""
    h, w = x.shape[-2], x.shape[-1]
    if size >= h and size >= w:
        return x
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top:top + size, left:left + size]


class GatedFusion(nn.Module):
    """Per-sample gated fusion of nucleus and context features.

    Projects both streams to ``out_dim``, learns a gate g in [0,1]^out_dim from the
    concatenation, and returns ``g * nucleus_proj + (1 - g) * context_proj``. The
    gate lets the model weight context for context-dependent classes (e.g. T-cells)
    and nucleus morphology for others, per sample and per feature.
    """

    def __init__(self, nucleus_dim: int, context_dim: int, out_dim: int) -> None:
        super().__init__()
        self.nuc_proj = nn.Linear(nucleus_dim, out_dim)
        self.ctx_proj = nn.Linear(context_dim, out_dim)
        self.gate = nn.Linear(out_dim * 2, out_dim)
        self.out_dim = out_dim

    def _project(self, nuc_feat: torch.Tensor, ctx_feat: torch.Tensor):
        return self.nuc_proj(nuc_feat), self.ctx_proj(ctx_feat)

    def gate_from_projections(self, n: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([n, c], dim=1)))

    def forward(self, nuc_feat: torch.Tensor, ctx_feat: torch.Tensor) -> torch.Tensor:
        n, c = self._project(nuc_feat, ctx_feat)
        g = self.gate_from_projections(n, c)
        return g * n + (1.0 - g) * c


class NuClassTwoStream(nn.Module):
    """Two-stream nucleus+context feature extractor.

    Takes a single ``[B, 1, H, W]`` patch, routes a tight center crop through the
    nucleus backbone and the full patch through the context backbone, and gated-
    fuses them. Exposes ``num_features`` (= fusion_dim) so it drops into the same
    ``backbone -> head`` plumbing as the single-stream backbones.
    """

    def __init__(
        self,
        nucleus_backbone: nn.Module,
        context_backbone: nn.Module,
        nucleus_dim: int,
        context_dim: int,
        nucleus_crop: int = NUCLASS_DEFAULT_CROP,
        fusion_dim: int = NUCLASS_FUSION_DIM,
    ) -> None:
        super().__init__()
        if nucleus_crop <= 0:
            raise ValueError(f"nucleus_crop must be positive, got {nucleus_crop}")
        self.nucleus_backbone = nucleus_backbone
        self.context_backbone = context_backbone
        self.nucleus_crop = nucleus_crop
        self.fusion = GatedFusion(nucleus_dim, context_dim, fusion_dim)
        self.num_features = fusion_dim

    def _features(self, x: torch.Tensor):
        nuc = center_crop(x, self.nucleus_crop)
        return self.nucleus_backbone(nuc), self.context_backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nf, cf = self._features(x)
        return self.fusion(nf, cf)

    def gate_values(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample gate in [0,1]^fusion_dim (interpretability: ~1 => nucleus-led,
        ~0 => context-led)."""
        nf, cf = self._features(x)
        n, c = self.fusion._project(nf, cf)
        return self.fusion.gate_from_projections(n, c)


def create_nuclass(
    pretrained: bool = True,
    nucleus: str = "nuspire",
    context: str = "microscopy_cnn_deep",
    nucleus_crop: int = NUCLASS_DEFAULT_CROP,
    fusion_dim: int = NUCLASS_FUSION_DIM,
) -> NuClassTwoStream:
    """Build a NuClass two-stream from two single-channel backbones.

    Defaults: NuSPIRe (DAPI foundation model) for the nucleus, the native
    1-channel microscopy CNN for context. Both are built via ``create_backbone``
    (imported lazily to avoid a backbone<->nuclass import cycle).
    """
    from dapidl.models.backbone import create_backbone

    nuc_bb, nuc_dim = create_backbone(nucleus, pretrained=pretrained, in_channels=1)
    ctx_bb, ctx_dim = create_backbone(context, pretrained=pretrained, in_channels=1)
    return NuClassTwoStream(
        nucleus_backbone=nuc_bb, context_backbone=ctx_bb,
        nucleus_dim=nuc_dim, context_dim=ctx_dim,
        nucleus_crop=nucleus_crop, fusion_dim=fusion_dim,
    )
