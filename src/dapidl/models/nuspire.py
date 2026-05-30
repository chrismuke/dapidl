"""NuSPIRe: a DAPI-native ViT-MAE foundation-model backbone for DAPIDL.

NuSPIRe (``TongjiZhanglab/NuSPIRe``, MIT) is a ViT-MAE pretrained on 15M nucleus
crops (NuCorpus-15M), 112x112 **single-channel** DAPI, 8x8 patches, 768-dim
encoder. Unlike the ImageNet backbones it ingests grayscale directly, so the
classifier needs no ``SingleChannelAdapter`` (``native_channels == 1``).

Loading (verified empirically 2026-05-30):
- Load as ``ViTMAEModel.from_pretrained(repo, mask_ratio=0.0)`` -> **0 missing
  encoder/embedding keys**; the 136 ``decoder.*`` keys are ignored.
- The model card's ``ViTModel`` + ``pooler_output`` path leaves ``pooler.dense``
  *randomly initialized* (ViT-MAE has no pooler) and HF warns the conversion is
  unsupported -- do NOT use it for frozen features.

Readout: mean of the patch tokens (MAE-standard linear-probe readout). ViT-MAE
shuffles patches inside the embeddings even at ``mask_ratio=0`` and ``ViTMAEModel``
never un-shuffles ``last_hidden_state``; mean pooling is permutation-invariant, and
we additionally pass an explicit ascending ``noise`` so the shuffle is the identity
-- features are deterministic and patch order is preserved.

Normalization: NuSPIRe was trained on [0,1] grayscale with
``mean=0.2187, std=0.1809`` (exposed as constants for the input transform; not
applied inside the backbone, mirroring the timm/PathologyFoundationModel convention).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

NUSPIRE_REPO = "TongjiZhanglab/NuSPIRe"
NUSPIRE_IMAGE_SIZE = 112
NUSPIRE_HIDDEN = 768
# Single-channel DAPI normalization the encoder was pretrained with.
NUSPIRE_NORM_MEAN = 0.21869252622127533
NUSPIRE_NORM_STD = 0.1809280514717102


class NuSPIReBackbone(nn.Module):
    """ViT-MAE encoder exposing DAPIDL's ``forward([B,1,H,W]) -> [B, num_features]``.

    Args:
        pretrained: Load the published NuSPIRe weights. If False, a randomly
            initialized encoder with the real geometry is built (no download) --
            useful for routing tests / training from scratch.
        pool: ``"mean"`` (patch-token mean, default) or ``"cls"`` (class token).
        encoder: Inject a prebuilt ``ViTMAEModel`` (test seam); bypasses loading.
    """

    def __init__(
        self,
        pretrained: bool = True,
        pool: str = "mean",
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if pool not in ("mean", "cls"):
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")
        self.pool = pool
        self.image_size = NUSPIRE_IMAGE_SIZE

        if encoder is not None:
            self.encoder = encoder
        else:
            from transformers import ViTMAEModel

            if pretrained:
                self.encoder = ViTMAEModel.from_pretrained(NUSPIRE_REPO, mask_ratio=0.0)
                logger.info(f"Loaded NuSPIRe encoder from {NUSPIRE_REPO} (mask_ratio=0)")
            else:
                from transformers import ViTMAEConfig

                cfg = ViTMAEConfig(
                    image_size=NUSPIRE_IMAGE_SIZE,
                    patch_size=8,
                    num_channels=1,
                    hidden_size=NUSPIRE_HIDDEN,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=NUSPIRE_HIDDEN * 4,
                    mask_ratio=0.0,
                )
                self.encoder = ViTMAEModel(cfg)

        self._force_no_masking()
        patch_size = int(self.encoder.config.patch_size)
        self._n_patches = (self.image_size // patch_size) ** 2
        self.num_features = int(self.encoder.config.hidden_size)

    def _force_no_masking(self) -> None:
        """Pin mask_ratio=0 on both the model and its embeddings config.

        ``from_pretrained(mask_ratio=0.0)`` already does this; this guards against
        an injected/legacy encoder whose config still carries a >0 ratio.
        """
        self.encoder.config.mask_ratio = 0.0
        emb = getattr(self.encoder, "embeddings", None)
        emb_cfg = getattr(emb, "config", None)
        if emb_cfg is not None:
            emb_cfg.mask_ratio = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        # Ascending noise -> argsort is the identity -> no patch shuffle, so
        # features are deterministic and patch tokens stay in natural order.
        noise = (
            torch.arange(self._n_patches, device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(x.shape[0], -1)
        )
        tokens = self.encoder(pixel_values=x, noise=noise).last_hidden_state
        if self.pool == "cls":
            return tokens[:, 0]
        return tokens[:, 1:].mean(dim=1)
