"""Node-feature encoders for the probe harness. Each exposes encode(rows) ->
[len(rows), out_dim] on `device`, hiding whether features come from a trainable conv
on 40px crops or a lookup into cached frozen embeddings."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dapidl.graph.gnn import NucleusNodeCNN


class FrozenFeatureEncoder(nn.Module):
    """Lookup into a cached (N, d) frozen-embedding array (numpy OR a device tensor —
    the caller may preload it once to avoid recopying per arm/fold). encode(rows)
    returns the rows on device; if proj_dim is set, a trainable LayerNorm->Linear
    projects d -> proj_dim. The frozen features themselves are never back-propagated."""

    def __init__(self, features, device: str, proj_dim: int | None = None):
        super().__init__()
        if isinstance(features, torch.Tensor):
            self._feats = features.to(device)
        else:
            self._feats = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
        self.device = device
        in_dim = int(self._feats.shape[1])
        if proj_dim is None:
            self.proj = None
            self.out_dim = in_dim
        else:
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, proj_dim)).to(device)
            self.out_dim = proj_dim

    def encode(self, rows: np.ndarray) -> torch.Tensor:
        x = self._feats[torch.as_tensor(rows, device=self.device, dtype=torch.long)]
        return self.proj(x) if self.proj is not None else x


class CropCNNEncoder(nn.Module):
    """Trainable conv encoder over 40px nucleus crops (reproduces Stage-2-proper's
    NucleusNodeCNN path). `crops` is the (N, crop, crop) uint16 array."""

    def __init__(self, crops: np.ndarray, device: str, out_dim: int = 128):
        super().__init__()
        self._crops = crops
        self.device = device
        self.out_dim = out_dim
        self.cnn = NucleusNodeCNN(out_dim=out_dim).to(device)

    def encode(self, rows: np.ndarray) -> torch.Tensor:
        x = self._crops[rows].astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        t = torch.from_numpy(x)[:, None].to(self.device)
        return self.cnn(t)
