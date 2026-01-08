"""Hierarchical blending module for HEIST.

Fuses high-level (spatial) and low-level (gene) embeddings
using multi-head attention.

Based on the HierarchicalBlending module from the original HEIST.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalBlending(nn.Module):
    """Fuse high and low level embeddings via attention.

    Uses multi-head self-attention over the two levels,
    followed by a learnable weighted combination.

    Args:
        hidden_dim: Hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        # Multi-head attention over the two levels
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Learnable blend weight (initialized to 0.5)
        self.beta = nn.Parameter(torch.tensor(0.5))

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            high_emb: (N, D) high-level (spatial) embeddings.
            low_emb: (N, D) low-level (gene) embeddings.

        Returns:
            (N, D) fused embeddings.
        """
        n_cells = high_emb.shape[0]

        # Stack as sequence of length 2: (N, 2, D)
        x = torch.stack([high_emb, low_emb], dim=1)

        # Self-attention over the two levels
        # Each level can attend to itself and the other
        attn_out, _ = self.attn(x, x, x)  # (N, 2, D)

        # Weighted combination using learnable beta
        # beta in [0, 1] via sigmoid
        weight = torch.sigmoid(self.beta)
        fused = weight * attn_out[:, 0] + (1 - weight) * attn_out[:, 1]

        # Residual from mean of inputs
        residual = (high_emb + low_emb) / 2

        # Project and normalize
        out = self.proj(fused)
        out = self.norm(out + residual)

        return out


class HierarchicalBlendingGated(nn.Module):
    """Gated fusion of high and low level embeddings.

    Alternative to attention-based blending using gating mechanism.

    Args:
        hidden_dim: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Gate computation
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Transform each level
        self.high_proj = nn.Linear(hidden_dim, hidden_dim)
        self.low_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            high_emb: (N, D) high-level embeddings.
            low_emb: (N, D) low-level embeddings.

        Returns:
            (N, D) fused embeddings.
        """
        # Compute gate from concatenation
        concat = torch.cat([high_emb, low_emb], dim=-1)
        gate = self.gate(concat)  # (N, D) in [0, 1]

        # Project each level
        high_proj = self.high_proj(high_emb)
        low_proj = self.low_proj(low_emb)

        # Gated combination
        fused = gate * high_proj + (1 - gate) * low_proj

        # Residual and normalize
        residual = (high_emb + low_emb) / 2
        out = self.norm(fused + residual)
        out = self.dropout(out)

        return out
