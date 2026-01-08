"""HEIST layer implementations.

This module provides the core neural network layers for HEIST:
1. GINConv wrapper for spatial (high-level) graph processing
2. TransformerConv for GRN (low-level) graph processing
3. CrossMessagePassing for bidirectional attention between levels
4. MultiLevelGraphLayer combining all components

Based on the original HEIST implementation from:
https://github.com/Graph-and-Geometric-Learning/HEIST
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, TransformerConv


class SpatialGINConv(nn.Module):
    """GIN convolution for spatial cell graph (high-level).

    Uses Graph Isomorphism Network for message passing on the
    spatial cell-cell graph.

    Args:
        hidden_dim: Hidden dimension.
        eps: GIN epsilon parameter.
        train_eps: Whether to learn epsilon.
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 0.0,
        train_eps: bool = True,
    ) -> None:
        super().__init__()

        # MLP for GIN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.conv = GINConv(self.mlp, eps=eps, train_eps=train_eps)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, D) node features.
            edge_index: (2, E) edge indices.

        Returns:
            (N, D) updated node features.
        """
        out = self.conv(x, edge_index)
        return self.norm(out)


class GeneTransformerConv(nn.Module):
    """Transformer convolution for GRN (low-level).

    Uses attention-based message passing on the gene regulatory network.

    Args:
        hidden_dim: Hidden dimension.
        heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # TransformerConv expects head_dim * heads = hidden_dim
        # So head_dim = hidden_dim // heads
        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,  # Output is heads * (hidden_dim // heads) = hidden_dim
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N_genes, D) gene embeddings.
            edge_index: (2, E) GRN edge indices.

        Returns:
            (N_genes, D) updated gene embeddings.
        """
        if edge_index.shape[1] == 0:
            # No edges - return input unchanged
            return x

        out = self.conv(x, edge_index)
        return self.norm(out)


class CrossMessagePassing(nn.Module):
    """Cross-level attention between high and low embeddings.

    Implements bidirectional Q-K-V attention:
    - high_emb queries low_emb (cell attends to genes)
    - low_emb queries high_emb (genes attend to cell)

    Args:
        hidden_dim: Hidden dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        # High → Low attention
        self.high_to_low_q = nn.Linear(hidden_dim, hidden_dim)
        self.high_to_low_k = nn.Linear(hidden_dim, hidden_dim)
        self.high_to_low_v = nn.Linear(hidden_dim, hidden_dim)

        # Low → High attention
        self.low_to_high_q = nn.Linear(hidden_dim, hidden_dim)
        self.low_to_high_k = nn.Linear(hidden_dim, hidden_dim)
        self.low_to_high_v = nn.Linear(hidden_dim, hidden_dim)

        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            high_emb: (N, D) high-level (cell) embeddings.
            low_emb: (N, D) low-level (gene-aggregated) embeddings.

        Returns:
            high_update: (N, D) update for high embeddings.
            low_update: (N, D) update for low embeddings.
        """
        # High queries low
        q_h2l = self.high_to_low_q(high_emb)  # (N, D)
        k_h2l = self.high_to_low_k(low_emb)  # (N, D)
        v_h2l = self.high_to_low_v(low_emb)  # (N, D)

        # Attention: each cell attends to its own gene aggregation
        # This is element-wise since we have per-cell gene aggregations
        attn_h2l = (q_h2l * k_h2l).sum(dim=-1, keepdim=True) * self.scale
        attn_h2l = torch.sigmoid(attn_h2l)  # (N, 1)
        high_update = attn_h2l * v_h2l  # (N, D)

        # Low queries high
        q_l2h = self.low_to_high_q(low_emb)
        k_l2h = self.low_to_high_k(high_emb)
        v_l2h = self.low_to_high_v(high_emb)

        attn_l2h = (q_l2h * k_l2h).sum(dim=-1, keepdim=True) * self.scale
        attn_l2h = torch.sigmoid(attn_l2h)
        low_update = attn_l2h * v_l2h

        return high_update, low_update


class GeneEmbedder(nn.Module):
    """Embed expression values to gene-level features.

    Creates per-cell, per-gene embeddings from expression values.

    Args:
        n_genes: Number of genes.
        hidden_dim: Hidden dimension.
    """

    def __init__(self, n_genes: int, hidden_dim: int) -> None:
        super().__init__()

        # Learnable gene embeddings
        self.gene_embed = nn.Embedding(n_genes, hidden_dim)

        # Project expression value
        self.expr_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )

        # Combine gene identity and expression
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            expression: (N_cells, N_genes) expression values.

        Returns:
            (N_cells, N_genes, D) gene embeddings per cell.
        """
        n_cells, n_genes = expression.shape
        device = expression.device

        # Gene identity embeddings: (N_genes, D)
        gene_idx = torch.arange(n_genes, device=device)
        gene_emb = self.gene_embed(gene_idx)  # (N_genes, D)

        # Expand for all cells: (N_cells, N_genes, D)
        gene_emb = gene_emb.unsqueeze(0).expand(n_cells, -1, -1)

        # Expression projection: (N_cells, N_genes, D)
        expr_emb = self.expr_proj(expression.unsqueeze(-1))

        # Combine
        combined = torch.cat([gene_emb, expr_emb], dim=-1)  # (N_cells, N_genes, 2D)
        out = self.combine(combined)  # (N_cells, N_genes, D)

        return out


class MultiLevelGraphLayer(nn.Module):
    """Single HEIST layer with high/low graph processing.

    Processes both spatial (high-level) and gene (low-level) graphs,
    then applies cross-message passing between levels.

    Args:
        hidden_dim: Hidden dimension.
        n_heads: Number of attention heads for gene transformer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # High-level (spatial) processing
        self.high_conv = SpatialGINConv(hidden_dim)

        # Low-level (gene) processing
        self.low_conv = GeneTransformerConv(
            hidden_dim, heads=n_heads, dropout=dropout
        )

        # Cross-level attention
        self.cross = CrossMessagePassing(hidden_dim)

        # Layer norms for residual connections
        self.norm_high = nn.LayerNorm(hidden_dim)
        self.norm_low = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        grn_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            high_emb: (N_cells, D) cell embeddings from spatial path.
            low_emb: (N_cells, D) cell embeddings from gene path.
            spatial_edge_index: (2, E_spatial) spatial graph edges.
            grn_edge_index: (2, E_grn) gene regulatory network edges.

        Returns:
            high_emb: (N_cells, D) updated cell embeddings (spatial).
            low_emb: (N_cells, D) updated cell embeddings (gene).
        """
        # 1. Process high-level (spatial) graph
        high_out = self.high_conv(high_emb, spatial_edge_index)
        high_out = self.dropout(high_out)

        # 2. Process low-level (gene) graph
        # Note: GRN operates on genes, but we aggregate to cell level
        # For simplicity, we use low_emb directly (already cell-level)
        # The GRN structure informs how information flows between cells
        # based on their gene expression patterns
        low_out = self._process_gene_level(low_emb, grn_edge_index)
        low_out = self.dropout(low_out)

        # 3. Cross-message passing
        high_cross, low_cross = self.cross(high_out, low_out)

        # 4. Residual connections
        high_emb = self.norm_high(high_emb + high_out + high_cross)
        low_emb = self.norm_low(low_emb + low_out + low_cross)

        return high_emb, low_emb

    def _process_gene_level(
        self,
        cell_emb: torch.Tensor,
        grn_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Process cell embeddings through gene-level GRN.

        For our cell-type GRN adaptation, we use the GRN structure
        to weight how neighboring cells influence each other based
        on shared gene regulatory patterns.

        This is a simplification from the original per-cell GRN.

        Args:
            cell_emb: (N_cells, D) cell embeddings.
            grn_edge_index: (2, E) GRN edges (gene-gene).

        Returns:
            (N_cells, D) processed embeddings.
        """
        # If no GRN edges, return unchanged
        if grn_edge_index.shape[1] == 0:
            return cell_emb

        # For the simplified version, we use the low_conv on cell embeddings
        # treating cells as if they were genes in the GRN
        # This propagates information based on gene-regulatory structure
        return self.low_conv(cell_emb, grn_edge_index)


class MultiLevelGraphLayerFull(nn.Module):
    """Full HEIST layer with per-cell gene embeddings.

    This is the more faithful implementation that maintains
    per-cell, per-gene embeddings and processes them through the GRN.

    More memory-intensive but closer to the original HEIST.

    Args:
        hidden_dim: Hidden dimension.
        n_genes: Number of genes.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_genes: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # High-level (spatial) processing
        self.high_conv = SpatialGINConv(hidden_dim)

        # Low-level (gene) processing
        self.low_conv = GeneTransformerConv(
            hidden_dim, heads=n_heads, dropout=dropout
        )

        # Cross-level attention
        self.cross = CrossMessagePassing(hidden_dim)

        # Gene aggregation for cross-attention
        self.gene_agg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Layer norms
        self.norm_high = nn.LayerNorm(hidden_dim)
        self.norm_low = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        grn_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            high_emb: (N_cells, D) cell embeddings from spatial path.
            low_emb: (N_cells, N_genes, D) per-cell gene embeddings.
            spatial_edge_index: (2, E_spatial) spatial graph edges.
            grn_edge_index: (2, E_grn) gene regulatory network edges.

        Returns:
            high_emb: (N_cells, D) updated cell embeddings.
            low_emb: (N_cells, N_genes, D) updated gene embeddings.
        """
        n_cells = high_emb.shape[0]

        # 1. Process high-level (spatial) graph
        high_out = self.high_conv(high_emb, spatial_edge_index)
        high_out = self.dropout(high_out)

        # 2. Process low-level (gene) graph for each cell
        # Reshape: (N_cells * N_genes, D)
        low_flat = low_emb.view(-1, self.hidden_dim)

        # Apply GRN convolution (same GRN for all cells)
        # Need to expand edge_index for batched processing
        grn_batch = self._batch_grn_edges(grn_edge_index, n_cells)
        low_out_flat = self.low_conv(low_flat, grn_batch)

        # Reshape back: (N_cells, N_genes, D)
        low_out = low_out_flat.view(n_cells, self.n_genes, self.hidden_dim)
        low_out = self.dropout(low_out)

        # 3. Aggregate genes for cross-attention
        gene_agg = self.gene_agg(low_out.mean(dim=1))  # (N_cells, D)

        # 4. Cross-message passing
        high_cross, low_cross = self.cross(high_out, gene_agg)

        # 5. Residual connections
        high_emb = self.norm_high(high_emb + high_out + high_cross)

        # Broadcast low_cross to all genes
        low_cross_broadcast = low_cross.unsqueeze(1).expand_as(low_out)
        low_emb = self.norm_low(low_emb + low_out + low_cross_broadcast)

        return high_emb, low_emb

    def _batch_grn_edges(
        self,
        grn_edge_index: torch.Tensor,
        n_cells: int,
    ) -> torch.Tensor:
        """Expand GRN edges for batched cells.

        Creates separate GRN graphs for each cell.

        Args:
            grn_edge_index: (2, E) gene-gene edges.
            n_cells: Number of cells.

        Returns:
            (2, E * n_cells) batched edge indices.
        """
        if grn_edge_index.shape[1] == 0:
            return grn_edge_index

        device = grn_edge_index.device
        n_edges = grn_edge_index.shape[1]

        # Offset for each cell's gene indices
        offsets = torch.arange(n_cells, device=device) * self.n_genes
        offsets = offsets.view(-1, 1, 1).expand(-1, 2, n_edges)

        # Expand edge index for all cells
        grn_batch = grn_edge_index.unsqueeze(0).expand(n_cells, -1, -1)
        grn_batch = grn_batch + offsets

        # Reshape to (2, E * n_cells)
        grn_batch = grn_batch.permute(1, 0, 2).reshape(2, -1)

        return grn_batch
