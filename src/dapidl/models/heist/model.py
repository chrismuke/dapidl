"""HEIST Classifier Model.

Main HEISTClassifier that combines:
1. Expression encoder (Linear + LayerNorm)
2. Sinusoidal positional encoding for spatial coordinates
3. Stack of MultiLevelGraphLayers
4. HierarchicalBlending for fusing levels
5. Classification head

Based on the original HEIST implementation, adapted for DAPIDL
with cell-type-specific GRNs and supervised training.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from dapidl.models.heist.blending import HierarchicalBlending
from dapidl.models.heist.layers import MultiLevelGraphLayer


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 2D spatial coordinates.

    Encodes (x, y) coordinates using sine/cosine functions at
    different frequencies, similar to transformer position encoding.

    Args:
        hidden_dim: Output dimension (must be divisible by 4).
        max_len: Maximum coordinate value for normalization.
    """

    def __init__(self, hidden_dim: int, max_len: int = 10000) -> None:
        super().__init__()

        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"

        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Precompute frequency bands
        dim_per_coord = hidden_dim // 2
        div_term = torch.exp(
            torch.arange(0, dim_per_coord, 2).float()
            * (-math.log(10000.0) / dim_per_coord)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: (N, 2) spatial coordinates.

        Returns:
            (N, D) positional encodings.
        """
        # Normalize coordinates to [0, max_len]
        coords_norm = coords / self.max_len * 1000

        # Separate x and y
        x = coords_norm[:, 0:1]  # (N, 1)
        y = coords_norm[:, 1:2]  # (N, 1)

        # Compute sin/cos for x
        x_sin = torch.sin(x * self.div_term)  # (N, D/4)
        x_cos = torch.cos(x * self.div_term)  # (N, D/4)

        # Compute sin/cos for y
        y_sin = torch.sin(y * self.div_term)  # (N, D/4)
        y_cos = torch.cos(y * self.div_term)  # (N, D/4)

        # Concatenate: (N, D)
        pe = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)

        return pe


class HEISTClassifier(nn.Module):
    """HEIST-based cell type classifier.

    Uses hierarchical graph processing with spatial and gene-regulatory
    networks for cell type classification from gene expression.

    Args:
        num_classes: Number of cell type classes.
        n_genes: Number of genes in expression panel.
        hidden_dim: Hidden dimension (must be divisible by 4 for pos enc).
        n_layers: Number of MultiLevelGraphLayers.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_classes: int,
        n_genes: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Expression encoder
        self.expr_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim)

        # Project concatenated features
        self.pos_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # HEIST layers
        self.layers = nn.ModuleList([
            MultiLevelGraphLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Hierarchical blending
        self.blending = HierarchicalBlending(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.info(
            f"HEISTClassifier: n_genes={n_genes}, hidden_dim={hidden_dim}, "
            f"n_layers={n_layers}, n_heads={n_heads}, num_classes={num_classes}"
        )

    def forward(
        self,
        expression: torch.Tensor,
        coords: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        grn_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            expression: (N, n_genes) gene expression values.
            coords: (N, 2) spatial coordinates.
            spatial_edge_index: (2, E_spatial) spatial graph edges.
            grn_edge_index: (2, E_grn) gene regulatory network edges.

        Returns:
            (N, num_classes) classification logits.
        """
        # Encode expression
        x = self.expr_encoder(expression)  # (N, D)

        # Add positional encoding
        pos = self.pos_encoder(coords)  # (N, D)
        x = self.pos_proj(torch.cat([x, pos], dim=-1))  # (N, D)

        # Initialize high and low level embeddings
        high_emb = x
        low_emb = x.clone()

        # Apply HEIST layers
        for layer in self.layers:
            high_emb, low_emb = layer(
                high_emb, low_emb,
                spatial_edge_index, grn_edge_index,
            )

        # Blend levels
        fused = self.blending(high_emb, low_emb)

        # Classify
        logits = self.classifier(fused)

        return logits

    def get_embeddings(
        self,
        expression: torch.Tensor,
        coords: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        grn_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Extract cell embeddings (before classification head).

        Useful for visualization (UMAP) and downstream analysis.

        Args:
            Same as forward().

        Returns:
            (N, hidden_dim) cell embeddings.
        """
        # Encode expression
        x = self.expr_encoder(expression)
        pos = self.pos_encoder(coords)
        x = self.pos_proj(torch.cat([x, pos], dim=-1))

        # Initialize levels
        high_emb = x
        low_emb = x.clone()

        # Apply HEIST layers
        for layer in self.layers:
            high_emb, low_emb = layer(
                high_emb, low_emb,
                spatial_edge_index, grn_edge_index,
            )

        # Blend
        fused = self.blending(high_emb, low_emb)

        return fused

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        metrics: dict | None = None,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            optimizer: Optional optimizer to save.
            epoch: Current epoch.
            metrics: Optional metrics dict.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "hparams": {
                "num_classes": self.num_classes,
                "n_genes": self.n_genes,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
            },
            "epoch": epoch,
            "metrics": metrics or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved HEISTClassifier checkpoint to {path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str = "cpu",
    ) -> "HEISTClassifier":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            map_location: Device to load checkpoint on.

        Returns:
            Loaded model.
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

        hparams = checkpoint.get("hparams", {})
        model = cls(
            num_classes=hparams.get("num_classes", 17),
            n_genes=hparams.get("n_genes", 350),
            hidden_dim=hparams.get("hidden_dim", 128),
            n_layers=hparams.get("n_layers", 4),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded HEISTClassifier from {checkpoint_path}")

        return model


class HEISTClassifierWithGRNSelection(HEISTClassifier):
    """HEIST classifier with dynamic GRN selection.

    Extends HEISTClassifier to handle cell-type-specific GRNs.
    During training, uses ground truth cell types.
    During inference, uses neighbor majority vote.

    Args:
        num_classes: Number of cell type classes.
        n_genes: Number of genes.
        cell_type_grns: Dict mapping cell_type_id -> edge_index.
        universal_grn: Optional fallback GRN for inference.
        **kwargs: Additional HEISTClassifier arguments.
    """

    def __init__(
        self,
        num_classes: int,
        n_genes: int,
        cell_type_grns: dict[int, torch.Tensor] | None = None,
        universal_grn: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, n_genes, **kwargs)

        self.cell_type_grns = cell_type_grns or {}
        self.universal_grn = universal_grn

        # Register GRNs as buffers so they move with the model
        for ct_id, grn in self.cell_type_grns.items():
            self.register_buffer(f"grn_{ct_id}", grn)

        if universal_grn is not None:
            self.register_buffer("universal_grn_buffer", universal_grn)

    def get_grn_for_cells(
        self,
        cell_types: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        use_neighbor_vote: bool = False,
    ) -> torch.Tensor:
        """Get GRN edges based on cell types.

        For training: use ground truth cell types.
        For inference: use neighbor majority vote or universal GRN.

        Args:
            cell_types: (N,) cell type labels (-1 for unknown).
            spatial_edge_index: (2, E) spatial graph for neighbor voting.
            use_neighbor_vote: Whether to use neighbor voting for unknown types.

        Returns:
            (2, E_grn) GRN edge indices.
        """
        device = cell_types.device

        # If all types are known, use majority type's GRN
        known_mask = cell_types >= 0
        if known_mask.all():
            # Use most common cell type's GRN
            majority_type = cell_types.mode().values.item()
            grn = getattr(self, f"grn_{majority_type}", None)
            if grn is not None:
                return grn.to(device)

        # If using neighbor vote for unknown types
        if use_neighbor_vote and known_mask.any():
            # Simple majority vote from known neighbors
            majority_type = cell_types[known_mask].mode().values.item()
            grn = getattr(self, f"grn_{majority_type}", None)
            if grn is not None:
                return grn.to(device)

        # Fallback to universal GRN
        if hasattr(self, "universal_grn_buffer"):
            return self.universal_grn_buffer.to(device)

        # Last resort: empty GRN
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    def forward_with_types(
        self,
        expression: torch.Tensor,
        coords: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        cell_types: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with cell type-aware GRN selection.

        Args:
            expression: (N, n_genes) expression.
            coords: (N, 2) coordinates.
            spatial_edge_index: (2, E) spatial edges.
            cell_types: (N,) cell type labels.

        Returns:
            (N, num_classes) logits.
        """
        grn_edge_index = self.get_grn_for_cells(
            cell_types, spatial_edge_index, use_neighbor_vote=True
        )
        return self.forward(expression, coords, spatial_edge_index, grn_edge_index)
