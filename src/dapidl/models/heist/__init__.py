"""HEIST: Hierarchical Graph Model for Spatial Transcriptomics.

This module provides a HEIST-based model for cell type classification
from gene expression, following the original HEIST paper with adaptations:

- Expression-only input (no images)
- Cell-type-specific GRNs (not per-cell)
- Supervised training (no pretraining)

Example usage:
    from dapidl.models.heist import (
        HEISTClassifier,
        SpatialGraphBuilder,
        CellTypeGRNBuilder,
    )

    # Build graphs
    spatial_builder = SpatialGraphBuilder(k=10, max_distance_um=50.0)
    spatial_edge_index, _ = spatial_builder.build_graph(coords)

    grn_builder = CellTypeGRNBuilder(mi_threshold=0.35)
    cell_type_grns = grn_builder.build_grns(expression, cell_types)

    # Create model
    model = HEISTClassifier(
        num_classes=17,
        n_genes=350,
        hidden_dim=128,
        n_layers=4,
    )

    # Forward pass
    logits = model(
        expression=expression,
        coords=coords,
        spatial_edge_index=spatial_edge_index,
        grn_edge_index=grn_edge_index,
    )
"""

from dapidl.models.heist.blending import (
    HierarchicalBlending,
    HierarchicalBlendingGated,
)
from dapidl.models.heist.graph_builder import (
    CellTypeGRNBuilder,
    SpatialGraphBuilder,
    compute_grn_statistics,
)
from dapidl.models.heist.layers import (
    CrossMessagePassing,
    GeneEmbedder,
    GeneTransformerConv,
    MultiLevelGraphLayer,
    MultiLevelGraphLayerFull,
    SpatialGINConv,
)
from dapidl.models.heist.model import (
    HEISTClassifier,
    HEISTClassifierWithGRNSelection,
    SinusoidalPositionalEncoding,
)

__all__ = [
    # Main model
    "HEISTClassifier",
    "HEISTClassifierWithGRNSelection",
    # Graph builders
    "SpatialGraphBuilder",
    "CellTypeGRNBuilder",
    "compute_grn_statistics",
    # Layers
    "MultiLevelGraphLayer",
    "MultiLevelGraphLayerFull",
    "SpatialGINConv",
    "GeneTransformerConv",
    "CrossMessagePassing",
    "GeneEmbedder",
    # Blending
    "HierarchicalBlending",
    "HierarchicalBlendingGated",
    # Encoders
    "SinusoidalPositionalEncoding",
]
