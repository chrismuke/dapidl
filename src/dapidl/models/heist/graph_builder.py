"""Graph construction utilities for HEIST.

This module provides:
1. SpatialGraphBuilder - k-NN spatial graph from cell coordinates
2. CellTypeGRNBuilder - Gene regulatory networks per cell type using mutual information

Following the original HEIST paper and implementation:
- High-level graph: spatial cell-cell relationships
- Low-level graphs: gene-gene relationships (one GRN per cell type)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from joblib import Parallel, delayed
from loguru import logger
from scipy.spatial import cKDTree
from sklearn.feature_selection import mutual_info_regression

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _compute_mi_for_gene(
    expression: np.ndarray,
    gene_idx: int,
) -> tuple[int, np.ndarray]:
    """Compute MI between one gene and all others (for parallel execution).

    Args:
        expression: (N_cells, N_genes) expression matrix.
        gene_idx: Index of the target gene.

    Returns:
        Tuple of (gene_idx, mi_values).
    """
    mi_values = mutual_info_regression(
        expression,
        expression[:, gene_idx],
        discrete_features=False,
        random_state=42,
    )
    return gene_idx, mi_values


class SpatialGraphBuilder:
    """Build k-NN spatial graph from cell coordinates.

    Creates a graph where cells are nodes and edges connect
    spatially proximal cells (k nearest neighbors within max distance).

    Args:
        k: Number of nearest neighbors per cell.
        max_distance_um: Maximum distance in micrometers for edges.
        symmetric: Whether to make edges bidirectional.
    """

    def __init__(
        self,
        k: int = 10,
        max_distance_um: float = 50.0,
        symmetric: bool = True,
    ) -> None:
        self.k = k
        self.max_distance_um = max_distance_um
        self.symmetric = symmetric

    def build_graph(
        self,
        coords: NDArray[np.floating],
        pixel_size_um: float = 0.2125,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build spatial k-NN graph from coordinates.

        Args:
            coords: (N, 2) array of cell centroid coordinates in pixels.
            pixel_size_um: Pixel size in micrometers (Xenium default: 0.2125).

        Returns:
            edge_index: (2, E) tensor of edges in PyG format.
            edge_attr: (E, 1) tensor of edge distances (normalized).
        """
        n_cells = len(coords)
        max_distance_px = self.max_distance_um / pixel_size_um

        logger.info(
            f"Building spatial graph: {n_cells} cells, k={self.k}, "
            f"max_dist={self.max_distance_um}um ({max_distance_px:.1f}px)"
        )

        # Build KD-tree for efficient neighbor search
        tree = cKDTree(coords)

        # Query k+1 neighbors (includes self)
        distances, indices = tree.query(coords, k=self.k + 1)

        # Build edge list (excluding self-loops)
        src_list = []
        dst_list = []
        dist_list = []

        for i in range(n_cells):
            for j, (neighbor_idx, dist) in enumerate(
                zip(indices[i, 1:], distances[i, 1:], strict=True)
            ):
                if dist <= max_distance_px:
                    src_list.append(i)
                    dst_list.append(neighbor_idx)
                    dist_list.append(dist)

        # Convert to tensors
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Normalize distances to [0, 1]
        distances_tensor = torch.tensor(dist_list, dtype=torch.float32)
        if len(distances_tensor) > 0:
            edge_attr = (distances_tensor / max_distance_px).unsqueeze(-1)
        else:
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        # Make symmetric if requested
        if self.symmetric:
            edge_index_rev = edge_index.flip(0)
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

            # Remove duplicates
            edge_index, unique_idx = torch.unique(
                edge_index, dim=1, return_inverse=True
            )
            # Average edge attributes for duplicates
            edge_attr_new = torch.zeros(
                (edge_index.shape[1], 1), dtype=torch.float32
            )
            edge_attr_new.scatter_reduce_(
                0,
                unique_idx.unsqueeze(-1).expand(-1, 1),
                edge_attr,
                reduce="mean",
            )
            edge_attr = edge_attr_new

        logger.info(f"Built spatial graph: {edge_index.shape[1]} edges")
        return edge_index, edge_attr


class CellTypeGRNBuilder:
    """Build gene regulatory networks per cell type using mutual information.

    For each cell type, computes pairwise mutual information between genes
    using cells of that type, then thresholds to create a sparse GRN.

    This is our adaptation of the original HEIST which builds per-cell GRNs.
    We use per-cell-type GRNs as a practical compromise.

    Args:
        mi_threshold: Minimum MI for edge creation.
        max_edges_per_gene: Maximum edges per gene (keeps top-k by MI).
        min_cells_per_type: Minimum cells required to build GRN for a type.
        max_cells_per_type: Maximum cells to sample for MI computation (for speed).
    """

    def __init__(
        self,
        mi_threshold: float = 0.35,
        max_edges_per_gene: int = 20,
        min_cells_per_type: int = 50,
        max_cells_per_type: int = 5000,
    ) -> None:
        self.mi_threshold = mi_threshold
        self.max_edges_per_gene = max_edges_per_gene
        self.min_cells_per_type = min_cells_per_type
        self.max_cells_per_type = max_cells_per_type

    def build_grns(
        self,
        expression: NDArray[np.floating],
        cell_types: NDArray[np.integer],
        cell_type_names: list[str] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Build one GRN per cell type.

        Args:
            expression: (N_cells, N_genes) expression matrix (log-normalized).
            cell_types: (N_cells,) integer cell type labels.
            cell_type_names: Optional list of cell type names for logging.

        Returns:
            Dict mapping cell_type_id -> edge_index (2, E) tensor.
        """
        n_cells, n_genes = expression.shape
        unique_types = np.unique(cell_types)

        logger.info(
            f"Building cell-type GRNs: {n_cells} cells, {n_genes} genes, "
            f"{len(unique_types)} cell types"
        )

        grns = {}

        n_jobs = max(1, int(os.cpu_count() * 0.8))
        logger.info(f"Using {n_jobs} parallel workers for MI computation")

        for idx, cell_type_id in enumerate(unique_types):
            # Get cells of this type
            mask = cell_types == cell_type_id
            n_type_cells = mask.sum()

            type_name = (
                cell_type_names[cell_type_id]
                if cell_type_names
                else str(cell_type_id)
            )

            if n_type_cells < self.min_cells_per_type:
                logger.warning(
                    f"Skipping GRN for {type_name}: only {n_type_cells} cells "
                    f"(need {self.min_cells_per_type})"
                )
                # Use empty graph
                grns[int(cell_type_id)] = torch.zeros((2, 0), dtype=torch.long)
                continue

            # Sample cells if too many (MI computation scales with n_cells)
            if n_type_cells > self.max_cells_per_type:
                n_sample = self.max_cells_per_type
                logger.info(
                    f"[{idx+1}/{len(unique_types)}] Computing GRN for {type_name} "
                    f"(sampling {n_sample} from {n_type_cells} cells, {n_genes} genes)..."
                )
                # Random sample of cell indices
                rng = np.random.default_rng(42)
                type_indices = np.where(mask)[0]
                sample_indices = rng.choice(type_indices, size=n_sample, replace=False)
                expr_subset = expression[sample_indices]
            else:
                logger.info(
                    f"[{idx+1}/{len(unique_types)}] Computing GRN for {type_name} "
                    f"({n_type_cells} cells, {n_genes} genes)..."
                )
                # Extract expression for this cell type
                expr_subset = expression[mask]

            # Compute GRN for this cell type (parallelized)
            edge_index = self._compute_mi_grn(expr_subset, n_jobs=n_jobs)

            grns[int(cell_type_id)] = edge_index

            logger.info(
                f"  {type_name}: {edge_index.shape[1]} GRN edges"
            )

        return grns

    def _compute_mi_grn(
        self,
        expression: NDArray[np.floating],
        n_jobs: int = -1,
    ) -> torch.Tensor:
        """Compute GRN using mutual information (parallelized).

        Args:
            expression: (N_cells, N_genes) expression for cells of one type.
            n_jobs: Number of parallel jobs (-1 for all cores).

        Returns:
            edge_index: (2, E) gene-gene edges.
        """
        n_genes = expression.shape[1]

        # Determine number of jobs (use 80% of cores to leave headroom)
        if n_jobs == -1:
            n_jobs = max(1, int(os.cpu_count() * 0.8))

        # Compute pairwise MI in parallel
        mi_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)

        # Parallel computation of MI for each gene
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_compute_mi_for_gene)(expression, i)
            for i in range(n_genes)
        )

        # Fill MI matrix from results
        for gene_idx, mi_values in results:
            mi_matrix[gene_idx, :] = mi_values

        # Make symmetric (take max of MI(i,j) and MI(j,i))
        mi_matrix = np.maximum(mi_matrix, mi_matrix.T)

        # Zero out diagonal (no self-loops)
        np.fill_diagonal(mi_matrix, 0)

        # Build edge list
        src_list = []
        dst_list = []

        for i in range(n_genes):
            # Get MI values for gene i
            mi_row = mi_matrix[i, :]

            # Filter by threshold
            candidates = np.where(mi_row >= self.mi_threshold)[0]

            if len(candidates) == 0:
                continue

            # Keep top-k by MI
            if len(candidates) > self.max_edges_per_gene:
                top_k_idx = np.argsort(mi_row[candidates])[-self.max_edges_per_gene:]
                candidates = candidates[top_k_idx]

            for j in candidates:
                if i < j:  # Only add upper triangle to avoid duplicates
                    src_list.append(i)
                    dst_list.append(j)

        # Make bidirectional
        if src_list:
            edge_index = torch.tensor(
                [src_list + dst_list, dst_list + src_list],
                dtype=torch.long,
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return edge_index

    def build_universal_grn(
        self,
        grns: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Build a universal GRN from union of all cell-type GRNs.

        Useful for inference when cell type is unknown.

        Args:
            grns: Dict of cell-type GRNs from build_grns().

        Returns:
            edge_index: (2, E) union of all GRN edges.
        """
        if not grns:
            return torch.zeros((2, 0), dtype=torch.long)

        all_edges = [grn for grn in grns.values() if grn.shape[1] > 0]

        if not all_edges:
            return torch.zeros((2, 0), dtype=torch.long)

        # Concatenate and deduplicate
        combined = torch.cat(all_edges, dim=1)
        unique_edges = torch.unique(combined, dim=1)

        logger.info(f"Universal GRN: {unique_edges.shape[1]} edges")
        return unique_edges


def compute_grn_statistics(
    grns: dict[int, torch.Tensor],
    n_genes: int,
) -> dict:
    """Compute statistics about the GRNs.

    Args:
        grns: Dict of cell-type GRNs.
        n_genes: Number of genes.

    Returns:
        Dict with statistics.
    """
    stats = {
        "n_cell_types": len(grns),
        "n_genes": n_genes,
        "edges_per_type": {},
        "avg_degree_per_type": {},
        "total_unique_edges": 0,
    }

    all_edges = []
    for cell_type, edge_index in grns.items():
        n_edges = edge_index.shape[1]
        stats["edges_per_type"][cell_type] = n_edges

        if n_edges > 0:
            # Compute average degree
            degree = torch.bincount(
                edge_index[0], minlength=n_genes
            ).float().mean().item()
            stats["avg_degree_per_type"][cell_type] = degree
            all_edges.append(edge_index)
        else:
            stats["avg_degree_per_type"][cell_type] = 0.0

    if all_edges:
        combined = torch.cat(all_edges, dim=1)
        unique = torch.unique(combined, dim=1)
        stats["total_unique_edges"] = unique.shape[1]

    return stats
