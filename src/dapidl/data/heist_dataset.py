"""HEIST Dataset with METIS partitioning.

This module provides a PyTorch Dataset for HEIST training that:
1. Loads expression, coordinates, and labels
2. Uses METIS graph partitioning for locality-preserving batches
3. Extracts subgraphs for each partition

Based on the original HEIST dataloader implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.utils import subgraph

try:
    import pymetis

    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    logger.warning("pymetis not available, using random partitioning")


class HEISTDataset(Dataset):
    """Dataset with METIS partitioning for HEIST training.

    Uses METIS graph partitioning to create locality-preserving batches
    where cells in the same partition are spatially close.

    Args:
        expression: (N, n_genes) expression matrix.
        coords: (N, 2) spatial coordinates.
        labels: (N,) cell type labels.
        spatial_edge_index: (2, E) spatial graph edges.
        partition_size: Target number of cells per partition.
        permute: Whether to shuffle nodes before partitioning.
    """

    def __init__(
        self,
        expression: np.ndarray | torch.Tensor,
        coords: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        spatial_edge_index: torch.Tensor,
        partition_size: int = 128,
        permute: bool = True,
    ) -> None:
        # Convert to tensors
        if isinstance(expression, np.ndarray):
            expression = torch.from_numpy(expression).float()
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        self.expression = expression
        self.coords = coords
        self.labels = labels
        self.spatial_edge_index = spatial_edge_index
        self.partition_size = partition_size

        self.n_cells = len(expression)
        self.n_genes = expression.shape[1]

        # Compute partitions
        self.partitions = self._compute_partitions(permute)

        logger.info(
            f"HEISTDataset: {self.n_cells} cells, {self.n_genes} genes, "
            f"{len(self.partitions)} partitions (target size {partition_size})"
        )

    def _compute_partitions(self, permute: bool) -> list[torch.Tensor]:
        """Compute METIS partitions of the spatial graph.

        Args:
            permute: Whether to shuffle nodes before partitioning.

        Returns:
            List of node index tensors, one per partition.
        """
        n_partitions = max(1, self.n_cells // self.partition_size)

        if permute:
            # Shuffle node indices
            perm = torch.randperm(self.n_cells)
        else:
            perm = torch.arange(self.n_cells)

        if METIS_AVAILABLE and self.spatial_edge_index.shape[1] > 0:
            return self._metis_partition(perm, n_partitions)
        else:
            return self._random_partition(perm, n_partitions)

    def _metis_partition(
        self,
        perm: torch.Tensor,
        n_partitions: int,
    ) -> list[torch.Tensor]:
        """Partition using METIS algorithm.

        Args:
            perm: Node permutation.
            n_partitions: Number of partitions.

        Returns:
            List of node index tensors.
        """
        # Build adjacency list for METIS
        edge_index = self.spatial_edge_index

        # Create inverse permutation for relabeling
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(self.n_cells)

        # Relabel edges according to permutation
        edge_src = inv_perm[edge_index[0]].numpy()
        edge_dst = inv_perm[edge_index[1]].numpy()

        # Build adjacency list
        adjacency = [[] for _ in range(self.n_cells)]
        for src, dst in zip(edge_src, edge_dst, strict=True):
            adjacency[src].append(dst)

        # Run METIS partitioning
        try:
            _, membership = pymetis.part_graph(n_partitions, adjacency=adjacency)
            membership = np.array(membership)
        except Exception as e:
            logger.warning(f"METIS failed: {e}, falling back to random")
            return self._random_partition(perm, n_partitions)

        # Group nodes by partition
        partitions = []
        for p in range(n_partitions):
            mask = membership == p
            if mask.sum() > 0:
                # Map back to original indices
                partition_indices = perm[torch.from_numpy(mask)]
                partitions.append(partition_indices)

        return partitions

    def _random_partition(
        self,
        perm: torch.Tensor,
        n_partitions: int,
    ) -> list[torch.Tensor]:
        """Partition randomly (fallback when METIS unavailable).

        Args:
            perm: Node permutation.
            n_partitions: Number of partitions.

        Returns:
            List of node index tensors.
        """
        partitions = []
        chunk_size = self.n_cells // n_partitions

        for i in range(n_partitions):
            start = i * chunk_size
            end = start + chunk_size if i < n_partitions - 1 else self.n_cells
            partitions.append(perm[start:end])

        return partitions

    def __len__(self) -> int:
        return len(self.partitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a partition.

        Args:
            idx: Partition index.

        Returns:
            Dict with:
                - expression: (P, n_genes) expression for partition cells.
                - coords: (P, 2) coordinates.
                - labels: (P,) cell type labels.
                - spatial_edge_index: (2, E) subgraph edges (relabeled).
                - node_indices: (P,) original node indices.
        """
        node_indices = self.partitions[idx]

        # Extract data for partition
        expression = self.expression[node_indices]
        coords = self.coords[node_indices]
        labels = self.labels[node_indices]

        # Extract subgraph
        spatial_sub, _ = subgraph(
            node_indices,
            self.spatial_edge_index,
            relabel_nodes=True,
            num_nodes=self.n_cells,
        )

        return {
            "expression": expression,
            "coords": coords,
            "labels": labels,
            "spatial_edge_index": spatial_sub,
            "node_indices": node_indices,
        }


def heist_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for HEIST batches.

    Combines multiple partitions into a single batch with
    properly offset edge indices.

    Args:
        batch: List of partition dicts from HEISTDataset.

    Returns:
        Batched dict with combined data.
    """
    # Concatenate node features
    expression = torch.cat([b["expression"] for b in batch], dim=0)
    coords = torch.cat([b["coords"] for b in batch], dim=0)
    labels = torch.cat([b["labels"] for b in batch], dim=0)
    node_indices = torch.cat([b["node_indices"] for b in batch], dim=0)

    # Combine edge indices with proper offsets
    edge_indices = []
    offset = 0
    for b in batch:
        edges = b["spatial_edge_index"]
        if edges.shape[1] > 0:
            edge_indices.append(edges + offset)
        offset += b["expression"].shape[0]

    if edge_indices:
        spatial_edge_index = torch.cat(edge_indices, dim=1)
    else:
        spatial_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create batch assignment tensor
    batch_idx = []
    for i, b in enumerate(batch):
        batch_idx.append(torch.full((b["expression"].shape[0],), i, dtype=torch.long))
    batch_tensor = torch.cat(batch_idx, dim=0)

    return {
        "expression": expression,
        "coords": coords,
        "labels": labels,
        "spatial_edge_index": spatial_edge_index,
        "node_indices": node_indices,
        "batch": batch_tensor,
    }


def create_heist_data_splits(
    expression: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    spatial_edge_index: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    partition_size: int = 128,
    seed: int = 42,
) -> tuple[HEISTDataset, HEISTDataset, HEISTDataset]:
    """Create train/val/test splits for HEIST.

    Uses stratified sampling to maintain class balance.

    Args:
        expression: (N, n_genes) expression matrix.
        coords: (N, 2) spatial coordinates.
        labels: (N,) cell type labels.
        spatial_edge_index: (2, E) spatial graph edges.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        partition_size: Target partition size.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    from sklearn.model_selection import train_test_split

    n_cells = len(expression)
    indices = np.arange(n_cells)

    # First split: train+val vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx],
        random_state=seed,
    )

    logger.info(
        f"Data splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    # Create datasets for each split
    # Note: Each split gets its own partition of the spatial graph
    def make_dataset(idx: np.ndarray) -> HEISTDataset:
        idx_tensor = torch.from_numpy(idx)

        # Extract subgraph for this split
        sub_edge_index, _ = subgraph(
            idx_tensor,
            spatial_edge_index,
            relabel_nodes=True,
            num_nodes=n_cells,
        )

        return HEISTDataset(
            expression=expression[idx],
            coords=coords[idx],
            labels=labels[idx],
            spatial_edge_index=sub_edge_index,
            partition_size=partition_size,
            permute=True,
        )

    train_ds = make_dataset(train_idx)
    val_ds = make_dataset(val_idx)
    test_ds = make_dataset(test_idx)

    return train_ds, val_ds, test_ds


def load_heist_data(
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, dict[int, torch.Tensor]]:
    """Load HEIST data from prepared directory.

    Expects files:
        - expression.npy: (N, n_genes) expression matrix
        - coords.npy: (N, 2) spatial coordinates
        - labels.npy: (N,) cell type labels
        - spatial_graph.pt: (2, E) spatial edge index
        - cell_type_grns.pt: dict of cell-type GRNs

    Args:
        data_dir: Directory containing prepared data.

    Returns:
        Tuple of (expression, coords, labels, spatial_edge_index, cell_type_grns).
    """
    data_dir = Path(data_dir)

    expression = np.load(data_dir / "expression.npy")
    coords = np.load(data_dir / "coords.npy")
    labels = np.load(data_dir / "labels.npy")
    spatial_edge_index = torch.load(data_dir / "spatial_graph.pt", weights_only=True)
    cell_type_grns = torch.load(data_dir / "cell_type_grns.pt", weights_only=True)

    logger.info(
        f"Loaded HEIST data: {len(expression)} cells, "
        f"{expression.shape[1]} genes, {len(cell_type_grns)} cell-type GRNs"
    )

    return expression, coords, labels, spatial_edge_index, cell_type_grns
