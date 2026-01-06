"""Hierarchical Dataset for multi-level cell type classification.

Extends the base DAPIDLDataset to provide labels at multiple hierarchy levels
(coarse, medium, fine) using the Cell Ontology module.

Returns:
    (patch, labels_dict) where labels_dict = {
        "coarse": coarse_label_idx,
        "medium": medium_label_idx,
        "fine": fine_label_idx,
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
import zarr
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from loguru import logger

from dapidl.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    compute_dataset_stats,
)
from dapidl.models.hierarchical import HierarchyConfig


class HierarchicalLabels:
    """Container for hierarchical label mappings.

    Converts fine-grained labels to coarse/medium/fine indices
    using Cell Ontology mappings.
    """

    def __init__(
        self,
        fine_class_mapping: dict[str, int],
        hierarchy_config: HierarchyConfig | None = None,
    ) -> None:
        """Initialize hierarchical labels.

        Args:
            fine_class_mapping: Fine-grained class name → index mapping
            hierarchy_config: Pre-computed hierarchy config (or compute from CL)
        """
        self.fine_class_mapping = fine_class_mapping
        self.fine_names = list(fine_class_mapping.keys())
        self.num_fine = len(fine_class_mapping)

        if hierarchy_config is None:
            hierarchy_config = self._build_hierarchy_from_cl()

        self.hierarchy_config = hierarchy_config

        # Build label index mappings
        self._build_label_maps()

        logger.info(
            f"HierarchicalLabels: fine={self.num_fine}, "
            f"medium={self.num_medium}, coarse={self.num_coarse}"
        )

    def _build_hierarchy_from_cl(self) -> HierarchyConfig:
        """Build hierarchy configuration from Cell Ontology.

        Returns:
            HierarchyConfig with all mappings
        """
        try:
            from dapidl.ontology import (
                get_broad_category,
                get_coarse_category,
                get_term_by_name,
                map_label,
            )
            has_ontology = True
        except ImportError:
            has_ontology = False
            logger.warning("Ontology module not available, using fallback hierarchy")

        # Collect categories at each level
        coarse_set = set()
        medium_set = set()

        fine_to_medium_name = {}
        fine_to_coarse_name = {}

        for fine_name in self.fine_names:
            if has_ontology:
                # Try to map through ontology
                cl_id = map_label(fine_name)
                if cl_id != "UNMAPPED":
                    medium = get_coarse_category(cl_id) or fine_name
                    coarse = get_broad_category(cl_id) or "Unknown"
                else:
                    # Use heuristic fallback
                    medium = fine_name
                    coarse = self._guess_coarse_category(fine_name)
            else:
                medium = fine_name
                coarse = self._guess_coarse_category(fine_name)

            coarse_set.add(coarse)
            medium_set.add(medium)
            fine_to_medium_name[fine_name] = medium
            fine_to_coarse_name[fine_name] = coarse

        # Create sorted lists and index mappings
        coarse_names = sorted(coarse_set)
        medium_names = sorted(medium_set)

        coarse_to_idx = {name: i for i, name in enumerate(coarse_names)}
        medium_to_idx = {name: i for i, name in enumerate(medium_names)}

        # Build index-based mappings
        fine_to_medium = {}
        fine_to_coarse = {}
        medium_to_coarse = {}

        for fine_name, fine_idx in self.fine_class_mapping.items():
            medium_name = fine_to_medium_name[fine_name]
            coarse_name = fine_to_coarse_name[fine_name]

            medium_idx = medium_to_idx[medium_name]
            coarse_idx = coarse_to_idx[coarse_name]

            fine_to_medium[fine_idx] = medium_idx
            fine_to_coarse[fine_idx] = coarse_idx

            if medium_idx not in medium_to_coarse:
                medium_to_coarse[medium_idx] = coarse_idx

        return HierarchyConfig(
            num_coarse=len(coarse_names),
            num_medium=len(medium_names),
            num_fine=self.num_fine,
            coarse_names=coarse_names,
            medium_names=medium_names,
            fine_names=self.fine_names,
            fine_to_medium=fine_to_medium,
            medium_to_coarse=medium_to_coarse,
            fine_to_coarse=fine_to_coarse,
        )

    def _guess_coarse_category(self, name: str) -> str:
        """Guess coarse category from name when ontology unavailable.

        Args:
            name: Cell type name

        Returns:
            Guessed coarse category
        """
        name_lower = name.lower()

        # Immune keywords
        immune_keywords = [
            "t_cell", "t cell", "b_cell", "b cell", "macrophage", "monocyte",
            "dendritic", "nk", "natural killer", "mast", "neutrophil",
            "lymphocyte", "plasma", "immune",
        ]
        if any(kw in name_lower for kw in immune_keywords):
            return "Immune"

        # Epithelial keywords
        epithelial_keywords = [
            "epithelial", "tumor", "cancer", "carcinoma", "dcis",
            "luminal", "basal", "myoepithelial", "keratinocyte",
        ]
        if any(kw in name_lower for kw in epithelial_keywords):
            return "Epithelial"

        # Stromal keywords
        stromal_keywords = [
            "fibroblast", "stromal", "stroma", "pericyte", "smooth muscle",
            "myofibroblast", "adipocyte",
        ]
        if any(kw in name_lower for kw in stromal_keywords):
            return "Stromal"

        # Endothelial
        if "endothelial" in name_lower or "vascular" in name_lower:
            return "Endothelial"

        return "Unknown"

    def _build_label_maps(self) -> None:
        """Build numpy arrays for fast label conversion."""
        hc = self.hierarchy_config

        self.num_coarse = hc.num_coarse
        self.num_medium = hc.num_medium
        self.coarse_names = hc.coarse_names
        self.medium_names = hc.medium_names

        # Create lookup arrays
        self._fine_to_medium = np.zeros(self.num_fine, dtype=np.int64)
        self._fine_to_coarse = np.zeros(self.num_fine, dtype=np.int64)

        for fine_idx in range(self.num_fine):
            self._fine_to_medium[fine_idx] = hc.fine_to_medium.get(fine_idx, 0)
            self._fine_to_coarse[fine_idx] = hc.fine_to_coarse.get(fine_idx, 0)

    def convert_labels(
        self,
        fine_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert fine labels to all hierarchy levels.

        Args:
            fine_labels: Array of fine-grained label indices

        Returns:
            Tuple of (coarse_labels, medium_labels, fine_labels)
        """
        coarse_labels = self._fine_to_coarse[fine_labels]
        medium_labels = self._fine_to_medium[fine_labels]
        return coarse_labels, medium_labels, fine_labels

    def get_class_weights_per_level(
        self,
        fine_labels: np.ndarray,
        max_weight_ratio: float = 10.0,
    ) -> dict[str, torch.Tensor]:
        """Compute class weights at each hierarchy level.

        Args:
            fine_labels: Array of fine-grained label indices
            max_weight_ratio: Maximum ratio between weights

        Returns:
            Dict with coarse_weights, medium_weights, fine_weights tensors
        """
        coarse_labels, medium_labels, _ = self.convert_labels(fine_labels)

        def compute_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
            counts = np.bincount(labels, minlength=num_classes).astype(float)
            counts = np.maximum(counts, 1)
            weights = 1.0 / counts

            if max_weight_ratio > 0:
                min_weight = weights.min()
                max_allowed = min_weight * max_weight_ratio
                weights = np.minimum(weights, max_allowed)

            weights = weights / weights.sum() * num_classes
            return torch.FloatTensor(weights)

        return {
            "coarse_weights": compute_weights(coarse_labels, self.num_coarse),
            "medium_weights": compute_weights(medium_labels, self.num_medium),
            "fine_weights": compute_weights(fine_labels, self.num_fine),
        }


class HierarchicalDataset(Dataset):
    """PyTorch Dataset for hierarchical cell type classification.

    Extends base dataset to return labels at multiple hierarchy levels.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        indices: np.ndarray | None = None,
        adaptive_norm: bool = True,
        hierarchical_labels: HierarchicalLabels | None = None,
    ) -> None:
        """Initialize hierarchical dataset.

        Args:
            data_path: Path to prepared dataset directory
            split: One of 'train', 'val', 'test'
            transform: Optional custom transform
            indices: Optional subset indices
            adaptive_norm: Use adaptive percentile-based normalization
            hierarchical_labels: Pre-computed hierarchical label mapper
        """
        self.data_path = Path(data_path)
        self.split = split

        # Load base data
        self.patches = zarr.open(self.data_path / "patches.zarr", mode="r")
        self.fine_labels = np.load(self.data_path / "labels.npy")
        self.metadata = pl.read_parquet(self.data_path / "metadata.parquet")

        # Load class mapping
        with open(self.data_path / "class_mapping.json") as f:
            self.class_mapping = json.load(f)

        # Handle indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.fine_labels))

        # Build hierarchical labels
        if hierarchical_labels is None:
            hierarchical_labels = HierarchicalLabels(self.class_mapping)
        self.hierarchical_labels = hierarchical_labels
        self.hierarchy_config = hierarchical_labels.hierarchy_config

        # Convert all labels upfront for efficiency
        subset_fine = self.fine_labels[self.indices]
        self.coarse_labels, self.medium_labels, _ = (
            hierarchical_labels.convert_labels(subset_fine)
        )

        # Compute normalization stats
        self.stats = None
        if adaptive_norm:
            self.stats = compute_dataset_stats(self.data_path)

        # Set transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(stats=self.stats)
        else:
            self.transform = get_val_transforms(stats=self.stats)

        logger.info(
            f"HierarchicalDataset: {len(self)} samples, split={split}, "
            f"coarse={hierarchical_labels.num_coarse}, "
            f"medium={hierarchical_labels.num_medium}, "
            f"fine={hierarchical_labels.num_fine}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int]]:
        """Get a single sample with hierarchical labels.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, labels_dict) where labels_dict contains
            'coarse', 'medium', 'fine' indices
        """
        actual_idx = self.indices[idx]

        # Load patch
        patch = np.array(self.patches[actual_idx])

        # Get labels at each level
        coarse_label = self.coarse_labels[idx]
        medium_label = self.medium_labels[idx]
        fine_label = self.fine_labels[actual_idx]

        # Apply transform
        if self.transform is not None:
            transformed = self.transform(image=patch)
            patch = transformed["image"]

        labels = {
            "coarse": int(coarse_label),
            "medium": int(medium_label),
            "fine": int(fine_label),
        }

        return patch, labels

    def get_class_weights(
        self, max_weight_ratio: float = 10.0
    ) -> dict[str, torch.Tensor]:
        """Get class weights at each hierarchy level.

        Args:
            max_weight_ratio: Maximum ratio between weights

        Returns:
            Dict with coarse_weights, medium_weights, fine_weights
        """
        subset_fine = self.fine_labels[self.indices]
        return self.hierarchical_labels.get_class_weights_per_level(
            subset_fine, max_weight_ratio
        )

    def get_sample_weights(self, level: str = "fine") -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler.

        Args:
            level: Which hierarchy level to use ('coarse', 'medium', 'fine')

        Returns:
            Array of sample weights
        """
        weights_dict = self.get_class_weights()
        class_weights = weights_dict[f"{level}_weights"].numpy()

        if level == "coarse":
            labels = self.coarse_labels
        elif level == "medium":
            labels = self.medium_labels
        else:
            labels = self.fine_labels[self.indices]

        return class_weights[labels]


def create_hierarchical_data_splits(
    data_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    stratify_level: str = "fine",
    min_samples_per_class: int | None = None,
) -> tuple[HierarchicalDataset, HierarchicalDataset, HierarchicalDataset, HierarchicalLabels]:
    """Create train/val/test splits for hierarchical dataset.

    Args:
        data_path: Path to prepared dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        stratify: Whether to stratify by labels
        stratify_level: Which level to stratify by
        min_samples_per_class: Minimum samples per class

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, hierarchical_labels)
    """
    data_path = Path(data_path)

    # Load labels
    labels = np.load(data_path / "labels.npy")
    n_samples = len(labels)
    indices = np.arange(n_samples)

    # Load class mapping
    with open(data_path / "class_mapping.json") as f:
        class_mapping = json.load(f)

    # Build hierarchical labels (shared across splits)
    hierarchical_labels = HierarchicalLabels(class_mapping)

    # Convert labels for stratification
    if stratify_level == "coarse":
        coarse_labels, _, _ = hierarchical_labels.convert_labels(labels)
        stratify_labels = coarse_labels
    elif stratify_level == "medium":
        _, medium_labels, _ = hierarchical_labels.convert_labels(labels)
        stratify_labels = medium_labels
    else:
        stratify_labels = labels

    # Filter rare classes if needed
    if min_samples_per_class is not None:
        unique_classes, counts = np.unique(stratify_labels, return_counts=True)
        valid_classes = unique_classes[counts >= min_samples_per_class]
        mask = np.isin(stratify_labels, valid_classes)
        indices = indices[mask]
        stratify_labels = stratify_labels[mask]
        logger.info(f"Filtered to {len(indices)} samples with min {min_samples_per_class} per class")

    # Split
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify_labels if stratify else None,
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    if stratify:
        temp_labels = stratify_labels[np.isin(indices, temp_indices)]
        temp_mask = np.isin(indices, temp_indices)
        temp_stratify = stratify_labels[temp_mask]
    else:
        temp_stratify = None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=seed,
        stratify=temp_stratify if stratify else None,
    )

    logger.info(
        f"Hierarchical splits: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    # Create datasets
    train_dataset = HierarchicalDataset(
        data_path, split="train", indices=train_indices,
        hierarchical_labels=hierarchical_labels,
    )
    val_dataset = HierarchicalDataset(
        data_path, split="val", indices=val_indices,
        hierarchical_labels=hierarchical_labels,
    )
    test_dataset = HierarchicalDataset(
        data_path, split="test", indices=test_indices,
        hierarchical_labels=hierarchical_labels,
    )

    return train_dataset, val_dataset, test_dataset, hierarchical_labels


def create_hierarchical_dataloaders(
    train_dataset: HierarchicalDataset,
    val_dataset: HierarchicalDataset,
    test_dataset: HierarchicalDataset | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    use_weighted_sampler: bool = True,
    sample_weight_level: str = "fine",
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders for hierarchical datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        use_weighted_sampler: Use WeightedRandomSampler
        sample_weight_level: Which level to use for sampling weights
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    def hierarchical_collate(batch):
        """Custom collate function for hierarchical labels."""
        images = torch.stack([item[0] for item in batch])
        labels = {
            "coarse": torch.tensor([item[1]["coarse"] for item in batch]),
            "medium": torch.tensor([item[1]["medium"] for item in batch]),
            "fine": torch.tensor([item[1]["fine"] for item in batch]),
        }
        return images, labels

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers and num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "collate_fn": hierarchical_collate,
    }

    # Training loader
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights(level=sample_weight_level)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader
