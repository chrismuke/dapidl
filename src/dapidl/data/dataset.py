"""PyTorch Dataset for DAPIDL."""

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
    get_heavy_augmentation_transforms,
    compute_dataset_stats,
)


class DAPIDLDataset(Dataset):
    """PyTorch Dataset for DAPIDL cell type classification.

    Loads preprocessed patches from Zarr format and applies transforms.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        indices: np.ndarray | None = None,
        adaptive_norm: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to prepared dataset directory
            split: One of 'train', 'val', 'test' (used for default transforms)
            transform: Optional custom transform (uses default if None)
            indices: Optional subset indices (for train/val/test splits)
            adaptive_norm: Use adaptive percentile-based normalization (default True)
        """
        self.data_path = Path(data_path)
        self.split = split

        # Load data
        self.patches = zarr.open(self.data_path / "patches.zarr", mode="r")
        self.labels = np.load(self.data_path / "labels.npy")
        self.metadata = pl.read_parquet(self.data_path / "metadata.parquet")

        # Load class mapping
        with open(self.data_path / "class_mapping.json") as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)
        self.class_names = list(self.class_mapping.keys())

        # Handle indices for splits
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.labels))

        # Compute/load normalization stats if using adaptive normalization
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
            f"DAPIDLDataset: {len(self)} samples, {self.num_classes} classes, "
            f"split={split}, adaptive_norm={adaptive_norm}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label)
        """
        actual_idx = self.indices[idx]

        # Load patch (shape: H, W)
        patch = np.array(self.patches[actual_idx])
        label = self.labels[actual_idx]

        # Apply transform
        if self.transform is not None:
            transformed = self.transform(image=patch)
            patch = transformed["image"]

        return patch, int(label)

    def get_class_weights(self, max_weight_ratio: float = 10.0) -> torch.Tensor:
        """Compute class weights for imbalanced data.

        Args:
            max_weight_ratio: Maximum ratio between largest and smallest weight.
                              Prevents extreme over-weighting of rare classes.
                              Default 10.0 means rare classes get at most 10x
                              the weight of the most common class.

        Returns:
            Tensor of class weights (inverse frequency, capped)
        """
        labels = self.labels[self.indices]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts

        # Cap the weight ratio to prevent mode collapse
        if max_weight_ratio is not None and max_weight_ratio > 0:
            min_weight = weights.min()
            max_allowed = min_weight * max_weight_ratio
            weights = np.minimum(weights, max_allowed)
            logger.debug(
                f"Class weights capped at {max_weight_ratio}x ratio "
                f"(min={min_weight:.6f}, max={weights.max():.6f})"
            )

        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler.

        Returns:
            Array of sample weights
        """
        class_weights = self.get_class_weights().numpy()
        labels = self.labels[self.indices]
        return class_weights[labels]


class DAPIDLDatasetWithHeavyAug(Dataset):
    """Dataset with class-conditional augmentation for rare classes.

    Applies heavy augmentation to samples from rare classes (< threshold % of data)
    to increase variability and help the model generalize better.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        indices: np.ndarray | None = None,
        adaptive_norm: bool = True,
        rare_class_threshold: float = 0.05,
    ) -> None:
        """Initialize dataset with class-conditional augmentation.

        Args:
            data_path: Path to prepared dataset directory
            split: One of 'train', 'val', 'test'
            indices: Optional subset indices
            adaptive_norm: Use adaptive percentile-based normalization
            rare_class_threshold: Classes with < this fraction are "rare" (default 5%)
        """
        self.data_path = Path(data_path)
        self.split = split

        # Load data
        self.patches = zarr.open(self.data_path / "patches.zarr", mode="r")
        self.labels = np.load(self.data_path / "labels.npy")
        self.metadata = pl.read_parquet(self.data_path / "metadata.parquet")

        # Load class mapping
        with open(self.data_path / "class_mapping.json") as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)
        self.class_names = list(self.class_mapping.keys())

        # Handle indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.labels))

        # Compute stats
        self.stats = None
        if adaptive_norm:
            self.stats = compute_dataset_stats(self.data_path)

        # Identify rare classes based on training distribution
        labels_subset = self.labels[self.indices]
        class_counts = np.bincount(labels_subset, minlength=self.num_classes)
        total = len(labels_subset)
        threshold_count = rare_class_threshold * total

        self.rare_classes = set(
            idx for idx, count in enumerate(class_counts)
            if 0 < count < threshold_count
        )

        # Create transforms
        if split == "train":
            self.normal_transform = get_train_transforms(stats=self.stats)
            self.heavy_transform = get_heavy_augmentation_transforms(stats=self.stats)
        else:
            self.normal_transform = get_val_transforms(stats=self.stats)
            self.heavy_transform = None  # No heavy aug for val/test

        logger.info(
            f"DAPIDLDatasetWithHeavyAug: {len(self)} samples, {self.num_classes} classes, "
            f"{len(self.rare_classes)} rare classes (<{rare_class_threshold*100:.0f}%)"
        )
        if self.rare_classes:
            rare_names = [self.class_names[i] for i in self.rare_classes]
            logger.info(f"Rare classes: {rare_names}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        actual_idx = self.indices[idx]
        patch = np.array(self.patches[actual_idx])
        label = self.labels[actual_idx]

        # Use heavy augmentation for rare classes during training
        if self.split == "train" and label in self.rare_classes and self.heavy_transform is not None:
            transformed = self.heavy_transform(image=patch)
        else:
            transformed = self.normal_transform(image=patch)

        patch = transformed["image"]
        return patch, int(label)

    def get_class_weights(self, max_weight_ratio: float = 10.0) -> torch.Tensor:
        """Compute class weights (same as base class)."""
        labels = self.labels[self.indices]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        if max_weight_ratio is not None and max_weight_ratio > 0:
            min_weight = weights.min()
            max_allowed = min_weight * max_weight_ratio
            weights = np.minimum(weights, max_allowed)
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        labels = self.labels[self.indices]
        return class_weights[labels]


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning (classification + segmentation).

    Extends the base dataset to also return nucleus masks for the
    auxiliary segmentation task. Masks can come from:
    1. Pre-computed masks stored in masks.zarr
    2. Simple circular masks based on patch center (fallback)

    The segmentation task is auxiliary - it helps the backbone learn better
    features for classification by forcing it to understand nuclear morphology.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        indices: np.ndarray | None = None,
        adaptive_norm: bool = True,
        mask_source: str = "auto",
        fallback_mask_radius: int = 32,
    ) -> None:
        """Initialize multi-task dataset.

        Args:
            data_path: Path to prepared dataset directory
            split: One of 'train', 'val', 'test'
            transform: Optional custom transform
            indices: Optional subset indices
            adaptive_norm: Use adaptive percentile-based normalization
            mask_source: Source for masks:
                - 'auto': Try masks.zarr, fallback to circular
                - 'zarr': Load from masks.zarr (error if not found)
                - 'circular': Generate circular masks at patch center
            fallback_mask_radius: Radius for circular fallback masks
        """
        self.data_path = Path(data_path)
        self.split = split
        self.mask_source = mask_source
        self.fallback_mask_radius = fallback_mask_radius

        # Load data
        self.patches = zarr.open(self.data_path / "patches.zarr", mode="r")
        self.labels = np.load(self.data_path / "labels.npy")
        self.metadata = pl.read_parquet(self.data_path / "metadata.parquet")

        # Load class mapping
        with open(self.data_path / "class_mapping.json") as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)
        self.class_names = list(self.class_mapping.keys())

        # Handle indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.labels))

        # Load masks if available
        self.masks = None
        masks_path = self.data_path / "masks.zarr"
        if mask_source == "zarr":
            if not masks_path.exists():
                raise FileNotFoundError(
                    f"Mask zarr not found at {masks_path}. "
                    "Run mask generation first or use mask_source='circular'"
                )
            self.masks = zarr.open(masks_path, mode="r")
        elif mask_source == "auto":
            if masks_path.exists():
                self.masks = zarr.open(masks_path, mode="r")
                logger.info(f"Loaded masks from {masks_path}")
            else:
                logger.info(
                    f"No masks.zarr found, using circular masks "
                    f"(radius={fallback_mask_radius})"
                )

        # Compute stats
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

        # Pre-compute circular mask template for fallback
        self._circular_mask_template = None
        if self.masks is None:
            self._create_circular_mask_template()

        logger.info(
            f"MultiTaskDataset: {len(self)} samples, {self.num_classes} classes, "
            f"split={split}, mask_source={mask_source}"
        )

    def _create_circular_mask_template(self) -> None:
        """Create circular mask template for fallback."""
        # Get patch size from first patch
        sample_patch = np.array(self.patches[0])
        h, w = sample_patch.shape[:2]
        center = (h // 2, w // 2)

        # Create circular mask
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        self._circular_mask_template = (dist <= self.fallback_mask_radius).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Get a single sample with mask.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label, mask_tensor)
            mask_tensor has shape (1, H, W) with values 0 or 1
        """
        actual_idx = self.indices[idx]

        # Load patch
        patch = np.array(self.patches[actual_idx])
        label = self.labels[actual_idx]

        # Load or generate mask
        if self.masks is not None:
            mask = np.array(self.masks[actual_idx]).astype(np.float32)
            # Ensure binary mask
            mask = (mask > 0).astype(np.float32)
        else:
            # Use circular fallback
            mask = self._circular_mask_template.copy()

        # Apply transform (to both image and mask)
        if self.transform is not None:
            transformed = self.transform(image=patch, mask=mask)
            patch = transformed["image"]
            mask = transformed["mask"]

        # Ensure mask has channel dimension
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return patch, int(label), mask

    def get_class_weights(self, max_weight_ratio: float = 10.0) -> torch.Tensor:
        """Compute class weights (same as base class)."""
        labels = self.labels[self.indices]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        if max_weight_ratio is not None and max_weight_ratio > 0:
            min_weight = weights.min()
            max_allowed = min_weight * max_weight_ratio
            weights = np.minimum(weights, max_allowed)
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        labels = self.labels[self.indices]
        return class_weights[labels]


def create_multitask_data_splits(
    data_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    min_samples_per_class: int | None = None,
    mask_source: str = "auto",
    fallback_mask_radius: int = 32,
) -> tuple[MultiTaskDataset, MultiTaskDataset, MultiTaskDataset]:
    """Create train/val/test splits of the multi-task dataset.

    Args:
        data_path: Path to prepared dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        stratify: Whether to stratify by class labels
        min_samples_per_class: Minimum samples per class
        mask_source: Source for segmentation masks
        fallback_mask_radius: Radius for circular fallback masks

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Get split indices using existing function
    train_indices, val_indices, test_indices = get_split_indices(
        data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify=stratify,
        min_samples_per_class=min_samples_per_class,
    )

    # Create multi-task datasets
    train_dataset = MultiTaskDataset(
        data_path, split="train", indices=train_indices,
        mask_source=mask_source, fallback_mask_radius=fallback_mask_radius
    )
    val_dataset = MultiTaskDataset(
        data_path, split="val", indices=val_indices,
        mask_source=mask_source, fallback_mask_radius=fallback_mask_radius
    )
    test_dataset = MultiTaskDataset(
        data_path, split="test", indices=test_indices,
        mask_source=mask_source, fallback_mask_radius=fallback_mask_radius
    )

    return train_dataset, val_dataset, test_dataset


def create_data_splits(
    data_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    min_samples_per_class: int | None = None,
    use_heavy_aug: bool = False,
) -> tuple[DAPIDLDataset, DAPIDLDataset, DAPIDLDataset]:
    """Create train/val/test splits of the dataset.

    Args:
        data_path: Path to prepared dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        stratify: Whether to stratify by class labels
        min_samples_per_class: Minimum samples per class (filter rare classes)
        use_heavy_aug: Use heavy augmentation for rare classes (only for training)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_path)

    # Load labels for stratification
    labels = np.load(data_path / "labels.npy")
    n_samples = len(labels)
    indices = np.arange(n_samples)

    # Filter rare classes if min_samples_per_class is set
    if min_samples_per_class is not None:
        unique_classes, counts = np.unique(labels, return_counts=True)
        valid_classes = unique_classes[counts >= min_samples_per_class]
        rare_classes = unique_classes[counts < min_samples_per_class]
        if len(rare_classes) > 0:
            logger.warning(
                f"Filtering {len(rare_classes)} classes with < {min_samples_per_class} samples"
            )
            mask = np.isin(labels, valid_classes)
            indices = indices[mask]
            labels = labels[mask]
            n_samples = len(indices)
            logger.info(f"Remaining samples after filtering: {n_samples}")

    # For stratification, we need the labels corresponding to current indices
    # When filtering is done, labels are filtered alongside indices
    # Create a mapping from indices to their positions for stratification
    stratify_labels = labels if stratify else None

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    # Second split: val vs test
    # Get the labels for temp_indices for stratification
    val_size = val_ratio / (val_ratio + test_ratio)
    if stratify:
        # Find positions of temp_indices in the indices array to get their labels
        temp_positions = np.searchsorted(indices, temp_indices)
        temp_labels = labels[temp_positions] if len(indices) == len(labels) else None
        if temp_labels is None:
            # Fallback: just get labels by the indices themselves from original
            all_labels = np.load(data_path / "labels.npy")
            temp_labels = all_labels[temp_indices]
    else:
        temp_labels = None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=seed,
        stratify=temp_labels,
    )

    logger.info(
        f"Data splits: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    # Create datasets
    if use_heavy_aug:
        # Use class-conditional augmentation for training
        train_dataset = DAPIDLDatasetWithHeavyAug(data_path, split="train", indices=train_indices)
        logger.info("Using heavy augmentation for rare classes in training set")
    else:
        train_dataset = DAPIDLDataset(data_path, split="train", indices=train_indices)

    val_dataset = DAPIDLDataset(data_path, split="val", indices=val_indices)
    test_dataset = DAPIDLDataset(data_path, split="test", indices=test_indices)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: DAPIDLDataset,
    val_dataset: DAPIDLDataset,
    test_dataset: DAPIDLDataset | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    use_weighted_sampler: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders from datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of worker processes (default 8 for better GPU utilization)
        use_weighted_sampler: Use WeightedRandomSampler for balanced batches
        prefetch_factor: Number of batches to prefetch per worker (default 4)
        persistent_workers: Keep workers alive between epochs (default True)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Common DataLoader kwargs for GPU saturation
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers and num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }

    # Training loader with optional weighted sampling
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
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

    # Validation loader (no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Test loader (no shuffling)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader


def get_split_indices(
    data_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    min_samples_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get train/val/test split indices without creating datasets.

    This is useful for DALI backend which needs raw indices.

    Args:
        data_path: Path to prepared dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        stratify: Whether to stratify by class labels
        min_samples_per_class: Minimum samples per class (filter rare classes)

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    data_path = Path(data_path)

    # Load labels for stratification
    labels = np.load(data_path / "labels.npy")
    n_samples = len(labels)
    indices = np.arange(n_samples)

    # Filter rare classes if min_samples_per_class is set
    if min_samples_per_class is not None:
        unique_classes, counts = np.unique(labels, return_counts=True)
        valid_classes = unique_classes[counts >= min_samples_per_class]
        rare_classes = unique_classes[counts < min_samples_per_class]
        if len(rare_classes) > 0:
            logger.warning(
                f"Filtering {len(rare_classes)} classes with < {min_samples_per_class} samples"
            )
            mask = np.isin(labels, valid_classes)
            indices = indices[mask]
            labels = labels[mask]
            n_samples = len(indices)
            logger.info(f"Remaining samples after filtering: {n_samples}")

    stratify_labels = labels if stratify else None

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    if stratify:
        temp_positions = np.searchsorted(indices, temp_indices)
        temp_labels = labels[temp_positions] if len(indices) == len(labels) else None
        if temp_labels is None:
            all_labels = np.load(data_path / "labels.npy")
            temp_labels = all_labels[temp_indices]
    else:
        temp_labels = None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=seed,
        stratify=temp_labels,
    )

    logger.info(
        f"Data splits: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    return train_indices, val_indices, test_indices


def create_dataloaders_with_backend(
    data_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 8,
    backend: str = "pytorch",
    use_weighted_sampler: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    seed: int = 42,
    min_samples_per_class: int | None = None,
    device_id: int = 0,
) -> tuple:
    """Create DataLoaders with selectable backend (PyTorch, DALI, or DALI-LMDB).

    Args:
        data_path: Path to prepared dataset directory
        batch_size: Batch size
        num_workers: Number of worker processes
        backend: "pytorch", "dali", or "dali-lmdb"
        use_weighted_sampler: Use WeightedRandomSampler (PyTorch only)
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        seed: Random seed
        min_samples_per_class: Minimum samples per class to include
        device_id: GPU device ID (DALI only)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata)
        metadata contains num_classes, class_names, class_weights
    """
    data_path = Path(data_path)

    # Get split indices
    train_indices, val_indices, test_indices = get_split_indices(
        data_path,
        seed=seed,
        min_samples_per_class=min_samples_per_class,
    )

    # Load class info for metadata
    with open(data_path / "class_mapping.json") as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping)
    class_names = list(class_mapping.keys())

    # Compute class weights
    labels = np.load(data_path / "labels.npy")
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    # Cap at 10x ratio
    min_weight = weights.min()
    max_allowed = min_weight * 10.0
    weights = np.minimum(weights, max_allowed)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights)

    metadata = {
        "num_classes": num_classes,
        "class_names": class_names,
        "class_weights": class_weights,
        "backend": backend,
    }

    if backend == "dali-lmdb":
        # Use DALI with LMDB backend (fastest)
        from dapidl.data.dali_native import (
            is_lmdb_available,
            create_dali_lmdb_dataloaders,
        )
        from dapidl.data.dali_pipeline import is_dali_available

        if not is_dali_available():
            raise RuntimeError(
                "DALI backend requested but DALI is not installed. "
                "Install with: pip install nvidia-dali-cuda120"
            )
        if not is_lmdb_available():
            raise RuntimeError(
                "LMDB backend requested but LMDB is not installed. "
                "Install with: pip install lmdb"
            )

        # Check if LMDB exists
        lmdb_path = data_path / "patches.lmdb"
        if not lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB database not found at {lmdb_path}. "
                "Run 'dapidl export-lmdb -d <dataset_path>' to create it."
            )

        train_loader, val_loader, test_loader = create_dali_lmdb_dataloaders(
            data_path=data_path,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_factor,
        )

        logger.info(f"Created DALI-LMDB DataLoaders: backend={backend}, device={device_id}")

    elif backend == "dali":
        # Use DALI backend with Zarr (slower due to Python external source)
        from dapidl.data.dali_pipeline import is_dali_available, create_dali_dataloaders

        if not is_dali_available():
            raise RuntimeError(
                "DALI backend requested but DALI is not installed. "
                "Install with: pip install nvidia-dali-cuda120"
            )

        train_loader, val_loader, test_loader = create_dali_dataloaders(
            data_path=data_path,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            batch_size=batch_size,
            num_threads=num_workers,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_factor,
        )

        logger.info(f"Created DALI DataLoaders: backend={backend}, device={device_id}")

    else:
        # Use PyTorch backend (default)
        train_dataset = DAPIDLDataset(data_path, split="train", indices=train_indices)
        val_dataset = DAPIDLDataset(data_path, split="val", indices=val_indices)
        test_dataset = DAPIDLDataset(data_path, split="test", indices=test_indices)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            use_weighted_sampler=use_weighted_sampler,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

        logger.info(f"Created PyTorch DataLoaders: backend={backend}")

    return train_loader, val_loader, test_loader, metadata
