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

from dapidl.data.transforms import get_train_transforms, get_val_transforms


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
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to prepared dataset directory
            split: One of 'train', 'val', 'test' (used for default transforms)
            transform: Optional custom transform (uses default if None)
            indices: Optional subset indices (for train/val/test splits)
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

        # Set transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

        logger.info(
            f"DAPIDLDataset: {len(self)} samples, {self.num_classes} classes, split={split}"
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

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data.

        Returns:
            Tensor of class weights (inverse frequency)
        """
        labels = self.labels[self.indices]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
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


def create_data_splits(
    data_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[DAPIDLDataset, DAPIDLDataset, DAPIDLDataset]:
    """Create train/val/test splits of the dataset.

    Args:
        data_path: Path to prepared dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        stratify: Whether to stratify by class labels

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_path)

    # Load labels for stratification
    labels = np.load(data_path / "labels.npy")
    n_samples = len(labels)
    indices = np.arange(n_samples)

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels if stratify else None,
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=seed,
        stratify=labels[temp_indices] if stratify else None,
    )

    logger.info(
        f"Data splits: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    # Create datasets
    train_dataset = DAPIDLDataset(data_path, split="train", indices=train_indices)
    val_dataset = DAPIDLDataset(data_path, split="val", indices=val_indices)
    test_dataset = DAPIDLDataset(data_path, split="test", indices=test_indices)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: DAPIDLDataset,
    val_dataset: DAPIDLDataset,
    test_dataset: DAPIDLDataset | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders from datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        use_weighted_sampler: Use WeightedRandomSampler for balanced batches

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
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
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Validation loader (no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test loader (no shuffling)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
