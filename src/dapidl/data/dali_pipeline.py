"""NVIDIA DALI pipeline for GPU-accelerated data loading and augmentation.

This module provides an optional DALI backend for training that moves
data augmentation from CPU to GPU, significantly improving training throughput.

Requires: nvidia-dali-cuda12 (install with: pip install nvidia-dali-cuda120)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr
from loguru import logger

# DALI imports are optional
try:
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import math as dali_math

    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    pipeline_def = None
    DALIGenericIterator = None
    LastBatchPolicy = None
    fn = None
    types = None
    dali_math = None

if TYPE_CHECKING:
    from nvidia.dali import Pipeline


def is_dali_available() -> bool:
    """Check if DALI is available."""
    return DALI_AVAILABLE


class ZarrExternalSource:
    """External source for reading patches from Zarr arrays.

    This class provides data to DALI from Zarr storage, supporting
    both training (with shuffling) and validation modes.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        labels_path: str | Path,
        indices: np.ndarray | None = None,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """Initialize Zarr external source.

        Args:
            zarr_path: Path to patches.zarr
            labels_path: Path to labels.npy
            indices: Optional subset indices for train/val/test splits
            batch_size: Batch size for iteration
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.zarr_path = Path(zarr_path)
        self.patches = zarr.open(self.zarr_path, mode="r")
        self.all_labels = np.load(labels_path)

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.all_labels))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.n_samples = len(self.indices)
        self.current_idx = 0
        self._epoch_indices = self.indices.copy()

        if self.shuffle:
            self.rng.shuffle(self._epoch_indices)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __call__(self, sample_info) -> tuple[np.ndarray, np.ndarray]:
        """Get a single sample for DALI.

        Args:
            sample_info: DALI sample info with idx_in_epoch

        Returns:
            Tuple of (image, label) as numpy arrays
        """
        idx = sample_info.idx_in_epoch

        if idx >= self.n_samples:
            raise StopIteration()

        actual_idx = int(self._epoch_indices[idx])

        # Load patch (H, W) as uint16
        patch = np.array(self.patches[actual_idx])
        label = np.array([self.all_labels[actual_idx]], dtype=np.int64)

        # DALI expects HWC format, add channel dimension
        patch = patch[:, :, np.newaxis]

        return patch, label

    def reset(self) -> None:
        """Reset for new epoch."""
        self.current_idx = 0
        if self.shuffle:
            self.rng.shuffle(self._epoch_indices)


def create_dali_train_pipeline(
    zarr_path: str | Path,
    labels_path: str | Path,
    indices: np.ndarray,
    batch_size: int = 64,
    num_threads: int = 4,
    device_id: int = 0,
    seed: int = 42,
    stats: dict[str, float] | None = None,
    prefetch_queue_depth: int = 2,
) -> "Pipeline":
    """Create DALI training pipeline with GPU augmentations.

    Args:
        zarr_path: Path to patches.zarr
        labels_path: Path to labels.npy
        indices: Training indices
        batch_size: Batch size
        num_threads: Number of CPU threads for data loading
        device_id: GPU device ID
        seed: Random seed
        stats: Normalization statistics (p_low, p_high, mean, std)
        prefetch_queue_depth: Number of batches to prefetch

    Returns:
        DALI Pipeline for training
    """
    if not DALI_AVAILABLE:
        raise RuntimeError(
            "DALI is not available. Install with: pip install nvidia-dali-cuda120"
        )

    # Default normalization stats
    if stats is None:
        stats = {"p_low": 0.0, "p_high": 65535.0, "mean": 0.5, "std": 0.25}

    p_low = stats["p_low"]
    p_high = stats["p_high"]
    mean = stats["mean"]
    std = stats["std"]

    # Create external source
    source = ZarrExternalSource(
        zarr_path=zarr_path,
        labels_path=labels_path,
        indices=indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed, py_start_method="spawn")
    def train_pipeline():
        # Read from external source
        images, labels = fn.external_source(
            source=source,
            num_outputs=2,
            batch=False,
            parallel=True,
            prefetch_queue_depth=prefetch_queue_depth,
        )

        # Transfer to GPU
        images = images.gpu()

        # Convert uint16 to float32 and normalize to [0, 1] based on percentiles
        # Clip to percentile range first
        images = fn.cast(images, dtype=types.FLOAT)
        images = dali_math.clamp(images, p_low, p_high)
        # Scale to [0, 1]
        images = (images - p_low) / (p_high - p_low + 1e-8)

        # GPU-accelerated augmentations
        # Random rotation (0, 90, 180, 270 degrees)
        angle = fn.random.uniform(range=(0, 4), dtype=types.INT32)
        angle = fn.cast(angle, dtype=types.FLOAT) * 90.0
        images = fn.rotate(images, angle=angle, fill_value=0, keep_size=True)

        # Random horizontal flip
        images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

        # Random vertical flip
        images = fn.flip(images, vertical=fn.random.coin_flip(probability=0.5))

        # Brightness/contrast adjustment
        brightness = fn.random.uniform(range=(0.8, 1.2))
        contrast = fn.random.uniform(range=(0.8, 1.2))
        images = fn.brightness_contrast(images, brightness=brightness, contrast=contrast)

        # Gaussian blur (applied with probability)
        blur_sigma = fn.random.uniform(range=(0.1, 1.0))
        do_blur = fn.random.coin_flip(probability=0.2)
        blurred = fn.gaussian_blur(images, sigma=blur_sigma, window_size=5)
        images = do_blur * blurred + (1 - do_blur) * images

        # Gaussian noise
        do_noise = fn.random.coin_flip(probability=0.3)
        noise_std = fn.random.uniform(range=(0.01, 0.05))
        noise = fn.random.normal(images, stddev=noise_std)
        images = do_noise * (images + noise) + (1 - do_noise) * images
        images = dali_math.clamp(images, 0.0, 1.0)

        # Final normalization
        images = (images - mean) / std

        # Convert from HWC to CHW format for PyTorch
        # Keep single channel - model's SingleChannelAdapter will handle 1->3 conversion
        images = fn.transpose(images, perm=[2, 0, 1])

        return images, labels

    pipe = train_pipeline()
    pipe.build()
    logger.info(f"DALI training pipeline created: batch_size={batch_size}, device={device_id}")
    return pipe, source


def create_dali_val_pipeline(
    zarr_path: str | Path,
    labels_path: str | Path,
    indices: np.ndarray,
    batch_size: int = 64,
    num_threads: int = 4,
    device_id: int = 0,
    stats: dict[str, float] | None = None,
    prefetch_queue_depth: int = 2,
) -> "Pipeline":
    """Create DALI validation pipeline (no augmentation).

    Args:
        zarr_path: Path to patches.zarr
        labels_path: Path to labels.npy
        indices: Validation indices
        batch_size: Batch size
        num_threads: Number of CPU threads
        device_id: GPU device ID
        stats: Normalization statistics

    Returns:
        DALI Pipeline for validation
    """
    if not DALI_AVAILABLE:
        raise RuntimeError(
            "DALI is not available. Install with: pip install nvidia-dali-cuda120"
        )

    # Default normalization stats
    if stats is None:
        stats = {"p_low": 0.0, "p_high": 65535.0, "mean": 0.5, "std": 0.25}

    p_low = stats["p_low"]
    p_high = stats["p_high"]
    mean = stats["mean"]
    std = stats["std"]

    # Create external source (no shuffle for validation)
    source = ZarrExternalSource(
        zarr_path=zarr_path,
        labels_path=labels_path,
        indices=indices,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id, py_start_method="spawn")
    def val_pipeline():
        # Read from external source
        images, labels = fn.external_source(
            source=source,
            num_outputs=2,
            batch=False,
            parallel=True,
            prefetch_queue_depth=prefetch_queue_depth,
        )

        # Transfer to GPU
        images = images.gpu()

        # Convert uint16 to float32 and normalize
        images = fn.cast(images, dtype=types.FLOAT)
        images = dali_math.clamp(images, p_low, p_high)
        images = (images - p_low) / (p_high - p_low + 1e-8)

        # Final normalization (no augmentation)
        images = (images - mean) / std

        # Convert from HWC to CHW format for PyTorch
        # Keep single channel - model's SingleChannelAdapter will handle 1->3 conversion
        images = fn.transpose(images, perm=[2, 0, 1])

        return images, labels

    pipe = val_pipeline()
    pipe.build()
    logger.info(f"DALI validation pipeline created: batch_size={batch_size}, device={device_id}")
    return pipe, source


class DALIDataLoader:
    """PyTorch-compatible DataLoader wrapper for DALI pipelines.

    This class wraps DALI pipelines to provide an interface compatible
    with PyTorch training loops.
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        source: ZarrExternalSource,
        output_map: list[str] = None,
        auto_reset: bool = True,
    ):
        """Initialize DALI DataLoader.

        Args:
            pipeline: DALI pipeline
            source: External source for epoch reset
            output_map: Names for pipeline outputs
            auto_reset: Whether to auto-reset between epochs
        """
        if not DALI_AVAILABLE:
            raise RuntimeError("DALI is not available")

        if output_map is None:
            output_map = ["images", "labels"]

        self.pipeline = pipeline
        self.source = source
        self.auto_reset = auto_reset

        self.iterator = DALIGenericIterator(
            pipelines=[pipeline],
            output_map=output_map,
            auto_reset=auto_reset,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        self._len = len(source)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        """Get next batch.

        Returns:
            Tuple of (images, labels) as PyTorch tensors on GPU
        """
        try:
            data = next(self.iterator)
            images = data[0]["images"]
            labels = data[0]["labels"].squeeze(-1)  # Remove extra dimension
            return images, labels
        except StopIteration:
            self.source.reset()
            raise

    def reset(self) -> None:
        """Reset iterator for new epoch."""
        self.source.reset()
        self.iterator.reset()


def create_dali_dataloaders(
    data_path: str | Path,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None = None,
    batch_size: int = 64,
    num_threads: int = 8,
    device_id: int = 0,
    seed: int = 42,
    prefetch_queue_depth: int = 2,
) -> tuple[DALIDataLoader, DALIDataLoader, DALIDataLoader | None]:
    """Create DALI DataLoaders for training, validation, and testing.

    Args:
        data_path: Path to dataset directory (containing patches.zarr, labels.npy)
        train_indices: Training sample indices
        val_indices: Validation sample indices
        test_indices: Optional test sample indices
        batch_size: Batch size
        num_threads: Number of CPU threads for data loading
        device_id: GPU device ID
        seed: Random seed
        prefetch_queue_depth: Number of batches to prefetch

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if not DALI_AVAILABLE:
        raise RuntimeError(
            "DALI is not available. Install with: pip install nvidia-dali-cuda120"
        )

    data_path = Path(data_path)
    zarr_path = data_path / "patches.zarr"
    labels_path = data_path / "labels.npy"
    stats_path = data_path / "normalization_stats.json"

    # Load normalization stats
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        logger.info(f"Loaded normalization stats: {stats}")
    else:
        stats = None
        logger.warning("No normalization stats found, using defaults")

    # Create training pipeline and loader
    train_pipe, train_source = create_dali_train_pipeline(
        zarr_path=zarr_path,
        labels_path=labels_path,
        indices=train_indices,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        stats=stats,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    train_loader = DALIDataLoader(train_pipe, train_source)

    # Create validation pipeline and loader
    val_pipe, val_source = create_dali_val_pipeline(
        zarr_path=zarr_path,
        labels_path=labels_path,
        indices=val_indices,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        stats=stats,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    val_loader = DALIDataLoader(val_pipe, val_source)

    # Create test pipeline and loader if needed
    test_loader = None
    if test_indices is not None:
        test_pipe, test_source = create_dali_val_pipeline(
            zarr_path=zarr_path,
            labels_path=labels_path,
            indices=test_indices,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            stats=stats,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        test_loader = DALIDataLoader(test_pipe, test_source)

    logger.info(
        f"DALI DataLoaders created: train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader) if test_loader else 0} batches"
    )

    return train_loader, val_loader, test_loader
