"""DALI native format conversion and loading.

This module provides conversion from Zarr to DALI-native formats (LMDB)
for maximum data loading performance with NVIDIA DALI.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr
from loguru import logger
from tqdm import tqdm

# LMDB is optional
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    lmdb = None

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


def is_lmdb_available() -> bool:
    """Check if LMDB is available."""
    return LMDB_AVAILABLE


def _read_zarr_batch(args: tuple) -> list[tuple[int, bytes]]:
    """Read a batch of patches from Zarr (for parallel processing).

    Args:
        args: Tuple of (zarr_path, labels, start_idx, end_idx)

    Returns:
        List of (idx, packed_data) tuples
    """
    zarr_path, labels, start_idx, end_idx = args
    patches = zarr.open(zarr_path, mode="r")
    results = []

    for idx in range(start_idx, end_idx):
        patch = np.array(patches[idx])
        label = labels[idx]
        key = struct.pack(">Q", idx)
        value = struct.pack(">q", label) + patch.tobytes()
        results.append((key, value))

    return results


def convert_zarr_to_lmdb(
    zarr_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    map_size_gb: float = 50.0,
    batch_size: int = 5000,
    num_workers: int = 8,
) -> dict:
    """Convert Zarr patches to LMDB format for DALI native reading.

    Uses parallel reading from Zarr for ~4-8x speedup.

    Args:
        zarr_path: Path to patches.zarr
        labels_path: Path to labels.npy
        output_path: Path for output LMDB database
        map_size_gb: Maximum database size in GB
        batch_size: Number of samples per batch (larger = more memory, faster)
        num_workers: Number of parallel workers for reading

    Returns:
        Metadata dict with n_samples, patch_shape, dtype
    """
    if not LMDB_AVAILABLE:
        raise RuntimeError("LMDB is not installed. Install with: pip install lmdb")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    zarr_path = Path(zarr_path)
    labels_path = Path(labels_path)
    output_path = Path(output_path)

    # Load source data
    patches = zarr.open(zarr_path, mode="r")
    labels = np.load(labels_path)

    n_samples = len(labels)
    patch_shape = patches[0].shape
    dtype = patches[0].dtype

    logger.info(
        f"Converting {n_samples} patches ({patch_shape}, {dtype}) to LMDB at {output_path}"
    )
    logger.info(f"Using {num_workers} workers, batch_size={batch_size}")

    # Create LMDB environment
    output_path.mkdir(parents=True, exist_ok=True)
    map_size = int(map_size_gb * 1024 * 1024 * 1024)

    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=True,
        meminit=False,
        map_async=True,
    )

    # Create batch ranges
    batch_ranges = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_ranges.append((str(zarr_path), labels, start_idx, end_idx))

    # Process in parallel with progress bar
    total_written = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_read_zarr_batch, args): args[2] for args in batch_ranges}

        with tqdm(total=n_samples, desc="Converting") as pbar:
            for future in as_completed(futures):
                results = future.result()

                # Write batch to LMDB
                with env.begin(write=True) as txn:
                    for key, value in results:
                        txn.put(key, value)

                total_written += len(results)
                pbar.update(len(results))

    env.sync()
    env.close()

    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "patch_shape": list(patch_shape),
        "dtype": str(dtype),
        "format": "lmdb",
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"LMDB conversion complete: {n_samples} samples, metadata at {metadata_path}")
    return metadata


def convert_dataset_to_lmdb(
    data_path: str | Path,
    output_path: str | Path | None = None,
    map_size_gb: float = 50.0,
    num_workers: int = 8,
) -> Path:
    """Convert a full dataset directory to LMDB format.

    Args:
        data_path: Path to dataset directory (containing patches.zarr, labels.npy)
        output_path: Output path for LMDB. If None, creates patches.lmdb in data_path
        map_size_gb: Maximum database size in GB
        num_workers: Number of parallel workers for reading Zarr data

    Returns:
        Path to created LMDB database
    """
    data_path = Path(data_path)
    zarr_path = data_path / "patches.zarr"
    labels_path = data_path / "labels.npy"

    if output_path is None:
        output_path = data_path / "patches.lmdb"
    else:
        output_path = Path(output_path)

    # Copy normalization stats if they exist
    stats_path = data_path / "normalization_stats.json"
    if stats_path.exists():
        import shutil
        output_stats = output_path / "normalization_stats.json"
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(stats_path, output_stats)
        logger.info(f"Copied normalization stats to {output_stats}")

    convert_zarr_to_lmdb(
        zarr_path=zarr_path,
        labels_path=labels_path,
        output_path=output_path,
        map_size_gb=map_size_gb,
        num_workers=num_workers,
    )

    return output_path


class LMDBExternalSource:
    """DALI external source for reading from LMDB.

    This provides much faster data loading than Zarr because LMDB
    can be read directly without Python multiprocessing overhead.

    Note: LMDB environment is opened lazily to support pickling for
    DALI's parallel external source workers.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        indices: np.ndarray | None = None,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 42,
        patch_shape: tuple[int, int] = (128, 128),
        dtype: np.dtype = np.uint16,
    ):
        """Initialize LMDB external source.

        Args:
            lmdb_path: Path to LMDB database
            indices: Optional subset indices for train/val/test splits
            batch_size: Batch size for iteration
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
            patch_shape: Shape of patches (H, W)
            dtype: Data type of patches
        """
        if not LMDB_AVAILABLE:
            raise RuntimeError("LMDB is not installed")

        # Store path as string for pickling
        self.lmdb_path_str = str(Path(lmdb_path))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.patch_shape = patch_shape
        self.dtype = dtype

        # Load metadata (file-based, picklable)
        metadata_path = Path(self.lmdb_path_str) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.n_total = metadata["n_samples"]
            self.patch_shape = tuple(metadata["patch_shape"])
            self.dtype = np.dtype(metadata["dtype"])
        else:
            raise RuntimeError(f"LMDB metadata not found at {metadata_path}")

        # LMDB environment is opened lazily (not picklable)
        self._env = None

        # Set up indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.n_total)

        self.n_samples = len(self.indices)
        self._epoch_indices = self.indices.copy()

        if self.shuffle:
            self.rng.shuffle(self._epoch_indices)

    @property
    def env(self):
        """Lazily open LMDB environment (supports multiprocessing)."""
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path_str,
                readonly=True,
                lock=False,
                meminit=False,
                map_async=True,
            )
        return self._env

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
        key = struct.pack(">Q", actual_idx)

        with self.env.begin(buffers=True) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {actual_idx} not found in LMDB")

            # Unpack label and patch
            label = struct.unpack(">q", value[:8])[0]
            patch_bytes = value[8:]
            patch = np.frombuffer(patch_bytes, dtype=self.dtype).reshape(self.patch_shape)

        # DALI expects HWC format, add channel dimension
        patch = patch[:, :, np.newaxis].copy()  # Copy to avoid memoryview issues
        label = np.array([label], dtype=np.int64)

        return patch, label

    def reset(self) -> None:
        """Reset for new epoch."""
        if self.shuffle:
            self.rng.shuffle(self._epoch_indices)

    def close(self) -> None:
        """Close LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self):
        """Support pickling by excluding LMDB environment."""
        state = self.__dict__.copy()
        state['_env'] = None  # Don't pickle the LMDB environment
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Environment will be reopened lazily


def create_dali_lmdb_train_pipeline(
    lmdb_path: str | Path,
    indices: np.ndarray,
    batch_size: int = 64,
    num_threads: int = 4,
    device_id: int = 0,
    seed: int = 42,
    stats: dict[str, float] | None = None,
    prefetch_queue_depth: int = 2,
) -> tuple["Pipeline", LMDBExternalSource]:
    """Create DALI training pipeline reading from LMDB.

    Args:
        lmdb_path: Path to LMDB database
        indices: Training indices
        batch_size: Batch size
        num_threads: Number of CPU threads for data loading
        device_id: GPU device ID
        seed: Random seed
        stats: Normalization statistics (p_low, p_high, mean, std)
        prefetch_queue_depth: Number of batches to prefetch

    Returns:
        Tuple of (DALI Pipeline, LMDBExternalSource)
    """
    if not DALI_AVAILABLE:
        raise RuntimeError("DALI is not available")

    # Default normalization stats
    if stats is None:
        stats = {"p_low": 0.0, "p_high": 65535.0, "mean": 0.5, "std": 0.25}

    p_low = stats["p_low"]
    p_high = stats["p_high"]
    mean = stats["mean"]
    std = stats["std"]

    # Create external source
    source = LMDBExternalSource(
        lmdb_path=lmdb_path,
        indices=indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        py_start_method="spawn",
    )
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
        images = fn.cast(images, dtype=types.FLOAT)
        images = dali_math.clamp(images, p_low, p_high)
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
        images = fn.transpose(images, perm=[2, 0, 1])

        return images, labels

    pipe = train_pipeline()
    pipe.build()
    logger.info(f"DALI LMDB training pipeline created: batch_size={batch_size}, device={device_id}")
    return pipe, source


def create_dali_lmdb_val_pipeline(
    lmdb_path: str | Path,
    indices: np.ndarray,
    batch_size: int = 64,
    num_threads: int = 4,
    device_id: int = 0,
    stats: dict[str, float] | None = None,
    prefetch_queue_depth: int = 2,
) -> tuple["Pipeline", LMDBExternalSource]:
    """Create DALI validation pipeline reading from LMDB (no augmentation).

    Args:
        lmdb_path: Path to LMDB database
        indices: Validation indices
        batch_size: Batch size
        num_threads: Number of CPU threads
        device_id: GPU device ID
        stats: Normalization statistics
        prefetch_queue_depth: Number of batches to prefetch

    Returns:
        Tuple of (DALI Pipeline, LMDBExternalSource)
    """
    if not DALI_AVAILABLE:
        raise RuntimeError("DALI is not available")

    # Default normalization stats
    if stats is None:
        stats = {"p_low": 0.0, "p_high": 65535.0, "mean": 0.5, "std": 0.25}

    p_low = stats["p_low"]
    p_high = stats["p_high"]
    mean = stats["mean"]
    std = stats["std"]

    # Create external source (no shuffle for validation)
    source = LMDBExternalSource(
        lmdb_path=lmdb_path,
        indices=indices,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        py_start_method="spawn",
    )
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
        images = fn.transpose(images, perm=[2, 0, 1])

        return images, labels

    pipe = val_pipeline()
    pipe.build()
    logger.info(f"DALI LMDB validation pipeline created: batch_size={batch_size}, device={device_id}")
    return pipe, source


class DALILMDBDataLoader:
    """PyTorch-compatible DataLoader wrapper for DALI LMDB pipelines."""

    def __init__(
        self,
        pipeline: "Pipeline",
        source: LMDBExternalSource,
        output_map: list[str] | None = None,
        auto_reset: bool = True,
    ):
        """Initialize DALI LMDB DataLoader.

        Args:
            pipeline: DALI pipeline
            source: LMDB external source for epoch reset
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
            labels = data[0]["labels"].squeeze(-1)
            return images, labels
        except StopIteration:
            self.source.reset()
            raise

    def reset(self) -> None:
        """Reset iterator for new epoch."""
        self.source.reset()
        self.iterator.reset()


def create_dali_lmdb_dataloaders(
    data_path: str | Path,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None = None,
    batch_size: int = 64,
    num_threads: int = 8,
    device_id: int = 0,
    seed: int = 42,
    prefetch_queue_depth: int = 2,
) -> tuple[DALILMDBDataLoader, DALILMDBDataLoader, DALILMDBDataLoader | None]:
    """Create DALI DataLoaders reading from LMDB format.

    Args:
        data_path: Path to dataset directory (containing patches.lmdb)
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
        raise RuntimeError("DALI is not available")
    if not LMDB_AVAILABLE:
        raise RuntimeError("LMDB is not available")

    data_path = Path(data_path)
    lmdb_path = data_path / "patches.lmdb"

    if not lmdb_path.exists():
        raise FileNotFoundError(
            f"LMDB database not found at {lmdb_path}. "
            "Run 'dapidl export-lmdb' to create it."
        )

    # Load normalization stats
    stats_path = data_path / "normalization_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        logger.info(f"Loaded normalization stats: {stats}")
    else:
        stats = None
        logger.warning("No normalization stats found, using defaults")

    # Create training pipeline and loader
    train_pipe, train_source = create_dali_lmdb_train_pipeline(
        lmdb_path=lmdb_path,
        indices=train_indices,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        stats=stats,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    train_loader = DALILMDBDataLoader(train_pipe, train_source)

    # Create validation pipeline and loader
    val_pipe, val_source = create_dali_lmdb_val_pipeline(
        lmdb_path=lmdb_path,
        indices=val_indices,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        stats=stats,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    val_loader = DALILMDBDataLoader(val_pipe, val_source)

    # Create test pipeline and loader if needed
    test_loader = None
    if test_indices is not None:
        test_pipe, test_source = create_dali_lmdb_val_pipeline(
            lmdb_path=lmdb_path,
            indices=test_indices,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            stats=stats,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        test_loader = DALILMDBDataLoader(test_pipe, test_source)

    logger.info(
        f"DALI LMDB DataLoaders created: train={len(train_loader)}, "
        f"val={len(val_loader)}, test={len(test_loader) if test_loader else 0} batches"
    )

    return train_loader, val_loader, test_loader
