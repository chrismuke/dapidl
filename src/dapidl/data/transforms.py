"""Data augmentation transforms for DAPI patches."""

import json
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import zarr
from albumentations.pytorch import ToTensorV2
from loguru import logger


class InvalidNormalizationStatsError(ValueError):
    """Raised when normalization statistics are invalid."""
    pass


def validate_normalization_stats(
    stats: dict[str, float],
    auto_recompute: bool = False,
    samples: Optional[np.ndarray] = None,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> dict[str, float]:
    """Validate normalization statistics and optionally recompute if invalid.

    This prevents NaN losses during training due to invalid cached stats
    (e.g., p_low=p_high=0 from corrupt cache files).

    Args:
        stats: Dictionary with p_low, p_high, mean, std values
        auto_recompute: If True and stats are invalid, recompute from samples
        samples: Sample patches for recomputation (required if auto_recompute=True)
        percentile_low: Lower percentile for recomputation (default 1%)
        percentile_high: Upper percentile for recomputation (default 99%)

    Returns:
        Validated (and possibly recomputed) stats dictionary

    Raises:
        InvalidNormalizationStatsError: If stats are invalid and cannot be recomputed
    """
    p_low = stats.get("p_low", 0)
    p_high = stats.get("p_high", 0)
    mean = stats.get("mean", 0)
    std = stats.get("std", 0)

    issues = []

    # Check for NaN values
    if np.isnan(p_low) or np.isnan(p_high):
        issues.append(f"NaN values in percentiles: p_low={p_low}, p_high={p_high}")

    if np.isnan(mean) or np.isnan(std):
        issues.append(f"NaN values in mean/std: mean={mean}, std={std}")

    # Check dynamic range (critical - causes division by zero)
    if p_high <= p_low:
        issues.append(f"No dynamic range: p_high ({p_high}) must be > p_low ({p_low})")

    # Check for suspicious values
    if p_low < 0:
        issues.append(f"Negative p_low: {p_low}")

    if std <= 0:
        issues.append(f"Invalid std: {std} (must be positive)")

    # If no issues, return stats as-is
    if not issues:
        return stats

    # Handle invalid stats
    issues_str = "; ".join(issues)

    if auto_recompute and samples is not None:
        logger.warning(
            f"Invalid normalization stats ({issues_str}). Recomputing from {len(samples)} samples..."
        )

        # Recompute stats
        p_low_new = float(np.percentile(samples, percentile_low))
        p_high_new = float(np.percentile(samples, percentile_high))

        # Final validation of recomputed values
        if p_high_new <= p_low_new:
            raise InvalidNormalizationStatsError(
                f"Recomputed stats still invalid: p_low={p_low_new}, p_high={p_high_new}. "
                "Check if image data is constant or corrupt."
            )

        # Compute mean/std after percentile normalization
        clipped = np.clip(samples, p_low_new, p_high_new)
        scaled = (clipped - p_low_new) / (p_high_new - p_low_new + 1e-8)
        mean_new = float(np.mean(scaled))
        std_new = float(np.std(scaled))

        new_stats = {
            "p_low": p_low_new,
            "p_high": p_high_new,
            "mean": mean_new,
            "std": std_new,
        }

        logger.info(f"Recomputed normalization stats: {new_stats}")
        return new_stats

    # Cannot auto-recompute, raise error with clear guidance
    raise InvalidNormalizationStatsError(
        f"Invalid normalization stats: {issues_str}. "
        "Delete the cached normalization_stats.json and rerun to recompute, "
        "or check if the source image data is corrupt."
    )


def compute_dataset_stats(
    data_path: str | Path,
    n_samples: int = 1000,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> dict[str, float]:
    """Compute normalization statistics from dataset patches.

    Uses percentile-based normalization to handle different intensity ranges
    across platforms (Xenium vs MERSCOPE).

    Args:
        data_path: Path to prepared dataset directory
        n_samples: Number of random patches to sample for statistics
        percentile_low: Lower percentile for clipping (default 1%)
        percentile_high: Upper percentile for clipping (default 99%)

    Returns:
        Dictionary with p_low, p_high, mean, std values
    """
    data_path = Path(data_path)
    stats_file = data_path / "normalization_stats.json"

    # Return cached stats if they exist and are valid
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
            logger.info(f"Loaded cached normalization stats: {stats}")

            # Validate cached stats - this prevents NaN losses from corrupt caches
            try:
                validated = validate_normalization_stats(stats)
                return validated
            except InvalidNormalizationStatsError as e:
                logger.warning(
                    f"Cached stats are invalid: {e}. "
                    f"Deleting {stats_file} and recomputing..."
                )
                stats_file.unlink()
                # Fall through to recompute

    # Check for LMDB or Zarr format
    lmdb_path = data_path / "patches.lmdb"
    zarr_path = data_path / "patches.zarr"

    if lmdb_path.exists():
        # Compute from LMDB
        import lmdb as lmdb_lib

        env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            # Get metadata for patch size
            metadata_bytes = txn.get(b"__metadata__")
            if metadata_bytes:
                metadata = json.loads(metadata_bytes.decode())
                patch_size = metadata.get("patch_size", 128)
            else:
                patch_size = 128

            # Sample patches
            cursor = txn.cursor()
            samples = []
            expected_patch_bytes = patch_size * patch_size * 2  # uint16

            for key, value in cursor:
                if key == b"__metadata__":
                    continue
                # Skip label keys from HQ format (label_00000000)
                if key.startswith(b"label_"):
                    continue

                # Detect format based on value length:
                # - HQ format: pure uint16 patch (no label prefix)
                # - Standard format: 8-byte int64 label + uint16 patch
                if len(value) == expected_patch_bytes:
                    patch_bytes = value  # HQ format
                elif len(value) == expected_patch_bytes + 8:
                    patch_bytes = value[8:]  # Standard format
                else:
                    continue  # Unknown format, skip

                patch = np.frombuffer(patch_bytes, dtype=np.uint16)
                patch = patch.reshape(patch_size, patch_size)
                # Convert to float for stats computation
                samples.append(patch.astype(np.float32))
                if len(samples) >= n_samples:
                    break
        env.close()
        samples = np.array(samples)
    elif zarr_path.exists():
        patches = zarr.open(zarr_path, mode="r")
        n_total = patches.shape[0]
        n_sample = min(n_samples, n_total)

        # Random sample indices
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, size=n_sample, replace=False)

        # Load samples
        samples = np.array([patches[int(i)] for i in indices])
    else:
        raise FileNotFoundError(
            f"No patches.lmdb or patches.zarr found in {data_path}"
        )

    # Compute percentiles from samples
    p_low = float(np.percentile(samples, percentile_low))
    p_high = float(np.percentile(samples, percentile_high))

    # Compute mean/std after percentile normalization
    # Clip and scale to [0, 1]
    clipped = np.clip(samples, p_low, p_high)
    scaled = (clipped - p_low) / (p_high - p_low + 1e-8)
    mean = float(np.mean(scaled))
    std = float(np.std(scaled))

    stats = {
        "p_low": p_low,
        "p_high": p_high,
        "mean": mean,
        "std": std,
    }

    # Validate computed stats before caching (catch corrupt source data early)
    try:
        validated = validate_normalization_stats(stats)
    except InvalidNormalizationStatsError as e:
        raise InvalidNormalizationStatsError(
            f"Computed stats are invalid: {e}. "
            "This usually indicates corrupt or constant-value image data. "
            f"Samples shape: {samples.shape}, "
            f"min={samples.min()}, max={samples.max()}, dtype={samples.dtype}"
        ) from e

    # Cache validated stats
    with open(stats_file, "w") as f:
        json.dump(validated, f, indent=2)
    logger.info(f"Computed normalization stats: {validated}")

    return validated


class PercentileNormalize(A.ImageOnlyTransform):
    """Normalize image using percentile-based clipping.

    This handles different intensity ranges across platforms by:
    1. Clipping values to [p_low, p_high] percentile range
    2. Scaling to [0, 1]
    3. Applying standard normalization with computed mean/std
    """

    def __init__(
        self,
        p_low: float,
        p_high: float,
        mean: float = 0.5,
        std: float = 0.25,
        always_apply: bool = True,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.p_low = p_low
        self.p_high = p_high
        self.mean = mean
        self.std = std

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Clip to percentile range
        img = np.clip(img, self.p_low, self.p_high)
        # Scale to [0, 1]
        img = (img - self.p_low) / (self.p_high - self.p_low + 1e-8)
        # Normalize
        img = (img - self.mean) / self.std
        return img.astype(np.float32)

    def get_transform_init_args_names(self):
        return ("p_low", "p_high", "mean", "std")


class ReinhardNormalize(A.ImageOnlyTransform):
    """Reinhard color/intensity normalization for domain transfer.

    Transforms target domain images to match source (reference) domain statistics.
    Originally designed for histopathology stain normalization, this technique
    works well for aligning DAPI intensity distributions across platforms.

    The transformation:
        1. Clip target to percentile range and scale to [0, 1]
        2. Apply Reinhard formula: (x - target_mean) / target_std * source_std + source_mean
        3. Apply final normalization for model input

    This allows a model trained on source domain (e.g., Xenium) to work on
    target domain (e.g., MERSCOPE) by aligning intensity distributions.

    Reference:
        Reinhard et al. "Color Transfer between Images" (2001)
    """

    def __init__(
        self,
        source_stats: dict[str, float],
        target_stats: dict[str, float],
        final_mean: float = 0.5,
        final_std: float = 0.25,
        always_apply: bool = True,
        p: float = 1.0,
    ):
        """Initialize Reinhard normalization.

        Args:
            source_stats: Reference domain statistics (p_low, p_high, mean, std)
            target_stats: Target domain statistics (p_low, p_high, mean, std)
            final_mean: Mean for final normalization (model input)
            final_std: Std for final normalization (model input)
        """
        super().__init__(always_apply=always_apply, p=p)
        self.source_stats = source_stats
        self.target_stats = target_stats
        self.final_mean = final_mean
        self.final_std = final_std

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Step 1: Clip and scale target image to [0, 1] using target stats
        img = np.clip(img, self.target_stats["p_low"], self.target_stats["p_high"])
        img = (img - self.target_stats["p_low"]) / (
            self.target_stats["p_high"] - self.target_stats["p_low"] + 1e-8
        )

        # Step 2: Reinhard transfer - align target distribution to source
        # (x - target_mean) / target_std * source_std + source_mean
        img = (img - self.target_stats["mean"]) / (self.target_stats["std"] + 1e-8)
        img = img * self.source_stats["std"] + self.source_stats["mean"]

        # Step 3: Clip to valid range [0, 1] after transfer
        img = np.clip(img, 0, 1)

        # Step 4: Final normalization for model input
        img = (img - self.final_mean) / self.final_std

        return img.astype(np.float32)

    def get_transform_init_args_names(self):
        return ("source_stats", "target_stats", "final_mean", "final_std")


def get_train_transforms(
    patch_size: int = 128,
    stats: dict[str, float] | None = None,
    cross_platform: bool = False,
) -> A.Compose:
    """Get training augmentation pipeline.

    Args:
        patch_size: Expected input patch size
        stats: Dataset normalization stats (p_low, p_high, mean, std).
               If None, uses legacy fixed normalization for backward compatibility.
        cross_platform: If True, use aggressive scale augmentation (±50%) for
                       cross-platform transfer. Xenium (0.2125 µm/px) and MERSCOPE
                       (0.108 µm/px) have 2x resolution difference, so a nucleus
                       appears 2x larger on MERSCOPE. This augmentation helps the
                       model become scale-invariant.

    Returns:
        Albumentations Compose transform
    """
    # Scale limit: 0.1 for single-platform, 0.5 for cross-platform (2x range)
    scale_limit = 0.5 if cross_platform else 0.1

    # Build normalization based on whether stats are provided
    if stats is not None:
        # Adaptive normalization using dataset-specific percentiles
        normalize = PercentileNormalize(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            mean=stats["mean"],
            std=stats["std"],
        )
        # For augmentations, we need to work on raw uint16 values
        # then normalize at the end
        return A.Compose(
            [
                # Geometric transforms (work on uint16)
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=scale_limit,  # 0.1 single-platform, 0.5 cross-platform
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                # Elastic deformation (important for nuclear morphology)
                A.ElasticTransform(
                    alpha=50,
                    sigma=10,
                    border_mode=0,
                    p=0.3,
                ),
                # Adaptive normalization (clips, scales, normalizes)
                normalize,
                # Intensity transforms (now on normalized float images)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                # Convert to tensor
                ToTensorV2(),
            ]
        )
    else:
        # Legacy fixed normalization (for backward compatibility)
        return A.Compose(
            [
                # Convert to float first (DAPI images are uint16)
                A.ToFloat(max_value=65535.0),
                # Geometric transforms
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=scale_limit,  # 0.1 single-platform, 0.5 cross-platform
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                # Elastic deformation (important for nuclear morphology)
                A.ElasticTransform(
                    alpha=50,
                    sigma=10,
                    border_mode=0,
                    p=0.3,
                ),
                # Intensity transforms (now on float images)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),  # Variance for [0,1] range
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                # Normalize
                A.Normalize(mean=0.5, std=0.25, max_pixel_value=1.0),
                # Convert to tensor
                ToTensorV2(),
            ]
        )


def get_val_transforms(
    patch_size: int = 128,
    stats: dict[str, float] | None = None,
) -> A.Compose:
    """Get validation/test transform pipeline.

    Args:
        patch_size: Expected input patch size
        stats: Dataset normalization stats (p_low, p_high, mean, std).
               If None, uses legacy fixed normalization for backward compatibility.

    Returns:
        Albumentations Compose transform
    """
    if stats is not None:
        # Adaptive normalization using dataset-specific percentiles
        normalize = PercentileNormalize(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            mean=stats["mean"],
            std=stats["std"],
        )
        return A.Compose(
            [
                normalize,
                ToTensorV2(),
            ]
        )
    else:
        # Legacy fixed normalization
        return A.Compose(
            [
                # Only normalization for validation
                A.ToFloat(max_value=65535.0),
                A.Normalize(mean=0.5, std=0.25, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )


def get_inference_transforms(
    patch_size: int = 128,
    stats: dict[str, float] | None = None,
) -> A.Compose:
    """Get inference transform pipeline (same as validation).

    Args:
        patch_size: Expected input patch size
        stats: Dataset normalization stats (p_low, p_high, mean, std).

    Returns:
        Albumentations Compose transform
    """
    return get_val_transforms(patch_size, stats=stats)


def get_reinhard_inference_transforms(
    source_stats: dict[str, float],
    target_stats: dict[str, float],
    patch_size: int = 128,
) -> A.Compose:
    """Get inference transforms with Reinhard normalization for domain transfer.

    Use this when applying a model trained on source domain (e.g., Xenium)
    to target domain data (e.g., MERSCOPE). Reinhard normalization aligns
    the target intensity distribution to match the source, allowing the
    model to work on cross-platform data without retraining.

    Args:
        source_stats: Reference domain statistics (from training data)
        target_stats: Target domain statistics (from inference data)
        patch_size: Expected input patch size

    Returns:
        Albumentations Compose transform with Reinhard normalization

    Example:
        >>> # Load stats from each dataset
        >>> xenium_stats = compute_dataset_stats("/path/to/xenium")
        >>> merscope_stats = compute_dataset_stats("/path/to/merscope")
        >>> # Create transform that aligns MERSCOPE to Xenium distribution
        >>> transform = get_reinhard_inference_transforms(xenium_stats, merscope_stats)
        >>> # Apply to MERSCOPE patches before inference with Xenium-trained model
        >>> normalized = transform(image=patch)["image"]
    """
    reinhard = ReinhardNormalize(
        source_stats=source_stats,
        target_stats=target_stats,
        final_mean=source_stats["mean"],
        final_std=source_stats["std"],
    )
    return A.Compose(
        [
            reinhard,
            ToTensorV2(),
        ]
    )


def get_heavy_augmentation_transforms(
    patch_size: int = 128,
    stats: dict[str, float] | None = None,
) -> A.Compose:
    """Get heavy augmentation pipeline for rare classes.

    This applies more aggressive augmentations to increase variability
    for underrepresented classes, helping the model generalize better.

    Args:
        patch_size: Expected input patch size
        stats: Dataset normalization stats (p_low, p_high, mean, std).

    Returns:
        Albumentations Compose transform
    """
    if stats is not None:
        normalize = PercentileNormalize(
            p_low=stats["p_low"],
            p_high=stats["p_high"],
            mean=stats["mean"],
            std=stats["std"],
        )
        return A.Compose(
            [
                # More aggressive geometric transforms
                A.RandomRotate90(p=1.0),  # Always rotate
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,  # Increased from 0.1
                    scale_limit=0.2,   # Increased from 0.1
                    rotate_limit=30,   # Increased from 15
                    border_mode=0,
                    p=0.8,  # Increased from 0.5
                ),
                # More aggressive elastic deformation
                A.ElasticTransform(
                    alpha=80,   # Increased from 50
                    sigma=12,   # Increased from 10
                    border_mode=0,
                    p=0.5,  # Increased from 0.3
                ),
                # Grid distortion for additional deformation
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    p=0.3,
                ),
                # Adaptive normalization
                normalize,
                # More aggressive intensity transforms
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,  # Increased from 0.2
                    contrast_limit=0.3,    # Increased from 0.2
                    p=0.7,  # Increased from 0.5
                ),
                A.GaussNoise(var_limit=(0.005, 0.02), p=0.5),  # Increased
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # Increased blur
                # Additional augmentations for rare classes
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.2),  # Contrast enhancement
                # Convert to tensor
                ToTensorV2(),
            ]
        )
    else:
        # Legacy version without stats
        return A.Compose(
            [
                A.ToFloat(max_value=65535.0),
                A.RandomRotate90(p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=0,
                    p=0.8,
                ),
                A.ElasticTransform(
                    alpha=80,
                    sigma=12,
                    border_mode=0,
                    p=0.5,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    p=0.3,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7,
                ),
                A.GaussNoise(var_limit=(0.005, 0.02), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.Normalize(mean=0.5, std=0.25, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
