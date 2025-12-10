"""Data augmentation transforms for DAPI patches."""

import json
from pathlib import Path

import albumentations as A
import numpy as np
import zarr
from albumentations.pytorch import ToTensorV2
from loguru import logger


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

    # Return cached stats if they exist
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
            logger.info(f"Loaded cached normalization stats: {stats}")
            return stats

    # Compute stats from patches
    patches = zarr.open(data_path / "patches.zarr", mode="r")
    n_total = patches.shape[0]
    n_samples = min(n_samples, n_total)

    # Random sample indices
    rng = np.random.default_rng(42)
    indices = rng.choice(n_total, size=n_samples, replace=False)

    # Load samples
    samples = np.array([patches[int(i)] for i in indices])

    # Compute percentiles
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

    # Cache stats
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Computed normalization stats: {stats}")

    return stats


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


def get_train_transforms(
    patch_size: int = 128,
    stats: dict[str, float] | None = None,
) -> A.Compose:
    """Get training augmentation pipeline.

    Args:
        patch_size: Expected input patch size
        stats: Dataset normalization stats (p_low, p_high, mean, std).
               If None, uses legacy fixed normalization for backward compatibility.

    Returns:
        Albumentations Compose transform
    """
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
                    scale_limit=0.1,
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
                    scale_limit=0.1,
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
