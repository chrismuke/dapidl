"""Albumentations 2.0 augmentation pipeline for instance-segmentation tiles.

Designed for STHELAR breast DAPI tiles. Key constraints:
- Instance maps must use NEAREST interpolation (never LINEAR) to preserve IDs.
- No color jitter (DAPI is single-channel).
- No `Normalize` (per-image percentile normalization is done in the dataset).
- All transform names match Albumentations ≥2.0.8 API (RandomGamma,
  GaussianBlur, CoarseDropout — NOT GammaCorrection / GaussBlur / Cutout).
"""

import albumentations as A
import cv2
import numpy as np


def build_train_transform(tile_size: int = 1024) -> A.Compose:
    """Strong augmentation pipeline for training tiles."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=0.6
            ),
            A.RandomGamma(gamma_limit=(70, 150), p=0.4),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.5), p=0.2),
            A.ElasticTransform(
                alpha=30,
                sigma=4,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.2,
            ),
            A.Affine(
                scale=(0.85, 1.15),
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(32, 32),
                hole_width_range=(32, 32),
                fill=0,
                p=0.2,
            ),
        ],
        additional_targets={"instance_map": "mask"},
    )


def build_val_transform(tile_size: int = 1024) -> A.Compose:
    """No-op transform for validation/test."""
    return A.Compose([], additional_targets={"instance_map": "mask"})


def apply_transform(
    image: np.ndarray, instance_map: np.ndarray, transform: A.Compose
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an Albumentations Compose with an instance_map auxiliary target.

    Args:
        image: (H, W) float32, already normalized to roughly zero-mean.
        instance_map: (H, W) integer instance IDs.
        transform: result of `build_train_transform` / `build_val_transform`.

    Returns:
        (image, instance_map) — both transformed; instance_map is recovered
        as integer dtype after Albumentations passes.
    """
    out = transform(image=image, instance_map=instance_map)
    image_out = out["image"]
    map_out = out["instance_map"]
    if map_out.dtype != instance_map.dtype:
        map_out = map_out.astype(instance_map.dtype)
    return image_out, map_out
