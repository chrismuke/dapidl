"""Data augmentation transforms for DAPI patches."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(patch_size: int = 128) -> A.Compose:
    """Get training augmentation pipeline.

    Args:
        patch_size: Expected input patch size

    Returns:
        Albumentations Compose transform
    """
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


def get_val_transforms(patch_size: int = 128) -> A.Compose:
    """Get validation/test transform pipeline.

    Args:
        patch_size: Expected input patch size

    Returns:
        Albumentations Compose transform
    """
    return A.Compose(
        [
            # Only normalization for validation
            A.ToFloat(max_value=65535.0),
            A.Normalize(mean=0.5, std=0.25, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(patch_size: int = 128) -> A.Compose:
    """Get inference transform pipeline (same as validation).

    Args:
        patch_size: Expected input patch size

    Returns:
        Albumentations Compose transform
    """
    return get_val_transforms(patch_size)
