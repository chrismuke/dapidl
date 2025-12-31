"""Hyperparameter search space definitions for DAPIDL.

Defines the search space for ClearML HPO with Optuna backend.
"""

from pathlib import Path
from typing import Any

# Backbone configurations with GPU memory requirements
BACKBONE_CONFIGS = {
    "efficientnetv2_rw_s": {"max_batch_size": 128, "params": "20M", "pretrained": True},
    "resnet18": {"max_batch_size": 512, "params": "11M", "pretrained": True},
    "resnet50": {"max_batch_size": 256, "params": "25M", "pretrained": True},
    "convnext_tiny": {"max_batch_size": 128, "params": "28M", "pretrained": True},
    "microscopy_cnn": {"max_batch_size": 512, "params": "1M", "pretrained": False},
}

# Full search space
SEARCH_SPACE = {
    # Model architecture
    "backbone": list(BACKBONE_CONFIGS.keys()),

    # Dataset configuration
    "patch_size": [32, 64, 128, 256],
    "centering": ["xenium", "cellpose"],
    "granularity": ["coarse", "finegrained"],

    # Training hyperparameters
    "batch_size": [32, 64, 128, 256],
    "learning_rate": {"min": 1e-5, "max": 1e-3, "log": True},
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "max_weight_ratio": [5.0, 10.0, 15.0, 20.0],

    # Fixed parameters (not tuned)
    "epochs": 50,
    "warmup_epochs": 5,
    "early_stopping_patience": 15,
    "label_smoothing": 0.1,
    "weight_decay": 1e-5,
}


def resolve_dataset_path(
    centering: str,
    granularity: str,
    patch_size: int,
    base_path: Path | str | None = None,
) -> Path:
    """Resolve dataset path from hyperparameters.

    Args:
        centering: 'xenium' or 'cellpose'
        granularity: 'coarse' or 'finegrained'
        patch_size: Patch size in pixels (32, 64, 128, 256)
        base_path: Base datasets directory (defaults to ~/datasets/derived)

    Returns:
        Path to the LMDB dataset directory
    """
    if base_path is None:
        base_path = Path.home() / "datasets" / "derived"
    else:
        base_path = Path(base_path)

    dataset_name = f"xenium-breast-{centering}-{granularity}-p{patch_size}"
    dataset_path = base_path / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def get_valid_batch_size(backbone: str, requested_batch_size: int) -> int:
    """Get valid batch size considering GPU memory constraints.

    Args:
        backbone: Model backbone name
        requested_batch_size: Requested batch size

    Returns:
        Valid batch size (capped by backbone's max)
    """
    max_batch = BACKBONE_CONFIGS.get(backbone, {}).get("max_batch_size", 128)
    return min(requested_batch_size, max_batch)


def get_clearml_hyper_parameters() -> list[dict[str, Any]]:
    """Get ClearML HyperParameterOptimizer parameter definitions.

    Returns:
        List of parameter dictionaries for ClearML HPO
    """
    from clearml.automation import (
        UniformParameterRange,
        DiscreteParameterRange,
    )

    return [
        # Categorical: backbone architecture
        DiscreteParameterRange(
            "General/backbone",
            values=SEARCH_SPACE["backbone"]
        ),

        # Discrete: patch size
        DiscreteParameterRange(
            "General/patch_size",
            values=SEARCH_SPACE["patch_size"]
        ),

        # Discrete: batch size
        DiscreteParameterRange(
            "General/batch_size",
            values=SEARCH_SPACE["batch_size"]
        ),

        # Discrete: dropout
        DiscreteParameterRange(
            "General/dropout",
            values=SEARCH_SPACE["dropout"]
        ),

        # Discrete: max weight ratio
        DiscreteParameterRange(
            "General/max_weight_ratio",
            values=SEARCH_SPACE["max_weight_ratio"]
        ),

        # Continuous: learning rate (log scale)
        UniformParameterRange(
            "General/learning_rate",
            min_value=SEARCH_SPACE["learning_rate"]["min"],
            max_value=SEARCH_SPACE["learning_rate"]["max"],
            step_size=1e-6,
        ),

        # Dataset selection
        DiscreteParameterRange(
            "General/centering",
            values=SEARCH_SPACE["centering"]
        ),
        DiscreteParameterRange(
            "General/granularity",
            values=SEARCH_SPACE["granularity"]
        ),
    ]


def get_num_classes(granularity: str) -> int:
    """Get number of classes for a granularity level.

    Args:
        granularity: 'coarse' or 'finegrained'

    Returns:
        Number of classes
    """
    return 3 if granularity == "coarse" else 17
