"""DAPIDL Hyperparameter Optimization module.

This module provides ClearML-based hyperparameter optimization for DAPIDL training,
including search space definitions, visualization, and result aggregation.
"""

from .optimizer import create_hpo_optimizer, run_hpo
from .search_space import (
    BACKBONE_CONFIGS,
    SEARCH_SPACE,
    get_clearml_hyper_parameters,
    resolve_dataset_path,
)
from .visualization import (
    create_class_prediction_grid,
    generate_sample_predictions,
    log_confusion_matrix,
)

__all__ = [
    "SEARCH_SPACE",
    "BACKBONE_CONFIGS",
    "resolve_dataset_path",
    "get_clearml_hyper_parameters",
    "create_hpo_optimizer",
    "run_hpo",
    "generate_sample_predictions",
    "create_class_prediction_grid",
    "log_confusion_matrix",
]
