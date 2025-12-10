"""Experiment tracking and reproducibility.

This module provides:
1. Reproducibility info capture (git, environment, dataset)
2. Abstract ExperimentTracker protocol for backend-agnostic logging
3. NullTracker for when tracking is disabled

Future Migration to ClearML/ZenML:
----------------------------------
To add a new tracking backend:
1. Create tracking/backends/<name>.py implementing ExperimentTracker
2. Create a factory function in tracking/factory.py
3. Update CLI to accept --tracker flag

Example:
    from dapidl.tracking import get_reproducibility_info, ExperimentTracker

    info = get_reproducibility_info(dataset_path, cli_command)
    tracker.init(project="dapidl", config=config, reproducibility=info)
"""

from dapidl.tracking.reproducibility import (
    ReproducibilityInfo,
    GitInfo,
    EnvironmentInfo,
    DatasetInfo,
    get_reproducibility_info,
    get_git_info,
    get_environment_info,
    get_dataset_info,
    compute_dataset_hash,
    set_cli_command,
    get_cli_command,
)
from dapidl.tracking.base import ExperimentTracker, NullTracker

__all__ = [
    # Reproducibility
    "ReproducibilityInfo",
    "GitInfo",
    "EnvironmentInfo",
    "DatasetInfo",
    "get_reproducibility_info",
    "get_git_info",
    "get_environment_info",
    "get_dataset_info",
    "compute_dataset_hash",
    "set_cli_command",
    "get_cli_command",
    # Tracker protocol
    "ExperimentTracker",
    "NullTracker",
]
