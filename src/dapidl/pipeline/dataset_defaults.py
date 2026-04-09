"""Loader for per-dataset default settings from configs/dataset_defaults.yaml.

This module loads dataset-specific configuration (segmenter, annotation_models,
recipe, confidence_tier, etc.) and merges them with _defaults for unknown datasets.

Usage::

    from dapidl.pipeline.dataset_defaults import get_dataset_defaults

    # Returns merged settings dict for a known dataset
    settings = get_dataset_defaults("xenium-lung-2fov")

    # Returns _defaults for an unknown dataset
    settings = get_dataset_defaults("some-unknown-dataset")
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def _find_defaults_file() -> Path | None:
    """Locate dataset_defaults.yaml.

    Search order:
    1. ``DAPIDL_DATASET_DEFAULTS`` env var (explicit path)
    2. ``configs/dataset_defaults.yaml`` relative to repo root
    """
    env_path = os.environ.get("DAPIDL_DATASET_DEFAULTS")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        logger.warning(f"DAPIDL_DATASET_DEFAULTS={env_path} does not exist -- ignoring")

    # Repo root: src/dapidl/pipeline → 3 parents up
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "configs" / "dataset_defaults.yaml"
    if candidate.is_file():
        return candidate

    return None


@functools.lru_cache(maxsize=1)
def _load_all_defaults() -> dict[str, dict[str, Any]]:
    """Load and cache the full dataset_defaults.yaml.

    Returns a dict where keys are dataset names and values are their settings.
    The ``_defaults`` key is preserved for merge logic.
    """
    path = _find_defaults_file()
    if path is None:
        logger.debug("No dataset_defaults.yaml found -- per-dataset defaults disabled")
        return {}

    try:
        data = yaml.safe_load(path.read_text())
    except Exception as e:
        logger.warning(f"Failed to parse {path}: {e}")
        return {}

    if not isinstance(data, dict):
        logger.warning("dataset_defaults.yaml is not a dict -- ignoring")
        return {}

    logger.info(f"Loaded dataset defaults for {len(data) - 1} datasets from {path}")
    return data


def get_dataset_defaults(dataset_name: str) -> dict[str, Any]:
    """Get merged default settings for a dataset.

    Merges ``_defaults`` with per-dataset overrides. Dataset-specific values
    take precedence over ``_defaults``. Keys starting with ``_`` (like
    ``_notes``) are stripped from the result.

    Args:
        dataset_name: Dataset identifier (e.g. "xenium-lung-2fov",
            "merscope-breast", "sthelar-breast_s0").

    Returns:
        Merged settings dict. Empty dict if no defaults file is found.
    """
    all_defaults = _load_all_defaults()
    if not all_defaults:
        return {}

    base = dict(all_defaults.get("_defaults", {}))
    per_dataset = all_defaults.get(dataset_name, {})

    # Merge: per-dataset overrides _defaults
    merged = {**base, **per_dataset}

    # Strip internal keys (e.g. _notes, _defaults itself)
    merged = {k: v for k, v in merged.items() if not k.startswith("_")}

    return merged


def clear_cache() -> None:
    """Clear the cached defaults (useful for testing or config reload)."""
    _load_all_defaults.cache_clear()
