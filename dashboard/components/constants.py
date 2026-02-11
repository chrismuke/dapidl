"""Dashboard constants — recipes, queues, options, and display mappings."""

from __future__ import annotations

# Pipeline recipes and their step sequences (mirrored from src/dapidl/pipeline/orchestrator.py)
BUILTIN_RECIPES: dict[str, list[str]] = {
    "default": ["data_loader", "ensemble_annotation", "cl_standardization", "lmdb_creation"],
    "gt": ["data_loader", "gt_annotation", "lmdb_creation"],
    "no_cl": ["data_loader", "ensemble_annotation", "lmdb_creation"],
    "annotate_only": ["data_loader", "ensemble_annotation", "cl_standardization"],
}

RECIPE_DESCRIPTIONS: dict[str, str] = {
    "default": "Full pipeline: annotation + CL standardization + LMDB",
    "gt": "Ground truth labels + LMDB creation",
    "no_cl": "Annotation + LMDB (skip CL standardization)",
    "annotate_only": "Annotation + standardization only (no LMDB)",
}

QUEUES = ["gpu-local", "gpu-cloud", "gpu-training", "cpu-local", "default"]
BACKBONES = ["efficientnetv2_rw_s", "convnext_tiny", "resnet50", "densenet121"]
PATCH_SIZES = ["32", "64", "128", "256"]
SAMPLING_STRATEGIES = ["sqrt", "equal", "proportional"]
ANNOTATORS = ["celltypist", "ground_truth", "popv"]
SEGMENTERS = ["native", "cellpose"]

# Status display colors (Streamlit markdown syntax)
STATUS_COLORS: dict[str, str] = {
    "completed": ":green[completed]",
    "in_progress": ":orange[in_progress]",
    "failed": ":red[failed]",
    "stopped": ":red[stopped]",
    "created": ":blue[created]",
    "queued": ":blue[queued]",
    "published": ":violet[published]",
    "closed": ":gray[closed]",
}

# Worker health thresholds (seconds since last activity)
WORKER_HEALTH_OK = 120       # < 2 min = healthy
WORKER_HEALTH_WARN = 600     # < 10 min = warning, else = error

# Confidence tier descriptions
TIER_DESCRIPTIONS: dict[int, str] = {
    1: "Ground truth labels",
    2: "Consensus annotations",
    3: "Single-method predictions",
}
