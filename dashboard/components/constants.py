"""Dashboard constants — recipes, queues, options, and display mappings.

All enum values and defaults mirror src/dapidl/pipeline/unified_config.py
so the dashboard emits parameter keys that from_clearml_parameters() can
parse directly without a translation layer.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pipeline recipes and step sequences
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# GPU targets — maps display label → (gpu_queue, default_queue)
# ---------------------------------------------------------------------------

GPU_TARGETS: dict[str, tuple[str, str]] = {
    "ubuntu3090 (RTX 3090)": ("gpu-local", "cpu-local"),
    "AWS Cloud (g6.xlarge, max 2)": ("gpu-cloud", "gpu-cloud"),
}

# ---------------------------------------------------------------------------
# Enum choices (matching unified_config.py Enums)
# ---------------------------------------------------------------------------

BACKBONES: list[str] = [
    "efficientnetv2_rw_s",
    "efficientnet_b0",
    "convnext_tiny",
    "resnet50",
    "resnet18",
]

BACKBONE_DESCRIPTIONS: dict[str, str] = {
    "efficientnetv2_rw_s": "EfficientNetV2-S — best accuracy (default)",
    "efficientnet_b0": "EfficientNet-B0 — lightweight",
    "convnext_tiny": "ConvNeXt-Tiny — modern CNN",
    "resnet50": "ResNet-50 — baseline",
    "resnet18": "ResNet-18 — fast baseline",
}

PATCH_SIZES: list[int] = [32, 64, 128, 256]

ANNOTATION_STRATEGIES: list[str] = ["ensemble", "single", "consensus", "ground_truth"]
ANNOTATION_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "ensemble": "Multiple methods with majority voting (recommended)",
    "single": "Single CellTypist model",
    "consensus": "Require agreement between methods",
    "ground_truth": "Use known ground truth labels",
}

CELLTYPIST_MODELS: list[str] = [
    "Cells_Adult_Breast.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Cells_Lung_Airway.pkl",
    "Cells_Intestinal_Tract.pkl",
    "Cells_Fetal_Lung.pkl",
    "Human_Lung_Atlas.pkl",
    "Pan_Fetal_Human.pkl",
    "Human_AdultAged_Gut.pkl",
    "Healthy_COVID19_PBMC.pkl",
    "Developing_Human_Brain.pkl",
]

SINGLER_REFERENCES: list[str] = ["blueprint", "hpca", "monaco"]
SINGLER_REFERENCE_DESCRIPTIONS: dict[str, str] = {
    "blueprint": "Blueprint Encode — broad cell types (recommended for stromal)",
    "hpca": "Human Primary Cell Atlas — immune-focused",
    "monaco": "Monaco Immune — detailed immune subtypes",
}

CL_TARGET_LEVELS: list[str] = ["broad", "coarse", "fine"]
CL_TARGET_LEVEL_DESCRIPTIONS: dict[str, str] = {
    "broad": "3-4 classes (Epithelial, Immune, Stromal, Other)",
    "coarse": "~10-15 classes (recommended)",
    "fine": "~30+ classes (detailed)",
}

NORMALIZATION_METHODS: list[str] = ["adaptive", "percentile", "minmax"]
NORMALIZATION_METHOD_DESCRIPTIONS: dict[str, str] = {
    "adaptive": "Adaptive percentile (recommended, cross-platform safe)",
    "percentile": "Fixed percentile normalization",
    "minmax": "Simple min-max scaling",
}

TRAINING_MODES: list[str] = ["flat", "hierarchical"]
TRAINING_MODE_DESCRIPTIONS: dict[str, str] = {
    "flat": "Standard flat classification",
    "hierarchical": "Curriculum learning: coarse → medium → fine (recommended)",
}

AUGMENTATION_LEVELS: list[str] = ["none", "standard", "heavy"]
AUGMENTATION_LEVEL_DESCRIPTIONS: dict[str, str] = {
    "none": "No augmentation",
    "standard": "Flip, rotate, brightness, noise (recommended)",
    "heavy": "Aggressive augmentation + elastic transforms",
}

SAMPLING_STRATEGIES: list[str] = ["equal", "proportional", "sqrt"]
SAMPLING_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "equal": "Equal samples per tissue",
    "proportional": "Proportional to tissue size",
    "sqrt": "Square-root balancing (recommended)",
}

SEGMENTERS: list[str] = ["cellpose", "native"]

# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

TIER_DESCRIPTIONS: dict[int, str] = {
    1: "Ground truth labels",
    2: "Consensus annotations",
    3: "Single-method predictions",
}

# ---------------------------------------------------------------------------
# Status display colors (Streamlit markdown syntax)
# ---------------------------------------------------------------------------

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

# Template task ID for cloning pipeline controllers via REST API
PIPELINE_TEMPLATE_TASK_ID = "26a8c58439514026b9b3d789da71c135"

# ---------------------------------------------------------------------------
# Admin settings
# ---------------------------------------------------------------------------

# ClearML user ID for the "Admin User" account (fixed users mode)
ADMIN_USER_ID = "98084d9e341946bbd6445bf2d44819d2"

# Default queue assignments per worker — used to restore queues after disabling
WORKER_DEFAULT_QUEUES: dict[str, list[str]] = {
    "clearml-agent-gpu-ubuntu3090": ["gpu-training", "gpu-local"],
    "clearml-agent-cpu-ubuntu3090": ["default", "cpu-local"],
    "clearml-agent-services": ["services"],
    "clearml-agent-cpu": ["default"],
}
