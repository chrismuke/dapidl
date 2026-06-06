"""Biologist-facing parameter schema for the DAPIDL pipeline studio.

Maps friendly UI selections to ``DAPIDLPipelineConfig.to_clearml_parameters()`` —
the single source of truth for the ClearML controller-task parameters. We load
``unified_config`` *by path* so importing this module does not drag in the heavy
``dapidl.pipeline.__init__`` chain (torch / cellpose / scanpy); the same trick the
ClearML controller script uses.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # scripts/studio -> repo
_UNIFIED_CONFIG_PATH = _REPO_ROOT / "src" / "dapidl" / "pipeline" / "unified_config.py"

_uc = None  # cached unified_config module


def _load_unified_config():
    """Import unified_config.py by path (cached), bypassing dapidl's heavy __init__."""
    global _uc
    if _uc is None:
        spec = importlib.util.spec_from_file_location(
            "dapidl_studio_unified_config", _UNIFIED_CONFIG_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Register before exec so pydantic can resolve the models' forward
        # references (unified_config uses `from __future__ import annotations`).
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _uc = module
    return _uc


# --- Biologist label -> backend value maps (also drive the UI dropdowns) ----------
DETECT_LEVELS: dict[str, bool] = {"Broad (4 classes)": False, "Fine (~12)": True}
LABEL_METHODS: dict[str, str] = {
    "popV ensemble": "ensemble",
    "CellTypist": "single",
    "Ground truth": "ground_truth",
}
SEGMENTATIONS: dict[str, str] = {
    "Vendor (native)": "native",
    "Cellpose": "cellpose",
    "StarDist": "stardist",
}
IMAGE_MODELS: dict[str, str] = {
    "EfficientNet-V2-S": "efficientnetv2_rw_s",
    "ConvNeXt-Tiny": "convnext_tiny",
    "ResNet50": "resnet50",
}
PATCH_SIZES: list[int] = [32, 64, 128, 256]


def build_param_overrides(
    selections: dict[str, Any],
    datasets: list[tuple],
    compute_targets: dict[str, str],
    services_queue: str = "services",
) -> dict[str, str]:
    """Map biologist selections + datasets to a ClearML controller-task param dict.

    ``selections`` keys: detect_level, label_method, segmentation, image_model,
    patch_size, epochs, compute_target, and optional advanced batch_size /
    learning_rate / patience. ``datasets`` is a list of (tissue, source, platform,
    tier); ``compute_targets`` maps the compute-target label to a ClearML queue.
    """
    uc = _load_unified_config()

    fine = DETECT_LEVELS.get(selections.get("detect_level", "Broad (4 classes)"), False)
    strategy = LABEL_METHODS.get(selections.get("label_method", "popV ensemble"), "ensemble")
    segmenter = SEGMENTATIONS.get(selections.get("segmentation", "Vendor (native)"), "native")
    backbone = IMAGE_MODELS.get(selections.get("image_model", "EfficientNet-V2-S"), "efficientnetv2_rw_s")
    patch_size = int(selections.get("patch_size", 128))
    gpu_queue = compute_targets.get(
        selections.get("compute_target", ""), next(iter(compute_targets.values()))
    )

    training_kwargs: dict[str, Any] = {
        "backbone": uc.BackboneType(backbone),
        "epochs": int(selections.get("epochs", 50)),
        "batch_size": int(selections.get("batch_size", 64)),
    }
    if "learning_rate" in selections:
        training_kwargs["learning_rate"] = float(selections["learning_rate"])
    if "patience" in selections:
        training_kwargs["patience"] = int(selections["patience"])

    config = uc.DAPIDLPipelineConfig(
        training=uc.TrainingConfig(**training_kwargs),
        annotation=uc.AnnotationConfig(
            strategy=uc.AnnotationStrategy(strategy), fine_grained=fine
        ),
        segmentation=uc.SegmentationConfig(segmenter=uc.SegmenterType(segmenter)),
        lmdb=uc.LMDBConfig(patch_sizes=[patch_size]),
        execution=uc.ExecutionConfig(
            execute_remotely=True, gpu_queue=gpu_queue, default_queue=services_queue
        ),
    )

    for tissue, source, platform, tier in datasets:
        src = pathlib.Path(str(source))
        if src.exists():
            config.input.add_tissue(
                tissue=tissue, local_path=str(src),
                platform=uc.Platform(platform), confidence_tier=int(tier),
            )
        else:
            config.input.add_tissue(
                tissue=tissue, dataset_id=str(source),
                platform=uc.Platform(platform), confidence_tier=int(tier),
            )

    return config.to_clearml_parameters()
