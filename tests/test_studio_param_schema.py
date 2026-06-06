"""Tests for the biologist studio param schema (pure logic, no Streamlit/network)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from studio import param_schema as ps  # noqa: E402

DATASETS = [("breast", "abc123", "xenium", 2)]
COMPUTE = {"Local 3090": "gpu-local", "AWS (scales)": "gpu-cloud"}


def test_build_param_overrides_maps_biologist_selections():
    sel = {
        "detect_level": "Fine (~12)",
        "label_method": "popV ensemble",
        "segmentation": "Vendor (native)",
        "image_model": "EfficientNet-V2-S",
        "patch_size": 128,
        "epochs": 50,
        "compute_target": "AWS (scales)",
        "batch_size": 64,
    }
    out = ps.build_param_overrides(sel, DATASETS, COMPUTE, services_queue="services")
    assert out["training/backbone"] == "efficientnetv2_rw_s"
    assert out["annotation/fine_grained"] == "True"
    assert out["training/epochs"] == "50"
    assert out["lmdb/patch_sizes"] == "128"
    assert out["segmentation/segmenter"] == "native"
    assert out["execution/gpu_queue"] == "gpu-cloud"
    assert out["execution/default_queue"] == "services"


def test_build_param_overrides_broad_sets_fine_grained_false():
    sel = {
        "detect_level": "Broad (4 classes)",
        "label_method": "CellTypist",
        "segmentation": "Cellpose",
        "image_model": "ConvNeXt-Tiny",
        "patch_size": 64,
        "epochs": 25,
        "compute_target": "Local 3090",
        "batch_size": 64,
    }
    out = ps.build_param_overrides(sel, DATASETS, COMPUTE, services_queue="services")
    assert out["annotation/fine_grained"] == "False"
    assert out["training/backbone"] == "convnext_tiny"
    assert out["segmentation/segmenter"] == "cellpose"
    assert out["execution/gpu_queue"] == "gpu-local"
