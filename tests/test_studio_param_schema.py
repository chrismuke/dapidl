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


BASE = {
    "detect_level": "Broad (4 classes)", "label_method": "popV ensemble",
    "segmentation": "Vendor (native)", "image_model": "EfficientNet-V2-S",
    "patch_size": 128, "epochs": 50, "compute_target": "Local 3090", "batch_size": 64,
}


def test_expand_sweep_cartesian():
    runs = ps.expand_sweep(
        BASE,
        {"patch_size": [64, 128], "image_model": ["EfficientNet-V2-S", "ConvNeXt-Tiny"]},
        DATASETS, COMPUTE, sweep_id="42",
    )
    assert len(runs) == 4
    assert all(r["tag"] == "sweep-42" for r in runs)
    assert {r["params"]["lmdb/patch_sizes"] for r in runs} == {"64", "128"}
    assert {r["params"]["training/backbone"] for r in runs} == {"efficientnetv2_rw_s", "convnext_tiny"}
    assert len({r["name"] for r in runs}) == 4  # unique run names


def test_expand_sweep_no_axes_yields_single_run():
    runs = ps.expand_sweep(BASE, {}, DATASETS, COMPUTE, sweep_id="7")
    assert len(runs) == 1
    assert runs[0]["params"]["lmdb/patch_sizes"] == "128"


def test_validate_flags_missing_dataset_and_empty_axis():
    errs = ps.validate({}, [], {"patch_size": []})
    assert any("dataset" in e.lower() for e in errs)
    assert any("patch_size" in e for e in errs)


def test_validate_ok_returns_empty():
    assert ps.validate({}, DATASETS, None) == []
