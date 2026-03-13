"""Tests for declarative annotation registry refactor."""

from dapidl.pipeline.base import AnnotationConfig


def test_annotation_config_has_singler_reference():
    """AnnotationConfig must have singler_reference for SingleR param passing."""
    cfg = AnnotationConfig()
    assert hasattr(cfg, "singler_reference")
    assert cfg.singler_reference == "blueprint"


def test_annotation_config_singler_reference_custom():
    cfg = AnnotationConfig(singler_reference="hpca")
    assert cfg.singler_reference == "hpca"


import json


def test_method_spec_creation():
    from dapidl.pipeline.steps.ensemble_annotation import MethodSpec

    spec = MethodSpec("celltypist", {"model": "Cells_Adult_Breast.pkl"})
    assert spec.name == "celltypist"
    assert spec.params == {"model": "Cells_Adult_Breast.pkl"}


def test_method_spec_roundtrip():
    from dapidl.pipeline.steps.ensemble_annotation import MethodSpec

    spec = MethodSpec("singler", {"reference": "blueprint"})
    d = spec.to_dict()
    assert d == {"name": "singler", "params": {"reference": "blueprint"}}
    restored = MethodSpec.from_dict(d)
    assert restored.name == "singler"
    assert restored.params == {"reference": "blueprint"}


def test_method_spec_empty_params():
    from dapidl.pipeline.steps.ensemble_annotation import MethodSpec

    spec = MethodSpec("sctype")
    assert spec.params == {}
    d = spec.to_dict()
    restored = MethodSpec.from_dict(d)
    assert restored.params == {}


def test_ensemble_config_creation():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        MethodSpec,
    )

    cfg = EnsembleAnnotationConfig(methods=[
        MethodSpec("celltypist", {"model": "Cells_Adult_Breast.pkl"}),
        MethodSpec("singler", {"reference": "blueprint"}),
    ])
    assert len(cfg.methods) == 2
    assert cfg.min_agreement == 2
    assert cfg.confidence_threshold == 0.5


def test_ensemble_config_roundtrip():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        MethodSpec,
    )

    cfg = EnsembleAnnotationConfig(
        methods=[
            MethodSpec("celltypist", {"model": "Immune_All_High.pkl"}),
            MethodSpec("singler", {"reference": "hpca"}),
        ],
        min_agreement=3,
        fine_grained=False,
    )
    d = cfg.to_dict()
    restored = EnsembleAnnotationConfig.from_dict(d)
    assert len(restored.methods) == 2
    assert restored.methods[0].name == "celltypist"
    assert restored.methods[1].params == {"reference": "hpca"}
    assert restored.min_agreement == 3
    assert restored.fine_grained is False


def test_ensemble_config_from_clearml_format():
    """ClearML stores params with General/ prefix and methods as JSON string."""
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        MethodSpec,
    )

    clearml_params = {
        "General/methods": json.dumps([
            {"name": "celltypist", "params": {"model": "Breast.pkl"}},
        ]),
        "General/min_agreement": "2",
        "General/confidence_threshold": "0.7",
        "General/fine_grained": "True",
        "General/use_confidence_weighting": "False",
    }
    cfg = EnsembleAnnotationConfig.from_dict(clearml_params)
    assert len(cfg.methods) == 1
    assert cfg.methods[0].name == "celltypist"
    assert cfg.min_agreement == 2
    assert cfg.confidence_threshold == 0.7
    assert cfg.use_confidence_weighting is False  # "False" string parsed correctly


def test_ensemble_config_from_dict_bool_parsing():
    """ClearML sends booleans as strings — 'False' must not become True."""
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
    )

    d = {"methods": "[]", "fine_grained": "False", "use_confidence_weighting": "True"}
    cfg = EnsembleAnnotationConfig.from_dict(d)
    assert cfg.fine_grained is False
    assert cfg.use_confidence_weighting is True


def test_registry_list_annotators():
    """Verify list_annotators() returns registered annotators."""
    from dapidl.pipeline.registry import list_annotators

    # Import annotators module to trigger registration
    import dapidl.pipeline.components.annotators  # noqa: F401

    available = list_annotators()
    assert "celltypist" in available
    assert "ground_truth" in available
    # singler, sctype etc. may or may not be available depending on deps
    assert len(available) >= 2


def test_registry_get_annotator_unknown_raises():
    from dapidl.pipeline.registry import get_annotator

    import pytest
    with pytest.raises(ValueError, match="Unknown annotator 'nonexistent'"):
        get_annotator("nonexistent")


def test_validate_methods_passes_for_registered():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )
    import dapidl.pipeline.components.annotators  # noqa: F401

    step = EnsembleAnnotationStep()
    # celltypist is always registered
    step._validate_methods([MethodSpec("celltypist", {"model": "Breast.pkl"})])


def test_validate_methods_fails_for_unregistered():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )
    import pytest

    step = EnsembleAnnotationStep()
    with pytest.raises(ValueError, match="not registered"):
        step._validate_methods([MethodSpec("nonexistent_method")])


def test_build_annotator_config_celltypist():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    spec = MethodSpec("celltypist", {"model": "Cells_Adult_Breast.pkl"})
    cfg = step._build_annotator_config(spec)
    assert cfg.method == "celltypist"
    assert cfg.model_names == ["Cells_Adult_Breast.pkl"]


def test_build_annotator_config_singler():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    spec = MethodSpec("singler", {"reference": "hpca"})
    cfg = step._build_annotator_config(spec)
    assert cfg.method == "singler"
    assert cfg.singler_reference == "hpca"


def test_build_annotator_config_generic_passthrough():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    spec = MethodSpec("sctype", {"confidence_threshold": 0.8})
    cfg = step._build_annotator_config(spec)
    assert cfg.method == "sctype"
    assert cfg.confidence_threshold == 0.8


def test_build_annotator_config_unknown_param_ignored():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    spec = MethodSpec("celltypist", {"model": "A.pkl", "nonexistent_key": "value"})
    cfg = step._build_annotator_config(spec)
    # Unknown params are silently dropped (no hasattr match)
    assert cfg.model_names == ["A.pkl"]


def test_make_source_label():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )

    assert EnsembleAnnotationStep._make_source_label(MethodSpec("celltypist", {"model": "Breast.pkl"})) == "celltypist_Breast.pkl"
    assert EnsembleAnnotationStep._make_source_label(MethodSpec("singler", {"reference": "hpca"})) == "singler_hpca"
    assert EnsembleAnnotationStep._make_source_label(MethodSpec("sctype")) == "sctype"


def test_config_hash_deterministic():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg = EnsembleAnnotationConfig(methods=[
        MethodSpec("celltypist", {"model": "A.pkl"}),
        MethodSpec("singler", {"reference": "blueprint"}),
    ])
    h1 = step._get_config_hash(cfg, "dataset123")
    h2 = step._get_config_hash(cfg, "dataset123")
    assert h1 == h2


def test_config_hash_order_independent():
    """Method order should not change the hash."""
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg1 = EnsembleAnnotationConfig(methods=[
        MethodSpec("celltypist", {"model": "A.pkl"}),
        MethodSpec("singler", {"reference": "blueprint"}),
    ])
    cfg2 = EnsembleAnnotationConfig(methods=[
        MethodSpec("singler", {"reference": "blueprint"}),
        MethodSpec("celltypist", {"model": "A.pkl"}),
    ])
    assert step._get_config_hash(cfg1, "ds") == step._get_config_hash(cfg2, "ds")


def test_config_hash_different_methods():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg1 = EnsembleAnnotationConfig(methods=[MethodSpec("celltypist", {"model": "A.pkl"})])
    cfg2 = EnsembleAnnotationConfig(methods=[MethodSpec("celltypist", {"model": "B.pkl"})])
    assert step._get_config_hash(cfg1, "ds") != step._get_config_hash(cfg2, "ds")


import polars as pl
from dapidl.pipeline.base import AnnotationResult


def _make_result(source: str, data: list[tuple[str, str, str, float]]) -> AnnotationResult:
    """Helper: create AnnotationResult from (cell_id, predicted_type, broad_category, confidence)."""
    df = pl.DataFrame(
        data,
        schema={"cell_id": pl.Utf8, "predicted_type": pl.Utf8, "broad_category": pl.Utf8, "confidence": pl.Float64},
        orient="row",
    )
    return AnnotationResult(annotations_df=df, stats={"source": source})


def test_consensus_unanimous():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg = EnsembleAnnotationConfig(
        methods=[MethodSpec("a"), MethodSpec("b")],
        min_agreement=2,
        use_confidence_weighting=False,
    )
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.9), ("c2", "Luminal", "Epithelial", 0.8)]),
        _make_result("b", [("c1", "T-cell", "Immune", 0.7), ("c2", "Luminal", "Epithelial", 0.6)]),
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 2
    assert stats["unanimous_agreement"] == 2
    row_c1 = consensus.filter(pl.col("cell_id") == "c1")
    assert row_c1["broad_category"][0] == "Immune"
    assert row_c1["n_agreement"][0] == 2


def test_consensus_majority():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg = EnsembleAnnotationConfig(
        methods=[MethodSpec("a"), MethodSpec("b"), MethodSpec("c")],
        min_agreement=2,
        use_confidence_weighting=False,
    )
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.9)]),
        _make_result("b", [("c1", "T-cell", "Immune", 0.7)]),
        _make_result("c", [("c1", "Fibroblast", "Stromal", 0.5)]),
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 1
    row = consensus.row(0, named=True)
    assert row["broad_category"] == "Immune"
    assert row["n_agreement"] == 2
    assert row["n_votes"] == 3


def test_consensus_min_agreement_filter():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg = EnsembleAnnotationConfig(
        methods=[MethodSpec("a"), MethodSpec("b"), MethodSpec("c")],
        min_agreement=3,
        use_confidence_weighting=False,
    )
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.9)]),
        _make_result("b", [("c1", "T-cell", "Immune", 0.7)]),
        _make_result("c", [("c1", "Fibroblast", "Stromal", 0.5)]),
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 0
    assert stats["insufficient_votes"] == 1


def test_consensus_weighted():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg = EnsembleAnnotationConfig(
        methods=[MethodSpec("a"), MethodSpec("b")],
        min_agreement=1,
        use_confidence_weighting=True,
    )
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.9)]),
        _make_result("b", [("c1", "Fibroblast", "Stromal", 0.1)]),
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 1
    assert consensus["broad_category"][0] == "Immune"


def test_consensus_weighted_overrides_majority():
    """One high-confidence vote should beat two low-confidence votes."""
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationConfig,
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    cfg = EnsembleAnnotationConfig(
        methods=[MethodSpec("a"), MethodSpec("b"), MethodSpec("c")],
        min_agreement=1,
        use_confidence_weighting=True,
    )
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.95)]),
        _make_result("b", [("c1", "Fibroblast", "Stromal", 0.1)]),
        _make_result("c", [("c1", "Pericyte", "Stromal", 0.1)]),
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 1
    assert consensus["broad_category"][0] == "Immune"
