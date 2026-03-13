# Declarative Annotation Registry Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded method orchestration in `ensemble_annotation.py` with a declarative, config-driven system using the existing annotator registry, so adding new methods (e.g., BANKSY) requires zero changes to pipeline infrastructure.

**Architecture:** A `MethodSpec(name, params)` dataclass drives a generic loop that calls `get_annotator()` from the existing registry. The consensus rewrite operates on `AnnotationResult` objects via polars DataFrames. Both the Pydantic pipeline config and the step-level dataclass config are updated. ClearML serialization uses JSON for the methods list.

**Tech Stack:** Python dataclasses, Pydantic v2, polars, ClearML SDK, Streamlit (dashboard)

**Spec:** `docs/superpowers/specs/2026-03-13-declarative-annotation-registry-design.md`

---

## Chunk 1: Core Config + Registry Verification

### Task 1: Add `singler_reference` field to base AnnotationConfig

**Files:**
- Modify: `src/dapidl/pipeline/base.py:423-452`
- Test: `tests/test_declarative_annotation.py` (create)

- [ ] **Step 1: Write failing test**

```python
# tests/test_declarative_annotation.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_declarative_annotation.py::test_annotation_config_has_singler_reference -v`
Expected: FAIL — `AnnotationConfig` has no `singler_reference` field

- [ ] **Step 3: Add singler_reference field**

In `src/dapidl/pipeline/base.py`, add after line 451 (`fine_grained: bool = False`):

```python
    # SingleR reference dataset (used when method="singler")
    singler_reference: str = "blueprint"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_declarative_annotation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_declarative_annotation.py src/dapidl/pipeline/base.py
git commit -m "feat: add singler_reference field to base AnnotationConfig"
```

### Task 2: Create MethodSpec and new EnsembleAnnotationConfig

**Files:**
- Modify: `src/dapidl/pipeline/steps/ensemble_annotation.py:1-74`
- Test: `tests/test_declarative_annotation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_declarative_annotation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_declarative_annotation.py -k "method_spec or ensemble_config" -v`
Expected: FAIL — `MethodSpec` not importable

- [ ] **Step 3: Implement MethodSpec and new EnsembleAnnotationConfig**

Replace the entire `EnsembleAnnotationConfig` class and add `MethodSpec` at the top of `src/dapidl/pipeline/steps/ensemble_annotation.py` (lines 39-73). The old fields (`celltypist_models`, `include_singler`, `singler_reference`, `include_sctype`) are removed. Full implementation is in the spec (lines 37-102).

Keep imports at the top unchanged. The new code:

```python
@dataclass
class MethodSpec:
    """Declarative specification for a single annotation method."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MethodSpec:
        return cls(name=d["name"], params=d.get("params", {}))


@dataclass
class EnsembleAnnotationConfig:
    """Configuration for ensemble annotation step."""
    methods: list[MethodSpec] = field(default_factory=list)

    # Consensus settings
    min_agreement: int = 2
    confidence_threshold: float = 0.5
    use_confidence_weighting: bool = True

    # Output settings
    fine_grained: bool = True
    create_derived_dataset: bool = True
    parent_dataset_id: str | None = None
    skip_if_exists: bool = True
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "methods": [m.to_dict() for m in self.methods],
            "min_agreement": self.min_agreement,
            "confidence_threshold": self.confidence_threshold,
            "use_confidence_weighting": self.use_confidence_weighting,
            "fine_grained": self.fine_grained,
            "create_derived_dataset": self.create_derived_dataset,
            "parent_dataset_id": self.parent_dataset_id,
            "skip_if_exists": self.skip_if_exists,
            "upload_to_s3": self.upload_to_s3,
            "s3_bucket": self.s3_bucket,
            "s3_endpoint": self.s3_endpoint,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnsembleAnnotationConfig:
        methods_raw = d.get("methods", d.get("General/methods", "[]"))
        if isinstance(methods_raw, str):
            methods_raw = json.loads(methods_raw)
        if isinstance(methods_raw, list) and all(isinstance(m, dict) for m in methods_raw):
            methods = [MethodSpec.from_dict(m) for m in methods_raw]
        else:
            methods = []

        def _pb(val: Any, default: bool = True) -> bool:
            """Parse bool from string or bool (ClearML sends 'True'/'False' strings)."""
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val) if val is not None else default

        return cls(
            methods=methods,
            min_agreement=int(d.get("min_agreement", d.get("General/min_agreement", 2))),
            confidence_threshold=float(d.get("confidence_threshold", d.get("General/confidence_threshold", 0.5))),
            use_confidence_weighting=_pb(d.get("use_confidence_weighting", d.get("General/use_confidence_weighting", True))),
            fine_grained=_pb(d.get("fine_grained", d.get("General/fine_grained", True))),
            create_derived_dataset=_pb(d.get("create_derived_dataset", True)),
            parent_dataset_id=d.get("parent_dataset_id"),
            skip_if_exists=_pb(d.get("skip_if_exists", True)),
            upload_to_s3=_pb(d.get("upload_to_s3", True)),
            s3_bucket=str(d.get("s3_bucket", "dapidl")),
            s3_endpoint=str(d.get("s3_endpoint", "")),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_declarative_annotation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/pipeline/steps/ensemble_annotation.py tests/test_declarative_annotation.py
git commit -m "feat: add MethodSpec and declarative EnsembleAnnotationConfig"
```

### Task 3: Verify registry has list_annotators and all annotators are registered

**Files:**
- Read: `src/dapidl/pipeline/registry.py` (no changes needed — `list_annotators()` exists at line 171)
- Test: `tests/test_declarative_annotation.py`

- [ ] **Step 1: Write test that verifies core annotators are registered**

Append to `tests/test_declarative_annotation.py`:

```python
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
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_declarative_annotation.py::test_registry_list_annotators tests/test_declarative_annotation.py::test_registry_get_annotator_unknown_raises -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_declarative_annotation.py
git commit -m "test: verify annotator registry has core annotators registered"
```

---

## Chunk 2: Validation + Execution Loop + Config Hash

> **CROSS-CHUNK DEPENDENCY**: Tasks 5 (delete old `_run_*` methods), 7 (rewrite `_build_consensus`), and 8 (update `execute()`) all modify the same file and depend on each other. The old `execute()` calls `_run_celltypist()` → `_build_consensus(list[dict])`. After Task 5 deletes `_run_*` and Task 7 rewrites consensus, `execute()` is broken until Task 8 rewires it. **If implementing sequentially**, do Tasks 5+7+8 in a single session and only commit after Task 8 passes tests. The tasks are separated for clarity, not for independent commits.

### Task 4: Implement _validate_methods and _build_annotator_config

**Files:**
- Modify: `src/dapidl/pipeline/steps/ensemble_annotation.py`
- Test: `tests/test_declarative_annotation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_declarative_annotation.py`:

```python
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
    assert not hasattr(cfg, "nonexistent_key") or cfg.__dict__.get("nonexistent_key") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_declarative_annotation.py -k "validate_methods or build_annotator_config" -v`
Expected: FAIL — methods don't exist yet

- [ ] **Step 3: Implement _validate_methods and _build_annotator_config**

Add these methods to the `EnsembleAnnotationStep` class in `ensemble_annotation.py`:

```python
def _validate_methods(self, methods: list[MethodSpec]) -> None:
    """Fail fast if any requested method is unavailable."""
    from dapidl.pipeline.registry import list_annotators

    available = list_annotators()
    for spec in methods:
        if spec.name not in available:
            raise ValueError(
                f"Annotator '{spec.name}' is not registered. "
                f"Available: {available}. "
                f"Check that required dependencies are installed."
            )

def _build_annotator_config(self, spec: MethodSpec) -> AnnotationConfig:
    """Translate MethodSpec params into annotator's expected config."""
    base = AnnotationConfig(method=spec.name)
    for key, value in spec.params.items():
        if key == "model":
            base.model_names = [value]
        elif key == "reference":
            base.singler_reference = value
        elif hasattr(base, key):
            setattr(base, key, value)
    return base

@staticmethod
def _make_source_label(spec: MethodSpec) -> str:
    """Create a human-readable source label for consensus tracking."""
    suffix = spec.params.get("model", spec.params.get("reference", ""))
    if suffix:
        return f"{spec.name}_{suffix}"
    return spec.name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_declarative_annotation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/pipeline/steps/ensemble_annotation.py tests/test_declarative_annotation.py
git commit -m "feat: add _validate_methods and _build_annotator_config"
```

### Task 5: Implement _run_methods generic execution loop

**Files:**
- Modify: `src/dapidl/pipeline/steps/ensemble_annotation.py`
- Test: `tests/test_declarative_annotation.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_declarative_annotation.py`:

```python
def test_make_source_label():
    from dapidl.pipeline.steps.ensemble_annotation import (
        EnsembleAnnotationStep,
        MethodSpec,
    )

    step = EnsembleAnnotationStep()
    assert step._make_source_label(MethodSpec("celltypist", {"model": "Breast.pkl"})) == "celltypist_Breast.pkl"
    assert step._make_source_label(MethodSpec("singler", {"reference": "hpca"})) == "singler_hpca"
    assert step._make_source_label(MethodSpec("sctype")) == "sctype"
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_declarative_annotation.py::test_make_source_label -v`
Expected: PASS (already implemented in Task 4)

- [ ] **Step 3: Implement _run_methods**

Add to `EnsembleAnnotationStep`:

```python
def _run_methods(
    self,
    methods: list[MethodSpec],
    adata: Any,
    data_path: Path,
) -> list[AnnotationResult]:
    """Run all declared annotation methods via the registry."""
    from dapidl.pipeline.registry import get_annotator

    results = []
    for spec in methods:
        annotator = get_annotator(spec.name, config=self._build_annotator_config(spec))
        result = annotator.annotate(
            adata=adata,
            expression_path=data_path,
        )
        result.stats["source"] = self._make_source_label(spec)
        results.append(result)
        logger.info(f"{spec.name}: {len(result.annotations_df)} cells annotated")
    return results
```

Also add `AnnotationResult` to the imports from `dapidl.pipeline.base` at the top of the file (line 30-36).

- [ ] **Step 4: Delete old _run_celltypist, _run_singler, _run_sctype methods**

Remove the three private methods (approximately lines 460-648 of the original file). These are replaced by the generic `_run_methods`.

- [ ] **Step 5: Run existing tests to check nothing is broken**

Run: `uv run pytest tests/ -v`
Expected: PASS (existing tests should still pass)

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/pipeline/steps/ensemble_annotation.py tests/test_declarative_annotation.py
git commit -m "feat: add generic _run_methods loop, delete hardcoded _run_* methods"
```

### Task 6: Rewrite _get_config_hash for methods list

**Files:**
- Modify: `src/dapidl/pipeline/steps/ensemble_annotation.py:158-177`
- Test: `tests/test_declarative_annotation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_declarative_annotation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_declarative_annotation.py -k "config_hash" -v`
Expected: FAIL — old `_get_config_hash` references removed fields

- [ ] **Step 3: Rewrite _get_config_hash**

Replace the existing `_get_config_hash` method (lines 158-177) with:

```python
def _get_config_hash(self, cfg: EnsembleAnnotationConfig, raw_dataset_id: str) -> str:
    """Generate deterministic hash of annotation configuration."""
    methods_normalized = sorted(
        [json.dumps(m.to_dict(), sort_keys=True) for m in cfg.methods]
    )
    config_str = (
        f"methods={'|'.join(methods_normalized)}|"
        f"min_agree={cfg.min_agreement}|"
        f"conf={cfg.confidence_threshold}|"
        f"fine={cfg.fine_grained}|"
        f"raw={raw_dataset_id}"
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Update skip logic in _check_existing_annotations**

In `_check_existing_annotations` (around line 278-289), replace the old field comparisons:

```python
# Old:
if (saved_config.get("celltypist_models") == ... and
    saved_config.get("singler_reference") == ...):

# New:
saved_methods = saved_config.get("methods", [])
current_methods = sorted([json.dumps(m.to_dict(), sort_keys=True) for m in cfg.methods])
saved_methods_normalized = sorted([json.dumps(m, sort_keys=True) for m in saved_methods])
if saved_methods_normalized == current_methods:
```

Also update the config saving in `execute()` (around line 406-420) to save the new format:

```python
json.dump({"methods": [m.to_dict() for m in cfg.methods], ...}, f, indent=2)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_declarative_annotation.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/pipeline/steps/ensemble_annotation.py tests/test_declarative_annotation.py
git commit -m "feat: rewrite config hash and skip logic for methods list"
```

---

## Chunk 3: Consensus Rewrite

### Task 7: Rewrite _build_consensus to use AnnotationResult

**Files:**
- Modify: `src/dapidl/pipeline/steps/ensemble_annotation.py:650-769`
- Test: `tests/test_declarative_annotation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_declarative_annotation.py`:

```python
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
    # Both cells have 2/2 agreement
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
        min_agreement=3,  # require unanimous
        use_confidence_weighting=False,
    )
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.9)]),
        _make_result("b", [("c1", "T-cell", "Immune", 0.7)]),
        _make_result("c", [("c1", "Fibroblast", "Stromal", 0.5)]),  # disagrees
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 0  # c1 filtered out — only 2/3 agree
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
    # One vote Immune (confidence 0.9), one vote Stromal (confidence 0.1)
    # Weighted: Immune should win
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
    # 2 votes Stromal (low conf), 1 vote Immune (high conf)
    # Weighted: Immune (0.95) > Stromal (0.1 + 0.1 = 0.2)
    results = [
        _make_result("a", [("c1", "T-cell", "Immune", 0.95)]),
        _make_result("b", [("c1", "Fibroblast", "Stromal", 0.1)]),
        _make_result("c", [("c1", "Pericyte", "Stromal", 0.1)]),
    ]
    consensus, stats = step._build_consensus(results, cfg)
    assert consensus.height == 1
    assert consensus["broad_category"][0] == "Immune"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_declarative_annotation.py -k "consensus" -v`
Expected: FAIL — old `_build_consensus` expects `list[dict]`, not `list[AnnotationResult]`

- [ ] **Step 3: Rewrite _build_consensus**

Replace the existing `_build_consensus` method with the polars-native implementation from the spec (lines 214-299). Full code:

```python
def _build_consensus(
    self,
    results: list[AnnotationResult],
    cfg: EnsembleAnnotationConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build consensus from AnnotationResult objects using polars."""
    all_votes = pl.concat([
        r.annotations_df.select([
            pl.col("cell_id"),
            pl.col("predicted_type"),
            pl.col("broad_category"),
            pl.col("confidence"),
        ]).with_columns(
            pl.lit(r.stats["source"]).alias("source")
        )
        for r in results
    ])

    if cfg.use_confidence_weighting:
        vote_scores = (
            all_votes
            .group_by(["cell_id", "broad_category"])
            .agg([
                pl.col("confidence").sum().alias("vote_score"),
                pl.len().alias("vote_count"),
                pl.col("predicted_type").sort_by("confidence", descending=True).first().alias("best_predicted_type"),
            ])
        )
    else:
        vote_scores = (
            all_votes
            .group_by(["cell_id", "broad_category"])
            .agg([
                pl.len().alias("vote_score"),  # Unweighted: count = score
                pl.len().alias("vote_count"),
                pl.col("predicted_type").first().alias("best_predicted_type"),
            ])
        )

    winners = (
        vote_scores
        .sort("vote_score", descending=True)
        .group_by("cell_id")
        .first()
    )

    total_votes_per_cell = (
        all_votes
        .group_by("cell_id")
        .agg(pl.len().alias("n_votes"))
    )

    consensus = (
        winners
        .join(total_votes_per_cell, on="cell_id")
        .select([
            pl.col("cell_id"),
            pl.col("best_predicted_type").alias("predicted_type"),
            pl.col("broad_category"),
            (pl.col("vote_score") / pl.col("n_votes")).alias("confidence"),
            pl.col("n_votes"),
            pl.col("vote_count").alias("n_agreement"),
        ])
    )

    consensus = consensus.filter(pl.col("n_agreement") >= cfg.min_agreement)

    n_total = all_votes.select("cell_id").n_unique()
    n_unanimous = consensus.filter(pl.col("n_agreement") == pl.col("n_votes")).height
    stats = {
        "total_cells": n_total,
        "annotated_cells": consensus.height,
        "unanimous_agreement": n_unanimous,
        "majority_agreement": consensus.height - n_unanimous,
        "insufficient_votes": n_total - consensus.height,
        "methods_used": [r.stats["source"] for r in results],
    }

    return consensus, stats
```

- [ ] **Step 4: Delete old _standardize_label method**

The old `_standardize_label` method (approximately lines 771-851) was used by the dict-based consensus. With the new approach, `broad_category` comes directly from each `AnnotationResult.annotations_df` (each annotator already maps to broad categories via `map_to_broad_category()`). Delete this method.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_declarative_annotation.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/pipeline/steps/ensemble_annotation.py tests/test_declarative_annotation.py
git commit -m "feat: rewrite consensus to use AnnotationResult with polars-native voting"
```

---

## Chunk 4: Wire Up execute() + Update Pipeline Plumbing

### Task 8: Update execute() to use the new methods

**Files:**
- Modify: `src/dapidl/pipeline/steps/ensemble_annotation.py:352-460`

- [ ] **Step 1: Update execute() method**

In `execute()`, replace the hardcoded annotation blocks (lines 357-384) with:

```python
# Validate all methods are available before doing anything
cfg = self.config
self._validate_methods(cfg.methods)

# Load expression data
adata = self._load_expression(expression_path, data_path, platform)
logger.info(f"Loaded expression data: {adata.n_obs} cells, {adata.n_vars} genes")

# Run all declared methods via registry
results = self._run_methods(cfg.methods, adata, data_path)

if not results:
    raise ValueError("No annotation methods succeeded")

# Build consensus
consensus_df, stats = self._build_consensus(results, cfg)
logger.info(f"Consensus built: {len(consensus_df)} cells with agreement")
```

- [ ] **Step 2: Update config saving in execute() (around line 406-420)**

Replace the old config save block:

```python
# Old:
json.dump({
    "celltypist_models": cfg.celltypist_models,
    "include_singler": cfg.include_singler,
    "singler_reference": cfg.singler_reference,
    "include_sctype": cfg.include_sctype,
    ...
}, f, indent=2)

# New:
json.dump({
    "methods": [m.to_dict() for m in cfg.methods],
    "min_agreement": cfg.min_agreement,
    "confidence_threshold": cfg.confidence_threshold,
    "fine_grained": cfg.fine_grained,
    "use_confidence_weighting": cfg.use_confidence_weighting,
}, f, indent=2)
```

- [ ] **Step 3: Update skip logic field comparison (around line 278-289)**

Replace the old field comparisons in `_check_existing_annotations`:

```python
# Old:
if (saved_config.get("celltypist_models") == list(cfg.celltypist_models)
    and saved_config.get("singler_reference") == cfg.singler_reference
    and ...):

# New:
saved_methods = sorted([json.dumps(m, sort_keys=True) for m in saved_config.get("methods", [])])
current_methods = sorted([json.dumps(m.to_dict(), sort_keys=True) for m in cfg.methods])
if saved_methods == current_methods:
```

- [ ] **Step 4: Update logging/stats output**

Search for any `cfg.celltypist_models` or `cfg.include_singler` references in logging/stats within execute(). Replace with `cfg.methods` references:

```python
# Old:
"annotation_methods": [p["source"] for p in all_predictions]

# New (already works — source comes from results):
"annotation_methods": [r.stats["source"] for r in results]
```

- [ ] **Step 5: Update get_parameter_schema() if it exists**

Search for `get_parameter_schema` in the file. If it defines ClearML UI params, update it to expose `methods` as a JSON text field instead of individual boolean/list params.

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/dapidl/pipeline/steps/ensemble_annotation.py
git commit -m "feat: wire execute() to use declarative _run_methods + consensus"
```

### Task 9: Update Pydantic config + controller + step runner (single atomic commit)

> **IMPORTANT**: Tasks 9a-9c are done together in a single commit to avoid broken intermediate states. The Pydantic config, controller, and step runner all reference the old fields — changing one without the others leaves the codebase broken.

**Files:**
- Modify: `src/dapidl/pipeline/unified_config.py:288-373`
- Modify: `src/dapidl/pipeline/unified_controller.py:265-285, 480-494, 620-636, 795-806`
- Modify: `scripts/clearml_step_runner.py:235-257`

#### 9a: Update unified_config.py Pydantic AnnotationConfig

- [ ] **Step 1: Replace method-specific fields with methods list**

Replace the `celltypist_models`, `include_singler`, `singler_reference`, `include_sctype`, `include_popv` fields (lines 302-326) with:

```python
    # Annotation methods (declarative list)
    methods: list[dict[str, Any]] = Field(
        default=[
            {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
            {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
            {"name": "singler", "params": {"reference": "blueprint"}},
        ],
        description="List of annotation method specs. Each dict has 'name' (registry name) and 'params' (method-specific parameters).",
    )

    # Preset name (optional, loads from configs/default_methods.json)
    method_preset: str | None = Field(
        default=None,
        description="Named preset from configs/default_methods.json (overrides methods list)",
    )
```

Keep `strategy`, `ground_truth_*`, consensus settings, `fine_grained`, and `extended_consensus` fields as-is. Remove `celltypist_models`, `include_singler`, `singler_reference`, `include_sctype`, `include_popv`.

Add `from typing import Any` to imports if not already present.

Also update the migration/compatibility helpers that reference old fields:
- **Lines 1128-1134**: `from_clearml_params()` — update to parse `methods` JSON instead of individual fields
- **Lines 1265**: `from_old_config()` — update to convert `old_config.model_names` into `methods` list
- **Lines 1332-1334**: Another migration path — update to build `methods` from old fields
- **Lines 1460-1501**: Documentation comments — update field mapping table

#### 9b: Update unified_controller.py (ALL 4 reference sites)

- [ ] **Step 2: Update ClearML DAG parameter_override (line 272-277)**

```python
# Old:
"step_config/celltypist_models": "${pipeline.annotation/celltypist_models}",
"step_config/include_singler": "${pipeline.annotation/include_singler}",
"step_config/singler_reference": "${pipeline.annotation/singler_reference}",

# New:
"step_config/methods": "${pipeline.annotation/methods}",
"step_config/use_confidence_weighting": "${pipeline.annotation/use_confidence_weighting}",
```

- [ ] **Step 3: Update multi-tissue DAG step (line 487)**

```python
# Old:
"step_config/model_names": ",".join(cfg.annotation.celltypist_models),

# New: This is the legacy annotation step (not ensemble), convert methods to model_names for backward compat
"step_config/model_names": ",".join(
    m["params"]["model"] for m in cfg.annotation.methods if m["name"] == "celltypist"
),
```

- [ ] **Step 4: Update local execution path (lines 627-636)**

```python
# Old:
annot_config = EnsembleAnnotationConfig(
    celltypist_models=cfg.annotation.celltypist_models,
    include_singler=cfg.annotation.include_singler,
    singler_reference=cfg.annotation.singler_reference,
    ...
)

# New:
from dapidl.pipeline.steps.ensemble_annotation import MethodSpec
annot_config = EnsembleAnnotationConfig(
    methods=[MethodSpec.from_dict(m) for m in cfg.annotation.methods],
    min_agreement=cfg.annotation.min_agreement,
    confidence_threshold=cfg.annotation.confidence_threshold,
    fine_grained=cfg.annotation.fine_grained,
    upload_to_s3=cfg.output.upload_to_s3,
    s3_bucket=cfg.output.s3_bucket,
)
```

- [ ] **Step 5: Update multi-tissue local execution path (line 798-803)**

```python
# Old:
annot_config = AnnotationStepConfig(
    annotator="celltypist",
    strategy="consensus",
    model_names=cfg.annotation.celltypist_models,
    ...
)

# New: Extract celltypist models from methods list for legacy AnnotationStep
celltypist_models = [m["params"]["model"] for m in cfg.annotation.methods if m["name"] == "celltypist"]
annot_config = AnnotationStepConfig(
    annotator="celltypist",
    strategy="consensus",
    model_names=celltypist_models or ["Cells_Adult_Breast.pkl"],
    fine_grained=cfg.annotation.fine_grained,
)
```

#### 9c: Update clearml_step_runner.py deserialization

- [ ] **Step 6: Replace manual construction with from_dict() (lines 241-256)**

```python
# Old:
models_str = step_config.get("celltypist_models", "...")
models = models_str.split(",") if isinstance(models_str, str) else models_str
config = EnsembleAnnotationConfig(
    celltypist_models=models,
    include_singler=_parse_bool(...),
    ...
)

# New:
config = EnsembleAnnotationConfig.from_dict(step_config)
```

#### Verify + Commit

- [ ] **Step 7: Run lint on all changed files**

Run: `uv run ruff check src/dapidl/pipeline/unified_config.py src/dapidl/pipeline/unified_controller.py scripts/clearml_step_runner.py`
Expected: PASS

- [ ] **Step 8: Run grep to verify no remaining references to removed fields**

Run: `uv run python3 -c "import subprocess; r = subprocess.run(['grep', '-rn', 'celltypist_models\|include_singler\|include_sctype\|include_popv', 'src/dapidl/pipeline/unified_config.py', 'src/dapidl/pipeline/unified_controller.py', 'scripts/clearml_step_runner.py'], capture_output=True, text=True); print(r.stdout or 'CLEAN')"`
Expected: CLEAN (no matches)

- [ ] **Step 9: Commit all three files together**

```bash
git add src/dapidl/pipeline/unified_config.py src/dapidl/pipeline/unified_controller.py scripts/clearml_step_runner.py
git commit -m "feat: replace method-specific config with declarative methods list across pipeline plumbing"
```

---

## Chunk 5: CLI + Default Presets + Dashboard

### Task 12: Update CLI annotation arguments (ALL reference sites)

**Files:**
- Modify: `src/dapidl/cli.py` (lines 316-318, 444-447, 1603-1606, 2700-2830, 3178-3179)

> **WARNING**: `cli.py` references the old fields in 5+ locations across different CLI subcommands. All must be updated.

- [ ] **Step 1: Find ALL references to old fields**

Run: `grep -n 'celltypist_models\|include_singler\|singler_reference\|singler_ref' src/dapidl/cli.py`

This will show all locations. There are references in:
- `clearml-pipeline run` command (lines 2700-2830) — main target
- `annotate` command (line 318) — passes `singler_reference` to base `AnnotationConfig`
- Another command (line 446) — passes `singler_reference`
- SOTA pipeline (line 1605) — passes `singler_reference`
- SOTA output (lines 3178-3179) — prints `config.annotation.celltypist_models` and `config.annotation.singler_reference`

- [ ] **Step 2: Update Click options for `clearml-pipeline run` (lines 2700-2710)**

Replace `--celltypist-models` and `--include-singler/--no-singler` with:

```python
@click.option(
    "--methods",
    type=str,
    multiple=True,
    help="Annotation method specs as JSON: '{\"name\":\"celltypist\",\"params\":{\"model\":\"Breast.pkl\"}}'",
)
@click.option(
    "--method-preset",
    type=str,
    default=None,
    help="Named preset from configs/default_methods.json (e.g., 'breast_standard', 'universal')",
)
```

- [ ] **Step 3: Update function signature (lines 2767-2768)**

Replace `celltypist_models: tuple, include_singler: bool` with `methods: tuple, method_preset: str | None`.

- [ ] **Step 4: Update config construction (lines 2810-2811)**

```python
import json
from pathlib import Path

if method_preset:
    presets_path = Path(__file__).parent.parent.parent / "configs" / "default_methods.json"
    with open(presets_path) as f:
        presets = json.load(f)
    if method_preset not in presets:
        raise click.BadParameter(f"Unknown preset '{method_preset}'. Available: {list(presets.keys())}")
    methods_list = presets[method_preset]
elif methods:
    methods_list = [json.loads(m) for m in methods]
else:
    # Default
    methods_list = [
        {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
        {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
        {"name": "singler", "params": {"reference": "blueprint"}},
    ]

# Pass to Pydantic config:
# config = ... AnnotationConfig(methods=methods_list, ...)
```

- [ ] **Step 5: Update summary output (lines 2826-2828)**

```python
console.print(f"Annotation: {len(methods_list)} methods")
for m in methods_list:
    console.print(f"  - {m['name']}: {m.get('params', {})}")
```

- [ ] **Step 6: Update SOTA pipeline output (lines 3178-3179)**

```python
# Old:
console.print(f"  Models: {config.annotation.celltypist_models}")
console.print(f"  SingleR: {config.annotation.singler_reference} (CRITICAL for Stromal)")

# New:
console.print(f"  Methods: {len(config.annotation.methods)} configured")
for m in config.annotation.methods:
    console.print(f"    - {m['name']}: {m.get('params', {})}")
```

- [ ] **Step 7: Update other CLI commands using singler_reference (lines 318, 446, 1605)**

These pass `singler_reference=singler_ref` to the base `AnnotationConfig` dataclass. Since Task 1 added `singler_reference` as a field on the base dataclass, these remain valid — no change needed. Verify with:

Run: `grep -n 'singler_ref' src/dapidl/cli.py | head -10`
Expected: References pass to `AnnotationConfig(singler_reference=...)` which still has this field.

- [ ] **Step 8: Verify no remaining references to removed Pydantic fields**

Run: `grep -n 'cfg\.annotation\.celltypist_models\|cfg\.annotation\.include_singler\|cfg\.annotation\.singler_reference\|config\.annotation\.celltypist_models\|config\.annotation\.include_singler' src/dapidl/cli.py`
Expected: No matches

- [ ] **Step 9: Run lint**

Run: `uv run ruff check src/dapidl/cli.py`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/dapidl/cli.py
git commit -m "feat: update CLI with --methods and --method-preset options"
```

### Task 13: Create default_methods.json

**Files:**
- Create: `configs/default_methods.json`

- [ ] **Step 1: Create presets file**

```json
{
  "breast_standard": [
    {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}}
  ],
  "lung_standard": [
    {"name": "celltypist", "params": {"model": "Human_Lung_Atlas.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}}
  ],
  "universal": [
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_Low.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}},
    {"name": "singler", "params": {"reference": "hpca"}}
  ],
  "immune_focused": [
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_Low.pkl"}},
    {"name": "celltypist", "params": {"model": "Adult_COVID19_PBMC.pkl"}},
    {"name": "singler", "params": {"reference": "monaco"}}
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add configs/default_methods.json
git commit -m "feat: add default method presets for annotation pipeline"
```

### Task 14: Update dashboard Pipeline Launcher

**Files:**
- Modify: `dashboard/pages/1_Pipeline_Launcher.py` (lines around 210-225, 390-393)

- [ ] **Step 1: Replace annotation UI widgets**

Find the annotation config section (around lines 212-224). Replace with a dynamic method builder:

```python
with st.expander("Annotation Methods", expanded=True):
    # Load presets
    presets_path = Path(__file__).parent.parent.parent / "configs" / "default_methods.json"
    presets = {}
    if presets_path.exists():
        with open(presets_path) as f:
            presets = json.load(f)

    preset_choice = st.selectbox(
        "Method Preset",
        options=["custom"] + list(presets.keys()),
        index=list(presets.keys()).index("breast_standard") + 1 if "breast_standard" in presets else 0,
    )

    if preset_choice != "custom":
        methods = presets[preset_choice]
        st.json(methods)
    else:
        methods_json = st.text_area(
            "Methods (JSON)",
            value=json.dumps(presets.get("breast_standard", []), indent=2),
            height=200,
        )
        try:
            methods = json.loads(methods_json)
        except json.JSONDecodeError:
            st.error("Invalid JSON")
            methods = []

    cfg["methods"] = methods
```

- [ ] **Step 2: Update params serialization**

Replace lines 391-393:

```python
# Old:
params["annotation/celltypist_models"] = ",".join(cfg["celltypist_models"])
params["annotation/include_singler"] = str(cfg["include_singler"])
params["annotation/singler_reference"] = cfg["singler_reference"]

# New:
params["annotation/methods"] = json.dumps(cfg["methods"])
```

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/1_Pipeline_Launcher.py
git commit -m "feat: update dashboard with dynamic method list builder"
```

---

## Chunk 6: Final Integration + Verification

### Task 15: Run full test suite and lint

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/ scripts/ dashboard/`
Expected: PASS (or fix any issues)

- [ ] **Step 3: Run type checker**

Run: `uv run mypy src/dapidl/pipeline/steps/ensemble_annotation.py src/dapidl/pipeline/base.py --ignore-missing-imports`
Expected: PASS

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: lint and type check cleanup for declarative annotation"
```

### Task 16: Manual E2E verification

- [ ] **Step 1: Test local pipeline with new config**

```bash
uv run dapidl clearml-pipeline run \
  -t lung bf8f913f xenium 2 \
  --method-preset breast_standard \
  --epochs 1 \
  --local
```

Verify: Pipeline starts, annotation step uses methods from preset, consensus produces output.

- [ ] **Step 2: Verify ClearML serialization**

Check the ClearML task hyperparams in the web UI — `annotation/methods` should contain the JSON methods list.

- [ ] **Step 3: Document any issues found and fix**

If anything fails, fix and commit with descriptive message.
