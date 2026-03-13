# Declarative Annotation Registry — Design Spec

**Date**: 2026-03-13
**Status**: Approved (design phase)
**Scope**: Refactor `ensemble_annotation.py` from hardcoded method blocks to a declarative, config-driven system using the existing annotator registry.

## Problem

The ensemble annotation pipeline step (`ensemble_annotation.py`) hardcodes each annotation method as a separate `if` block in `execute()`:

```python
for model_name in cfg.celltypist_models:   # CellTypist loop
if cfg.include_singler:                     # SingleR block
if cfg.include_sctype:                      # scType block (NotImplementedError!)
```

Adding a new method (e.g., BANKSY) requires modifying `execute()`, `EnsembleAnnotationConfig`, and adding a new `_run_*` private method. Meanwhile, an annotator registry (`registry.py`) and protocol (`AnnotatorProtocol` in `base.py`) already exist but are unused by the ensemble step.

## Goal

Replace hardcoded method orchestration with a declarative config that lists method specs. The step becomes a generic loop: validate availability, instantiate from registry, run, feed results to consensus. Adding a new annotator requires only implementing `AnnotatorProtocol` and registering it — zero changes to the ensemble step.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config format | List of `MethodSpec(name, params)` | Flexible per-method params without hardcoded boolean flags |
| Backward compat | Clean break | Old `celltypist_models` / `include_singler` fields removed entirely. Existing ClearML pipeline templates and cached annotation datasets will need updating — see Breaking Changes section |
| Consensus input | Rewrite to use `AnnotationResult` | Cleaner than adapter dicts; polars-native for 700K+ cell performance |
| Unavailable methods | Fail fast | Don't waste 20 min on CellTypist only to discover SingleR is missing |
| Dashboard | Update in scope | Replace checkboxes with dynamic method list builder |

## Architecture

### Config Schema

```python
@dataclass
class MethodSpec:
    """Declarative specification for a single annotation method."""
    name: str                                          # Registry name
    params: dict[str, Any] = field(default_factory=dict)  # Method-specific params

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MethodSpec:
        return cls(name=d["name"], params=d.get("params", {}))

@dataclass
class EnsembleAnnotationConfig:
    methods: list[MethodSpec]                           # Replaces all method-specific fields

    # Consensus settings (unchanged)
    min_agreement: int = 2
    confidence_threshold: float = 0.5
    use_confidence_weighting: bool = True

    # Output settings (unchanged)
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
        methods = [MethodSpec.from_dict(m) for m in methods_raw]
        return cls(
            methods=methods,
            min_agreement=int(d.get("min_agreement", d.get("General/min_agreement", 2))),
            confidence_threshold=float(d.get("confidence_threshold", d.get("General/confidence_threshold", 0.5))),
            use_confidence_weighting=bool(d.get("use_confidence_weighting", d.get("General/use_confidence_weighting", True))),
            fine_grained=bool(d.get("fine_grained", d.get("General/fine_grained", True))),
            create_derived_dataset=bool(d.get("create_derived_dataset", True)),
            parent_dataset_id=d.get("parent_dataset_id"),
            skip_if_exists=bool(d.get("skip_if_exists", True)),
            upload_to_s3=bool(d.get("upload_to_s3", True)),
            s3_bucket=str(d.get("s3_bucket", "dapidl")),
            s3_endpoint=str(d.get("s3_endpoint", "")),
        )
```

**Old config:**
```python
EnsembleAnnotationConfig(
    celltypist_models=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
    include_singler=True,
    singler_reference="blueprint",
    include_sctype=False,
)
```

**New config:**
```python
EnsembleAnnotationConfig(methods=[
    MethodSpec("celltypist", {"model": "Cells_Adult_Breast.pkl"}),
    MethodSpec("celltypist", {"model": "Immune_All_High.pkl"}),
    MethodSpec("singler", {"reference": "blueprint"}),
])
```

### Two AnnotationConfig Classes

The codebase has two separate `AnnotationConfig` types that must both be updated:

1. **`base.py` AnnotationConfig** (dataclass) — used by individual annotator components. Each annotator reads its params from this config (e.g., `cfg.model_names`, `cfg.strategy`). This is what `_build_annotator_config()` constructs.

2. **`unified_config.py` AnnotationConfig** (Pydantic BaseModel) — used by the pipeline controller, CLI, and dashboard. Has fields like `celltypist_models`, `include_singler`, `singler_reference`. This is what gets serialized to ClearML hyperparams.

The refactor touches both:
- **Pydantic `AnnotationConfig`**: Replace `celltypist_models`, `include_singler`, `singler_reference`, `include_sctype`, `include_popv` with a single `methods: list[dict]` field (JSON-serializable list of method specs)
- **Dataclass `AnnotationConfig`**: Unchanged — annotators still read their params from it. The `_build_annotator_config()` method in the step translates `MethodSpec.params` into the appropriate dataclass fields.

### Validation (Fail Fast)

Before loading expression data or running any annotation:

```python
def _validate_methods(self, methods: list[MethodSpec]) -> None:
    available = list_annotators()
    for spec in methods:
        if spec.name not in available:
            raise ValueError(
                f"Annotator '{spec.name}' is not registered. "
                f"Available: {available}. "
                f"Check that required dependencies are installed."
            )
```

The existing `try/except ImportError` pattern in `annotators/__init__.py` handles optional deps — uninstalled annotators never register, so they won't appear in `list_annotators()`.

### Execution Loop

Replaces `_run_celltypist()`, `_run_singler()`, `_run_sctype()` with a single generic loop:

```python
def _run_methods(
    self,
    methods: list[MethodSpec],
    adata: anndata.AnnData,
    data_path: Path,
) -> list[AnnotationResult]:
    results = []
    for spec in methods:
        annotator = get_annotator(spec.name, config=self._build_annotator_config(spec))
        result = annotator.annotate(
            adata=adata,
            expression_path=data_path,  # Some annotators need data_path (e.g., SingleR uses it)
        )  # config already set at instantiation via get_annotator()
        result.stats["source"] = self._make_source_label(spec)
        results.append(result)
        logger.info(f"{spec.name}: {len(result.annotations_df)} cells annotated")
    return results
```

**Key details:**
- Config is passed at instantiation via `get_annotator(name, config=...)` so that annotators that lazy-init in `__init__` (like CellTypistAnnotator) get the right config immediately
- `expression_path=data_path` is passed through because some annotators (SingleR, Azimuth) need the data directory for R bridge file I/O
- Error handling: exceptions from individual annotators propagate up (no silent swallowing — consistent with fail-fast design)

`_build_annotator_config()` translates `MethodSpec.params` into the `AnnotationConfig` dataclass (from `base.py`). Param names are mapped to the actual field names each annotator reads:

```python
def _build_annotator_config(self, spec: MethodSpec) -> AnnotationConfig:
    base = AnnotationConfig(method=spec.name)
    for key, value in spec.params.items():
        if key == "model":
            base.model_names = [value]
        elif key == "reference":
            # SingleR reads cfg.singler_reference (not an "extras" dict)
            # We need to add singler_reference as a field or use setattr
            base.singler_reference = value  # Note: requires adding field to AnnotationConfig
        elif hasattr(base, key):
            setattr(base, key, value)
    return base
```

**Required change to `base.py` AnnotationConfig**: Add `singler_reference: str = "blueprint"` field so SingleR params are properly passed. The current `AnnotationConfig` dataclass has no field for this — `SingleRAnnotator` reads it via `getattr(cfg, "singler_reference", "blueprint")`. Adding the field makes this explicit.

The private methods `_run_celltypist`, `_run_singler`, `_run_sctype` are deleted.

### Protocol Conformance

**Known non-conforming annotator**: `PopVStyleEnsembleAnnotator` has signature `annotate(self, adata) -> EnsembleResult` instead of the protocol's `annotate(self, config, adata, ...) -> AnnotationResult`. If added to a `methods` list, the generic loop would fail.

**Resolution**: `PopVStyleEnsembleAnnotator` is itself an ensemble orchestrator (it runs CellTypist + SingleR internally). It should NOT be used as a method inside the new declarative ensemble — that would be nesting ensembles. It remains available as a standalone annotator for direct use but is excluded from registry-based ensemble composition. The `_validate_methods()` step will catch this if someone tries to use it, since it either won't be registered or can be explicitly excluded.

### Consensus Rewrite

`_build_consensus()` is rewritten to accept `list[AnnotationResult]` instead of `list[dict]`. The current dict-based format (`{source, predictions, confidence, cell_ids}` — flat lists) is replaced by operating directly on the polars DataFrames inside `AnnotationResult.annotations_df` (columns: `cell_id`, `predicted_type`, `broad_category`, `confidence`).

```python
def _build_consensus(
    self,
    results: list[AnnotationResult],
    cfg: EnsembleAnnotationConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    # Stack all predictions with source column
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

    # For each cell, count votes per broad_category (optionally weighted by confidence)
    if cfg.use_confidence_weighting:
        vote_scores = (
            all_votes
            .group_by(["cell_id", "broad_category"])
            .agg([
                pl.col("confidence").sum().alias("vote_score"),
                pl.len().alias("vote_count"),
                # Keep one predicted_type as representative (most confident)
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

    # Pick winner per cell (highest vote_score)
    winners = (
        vote_scores
        .sort("vote_score", descending=True)
        .group_by("cell_id")
        .first()
    )

    # Compute agreement metrics
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

    # Filter by min_agreement
    consensus = consensus.filter(pl.col("n_agreement") >= cfg.min_agreement)

    # Stats
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

The output schema is unchanged: `cell_id`, `predicted_type`, `broad_category`, `confidence`, `n_votes`, `n_agreement`. Downstream steps need no changes.

### Config Hash and Skip Logic

The existing `_get_config_hash()` hashes individual fields (`celltypist_models`, `include_singler`, etc.). This must be rewritten to hash the `methods` list:

```python
def _get_config_hash(self, cfg: EnsembleAnnotationConfig, raw_dataset_id: str) -> str:
    # Sort methods by (name, sorted params) for deterministic hashing
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

The skip logic in `_check_existing_annotations()` (line 278-289) that compares `saved_config.get("celltypist_models")` etc. must also be updated to compare the serialized `methods` list instead.

**Cache invalidation**: Existing cached annotation datasets (on disk and in ClearML) will have old-format config hashes. They will not match new-format hashes, so they will be re-computed on first run with the new config format. This is acceptable for a clean break.

### ClearML Serialization

`MethodSpec` serializes as plain dicts for ClearML hyperparameters. The `from_dict()` / `to_dict()` methods on `EnsembleAnnotationConfig` handle the round-trip:

```python
# Controller serializes methods list as JSON string
parameter_override={
    "General/methods": json.dumps([m.to_dict() for m in cfg.annotation.methods]),
    "General/min_agreement": cfg.annotation.min_agreement,
    ...
}

# Step runner deserializes (in clearml_step_runner.py)
config = EnsembleAnnotationConfig.from_dict(task.get_parameters_as_dict())
```

The step runner (`scripts/clearml_step_runner.py`) currently constructs `EnsembleAnnotationConfig` field-by-field (lines 245-256). This manual construction is replaced by calling `EnsembleAnnotationConfig.from_dict()` which handles JSON parsing of the `methods` field and ClearML's `General/` prefix convention.

### Default Method Presets

A new `configs/default_methods.json` provides named presets:

```json
{
  "breast_standard": [
    {"name": "celltypist", "params": {"model": "Cells_Adult_Breast.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}}
  ],
  "universal": [
    {"name": "celltypist", "params": {"model": "Immune_All_High.pkl"}},
    {"name": "celltypist", "params": {"model": "Immune_All_Low.pkl"}},
    {"name": "singler", "params": {"reference": "blueprint"}},
    {"name": "singler", "params": {"reference": "hpca"}}
  ]
}
```

## Files Changed

| File | Change | Scope |
|------|--------|-------|
| `src/dapidl/pipeline/steps/ensemble_annotation.py` | Core refactor: new config with `to_dict()`/`from_dict()`, generic execution loop, consensus rewrite, config hash update, skip logic update | Major rewrite (~700 lines) |
| `src/dapidl/pipeline/unified_controller.py` | Serialize `methods` list as JSON in `parameter_override` | ~20 lines |
| `src/dapidl/pipeline/unified_config.py` | Replace `celltypist_models`, `include_singler`, `singler_reference`, `include_sctype`, `include_popv` with `methods: list[dict]` field in Pydantic `AnnotationConfig` | ~40 lines |
| `src/dapidl/pipeline/registry.py` | Ensure `list_annotators()` exists, verify all annotators register | ~10 lines |
| `src/dapidl/pipeline/base.py` | Add `singler_reference: str = "blueprint"` field to dataclass `AnnotationConfig` | ~2 lines |
| `scripts/clearml_step_runner.py` | Replace manual field-by-field construction with `EnsembleAnnotationConfig.from_dict()` | ~20 lines |
| `dashboard/pages/1_Pipeline_Launcher.py` | Replace SingleR/scType checkboxes and CellTypist multi-select with dynamic method list builder UI | ~100 lines |
| `configs/default_methods.json` | **New** — default method presets | ~30 lines |

**Not changed**: Individual annotator files (`celltypist.py`, `singler.py`, etc.) — they already implement `AnnotatorProtocol` and register via the existing registry. No changes needed.

**Note on `cli.py`**: The CLI does not directly construct `EnsembleAnnotationConfig`. It builds the Pydantic `AnnotationConfig` from `unified_config.py`, which gets serialized to ClearML hyperparams. CLI changes are limited to updating the Pydantic config args (handled in the `unified_config.py` row above). The CLI's Click options (`--celltypist-models`, etc.) will need updating to accept the new `--methods` format or a `--preset` option.

## Breaking Changes

This is a clean break. The following will need updating:

1. **ClearML pipeline templates**: Existing templates have `celltypist_models`, `include_singler` etc. as hyperparams. Must be reset and re-created with new `methods` param format.
2. **Cached annotation datasets**: Old config hashes won't match new hashes. Annotations will be re-computed on first run (acceptable — annotation is fast compared to training).
3. **Step runner deserialization**: `clearml_step_runner.py` manual construction replaced by `from_dict()`.
4. **Dashboard saved configs**: Any saved pipeline configs in the dashboard will use old field names.

## Testing

- Unit test: `MethodSpec` round-trip through `to_dict()` / `from_dict()`
- Unit test: `EnsembleAnnotationConfig` round-trip through `to_dict()` / `from_dict()` including ClearML `General/` prefix format
- Unit test: `_validate_methods()` raises for unregistered annotator
- Unit test: `_build_annotator_config()` maps params correctly for celltypist (model), singler (reference), and generic fields
- Unit test: `_get_config_hash()` produces consistent hashes regardless of method order
- Integration test: `_build_consensus()` produces correct output from `AnnotationResult` list (test weighted and unweighted voting)
- Integration test: Consensus filters by `min_agreement` correctly
- E2E: Run pipeline with new config format on lung_2fov, verify ClearML serialization round-trip and annotation output

## What This Enables

After this refactor, adding BANKSY (or any new annotator) requires:
1. Implement `AnnotatorProtocol` in `src/dapidl/pipeline/components/annotators/banksy.py`
2. Decorate with `@register_annotator`
3. Add to config: `MethodSpec("banksy", {"resolution": 0.5})`

Zero changes to `ensemble_annotation.py`, `unified_controller.py`, or any other pipeline infrastructure.
