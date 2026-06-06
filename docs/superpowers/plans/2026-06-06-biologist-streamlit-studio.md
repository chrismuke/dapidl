# Biologist Pipeline Studio — Implementation Plan

> **For agentic workers:** Inline TDD execution (superpowers:executing-plans). Steps use `- [ ]` tracking. DRY · YAGNI · TDD · frequent commits.

**Goal:** A biologist-friendly Streamlit UI to configure, sweep, and analyze the DAPIDL pipeline on local/AWS/DO compute, by cloning a ClearML controller template task and enqueuing it.

**Architecture:** Extract dependency-light logic into `scripts/studio/` (pure Python + `requests`, NO streamlit/torch). `param_schema` maps biologist selections → `DAPIDLPipelineConfig.to_clearml_parameters()` (single source of truth, loaded by-path to skip the heavy `__init__`). `clearml_launch` does REST clone/set-params/enqueue/scalars. `streamlit_pipeline.py` becomes UI glue over these.

**Tech Stack:** Streamlit, requests (ClearML REST), pydantic (via reused `unified_config`), pytest.

---

## File Structure
- **Create** `scripts/studio/__init__.py` — package marker.
- **Create** `scripts/studio/param_schema.py` — `PARAMS`, `build_param_overrides`, `expand_sweep`, `validate` (pure).
- **Create** `scripts/studio/clearml_launch.py` — `ClearMLClient` moved from `streamlit_pipeline.py` + new `clone_task`/`set_task_params`/`enqueue_task`/`get_tasks_by_tag`/`get_task_scalars`.
- **Create** `configs/studio.json` — `{"template_task_id": "<id>", "compute_targets": {"Local 3090": "gpu-local", "AWS (scales)": "gpu-cloud", "DigitalOcean": "gpu-do"}, "services_queue": "services"}`.
- **Modify** `scripts/streamlit_pipeline.py` — import `studio`, add Configure/Sweep/Results tabs.
- **Create** `tests/test_studio_param_schema.py`, `tests/test_studio_clearml_launch.py`.

Run tests with: `uv run pytest tests/test_studio_param_schema.py tests/test_studio_clearml_launch.py -v`

---

## Task 1: param_schema — selection → ClearML param overrides

**Files:** Create `scripts/studio/__init__.py`, `scripts/studio/param_schema.py`; Test `tests/test_studio_param_schema.py`.

Selection schema (biologist label → backend). `PARAMS` entries: `{name, label, widget, options:[(label,value)], default, sweepable, advanced, help}`. Derived mappings in `build_param_overrides`:
- `detect_level`: "Broad (4 classes)"→fine_grained False, "Fine (~12)"→True
- `label_method` → `AnnotatorType` (popV ensemble→`popv`, CellTypist→`celltypist`, Ground truth→`ground_truth`)
- `segmentation` → `SegmenterType` (Vendor (native)→`native`, Cellpose→`cellpose`, StarDist→`stardist`)
- `image_model` → `BackboneType` (EfficientNet-V2-S→`efficientnetv2_rw_s`, ConvNeXt-Tiny→`convnext_tiny`, ResNet50→`resnet50`)
- `patch_size` → `lmdb.patch_sizes=[int]`; `epochs`→`training.epochs`; advanced: `batch_size`,`learning_rate`,`patience`
- `compute_target` → `execution.gpu_queue` (via configs/studio.json map), `execution.default_queue=services_queue`

`build_param_overrides(selections: dict, datasets: list[tuple], compute_targets: dict, services_queue: str="services") -> dict[str,str]`:
1. `_load_unified_config()` imports `unified_config.py` by path (repo_root/src/dapidl/pipeline/unified_config.py) via `importlib.util` (controller pattern) — avoids heavy `__init__`.
2. Build `DAPIDLPipelineConfig(training=TrainingConfig(epochs, batch_size, backbone, ...), annotation=AnnotationConfig(fine_grained, annotator), lmdb=LMDBConfig(patch_sizes=[ps]), execution=ExecutionConfig(execute_remotely=True, gpu_queue=mapped, default_queue=services_queue))`; `config.input.add_tissue(...)` per dataset.
3. Return `config.to_clearml_parameters()`.

- [ ] **Step 1 — failing test** (`tests/test_studio_param_schema.py`):
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from studio import param_schema as ps

DATASETS = [("breast", "abc123", "xenium", 2)]
COMPUTE = {"Local 3090": "gpu-local", "AWS (scales)": "gpu-cloud"}

def test_build_param_overrides_maps_biologist_selections():
    sel = {"detect_level": "Fine (~12)", "label_method": "popV ensemble",
           "segmentation": "Vendor (native)", "image_model": "EfficientNet-V2-S",
           "patch_size": 128, "epochs": 50, "compute_target": "AWS (scales)",
           "batch_size": 64}
    out = ps.build_param_overrides(sel, DATASETS, COMPUTE, services_queue="services")
    assert out["training/backbone"] == "efficientnetv2_rw_s"
    assert out["annotation/fine_grained"] == "True"
    assert out["training/epochs"] == "50"
    assert "128" in out["lmdb/patch_sizes"]
    assert out["execution/gpu_queue"] == "gpu-cloud"
    assert out["execution/default_queue"] == "services"
```
- [ ] **Step 2 — run, expect FAIL** `ModuleNotFoundError: studio` → after stub, assert mismatches. (`uv run pytest tests/test_studio_param_schema.py -v`)
- [ ] **Step 3 — implement** `param_schema.py` (PARAMS + `_load_unified_config` + `build_param_overrides`). NOTE: confirm `to_clearml_parameters()` key format against the real output; adjust asserts if keys differ (e.g. nested vs slash) — the real serialization is authoritative.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `git add scripts/studio/__init__.py scripts/studio/param_schema.py tests/test_studio_param_schema.py && git commit -m "feat(studio): biologist param schema → ClearML overrides"`

## Task 2: expand_sweep + validate

**Files:** Modify `scripts/studio/param_schema.py`; `tests/test_studio_param_schema.py`.

`expand_sweep(base_selections, sweep_axes: dict[str, list], datasets, compute_targets, sweep_id) -> list[dict]`: cartesian product over `sweep_axes`; for each combo, merge into `base_selections`, call `build_param_overrides`, attach `"_tag": f"sweep-{sweep_id}"` and `"_run_name"`. `validate(selections, datasets, sweep_axes=None) -> list[str]`: errors for empty datasets, empty sweep axis.

- [ ] **Step 1 — failing tests:**
```python
def test_expand_sweep_cartesian():
    runs = ps.expand_sweep({"epochs":50,"detect_level":"Broad (4 classes)","label_method":"popV ensemble","segmentation":"Vendor (native)","image_model":"EfficientNet-V2-S","patch_size":128,"compute_target":"Local 3090","batch_size":64},
                           {"patch_size":[64,128], "image_model":["EfficientNet-V2-S","ConvNeXt-Tiny"]},
                           DATASETS, COMPUTE, sweep_id="42")
    assert len(runs) == 4
    assert all(r["_tag"] == "sweep-42" for r in runs)
    assert {r["lmdb/patch_sizes"] for r in runs} >= {"[64]", "[128]"} or any("64" in r["lmdb/patch_sizes"] for r in runs)

def test_validate_flags_no_dataset():
    assert any("dataset" in e.lower() for e in ps.validate({}, [], None))
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `expand_sweep` + `validate`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `-m "feat(studio): sweep expansion + config validation"`

## Task 3: ClearMLClient launch methods (REST)

**Files:** Create `scripts/studio/clearml_launch.py` (move `ClearMLClient` from `streamlit_pipeline.py`, extend); Test `tests/test_studio_clearml_launch.py`.

New methods (REST `_post` already exists): `clone_task(template_id, name)` → POST `tasks.clone` `{task, new_task_name}` → returns `id`; `set_task_params(task_id, params)` → POST `tasks.edit`/`tasks.set_parameters` with `hyperparams`/`{task, parameters}`; `enqueue_task(task_id, queue_name)` → POST `tasks.enqueue` `{task, queue_name}`; `get_tasks_by_tag(tag)` → `tasks.get_all` `{tags:[tag]}`; `get_task_scalars(task_id)` → `events.get_task_latest_scalar_values` or `tasks.get_by_id` last_metrics → `{metric: value}`.

- [ ] **Step 1 — failing test** (monkeypatch `_post` to capture endpoint+payload):
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from studio.clearml_launch import ClearMLClient

def test_clone_and_enqueue_payloads(monkeypatch):
    c = ClearMLClient(api_server="http://x")
    calls = []
    monkeypatch.setattr(c, "_post", lambda ep, pl: (calls.append((ep, pl)), {"id": "new1"})[1])
    new_id = c.clone_task("tmpl1", "run-A")
    assert new_id == "new1"
    assert calls[-1][0] == "tasks.clone" and calls[-1][1]["task"] == "tmpl1"
    c.enqueue_task("new1", "gpu-cloud")
    assert calls[-1][0] == "tasks.enqueue" and calls[-1][1]["queue_name"] == "gpu-cloud"
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** `clearml_launch.py` (moved client + methods). Verify endpoint names against the ClearML REST of the self-hosted server during smoke; adjust if the server uses `tasks.set_parameters` vs `tasks.edit`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `-m "feat(studio): ClearML REST clone/set-params/enqueue/scalars"`

## Task 4: Wire the UI (Configure / Sweep / Results)

**Files:** Modify `scripts/streamlit_pipeline.py`; Create `configs/studio.json`.

Replace `pipeline_config`/`build_cli_command`/`launch_section` with schema-driven tabs that call `param_schema` + `clearml_launch`. Keep `dataset_picker`, Monitor, Datasets. Configure&Run: render `PARAMS` (Advanced in expander) → on Launch: load template id from `configs/studio.json` (banner+command if missing) → clone→set→enqueue→show ClearML link. Sweep: multiselect sweepable params + values → `expand_sweep` → loop clone/set/enqueue → show N. Results: `get_tasks_by_tag` → `get_task_scalars` → dataframe + compare link.

- [ ] **Step 1 — implement** UI (no unit test — Streamlit glue).
- [ ] **Step 2 — manual smoke:** `uv run --with streamlit streamlit run scripts/streamlit_pipeline.py`; launch 1 run + a 2-point patch-size sweep on `gpu-local`; confirm tasks appear enqueued in ClearML.
- [ ] **Step 3 — commit** `-m "feat(studio): biologist Configure/Sweep/Results tabs over ClearML clone+enqueue"`

---

## Self-Review
- **Spec coverage:** launch mechanism (T3+T4), compute-as-queue (T1 mapping + studio.json), curated params (T1), sweeps (T2+T4), results (T3+T4), error handling (T2 validate + T4 missing-template banner), UI/logic split (file structure). ✓
- **Type consistency:** `build_param_overrides`/`expand_sweep`/`validate` signatures consistent across T1/T2/T4; `ClearMLClient` method names consistent T3/T4.
- **Open risk (verify at smoke):** exact ClearML REST endpoint for setting params (`tasks.edit` vs `tasks.set_parameters`) and the `to_clearml_parameters()` key format — both confirmed against the live server in T3/T1 and asserts adjusted to the real output.
