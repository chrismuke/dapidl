# Biologist Pipeline Studio — Design Spec (2026-06-06)

## Goal
A Streamlit web UI that lets a **non-expert (biologist)** configure, launch, sweep, and
analyze the DAPIDL pipeline with different settings — on local / AWS / DigitalOcean
compute — with minimal friction. **Enhances** the existing `scripts/streamlit_pipeline.py`
(which already has a ClearML REST client, dataset picker, single-run launcher, monitor).

## Approved decisions
- **Launch mechanism:** clone a registered **controller template task** → override its
  parameters → **enqueue to a queue**, all via the ClearML **REST API** (no SDK dep).
  The run executes on a ClearML agent, not the Streamlit host → robust to host restarts.
  This is the pattern `scripts/clearml_pipeline_controller.py` was built for.
- **Compute target = ClearML queue.** One dropdown maps to a queue: Local 3090
  (`gpu-local`), AWS-autoscaler (`gpu-cloud`), optional DigitalOcean (`gpu-do`). No
  code difference between targets. Sweep parallelism comes free from the AWS autoscaler.
- **Curated biologist parameters** with technical knobs under an "Advanced" expander.
- **Tabs:** Configure & Run, Sweep, Results (keep existing Monitor, Datasets).

## Architecture — separate UI from logic (so the logic is unit-testable)
The existing `streamlit_pipeline.py` mixes the testable `ClearMLClient` with `import streamlit`.
We extract the dependency-light logic into a small `scripts/studio/` package (pure Python +
`requests` only — NO streamlit, NO torch), which both the UI and the tests import.

**Files:**
- **Create `scripts/studio/__init__.py`** — package marker.
- **Create `scripts/studio/param_schema.py`** — single source of truth.
  - `PARAMS`: list of param specs `{key, label, widget, options, default, help, sweepable, advanced}`
    where `key` is the ClearML controller param key and `label` is biologist-facing.
  - `build_param_overrides(selections: dict) -> dict[str, str]` — map UI selections to
    ClearML param keys (incl. derived: `detect_level` → `fine_grained`, `compute_target` →
    `gpu_queue`/`default_queue`). Pure.
  - `expand_sweep(sweep_selections: dict, base: dict, sweep_id: str) -> list[dict]` —
    cartesian product over chosen sweepable params; returns one override-dict per run,
    each carrying the shared `sweep_id` tag. Pure.
  - `validate(selections, datasets) -> list[str]` — human-readable errors (no dataset,
    empty sweep axis, etc.). Pure.
- **Create `scripts/studio/clearml_launch.py`** — move `ClearMLClient` here from
  `streamlit_pipeline.py` and extend (REST only):
  - existing: auth, `get_queues/get_datasets/get_pipelines/get_task/...`
  - new: `clone_task(template_id, name) -> new_id`, `set_task_params(task_id, params)`,
    `enqueue_task(task_id, queue)`, `get_tasks_by_tag(tag) -> list`,
    `get_task_scalars(task_id) -> dict` (last reported scalar values; macro_f1, per-class F1).
  - `TEMPLATE_TASK_ID` resolved from `configs/studio.json` (new) with a clear error +
    the `create-controller-task` command if absent.
- **Modify `scripts/streamlit_pipeline.py`** — import from `scripts.studio`; add the three
  tabs; keep Monitor/Datasets. UI glue only.
- **Create `tests/test_studio_param_schema.py`**, **`tests/test_studio_clearml_launch.py`**.

## Data flow
- **Configure & Run:** schema-driven form → `selections` → `build_param_overrides` →
  `clone(template)` → `set_task_params` → `enqueue(queue)` → show run + ClearML link.
- **Sweep:** pick sweepable params + value sets → `expand_sweep` → loop
  clone+set+enqueue, all tagged `sweep-<id>` → show N enqueued.
- **Results:** `get_tasks_by_tag('sweep-<id>')` (or recent) → `get_task_scalars` per run →
  table (macro-F1, per-class F1, status, runtime) + "Compare in ClearML" deep-link.

## Parameter set (initial)
dataset(s) [existing picker], **detect level** (Broad 4 / Fine ~12 → `fine_grained`),
**label method** (→ `annotator`), **segmentation** (→ `segmenter`), **image model**
(→ `backbone`), **patch size**, **epochs**, **compute target** (→ queues).
Advanced: batch size, learning rate, patience, loss.

## Error handling
- Missing template task → banner + the one command to create it (no crash).
- `validate()` blocks launch on bad config (no dataset / empty sweep axis) with clear text.
- ClearML unreachable → existing graceful banner.
- Sweep: per-run enqueue wrapped; one failure is reported in-row, the rest continue.

## Testing (TDD)
Pure-logic unit tests, no Streamlit and no live network:
- `param_schema`: `build_param_overrides` maps labels→ClearML keys with defaults; derived
  fields (`detect_level`→`fine_grained`, `compute_target`→queues); advanced included.
- `expand_sweep`: single-axis and multi-axis cartesian expansion; shared `sweep_id`;
  base values preserved for non-swept params.
- `validate`: catches empty dataset and empty sweep axis.
- `ClearMLClient`: payload + endpoint construction for `clone_task`/`set_task_params`/
  `enqueue_task` (monkeypatched `requests`); `get_task_scalars` parses a sample response.
UI layer = manual smoke (1 run + a 2-point sweep on `gpu-local`).

## Out of scope (YAGNI)
Bayesian HPO (grid/explicit sweeps only for v1), auth/multi-user, editing the pipeline DAG
from the UI, DigitalOcean agent provisioning (just leave `gpu-do` selectable).
