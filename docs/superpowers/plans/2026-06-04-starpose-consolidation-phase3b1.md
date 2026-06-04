# StarPose Consolidation — Phase 3b-1 (Benchmark Evaluation Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `starpose` self-sufficient for PQ/F1 benchmarking by moving dapidl's **pure** benchmark evaluation layer (instance metrics, biological metrics, FOV selection, reporting) into starpose, with dapidl back-compat shims.

**Architecture:** Each module is pure (numpy/scipy/polars/stdlib, no `SegmentationOutput` coupling, no cross-imports), so each moves like the Phase-1 QC scorer: copy into starpose, add `__all__`, export from the starpose package, leave a `from … import *` shim at the old dapidl path. dapidl's `BenchmarkRunner` and the 3 consumer scripts keep working through the shims.

**Tech stack:** Python 3.12, `uv`, pytest, numpy/scipy/polars. starpose at `/home/chrism/git/starpose` (branch `feat/qc-consolidation`), dapidl at `/mnt/work/git/dapidl` (branch `feat/qc-consolidation`).

**Scope note:** 3b-1 is the **pure eval layer only**. The `SegmentationOutput→SegmentationResult` type migration — the benchmark runner rewrite, the segmenter-adapter de-dup (incl. the ratchet'd `benchmark/segmenters/stardist_adapter.py`), and the coupled `consensus/iou_weighted.py` + `evaluation/cross_method.py` — is **Phase 3b-2** (deferred). `consensus/instance_matching.py` also defers to 3b-2 (it's a consensus helper, not needed for metric scoring).

**Do NOT touch:** `starpose/methods/{cellotype,joint}.py`, `dapidl/training/instance/*`, the joint WIP.

> **⚠️ REVISION (2026-06-04, during execution):** starpose **already has** `starpose/benchmark/` with its own `fov_selector.py` (`select_fovs`/`extract_tile` on `starpose.types.FOVTile`), `reporting.py`, **and `runner.py`**. So **Tasks 3 and 4 below are CUT** — those are duplicate parallel implementations (different APIs), and reconciling them belongs with the framework de-dup in **Phase 3b-2**, not a one-way move. **3b-1 = Tasks 1 & 2 only** (instance_metrics + biological — the modules starpose genuinely lacked). Both are done and committed.

---

## Task 1: Move instance metrics into `starpose.evaluate`

**Files:**
- Create: `/home/chrism/git/starpose/src/starpose/evaluate/instance_metrics.py`
- Modify: `/home/chrism/git/starpose/src/starpose/evaluate/__init__.py`
- Create: `/home/chrism/git/starpose/tests/test_instance_metrics.py`
- Modify (→ shim): `/mnt/work/git/dapidl/src/dapidl/benchmark/instance_metrics.py`

- [ ] **Step 1: Copy the module + its test into starpose, verbatim**
```bash
cp /mnt/work/git/dapidl/src/dapidl/benchmark/instance_metrics.py \
   /home/chrism/git/starpose/src/starpose/evaluate/instance_metrics.py
cp /mnt/work/git/dapidl/tests/test_instance_metrics.py \
   /home/chrism/git/starpose/tests/test_instance_metrics.py
```
The module imports only `numpy` + `loguru` (both in starpose's env) — no edits to the body needed.

- [ ] **Step 2: Append `__all__` to the starpose copy** (so the dapidl shim can `import *`)

Append to `/home/chrism/git/starpose/src/starpose/evaluate/instance_metrics.py`:
```python
__all__ = [
    "match_instances_iou", "panoptic_quality", "segmentation_pq", "aji_plus",
    "average_precision", "per_class_f1", "confusion_matrix",
]
```

- [ ] **Step 3: Export from `starpose.evaluate`**

Replace `/home/chrism/git/starpose/src/starpose/evaluate/__init__.py` with:
```python
"""Segmentation evaluation metrics and utilities."""

from starpose.evaluate.agreement import compute_agreement
from starpose.evaluate.instance_metrics import (
    aji_plus,
    average_precision,
    confusion_matrix,
    match_instances_iou,
    panoptic_quality,
    per_class_f1,
    segmentation_pq,
)
from starpose.evaluate.morphometric import compute_morphometric
from starpose.evaluate.recovery import compute_recovery

__all__ = [
    "compute_morphometric", "compute_recovery", "compute_agreement",
    "match_instances_iou", "panoptic_quality", "segmentation_pq", "aji_plus",
    "average_precision", "per_class_f1", "confusion_matrix",
]
```

- [ ] **Step 4: Run the moved test in starpose**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_instance_metrics.py -q`
Expected: PASS (same tests that passed in dapidl, now against the starpose copy).

- [ ] **Step 5: Replace the dapidl module with a shim**

Overwrite `/mnt/work/git/dapidl/src/dapidl/benchmark/instance_metrics.py`:
```python
"""DEPRECATED shim — moved to starpose.evaluate.instance_metrics (starpose 0.3.0).

Import from starpose.evaluate in new code.
"""
from starpose.evaluate.instance_metrics import *  # noqa: F401,F403
from starpose.evaluate.instance_metrics import __all__  # noqa: F401
```

- [ ] **Step 6: Verify dapidl's test still passes through the shim**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_instance_metrics.py -q`
Expected: PASS — `from dapidl.benchmark.instance_metrics import panoptic_quality` resolves via the shim.

- [ ] **Step 7: Commit both repos**
```bash
cd /home/chrism/git/starpose && git add src/starpose/evaluate/instance_metrics.py src/starpose/evaluate/__init__.py tests/test_instance_metrics.py && git commit -m "feat(evaluate): instance metrics (PQ/AJI+/AP/F1) into starpose.evaluate"
cd /mnt/work/git/dapidl && git add src/dapidl/benchmark/instance_metrics.py && git commit -m "refactor(benchmark): instance_metrics becomes a starpose shim"
```

---

## Task 2: Move biological metrics into `starpose.evaluate`

`compute_biological_metrics` is pure numpy.

**Files:**
- Create: `/home/chrism/git/starpose/src/starpose/evaluate/biological.py`
- Modify: `/home/chrism/git/starpose/src/starpose/evaluate/__init__.py`
- Modify (→ shim): `/mnt/work/git/dapidl/src/dapidl/benchmark/evaluation/biological.py`

- [ ] **Step 1: Copy the module into starpose**
```bash
cp /mnt/work/git/dapidl/src/dapidl/benchmark/evaluation/biological.py \
   /home/chrism/git/starpose/src/starpose/evaluate/biological.py
```
Imports are numpy-only — no body edits.

- [ ] **Step 2: Append `__all__`**

Append to `/home/chrism/git/starpose/src/starpose/evaluate/biological.py`:
```python
__all__ = ["compute_biological_metrics"]
```

- [ ] **Step 3: Add to `starpose.evaluate` exports**

In `/home/chrism/git/starpose/src/starpose/evaluate/__init__.py`, add the import and the `__all__` entry:
```python
from starpose.evaluate.biological import compute_biological_metrics
```
and add `"compute_biological_metrics"` to `__all__`.

- [ ] **Step 4: Write a smoke test**

Create `/home/chrism/git/starpose/tests/test_biological.py`:
```python
import numpy as np

from starpose.evaluate import compute_biological_metrics


def test_compute_biological_metrics_runs_on_a_label_image():
    masks = np.zeros((32, 32), dtype=np.int32)
    masks[2:8, 2:8] = 1
    masks[20:28, 20:28] = 2
    out = compute_biological_metrics(masks)
    assert isinstance(out, dict) and len(out) > 0
```
(If `compute_biological_metrics` needs extra args, inspect its signature with
`uv run python -c "import inspect, starpose.evaluate as e; print(inspect.signature(e.compute_biological_metrics))"` and pass the required positional args; keep the assertion that it returns a non-empty dict.)

- [ ] **Step 5: Run the test**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_biological.py -q`
Expected: PASS.

- [ ] **Step 6: Replace the dapidl module with a shim**

Overwrite `/mnt/work/git/dapidl/src/dapidl/benchmark/evaluation/biological.py`:
```python
"""DEPRECATED shim — moved to starpose.evaluate.biological (starpose 0.3.0)."""
from starpose.evaluate.biological import *  # noqa: F401,F403
from starpose.evaluate.biological import __all__  # noqa: F401
```

- [ ] **Step 7: Verify the dapidl runner still imports it**

Run: `cd /mnt/work/git/dapidl && uv run python -c "from dapidl.benchmark.evaluation.biological import compute_biological_metrics as f; print('ok', f.__module__)"`
Expected: `ok starpose.evaluate.biological`

- [ ] **Step 8: Commit both repos**
```bash
cd /home/chrism/git/starpose && git add src/starpose/evaluate/biological.py src/starpose/evaluate/__init__.py tests/test_biological.py && git commit -m "feat(evaluate): biological metrics into starpose.evaluate"
cd /mnt/work/git/dapidl && git add src/dapidl/benchmark/evaluation/biological.py && git commit -m "refactor(benchmark): biological metrics becomes a starpose shim"
```

---

## Task 3: Move FOV selection into `starpose.benchmark`

`fov_selector.py` (FOVTile, load_transform, select_fovs, load_dapi_mosaic, extract_fov_tile, …) is pure numpy/polars. Create a new `starpose.benchmark` subpackage for benchmark-orchestration support.

**Note on `FOVTile`:** `starpose.types` already defines a *different* `FOVTile`. They will coexist by module path (`starpose.benchmark.fov_selector.FOVTile` vs `starpose.types.FOVTile`); do NOT merge them in 3b-1 — a later cleanup can unify. Keep `fov_selector`'s own `FOVTile` in its module.

**Files:**
- Create: `/home/chrism/git/starpose/src/starpose/benchmark/__init__.py`
- Create: `/home/chrism/git/starpose/src/starpose/benchmark/fov_selector.py`
- Modify (→ shim): `/mnt/work/git/dapidl/src/dapidl/benchmark/fov_selector.py`

- [ ] **Step 1: Create the starpose.benchmark package + copy the module**
```bash
mkdir -p /home/chrism/git/starpose/src/starpose/benchmark
printf '"""Benchmark orchestration support (FOV selection, reporting)."""\n' \
  > /home/chrism/git/starpose/src/starpose/benchmark/__init__.py
cp /mnt/work/git/dapidl/src/dapidl/benchmark/fov_selector.py \
   /home/chrism/git/starpose/src/starpose/benchmark/fov_selector.py
```
Imports are numpy/polars/stdlib — no body edits.

- [ ] **Step 2: Append `__all__` to the starpose copy**

Inspect the public names first:
`cd /home/chrism/git/starpose && grep -nE "^def |^class " src/starpose/benchmark/fov_selector.py`
Then append an `__all__` listing exactly those public (non-underscore) names, e.g.:
```python
__all__ = ["FOVTile", "load_transform", "select_fovs", "load_dapi_mosaic", "extract_fov_tile"]
```
(Match the actual public defs from the grep; include every non-underscore def/class.)

- [ ] **Step 3: Write a smoke test**

Create `/home/chrism/git/starpose/tests/test_fov_selector.py`:
```python
from starpose.benchmark import fov_selector


def test_fov_selector_public_api_present():
    for name in fov_selector.__all__:
        assert hasattr(fov_selector, name), f"missing public symbol {name}"
    # FOVTile is a dataclass with the documented fields
    assert hasattr(fov_selector, "FOVTile")
```

- [ ] **Step 4: Run the test**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_fov_selector.py -q`
Expected: PASS.

- [ ] **Step 5: Replace the dapidl module with a shim**

Overwrite `/mnt/work/git/dapidl/src/dapidl/benchmark/fov_selector.py`:
```python
"""DEPRECATED shim — moved to starpose.benchmark.fov_selector (starpose 0.3.0)."""
from starpose.benchmark.fov_selector import *  # noqa: F401,F403
from starpose.benchmark.fov_selector import __all__  # noqa: F401
```

- [ ] **Step 6: Verify the dapidl runner's fov_selector imports still resolve**

The runner imports several names from `dapidl.benchmark.fov_selector`. Confirm:
`cd /mnt/work/git/dapidl && uv run python -c "import dapidl.benchmark.runner as r; print('ok')"`
Expected: `ok` (runner imports resolve through the shim).

- [ ] **Step 7: Commit both repos**
```bash
cd /home/chrism/git/starpose && git add src/starpose/benchmark/__init__.py src/starpose/benchmark/fov_selector.py tests/test_fov_selector.py && git commit -m "feat(benchmark): fov_selector into starpose.benchmark"
cd /mnt/work/git/dapidl && git add src/dapidl/benchmark/fov_selector.py && git commit -m "refactor(benchmark): fov_selector becomes a starpose shim"
```

---

## Task 4: Move reporting into `starpose.benchmark`

`reporting.py` (`generate_report` + helpers) is pure json/numpy.

**Files:**
- Create: `/home/chrism/git/starpose/src/starpose/benchmark/reporting.py`
- Modify (→ shim): `/mnt/work/git/dapidl/src/dapidl/benchmark/reporting.py`

- [ ] **Step 1: Copy the module into starpose**
```bash
cp /mnt/work/git/dapidl/src/dapidl/benchmark/reporting.py \
   /home/chrism/git/starpose/src/starpose/benchmark/reporting.py
```
Imports are json/numpy/pathlib — no body edits.

- [ ] **Step 2: Append `__all__`**

Confirm the public entrypoint(s) with `grep -nE "^def " src/starpose/benchmark/reporting.py` (expect `generate_report`; helpers are underscore-prefixed). Append:
```python
__all__ = ["generate_report"]
```
(Add any other non-underscore public defs the grep shows.)

- [ ] **Step 3: Write a smoke test**

Create `/home/chrism/git/starpose/tests/test_reporting.py`:
```python
from starpose.benchmark import reporting


def test_reporting_exposes_generate_report():
    assert hasattr(reporting, "generate_report")
    for name in reporting.__all__:
        assert hasattr(reporting, name)
```

- [ ] **Step 4: Run the test**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_reporting.py -q`
Expected: PASS.

- [ ] **Step 5: Replace the dapidl module with a shim**

Overwrite `/mnt/work/git/dapidl/src/dapidl/benchmark/reporting.py`:
```python
"""DEPRECATED shim — moved to starpose.benchmark.reporting (starpose 0.3.0)."""
from starpose.benchmark.reporting import *  # noqa: F401,F403
from starpose.benchmark.reporting import __all__  # noqa: F401
```

- [ ] **Step 6: Verify the runner still imports it**

Run: `cd /mnt/work/git/dapidl && uv run python -c "from dapidl.benchmark.reporting import generate_report as g; print('ok', g.__module__)"`
Expected: `ok starpose.benchmark.reporting`

- [ ] **Step 7: Commit both repos**
```bash
cd /home/chrism/git/starpose && git add src/starpose/benchmark/reporting.py tests/test_reporting.py && git commit -m "feat(benchmark): reporting into starpose.benchmark"
cd /mnt/work/git/dapidl && git add src/dapidl/benchmark/reporting.py && git commit -m "refactor(benchmark): reporting becomes a starpose shim"
```

---

## Final verification

- [ ] starpose suite green: `cd /home/chrism/git/starpose && uv run pytest -q`.
- [ ] dapidl benchmark imports resolve through shims: `cd /mnt/work/git/dapidl && uv run python -c "import dapidl.benchmark.runner; from dapidl.benchmark import instance_metrics; print('ok')"`.
- [ ] starpose now exposes the metrics: `uv run python -c "from starpose.evaluate import panoptic_quality, per_class_f1, aji_plus; print('PQ/F1/AJI in starpose')"`.
- [ ] `starpose.benchmark` exists with `fov_selector` + `reporting`.
- [ ] Phase 3b-1 commits present on `feat/qc-consolidation` in both repos.

**After 3b-1:** starpose can score PQ/F1/AJI/AP and select/report FOVs — it's benchmark-capable. **Phase 3b-2** then does the type migration (SegmentationOutput→SegmentationResult), the runner rewrite onto `starpose.methods`, the segmenter-adapter de-dup (incl. the ratchet'd `stardist_adapter`), and the coupled `consensus/iou_weighted` + `evaluation/cross_method` + `consensus/instance_matching`. **Phase 3a** (the actual benchmark RUN vs STHELAR + hand-annotated Xenium → set the density threshold + validate the default) then runs on the consolidated starpose.
