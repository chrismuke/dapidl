# StarPose Consolidation — Phase 1 (Plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `starpose` the single home for the StarDist-grounded QC scorer, so `dapidl` no longer calls StarDist directly — without breaking any existing dapidl call site.

**Architecture:** Move `dapidl/qc/segmentation_grounded.py` into `starpose/qc/`, swapping its private StarDist call for the existing `StarDist2DAdapter` (which must first be taught to return per-object probabilities). dapidl keeps a thin re-export shim so existing imports keep working. A guardrail test forbids new direct `stardist` imports in dapidl.

**Tech stack:** Python 3.12, `uv`, pytest, numpy/scipy/scikit-image, StarDist (TF) behind the starpose adapter. Two repos: `starpose` at `/home/chrism/git/starpose` (branch `master`), `dapidl` at `/mnt/work/git/dapidl` (branch `feat/nucleus-qc-scorer`).

**Scope note:** The spec listed "de-dup the consensus code" under Phase 1. Planning revealed dapidl's `topological_voting_consensus` operates on a different type (`dapidl.benchmark.segmenters.base.SegmentationOutput`) than starpose's `topological_voting` (`starpose.types.SegmentationResult`) and is entangled with dapidl's parallel benchmark framework. **Consensus/benchmark de-dup is therefore deferred to Phase 3** (benchmark consolidation). Phase 1 is the QC-scorer move only.

**Do NOT touch:** `starpose/methods/{cellotype,joint}.py`, `dapidl/training/instance/*` (instance-seg WIP).

---

## Task 1: Expose per-object probabilities through the StarDist adapter (starpose)

The QC scorer needs StarDist's per-object `prob` (its `prob_min` objectness gate). The adapter currently discards it (`labels, _ = predict_instances(img)`) and `SegmentationResult` has no field for it.

**Files:**
- Modify: `/home/chrism/git/starpose/src/starpose/types.py`
- Modify: `/home/chrism/git/starpose/src/starpose/methods/stardist.py`
- Test: `/home/chrism/git/starpose/tests/test_stardist_probs.py` (create)

- [ ] **Step 1: Write the failing test** (GPU-free — fakes the StarDist model)

Create `/home/chrism/git/starpose/tests/test_stardist_probs.py`:
```python
import numpy as np

from starpose.methods.stardist import StarDist2DAdapter
from starpose.types import SegmentationResult


class _FakeModel:
    """Stand-in for StarDist2D: returns fixed labels + a details dict with probs."""
    def __init__(self, labels, probs):
        self._labels, self._probs = labels, probs
    def predict_instances(self, img):
        return self._labels, {"prob": self._probs}


def test_segmentation_result_has_probs_field_defaulting_none():
    res = SegmentationResult(masks=np.zeros((4, 4), np.int32), centroids=np.empty((0, 2)),
                             n_cells=0, runtime=0.0, method_name="x")
    assert res.probs is None


def test_run_threads_labels_and_probs(monkeypatch):
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    probs = np.array([0.91, 0.42], dtype=float)
    adapter = StarDist2DAdapter(gpu=False)
    monkeypatch.setattr(adapter, "_get_model", lambda: _FakeModel(labels, probs))
    img = np.array([[0, 100], [200, 300]], dtype=np.uint16)   # non-flat
    m, p = adapter._run(img)
    assert np.array_equal(m, labels) and np.allclose(p, probs)


def test_run_flat_patch_skips_model(monkeypatch):
    def _boom():
        raise AssertionError("model must not be called on a flat patch")
    adapter = StarDist2DAdapter(gpu=False)
    monkeypatch.setattr(adapter, "_get_model", _boom)
    m, p = adapter._run(np.full((8, 8), 7, dtype=np.uint16))   # flat: p_high-p_low < 1e-6
    assert m.shape == (8, 8) and int(m.max()) == 0
    assert p.size == 0


def test_segment_sets_probs_field(monkeypatch):
    """segment() must put the probs onto the result; GPU-memory probes neutralised."""
    labels = np.array([[0, 1], [1, 1]], dtype=np.int32)
    probs = np.array([0.77], dtype=float)
    adapter = StarDist2DAdapter(gpu=False)
    monkeypatch.setattr(adapter, "_get_model", lambda: _FakeModel(labels, probs))
    monkeypatch.setattr("starpose.methods.stardist.measure_gpu_memory", lambda: 0.0)
    monkeypatch.setattr("starpose.methods.stardist.reset_gpu_memory", lambda: None)
    res = adapter.segment(np.array([[0, 100], [200, 300]], dtype=np.uint16))
    assert np.allclose(res.probs, probs) and np.array_equal(res.masks, labels)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_stardist_probs.py -q`
Expected: FAIL — `SegmentationResult` has no `probs` arg / adapter doesn't thread probs.

- [ ] **Step 3: Add the `probs` field to `SegmentationResult`**

In `/home/chrism/git/starpose/src/starpose/types.py`, add a field at the END of the `SegmentationResult` dataclass (after `cell_centroids`, before `__post_init__`):
```python
    cell_centroids: np.ndarray | None = None

    # Per-object detection confidence (StarDist `prob`); None for backends that
    # don't expose it. Index alignment: label k -> probs[k-1].
    probs: np.ndarray | None = None
```

- [ ] **Step 4: Thread probs through the StarDist adapter + add the flat-patch fast-path**

In `/home/chrism/git/starpose/src/starpose/methods/stardist.py`, replace `_run` and the `segment` body:
```python
    def _run(self, image: np.ndarray):
        """Return (label mask int32, per-object prob array). Label k -> prob[k-1]."""
        model = self._get_model()
        p_low, p_high = np.percentile(image, [1, 99.8])
        if p_high - p_low < 1e-6:
            # Flat/empty patch: skip the model (matches the legacy QC fast-path).
            return np.zeros(image.shape, dtype=np.int32), np.array([], dtype=float)
        img = ((image.astype(np.float32) - p_low) / (p_high - p_low)).clip(0, 1)
        labels, details = model.predict_instances(img)
        return labels.astype(np.int32), np.asarray(details["prob"], dtype=float)

    def segment(self, image: np.ndarray, pixel_size: float = 0.2125) -> SegmentationResult:
        mem_before = measure_gpu_memory()
        t0 = time.time()
        masks, probs = self._run(image)
        runtime = time.time() - t0
        peak_mem = max(measure_gpu_memory() - mem_before, 0.0)
        reset_gpu_memory()
        centroids = centroids_from_masks(masks)
        return SegmentationResult(
            masks=masks,
            centroids=centroids,
            n_cells=int(masks.max()),
            runtime=runtime,
            method_name=self.name,
            peak_memory_mb=peak_mem,
            probs=probs,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_stardist_probs.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Verify nothing else broke + lint**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_methods.py -q && uv run ruff check src/starpose/types.py src/starpose/methods/stardist.py`
Expected: PASS, no new lint errors.

- [ ] **Step 7: Commit (starpose)**
```bash
cd /home/chrism/git/starpose
git add src/starpose/types.py src/starpose/methods/stardist.py tests/test_stardist_probs.py
git commit -m "feat(stardist): expose per-object probs on SegmentationResult"
```

---

## Task 2: Public segmentation API on `starpose` (create / segment)

The spec's stable surface is `starpose.create(...)` + `starpose.segment(...)`. The registry function is `get_method`; expose a public alias + a one-shot convenience so dapidl never imports submodules.

**Files:**
- Modify: `/home/chrism/git/starpose/src/starpose/__init__.py`
- Test: `/home/chrism/git/starpose/tests/test_public_api.py` (create)

- [ ] **Step 1: Write the failing test**

Create `/home/chrism/git/starpose/tests/test_public_api.py`:
```python
import numpy as np

import starpose


def test_create_is_exposed_and_builds_a_registered_method():
    seg = starpose.create("stardist", gpu=False)
    assert seg.name == "stardist"


def test_segment_convenience_runs_a_named_method(monkeypatch):
    class _Fake:
        name = "stardist"
        def segment(self, image, pixel_size=0.2125):
            from starpose.types import SegmentationResult
            return SegmentationResult(masks=np.zeros((2, 2), np.int32),
                                      centroids=np.empty((0, 2)), n_cells=0,
                                      runtime=0.0, method_name="stardist")
    monkeypatch.setattr(starpose, "create", lambda *a, **k: _Fake())
    res = starpose.segment(np.zeros((2, 2), np.uint16), method="stardist")
    assert res.method_name == "stardist"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_public_api.py -q`
Expected: FAIL — `starpose.create` / `starpose.segment` do not exist.

- [ ] **Step 3: Add the public API to `starpose/__init__.py`**

Append to `/home/chrism/git/starpose/src/starpose/__init__.py` (lazy imports inside the
functions so `import starpose` stays light and there's no package-load import cycle):
```python
def create(method, **kwargs):
    """Instantiate a registered segmentation method by name (public alias)."""
    from starpose.methods import get_method
    return get_method(method, **kwargs)


def segment(image, method="stardist", *, pixel_size=0.2125, gpu=True, **kwargs):
    """One-shot segmentation with a named registered method.

    For modality-adaptive selection use starpose.dispatch.AdaptiveDispatcher.
    """
    return create(method, gpu=gpu, **kwargs).segment(image, pixel_size=pixel_size)
```
Add `"create"` and `"segment"` to the module's `__all__` if one is defined.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_public_api.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit (starpose)**
```bash
cd /home/chrism/git/starpose
git add src/starpose/__init__.py tests/test_public_api.py
git commit -m "feat(api): public starpose.create + starpose.segment"
```

---

## Task 3: Move the QC scorer into `starpose.qc.segmentation_grounded` (starpose)

Relocate the scorer + its pure functions. The pure functions are GPU-free and copied verbatim; only the class's segmentation path changes (StarDist via the adapter, not a private model).

**Files:**
- Create: `/home/chrism/git/starpose/src/starpose/qc/segmentation_grounded.py`
- Modify: `/home/chrism/git/starpose/src/starpose/qc/__init__.py`
- Test: `/home/chrism/git/starpose/tests/test_qc_segmentation_grounded.py` (create)

- [ ] **Step 1: Create the new module — copy the source, then apply 4 edits**

Copy the entire current contents of `/mnt/work/git/dapidl/src/dapidl/qc/segmentation_grounded.py` to `/home/chrism/git/starpose/src/starpose/qc/segmentation_grounded.py`, then make exactly these four changes:

(a) Replace the module docstring (lines 1-6) with:
```python
"""Segmentation-grounded per-nucleus QC: reject obviously broken training patches.

Pure scoring functions (GPU-free) operate on a patch + a StarDist segmentation
(label mask + per-object prob). SegmentationGroundedScorer supplies the
segmentation via the starpose StarDist backend (one StarDist path).
"""
```

(b) Confirm the ABC import (old line 16) reads exactly this and leave it unchanged — the ABC already lives in starpose, so it resolves identically from the new location:
```python
from starpose.qc.base import NormRef, QualityScore, QualityScorer
```

(c) In `class SegmentationGroundedScorer`, DELETE the `_get_model` method entirely and REPLACE `__init__` + `_segment` with the adapter-backed versions:
```python
    def __init__(self, cfg: SegQCConfig | None = None, gpu: bool = True,
                 pixel_size: float = 0.2125):
        self.cfg = cfg or SegQCConfig()
        self.gpu = gpu
        self.pixel_size = pixel_size
        self._adapter = None

    @property
    def name(self) -> str:
        return "segmentation_grounded"

    def _segmenter(self):
        if self._adapter is None:
            from starpose.methods import get_method
            self._adapter = get_method("stardist", gpu=self.gpu)
        return self._adapter

    def _segment(self, patch: np.ndarray):
        """Return (label mask int32, per-object prob array) via the StarDist
        backend. Label k -> prob[k-1]. One StarDist path for the whole repo."""
        res = self._segmenter().segment(patch, pixel_size=self.pixel_size)
        masks = np.asarray(res.masks, dtype=np.int32)
        probs = np.asarray(res.probs, dtype=float) if res.probs is not None else np.array([])
        return masks, probs
```
Remove the now-unused `import os` ONLY if nothing else in the file uses it (the deleted `_get_model` was the sole user — verify with grep; if so, delete the `import os` line).

(d) Add an `__all__` at the end of the module so the dapidl shim can re-export cleanly:
```python
__all__ = [
    "SegQCConfig", "CenterNucleus", "select_center_nucleus", "structure_raw",
    "structure_score", "centeredness_score", "touches_edge", "area_um2",
    "dominant_central_fraction", "objectness_metrics", "interior_cov",
    "brenner_focus", "glcm_texture", "score_from_segmentation", "decide_broken",
    "SegmentationGroundedScorer",
]
```

- [ ] **Step 2: Update the qc package exports**

In `/home/chrism/git/starpose/src/starpose/qc/__init__.py`, add the scorer + config + decision fn to the imports and `__all__`:
```python
"""starpose quality-control scoring."""

from starpose.qc.base import NormRef, QualityScore, QualityScorer
from starpose.qc.classical import ClassicalQualityScorer
from starpose.qc.segmentation_grounded import (
    SegmentationGroundedScorer,
    SegQCConfig,
    decide_broken,
    score_from_segmentation,
    select_center_nucleus,
)

__all__ = [
    "NormRef", "QualityScore", "QualityScorer", "ClassicalQualityScorer",
    "SegmentationGroundedScorer", "SegQCConfig", "decide_broken",
    "score_from_segmentation", "select_center_nucleus",
]
```

- [ ] **Step 3: Write the test (GPU-free: pure fns + fake-adapter wiring)**

Create `/home/chrism/git/starpose/tests/test_qc_segmentation_grounded.py`:
```python
import numpy as np

from starpose.qc.segmentation_grounded import (
    SegQCConfig, SegmentationGroundedScorer, decide_broken, select_center_nucleus,
)


def _disc(cx, cy, r, shape=(64, 64)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r)


def test_select_center_nucleus_picks_central_object():
    masks = np.zeros((64, 64), np.int32)
    masks[_disc(32, 32, 8)] = 1
    masks[_disc(5, 5, 3)] = 2
    cn = select_center_nucleus(masks, np.array([0.9, 0.8]), SegQCConfig())
    assert cn is not None and cn.label == 1


def test_select_center_nucleus_none_when_empty():
    assert select_center_nucleus(np.zeros((64, 64), np.int32), np.array([]), SegQCConfig()) is None


def test_decide_broken_flags_no_nucleus():
    from starpose.qc.base import QualityScore
    qs = QualityScore(focus_score=0.0, detection_score=0.0, qc_score=0.0,
                      metrics={"has_nucleus": 0.0})
    broken, reason = decide_broken(qs, SegQCConfig())
    assert broken and reason == "no_nucleus"


def test_scorer_segment_uses_starpose_backend(monkeypatch):
    """_segment must go through the starpose StarDist adapter, returning probs."""
    labels = np.zeros((64, 64), np.int32); labels[_disc(32, 32, 8)] = 1
    probs = np.array([0.95])

    class _FakeAdapter:
        def segment(self, patch, pixel_size=0.2125):
            from starpose.types import SegmentationResult
            return SegmentationResult(masks=labels, centroids=np.empty((0, 2)),
                                      n_cells=1, runtime=0.0, method_name="stardist",
                                      probs=probs)

    scorer = SegmentationGroundedScorer(gpu=False)
    monkeypatch.setattr(scorer, "_segmenter", lambda: _FakeAdapter())
    m, p = scorer._segment(np.zeros((64, 64), np.uint16))
    assert np.array_equal(m, labels) and np.allclose(p, probs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Confirm no direct stardist import remains in the moved module**

Run: `cd /home/chrism/git/starpose && ! grep -rn "from stardist" src/starpose/qc/segmentation_grounded.py`
Expected: exit 0 (no match) — the scorer no longer imports StarDist directly.

- [ ] **Step 6: Commit (starpose)**
```bash
cd /home/chrism/git/starpose
git add src/starpose/qc/segmentation_grounded.py src/starpose/qc/__init__.py tests/test_qc_segmentation_grounded.py
git commit -m "feat(qc): move SegmentationGroundedScorer into starpose.qc (StarDist via adapter)"
```

---

## Task 4: dapidl back-compat shim (dapidl)

Replace dapidl's module with a re-export so every existing import keeps working.

**Files:**
- Modify (replace contents): `/mnt/work/git/dapidl/src/dapidl/qc/segmentation_grounded.py`
- Test: `/mnt/work/git/dapidl/tests/test_qc_segmentation_grounded.py` (keep; it now exercises the shim)

- [ ] **Step 1: Verify dapidl currently has starpose ≥ 0.3.0 available**

Run: `cd /mnt/work/git/dapidl && uv run python -c "from starpose.qc.segmentation_grounded import SegmentationGroundedScorer; print('ok')"`
Expected: `ok` (the editable starpose dep already points at the local repo).

- [ ] **Step 2: Replace the dapidl module with a shim**

Overwrite `/mnt/work/git/dapidl/src/dapidl/qc/segmentation_grounded.py` with:
```python
"""DEPRECATED shim — moved to starpose.qc.segmentation_grounded (starpose 0.3.0).

Kept so existing dapidl imports keep working; import from starpose.qc directly
in new code. Removed in a later cleanup.
"""
from starpose.qc.segmentation_grounded import *  # noqa: F401,F403
from starpose.qc.segmentation_grounded import __all__  # noqa: F401
```

- [ ] **Step 3: Run the existing dapidl QC test through the shim**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS — `from dapidl.qc.segmentation_grounded import SegQCConfig, select_center_nucleus` resolves via the shim.

- [ ] **Step 4: Commit (dapidl)**
```bash
cd /mnt/work/git/dapidl
git add src/dapidl/qc/segmentation_grounded.py
git commit -m "refactor(qc): dapidl segmentation_grounded becomes a starpose shim"
```

---

## Task 5: Update dapidl primary call sites (dapidl)

Point the two first-party call sites at `starpose.qc` directly (the shim covers scripts/tests).

**Files:**
- Modify: `/mnt/work/git/dapidl/src/dapidl/pipeline/steps/quality_control_seg.py:24`
- Modify: `/mnt/work/git/dapidl/scripts/seg_qc_smoke.py:27-29`

- [ ] **Step 1: Update the pipeline step import**

In `/mnt/work/git/dapidl/src/dapidl/pipeline/steps/quality_control_seg.py`, change line 24 from:
```python
from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer, decide_broken
```
to:
```python
from starpose.qc import SegmentationGroundedScorer, decide_broken
```

- [ ] **Step 2: Update the smoke script import**

In `/mnt/work/git/dapidl/scripts/seg_qc_smoke.py`, change the import block (lines 27-30) from `from dapidl.qc.segmentation_grounded import (...)` to:
```python
from starpose.qc import (  # noqa: E402
    SegmentationGroundedScorer,
    decide_broken,
)
```

- [ ] **Step 3: Verify the pipeline step imports cleanly**

Run: `cd /mnt/work/git/dapidl && uv run python -c "import dapidl.pipeline.steps.quality_control_seg as m; print('ok', m.SegmentationGroundedScorer.__module__)"`
Expected: `ok starpose.qc.segmentation_grounded`

- [ ] **Step 4: Commit (dapidl)**
```bash
cd /mnt/work/git/dapidl
git add src/dapidl/pipeline/steps/quality_control_seg.py scripts/seg_qc_smoke.py
git commit -m "refactor(qc): import SegmentationGroundedScorer from starpose.qc"
```

---

## Task 6: Guardrail — forbid new direct `stardist` imports in dapidl (dapidl)

Lock the seam: dapidl must not import StarDist directly (segmentation lives in starpose).

**Files:**
- Test: `/mnt/work/git/dapidl/tests/test_no_direct_stardist_import.py` (create)

- [ ] **Step 1: Write the test**

Create `/mnt/work/git/dapidl/tests/test_no_direct_stardist_import.py`:
```python
"""The seam: dapidl consumes segmentation via starpose, never StarDist directly."""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "dapidl"
PATTERN = re.compile(r"^\s*(from\s+stardist|import\s+stardist)\b", re.MULTILINE)


def test_no_direct_stardist_imports_in_dapidl_src():
    offenders = []
    for py in SRC.rglob("*.py"):
        if PATTERN.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(SRC)))
    assert not offenders, f"dapidl must import StarDist via starpose, not directly: {offenders}"
```

- [ ] **Step 2: Run the test to verify it passes now**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_no_direct_stardist_import.py -q`
Expected: PASS — after Tasks 4-5 no `dapidl/src` module imports stardist directly.

(If it FAILS, the offending files it lists still import StarDist directly and must be routed through `starpose.qc`/`starpose.create` first.)

- [ ] **Step 3: Commit (dapidl)**
```bash
cd /mnt/work/git/dapidl
git add tests/test_no_direct_stardist_import.py
git commit -m "test(qc): forbid direct stardist imports in dapidl src"
```

---

## Task 7: Version bump + dependency refresh

**Files:**
- Modify: `/home/chrism/git/starpose/pyproject.toml:3`
- Refresh: `/mnt/work/git/dapidl/uv.lock`

- [ ] **Step 1: Bump starpose version**

In `/home/chrism/git/starpose/pyproject.toml`, change `version = "0.2.0"` to `version = "0.3.0"`.

- [ ] **Step 2: Run the full starpose test suite**

Run: `cd /home/chrism/git/starpose && uv run pytest -q`
Expected: PASS (all existing tests + the new ones).

- [ ] **Step 3: Refresh dapidl's lock + run the affected dapidl tests**

Run:
```bash
cd /mnt/work/git/dapidl && uv lock && uv run pytest tests/test_qc_segmentation_grounded.py tests/test_no_direct_stardist_import.py -q
```
Expected: PASS.

- [ ] **Step 4: Commit (both repos)**
```bash
cd /home/chrism/git/starpose && git add pyproject.toml && git commit -m "chore: bump starpose 0.2.0 -> 0.3.0 (qc consolidation)"
cd /mnt/work/git/dapidl && git add uv.lock && git commit -m "chore: refresh lock for starpose 0.3.0"
```

---

## Final verification

- [ ] Run both suites: `cd /home/chrism/git/starpose && uv run pytest -q` and `cd /mnt/work/git/dapidl && uv run pytest -q`. Both green.
- [ ] `grep -rn "from stardist\|import stardist" /mnt/work/git/dapidl/src` returns nothing.
- [ ] `cd /mnt/work/git/dapidl && uv run python -c "from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer as S; print(S.__module__)"` prints `starpose.qc.segmentation_grounded` (shim works).
