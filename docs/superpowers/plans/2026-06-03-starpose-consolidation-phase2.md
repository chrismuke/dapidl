# StarPose Consolidation — Phase 2 (Backend Hygiene) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden starpose's backends — fix the broken InstanSeg adapter, surface (informational) license metadata, and add a density-estimation mechanism + configurable dispatch default — without changing default behavior.

**Architecture:** Three independent changes in the starpose repo (branch `feat/qc-consolidation`): (1) rewrite `methods/instanseg.py` to the current `instanseg` API with model caching + nucleus-channel extraction, validated by a real GPU smoke; (2) thread a `license` string through the registry → `MethodInfo` → `starpose methods` CLI; (3) add `density.estimate_nucleus_density()` + make the dispatcher's DAPI default configurable, with density routing wired but **inert by default** (threshold `None`) so Phase 3 can set the threshold from benchmark data.

**Tech stack:** Python 3.12, `uv`, pytest, numpy/scipy/scikit-image, torch, instanseg-torch (to be installed). starpose at `/home/chrism/git/starpose`.

**Do NOT touch:** `starpose/methods/{cellotype,joint}.py`, `dapidl/training/instance/*`.

---

## ⚠️ EXECUTION PREREQUISITE (read before Task 2)

`src/starpose/methods/__init__.py` currently carries **uncommitted joint/cellotype WIP** in the working tree (a `from starpose.methods.joint import ...` line + `__all__` change; plus untracked `cellotype.py`/`joint.py`/`test_joint_abc.py`). **Task 2 edits this file.** Before executing Task 2, the WIP owner must either commit or `git stash` that WIP so the license edits don't entangle with it. Tasks 1 and 3 touch different files and are unaffected. If the WIP cannot be moved, stage Task 2's hunks selectively (`git add -p src/starpose/methods/__init__.py`) — but committing/stashing the WIP first is strongly preferred.

---

## Task 1: Fix the InstanSeg adapter

The adapter uses the **old API** (`instanseg.inference_class.Instanseg`), **reloads the model every call**, and **squeezes the 2-channel nuclei+cells output** as one label map (the "2 cells in dense" bug). instanseg-torch is also **not installed**.

**Files:**
- Modify: `/home/chrism/git/starpose/src/starpose/methods/instanseg.py`
- Test: `/home/chrism/git/starpose/tests/test_instanseg_adapter.py` (create)

- [ ] **Step 1: Install instanseg-torch into the starpose env**

Run: `cd /home/chrism/git/starpose && uv pip install "instanseg-torch>=0.1"`
Expected: installs `instanseg` (+ deps). Confirm: `uv run python -c "import instanseg; print('ok', instanseg.__version__)"` prints `ok <version>`.

- [ ] **Step 2: Inspect the real API (informs Step 4's exact call)**

Run:
```bash
cd /home/chrism/git/starpose && uv run python -c "
import inspect
from instanseg import InstanSeg
print('init:', inspect.signature(InstanSeg.__init__))
print('eval_small_image:', inspect.signature(InstanSeg.eval_small_image))
"
```
Note the `eval_small_image` parameter names (esp. whether `pixel_size` is positional/kw) and return shape. The implementation in Step 4 uses a shape-robust extractor, so minor signature differences are tolerated; record the exact `__init__`/`eval_small_image` call for the model + pixel size.

- [ ] **Step 3: Write the failing unit test (GPU-free, mocked model)**

Create `/home/chrism/git/starpose/tests/test_instanseg_adapter.py`:
```python
import numpy as np
import pytest

from starpose.methods.instanseg import InstanSegAdapter, _to_nucleus_labels


def test_to_nucleus_labels_extracts_channel0_from_2ch_output():
    # InstanSeg fluorescence model returns (nuclei, cells); we want nuclei (ch 0).
    out = np.zeros((1, 2, 8, 8), dtype=np.int32)
    out[0, 0, 1:4, 1:4] = 1          # nucleus instance in channel 0
    out[0, 0, 5:7, 5:7] = 2
    out[0, 1, :, :] = 9              # channel 1 (cells) must be ignored
    labels = _to_nucleus_labels(out)
    assert labels.shape == (8, 8)
    assert set(np.unique(labels)) == {0, 1, 2}   # NOT collapsed, NOT the cell channel


def test_to_nucleus_labels_handles_plain_2d():
    arr = np.array([[0, 1], [2, 2]], dtype=np.int32)
    assert np.array_equal(_to_nucleus_labels(arr), arr)


def test_run_extracts_nucleus_channel_via_mocked_model(monkeypatch):
    """_run must return the channel-0 (nucleus) label map, never the cell channel
    or a squeezed 2-channel array. This is the exact bug being fixed, tested
    GPU-free by mocking the model (instanseg need not be installed)."""
    out = np.zeros((1, 2, 8, 8), dtype=np.int32)
    out[0, 0, 1:4, 1:4] = 1          # nucleus instance in channel 0
    out[0, 1, :, :] = 9              # cell channel -- must be ignored

    class _FakeModel:
        def eval_small_image(self, image, pixel_size=None, **kw):
            return out

    adapter = InstanSegAdapter(gpu=False)
    monkeypatch.setattr(adapter, "_get_model", lambda: _FakeModel())
    labels = adapter._run(np.zeros((8, 8), np.uint16), 0.2125)
    assert labels.shape == (8, 8)
    assert set(np.unique(labels)) == {0, 1}   # channel 0 only, NOT 9 (cells), NOT collapsed
```

- [ ] **Step 4: Rewrite the adapter**

Replace the body of `/home/chrism/git/starpose/src/starpose/methods/instanseg.py` (keep the module docstring) with:
```python
"""InstanSeg segmentation adapter."""

import time

import numpy as np

from starpose.methods.base import Segmenter
from starpose.methods.cellpose import centroids_from_masks, measure_gpu_memory, reset_gpu_memory
from starpose.types import SegmentationResult

# InstanSeg fluorescence model emits a multi-channel label stack; nuclei = channel 0.
_MODEL_NAME = "fluorescence_nuclei_and_cells"
_NUCLEUS_CHANNEL = 0


def _to_nucleus_labels(output) -> np.ndarray:
    """Extract a 2D nucleus label map from whatever eval_small_image returns.

    Handles (B, C, H, W), (C, H, W), (1, H, W) and (H, W); torch tensors or arrays.
    For multi-channel outputs the nucleus channel (0) is selected — NEVER squeezed
    into a 2-channel array (the old bug that collapsed everything to ~2 labels).
    """
    t = output[0] if isinstance(output, (tuple, list)) else output
    try:
        import torch
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(t)
    while arr.ndim > 3:               # drop leading batch dims
        arr = arr[0]
    if arr.ndim == 3:                 # (C, H, W) -> nucleus channel
        arr = arr[_NUCLEUS_CHANNEL]
    return arr.astype(np.int32)


class InstanSegAdapter(Segmenter):
    """InstanSeg instance segmentation model (DAPI nuclei)."""

    def __init__(self, gpu: bool = True, **kwargs):
        self.gpu = gpu
        self._kwargs = kwargs
        self._model = None

    @property
    def name(self) -> str:
        return "instanseg"

    def _get_model(self):
        if self._model is None:
            import torch
            from instanseg import InstanSeg

            device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
            self._model = InstanSeg(_MODEL_NAME, verbosity=0, device=device)
        return self._model

    def _run(self, image: np.ndarray, pixel_size: float) -> np.ndarray:
        model = self._get_model()
        output = model.eval_small_image(image, pixel_size)
        return _to_nucleus_labels(output)

    def segment(self, image: np.ndarray, pixel_size: float = 0.2125) -> SegmentationResult:
        mem_before = measure_gpu_memory()
        t0 = time.time()
        masks = self._run(image, pixel_size)
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
        )
```
If Step 2 showed `eval_small_image` needs `pixel_size` as a keyword or a different name, adjust the `model.eval_small_image(image, pixel_size)` call accordingly; the rest is signature-independent.

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_instanseg_adapter.py -q`
Expected: PASS (3 passed). (These are GPU-free; they validate channel extraction + caching, the heart of the bug.)

- [ ] **Step 6: Write + run the real-model GPU smoke (catches the 2-cells pathology)**

Append to `/home/chrism/git/starpose/tests/test_instanseg_adapter.py`:
```python
def _has_instanseg_gpu():
    try:
        import torch
        import instanseg  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _has_instanseg_gpu(), reason="needs instanseg + GPU")
def test_real_model_counts_many_nuclei_on_dense_synthetic_field():
    """Ground truth: a field of ~30 bright blobs must yield many nuclei, not ~2.

    This is exactly the failure the old adapter produced (2 cells in dense).
    Downloads the bioimage.io model on first run.
    """
    rng = np.random.default_rng(0)
    img = np.zeros((256, 256), dtype=np.uint16)
    yy, xx = np.ogrid[:256, :256]
    centers = [(int(r), int(c)) for r in range(24, 232, 40) for c in range(24, 232, 40)]
    for (cy, cx) in centers:                       # 6x6 = 36 well-separated blobs
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 64] = 3000 + int(rng.integers(0, 500))
    res = InstanSegAdapter(gpu=True).segment(img, pixel_size=0.2125)
    assert res.n_cells >= 15, f"expected many nuclei, got {res.n_cells} (the old 2-cells bug?)"
```

Run: `cd /home/chrism/git/starpose && nvidia-smi --query-gpu=memory.free --format=csv,noheader && uv run pytest tests/test_instanseg_adapter.py -q`
Expected: 4 passed (the smoke downloads the model first run, then asserts ≥15 nuclei on the 36-blob field).

- [ ] **Step 7: Lint + commit**
```bash
cd /home/chrism/git/starpose
uv run ruff check src/starpose/methods/instanseg.py tests/test_instanseg_adapter.py
git add src/starpose/methods/instanseg.py tests/test_instanseg_adapter.py
git commit -m "fix(instanseg): current API + model caching + nucleus-channel extraction

Old adapter used the removed instanseg.inference_class API, rebuilt the model
every call, and squeezed the 2-channel nuclei+cells output into one map (the
'2 cells in dense' bug). Now caches the model and extracts the nucleus channel;
a real-model GPU smoke asserts many nuclei on a dense synthetic field."
```
(Note: instanseg-torch was installed into the venv; it is already declared in pyproject's `[project.optional-dependencies] instanseg`, so no pyproject change is needed.)

---

## Task 2: Informational license metadata in the registry

⚠️ See the EXECUTION PREREQUISITE above — resolve the joint WIP in `methods/__init__.py` first.

Add a `license` string to the registry, surfaced by `list_methods()` and `starpose methods`. **No gating** (academic use).

**Files:**
- Modify: `/home/chrism/git/starpose/src/starpose/types.py` (`MethodInfo`)
- Modify: `/home/chrism/git/starpose/src/starpose/methods/__init__.py`
- Modify: `/home/chrism/git/starpose/src/starpose/cli.py` (`methods_cmd`)
- Test: `/home/chrism/git/starpose/tests/test_method_license.py` (create)

- [ ] **Step 1: Write the failing test**

Create `/home/chrism/git/starpose/tests/test_method_license.py`:
```python
from starpose.methods import list_methods, register
from starpose.types import MethodInfo


def test_methodinfo_has_license_field():
    mi = MethodInfo(name="x", description="d", available=True,
                    supports_cell_boundaries=False, license="MIT")
    assert mi.license == "MIT"


def test_register_records_license_and_list_methods_surfaces_it():
    register("_lic_probe", None, "probe", "", license="Apache-2.0")
    by_name = {m.name: m for m in list_methods()}
    assert by_name["_lic_probe"].license == "Apache-2.0"


def test_builtin_licenses_are_populated():
    by_name = {m.name: m for m in list_methods()}
    assert by_name["stardist"].license == "BSD-3-Clause"
    assert by_name["instanseg"].license == "Apache-2.0"
    assert "Commons-Clause" in by_name["cellvit"].license
    assert by_name["topological"].license == "MIT"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_method_license.py -q`
Expected: FAIL — `MethodInfo` has no `license`; `register` takes no `license`.

- [ ] **Step 3: Add `license` to `MethodInfo`**

In `/home/chrism/git/starpose/src/starpose/types.py`, the `MethodInfo` dataclass — add a field:
```python
@dataclass
class MethodInfo:
    """Metadata about an available segmentation method."""

    name: str
    description: str
    available: bool
    supports_cell_boundaries: bool
    license: str = "unknown"
```

- [ ] **Step 4: Thread `license` through the registry**

In `/home/chrism/git/starpose/src/starpose/methods/__init__.py`:

(a) Registry value becomes a 4-tuple; update the type comment and `register`:
```python
# Registry: name -> (class | None, description, install_hint, license)
_METHODS: dict[str, tuple[type | None, str, str, str]] = {}


def register(name: str, cls: type | None, description: str,
             install_hint: str = "", license: str = "unknown") -> None:
    """Register a segmentation method."""
    _METHODS[name] = (cls, description, install_hint, license)
```

(The `license` parameter shadows a Python builtin. If `ruff` flags `A002` on the `def register(...)` line in Step 7, add `# noqa: A002` to it; if A002 is not enabled in the repo's ruff config, leave it as-is — do not add an unused noqa.)

(b) `get_method` unpacks 4:
```python
    cls, _desc, hint, _license = _METHODS[name]
```

(c) `list_methods` unpacks 4 and passes license:
```python
    for name, (cls, desc, _hint, license) in sorted(_METHODS.items()):
        available = cls is not None
        boundaries = False
        if available:
            try:
                boundaries = cls.supports_cell_boundaries.fget(None)  # type: ignore[union-attr]
            except Exception:
                boundaries = False
        result.append(
            MethodInfo(
                name=name,
                description=desc,
                available=available,
                supports_cell_boundaries=boundaries,
                license=license,
            )
        )
```

(d) In `_register_builtins`, add `license=...` to each built-in registration (both the success and the ImportError-fallback `register(...)` calls for a method get the SAME license):
- all three `cellpose*` → `license="BSD-3-Clause"`
- `stardist` → `license="BSD-3-Clause"`
- `instanseg` → `license="Apache-2.0"`
- `cellvit` → `license="Apache-2.0 + Commons-Clause (non-commercial)"`
- `adaptive`, `majority`, `topological` → `license="MIT"`

Example (cellpose success branch):
```python
        register("cellpose", CellposeSAM, "Cellpose SAM default model", license="BSD-3-Clause")
        register("cellpose_cyto3", CellposeCyto3, "Cellpose cyto3 model", license="BSD-3-Clause")
        register("cellpose_nuclei", CellposeNuclei, "Cellpose nuclei model", license="BSD-3-Clause")
```
and the matching ImportError branch adds `license="BSD-3-Clause"` to each fallback `register(...)`. Apply the analogous `license=...` to stardist/instanseg/cellvit (both branches) and the three consensus registrations.

- [ ] **Step 5: Add a License column to the CLI**

In `/home/chrism/git/starpose/src/starpose/cli.py`, `methods_cmd`:
```python
    table = Table(title="Available Methods")
    table.add_column("Name", style="bold")
    table.add_column("Available", justify="center")
    table.add_column("License")
    table.add_column("Description")
    for m in list_methods():
        status = "[green]yes[/green]" if m.available else "[red]no[/red]"
        table.add_row(m.name, status, m.license, m.description)
    console.print(table)
```

- [ ] **Step 6: Run tests + the CLI smoke**

Run:
```bash
cd /home/chrism/git/starpose
uv run pytest tests/test_method_license.py tests/test_methods.py -q
uv run starpose methods
```
Expected: tests PASS; the `starpose methods` table shows a License column.

- [ ] **Step 7: Lint + commit** (stage ONLY the four intended files — not the WIP)
```bash
cd /home/chrism/git/starpose
uv run ruff check src/starpose/types.py src/starpose/methods/__init__.py src/starpose/cli.py tests/test_method_license.py
git add src/starpose/types.py src/starpose/methods/__init__.py src/starpose/cli.py tests/test_method_license.py
git commit -m "feat(registry): informational license metadata (academic use, no gating)"
```

---

## Task 3: Density-estimation mechanism + configurable dispatch default

Build the density tooling and de-hardcode the dispatch default, but keep behavior unchanged (density routing inert until Phase 3 sets a threshold).

**Files:**
- Create: `/home/chrism/git/starpose/src/starpose/density.py`
- Modify: `/home/chrism/git/starpose/src/starpose/dispatch.py`
- Test: `/home/chrism/git/starpose/tests/test_density.py` (create)
- Test: `/home/chrism/git/starpose/tests/test_dispatch_density.py` (create)

- [ ] **Step 1: Write the density-estimator test**

Create `/home/chrism/git/starpose/tests/test_density.py`:
```python
import numpy as np

from starpose.density import estimate_nucleus_density


def _field(n_per_side, shape=256, r=8, val=3000):
    img = np.zeros((shape, shape), dtype=np.uint16)
    yy, xx = np.ogrid[:shape, :shape]
    step = shape // (n_per_side + 1)
    for i in range(1, n_per_side + 1):
        for j in range(1, n_per_side + 1):
            cy, cx = i * step, j * step
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = val
    return img


def test_density_zero_on_blank_image():
    assert estimate_nucleus_density(np.zeros((128, 128), np.uint16)) == 0.0


def test_density_increases_with_blob_count():
    sparse = estimate_nucleus_density(_field(3))    # 9 blobs
    dense = estimate_nucleus_density(_field(8))      # 64 blobs
    assert dense > sparse > 0.0


def test_density_is_per_area_and_pixel_size_aware():
    # Same blob layout, finer pixel_size -> fewer nuclei per physical area.
    img = _field(5)
    coarse = estimate_nucleus_density(img, pixel_size=0.5)
    fine = estimate_nucleus_density(img, pixel_size=0.1)
    assert coarse > fine > 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_density.py -q`
Expected: FAIL — `starpose.density` does not exist.

- [ ] **Step 3: Implement the density estimator**

Create `/home/chrism/git/starpose/src/starpose/density.py`:
```python
"""Fast, GPU-free nucleus-density estimate for dispatch routing.

Threshold (Otsu) + connected-components count over physical area. This is a
cheap PROXY (not a segmentation) used only to choose a nucleus method; the
thresholds that turn it into a routing decision are set in Phase 3 from the
PQ/F1 benchmark.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def estimate_nucleus_density(image: np.ndarray, pixel_size: float = 0.2125) -> float:
    """Approximate nuclei per 1000 um^2.

    Returns 0.0 for a blank/flat image.
    """
    img = np.asarray(image)
    if img.size == 0 or float(img.max()) - float(img.min()) < 1e-6:
        return 0.0
    try:
        thr = threshold_otsu(img)
    except Exception:
        return 0.0
    fg = img > thr
    if not fg.any():
        return 0.0
    _labels, n = ndimage.label(fg)
    area_um2 = img.shape[0] * img.shape[1] * (pixel_size ** 2)
    if area_um2 <= 0:
        return 0.0
    return float(n) / area_um2 * 1000.0
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_density.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the dispatcher-routing test**

Create `/home/chrism/git/starpose/tests/test_dispatch_density.py`:
```python
import numpy as np

from starpose.dispatch import AdaptiveDispatcher
from starpose.types import ModalityBundle


def _dapi_bundle():
    # a small DAPI image; density routing is what we test, not the value
    img = np.zeros((64, 64), dtype=np.uint16)
    img[20:30, 20:30] = 3000
    return ModalityBundle(dapi=img, platform="xenium")


def test_default_dapi_choice_is_topological_unchanged():
    spec = AdaptiveDispatcher(gpu=False).plan(_dapi_bundle())
    assert spec.nucleus_method == "topological"


def test_configurable_dapi_default():
    spec = AdaptiveDispatcher(gpu=False, dapi_default="stardist").plan(_dapi_bundle())
    assert spec.nucleus_method == "stardist"


def test_density_routing_when_threshold_set(monkeypatch):
    import starpose.dispatch as d
    # force a high density so the dense branch is taken
    monkeypatch.setattr(d, "estimate_nucleus_density", lambda *a, **k: 999.0)
    disp = AdaptiveDispatcher(gpu=False, dapi_default="stardist",
                              density_threshold=10.0, dense_method="cellpose_nuclei")
    assert disp.plan(_dapi_bundle()).nucleus_method == "cellpose_nuclei"
    # below threshold -> sparse default
    monkeypatch.setattr(d, "estimate_nucleus_density", lambda *a, **k: 1.0)
    assert disp.plan(_dapi_bundle()).nucleus_method == "stardist"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_dispatch_density.py -q`
Expected: FAIL — dispatcher takes no `dapi_default`/`density_threshold`/`dense_method`; `estimate_nucleus_density` not imported there.

- [ ] **Step 7: Wire density + config into the dispatcher**

In `/home/chrism/git/starpose/src/starpose/dispatch.py`:

(a) Add the import near the top:
```python
from starpose.density import estimate_nucleus_density
```

(b) Extend `AdaptiveDispatcher.__init__` to accept the new (defaulted) params and store them:
```python
    def __init__(
        self,
        gpu: bool = True,
        nucleus_method: str | None = None,
        expansion_method: str | None = None,
        dapi_default: str = "topological",
        density_threshold: float | None = None,
        dense_method: str = "cellpose_nuclei",
    ):
        self.gpu = gpu
        self._force_nucleus = nucleus_method
        self._force_expansion = expansion_method
        self._dapi_default = dapi_default
        self._density_threshold = density_threshold
        self._dense_method = dense_method
```

(c) In `plan()`, compute the nucleus method with optional density routing. Replace the line `nucleus = self._force_nucleus or self._pick_nucleus_method(available)` with:
```python
        if self._force_nucleus:
            nucleus = self._force_nucleus
        else:
            density = None
            if self._density_threshold is not None and modalities.dapi is not None:
                density = estimate_nucleus_density(modalities.dapi, modalities.pixel_size)
            nucleus = self._pick_nucleus_method(available, density)
```

(d) Update `_pick_nucleus_method` to use the configurable default + density:
```python
    def _pick_nucleus_method(self, available: set[str], density: float | None = None) -> str:
        """Choose the best nucleus segmentation method.

        For DAPI: density routing applies only when a threshold is configured
        (Phase 3 sets it from the benchmark); otherwise the configurable default
        (``dapi_default``, historically 'topological') is used.
        """
        if "dapi" in available:
            if (self._density_threshold is not None and density is not None
                    and density >= self._density_threshold):
                return self._dense_method
            return self._dapi_default
        if "he" in available:
            return "cellvit"
        return "cellpose"  # fallback
```

- [ ] **Step 8: Run both dispatch + density tests**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_density.py tests/test_dispatch_density.py -q`
Expected: PASS (6 passed). Default DAPI choice is still `topological` (behavior unchanged).

- [ ] **Step 9: Lint + commit**
```bash
cd /home/chrism/git/starpose
uv run ruff check src/starpose/density.py src/starpose/dispatch.py tests/test_density.py tests/test_dispatch_density.py
git add src/starpose/density.py src/starpose/dispatch.py tests/test_density.py tests/test_dispatch_density.py
git commit -m "feat(dispatch): density estimator + configurable DAPI default (routing inert until Phase 3)"
```

---

## Final verification

- [ ] Full suite: `cd /home/chrism/git/starpose && uv run pytest -q` — green (incl. the gated InstanSeg smoke if GPU present).
- [ ] `uv run starpose methods` shows a populated License column.
- [ ] `AdaptiveDispatcher(gpu=False).plan(<dapi bundle>).nucleus_method == "topological"` — default behavior unchanged.
- [ ] Phase 2 commits present in `git log` on `feat/qc-consolidation`.
