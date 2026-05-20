# Nucleus QC + Inspection Montage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast classical quality scorer for DAPI nucleus patches (in starpose) plus a re-runnable dapidl pass that scores existing datasets, writes a sidecar, and logs per-class worst-first montages to ClearML.

**Architecture:** starpose gains a `qc/` package — a `QualityScorer` ABC and a `ClassicalQualityScorer` (focus + detection metrics, per-slide relative + absolute-floor scoring). dapidl gains a `qc/` package (patch reader + montage builder) and a `quality_control` pipeline step + `dapidl qc` CLI command that orchestrates read → group-by-slide → sample-fit reference → score → write `qc/qc_scores.parquet` → build montages → log to ClearML. Nothing is dropped; `metadata.parquet` is never modified.

**Tech Stack:** Python 3.10+, numpy, scipy.ndimage, scikit-image (focus/detection metrics), polars (sidecar parquet), matplotlib (montage), lmdb + zarr (patch read), typer (starpose CLI), click (dapidl CLI), ClearML (logging), pytest + typer.testing.CliRunner.

**Spec:** `docs/superpowers/specs/2026-05-20-nucleus-qc-design.md`

**Two repos:**
- starpose: `/home/chrism/git/starpose` (editable-installed in dapidl's venv, so new `starpose.qc` modules are importable from dapidl immediately — no reinstall).
- dapidl: `/mnt/work/git/dapidl`.
- Run starpose tests from the starpose dir; dapidl tests from the dapidl dir. All commands use `uv run`.

**Slide grouping note:** per-slide normalization needs a per-patch slide key. The derived datasets ship `slide_stats.json` (a JSON list of `{"source": ..., "n_written": N}` in patch order) — reconstruct the grouping from that (safe JSON). Do NOT load the object-dtype `sources.npy` (requires `allow_pickle`, which the environment's security hook blocks). Single-slide datasets with no `slide_stats.json` form one group.

---

## Task 1: starpose `QualityScore` + `QualityScorer` ABC + `NormRef`

**Repo:** starpose (`/home/chrism/git/starpose`)

**Files:**
- Create: `src/starpose/qc/__init__.py`
- Create: `src/starpose/qc/base.py`
- Test: `tests/test_qc_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_base.py`:

```python
"""Tests for starpose.qc.base."""

import pytest

from starpose.qc.base import NormRef, QualityScore, QualityScorer


def test_quality_score_fields():
    qs = QualityScore(focus_score=0.9, detection_score=0.8, qc_score=0.8,
                      metrics={"var_laplacian": 123.0})
    assert qs.qc_score == 0.8
    assert qs.metrics["var_laplacian"] == 123.0


def test_normref_fields():
    ref = NormRef(varlap_p90=100.0, tenengrad_p90=50.0)
    assert ref.varlap_p90 == 100.0


def test_qualityscorer_is_abstract():
    with pytest.raises(TypeError):
        QualityScorer()  # abstract, cannot instantiate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'starpose.qc'`

- [ ] **Step 3: Write minimal implementation**

Create `src/starpose/qc/base.py`:

```python
"""Quality-control scoring of nucleus image patches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QualityScore:
    """Per-patch quality scores. All scores are 0..1, higher = better."""

    focus_score: float       # higher = sharper
    detection_score: float   # higher = clearer real nucleus
    qc_score: float          # combined (see ClassicalQualityScorer)
    metrics: dict[str, float] = field(default_factory=dict)  # raw named metrics


@dataclass(frozen=True)
class NormRef:
    """Per-slide normalization reference fitted from a sample of patches."""

    varlap_p90: float
    tenengrad_p90: float


class QualityScorer(ABC):
    """Base class for patch quality scorers (mirrors methods/base.py)."""

    @abstractmethod
    def score_batch(
        self, patches: np.ndarray, ref: NormRef | None = None
    ) -> list[QualityScore]:
        """Score an (N, H, W) batch of raw patches."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this scorer."""
        ...
```

Create `src/starpose/qc/__init__.py`:

```python
"""starpose quality-control scoring."""

from starpose.qc.base import NormRef, QualityScore, QualityScorer

__all__ = ["NormRef", "QualityScore", "QualityScorer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_base.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/chrism/git/starpose
git add src/starpose/qc/__init__.py src/starpose/qc/base.py tests/test_qc_base.py
git commit -m "feat(qc): add QualityScorer ABC + QualityScore/NormRef types"
```

---

## Task 2: starpose focus metrics (variance-of-Laplacian, Tenengrad)

**Repo:** starpose

**Files:**
- Create: `src/starpose/qc/classical.py`
- Test: `tests/test_qc_classical.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_classical.py`:

```python
"""Tests for starpose.qc.classical."""

import numpy as np

from starpose.qc.classical import tenengrad, variance_of_laplacian


def _gaussian_blob(size=128, sigma=8.0, amp=4000.0):
    """A bright Gaussian nucleus on a dim background (uint16)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2
    g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    return (200 + amp * g).astype(np.uint16)


def _blur(patch, k=6):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(patch.astype(np.float32), sigma=k).astype(np.uint16)


def test_sharp_beats_blurred_focus():
    sharp = _gaussian_blob()
    blurred = _blur(sharp, k=6)
    assert variance_of_laplacian(sharp) > variance_of_laplacian(blurred)
    assert tenengrad(sharp) > tenengrad(blurred)


def test_saturated_flat_has_low_focus():
    # Known limitation: a saturated/clipped nucleus is texture-less and looks
    # blurry to focus metrics. Documented, not a bug — detection_score keeps it
    # distinguishable (see Task 4).
    saturated = np.full((128, 128), 65535, dtype=np.uint16)
    assert variance_of_laplacian(saturated) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_classical.py -v`
Expected: FAIL with `ImportError: cannot import name 'variance_of_laplacian'`

- [ ] **Step 3: Write minimal implementation**

Create `src/starpose/qc/classical.py`:

```python
"""Classical (no-model) quality scoring for nucleus patches."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_float(patch: np.ndarray) -> np.ndarray:
    return patch.astype(np.float64)


def variance_of_laplacian(patch: np.ndarray) -> float:
    """High-frequency energy via Laplacian variance. Higher = sharper."""
    lap = ndimage.laplace(_to_float(patch))
    return float(lap.var())


def tenengrad(patch: np.ndarray) -> float:
    """Mean squared Sobel gradient magnitude. Higher = sharper."""
    f = _to_float(patch)
    gx = ndimage.sobel(f, axis=1)
    gy = ndimage.sobel(f, axis=0)
    return float(np.mean(gx * gx + gy * gy))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_classical.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/chrism/git/starpose
git add src/starpose/qc/classical.py tests/test_qc_classical.py
git commit -m "feat(qc): add focus metrics (variance-of-laplacian, tenengrad)"
```

---

## Task 3: starpose detection metrics (foreground fraction, central blob)

**Repo:** starpose

**Files:**
- Modify: `src/starpose/qc/classical.py` (append functions)
- Modify: `tests/test_qc_classical.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qc_classical.py`:

```python
from starpose.qc.classical import central_blob_fraction, foreground_fraction


def test_centered_blob_detection():
    blob = _gaussian_blob()
    assert foreground_fraction(blob) > 0.01
    assert central_blob_fraction(blob) > 0.8  # most foreground is central


def test_empty_patch_detection_zero():
    flat = np.full((128, 128), 200, dtype=np.uint16)
    assert foreground_fraction(flat) == 0.0
    assert central_blob_fraction(flat) == 0.0


def test_corner_blob_not_central():
    yy, xx = np.mgrid[0:128, 0:128]
    g = np.exp(-(((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 8.0**2)))
    corner = (200 + 4000 * g).astype(np.uint16)
    assert central_blob_fraction(corner) < 0.2  # blob is off-center
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_classical.py -v`
Expected: FAIL with `ImportError: cannot import name 'foreground_fraction'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/starpose/qc/classical.py`:

```python
from skimage.filters import threshold_otsu


def _otsu_mask(patch: np.ndarray) -> np.ndarray | None:
    f = _to_float(patch)
    if f.max() <= f.min():
        return None
    try:
        t = threshold_otsu(f)
    except ValueError:
        return None
    return f > t


def foreground_fraction(patch: np.ndarray) -> float:
    """Fraction of pixels above the Otsu threshold (0 if flat)."""
    mask = _otsu_mask(patch)
    if mask is None:
        return 0.0
    return float(mask.mean())


def central_blob_fraction(patch: np.ndarray) -> float:
    """Fraction of foreground mass within the central 50% box (0 if flat)."""
    mask = _otsu_mask(patch)
    if mask is None:
        return 0.0
    total = mask.sum()
    if total == 0:
        return 0.0
    h, w = mask.shape
    central = mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].sum()
    return float(central / total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_classical.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/chrism/git/starpose
git add src/starpose/qc/classical.py tests/test_qc_classical.py
git commit -m "feat(qc): add detection metrics (foreground fraction, central blob)"
```

---

## Task 4: starpose `ClassicalQualityScorer` (fit_reference + score_batch)

**Repo:** starpose

**Files:**
- Modify: `src/starpose/qc/classical.py` (add class)
- Modify: `src/starpose/qc/__init__.py` (export)
- Modify: `tests/test_qc_classical.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qc_classical.py`:

```python
from starpose.qc.classical import ClassicalQualityScorer


def test_scorer_orders_sharp_above_blurred():
    sharp = _gaussian_blob()
    blurred = _blur(sharp, k=6)
    batch = np.stack([sharp, blurred, sharp, blurred])
    scorer = ClassicalQualityScorer()
    scores = scorer.score_batch(batch)
    assert scorer.name == "classical"
    assert scores[0].focus_score > scores[1].focus_score
    assert 0.0 <= scores[0].qc_score <= 1.0
    assert "var_laplacian" in scores[0].metrics


def test_all_bad_slide_does_not_manufacture_high_scores():
    # An all-blurry slide: relative ranking alone would crown a "best" patch,
    # but the absolute floor must keep focus_score near 0 for all of them.
    blurred = _blur(_gaussian_blob(), k=10)
    batch = np.stack([blurred] * 8)
    scorer = ClassicalQualityScorer()
    scores = scorer.score_batch(batch)
    assert max(s.focus_score for s in scores) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_classical.py -v`
Expected: FAIL with `ImportError: cannot import name 'ClassicalQualityScorer'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/starpose/qc/classical.py`:

```python
from starpose.qc.base import NormRef, QualityScore, QualityScorer


class ClassicalQualityScorer(QualityScorer):
    """Fast, model-free scorer: focus (sharpness) + detection (real nucleus).

    Scoring combines a per-slide relative position with an absolute floor so an
    all-bad slide cannot manufacture high scores. Score raw patches, never
    normalized/augmented training tensors.
    """

    def __init__(
        self, varlap_floor: float = 5.0, fg_lo: float = 0.03, fg_hi: float = 0.9
    ) -> None:
        self.varlap_floor = varlap_floor
        self.fg_lo = fg_lo
        self.fg_hi = fg_hi

    @property
    def name(self) -> str:
        return "classical"

    def fit_reference(self, patches: np.ndarray) -> NormRef:
        vl = np.array([variance_of_laplacian(p) for p in patches])
        tg = np.array([tenengrad(p) for p in patches])
        return NormRef(
            varlap_p90=float(np.percentile(vl, 90)),
            tenengrad_p90=float(np.percentile(tg, 90)),
        )

    def _focus_score(self, vl: float, ref: NormRef) -> float:
        denom = max(ref.varlap_p90 - self.varlap_floor, 1e-6)
        return float(np.clip((vl - self.varlap_floor) / denom, 0.0, 1.0))

    def _detection_score(self, fg: float, cb: float) -> float:
        sanity = 1.0 if self.fg_lo <= fg <= self.fg_hi else 0.2
        return float(np.clip(cb * sanity, 0.0, 1.0))

    def score_batch(
        self, patches: np.ndarray, ref: NormRef | None = None
    ) -> list[QualityScore]:
        if ref is None:
            ref = self.fit_reference(patches)
        out: list[QualityScore] = []
        for p in patches:
            vl = variance_of_laplacian(p)
            tg = tenengrad(p)
            fg = foreground_fraction(p)
            cb = central_blob_fraction(p)
            focus = self._focus_score(vl, ref)
            detection = self._detection_score(fg, cb)
            out.append(
                QualityScore(
                    focus_score=focus,
                    detection_score=detection,
                    qc_score=min(focus, detection),
                    metrics={
                        "var_laplacian": vl,
                        "tenengrad": tg,
                        "foreground_frac": fg,
                        "central_blob_frac": cb,
                    },
                )
            )
        return out
```

Update `src/starpose/qc/__init__.py`:

```python
"""starpose quality-control scoring."""

from starpose.qc.base import NormRef, QualityScore, QualityScorer
from starpose.qc.classical import ClassicalQualityScorer

__all__ = [
    "NormRef",
    "QualityScore",
    "QualityScorer",
    "ClassicalQualityScorer",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chrism/git/starpose && uv run pytest tests/test_qc_classical.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/chrism/git/starpose
git add src/starpose/qc/classical.py src/starpose/qc/__init__.py tests/test_qc_classical.py
git commit -m "feat(qc): add ClassicalQualityScorer with relative+absolute scoring"
```

---

## Task 5: dapidl patch reader (LMDB + Zarr)

**Repo:** dapidl (`/mnt/work/git/dapidl`)

**Files:**
- Create: `src/dapidl/qc/__init__.py`
- Create: `src/dapidl/qc/io.py`
- Test: `tests/test_qc_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_io.py`:

```python
"""Tests for dapidl.qc.io patch reader."""

import struct

import numpy as np

from dapidl.qc.io import read_patches


def _write_lmdb(path, patches):
    import lmdb
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, p in enumerate(patches):
            h, w = p.shape
            buf = struct.pack("I", h) + struct.pack("I", w) + p.astype(np.uint16).tobytes()
            txn.put(str(i).encode(), buf)
    env.close()


def test_read_patches_from_lmdb(tmp_path):
    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(5)]
    _write_lmdb(tmp_path / "patches.lmdb", patches)
    out = read_patches(tmp_path, [0, 2, 4])
    assert out.shape == (3, 8, 8)
    assert out[1, 0, 0] == 2  # index 2 -> filled with 2


def test_missing_store_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        read_patches(tmp_path, [0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.qc'`

- [ ] **Step 3: Write minimal implementation**

Create `src/dapidl/qc/__init__.py`:

```python
"""DAPIDL quality-control: scoring pass, patch reader, and montage."""
```

Create `src/dapidl/qc/io.py`:

```python
"""Read raw patches back from a built dataset (LMDB or Zarr)."""

import struct
from pathlib import Path

import numpy as np


def read_patches(dataset_path: Path | str, indices) -> np.ndarray:
    """Return an (N, H, W) uint16 array of patches for the given indices.

    Supports both storage formats produced by the pipeline:
    - patches.lmdb: key=str(idx), value=4-byte H + 4-byte W + uint16 bytes
    - patches.zarr: indexed array
    """
    dataset_path = Path(dataset_path)
    lmdb_path = dataset_path / "patches.lmdb"
    zarr_path = dataset_path / "patches.zarr"
    if lmdb_path.exists():
        return _read_lmdb(lmdb_path, indices)
    if zarr_path.exists():
        return _read_zarr(zarr_path, indices)
    raise FileNotFoundError(f"No patches.lmdb or patches.zarr in {dataset_path}")


def _read_lmdb(lmdb_path: Path, indices) -> np.ndarray:
    import lmdb

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    patches = []
    try:
        with env.begin() as txn:
            for idx in indices:
                data = txn.get(str(int(idx)).encode())
                if data is None:
                    raise KeyError(f"patch {idx} missing in {lmdb_path}")
                h = struct.unpack("I", data[:4])[0]
                w = struct.unpack("I", data[4:8])[0]
                patches.append(
                    np.frombuffer(data[8:], dtype=np.uint16).reshape(h, w)
                )
    finally:
        env.close()
    return np.stack(patches)


def _read_zarr(zarr_path: Path, indices) -> np.ndarray:
    import zarr

    arr = zarr.open(str(zarr_path), mode="r")
    return np.stack([np.asarray(arr[int(i)]) for i in indices])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_io.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/qc/__init__.py src/dapidl/qc/io.py tests/test_qc_io.py
git commit -m "feat(qc): add LMDB/Zarr patch reader for post-hoc scoring"
```

---

## Task 6: dapidl montage builder (per-class worst-first)

**Repo:** dapidl

**Files:**
- Create: `src/dapidl/qc/montage.py`
- Test: `tests/test_qc_montage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_montage.py`:

```python
"""Tests for dapidl.qc.montage."""

import numpy as np

from dapidl.qc.montage import build_class_montage


def test_montage_returns_rgb_image():
    rng = np.random.default_rng(0)
    patches = rng.integers(0, 4000, size=(10, 32, 32), dtype=np.uint16)
    scores = np.linspace(0.0, 1.0, 10)
    img = build_class_montage(patches, scores, cell_type="Immune", top_n=6, cols=3)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_montage_caps_at_top_n():
    patches = np.zeros((100, 16, 16), dtype=np.uint16)
    scores = np.linspace(0, 1, 100)
    # Should not raise and should render at most top_n tiles
    img = build_class_montage(patches, scores, cell_type="T", top_n=4, cols=2)
    assert img.shape[0] > 0 and img.shape[1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_montage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.qc.montage'`

- [ ] **Step 3: Write minimal implementation**

Create `src/dapidl/qc/montage.py`:

```python
"""Per-class worst-first montage of nucleus patches for visual QC."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def build_class_montage(
    patches: np.ndarray,
    scores: np.ndarray,
    cell_type: str,
    top_n: int = 64,
    cols: int = 8,
) -> np.ndarray:
    """Grid of the worst-scoring patches for one cell type.

    Patches are sorted by score ascending (worst first). Each tile's score is
    rendered as a title (margin), never overlaid on the 128px pixels. Returns an
    (H, W, 3) uint8 RGB image.
    """
    scores = np.asarray(scores)
    order = np.argsort(scores)[: min(top_n, len(scores))]
    n = len(order)
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, idx in enumerate(order):
        p = patches[idx].astype(np.float32)
        lo, hi = np.percentile(p, [1, 99])
        norm = np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1)
        axes[i].imshow(norm, cmap="gray")
        axes[i].set_title(f"{scores[idx]:.2f}", fontsize=7)
    fig.suptitle(f"{cell_type} — worst {n} by qc_score", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_montage.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/qc/montage.py tests/test_qc_montage.py
git commit -m "feat(qc): add per-class worst-first montage builder"
```

---

## Task 7: dapidl post-hoc QC pass (read → group → score → sidecar)

**Repo:** dapidl

**Files:**
- Create: `src/dapidl/pipeline/steps/quality_control.py`
- Test: `tests/test_qc_step.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_step.py`:

```python
"""Tests for the dapidl QC pass (no ClearML)."""

import json
import struct

import numpy as np
import polars as pl

from dapidl.pipeline.steps.quality_control import run_quality_control


def _build_dataset(path):
    """Tiny single-slide dataset: 6 patches (3 sharp blobs, 3 flat) + metadata."""
    import lmdb

    yy, xx = np.mgrid[0:32, 0:32]
    blob = (200 + 4000 * np.exp(-(((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 5.0**2)))).astype(np.uint16)
    flat = np.full((32, 32), 200, dtype=np.uint16)
    patches = [blob, blob, blob, flat, flat, flat]

    env = lmdb.open(str(path / "patches.lmdb"), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, p in enumerate(patches):
            buf = struct.pack("I", 32) + struct.pack("I", 32) + p.tobytes()
            txn.put(str(i).encode(), buf)
    env.close()

    pl.DataFrame({
        "cell_id": [f"c{i}" for i in range(6)],
        "broad_category": ["Epithelial"] * 6,
    }).write_parquet(path / "metadata.parquet")
    # slide_stats.json: JSON grouping (safe; avoids object-dtype sources.npy)
    (path / "slide_stats.json").write_text(json.dumps([{"source": "slideA", "n_written": 6}]))
    (path / "class_mapping.json").write_text(json.dumps({"Epithelial": 0}))


def test_run_writes_sidecar_with_scores(tmp_path):
    _build_dataset(tmp_path)
    run_quality_control(tmp_path, use_clearml=False, montage_top_n=4)

    sidecar = tmp_path / "qc" / "qc_scores.parquet"
    assert sidecar.exists()
    df = pl.read_parquet(sidecar)
    assert set(["cell_id", "focus_score", "detection_score", "qc_score"]).issubset(df.columns)
    assert df.height == 6
    # blobs (first 3) should out-score flat patches (last 3) on detection
    by_id = dict(zip(df["cell_id"], df["detection_score"]))
    assert by_id["c0"] > by_id["c3"]


def test_provenance_written(tmp_path):
    _build_dataset(tmp_path)
    run_quality_control(tmp_path, use_clearml=False, montage_top_n=4)
    meta = json.loads((tmp_path / "qc" / "qc_scores.meta.json").read_text())
    assert meta["scorer"] == "classical"
    assert "date" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_step.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.pipeline.steps.quality_control'`

- [ ] **Step 3: Write minimal implementation**

Create `src/dapidl/pipeline/steps/quality_control.py`:

```python
"""Post-hoc quality-control pass over a built DAPIDL dataset.

Reads patches, groups by slide, fits a per-slide normalization reference from a
sample, scores every patch, and writes a sidecar qc/qc_scores.parquet (plus
provenance). metadata.parquet is never modified. Optionally logs montages and
score histograms to ClearML.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from starpose.qc.classical import ClassicalQualityScorer

from dapidl.qc.io import read_patches
from dapidl.qc.montage import build_class_montage

REFERENCE_SAMPLE = 2000  # patches sampled per slide to fit the normalization ref


def _slide_groups(dataset_path: Path, n: int) -> np.ndarray:
    """Per-patch slide labels, reconstructed from slide_stats.json (safe JSON).

    Falls back to a single group if slide_stats.json is absent.
    """
    stats_path = dataset_path / "slide_stats.json"
    if not stats_path.exists():
        return np.array(["__single__"] * n, dtype=object)
    stats = json.loads(stats_path.read_text())
    sources = np.empty(n, dtype=object)
    i = 0
    for s in stats:
        cnt = int(s["n_written"])
        sources[i : i + cnt] = s["source"]
        i += cnt
    if i != n:
        raise ValueError(f"slide_stats.json sums to {i} != {n} metadata rows")
    return sources


def run_quality_control(
    dataset_path: Path | str,
    use_clearml: bool = True,
    montage_top_n: int = 64,
    seed: int = 42,
) -> Path:
    """Score a dataset, write the sidecar, build montages. Returns the qc/ dir."""
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.parquet in {dataset_path}")
    meta = pl.read_parquet(meta_path)
    n = meta.height

    sources = _slide_groups(dataset_path, n)
    scorer = ClassicalQualityScorer()
    rng = np.random.default_rng(seed)
    focus = np.zeros(n)
    detection = np.zeros(n)
    qc = np.zeros(n)
    raw = {k: np.zeros(n) for k in ("var_laplacian", "tenengrad", "foreground_frac", "central_blob_frac")}

    for slide in sorted(set(sources.tolist())):
        idx = np.where(sources == slide)[0]
        sample = idx if len(idx) <= REFERENCE_SAMPLE else rng.choice(idx, REFERENCE_SAMPLE, replace=False)
        ref = scorer.fit_reference(read_patches(dataset_path, sample))
        for start in range(0, len(idx), 1000):
            chunk = idx[start : start + 1000]
            scores = scorer.score_batch(read_patches(dataset_path, chunk), ref=ref)
            for j, gi in enumerate(chunk):
                s = scores[j]
                focus[gi], detection[gi], qc[gi] = s.focus_score, s.detection_score, s.qc_score
                for k in raw:
                    raw[k][gi] = s.metrics[k]
        logger.info(f"QC scored slide {slide}: {len(idx)} patches")

    out_dir = dataset_path / "qc"
    out_dir.mkdir(exist_ok=True)
    scores_df = meta.select("cell_id").with_columns(
        focus_score=pl.Series(focus),
        detection_score=pl.Series(detection),
        qc_score=pl.Series(qc),
        **{k: pl.Series(v) for k, v in raw.items()},
    )
    _atomic_write_parquet(scores_df, out_dir / "qc_scores.parquet")

    (out_dir / "qc_scores.meta.json").write_text(json.dumps({
        "scorer": scorer.name,
        "params": {"varlap_floor": scorer.varlap_floor, "fg_lo": scorer.fg_lo, "fg_hi": scorer.fg_hi},
        "reference_sample": REFERENCE_SAMPLE,
        "date": date.today().isoformat(),
    }, indent=2))

    montages = _build_montages(dataset_path, meta, qc, montage_top_n, out_dir)
    if use_clearml:
        _log_to_clearml(dataset_path, scores_df, montages)
    return out_dir


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _build_montages(dataset_path, meta, qc, top_n, out_dir) -> dict:
    """One worst-first montage PNG per cell type. Returns {cell_type: image}."""
    label_col = "broad_category" if "broad_category" in meta.columns else "predicted_type"
    labels = meta[label_col].to_numpy()
    montages = {}
    import matplotlib.image as mpimg
    for cell_type in sorted(set(labels.tolist())):
        idx = np.where(labels == cell_type)[0]
        if len(idx) == 0:
            continue
        patches = read_patches(dataset_path, idx)
        img = build_class_montage(patches, qc[idx], cell_type, top_n=top_n)
        mpimg.imsave(out_dir / f"montage_{_safe_name(cell_type)}.png", img)
        montages[cell_type] = img
    return montages


def _log_to_clearml(dataset_path, scores_df, montages) -> None:
    import clearml

    task = clearml.Task.current_task() or clearml.Task.init(
        project_name="DAPIDL/QC",
        task_name=f"qc_{dataset_path.name}",
        task_type=clearml.Task.TaskTypes.qc,
    )
    qc_logger = task.get_logger()
    for cell_type, img in montages.items():
        qc_logger.report_image(title="worst_qc", series=cell_type, iteration=0, image=img)
    qc_logger.report_histogram(
        title="qc_score", series="all", iteration=0,
        values=scores_df["qc_score"].to_numpy(),
    )
    task.upload_artifact(name="qc_scores", artifact_object=scores_df.to_pandas())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_step.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/pipeline/steps/quality_control.py tests/test_qc_step.py
git commit -m "feat(qc): add post-hoc QC pass writing sidecar scores + montages"
```

---

## Task 8: dapidl ClearML logging test (mocked)

**Repo:** dapidl

**Files:**
- Modify: `tests/test_qc_step.py` (append mocked-ClearML test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qc_step.py`:

```python
from unittest.mock import MagicMock, patch as mock_patch


def test_clearml_logging_called(tmp_path):
    _build_dataset(tmp_path)
    fake_task = MagicMock()
    fake_logger = MagicMock()
    fake_task.get_logger.return_value = fake_logger
    with mock_patch("clearml.Task") as TaskCls:
        TaskCls.current_task.return_value = None
        TaskCls.init.return_value = fake_task
        TaskCls.TaskTypes.qc = "qc"
        run_quality_control(tmp_path, use_clearml=True, montage_top_n=4)
    assert fake_logger.report_image.called
    assert fake_logger.report_histogram.called
    assert fake_task.upload_artifact.called
```

- [ ] **Step 2: Run test to verify it passes**

`_log_to_clearml` already imports the `clearml` module (not the symbol), so `mock_patch("clearml.Task")` is honored.

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_step.py -v`
Expected: PASS (3 passed). If it FAILS because `clearml` is not installed in the test env, install it (`uv add clearml`) or skip via `pytest.importorskip("clearml")` at the top of the test — but clearml is a project dependency, so it should import.

- [ ] **Step 3: Commit**

```bash
cd /mnt/work/git/dapidl
git add tests/test_qc_step.py
git commit -m "test(qc): verify ClearML montage/histogram/artifact logging"
```

---

## Task 9: dapidl `qc` CLI command

**Repo:** dapidl

**Files:**
- Modify: `src/dapidl/cli.py` (add command near other `@main.command` defs)
- Test: `tests/test_qc_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qc_cli.py`:

```python
"""Tests for the `dapidl qc` CLI command."""

from unittest.mock import patch

from click.testing import CliRunner

from dapidl.cli import main


def test_qc_command_help():
    result = CliRunner().invoke(main, ["qc", "--help"])
    assert result.exit_code == 0
    assert "dataset" in result.output.lower()


def test_qc_command_invokes_step(tmp_path):
    with patch("dapidl.pipeline.steps.quality_control.run_quality_control") as run:
        result = CliRunner().invoke(
            main, ["qc", "--dataset", str(tmp_path), "--no-clearml"]
        )
    assert result.exit_code == 0
    assert run.called
    _, kwargs = run.call_args
    assert kwargs["use_clearml"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_cli.py -v`
Expected: FAIL — `Error: No such command 'qc'.`

- [ ] **Step 3: Write minimal implementation**

Add to `src/dapidl/cli.py` (alongside the other `@main.command` definitions; uses the module-level `console` already defined there):

```python
@main.command(name="qc")
@click.option("--dataset", "dataset", required=True, type=click.Path(), help="Path to a built dataset dir (with patches.lmdb/zarr + metadata.parquet)")
@click.option("--montage-top-n", default=64, show_default=True, help="Worst-N patches per class in the montage")
@click.option("--clearml/--no-clearml", default=True, help="Log montages + histograms to ClearML")
def qc_cmd(dataset: str, montage_top_n: int, clearml: bool) -> None:
    """Score patch quality (focus + detection) and build inspection montages."""
    from dapidl.pipeline.steps.quality_control import run_quality_control

    console.print(f"[bold blue]QC scoring[/bold blue] {dataset}")
    out_dir = run_quality_control(
        dataset, use_clearml=clearml, montage_top_n=montage_top_n
    )
    console.print(f"[green]Wrote[/green] {out_dir}/qc_scores.parquet + montages")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/cli.py tests/test_qc_cli.py
git commit -m "feat(qc): add 'dapidl qc' CLI command"
```

---

## Task 10: End-to-end smoke run on a real dataset

**Repo:** dapidl (manual verification, no new code)

- [ ] **Step 1: Run QC on a small real dataset (no ClearML)**

Run:
```bash
cd /mnt/work/git/dapidl
uv run dapidl qc --dataset /mnt/work/datasets/derived/breast-6source-dapi-p128 --no-clearml --montage-top-n 36
```
Expected: writes `qc/qc_scores.parquet`, `qc/qc_scores.meta.json`, and `qc/montage_<class>.png` per class. No errors.

- [ ] **Step 2: Eyeball the worst patches**

Open a couple of `qc/montage_*.png` files. Confirm the worst-scoring tiles are visibly blurry or empty/false — i.e. the score tracks perceived quality. If a whole class looks fine despite low scores, revisit `varlap_floor` (Task 4) and re-run.

- [ ] **Step 3: Confirm the slide grouping is correct**

Run:
```bash
cd /mnt/work/git/dapidl
uv run python -c "import json,pathlib; s=json.loads(pathlib.Path('/mnt/work/datasets/derived/breast-6source-dapi-p128/slide_stats.json').read_text()); print([(x['source'], x['n_written']) for x in s])"
```
Expected: one entry per acquisition/slide (e.g. xenium_rep1, sthelar_breast_s0, ...) with counts summing to the dataset size. If `slide_stats.json` is missing or sources are NOT 1:1 with slides, the per-slide normalization grouping needs a different key (spec §11) — fix before trusting cross-slide scores.

- [ ] **Step 4: (Optional) Run with ClearML and inspect in the web UI**

Run:
```bash
cd /mnt/work/git/dapidl
uv run dapidl qc --dataset /mnt/work/datasets/derived/breast-6source-dapi-p128 --montage-top-n 36
```
Open the ClearML task under project `DAPIDL/QC`; confirm per-class montages appear under Debug Samples and the `qc_score` histogram + `qc_scores` artifact are present.

---

## Self-Review

**Spec coverage:**
- §3 starpose split → Tasks 1-4 (scorer in starpose); dapidl glue → Tasks 5-9. ✓
- §4.1 QualityScorer ABC + QualityScore → Task 1. ✓
- §4.2 ClassicalQualityScorer (focus, detection, raw metrics, raw-patch scoring) → Tasks 2-4. ✓
- §4.3 post-hoc pass (read LMDB/Zarr, group, sample-fit) → Tasks 5, 7. ✓
- §4.4 montage (worst-first, margin labels, capped) → Task 6. ✓
- §4.5 dapidl CLI → Task 9. starpose CLI intentionally deferred per spec §3 (not built). ✓
- §6 relative+absolute scoring, min combine, sampled reference, memory → Task 4 + Task 7 (chunked, sampled). ✓
- §7 sidecar parquet + provenance + atomic write + montage PNG path; metadata.parquet untouched → Task 7. ✓
- §8 error handling (missing store, missing metadata, grouping length mismatch) → Tasks 5, 7. ✓
- §9 tests incl. hard negatives (saturated, empty, corner blob, all-bad floor) → Tasks 2-4, 7. ✓
- ClearML logging → Task 8. ✓
- §10 future work (FM scorer, extraction hook, training integration, qc_scores.npy) → intentionally NOT built. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; every command has expected output. ✓

**Type consistency:** `QualityScore(focus_score, detection_score, qc_score, metrics)`, `NormRef(varlap_p90, tenengrad_p90)`, `QualityScorer.score_batch(patches, ref)/name`, `ClassicalQualityScorer(varlap_floor, fg_lo, fg_hi)` + `fit_reference`, `read_patches(dataset_path, indices)`, `build_class_montage(patches, scores, cell_type, top_n, cols)`, `run_quality_control(dataset_path, use_clearml, montage_top_n, seed)`, `_slide_groups(dataset_path, n)` — names match across all tasks. ✓

**Notes for executor:**
- `ClassicalQualityScorer` `varlap_floor=5.0` is a calibration constant to tune during Task 10, not a correctness invariant.
- Confirm `clearml.Task.TaskTypes.qc` exists in the installed ClearML version; if not, use `data_processing` in `_log_to_clearml` (Task 7).
