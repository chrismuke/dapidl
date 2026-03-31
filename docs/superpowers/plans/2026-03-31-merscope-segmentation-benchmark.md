# MERSCOPE Segmentation Grand Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark 7 segmentation methods + 3 consensus approaches on MERSCOPE DAPI, evaluated across biological, morphometric, cross-method, and practical axes.

**Architecture:** Modular benchmark framework under `src/dapidl/benchmark/` with adapter pattern for segmenters, pluggable evaluation modules, and consensus engine. FOV-based sampling for fast iteration. Script entry point at `scripts/run_segmentation_benchmark.py`.

**Tech Stack:** cellpose 4.0.8, stardist, deepcell (mesmer), instanseg, scikit-image (regionprops), scipy (Hungarian matching), polars, matplotlib, tifffile (mmap loading)

**Spec:** `docs/superpowers/specs/2026-03-31-merscope-segmentation-benchmark-design.md`

---

## File Map

```
src/dapidl/benchmark/
├── __init__.py                        # Exports: BenchmarkRunner, SegmenterAdapter, SegmentationOutput
├── fov_selector.py                    # select_fovs() → list[FOVTile]
├── segmenters/
│   ├── __init__.py                    # Exports all adapters
│   ├── base.py                        # SegmenterAdapter ABC, SegmentationOutput dataclass
│   ├── cellpose_adapter.py            # CellposeSAMAdapter, CellposeCyto3Adapter, CellposeNucleiAdapter
│   ├── stardist_adapter.py            # StarDistAdapter
│   ├── mesmer_adapter.py              # MesmerAdapter
│   ├── instanseg_adapter.py           # InstanSegAdapter
│   └── native_adapter.py             # NativeAdapter (MERSCOPE platform)
├── consensus/
│   ├── __init__.py                    # Exports all consensus methods
│   ├── instance_matching.py           # Shared IoU matching utilities
│   ├── majority_voting.py             # MajorityVotingConsensus
│   ├── iou_weighted.py                # IoUWeightedConsensus
│   └── topological_voting.py          # TopologicalVotingConsensus
├── evaluation/
│   ├── __init__.py                    # Exports: evaluate_all()
│   ├── morphometric.py                # compute_morphometric_metrics()
│   ├── biological.py                  # compute_biological_metrics()
│   ├── cross_method.py                # compute_cross_method_metrics()
│   └── practical.py                   # (recorded during segmentation, no separate module)
└── reporting.py                       # generate_report() → markdown + plots

scripts/run_segmentation_benchmark.py  # CLI entry point

tests/test_benchmark_fov_selector.py
tests/test_benchmark_evaluation.py
tests/test_benchmark_consensus.py
```

---

## Task 1: Adapter Base + SegmentationOutput

**Files:**
- Create: `src/dapidl/benchmark/__init__.py`
- Create: `src/dapidl/benchmark/segmenters/__init__.py`
- Create: `src/dapidl/benchmark/segmenters/base.py`
- Test: `tests/test_benchmark_evaluation.py` (initial)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/dapidl/benchmark/segmenters src/dapidl/benchmark/consensus src/dapidl/benchmark/evaluation
```

- [ ] **Step 2: Write the adapter base and output dataclass**

Create `src/dapidl/benchmark/segmenters/base.py`:

```python
"""Base adapter interface for segmentation benchmark."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentationOutput:
    """Standardized output from any segmenter."""

    masks: np.ndarray  # (H, W) int32 label image, 0=background
    centroids: np.ndarray  # (N, 2) float64 [y, x] in pixels
    n_cells: int
    runtime_seconds: float
    peak_memory_mb: float
    method_name: str
    metadata: dict = field(default_factory=dict)


class SegmenterAdapter(ABC):
    """Thin wrapper to standardize any segmentation method."""

    @abstractmethod
    def segment(
        self, image: np.ndarray, pixel_size_um: float = 0.108
    ) -> SegmentationOutput:
        """Segment a single FOV image tile.

        Args:
            image: (H, W) uint16 DAPI image
            pixel_size_um: Physical pixel size in microns

        Returns:
            SegmentationOutput with masks, centroids, timing
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""
        ...

    @property
    def supports_cell_boundaries(self) -> bool:
        """Whether this method produces whole-cell (not just nucleus) segmentation."""
        return False
```

- [ ] **Step 3: Write test for SegmentationOutput**

Create `tests/test_benchmark_evaluation.py`:

```python
"""Tests for benchmark framework core types and evaluation."""

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter


def test_segmentation_output_creation():
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:20, 10:20] = 1
    masks[50:60, 50:60] = 2
    centroids = np.array([[15.0, 15.0], [55.0, 55.0]])

    out = SegmentationOutput(
        masks=masks,
        centroids=centroids,
        n_cells=2,
        runtime_seconds=1.5,
        peak_memory_mb=100.0,
        method_name="test",
    )
    assert out.n_cells == 2
    assert out.masks.shape == (100, 100)
    assert out.centroids.shape == (2, 2)


def test_segmenter_adapter_is_abstract():
    """SegmenterAdapter cannot be instantiated directly."""
    import pytest

    with pytest.raises(TypeError):
        SegmenterAdapter()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmark_evaluation.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Create package __init__ files**

Create `src/dapidl/benchmark/__init__.py`:

```python
"""Segmentation benchmark framework for comparing methods on MERSCOPE data."""

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter

__all__ = ["SegmentationOutput", "SegmenterAdapter"]
```

Create `src/dapidl/benchmark/segmenters/__init__.py`:

```python
"""Segmenter adapters for benchmark framework."""
```

Create `src/dapidl/benchmark/consensus/__init__.py`:

```python
"""Consensus segmentation methods."""
```

Create `src/dapidl/benchmark/evaluation/__init__.py`:

```python
"""Evaluation metrics for segmentation benchmark."""
```

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/benchmark/ tests/test_benchmark_evaluation.py
git commit -m "feat(benchmark): add segmenter adapter base and SegmentationOutput"
```

---

## Task 2: FOV Selector

**Files:**
- Create: `src/dapidl/benchmark/fov_selector.py`
- Test: `tests/test_benchmark_fov_selector.py`

- [ ] **Step 1: Write test for FOV selection**

Create `tests/test_benchmark_fov_selector.py`:

```python
"""Tests for FOV selection logic."""

import numpy as np
import polars as pl
import pytest

from dapidl.benchmark.fov_selector import FOVTile, select_fovs


@pytest.fixture
def sample_cell_metadata() -> pl.DataFrame:
    """Create synthetic cell_metadata with 10 FOVs of varying density."""
    rng = np.random.default_rng(42)
    rows = []
    for fov_id in range(10):
        n_cells = 50 + fov_id * 30  # 50 to 320 cells per FOV
        base_x = fov_id * 200.0  # Each FOV offset by 200 microns
        base_y = 0.0
        for _ in range(n_cells):
            cx = base_x + rng.uniform(0, 200)
            cy = base_y + rng.uniform(0, 200)
            vol = rng.uniform(100, 800) if fov_id != 8 else rng.uniform(30, 150)
            rows.append({
                "fov": fov_id,
                "volume": vol,
                "center_x": cx,
                "center_y": cy,
                "min_x": cx - 5,
                "max_x": cx + 5,
                "min_y": cy - 5,
                "max_y": cy + 5,
            })
    return pl.DataFrame(rows)


def test_select_fovs_returns_5(sample_cell_metadata):
    fovs = select_fovs(sample_cell_metadata, n_fovs=5)
    assert len(fovs) == 5


def test_select_fovs_all_unique(sample_cell_metadata):
    fovs = select_fovs(sample_cell_metadata, n_fovs=5)
    fov_ids = [f.fov_id for f in fovs]
    assert len(set(fov_ids)) == 5


def test_fov_tile_has_required_fields(sample_cell_metadata):
    fovs = select_fovs(sample_cell_metadata, n_fovs=5)
    for fov in fovs:
        assert isinstance(fov, FOVTile)
        assert fov.fov_id >= 0
        assert fov.label in ("dense", "sparse", "mixed", "edge", "immune")
        assert fov.pixel_bbox is not None  # (y_min, y_max, x_min, x_max) in pixels
        assert fov.n_cells > 0


def test_dense_fov_has_highest_density(sample_cell_metadata):
    fovs = select_fovs(sample_cell_metadata, n_fovs=5)
    dense = next(f for f in fovs if f.label == "dense")
    sparse = next(f for f in fovs if f.label == "sparse")
    assert dense.density > sparse.density
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_benchmark_fov_selector.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement FOV selector**

Create `src/dapidl/benchmark/fov_selector.py`:

```python
"""FOV selection for segmentation benchmark.

Selects representative FOVs from MERSCOPE cell_metadata based on
density, volume, and spatial position criteria.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# MERSCOPE breast dataset affine transform (micron -> pixel)
# From micron_to_mosaic_pixel_transform.csv
DEFAULT_SCALE = 9.259259  # pixels per micron
DEFAULT_OFFSET_X = 357.2  # pixel offset
DEFAULT_OFFSET_Y = 2007.97  # pixel offset


@dataclass
class FOVTile:
    """A selected FOV with metadata for extraction."""

    fov_id: int
    label: str  # dense, sparse, mixed, edge, immune
    n_cells: int
    density: float  # cells per 1000 um2
    mean_volume: float
    pixel_bbox: tuple[int, int, int, int]  # (y_min, y_max, x_min, x_max) in pixels
    micron_bbox: tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max) in microns


def _micron_to_pixel(
    x_um: float,
    y_um: float,
    scale: float = DEFAULT_SCALE,
    offset_x: float = DEFAULT_OFFSET_X,
    offset_y: float = DEFAULT_OFFSET_Y,
) -> tuple[int, int]:
    """Convert micron coordinates to pixel coordinates."""
    px = int(x_um * scale + offset_x)
    py = int(y_um * scale + offset_y)
    return py, px  # Return as (row, col)


def load_transform(transform_path: Path) -> tuple[float, float, float]:
    """Load affine transform from MERSCOPE CSV.

    Returns (scale, offset_x, offset_y).
    """
    import csv

    with open(transform_path) as f:
        reader = csv.reader(f)
        rows = [list(map(float, row)) for row in reader]
    # 3x3 affine: [[sx, 0, tx], [0, sy, ty], [0, 0, 1]]
    scale_x = rows[0][0]
    offset_x = rows[0][2]
    offset_y = rows[1][2]
    return abs(scale_x), offset_x, offset_y


def select_fovs(
    cell_metadata: pl.DataFrame,
    n_fovs: int = 5,
    padding_um: float = 10.0,
    scale: float = DEFAULT_SCALE,
    offset_x: float = DEFAULT_OFFSET_X,
    offset_y: float = DEFAULT_OFFSET_Y,
) -> list[FOVTile]:
    """Select representative FOVs for benchmarking.

    Args:
        cell_metadata: DataFrame with columns: fov, volume, center_x, center_y,
                       min_x, max_x, min_y, max_y
        n_fovs: Number of FOVs to select (default 5)
        padding_um: Padding around FOV bounding box in microns
        scale: Pixels per micron
        offset_x: Pixel offset X
        offset_y: Pixel offset Y

    Returns:
        List of FOVTile objects sorted by label
    """
    # Compute per-FOV statistics
    fov_stats = cell_metadata.group_by("fov").agg(
        pl.len().alias("n_cells"),
        pl.col("volume").mean().alias("mean_volume"),
        pl.col("center_x").min().alias("cx_min"),
        pl.col("center_x").max().alias("cx_max"),
        pl.col("center_y").min().alias("cy_min"),
        pl.col("center_y").max().alias("cy_max"),
        pl.col("min_x").min().alias("bbox_x_min"),
        pl.col("max_x").max().alias("bbox_x_max"),
        pl.col("min_y").min().alias("bbox_y_min"),
        pl.col("max_y").max().alias("bbox_y_max"),
    )

    # Compute spatial extent and density
    fov_stats = fov_stats.with_columns(
        ((pl.col("cx_max") - pl.col("cx_min")) * (pl.col("cy_max") - pl.col("cy_min")))
        .alias("area_um2"),
    )
    # Avoid division by zero for single-cell FOVs
    fov_stats = fov_stats.with_columns(
        pl.when(pl.col("area_um2") > 0)
        .then(pl.col("n_cells") / pl.col("area_um2") * 1000)
        .otherwise(0.0)
        .alias("density"),
    )

    # Compute edge distance (max pixel coordinate = closest to mosaic boundary)
    fov_stats = fov_stats.with_columns(
        (pl.col("cx_max") * scale + offset_x).alias("max_pixel_x"),
        (pl.col("cy_max") * scale + offset_y).alias("max_pixel_y"),
    )
    fov_stats = fov_stats.with_columns(
        pl.max_horizontal("max_pixel_x", "max_pixel_y").alias("edge_distance"),
    )

    stats_sorted = fov_stats.sort("density")
    n_rows = len(stats_sorted)

    selected: dict[str, int] = {}

    # Dense: p95 density
    idx = min(int(n_rows * 0.95), n_rows - 1)
    selected["dense"] = stats_sorted.row(idx, named=True)["fov"]

    # Sparse: p5 density (skip zero-area FOVs)
    nonzero = stats_sorted.filter(pl.col("density") > 0)
    if len(nonzero) > 0:
        idx = max(int(len(nonzero) * 0.05), 0)
        selected["sparse"] = nonzero.row(idx, named=True)["fov"]
    else:
        selected["sparse"] = stats_sorted.row(0, named=True)["fov"]

    # Mixed: p50 density
    idx = n_rows // 2
    selected["mixed"] = stats_sorted.row(idx, named=True)["fov"]

    # Edge: max pixel coordinate (excluding already selected)
    remaining = fov_stats.filter(~pl.col("fov").is_in(list(selected.values())))
    edge_row = remaining.sort("edge_distance", descending=True).row(0, named=True)
    selected["edge"] = edge_row["fov"]

    # Immune: smallest mean volume (excluding already selected)
    remaining = fov_stats.filter(~pl.col("fov").is_in(list(selected.values())))
    immune_row = remaining.sort("mean_volume").row(0, named=True)
    selected["immune"] = immune_row["fov"]

    # Build FOVTile objects
    results = []
    for label, fov_id in selected.items():
        row = fov_stats.filter(pl.col("fov") == fov_id).row(0, named=True)

        # Bounding box in microns with padding
        x_min = row["bbox_x_min"] - padding_um
        x_max = row["bbox_x_max"] + padding_um
        y_min = row["bbox_y_min"] - padding_um
        y_max = row["bbox_y_max"] + padding_um

        # Convert to pixel coordinates
        py_min, px_min = _micron_to_pixel(x_min, y_min, scale, offset_x, offset_y)
        py_max, px_max = _micron_to_pixel(x_max, y_max, scale, offset_x, offset_y)

        # Ensure correct ordering (y might be inverted)
        py_min, py_max = min(py_min, py_max), max(py_min, py_max)
        px_min, px_max = min(px_min, px_max), max(px_min, px_max)

        results.append(FOVTile(
            fov_id=fov_id,
            label=label,
            n_cells=row["n_cells"],
            density=row["density"],
            mean_volume=row["mean_volume"],
            pixel_bbox=(py_min, py_max, px_min, px_max),
            micron_bbox=(x_min, x_max, y_min, y_max),
        ))

    logger.info(
        "Selected %d FOVs: %s",
        len(results),
        {f.label: f"fov={f.fov_id} ({f.n_cells} cells)" for f in results},
    )
    return sorted(results, key=lambda f: f.label)


def extract_fov_tile(
    dapi_path: Path,
    fov: FOVTile,
) -> np.ndarray:
    """Extract a FOV tile from the DAPI mosaic using memory-mapped access.

    Args:
        dapi_path: Path to mosaic DAPI TIFF
        fov: FOVTile with pixel bounding box

    Returns:
        (H, W) uint16 array of the FOV region
    """
    import tifffile

    y_min, y_max, x_min, x_max = fov.pixel_bbox

    with tifffile.TiffFile(dapi_path) as tif:
        page = tif.pages[0]
        img_h, img_w = page.shape

        # Clamp to image bounds
        y_min = max(0, y_min)
        y_max = min(img_h, y_max)
        x_min = max(0, x_min)
        x_max = min(img_w, x_max)

        # Read the full image and slice
        full = page.asarray()
        tile = full[y_min:y_max, x_min:x_max].copy()

    return tile
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmark_fov_selector.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/benchmark/fov_selector.py tests/test_benchmark_fov_selector.py
git commit -m "feat(benchmark): add FOV selector for representative tissue sampling"
```

---

## Task 3: Cellpose Adapters (SAM, cyto3, nuclei)

**Files:**
- Create: `src/dapidl/benchmark/segmenters/cellpose_adapter.py`
- Modify: `tests/test_benchmark_evaluation.py`

- [ ] **Step 1: Write test for Cellpose adapters**

Append to `tests/test_benchmark_evaluation.py`:

```python
import unittest.mock as mock


def _make_fake_cellpose_result(n_cells=5, h=512, w=512):
    """Create a fake cellpose masks array for testing."""
    masks = np.zeros((h, w), dtype=np.int32)
    for i in range(1, n_cells + 1):
        r = 40 + i * 80
        c = 40 + i * 80
        masks[r - 15 : r + 15, c - 15 : c + 15] = i
    return masks


def test_cellpose_sam_adapter_name():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter

    adapter = CellposeSAMAdapter()
    assert adapter.name == "cellpose_sam"


def test_cellpose_cyto3_adapter_name():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeCyto3Adapter

    adapter = CellposeCyto3Adapter()
    assert adapter.name == "cellpose_cyto3"


def test_cellpose_nuclei_adapter_name():
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeNucleiAdapter

    adapter = CellposeNucleiAdapter()
    assert adapter.name == "cellpose_nuclei"


def test_cellpose_adapter_segment_returns_output():
    """Test that adapter wraps cellpose and returns SegmentationOutput."""
    from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter

    adapter = CellposeSAMAdapter()
    fake_masks = _make_fake_cellpose_result(3, 256, 256)

    with mock.patch.object(adapter, "_run_cellpose", return_value=fake_masks):
        image = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
        result = adapter.segment(image, pixel_size_um=0.108)

    assert isinstance(result, SegmentationOutput)
    assert result.n_cells == 3
    assert result.masks.shape == (256, 256)
    assert result.method_name == "cellpose_sam"
    assert result.runtime_seconds >= 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmark_evaluation.py::test_cellpose_sam_adapter_name -v
```

Expected: FAIL (import error)

- [ ] **Step 3: Implement Cellpose adapters**

Create `src/dapidl/benchmark/segmenters/cellpose_adapter.py`:

```python
"""Cellpose segmenter adapters for benchmark.

Wraps Cellpose 4.0.8 with three model configurations:
- CellposeSAMAdapter: nuclei model with SAM backbone (default in CP4)
- CellposeCyto3Adapter: cyto3 model for cell+nucleus
- CellposeNucleiAdapter: nuclei model without SAM (CP3-style)
"""

import logging
import time

import numpy as np
from scipy import ndimage

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter

logger = logging.getLogger(__name__)


def _centroids_from_masks(masks: np.ndarray) -> np.ndarray:
    """Extract centroids from label mask.

    Returns (N, 2) array of [y, x] centroids.
    """
    labels = np.unique(masks)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.empty((0, 2), dtype=np.float64)

    centroids = ndimage.center_of_mass(masks > 0, masks, labels)
    return np.array(centroids, dtype=np.float64)


def _measure_gpu_memory() -> float:
    """Get peak GPU memory in MB, or 0 if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def _reset_gpu_memory():
    """Reset GPU memory tracking."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class _CellposeAdapterBase(SegmenterAdapter):
    """Shared base for all Cellpose adapters."""

    def __init__(self, model_type: str, gpu: bool = True, diameter: int = 0):
        self._model_type = model_type
        self._gpu = gpu
        self._diameter = diameter  # 0 = auto-detect
        self._model = None

    def _get_model(self):
        if self._model is None:
            from cellpose import models

            self._model = models.CellposeModel(model_type=self._model_type, gpu=self._gpu)
        return self._model

    def _run_cellpose(self, image: np.ndarray, pixel_size_um: float) -> np.ndarray:
        """Run cellpose and return masks. Separated for testability."""
        model = self._get_model()
        diameter = self._diameter if self._diameter > 0 else None
        masks, _, _ = model.eval_batch(
            [image],
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        return masks[0].astype(np.int32)

    def segment(
        self, image: np.ndarray, pixel_size_um: float = 0.108
    ) -> SegmentationOutput:
        _reset_gpu_memory()
        t0 = time.perf_counter()

        masks = self._run_cellpose(image, pixel_size_um)

        runtime = time.perf_counter() - t0
        peak_mem = _measure_gpu_memory()
        centroids = _centroids_from_masks(masks)

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=len(centroids),
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
            method_name=self.name,
        )


class CellposeSAMAdapter(_CellposeAdapterBase):
    """Cellpose 4 with SAM backbone (nuclei model)."""

    def __init__(self, gpu: bool = True, diameter: int = 0):
        super().__init__(model_type="nuclei", gpu=gpu, diameter=diameter)

    @property
    def name(self) -> str:
        return "cellpose_sam"


class CellposeCyto3Adapter(_CellposeAdapterBase):
    """Cellpose cyto3 model for whole-cell segmentation."""

    def __init__(self, gpu: bool = True, diameter: int = 0):
        super().__init__(model_type="cyto3", gpu=gpu, diameter=diameter)

    @property
    def name(self) -> str:
        return "cellpose_cyto3"

    @property
    def supports_cell_boundaries(self) -> bool:
        return True


class CellposeNucleiAdapter(_CellposeAdapterBase):
    """Cellpose nuclei model (CP3-style, non-SAM)."""

    def __init__(self, gpu: bool = True, diameter: int = 0):
        super().__init__(model_type="nuclei", gpu=gpu, diameter=diameter)

    @property
    def name(self) -> str:
        return "cellpose_nuclei"
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmark_evaluation.py -v
```

Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/benchmark/segmenters/cellpose_adapter.py tests/test_benchmark_evaluation.py
git commit -m "feat(benchmark): add Cellpose SAM/cyto3/nuclei adapters"
```

---

## Task 4: StarDist, InstanSeg, Mesmer, and Native Adapters

**Files:**
- Create: `src/dapidl/benchmark/segmenters/stardist_adapter.py`
- Create: `src/dapidl/benchmark/segmenters/instanseg_adapter.py`
- Create: `src/dapidl/benchmark/segmenters/native_adapter.py`
- Create: `src/dapidl/benchmark/segmenters/mesmer_adapter.py`

- [ ] **Step 1: Install new dependencies**

```bash
uv add stardist instanseg
```

If `stardist` fails due to TF conflicts, try:
```bash
uv add "stardist>=0.9.2" --no-deps
uv add "tensorflow>=2.15"
```

If TF install fails entirely, skip StarDist and Mesmer. Note the failure and move on.

For `deepcell` (Mesmer):
```bash
uv add deepcell
```

Same fallback: skip if dependency hell.

- [ ] **Step 2: Implement StarDist adapter**

Create `src/dapidl/benchmark/segmenters/stardist_adapter.py`:

```python
"""StarDist segmenter adapter for benchmark.

StarDist uses star-convex polygons for nucleus detection.
Requires stardist + tensorflow packages.
"""

import logging
import time

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.segmenters.cellpose_adapter import (
    _centroids_from_masks,
    _measure_gpu_memory,
    _reset_gpu_memory,
)

logger = logging.getLogger(__name__)


class StarDistAdapter(SegmenterAdapter):
    """StarDist 2D fluorescence model for nucleus segmentation."""

    def __init__(self, model_name: str = "2D_versatile_fluo"):
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return "stardist"

    def _get_model(self):
        if self._model is None:
            from stardist.models import StarDist2D

            self._model = StarDist2D.from_pretrained(self._model_name)
        return self._model

    def segment(
        self, image: np.ndarray, pixel_size_um: float = 0.108
    ) -> SegmentationOutput:
        _reset_gpu_memory()
        t0 = time.perf_counter()

        model = self._get_model()

        # StarDist expects normalized float input
        img = image.astype(np.float32)
        p_low, p_high = np.percentile(img, (1, 99.8))
        if p_high > p_low:
            img = (img - p_low) / (p_high - p_low)
        img = np.clip(img, 0, 1)

        masks, details = model.predict_instances(img)
        masks = masks.astype(np.int32)

        runtime = time.perf_counter() - t0
        peak_mem = _measure_gpu_memory()
        centroids = _centroids_from_masks(masks)

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=len(centroids),
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
            method_name=self.name,
        )
```

- [ ] **Step 3: Implement Mesmer adapter**

Create `src/dapidl/benchmark/segmenters/mesmer_adapter.py`:

```python
"""Mesmer/DeepCell segmenter adapter for benchmark.

Mesmer is a panoptic segmentation model trained on tissue data.
Requires deepcell + tensorflow packages.
"""

import logging
import time

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.segmenters.cellpose_adapter import (
    _centroids_from_masks,
    _measure_gpu_memory,
    _reset_gpu_memory,
)

logger = logging.getLogger(__name__)


class MesmerAdapter(SegmenterAdapter):
    """DeepCell Mesmer for nucleus and whole-cell segmentation."""

    def __init__(self):
        self._app = None

    @property
    def name(self) -> str:
        return "mesmer"

    @property
    def supports_cell_boundaries(self) -> bool:
        return True

    def _get_app(self):
        if self._app is None:
            from deepcell.applications import Mesmer

            self._app = Mesmer()
        return self._app

    def segment(
        self, image: np.ndarray, pixel_size_um: float = 0.108
    ) -> SegmentationOutput:
        _reset_gpu_memory()
        t0 = time.perf_counter()

        app = self._get_app()

        # Mesmer expects (batch, H, W, channels) with nuclear + membrane
        # For DAPI-only, use nuclear channel only (membrane=zeros)
        img_4d = np.stack([image, np.zeros_like(image)], axis=-1)
        img_4d = img_4d[np.newaxis, ...]  # Add batch dim

        # resolution in microns/pixel
        result = app.predict(
            img_4d,
            image_mpp=pixel_size_um,
            compartment="nuclear",
        )

        masks = result[0, :, :, 0].astype(np.int32)

        runtime = time.perf_counter() - t0
        peak_mem = _measure_gpu_memory()
        centroids = _centroids_from_masks(masks)

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=len(centroids),
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
            method_name=self.name,
        )
```

- [ ] **Step 4: Implement InstanSeg adapter**

Create `src/dapidl/benchmark/segmenters/instanseg_adapter.py`:

```python
"""InstanSeg segmenter adapter for benchmark.

InstanSeg uses embedding-based instance segmentation.
PyTorch-native, ~60% faster than alternatives.
"""

import logging
import time

import numpy as np

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.segmenters.cellpose_adapter import (
    _centroids_from_masks,
    _measure_gpu_memory,
    _reset_gpu_memory,
)

logger = logging.getLogger(__name__)


class InstanSegAdapter(SegmenterAdapter):
    """InstanSeg embedding-based nucleus/cell segmentation."""

    def __init__(self, gpu: bool = True):
        self._gpu = gpu
        self._model = None

    @property
    def name(self) -> str:
        return "instanseg"

    @property
    def supports_cell_boundaries(self) -> bool:
        return True

    def _get_model(self):
        if self._model is None:
            from instanseg import InstanSeg

            self._model = InstanSeg("instanseg_fluorescence_nuclei_cells")
        return self._model

    def segment(
        self, image: np.ndarray, pixel_size_um: float = 0.108
    ) -> SegmentationOutput:
        _reset_gpu_memory()
        t0 = time.perf_counter()

        model = self._get_model()

        # InstanSeg expects (H, W) or (H, W, C) float32
        img = image.astype(np.float32)

        # Run segmentation - returns labeled array or tuple
        labeled = model.run(img, pixel_size=pixel_size_um)

        # labeled can be (H, W) or tuple of (nuclei, cells)
        if isinstance(labeled, tuple):
            masks = labeled[0].astype(np.int32)  # nuclei
        else:
            masks = np.asarray(labeled).astype(np.int32)

        runtime = time.perf_counter() - t0
        peak_mem = _measure_gpu_memory()
        centroids = _centroids_from_masks(masks)

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=len(centroids),
            runtime_seconds=runtime,
            peak_memory_mb=peak_mem,
            method_name=self.name,
        )
```

- [ ] **Step 5: Implement Native (MERSCOPE platform) adapter**

Create `src/dapidl/benchmark/segmenters/native_adapter.py`:

```python
"""Native MERSCOPE segmenter adapter for benchmark.

Uses the platform's built-in cell segmentation from cell_metadata.csv.
Creates pseudo-masks from bounding boxes since MERSCOPE doesn't export
actual segmentation masks.
"""

import logging
import time

import numpy as np
import polars as pl

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter

logger = logging.getLogger(__name__)


class NativeAdapter(SegmenterAdapter):
    """MERSCOPE native segmentation from cell_metadata.

    Creates circular pseudo-masks from cell centroids and volumes,
    assuming spherical nuclei to estimate radius from volume.
    """

    def __init__(
        self,
        cell_metadata: pl.DataFrame,
        fov_id: int,
        scale: float = 9.259259,
        offset_x: float = 357.2,
        offset_y: float = 2007.97,
    ):
        self._cells = cell_metadata.filter(pl.col("fov") == fov_id)
        self._scale = scale
        self._offset_x = offset_x
        self._offset_y = offset_y

    @property
    def name(self) -> str:
        return "native"

    def segment(
        self, image: np.ndarray, pixel_size_um: float = 0.108
    ) -> SegmentationOutput:
        t0 = time.perf_counter()
        h, w = image.shape

        masks = np.zeros((h, w), dtype=np.int32)
        centroids_list = []

        for idx, row in enumerate(self._cells.iter_rows(named=True), start=1):
            # Convert micron coords to pixel coords in the full mosaic
            px = int(row["center_x"] * self._scale + self._offset_x)
            py = int(row["center_y"] * self._scale + self._offset_y)

            # Estimate radius from volume (assume sphere: V = 4/3 pi r^3)
            volume = row["volume"]
            radius_um = (3 * volume / (4 * np.pi)) ** (1 / 3)
            radius_px = int(radius_um * self._scale)
            radius_px = max(3, min(radius_px, 50))  # Clamp to reasonable range

            # Draw circle on mask
            yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
            circle = xx**2 + yy**2 <= radius_px**2

            y_start = py - radius_px
            x_start = px - radius_px
            y_end = py + radius_px + 1
            x_end = px + radius_px + 1

            # Clip to image bounds
            cy_start = max(0, -y_start)
            cx_start = max(0, -x_start)
            cy_end = circle.shape[0] - max(0, y_end - h)
            cx_end = circle.shape[1] - max(0, x_end - w)

            y_start = max(0, y_start)
            x_start = max(0, x_start)
            y_end = min(h, y_end)
            x_end = min(w, x_end)

            if y_end > y_start and x_end > x_start:
                region = circle[cy_start:cy_end, cx_start:cx_end]
                masks[y_start:y_end, x_start:x_end][region] = idx
                centroids_list.append([float(py), float(px)])

        runtime = time.perf_counter() - t0
        centroids = np.array(centroids_list, dtype=np.float64) if centroids_list else np.empty((0, 2))

        return SegmentationOutput(
            masks=masks,
            centroids=centroids,
            n_cells=len(centroids_list),
            runtime_seconds=runtime,
            peak_memory_mb=0.0,
            method_name=self.name,
        )
```

- [ ] **Step 6: Update segmenters __init__.py**

Update `src/dapidl/benchmark/segmenters/__init__.py`:

```python
"""Segmenter adapters for benchmark framework.

Import adapters with try/except for optional TF dependencies.
"""

from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.segmenters.cellpose_adapter import (
    CellposeCyto3Adapter,
    CellposeNucleiAdapter,
    CellposeSAMAdapter,
)
from dapidl.benchmark.segmenters.native_adapter import NativeAdapter

# Optional TF-based adapters
try:
    from dapidl.benchmark.segmenters.stardist_adapter import StarDistAdapter
except ImportError:
    StarDistAdapter = None  # type: ignore[assignment,misc]

try:
    from dapidl.benchmark.segmenters.mesmer_adapter import MesmerAdapter
except ImportError:
    MesmerAdapter = None  # type: ignore[assignment,misc]

# Optional PyTorch-based adapter
try:
    from dapidl.benchmark.segmenters.instanseg_adapter import InstanSegAdapter
except ImportError:
    InstanSegAdapter = None  # type: ignore[assignment,misc]

__all__ = [
    "SegmentationOutput",
    "SegmenterAdapter",
    "CellposeSAMAdapter",
    "CellposeCyto3Adapter",
    "CellposeNucleiAdapter",
    "StarDistAdapter",
    "MesmerAdapter",
    "InstanSegAdapter",
    "NativeAdapter",
]
```

- [ ] **Step 7: Commit**

```bash
git add src/dapidl/benchmark/segmenters/
git commit -m "feat(benchmark): add StarDist, Mesmer, InstanSeg, and Native adapters"
```

---

## Task 5: Morphometric Evaluation

**Files:**
- Create: `src/dapidl/benchmark/evaluation/morphometric.py`
- Modify: `tests/test_benchmark_evaluation.py`

- [ ] **Step 1: Write test**

Append to `tests/test_benchmark_evaluation.py`:

```python
from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics


def _make_two_cell_mask():
    """Create a mask with two round cells of known size."""
    masks = np.zeros((200, 200), dtype=np.int32)
    yy, xx = np.ogrid[:200, :200]
    # Cell 1: circle at (50, 50) radius 20
    masks[((yy - 50) ** 2 + (xx - 50) ** 2) <= 20**2] = 1
    # Cell 2: circle at (150, 150) radius 15
    masks[((yy - 150) ** 2 + (xx - 150) ** 2) <= 15**2] = 2
    return masks


def test_morphometric_metrics_structure():
    masks = _make_two_cell_mask()
    metrics = compute_morphometric_metrics(masks, pixel_size_um=0.108)
    assert "mean_area_um2" in metrics
    assert "median_area_um2" in metrics
    assert "mean_eccentricity" in metrics
    assert "mean_solidity" in metrics
    assert "size_outlier_rate" in metrics
    assert "n_detected" in metrics


def test_morphometric_area_values():
    masks = _make_two_cell_mask()
    metrics = compute_morphometric_metrics(masks, pixel_size_um=0.108)
    # Cell 1 area ~ pi*20^2 = 1257 px2 ~ 14.7 um2 (at 0.108 um/px)
    # Cell 2 area ~ pi*15^2 = 707 px2 ~ 8.2 um2
    assert metrics["n_detected"] == 2
    assert metrics["mean_area_um2"] > 5.0


def test_morphometric_empty_mask():
    masks = np.zeros((100, 100), dtype=np.int32)
    metrics = compute_morphometric_metrics(masks, pixel_size_um=0.108)
    assert metrics["n_detected"] == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmark_evaluation.py::test_morphometric_metrics_structure -v
```

Expected: FAIL

- [ ] **Step 3: Implement morphometric evaluation**

Create `src/dapidl/benchmark/evaluation/morphometric.py`:

```python
"""Morphometric evaluation metrics for segmentation benchmark.

Computes shape, size, and quality statistics from segmentation masks.
"""

import numpy as np
from skimage.measure import regionprops


def compute_morphometric_metrics(
    masks: np.ndarray,
    pixel_size_um: float = 0.108,
) -> dict:
    """Compute morphometric metrics from a label mask.

    Args:
        masks: (H, W) int32 label image, 0=background
        pixel_size_um: Physical pixel size in microns

    Returns:
        Dictionary of metric name to value
    """
    area_per_pixel_um2 = pixel_size_um**2

    props = regionprops(masks)
    n = len(props)

    if n == 0:
        return {
            "n_detected": 0,
            "mean_area_um2": 0.0,
            "median_area_um2": 0.0,
            "std_area_um2": 0.0,
            "mean_eccentricity": 0.0,
            "mean_solidity": 0.0,
            "size_outlier_rate": 0.0,
            "detection_density_per_1000um2": 0.0,
            "small_debris_count": 0,
            "large_merged_count": 0,
        }

    areas_px = np.array([p.area for p in props])
    areas_um2 = areas_px * area_per_pixel_um2
    eccentricities = np.array([p.eccentricity for p in props])
    solidities = np.array([p.solidity for p in props])

    # Size outliers
    small_threshold_um2 = 20.0  # Debris
    large_threshold_um2 = 500.0  # Merged nuclei
    n_small = int(np.sum(areas_um2 < small_threshold_um2))
    n_large = int(np.sum(areas_um2 > large_threshold_um2))
    outlier_rate = (n_small + n_large) / n

    # Detection density
    image_area_um2 = masks.shape[0] * masks.shape[1] * area_per_pixel_um2
    density = n / image_area_um2 * 1000  # per 1000 um2

    return {
        "n_detected": n,
        "mean_area_um2": float(np.mean(areas_um2)),
        "median_area_um2": float(np.median(areas_um2)),
        "std_area_um2": float(np.std(areas_um2)),
        "mean_eccentricity": float(np.mean(eccentricities)),
        "mean_solidity": float(np.mean(solidities)),
        "size_outlier_rate": float(outlier_rate),
        "detection_density_per_1000um2": float(density),
        "small_debris_count": n_small,
        "large_merged_count": n_large,
        "area_p10_um2": float(np.percentile(areas_um2, 10)),
        "area_p90_um2": float(np.percentile(areas_um2, 90)),
        "eccentricity_p90": float(np.percentile(eccentricities, 90)),
        "solidity_p10": float(np.percentile(solidities, 10)),
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmark_evaluation.py -k morphometric -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/benchmark/evaluation/morphometric.py tests/test_benchmark_evaluation.py
git commit -m "feat(benchmark): add morphometric evaluation metrics"
```

---

## Task 6: Biological Evaluation

**Files:**
- Create: `src/dapidl/benchmark/evaluation/biological.py`
- Modify: `tests/test_benchmark_evaluation.py`

- [ ] **Step 1: Write test**

Append to `tests/test_benchmark_evaluation.py`:

```python
from dapidl.benchmark.evaluation.biological import compute_biological_metrics


def test_biological_metrics_perfect_match():
    """When segmentation matches native exactly, recovery=1.0."""
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:30, 10:30] = 1
    masks[60:80, 60:80] = 2

    native_centroids = np.array([[20.0, 20.0], [70.0, 70.0]])

    metrics = compute_biological_metrics(
        masks=masks,
        native_centroids=native_centroids,
        pixel_size_um=0.108,
    )
    assert metrics["native_recovery_rate"] == 1.0
    assert metrics["n_native"] == 2
    assert metrics["n_recovered"] == 2


def test_biological_metrics_partial_recovery():
    """When segmentation misses cells, recovery < 1.0."""
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:30, 10:30] = 1

    native_centroids = np.array([[20.0, 20.0], [70.0, 70.0]])

    metrics = compute_biological_metrics(
        masks=masks,
        native_centroids=native_centroids,
        pixel_size_um=0.108,
    )
    assert metrics["native_recovery_rate"] == 0.5
    assert metrics["n_recovered"] == 1


def test_biological_metrics_oversegmentation():
    """Detect when one native cell is split across multiple segments."""
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:20, 10:30] = 1
    masks[20:30, 10:30] = 2

    native_centroids = np.array([[20.0, 20.0]])

    metrics = compute_biological_metrics(
        masks=masks,
        native_centroids=native_centroids,
        pixel_size_um=0.108,
    )
    assert metrics["n_native"] == 1
    assert "split_cell_rate" in metrics
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmark_evaluation.py::test_biological_metrics_perfect_match -v
```

Expected: FAIL

- [ ] **Step 3: Implement biological evaluation**

Create `src/dapidl/benchmark/evaluation/biological.py`:

```python
"""Biological evaluation metrics for segmentation benchmark.

Evaluates segmentation quality by comparing to MERSCOPE native cell
assignments. Uses centroid-to-mask matching since we have native
centroids but not native segmentation masks.
"""

import numpy as np


def compute_biological_metrics(
    masks: np.ndarray,
    native_centroids: np.ndarray,
    pixel_size_um: float = 0.108,
    search_radius_px: int = 20,
) -> dict:
    """Compute biological quality metrics.

    Matches native MERSCOPE cell centroids to segmented masks to measure:
    - Recovery rate: fraction of native cells found in segmentation
    - Split-cell rate: native cells matched to multiple segments
    - Under-segmentation: multiple native centroids in one segment

    Args:
        masks: (H, W) int32 label image from segmentation
        native_centroids: (N, 2) float64 [y, x] native cell centroids in pixels
        pixel_size_um: Physical pixel size
        search_radius_px: Radius to search for matching mask label

    Returns:
        Dictionary of metric name to value
    """
    n_native = len(native_centroids)
    h, w = masks.shape

    if n_native == 0:
        return {
            "n_native": 0,
            "n_recovered": 0,
            "native_recovery_rate": 0.0,
            "split_cell_rate": 0.0,
            "underseg_rate": 0.0,
            "n_segments": int(masks.max()),
        }

    # For each native centroid, find which segment label it falls in
    centroid_labels = []
    for cy, cx in native_centroids:
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < h and 0 <= ix < w:
            label = masks[iy, ix]
            if label > 0:
                centroid_labels.append(label)
            else:
                # Search nearby pixels in case centroid is slightly off
                found = False
                for dy in range(-search_radius_px, search_radius_px + 1, 2):
                    for dx in range(-search_radius_px, search_radius_px + 1, 2):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < h and 0 <= nx < w and masks[ny, nx] > 0:
                            centroid_labels.append(masks[ny, nx])
                            found = True
                            break
                    if found:
                        break
                if not found:
                    centroid_labels.append(0)
        else:
            centroid_labels.append(0)

    centroid_labels = np.array(centroid_labels)

    n_recovered = int(np.sum(centroid_labels > 0))
    recovery_rate = n_recovered / n_native if n_native > 0 else 0.0

    # Under-segmentation: multiple native centroids in one segment
    unique_labels, counts = np.unique(centroid_labels[centroid_labels > 0], return_counts=True)
    n_underseg = int(np.sum(counts > 1))
    underseg_rate = n_underseg / len(unique_labels) if len(unique_labels) > 0 else 0.0

    # Split-cell rate: measure via segment count vs native count
    n_segments = int(masks.max())
    split_cell_rate = max(0.0, (n_segments - n_native) / n_native) if n_native > 0 else 0.0

    return {
        "n_native": n_native,
        "n_recovered": n_recovered,
        "native_recovery_rate": float(recovery_rate),
        "split_cell_rate": float(split_cell_rate),
        "underseg_rate": float(underseg_rate),
        "n_segments": n_segments,
        "segments_per_native": float(n_segments / n_native) if n_native > 0 else 0.0,
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmark_evaluation.py -k biological -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/benchmark/evaluation/biological.py tests/test_benchmark_evaluation.py
git commit -m "feat(benchmark): add biological evaluation metrics"
```

---

## Task 7: Cross-Method Evaluation + Instance Matching

**Files:**
- Create: `src/dapidl/benchmark/evaluation/cross_method.py`
- Create: `src/dapidl/benchmark/consensus/instance_matching.py`
- Test: `tests/test_benchmark_consensus.py`

- [ ] **Step 1: Write test for instance matching**

Create `tests/test_benchmark_consensus.py`:

```python
"""Tests for consensus methods and cross-method evaluation."""

import numpy as np

from dapidl.benchmark.consensus.instance_matching import match_instances_iou


def _make_shifted_masks():
    """Two masks with overlapping but shifted cells."""
    m1 = np.zeros((100, 100), dtype=np.int32)
    m1[10:30, 10:30] = 1
    m1[50:70, 50:70] = 2

    m2 = np.zeros((100, 100), dtype=np.int32)
    m2[15:35, 15:35] = 1  # Shifted cell 1
    m2[50:70, 50:70] = 2  # Same cell 2

    return m1, m2


def test_match_instances_perfect():
    m = np.zeros((100, 100), dtype=np.int32)
    m[10:30, 10:30] = 1
    m[50:70, 50:70] = 2

    matches, ious = match_instances_iou(m, m)
    assert len(matches) == 2
    assert all(iou == 1.0 for iou in ious)


def test_match_instances_shifted():
    m1, m2 = _make_shifted_masks()
    matches, ious = match_instances_iou(m1, m2)
    assert len(matches) == 2
    assert any(iou == 1.0 for iou in ious)
    assert any(0 < iou < 1.0 for iou in ious)


def test_match_instances_no_overlap():
    m1 = np.zeros((100, 100), dtype=np.int32)
    m1[10:20, 10:20] = 1
    m2 = np.zeros((100, 100), dtype=np.int32)
    m2[80:90, 80:90] = 1

    matches, ious = match_instances_iou(m1, m2, iou_threshold=0.1)
    assert len(matches) == 0
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmark_consensus.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement instance matching**

Create `src/dapidl/benchmark/consensus/instance_matching.py`:

```python
"""Instance matching utilities for cross-method comparison and consensus.

Matches cells across different segmentation masks using IoU (Intersection
over Union) with Hungarian algorithm for optimal assignment.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _compute_iou_matrix(masks_a: np.ndarray, masks_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between all instances in two label masks.

    Args:
        masks_a: (H, W) int32 label image
        masks_b: (H, W) int32 label image

    Returns:
        (Na, Nb) IoU matrix
    """
    labels_a = np.unique(masks_a)
    labels_a = labels_a[labels_a > 0]
    labels_b = np.unique(masks_b)
    labels_b = labels_b[labels_b > 0]

    if len(labels_a) == 0 or len(labels_b) == 0:
        return np.empty((len(labels_a), len(labels_b)))

    iou_matrix = np.zeros((len(labels_a), len(labels_b)), dtype=np.float64)

    for i, la in enumerate(labels_a):
        mask_a = masks_a == la
        for j, lb in enumerate(labels_b):
            mask_b = masks_b == lb
            intersection = np.sum(mask_a & mask_b)
            if intersection == 0:
                continue
            union = np.sum(mask_a | mask_b)
            iou_matrix[i, j] = intersection / union

    return iou_matrix


def match_instances_iou(
    masks_a: np.ndarray,
    masks_b: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Match instances between two masks using Hungarian algorithm on IoU.

    Args:
        masks_a: (H, W) int32 label image
        masks_b: (H, W) int32 label image
        iou_threshold: Minimum IoU to accept a match

    Returns:
        Tuple of (matches, ious) where matches is list of (label_a, label_b)
        pairs and ious is list of corresponding IoU values.
    """
    labels_a = np.unique(masks_a)
    labels_a = labels_a[labels_a > 0]
    labels_b = np.unique(masks_b)
    labels_b = labels_b[labels_b > 0]

    if len(labels_a) == 0 or len(labels_b) == 0:
        return [], []

    iou_matrix = _compute_iou_matrix(masks_a, masks_b)

    # Hungarian algorithm (minimize cost = 1 - IoU)
    cost_matrix = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    ious = []
    for r, c in zip(row_idx, col_idx):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            matches.append((int(labels_a[r]), int(labels_b[c])))
            ious.append(float(iou))

    return matches, ious
```

- [ ] **Step 4: Implement cross-method evaluation**

Create `src/dapidl/benchmark/evaluation/cross_method.py`:

```python
"""Cross-method agreement metrics for segmentation benchmark."""

import numpy as np

from dapidl.benchmark.consensus.instance_matching import match_instances_iou
from dapidl.benchmark.segmenters.base import SegmentationOutput


def compute_cross_method_metrics(
    results: dict[str, SegmentationOutput],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute cross-method agreement metrics.

    Args:
        results: Mapping of method_name to SegmentationOutput
        iou_threshold: Minimum IoU to count as agreement

    Returns:
        Dictionary with pairwise_iou, consensus_scores, agreement_matrix
    """
    method_names = sorted(results.keys())
    n_methods = len(method_names)

    pairwise_mean_iou = np.zeros((n_methods, n_methods))
    pairwise_match_rate = np.zeros((n_methods, n_methods))

    for i, name_a in enumerate(method_names):
        for j, name_b in enumerate(method_names):
            if i == j:
                pairwise_mean_iou[i, j] = 1.0
                pairwise_match_rate[i, j] = 1.0
                continue
            if i > j:
                pairwise_mean_iou[i, j] = pairwise_mean_iou[j, i]
                pairwise_match_rate[i, j] = pairwise_match_rate[j, i]
                continue

            masks_a = results[name_a].masks
            masks_b = results[name_b].masks
            matches, ious = match_instances_iou(masks_a, masks_b, iou_threshold)

            n_a = results[name_a].n_cells
            n_b = results[name_b].n_cells
            n_max = max(n_a, n_b, 1)

            pairwise_mean_iou[i, j] = float(np.mean(ious)) if ious else 0.0
            pairwise_match_rate[i, j] = len(matches) / n_max

    method_consensus = {}
    for i, name in enumerate(method_names):
        agreements = [
            pairwise_match_rate[i, j] for j in range(n_methods) if i != j
        ]
        method_consensus[name] = float(np.mean(agreements)) if agreements else 0.0

    return {
        "method_names": method_names,
        "pairwise_mean_iou": pairwise_mean_iou.tolist(),
        "pairwise_match_rate": pairwise_match_rate.tolist(),
        "method_consensus_score": method_consensus,
    }
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_benchmark_consensus.py -v
```

Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/benchmark/consensus/instance_matching.py src/dapidl/benchmark/evaluation/cross_method.py tests/test_benchmark_consensus.py
git commit -m "feat(benchmark): add instance matching and cross-method evaluation"
```

---

## Task 8: Consensus Methods (Majority, IoU-Weighted, Topological)

**Files:**
- Create: `src/dapidl/benchmark/consensus/majority_voting.py`
- Create: `src/dapidl/benchmark/consensus/iou_weighted.py`
- Create: `src/dapidl/benchmark/consensus/topological_voting.py`
- Modify: `tests/test_benchmark_consensus.py`

- [ ] **Step 1: Write tests for consensus methods**

Append to `tests/test_benchmark_consensus.py`:

```python
from dapidl.benchmark.segmenters.base import SegmentationOutput
from dapidl.benchmark.consensus.majority_voting import majority_voting_consensus
from dapidl.benchmark.consensus.iou_weighted import iou_weighted_consensus
from dapidl.benchmark.consensus.topological_voting import topological_voting_consensus


def _make_three_method_results():
    """Three methods, two agree on cell positions, one disagrees."""
    m1 = np.zeros((100, 100), dtype=np.int32)
    m1[10:30, 10:30] = 1
    m1[50:70, 50:70] = 2

    m2 = np.zeros((100, 100), dtype=np.int32)
    m2[10:30, 10:30] = 1
    m2[50:70, 50:70] = 2

    m3 = np.zeros((100, 100), dtype=np.int32)
    m3[10:30, 10:30] = 1
    m3[55:75, 55:75] = 2  # Slightly shifted cell 2

    results = {}
    for name, masks in [("a", m1), ("b", m2), ("c", m3)]:
        centroids = np.array([[20.0, 20.0], [60.0, 60.0]])
        results[name] = SegmentationOutput(
            masks=masks, centroids=centroids, n_cells=2,
            runtime_seconds=1.0, peak_memory_mb=0.0, method_name=name,
        )
    return results


def test_majority_voting_returns_output():
    results = _make_three_method_results()
    out = majority_voting_consensus(results)
    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_majority"
    assert out.n_cells >= 1


def test_iou_weighted_returns_output():
    results = _make_three_method_results()
    out = iou_weighted_consensus(results)
    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_iou_weighted"
    assert out.n_cells >= 1


def test_topological_voting_returns_output():
    results = _make_three_method_results()
    out = topological_voting_consensus(results)
    assert isinstance(out, SegmentationOutput)
    assert out.method_name == "consensus_topological"
    assert out.n_cells >= 1
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_benchmark_consensus.py::test_majority_voting_returns_output -v
```

Expected: FAIL

- [ ] **Step 3: Implement majority voting**

Create `src/dapidl/benchmark/consensus/majority_voting.py`:

```python
"""Majority voting consensus segmentation.

For each pixel, assigns foreground if the majority of methods agree.
Connected components become the consensus instances.
"""

import time

import numpy as np
from scipy import ndimage

from dapidl.benchmark.segmenters.base import SegmentationOutput


def majority_voting_consensus(
    results: dict[str, SegmentationOutput],
    min_agreement: float = 0.5,
) -> SegmentationOutput:
    """Create consensus segmentation via pixel-level majority voting.

    Args:
        results: method_name to SegmentationOutput
        min_agreement: Minimum fraction of methods that must agree

    Returns:
        Consensus SegmentationOutput
    """
    t0 = time.perf_counter()

    method_names = sorted(results.keys())
    all_masks = [results[name].masks for name in method_names]
    n_methods = len(all_masks)
    h, w = all_masks[0].shape

    # Binary foreground voting
    foreground_votes = np.zeros((h, w), dtype=np.int32)
    for masks in all_masks:
        foreground_votes += (masks > 0).astype(np.int32)

    min_votes = max(1, int(n_methods * min_agreement))
    consensus_fg = foreground_votes >= min_votes

    # Label connected components
    consensus_masks, n_labels = ndimage.label(consensus_fg)
    consensus_masks = consensus_masks.astype(np.int32)

    if n_labels > 0:
        labels = np.arange(1, n_labels + 1)
        centroids = np.array(
            ndimage.center_of_mass(consensus_fg, consensus_masks, labels),
            dtype=np.float64,
        )
    else:
        centroids = np.empty((0, 2), dtype=np.float64)

    runtime = time.perf_counter() - t0

    return SegmentationOutput(
        masks=consensus_masks,
        centroids=centroids,
        n_cells=n_labels,
        runtime_seconds=runtime,
        peak_memory_mb=0.0,
        method_name="consensus_majority",
    )
```

- [ ] **Step 4: Implement IoU-weighted consensus**

Create `src/dapidl/benchmark/consensus/iou_weighted.py`:

```python
"""IoU-weighted consensus segmentation.

Each method is weighted by its morphometric quality score (solidity).
The weighted average of foreground masks produces the consensus.
"""

import time

import numpy as np
from scipy import ndimage

from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics
from dapidl.benchmark.segmenters.base import SegmentationOutput


def iou_weighted_consensus(
    results: dict[str, SegmentationOutput],
    pixel_size_um: float = 0.108,
) -> SegmentationOutput:
    """Create consensus via quality-weighted mask averaging.

    Args:
        results: method_name to SegmentationOutput
        pixel_size_um: Physical pixel size for morphometric scoring

    Returns:
        Consensus SegmentationOutput
    """
    t0 = time.perf_counter()

    method_names = sorted(results.keys())
    all_masks = [results[name].masks for name in method_names]
    h, w = all_masks[0].shape

    # Compute per-method quality weights
    weights = []
    for masks in all_masks:
        metrics = compute_morphometric_metrics(masks, pixel_size_um)
        solidity = metrics["mean_solidity"]
        outlier_rate = metrics["size_outlier_rate"]
        weight = solidity * (1.0 - outlier_rate) if metrics["n_detected"] > 0 else 0.01
        weights.append(max(weight, 0.01))

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted foreground voting
    weighted_fg = np.zeros((h, w), dtype=np.float64)
    for masks, weight in zip(all_masks, weights):
        weighted_fg += (masks > 0).astype(np.float64) * weight

    consensus_fg = weighted_fg >= 0.5

    consensus_masks, n_labels = ndimage.label(consensus_fg)
    consensus_masks = consensus_masks.astype(np.int32)

    if n_labels > 0:
        labels = np.arange(1, n_labels + 1)
        centroids = np.array(
            ndimage.center_of_mass(consensus_fg, consensus_masks, labels),
            dtype=np.float64,
        )
    else:
        centroids = np.empty((0, 2), dtype=np.float64)

    runtime = time.perf_counter() - t0

    return SegmentationOutput(
        masks=consensus_masks,
        centroids=centroids,
        n_cells=n_labels,
        runtime_seconds=runtime,
        peak_memory_mb=0.0,
        method_name="consensus_iou_weighted",
        metadata={"weights": dict(zip(method_names, weights))},
    )
```

- [ ] **Step 5: Implement topological voting**

Create `src/dapidl/benchmark/consensus/topological_voting.py`:

```python
"""Topological voting consensus segmentation.

Divides image into local patches, selects the best method per patch
based on local morphometric quality, and stitches results.
"""

import time

import numpy as np
from scipy import ndimage

from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics
from dapidl.benchmark.segmenters.base import SegmentationOutput


def _score_patch(masks_patch: np.ndarray, pixel_size_um: float) -> float:
    """Score a local patch based on morphometric quality."""
    if masks_patch.max() == 0:
        return 0.0
    metrics = compute_morphometric_metrics(masks_patch, pixel_size_um)
    if metrics["n_detected"] == 0:
        return 0.0
    return metrics["mean_solidity"] * (1.0 - metrics["size_outlier_rate"])


def topological_voting_consensus(
    results: dict[str, SegmentationOutput],
    patch_size: int = 128,
    overlap: int = 64,
    pixel_size_um: float = 0.108,
) -> SegmentationOutput:
    """Create consensus via local best-method selection.

    Args:
        results: method_name to SegmentationOutput
        patch_size: Size of local voting patches
        overlap: Overlap between patches
        pixel_size_um: Physical pixel size

    Returns:
        Consensus SegmentationOutput
    """
    t0 = time.perf_counter()

    method_names = sorted(results.keys())
    all_masks = {name: results[name].masks for name in method_names}
    h, w = next(iter(all_masks.values())).shape

    consensus = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    stride = patch_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            # Score each method on this patch
            best_score = -1.0
            best_method = method_names[0]

            for name in method_names:
                patch = all_masks[name][y:y_end, x:x_end]
                score = _score_patch(patch, pixel_size_um)
                if score > best_score:
                    best_score = score
                    best_method = name

            # Copy best method's patch, relabeling to avoid conflicts
            patch = all_masks[best_method][y:y_end, x:x_end]
            unique_labels = np.unique(patch)
            unique_labels = unique_labels[unique_labels > 0]

            for old_label in unique_labels:
                region = patch == old_label
                write_mask = region & (consensus[y:y_end, x:x_end] == 0)
                if write_mask.any():
                    consensus[y:y_end, x:x_end][write_mask] = next_label
                    next_label += 1

    # Relabel to ensure contiguous labels
    consensus_relabeled, n_labels = ndimage.label(consensus > 0)
    consensus_relabeled = consensus_relabeled.astype(np.int32)

    if n_labels > 0:
        labels = np.arange(1, n_labels + 1)
        centroids = np.array(
            ndimage.center_of_mass(consensus_relabeled > 0, consensus_relabeled, labels),
            dtype=np.float64,
        )
    else:
        centroids = np.empty((0, 2), dtype=np.float64)

    runtime = time.perf_counter() - t0

    return SegmentationOutput(
        masks=consensus_relabeled,
        centroids=centroids,
        n_cells=n_labels,
        runtime_seconds=runtime,
        peak_memory_mb=0.0,
        method_name="consensus_topological",
    )
```

- [ ] **Step 6: Run all consensus tests**

```bash
uv run pytest tests/test_benchmark_consensus.py -v
```

Expected: 6 PASS

- [ ] **Step 7: Commit**

```bash
git add src/dapidl/benchmark/consensus/ tests/test_benchmark_consensus.py
git commit -m "feat(benchmark): add majority, IoU-weighted, and topological voting consensus"
```

---

## Task 9: Reporting Module

**Files:**
- Create: `src/dapidl/benchmark/reporting.py`

- [ ] **Step 1: Implement reporting**

Create `src/dapidl/benchmark/reporting.py`:

```python
"""Reporting module for segmentation benchmark.

Generates markdown reports, comparison tables, and visualization plots.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_report(
    all_metrics: dict,
    output_dir: Path,
) -> Path:
    """Generate benchmark report with tables and plots.

    Args:
        all_metrics: Nested dict of {method_name: {fov_label: {metric: value}}}
        output_dir: Directory to write report and plots

    Returns:
        Path to generated report.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    lines = ["# MERSCOPE Segmentation Benchmark Report\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Method | Avg Cells | Avg Solidity | Avg Recovery | Avg Runtime (s) |")
    lines.append("|--------|-----------|-------------|-------------|----------------|")

    method_summaries = {}
    for method_name, fov_data in sorted(all_metrics.items()):
        cells = []
        solidity = []
        recovery = []
        runtime = []
        for fov_label, metrics in fov_data.items():
            morph = metrics.get("morphometric", {})
            bio = metrics.get("biological", {})
            prac = metrics.get("practical", {})
            if morph.get("n_detected", 0) > 0:
                cells.append(morph["n_detected"])
                solidity.append(morph.get("mean_solidity", 0))
            if bio.get("n_native", 0) > 0:
                recovery.append(bio.get("native_recovery_rate", 0))
            runtime.append(prac.get("runtime_seconds", 0))

        avg_cells = np.mean(cells) if cells else 0
        avg_solidity = np.mean(solidity) if solidity else 0
        avg_recovery = np.mean(recovery) if recovery else 0
        avg_runtime = np.mean(runtime) if runtime else 0

        method_summaries[method_name] = {
            "avg_cells": avg_cells,
            "avg_solidity": avg_solidity,
            "avg_recovery": avg_recovery,
            "avg_runtime": avg_runtime,
        }

        lines.append(
            f"| {method_name} | {avg_cells:.0f} | {avg_solidity:.3f} | "
            f"{avg_recovery:.3f} | {avg_runtime:.1f} |"
        )

    # Per-FOV breakdown
    lines.append("\n## Per-FOV Breakdown\n")
    for method_name, fov_data in sorted(all_metrics.items()):
        lines.append(f"### {method_name}\n")
        lines.append("| FOV | Cells | Area um2 | Solidity | Recovery | Runtime |")
        lines.append("|-----|-------|----------|----------|----------|---------|")
        for fov_label, metrics in sorted(fov_data.items()):
            morph = metrics.get("morphometric", {})
            bio = metrics.get("biological", {})
            prac = metrics.get("practical", {})
            lines.append(
                f"| {fov_label} | {morph.get('n_detected', 0)} | "
                f"{morph.get('mean_area_um2', 0):.1f} | "
                f"{morph.get('mean_solidity', 0):.3f} | "
                f"{bio.get('native_recovery_rate', 0):.3f} | "
                f"{prac.get('runtime_seconds', 0):.1f} |"
            )
        lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)

    # Save raw metrics as JSON
    metrics_path = output_dir / "all_metrics.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_convert)

    # Generate plots
    _generate_plots(method_summaries, output_dir)

    logger.info("Report saved to %s", report_path)
    return report_path


def _generate_plots(method_summaries: dict, output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        methods = list(method_summaries.keys())

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        metrics_to_plot = [
            ("avg_cells", "Avg Cells Detected"),
            ("avg_solidity", "Avg Solidity"),
            ("avg_recovery", "Avg Native Recovery"),
            ("avg_runtime", "Avg Runtime (s)"),
        ]

        for ax, (key, title) in zip(axes, metrics_to_plot):
            values = [method_summaries[m][key] for m in methods]
            bars = ax.barh(methods, values)
            ax.set_title(title)
            ax.set_xlim(0, max(values) * 1.2 if values else 1)

            # Color best value green
            if key == "avg_runtime":
                best_idx = int(np.argmin(values)) if values else 0
            else:
                best_idx = int(np.argmax(values)) if values else 0
            if bars:
                bars[best_idx].set_color("green")

        plt.tight_layout()
        plt.savefig(output_dir / "comparison_chart.png", dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
```

- [ ] **Step 2: Commit**

```bash
git add src/dapidl/benchmark/reporting.py
git commit -m "feat(benchmark): add reporting module with markdown tables and plots"
```

---

## Task 10: Benchmark Runner (Orchestrator)

**Files:**
- Create: `src/dapidl/benchmark/runner.py`
- Modify: `src/dapidl/benchmark/__init__.py`

- [ ] **Step 1: Implement BenchmarkRunner**

Create `src/dapidl/benchmark/runner.py`:

```python
"""Benchmark runner: orchestrates FOV selection, segmentation, evaluation, and reporting."""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from dapidl.benchmark.evaluation.biological import compute_biological_metrics
from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics
from dapidl.benchmark.fov_selector import FOVTile, extract_fov_tile, select_fovs
from dapidl.benchmark.reporting import generate_report
from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates the full segmentation benchmark pipeline."""

    def __init__(
        self,
        dapi_path: Path,
        cell_metadata_path: Path,
        output_dir: Path,
        pixel_size_um: float = 0.108,
        scale: float = 9.259259,
        offset_x: float = 357.2,
        offset_y: float = 2007.97,
    ):
        self.dapi_path = dapi_path
        self.cell_metadata_path = cell_metadata_path
        self.output_dir = output_dir
        self.pixel_size_um = pixel_size_um
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        segmenters: list[SegmenterAdapter],
        n_fovs: int = 5,
        run_consensus: bool = True,
    ) -> Path:
        """Run the full benchmark.

        Args:
            segmenters: List of segmenter adapters to benchmark
            n_fovs: Number of FOVs to select
            run_consensus: Whether to run consensus methods

        Returns:
            Path to generated report
        """
        logger.info("Loading cell metadata from %s", self.cell_metadata_path)
        cell_metadata = pl.read_csv(self.cell_metadata_path)

        # Step 1: FOV selection
        logger.info("Selecting %d representative FOVs", n_fovs)
        fovs = select_fovs(
            cell_metadata,
            n_fovs=n_fovs,
            scale=self.scale,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
        )

        # Step 2: Extract FOV tiles
        fov_tiles: dict[str, np.ndarray] = {}
        fov_dir = self.output_dir / "fovs"
        fov_dir.mkdir(exist_ok=True)

        for fov in fovs:
            logger.info("Extracting FOV '%s' (fov_id=%d, %d cells)", fov.label, fov.fov_id, fov.n_cells)
            tile = extract_fov_tile(self.dapi_path, fov)
            fov_tiles[fov.label] = tile

            import tifffile
            tifffile.imwrite(fov_dir / f"fov_{fov.label}.tif", tile)

        # Step 3: Run segmentation
        all_results: dict[str, dict[str, SegmentationOutput]] = {}

        for segmenter in segmenters:
            logger.info("Running segmenter: %s", segmenter.name)
            method_results = {}

            for fov in fovs:
                tile = fov_tiles[fov.label]
                logger.info("  FOV '%s' (%dx%d)", fov.label, tile.shape[0], tile.shape[1])

                try:
                    result = segmenter.segment(tile, self.pixel_size_um)
                    method_results[fov.label] = result
                    logger.info(
                        "  -> %d cells in %.1fs (%.0f MB GPU)",
                        result.n_cells,
                        result.runtime_seconds,
                        result.peak_memory_mb,
                    )
                except Exception as e:
                    logger.error("  -> FAILED: %s", e)
                    method_results[fov.label] = SegmentationOutput(
                        masks=np.zeros_like(tile, dtype=np.int32),
                        centroids=np.empty((0, 2)),
                        n_cells=0,
                        runtime_seconds=0.0,
                        peak_memory_mb=0.0,
                        method_name=segmenter.name,
                        metadata={"error": str(e)},
                    )

            all_results[segmenter.name] = method_results

            # Save results
            results_dir = self.output_dir / "results" / segmenter.name
            results_dir.mkdir(parents=True, exist_ok=True)
            for fov_label, result in method_results.items():
                np.save(results_dir / f"fov_{fov_label}_masks.npy", result.masks)

        # Step 4: Consensus methods
        if run_consensus and len(all_results) >= 2:
            logger.info("Running consensus methods")
            from dapidl.benchmark.consensus.iou_weighted import iou_weighted_consensus
            from dapidl.benchmark.consensus.majority_voting import majority_voting_consensus
            from dapidl.benchmark.consensus.topological_voting import topological_voting_consensus

            for fov in fovs:
                fov_results = {
                    name: results[fov.label]
                    for name, results in all_results.items()
                    if fov.label in results and results[fov.label].n_cells > 0
                }
                if len(fov_results) < 2:
                    continue

                for consensus_fn, consensus_name in [
                    (majority_voting_consensus, "consensus_majority"),
                    (iou_weighted_consensus, "consensus_iou_weighted"),
                    (topological_voting_consensus, "consensus_topological"),
                ]:
                    try:
                        result = consensus_fn(fov_results)
                        all_results.setdefault(consensus_name, {})[fov.label] = result
                        logger.info(
                            "  %s on '%s': %d cells",
                            consensus_name, fov.label, result.n_cells,
                        )
                    except Exception as e:
                        logger.error("  %s FAILED on '%s': %s", consensus_name, fov.label, e)

        # Step 5: Evaluation
        logger.info("Computing evaluation metrics")
        all_metrics: dict[str, dict[str, dict]] = {}

        for method_name, method_results in all_results.items():
            all_metrics[method_name] = {}

            for fov in fovs:
                if fov.label not in method_results:
                    continue
                result = method_results[fov.label]

                morph_metrics = compute_morphometric_metrics(
                    result.masks, self.pixel_size_um
                )

                # Biological: native centroids relative to FOV tile origin
                fov_cells = cell_metadata.filter(pl.col("fov") == fov.fov_id)
                native_centroids_px = np.column_stack([
                    fov_cells["center_y"].to_numpy() * self.scale + self.offset_y - fov.pixel_bbox[0],
                    fov_cells["center_x"].to_numpy() * self.scale + self.offset_x - fov.pixel_bbox[2],
                ])
                bio_metrics = compute_biological_metrics(
                    result.masks, native_centroids_px, self.pixel_size_um
                )

                prac_metrics = {
                    "runtime_seconds": result.runtime_seconds,
                    "peak_memory_mb": result.peak_memory_mb,
                    "n_detected": result.n_cells,
                }

                all_metrics[method_name][fov.label] = {
                    "morphometric": morph_metrics,
                    "biological": bio_metrics,
                    "practical": prac_metrics,
                }

        # Step 6: Cross-method evaluation
        logger.info("Computing cross-method agreement")
        from dapidl.benchmark.evaluation.cross_method import compute_cross_method_metrics

        cross_method = {}
        for fov in fovs:
            fov_results = {
                name: results[fov.label]
                for name, results in all_results.items()
                if fov.label in results and results[fov.label].n_cells > 0
            }
            if len(fov_results) >= 2:
                cross_method[fov.label] = compute_cross_method_metrics(fov_results)

        eval_dir = self.output_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        with open(eval_dir / "cross_method_metrics.json", "w") as f:
            json.dump(cross_method, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

        # Step 7: Generate report
        logger.info("Generating report")
        report_path = generate_report(all_metrics, self.output_dir)

        logger.info("Benchmark complete! Report at %s", report_path)
        return report_path
```

- [ ] **Step 2: Update benchmark __init__.py**

Update `src/dapidl/benchmark/__init__.py`:

```python
"""Segmentation benchmark framework for comparing methods on MERSCOPE data."""

from dapidl.benchmark.runner import BenchmarkRunner
from dapidl.benchmark.segmenters.base import SegmentationOutput, SegmenterAdapter

__all__ = ["BenchmarkRunner", "SegmentationOutput", "SegmenterAdapter"]
```

- [ ] **Step 3: Commit**

```bash
git add src/dapidl/benchmark/runner.py src/dapidl/benchmark/__init__.py
git commit -m "feat(benchmark): add BenchmarkRunner orchestrator"
```

---

## Task 11: CLI Entry Point Script

**Files:**
- Create: `scripts/run_segmentation_benchmark.py`

- [ ] **Step 1: Create the benchmark script**

Create `scripts/run_segmentation_benchmark.py`:

```python
#!/usr/bin/env python3
"""Run MERSCOPE segmentation grand benchmark.

Usage:
    uv run python scripts/run_segmentation_benchmark.py
    uv run python scripts/run_segmentation_benchmark.py --methods cellpose_sam,stardist
    uv run python scripts/run_segmentation_benchmark.py --n-fovs 3 --no-consensus
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths
MERSCOPE_DIR = Path("/mnt/work/datasets/raw/merscope/merscope-breast")
DAPI_PATH = MERSCOPE_DIR / "images" / "mosaic_DAPI_z3.tif"
CELL_METADATA_PATH = MERSCOPE_DIR / "cell_metadata.csv"
OUTPUT_DIR = Path("pipeline_output/segmentation_benchmark")

AVAILABLE_METHODS = [
    "cellpose_sam",
    "cellpose_cyto3",
    "cellpose_nuclei",
    "stardist",
    "mesmer",
    "instanseg",
]


def get_adapter(name: str):
    """Instantiate a segmenter adapter by name."""
    if name == "cellpose_sam":
        from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter
        return CellposeSAMAdapter()
    elif name == "cellpose_cyto3":
        from dapidl.benchmark.segmenters.cellpose_adapter import CellposeCyto3Adapter
        return CellposeCyto3Adapter()
    elif name == "cellpose_nuclei":
        from dapidl.benchmark.segmenters.cellpose_adapter import CellposeNucleiAdapter
        return CellposeNucleiAdapter()
    elif name == "stardist":
        from dapidl.benchmark.segmenters.stardist_adapter import StarDistAdapter
        return StarDistAdapter()
    elif name == "mesmer":
        from dapidl.benchmark.segmenters.mesmer_adapter import MesmerAdapter
        return MesmerAdapter()
    elif name == "instanseg":
        from dapidl.benchmark.segmenters.instanseg_adapter import InstanSegAdapter
        return InstanSegAdapter()
    else:
        raise ValueError(f"Unknown method: {name}")


def main():
    parser = argparse.ArgumentParser(description="MERSCOPE Segmentation Benchmark")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=f"Comma-separated list of methods. Available: {','.join(AVAILABLE_METHODS)}",
    )
    parser.add_argument("--n-fovs", type=int, default=5)
    parser.add_argument("--no-consensus", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dapi-path", type=Path, default=DAPI_PATH)
    parser.add_argument("--cell-metadata", type=Path, default=CELL_METADATA_PATH)
    args = parser.parse_args()

    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",")]
    else:
        method_names = AVAILABLE_METHODS

    # Instantiate adapters
    segmenters = []
    for name in method_names:
        try:
            adapter = get_adapter(name)
            segmenters.append(adapter)
            logger.info("Loaded: %s", name)
        except (ImportError, Exception) as e:
            logger.warning("Skipping %s: %s", name, e)

    if not segmenters:
        logger.error("No segmenters available!")
        return

    logger.info(
        "Running benchmark with %d methods on %d FOVs",
        len(segmenters),
        args.n_fovs,
    )

    # Load transform
    transform_path = args.dapi_path.parent / "micron_to_mosaic_pixel_transform.csv"
    scale, offset_x, offset_y = 9.259259, 357.2, 2007.97
    if transform_path.exists():
        from dapidl.benchmark.fov_selector import load_transform
        scale, offset_x, offset_y = load_transform(transform_path)
        logger.info("Loaded transform: scale=%.3f, offset=(%.1f, %.1f)", scale, offset_x, offset_y)

    from dapidl.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        dapi_path=args.dapi_path,
        cell_metadata_path=args.cell_metadata,
        output_dir=args.output_dir,
        pixel_size_um=0.108,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    report = runner.run(
        segmenters=segmenters,
        n_fovs=args.n_fovs,
        run_consensus=not args.no_consensus,
    )

    logger.info("Done! Report: %s", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/run_segmentation_benchmark.py
git add scripts/run_segmentation_benchmark.py
git commit -m "feat(benchmark): add CLI entry point script"
```

---

## Task 12: Install Dependencies and Smoke Test

- [ ] **Step 1: Install optional segmentation packages**

Try each, log failures:

```bash
uv add stardist
uv add instanseg
uv add deepcell
```

If any fail, note which and continue.

- [ ] **Step 2: Smoke test imports**

```bash
uv run python -c "
from dapidl.benchmark import BenchmarkRunner, SegmentationOutput, SegmenterAdapter
from dapidl.benchmark.fov_selector import select_fovs, FOVTile
from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter
from dapidl.benchmark.evaluation.morphometric import compute_morphometric_metrics
from dapidl.benchmark.evaluation.biological import compute_biological_metrics
from dapidl.benchmark.consensus.majority_voting import majority_voting_consensus
from dapidl.benchmark.reporting import generate_report
print('All core imports OK')
"
```

- [ ] **Step 3: Run all tests**

```bash
uv run pytest tests/test_benchmark_evaluation.py tests/test_benchmark_fov_selector.py tests/test_benchmark_consensus.py -v
```

Expected: All PASS

- [ ] **Step 4: Commit dependency changes**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(benchmark): add stardist, instanseg, deepcell dependencies"
```

---

## Task 13: Run the Benchmark

- [ ] **Step 1: Check GPU and memory**

```bash
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
free -h
```

- [ ] **Step 2: Run with Cellpose methods first (guaranteed to work)**

```bash
uv run python scripts/run_segmentation_benchmark.py \
    --methods cellpose_sam,cellpose_cyto3,cellpose_nuclei \
    --n-fovs 5 \
    --output-dir pipeline_output/segmentation_benchmark
```

- [ ] **Step 3: If StarDist/InstanSeg installed, run full benchmark**

```bash
uv run python scripts/run_segmentation_benchmark.py \
    --methods cellpose_sam,cellpose_cyto3,cellpose_nuclei,stardist,instanseg \
    --n-fovs 5 \
    --output-dir pipeline_output/segmentation_benchmark
```

- [ ] **Step 4: Review results**

```bash
cat pipeline_output/segmentation_benchmark/report.md
```

- [ ] **Step 5: Commit results**

```bash
git add pipeline_output/segmentation_benchmark/report.md
git add pipeline_output/segmentation_benchmark/all_metrics.json
git commit -m "results: MERSCOPE segmentation benchmark initial run"
```
