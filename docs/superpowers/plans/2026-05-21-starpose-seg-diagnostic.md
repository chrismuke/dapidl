# starpose Segmentation Diagnostic (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-segment the 6 breast sources with starpose (adaptive nuclei + proseg cells) on representative FOVs, compare masks against the source segmentation, and compare QC of patches built on our segmentation vs the source's.

**Architecture:** A dapidl orchestrator (`scripts/seg_diagnostic.py`) drives reusable units in `src/dapidl/seg_eval/` (per-platform source loaders, mask comparison, QC comparison) over the starpose library (`segment_multimodal`, `select_fovs`, `compute_iou_matrix`, `compute_morphometric`) and `starpose.qc.ClassicalQualityScorer`. starpose stays platform-agnostic; all Xenium/STHELAR specifics live in dapidl.

**Tech Stack:** Python 3.10+, numpy, polars, tifffile (OME-TIFF), spatialdata (STHELAR zarr), shapely + skimage.draw (polygon raster), matplotlib, starpose (editable), proseg (Rust CLI, installed in Task 0).

**Spec:** `docs/superpowers/specs/2026-05-21-starpose-seg-diagnostic-design.md`

**Repos:** dapidl `/mnt/work/git/dapidl`; starpose (editable) `/home/chrism/git/starpose`. Run from dapidl with `uv run`.

**Confirmed facts (do not re-derive):**
- starpose `core.segment_multimodal(modalities: ModalityBundle, gpu=True, nucleus_method=None, expansion_method=None) -> SegmentationResult` (BOTH level: `.nucleus_masks`, `.cell_masks`, `.nucleus_centroids` [y,x], `.cell_centroids`).
- `ModalityBundle(dapi=2D uint16, he=None, membrane=None, polyt=None, transcripts=polars df[x,y,gene] in px, pixel_size=µm/px, platform=str)`.
- `starpose.benchmark.fov_selector.select_fovs(centroids[y,x]_px, image_shape, pixel_size, n_fovs, tile_size_px) -> list[FOVTile(label,bbox=(y0,x0,y1,x1),n_cells,density)]`; `extract_tile(image, fov)`.
- `starpose.consensus.matching.compute_iou_matrix(masks_a, masks_b) -> (max_label_a, max_label_b) IoU matrix`.
- `starpose.evaluate.morphometric.compute_morphometric(masks, pixel_size) -> dict(n_detected, mean_area_um2, median_area_um2, mean_eccentricity, mean_solidity, size_outlier_rate, density, ...)`.
- `starpose.qc.ClassicalQualityScorer().fit_reference(patches)->NormRef`; `.score_batch(patches, ref)->list[QualityScore(focus_score,detection_score,qc_score,metrics)]`.
- Xenium DAPI: `<src>/morphology_focus.ome.tif` (YX uint16, px=0.2125). Source polygons: `<src>/outs/nucleus_boundaries.parquet` & `cell_boundaries.parquet` (cols `cell_id`,`vertex_x`,`vertex_y` in µm). Transcripts: `<src>/outs/transcripts.parquet` (cols `x_location`,`y_location`,`feature_name` in µm). Centroids: `<src>/outs/cells.parquet` (`x_centroid`,`y_centroid` µm).
- STHELAR sdata: `<src>.zarr/<src>.zarr/` → `images['morpho']`, `shapes['nucleus_boundaries']`,`shapes['cell_boundaries']` (GeoDataFrame, shapely polygons), `points['st']`, `tables['table_combined']`.

**Source registry (used across tasks):**
```python
SOURCES = {
    "xenium_rep1": {"kind": "xenium", "root": "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1"},
    "xenium_rep2": {"kind": "xenium", "root": "/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2"},
    "sthelar_breast_s0": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/sdata_breast_s0.zarr"},
    "sthelar_breast_s1": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s1.zarr/sdata_breast_s1.zarr"},
    "sthelar_breast_s3": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s3.zarr/sdata_breast_s3.zarr"},
    "sthelar_breast_s6": {"kind": "sthelar", "zarr": "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s6.zarr/sdata_breast_s6.zarr"},
}
```

---

## Task 0: Install proseg (cell expansion backend)

**Files:** none (environment setup).

- [ ] **Step 1: Install proseg via cargo**

Run: `cargo install proseg 2>&1 | tail -5`
Expected: builds and installs to `~/.cargo/bin/proseg` (a few minutes).

- [ ] **Step 2: Verify it's callable the way starpose expects**

Run: `proseg --version`
Expected: prints a version (starpose's `_check_proseg_available()` runs exactly this).

- [ ] **Step 3: Record the outcome**

If Step 2 fails, the orchestrator (Task 5) must use `expansion_method="watershed"`. Record which expander is active:
```bash
proseg --version >/dev/null 2>&1 && echo "EXPANDER=proseg" || echo "EXPANDER=watershed (proseg unavailable)"
```
No commit (environment only).

---

## Task 1: seg_eval package + Xenium source loader

**Files:**
- Create: `src/dapidl/seg_eval/__init__.py`
- Create: `src/dapidl/seg_eval/source_masks.py`
- Test: `tests/test_seg_eval_xenium.py`

- [ ] **Step 1: Write the failing test (polygon rasterization is the unit-testable core)**

```python
"""Tests for the Xenium source loader's polygon rasterization."""
import numpy as np
import polars as pl
from dapidl.seg_eval.source_masks import rasterize_polygons


def test_rasterize_two_squares():
    # Two cells as axis-aligned squares (vertices in pixel coords already).
    df = pl.DataFrame({
        "cell_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "px": [2, 6, 6, 2, 10, 14, 14, 10],
        "py": [2, 2, 6, 6, 10, 10, 14, 14],
    })
    mask = rasterize_polygons(df, id_col="cell_id", x_col="px", y_col="py",
                              bbox=(0, 0, 20, 20))
    assert mask.shape == (20, 20)
    labels = set(np.unique(mask)) - {0}
    assert len(labels) == 2          # two distinct instances
    assert mask[4, 4] != 0 and mask[12, 12] != 0
    assert mask[4, 4] != mask[12, 12]
```

- [ ] **Step 2: Run, verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_xenium.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'dapidl.seg_eval'`

- [ ] **Step 3: Implement**

Create `src/dapidl/seg_eval/__init__.py`:
```python
"""Segmentation diagnostic: source-mask loaders, comparison, QC comparison."""
```

Create `src/dapidl/seg_eval/source_masks.py`:
```python
"""Load source (built-in) segmentation masks + DAPI + transcripts per source.

Two platforms: Xenium (OME-TIFF DAPI + parquet polygons in microns) and
STHELAR (SpatialData zarr: morpho image + shapes polygons + points transcripts).
All returned masks/centroids are in PIXEL coordinates of the DAPI raster.
"""
from pathlib import Path

import numpy as np
import polars as pl
from skimage.draw import polygon as sk_polygon

XENIUM_PX = 0.2125  # microns per pixel (Xenium morphology)


def rasterize_polygons(df, id_col: str, x_col: str, y_col: str,
                       bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Rasterize per-instance polygons (pixel coords) into a label mask for a bbox.

    df: long-format polars DF, one row per polygon vertex, grouped by id_col.
    bbox: (y0, x0, y1, x1) in pixels. Returns (y1-y0, x1-x0) int32 label mask.
    """
    y0, x0, y1, x1 = bbox
    h, w = y1 - y0, x1 - x0
    mask = np.zeros((h, w), dtype=np.int32)
    label = 0
    for _id, grp in df.group_by(id_col, maintain_order=True):
        xs = grp[x_col].to_numpy() - x0
        ys = grp[y_col].to_numpy() - y0
        # skip polygons fully outside the bbox
        if xs.max() < 0 or ys.max() < 0 or xs.min() >= w or ys.min() >= h:
            continue
        rr, cc = sk_polygon(ys, xs, shape=(h, w))
        if rr.size == 0:
            continue
        label += 1
        mask[rr, cc] = label
    return mask


def load_xenium(root: str | Path):
    """Return a dict with the Xenium source's DAPI, polygons (px), transcripts, centroids.

    Keys: dapi (lazy callable -> 2D array), pixel_size, nucleus_polys/cell_polys
    (polars long-format with px columns), transcripts (polars x,y px + gene),
    centroids (Nx2 [y,x] px).
    """
    root = Path(root)
    outs = root / "outs"
    import tifffile

    def _dapi():
        return tifffile.imread(str(root / "morphology_focus.ome.tif"))

    def _polys(name):
        df = pl.read_parquet(outs / f"{name}.parquet")
        return df.with_columns(
            (pl.col("vertex_x") / XENIUM_PX).alias("px"),
            (pl.col("vertex_y") / XENIUM_PX).alias("py"),
        )

    cells = pl.read_parquet(outs / "cells.parquet")
    centroids = np.stack([
        cells["y_centroid"].to_numpy() / XENIUM_PX,
        cells["x_centroid"].to_numpy() / XENIUM_PX,
    ], axis=1)
    tx = pl.read_parquet(outs / "transcripts.parquet").select(
        (pl.col("x_location") / XENIUM_PX).alias("x"),
        (pl.col("y_location") / XENIUM_PX).alias("y"),
        pl.col("feature_name").alias("gene"),
    )
    return {
        "dapi": _dapi,
        "pixel_size": XENIUM_PX,
        "nucleus_polys": lambda: _polys("nucleus_boundaries"),
        "cell_polys": lambda: _polys("cell_boundaries"),
        "transcripts": tx,
        "centroids": centroids,
    }
```

- [ ] **Step 4: Run, verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_xenium.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Real-data smoke (catches format surprises early — like the QC LMDB lesson)**

Run:
```bash
cd /mnt/work/git/dapidl && uv run python -c "
from dapidl.seg_eval.source_masks import load_xenium, rasterize_polygons
import numpy as np
s = load_xenium('/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1')
print('centroids', s['centroids'].shape, 'px range y', s['centroids'][:,0].min(), s['centroids'][:,0].max())
nuc = s['nucleus_polys']()
print('nucleus_polys cols', nuc.columns, 'rows', nuc.height)
m = rasterize_polygons(nuc.filter((pl.col('px')<2000)&(pl.col('py')<2000)) if False else nuc, 'cell_id','px','py',(0,0,2000,2000))
print('mask cells in 2000x2000 corner:', len(set(np.unique(m))-{0}))
" 2>&1 | tail -5
```
Expected: prints centroid shape (~167k for rep1), polygon columns including `cell_id,vertex_x,vertex_y,px,py`, and a nonzero cell count in the corner tile. If columns differ, fix `load_xenium` to the real names and re-run.

- [ ] **Step 6: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/seg_eval/__init__.py src/dapidl/seg_eval/source_masks.py tests/test_seg_eval_xenium.py
git commit -m "feat(seg-eval): Xenium source loader + polygon rasterization"
```

---

## Task 2: STHELAR source loader

**Files:**
- Modify: `src/dapidl/seg_eval/source_masks.py` (add `load_sthelar`)
- Test: `tests/test_seg_eval_sthelar.py`

- [ ] **Step 1: Write the failing test (real-data smoke — sdata structure can't be faked meaningfully)**

```python
"""Smoke test for the STHELAR sdata loader against real data."""
import numpy as np
from dapidl.seg_eval.source_masks import load_sthelar

ZARR = "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/sdata_breast_s0.zarr"


def test_sthelar_loader_smoke():
    s = load_sthelar(ZARR)
    dapi = s["dapi"]()
    assert dapi.ndim == 2 and dapi.size > 0
    assert s["centroids"].shape[1] == 2
    nuc = s["nucleus_polys"]()
    assert set(["px", "py", "cell_id"]).issubset(nuc.columns)
```

- [ ] **Step 2: Run, verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_sthelar.py -v`
Expected: FAIL `ImportError: cannot import name 'load_sthelar'`

- [ ] **Step 3: Implement (append to `source_masks.py`)**

```python
def load_sthelar(zarr_path: str | Path):
    """Return the STHELAR source's DAPI, polygons (px), transcripts, centroids.

    SpatialData zarr: images['morpho'] (DAPI), shapes['nucleus_boundaries'] /
    ['cell_boundaries'] (GeoDataFrame of shapely polygons in pixel coords of the
    morpho image), points['st'] (transcripts), tables['table_combined'] (labels).
    """
    import spatialdata as sd

    sdata = sd.read_zarr(str(zarr_path))
    morpho = sdata.images["morpho"]
    # SpatialImage/DataArray -> 2D numpy (squeeze any singleton channel/z)
    arr = np.asarray(morpho.data if hasattr(morpho, "data") else morpho)
    arr = np.squeeze(arr)
    if arr.ndim == 3:           # (C,H,W) -> take first channel (DAPI/morpho)
        arr = arr[0]
    dapi_img = arr

    def _polys_from_shapes(key: str):
        gdf = sdata.shapes[key]  # geopandas GeoDataFrame, column 'geometry'
        rows = {"cell_id": [], "px": [], "py": []}
        for i, geom in enumerate(gdf.geometry):
            xs, ys = geom.exterior.coords.xy
            rows["cell_id"].extend([str(i)] * len(xs))
            rows["px"].extend(list(xs))
            rows["py"].extend(list(ys))
        return pl.DataFrame(rows)

    # centroids from cell polygons' representative points
    nuc_gdf = sdata.shapes["nucleus_boundaries"]
    centroids = np.stack([
        nuc_gdf.geometry.centroid.y.to_numpy(),
        nuc_gdf.geometry.centroid.x.to_numpy(),
    ], axis=1)

    pts = sdata.points["st"].compute() if hasattr(sdata.points["st"], "compute") else sdata.points["st"]
    pts = pl.from_pandas(pts[[c for c in pts.columns]].reset_index(drop=True)) if not isinstance(pts, pl.DataFrame) else pts
    # normalize transcript column names (x, y px + gene)
    xcol = next(c for c in pts.columns if c.lower() in ("x", "x_location"))
    ycol = next(c for c in pts.columns if c.lower() in ("y", "y_location"))
    gcol = next((c for c in pts.columns if "gene" in c.lower() or "feature" in c.lower()), pts.columns[-1])
    tx = pts.select(pl.col(xcol).alias("x"), pl.col(ycol).alias("y"), pl.col(gcol).alias("gene"))

    return {
        "dapi": lambda: dapi_img,
        "pixel_size": 0.2125,  # STHELAR Xenium-derived; refined per-slide if metadata present
        "nucleus_polys": lambda: _polys_from_shapes("nucleus_boundaries"),
        "cell_polys": lambda: _polys_from_shapes("cell_boundaries"),
        "transcripts": tx,
        "centroids": centroids,
    }
```

- [ ] **Step 4: Run, verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_sthelar.py -v`
Expected: PASS. If the smoke reveals different coords (µm not px) or column names, fix `load_sthelar` accordingly (the test pins the contract) and re-run.

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/seg_eval/source_masks.py tests/test_seg_eval_sthelar.py
git commit -m "feat(seg-eval): STHELAR sdata source loader"
```

---

## Task 3: Mask comparison (detection P/R/F1 + morphometrics)

**Files:**
- Create: `src/dapidl/seg_eval/compare.py`
- Test: `tests/test_seg_eval_compare.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for detection metrics between two label masks."""
import numpy as np
from dapidl.seg_eval.compare import detection_metrics


def _two_squares(shape=(40, 40)):
    m = np.zeros(shape, dtype=np.int32)
    m[2:10, 2:10] = 1
    m[20:28, 20:28] = 2
    return m


def test_identical_masks_perfect():
    m = _two_squares()
    r = detection_metrics(m, m, iou_thr=0.5)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["n_pred"] == 2 and r["n_true"] == 2
    assert r["count_ratio"] == 1.0


def test_one_missing_instance():
    pred = _two_squares()
    true = _two_squares()
    pred[20:28, 20:28] = 0          # pred drops instance 2 -> 1 pred, 2 true
    r = detection_metrics(pred, true, iou_thr=0.5)
    assert r["n_pred"] == 1 and r["n_true"] == 2
    assert r["recall"] == 0.5 and r["precision"] == 1.0
```

- [ ] **Step 2: Run, verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_compare.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'dapidl.seg_eval.compare'`

- [ ] **Step 3: Implement**

Create `src/dapidl/seg_eval/compare.py`:
```python
"""Compare two instance label masks: detection P/R/F1, IoU, count ratio."""
import numpy as np
from starpose.consensus.matching import compute_iou_matrix
from starpose.evaluate.morphometric import compute_morphometric


def detection_metrics(pred: np.ndarray, true: np.ndarray, iou_thr: float = 0.5) -> dict:
    """pred/true are (H,W) int label masks. 'true' = source segmentation."""
    n_pred = int(pred.max())
    n_true = int(true.max())
    if n_pred == 0 or n_true == 0:
        return {"n_pred": n_pred, "n_true": n_true, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "median_iou": 0.0,
                "count_ratio": (n_pred / n_true) if n_true else float("nan")}
    iou = compute_iou_matrix(pred, true)            # (n_pred, n_true)
    best_per_pred = iou.max(axis=1)
    best_per_true = iou.max(axis=0)
    matched_pred = int((best_per_pred >= iou_thr).sum())
    matched_true = int((best_per_true >= iou_thr).sum())
    precision = matched_pred / n_pred
    recall = matched_true / n_true
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_pred": n_pred, "n_true": n_true,
        "precision": precision, "recall": recall, "f1": f1,
        "median_iou": float(np.median(best_per_pred)),
        "count_ratio": n_pred / n_true,
    }


def morphometrics(masks: np.ndarray, pixel_size: float) -> dict:
    """Thin wrapper over starpose morphometric stats for one mask."""
    return compute_morphometric(masks, pixel_size)
```

- [ ] **Step 4: Run, verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_compare.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/seg_eval/compare.py tests/test_seg_eval_compare.py
git commit -m "feat(seg-eval): detection P/R/F1 + morphometric comparison"
```

---

## Task 4: QC comparison (own centroids vs source centroids)

**Files:**
- Create: `src/dapidl/seg_eval/qc_compare.py`
- Test: `tests/test_seg_eval_qc.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for QC patch scoring around centroids."""
import numpy as np
from dapidl.seg_eval.qc_compare import score_centroid_patches


def test_score_centroid_patches_shapes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 4000, size=(512, 512), dtype=np.uint16)
    centroids = np.array([[100, 100], [200, 250], [400, 400]], dtype=float)  # [y,x]
    scores = score_centroid_patches(img, centroids, patch=128)
    assert set(["qc_score", "focus_score", "detection_score"]).issubset(scores.columns)
    assert scores.height == 3


def test_edge_centroids_skipped():
    img = np.zeros((256, 256), dtype=np.uint16)
    centroids = np.array([[5, 5], [128, 128]], dtype=float)  # first is too close to edge
    scores = score_centroid_patches(img, centroids, patch=128)
    assert scores.height == 1  # only the in-bounds patch scored
```

- [ ] **Step 2: Run, verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_qc.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'dapidl.seg_eval.qc_compare'`

- [ ] **Step 3: Implement**

Create `src/dapidl/seg_eval/qc_compare.py`:
```python
"""Score QC of nucleus patches cropped around centroids (own vs source)."""
import numpy as np
import polars as pl
from starpose.qc.classical import ClassicalQualityScorer


def score_centroid_patches(image: np.ndarray, centroids: np.ndarray,
                           patch: int = 128) -> pl.DataFrame:
    """Crop patch x patch windows around each [y,x] centroid (in-bounds only),
    score with ClassicalQualityScorer (reference fitted on this set), return a
    polars DF with focus_score, detection_score, qc_score per kept centroid."""
    half = patch // 2
    h, w = image.shape
    patches, kept = [], []
    for i, (y, x) in enumerate(centroids):
        yi, xi = int(round(y)), int(round(x))
        if yi - half < 0 or yi + half > h or xi - half < 0 or xi + half > w:
            continue
        patches.append(image[yi - half:yi + half, xi - half:xi + half])
        kept.append(i)
    if not patches:
        return pl.DataFrame({"centroid_idx": [], "focus_score": [],
                             "detection_score": [], "qc_score": []})
    batch = np.stack(patches)
    scorer = ClassicalQualityScorer()
    ref = scorer.fit_reference(batch)
    scores = scorer.score_batch(batch, ref=ref)
    return pl.DataFrame({
        "centroid_idx": kept,
        "focus_score": [s.focus_score for s in scores],
        "detection_score": [s.detection_score for s in scores],
        "qc_score": [s.qc_score for s in scores],
    })
```

- [ ] **Step 4: Run, verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_eval_qc.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add src/dapidl/seg_eval/qc_compare.py tests/test_seg_eval_qc.py
git commit -m "feat(seg-eval): QC scoring of patches around centroids"
```

---

## Task 5: Orchestrator (`scripts/seg_diagnostic.py`)

**Files:**
- Create: `scripts/seg_diagnostic.py`
- Test: `tests/test_seg_diagnostic_smoke.py`

- [ ] **Step 1: Write the failing test (the orchestrator's per-FOV unit must be callable in isolation)**

```python
"""Smoke test for the per-FOV diagnostic step using synthetic inputs."""
import numpy as np
import polars as pl
from seg_diagnostic import diagnose_fov   # scripts/ on path


class _Res:  # minimal stand-in for starpose SegmentationResult
    def __init__(self, nuc, cell, ncen):
        self.nucleus_masks = nuc; self.cell_masks = cell
        self.nucleus_centroids = ncen


def test_diagnose_fov_rows(monkeypatch):
    import seg_diagnostic as sd
    dapi = (np.random.default_rng(0).integers(0, 4000, (256, 256))).astype(np.uint16)
    src_nuc = np.zeros((256, 256), np.int32); src_nuc[10:30, 10:30] = 1
    src_cell = src_nuc.copy()
    src_cen = np.array([[20, 20]], float)
    sp = _Res(src_nuc.copy(), src_cell.copy(), src_cen.copy())
    monkeypatch.setattr(sd, "_segment_fov", lambda *a, **k: sp)
    row = diagnose_fov("xenium_rep1", "dense", dapi, src_nuc, src_cell,
                       src_cen, transcripts=None, pixel_size=0.2125)
    assert row["source"] == "xenium_rep1" and row["fov"] == "dense"
    assert "nuc_f1" in row and "qc_own_mean" in row and "qc_src_mean" in row
```

- [ ] **Step 2: Run, verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_diagnostic_smoke.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'seg_diagnostic'`

- [ ] **Step 3: Implement**

Create `scripts/seg_diagnostic.py`:
```python
"""Phase 1 segmentation diagnostic: starpose vs source on representative FOVs.

Per source: pick FOVs (source centroids), segment each with starpose
(adaptive nuclei + proseg/watershed cells), compare masks to source, and
compare QC of patches around starpose vs source centroids.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from dapidl.seg_eval.source_masks import (SOURCES, load_sthelar, load_xenium,
                                          rasterize_polygons)
from dapidl.seg_eval.compare import detection_metrics, morphometrics
from dapidl.seg_eval.qc_compare import score_centroid_patches

from starpose.benchmark.fov_selector import select_fovs, extract_tile
from starpose.core import segment_multimodal
from starpose.types import ModalityBundle

OUT = Path("pipeline_output/seg_diagnostic_2026_05")


def _segment_fov(dapi_tile, transcripts_tile, pixel_size, expander):
    mb = ModalityBundle(dapi=dapi_tile, transcripts=transcripts_tile,
                        pixel_size=pixel_size, platform="diagnostic")
    return segment_multimodal(mb, gpu=True, nucleus_method="adaptive",
                              expansion_method=expander)


def _crop_polys(polys, bbox):
    y0, x0, y1, x1 = bbox
    return polys.filter((pl.col("px") >= x0) & (pl.col("px") < x1)
                        & (pl.col("py") >= y0) & (pl.col("py") < y1))


def _local_centroids(centroids, bbox):
    y0, x0, y1, x1 = bbox
    m = ((centroids[:, 0] >= y0) & (centroids[:, 0] < y1)
         & (centroids[:, 1] >= x0) & (centroids[:, 1] < x1))
    c = centroids[m].copy()
    c[:, 0] -= y0; c[:, 1] -= x0
    return c


def diagnose_fov(source, fov_label, dapi_tile, src_nuc, src_cell, src_centroids_local,
                 transcripts, pixel_size, expander="proseg"):
    """One FOV: segment, compare nucleus+cell masks, compare QC. Returns a row dict."""
    res = _segment_fov(dapi_tile, transcripts, pixel_size, expander)
    sp_nuc = res.nucleus_masks
    sp_cell = res.cell_masks if res.cell_masks is not None else res.nucleus_masks
    sp_cen = np.asarray(res.nucleus_centroids)

    nuc = detection_metrics(sp_nuc, src_nuc)
    cell = detection_metrics(sp_cell, src_cell)
    sp_morph = morphometrics(sp_nuc, pixel_size)
    src_morph = morphometrics(src_nuc, pixel_size)

    qc_own = score_centroid_patches(dapi_tile, sp_cen)
    qc_src = score_centroid_patches(dapi_tile, src_centroids_local)

    return {
        "source": source, "fov": fov_label,
        "n_starpose_nuc": nuc["n_pred"], "n_source_nuc": nuc["n_true"],
        "nuc_precision": nuc["precision"], "nuc_recall": nuc["recall"],
        "nuc_f1": nuc["f1"], "nuc_median_iou": nuc["median_iou"],
        "nuc_count_ratio": nuc["count_ratio"],
        "cell_f1": cell["f1"], "cell_count_ratio": cell["count_ratio"],
        "sp_mean_area_um2": sp_morph["mean_area_um2"],
        "src_mean_area_um2": src_morph["mean_area_um2"],
        "qc_own_mean": float(qc_own["qc_score"].mean()) if qc_own.height else float("nan"),
        "qc_src_mean": float(qc_src["qc_score"].mean()) if qc_src.height else float("nan"),
        "qc_own_n": qc_own.height, "qc_src_n": qc_src.height,
    }


def run_source(name: str, n_fovs: int, tile: int, expander: str) -> list[dict]:
    cfg = SOURCES[name]
    src = load_xenium(cfg["root"]) if cfg["kind"] == "xenium" else load_sthelar(cfg["zarr"])
    dapi = src["dapi"]()
    px = src["pixel_size"]
    fovs = select_fovs(src["centroids"], dapi.shape, pixel_size=px,
                       n_fovs=n_fovs, tile_size_px=tile)
    nuc_polys, cell_polys = src["nucleus_polys"](), src["cell_polys"]()
    rows = []
    for fov in fovs:
        y0, x0, y1, x1 = fov.bbox
        tile_img = extract_tile(dapi, fov)
        src_nuc = rasterize_polygons(_crop_polys(nuc_polys, fov.bbox), "cell_id", "px", "py", fov.bbox)
        src_cell = rasterize_polygons(_crop_polys(cell_polys, fov.bbox), "cell_id", "px", "py", fov.bbox)
        src_cen = _local_centroids(src["centroids"], fov.bbox)
        tx = src["transcripts"]
        tx_tile = tx.filter((pl.col("x") >= x0) & (pl.col("x") < x1)
                            & (pl.col("y") >= y0) & (pl.col("y") < y1)
                            ).with_columns((pl.col("x") - x0).alias("x"),
                                           (pl.col("y") - y0).alias("y"))
        try:
            rows.append(diagnose_fov(name, fov.label, tile_img, src_nuc, src_cell,
                                     src_cen, tx_tile, px, expander))
        except Exception as e:  # one bad FOV shouldn't kill the source
            logger.warning(f"{name}/{fov.label} failed: {e}")
    logger.info(f"{name}: {len(rows)} FOVs done")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--n-fovs", type=int, default=8)
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--expander", default="proseg", choices=["proseg", "watershed"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name in args.sources.split(","):
        all_rows.extend(run_source(name, args.n_fovs, args.tile, args.expander))
    df = pl.DataFrame(all_rows)
    df.write_parquet(OUT / "results.parquet")
    logger.info(f"wrote {OUT/'results.parquet'} ({df.height} rows)")
    # headline rollup per source
    print(df.group_by("source").agg(
        pl.col("nuc_f1").mean().round(3),
        pl.col("nuc_count_ratio").mean().round(3),
        pl.col("qc_own_mean").mean().round(3),
        pl.col("qc_src_mean").mean().round(3),
    ).sort("source"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, verify it passes**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_seg_diagnostic_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
cd /mnt/work/git/dapidl
git add scripts/seg_diagnostic.py tests/test_seg_diagnostic_smoke.py
git commit -m "feat(seg): segmentation diagnostic orchestrator"
```

---

## Task 6: End-to-end smoke + full run

**Files:** none (execution + figures generated by a follow-up; this task validates the pipeline on real data).

- [ ] **Step 1: One Xenium source, few FOVs, real segmentation (GPU)**

Check GPU first: `nvidia-smi --query-gpu=memory.free --format=csv,noheader`
Run:
```bash
cd /mnt/work/git/dapidl
EXP=$(proseg --version >/dev/null 2>&1 && echo proseg || echo watershed)
uv run python scripts/seg_diagnostic.py --sources xenium_rep1 --n-fovs 2 --tile 2048 --expander "$EXP" 2>&1 | tail -20
```
Expected: writes `pipeline_output/seg_diagnostic_2026_05/results.parquet`, prints a per-source rollup with `nuc_f1`, `nuc_count_ratio`, `qc_own_mean`, `qc_src_mean`. No crash. Sanity: `n_starpose_nuc` and `n_source_nuc` both > 0; `nuc_f1` in [0,1].

- [ ] **Step 2: Inspect the smoke result**

Run:
```bash
cd /mnt/work/git/dapidl && uv run python -c "
import polars as pl; print(pl.read_parquet('pipeline_output/seg_diagnostic_2026_05/results.parquet'))"
```
Confirm: nucleus counts are plausible for a 2048² FOV (hundreds–thousands), starpose vs source agreement is non-degenerate (f1 not 0 or exactly 1), and QC means are in [0,1]. If counts are ~0, the coordinate alignment (µm↔px) is wrong — fix the loader before the full run.

- [ ] **Step 3: Full run (all 6 sources, 8 FOVs)**

Run:
```bash
cd /mnt/work/git/dapidl
EXP=$(proseg --version >/dev/null 2>&1 && echo proseg || echo watershed)
uv run python scripts/seg_diagnostic.py --n-fovs 8 --tile 2048 --expander "$EXP" \
  > /tmp/dapidl_logs/seg_diagnostic.log 2>&1 &
```
Expected (when done): `results.parquet` with ~48 rows (6×8), per-source rollup. Watch for OOM (each 2048² FOV seg is small; full DAPI load per source is ~1.8 GB — fine on 62 GB).

- [ ] **Step 4: Headline readout (the actual question)**

Compare per-source `qc_own_mean` vs `qc_src_mean`, especially for `sthelar_breast_s6` (Prime): if starpose patches out-QC the source on Prime, segmentation is part of the Prime gap → Phase 2 worth it. Also report nucleus agreement (f1, count ratio) per source. Write a short readout and a go/no-go for Phase 2.

---

## Self-Review

**Spec coverage:**
- §3 decisions (Phase 1, ~8 FOVs, adaptive, proseg+watershed fallback) → Task 0 (proseg), Task 5 (`--n-fovs 8`, `adaptive`, `--expander`). ✓
- §4 SourceSegmentationLoader (Xenium polygons µm→px; STHELAR sdata) → Tasks 1, 2. ✓
- §4 FOV selection (region-matched) → Task 5 `select_fovs` on source centroids, same bbox for both. ✓
- §4 starpose seg (adaptive + proseg) → Task 5 `_segment_fov`/`segment_multimodal`. ✓
- §4 compare (IoU P/R/F1, count ratio, morphometrics) → Task 3. ✓
- §4 QC own vs source → Task 4 + Task 5 (`qc_own_mean`/`qc_src_mean`). ✓
- §5 data flow + §6 outputs (`results.parquet`, rollup, readout) → Task 5 + Task 6. ✓
- §7 risks: proseg fallback (Task 0/5/6 `$EXP`), STHELAR reading (Task 2 smoke), coordinate alignment (Task 1/2 smoke + Task 6 Step 2 check). ✓
- §9 testing (loader, compare, qc, smoke) → Tasks 1-6 all TDD. ✓
- Figures/montages (§6): the rollup table + parquet are produced; **figure plotting is deferred to the readout step** (Task 6 Step 4) rather than a separate task — flagged here as the one spec item delivered as analysis rather than code. Acceptable for a diagnostic; add a `figures.py` task if you want committed plots.

**Placeholder scan:** no TBD/TODO; every code step has complete code; commands have expected output. The loaders include real-data smoke steps to catch format drift (column names, coord units) — these are verification steps, not placeholders.

**Type consistency:** `load_xenium`/`load_sthelar` both return the same dict contract (`dapi` callable, `pixel_size`, `nucleus_polys`/`cell_polys` callables, `transcripts`, `centroids`); `rasterize_polygons(df,id_col,x_col,y_col,bbox)`; `detection_metrics(pred,true,iou_thr)->{n_pred,n_true,precision,recall,f1,median_iou,count_ratio}`; `score_centroid_patches(image,centroids,patch)->pl.DataFrame[centroid_idx,focus_score,detection_score,qc_score]`; `diagnose_fov(...)`/`run_source(...)` consistent. ✓

**Notes for executor:**
- STHELAR coordinate units (px vs µm in `shapes`) and exact `points['st']` column names are the highest-risk unknowns — Task 2's smoke test is the gate; adjust `load_sthelar` to whatever the real sdata exposes (the contract/test stays fixed).
- If proseg's per-FOV runtime is heavy, cell metrics can be dropped (`--expander watershed`) without affecting the nucleus + QC headline.
