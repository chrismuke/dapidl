# Spatial-GNN Probe ("Learned BANKSY") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged probe that tests whether a within-slide spatial-neighbour graph on DAPI features beats the EfficientNet 0.619 on held-out rep2 — especially on the context-defined classes (Endothelial/Stromal).

**Architecture:** A new `src/dapidl/graph/` package with focused, mostly-pure units (k-NN graph, BANKSY feature augmentation, evaluation helpers, spatial registry replay, embedding/PCA, a hand-rolled scatter-mean GraphSAGE), wired by `scripts/spatial_gnn_probe.py`. Stage 1 (cheap, no GNN) augments frozen-EffNet PCA-128 embeddings with BANKSY neighbour terms and classifies with LightGBM; a decision gate; Stage 2 (gated) trains a nucleus-local CNN + GraphSAGE end-to-end. Reuses the existing `breast-6source-dapi-p128` LMDB; train rep1+STHELAR s0/s1/s3/s6, test rep2.

**Tech Stack:** Python, numpy, polars, scipy (`cKDTree`), scikit-learn (PCA, metrics), LightGBM 4.6, PyTorch + timm (existing `DapiClassifier`), lmdb. **No torch_geometric** (hand-rolled scatter-mean). Spec: `docs/superpowers/specs/2026-06-09-spatial-gnn-probe-design.md`.

**Conventions for this plan:** all commands run from repo root `/mnt/work/git/dapidl` via `uv run`. Commit messages carry **no Claude/Anthropic attribution**. Pure-unit tasks (1,2,3,5a,6) are subagent-implementable; GPU/integration runs (4-run, 5b, 7) are **controller-run** (marked `[CONTROLLER RUN]`) because they touch the real 8 GB LMDB / GPU and the deterministic-replay assertion.

---

## File Structure

```
src/dapidl/graph/
  __init__.py            # exports the public functions
  knn_graph.py           # build_within_slide_knn(coords, slide_ids, k) -> edge_index (Task 1)
  banksy_features.py     # banksy_augment(feats, edge_index, coords, lambda_) -> (N, 3D) (Task 2)
  probe_eval.py          # mcnemar_test, bootstrap_macro_f1_ci, gate_decision (Task 3)
  registry.py            # replay_xenium/_sthelar (pure) + build_spatial_registry (I/O) (Task 4)
  embed.py               # read_patch, lmdb_patch_iter, pca_fit_transform; extract_embeddings (Task 5)
  gnn.py                 # NucleusNodeCNN, scatter_mean, SageCellTyper (Task 6)
scripts/
  spatial_gnn_probe.py   # driver: registry -> stage1 -> gate -> stage2 -> readout (Tasks 4,5,7)
tests/
  test_graph_knn.py  test_graph_banksy.py  test_graph_probe_eval.py
  test_graph_registry.py  test_graph_embed.py  test_graph_gnn.py
```

Outputs → `pipeline_output/spatial_gnn_probe_2026_06/`.

---

### Task 1: Within-slide k-NN graph

**Files:**
- Create: `src/dapidl/graph/__init__.py`
- Create: `src/dapidl/graph/knn_graph.py`
- Test: `tests/test_graph_knn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_knn.py
import numpy as np
from dapidl.graph.knn_graph import build_within_slide_knn


def test_edges_stay_within_slide():
    # slide A at x~0, slide B at x~1000 -> a global k-NN would wrongly connect them
    coords = np.array([[0, 0], [1, 0], [2, 0], [1000, 0], [1001, 0], [1002, 0]], float)
    slide_ids = np.array(["A", "A", "A", "B", "B", "B"])
    ei = build_within_slide_knn(coords, slide_ids, k=2)
    assert ei.shape[0] == 2
    # every edge connects two cells on the same slide
    assert np.all(slide_ids[ei[0]] == slide_ids[ei[1]])
    # no self loops
    assert np.all(ei[0] != ei[1])


def test_k_capped_to_slide_size():
    # a 2-cell slide can yield at most 1 neighbour per node even if k=8
    coords = np.array([[0, 0], [1, 0]], float)
    slide_ids = np.array(["A", "A"])
    ei = build_within_slide_knn(coords, slide_ids, k=8)
    assert ei.shape == (2, 2)  # each of 2 nodes -> its single neighbour
    assert set(map(tuple, ei.T.tolist())) == {(0, 1), (1, 0)}


def test_singleton_slide_contributes_no_edges():
    coords = np.array([[0, 0], [5, 5], [6, 6]], float)
    slide_ids = np.array(["solo", "pair", "pair"])
    ei = build_within_slide_knn(coords, slide_ids, k=3)
    assert "solo" not in set(slide_ids[ei[0]].tolist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_knn.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dapidl.graph.knn_graph'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/graph/__init__.py
"""Spatial-graph probe ("learned BANKSY"): within-slide k-NN over DAPI cells."""
```

```python
# src/dapidl/graph/knn_graph.py
"""Within-slide k-nearest-neighbour graph over cell centroids. Edges never cross
slides (each tissue section is its own connected component); no self-loops."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_within_slide_knn(coords, slide_ids, k: int = 8) -> np.ndarray:
    """Directed edge_index (2, E) of GLOBAL node indices. Edge (src, dst) means
    ``dst`` is one of ``src``'s k nearest neighbours on the same slide. Slides with
    a single cell contribute no edges; k is capped at (slide_size - 1)."""
    coords = np.asarray(coords, dtype=float)
    slide_ids = np.asarray(slide_ids)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for s in np.unique(slide_ids):
        idx = np.where(slide_ids == s)[0]
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        tree = cKDTree(coords[idx])
        _, nn = tree.query(coords[idx], k=kk + 1)  # col 0 is the node itself
        nn = np.atleast_2d(nn)
        for col in range(1, kk + 1):
            src_parts.append(idx)
            dst_parts.append(idx[nn[:, col]])
    if not src_parts:
        return np.empty((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src_parts), np.concatenate(dst_parts)]).astype(np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_knn.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/__init__.py src/dapidl/graph/knn_graph.py tests/test_graph_knn.py
git commit -m "feat(graph): within-slide k-NN graph over cell centroids"
```

---

### Task 2: BANKSY feature augmentation

**Files:**
- Create: `src/dapidl/graph/banksy_features.py`
- Test: `tests/test_graph_banksy.py`

BANKSY augments each cell's feature `x` with the **neighbour mean** and the **azimuthal gradient (AGF)** — the magnitude of the first angular harmonic of neighbour feature-differences around the cell. The output is `concat[√(1−λ)·x, √(λ/2)·mean, √(λ/2)·|AGF|]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_banksy.py
import numpy as np
from dapidl.graph.banksy_features import banksy_augment


def _line_graph():
    # node 1 has neighbours 0 and 2 (a directed star centred on 1 for this test)
    coords = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    feats = np.array([[2.0], [0.0], [4.0]])  # D=1
    edge_index = np.array([[1, 1], [0, 2]])  # 1->0, 1->2
    return feats, edge_index, coords


def test_lambda_zero_is_identity_padded_with_zeros():
    feats, ei, coords = _line_graph()
    aug = banksy_augment(feats, ei, coords, lambda_=0.0)
    assert aug.shape == (3, 3)            # N x 3D
    np.testing.assert_allclose(aug[:, 0:1], feats)   # self term == x
    np.testing.assert_allclose(aug[:, 1:], 0.0)      # neighbour terms vanish


def test_neighbour_mean_matches_hand_computation():
    feats, ei, coords = _line_graph()
    aug = banksy_augment(feats, ei, coords, lambda_=0.5)
    # node 1 neighbour mean of feats {2, 4} = 3.0; scaled by sqrt(0.5/2)=0.5
    assert abs(aug[1, 1] - 0.5 * 3.0) < 1e-9
    # node 0 has no out-edges -> neighbour mean 0
    assert abs(aug[0, 1]) < 1e-9


def test_output_width_is_three_times_feature_dim():
    feats = np.random.default_rng(0).normal(size=(10, 7))
    ei = np.array([[0, 1, 2], [1, 2, 0]])
    coords = np.random.default_rng(1).normal(size=(10, 2))
    assert banksy_augment(feats, ei, coords, lambda_=0.8).shape == (10, 21)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_banksy.py -v`
Expected: FAIL — `ModuleNotFoundError: ... banksy_features`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/graph/banksy_features.py
"""BANKSY-style neighbour augmentation: concatenate each cell's own feature with
its neighbour mean and azimuthal-gradient magnitude (AGF). lambda_ weights the
neighbourhood vs the self term (lambda_=0 -> self only)."""
from __future__ import annotations

import numpy as np


def banksy_augment(feats, edge_index, coords, lambda_: float = 0.5) -> np.ndarray:
    feats = np.asarray(feats, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n, d = feats.shape
    src, dst = np.asarray(edge_index)
    deg = np.zeros(n)
    nbr_mean = np.zeros((n, d))
    agf = np.zeros((n, d), dtype=complex)
    if src.size:
        theta = np.arctan2(coords[dst, 1] - coords[src, 1],
                           coords[dst, 0] - coords[src, 0])
        phase = np.exp(1j * theta)[:, None]            # (E, 1)
        diff = feats[dst] - feats[src]                  # (E, D)
        np.add.at(nbr_mean, src, feats[dst])
        np.add.at(agf, src, phase * diff)
        np.add.at(deg, src, 1.0)
    deg_safe = np.maximum(deg, 1.0)[:, None]
    nbr_mean = nbr_mean / deg_safe
    agf_mag = np.abs(agf / deg_safe)
    return np.concatenate([
        np.sqrt(1.0 - lambda_) * feats,
        np.sqrt(lambda_ / 2.0) * nbr_mean,
        np.sqrt(lambda_ / 2.0) * agf_mag,
    ], axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_banksy.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/banksy_features.py tests/test_graph_banksy.py
git commit -m "feat(graph): BANKSY neighbour-mean + azimuthal-gradient augmentation"
```

---

### Task 3: Evaluation helpers (McNemar, bootstrap CI, gate)

**Files:**
- Create: `src/dapidl/graph/probe_eval.py`
- Test: `tests/test_graph_probe_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_probe_eval.py
import numpy as np
from dapidl.graph.probe_eval import mcnemar_test, bootstrap_macro_f1_ci, gate_decision


def test_mcnemar_detects_directional_disagreement():
    truth = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    base = truth.copy(); base[:6] = (base[:6] + 1) % 4   # base wrong on 6
    new = truth.copy()                                    # new always right
    res = mcnemar_test(truth, base, new)
    assert res["n_new_right_base_wrong"] == 6
    assert res["n_base_right_new_wrong"] == 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, size=400)
    pred = truth.copy()
    flip = rng.choice(400, size=80, replace=False)
    pred[flip] = (pred[flip] + 1) % 4
    lo, point, hi = bootstrap_macro_f1_ci(truth, pred, n_boot=200, seed=0)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_gate_proceeds_unless_flat_dead():
    # clear lift on endothelial -> proceed
    assert gate_decision(macro_delta=0.04, endo_delta=0.06, stromal_delta=0.01)["proceed"]
    # flat-dead: no macro gain AND <0.01 on both context classes -> stop
    d = gate_decision(macro_delta=-0.01, endo_delta=0.0, stromal_delta=0.005)
    assert not d["proceed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_probe_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: ... probe_eval`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/graph/probe_eval.py
"""Probe evaluation helpers: McNemar paired test (new vs baseline predictions),
bootstrap CI on macro-F1, and the Stage-1 -> Stage-2 gate decision."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def mcnemar_test(truth, base_pred, new_pred) -> dict:
    """Exact McNemar (binomial) on the discordant pairs where exactly one model is
    right. Reports the two discordant counts and the two-sided p-value."""
    truth = np.asarray(truth); base_pred = np.asarray(base_pred); new_pred = np.asarray(new_pred)
    base_ok = base_pred == truth
    new_ok = new_pred == truth
    b = int(np.sum(new_ok & ~base_ok))   # new right, base wrong
    c = int(np.sum(base_ok & ~new_ok))   # base right, new wrong
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n > 0 else 1.0
    return {"n_new_right_base_wrong": b, "n_base_right_new_wrong": c, "p_value": float(p)}


def bootstrap_macro_f1_ci(truth, pred, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """(lo, point, hi) macro-F1 with a percentile bootstrap over resampled cells."""
    truth = np.asarray(truth); pred = np.asarray(pred)
    point = f1_score(truth, pred, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    n = len(truth)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, size=n)
        boots[i] = f1_score(truth[s], pred[s], average="macro", zero_division=0)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(point), float(hi)


def gate_decision(macro_delta: float, endo_delta: float, stromal_delta: float) -> dict:
    """Proceed to Stage 2 UNLESS Stage 1 is flat-dead: no macro gain AND <0.01 on
    BOTH context classes. (Permissive: Stage 2 is the real test.)"""
    flat_dead = (macro_delta <= 0.0) and (endo_delta < 0.01) and (stromal_delta < 0.01)
    return {"proceed": not flat_dead,
            "reason": "flat-dead: no spatial signal" if flat_dead else "spatial signal present"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_probe_eval.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/probe_eval.py tests/test_graph_probe_eval.py
git commit -m "feat(graph): McNemar + bootstrap-F1 CI + Stage-1->2 gate helpers"
```

---

### Task 4: Spatial registry (deterministic replay + verification)

**Files:**
- Create: `src/dapidl/graph/registry.py`
- Test: `tests/test_graph_registry.py`
- Driver step in: `scripts/spatial_gnn_probe.py` (`--phase registry`)

The pure replay functions mirror `scripts/breast_dapi_lmdb.py` (`extract_xenium_breast` candidate-filter + OOB-bounds; `extract_sthelar_breast` likewise). The I/O wrapper wires the real readers and **asserts the replayed (source, label) sequence equals the stored `sources.npy` / `labels.npy`** — the alignment proof.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_registry.py
import numpy as np
import pytest
from dapidl.graph.registry import replay_xenium, replay_sthelar, verify_alignment


def test_replay_xenium_filters_and_bounds():
    cell_ids = ["a", "b", "c", "d"]
    gt = {"a": "Endothelial", "b": "Unlabeled", "c": "CD8+_T_Cells", "d": "Invasive_Tumor"}
    fine_to_coarse = {"Endothelial": "Endothelial", "Unlabeled": None,
                      "CD8+_T_Cells": "Immune", "Invasive_Tumor": "Epithelial"}.get
    coarse_to_idx = {"Endothelial": 0, "Epithelial": 1, "Immune": 2, "Stromal": 3}
    centroids = np.array([[100, 100], [100, 100], [1, 1], [200, 200]], float)  # c is OOB (edge)
    rows = replay_xenium(cell_ids, gt, fine_to_coarse, coarse_to_idx,
                         centroids, h=300, w=300, half=64)
    # b dropped (Unlabeled->None); c dropped (OOB, x0=1-64<0); a and d kept, in order
    assert [r[0] for r in rows] == ["a", "d"]
    assert [r[3] for r in rows] == [0, 1]


def test_verify_alignment_raises_on_desync():
    rows = [("a", 100.0, 100.0, 0), ("d", 200.0, 200.0, 1)]
    sources = np.array(["xenium_rep1", "xenium_rep1"])
    labels = np.array([0, 1])
    verify_alignment(rows, sources_seq=["xenium_rep1", "xenium_rep1"],
                     stored_sources=sources, stored_labels=labels)  # ok, no raise
    with pytest.raises(AssertionError):
        verify_alignment(rows, sources_seq=["xenium_rep1", "xenium_rep1"],
                         stored_sources=sources, stored_labels=np.array([0, 2]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: ... registry`.

- [ ] **Step 3: Write minimal implementation (pure core)**

```python
# src/dapidl/graph/registry.py
"""Reconstruct a per-row spatial registry (row_idx, source, cell_id, x_px, y_px)
for a pre-registry LMDB by replaying scripts/breast_dapi_lmdb.py's DETERMINISTIC
build order, then proving alignment against the stored sources.npy / labels.npy.

Pure replay cores mirror extract_xenium_breast / extract_sthelar_breast exactly:
candidate label filter, then the int(round(centroid)) +/- half OOB-bounds check.
No patches are read; only reader metadata + the mosaic shape are needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def _in_bounds(cx, cy, h, w, half) -> bool:
    cx, cy = int(round(cx)), int(round(cy))
    return not (cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w)


def replay_xenium(cell_ids, gt_lookup, fine_to_coarse, coarse_to_idx,
                  centroids, h, w, half) -> list[tuple]:
    rows = []
    for i, cid in enumerate(cell_ids):
        fine = gt_lookup.get(str(cid))
        if fine is None:
            continue
        coarse = fine_to_coarse(fine)
        if coarse is None:
            continue
        if not _in_bounds(centroids[i, 0], centroids[i, 1], h, w, half):
            continue
        rows.append((str(cid), float(centroids[i, 0]), float(centroids[i, 1]),
                     coarse_to_idx[coarse]))
    return rows


def replay_sthelar(ordered_cids, coarse_by_cid, centroid_by_cid, coarse_to_idx,
                   h, w, half) -> list[tuple]:
    """``ordered_cids`` is reader.nucleus_df cell_id order AFTER the coarse-not-null
    filter (mirrors extract_sthelar_breast's nuc_df.iter_rows)."""
    rows = []
    for cid in ordered_cids:
        if cid not in centroid_by_cid:
            continue
        cx, cy = centroid_by_cid[cid]
        if not _in_bounds(cx, cy, h, w, half):
            continue
        rows.append((str(cid), float(cx), float(cy), coarse_to_idx[coarse_by_cid[cid]]))
    return rows


def verify_alignment(rows, sources_seq, stored_sources, stored_labels) -> None:
    assert len(rows) == len(stored_sources) == len(stored_labels), (
        f"length desync: rows={len(rows)} sources={len(stored_sources)} labels={len(stored_labels)}")
    replay_labels = np.array([r[3] for r in rows], dtype=np.int64)
    assert np.array_equal(np.asarray(sources_seq), np.asarray(stored_sources)), "source-sequence desync"
    assert np.array_equal(replay_labels, np.asarray(stored_labels)), "label-sequence desync"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_registry.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the I/O wrapper (no new test — exercised by the [CONTROLLER RUN] below)**

Append to `src/dapidl/graph/registry.py`:

```python
def build_spatial_registry(lmdb_dir: Path) -> pl.DataFrame:
    """Wire the real readers in build order, replay, verify, return the registry.
    Mirrors scripts/breast_dapi_lmdb.py main() source order: rep1, rep2, then
    sorted STHELAR breast zarrs. Imports the build script's constants so the label
    maps never drift."""
    import sys
    sys.path.insert(0, "scripts")
    import breast_dapi_lmdb as B  # noqa: E402  (constants + readers, no side effects)
    from dapidl.data.lazy_mosaic import open_xenium_mosaic, LazyMosaic
    from dapidl.data.sthelar import SthelarDataReader
    from dapidl.data.xenium import XeniumDataReader

    lmdb_dir = Path(lmdb_dir)
    # sources.npy is our own build artifact (object array of source-name strings),
    # so allow_pickle is safe here (not untrusted input).
    stored_sources = np.load(lmdb_dir / "sources.npy", allow_pickle=True)
    stored_labels = np.load(lmdb_dir / "labels.npy")
    half = 64  # patch_size 128
    all_rows: list[tuple] = []
    all_src: list[str] = []

    for rep in ["rep1", "rep2"]:
        raw = B.XENIUM_BASE / f"xenium-breast-tumor-{rep}"
        reader = XeniumDataReader(raw / "outs")
        gt = B._load_xenium_supervised_gt(rep)
        cents = reader.get_centroids_pixels()
        cids = reader.get_cell_ids()
        with open_xenium_mosaic(reader.image_path) as m:
            h, w = m.shape
        rows = replay_xenium(cids, gt, B._xenium_fine_to_coarse, B.COARSE_TO_IDX, cents, h, w, half)
        all_rows += rows
        all_src += [f"xenium_{rep}"] * len(rows)

    for z in sorted(B.STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        if not z.is_dir():
            continue
        name = z.name.replace("sdata_", "").replace(".zarr", "")
        reader = SthelarDataReader(z)
        ndf = reader.nucleus_df.with_columns(
            pl.col("label1").replace_strict(B.STHELAR_LABEL1_TO_COARSE, default=None).alias("coarse")
        ).filter(pl.col("coarse").is_not_null())
        ordered = ndf["cell_id"].to_list()
        coarse_by = dict(zip(ndf["cell_id"].to_list(), ndf["coarse"].to_list()))
        cents = reader.get_centroids_pixels()
        rcids = reader.get_cell_ids()
        cent_by = {cid: (cents[i, 0], cents[i, 1]) for i, cid in enumerate(rcids)}
        h, w = LazyMosaic(reader.dapi_lazy).shape
        rows = replay_sthelar(ordered, coarse_by, cent_by, B.COARSE_TO_IDX, h, w, half)
        all_rows += rows
        all_src += [f"sthelar_{name}"] * len(rows)

    verify_alignment(all_rows, all_src, stored_sources, stored_labels)
    return pl.DataFrame({
        "row_idx": np.arange(len(all_rows), dtype=np.int64),
        "source": all_src,
        "cell_id": [r[0] for r in all_rows],
        "x_px": [r[1] for r in all_rows],
        "y_px": [r[2] for r in all_rows],
        "coarse_idx": [r[3] for r in all_rows],
    })
```

- [ ] **Step 6: Create the driver skeleton with the registry phase**

```python
# scripts/spatial_gnn_probe.py
"""Spatial-GNN probe driver: registry -> stage1 -> gate -> stage2 -> readout.
Run phases individually; GPU phases print nvidia-smi guidance first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
OUT = Path("pipeline_output/spatial_gnn_probe_2026_06")


def phase_registry() -> None:
    from dapidl.graph.registry import build_spatial_registry
    OUT.mkdir(parents=True, exist_ok=True)
    reg = build_spatial_registry(LMDB_DIR)
    reg.write_parquet(OUT / "spatial_registry.parquet")
    logger.info(f"registry: {len(reg)} rows verified-aligned -> {OUT/'spatial_registry.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["registry", "embed", "stage1", "stage2"])
    args = ap.parse_args()
    {"registry": phase_registry}.get(args.phase, lambda: logger.error("phase not yet implemented"))()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add src/dapidl/graph/registry.py scripts/spatial_gnn_probe.py tests/test_graph_registry.py
git commit -m "feat(graph): spatial registry replay+verify + probe driver skeleton"
```

- [ ] **Step 8: [CONTROLLER RUN] Build & verify the registry on real data**

Run: `uv run python scripts/spatial_gnn_probe.py --phase registry`
Expected: logs `registry: 2277877 rows verified-aligned` and writes `pipeline_output/spatial_gnn_probe_2026_06/spatial_registry.parquet`. **If `verify_alignment` raises**, the replay drifted from the build — diagnose the first mismatching source/index before proceeding (do not weaken the assertion).

---

### Task 5: Embedding extraction + PCA-128

**Files:**
- Create: `src/dapidl/graph/embed.py`
- Test: `tests/test_graph_embed.py`
- Driver: add `phase_embed` to `scripts/spatial_gnn_probe.py`

- [ ] **Step 1: Write the failing test** (pure helpers: LMDB record decode + PCA)

```python
# tests/test_graph_embed.py
import struct
import numpy as np
from dapidl.graph.embed import decode_record, pca_fit_transform


def test_decode_record_splits_label_and_patch():
    label = np.array([2], dtype=np.int64).tobytes()
    patch = (np.arange(128 * 128, dtype=np.uint16)).tobytes()
    lab, arr = decode_record(label + patch, patch_size=128)
    assert lab == 2
    assert arr.shape == (128, 128) and arr.dtype == np.uint16
    assert arr[0, 1] == 1


def test_pca_reduces_width_and_is_finite():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 64)).astype(np.float32)
    red, model = pca_fit_transform(emb, n_components=8, fit_sample=200, seed=0)
    assert red.shape == (500, 8)
    assert np.all(np.isfinite(red))
    # variance is ordered (first component explains the most)
    assert model.explained_variance_[0] >= model.explained_variance_[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: ... embed`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/graph/embed.py
"""Frozen-EffNet embedding extraction over the LMDB + PCA reduction. The pure
helpers (decode_record, pca_fit_transform) are unit-tested; extract_embeddings is
GPU and run by the controller."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def decode_record(value: bytes, patch_size: int = 128):
    """Format B record -> (int label, uint16 patch). value = int64 label + uint16 square."""
    label = int(np.frombuffer(value[:8], dtype=np.int64)[0])
    patch = np.frombuffer(value[8:], dtype=np.uint16).reshape(patch_size, patch_size)
    return label, patch


def pca_fit_transform(emb, n_components: int = 128, fit_sample: int = 200_000, seed: int = 0):
    """Fit PCA on a random row-sample (RAM), transform all rows. Returns (reduced, model)."""
    emb = np.asarray(emb)
    rng = np.random.default_rng(seed)
    n = len(emb)
    sample = emb if n <= fit_sample else emb[rng.choice(n, size=fit_sample, replace=False)]
    model = PCA(n_components=min(n_components, emb.shape[1]), random_state=seed)
    model.fit(sample.astype(np.float32))
    return model.transform(emb.astype(np.float32)).astype(np.float32), model


def extract_embeddings(lmdb_dir: Path, ckpt: Path, out_path: Path,
                       n: int, batch_size: int = 256, patch_size: int = 128) -> None:
    """[GPU] Stream the LMDB in row order through the frozen DapiClassifier backbone
    (penultimate 1792-d features) -> float16 memmap (n, 1792)."""
    import sys
    import lmdb
    import torch
    sys.path.insert(0, "scripts")
    from breast_pooled_train import DapiClassifier  # same class the checkpoint was saved from

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DapiClassifier(num_classes=4, backbone="efficientnetv2_rw_s")
    # weights_only=True: our own trusted checkpoint, but keep the secure default
    # (a state_dict is tensors + basic types, so this loads fine).
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict") or state.get("model") or state
    model.load_state_dict(sd)
    model.eval().to(device)

    feat_dim = model.head.in_features
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, feat_dim))
    env = lmdb.open(str(lmdb_dir / "patches.lmdb"), readonly=True, lock=False)
    buf_imgs: list[np.ndarray] = []
    buf_rows: list[int] = []

    def flush():
        if not buf_rows:
            return
        x = np.stack(buf_imgs).astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        t = torch.from_numpy(x)[:, None, :, :].to(device)
        with torch.no_grad():
            feat = model.backbone(t.expand(-1, 3, -1, -1))
        out[buf_rows] = feat.cpu().numpy().astype(np.float16)
        buf_imgs.clear(); buf_rows.clear()

    with env.begin() as txn:
        for idx in range(n):
            _, patch = decode_record(txn.get(struct.pack(">Q", idx)), patch_size)
            buf_imgs.append(patch); buf_rows.append(idx)
            if len(buf_rows) == batch_size:
                flush()
        flush()
    out.flush()
    env.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_embed.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add `phase_embed` to the driver**

Add to `scripts/spatial_gnn_probe.py` (and wire into the `--phase` dispatch dict):

```python
def phase_embed() -> None:
    import numpy as np
    from dapidl.graph.embed import extract_embeddings, pca_fit_transform
    n = int(np.load(LMDB_DIR / "labels.npy").shape[0])
    ckpt = Path("pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt")
    emb_path = OUT / "embeddings_f16.npy"
    extract_embeddings(LMDB_DIR, ckpt, emb_path, n=n, batch_size=256)
    emb = np.load(emb_path, mmap_mode="r")
    red, _ = pca_fit_transform(emb, n_components=128)
    np.save(OUT / "embeddings_pca128.npy", red)
    logger.info(f"embeddings: {emb.shape} -> pca {red.shape}")
```

- [ ] **Step 6: Commit**

```bash
git add src/dapidl/graph/embed.py tests/test_graph_embed.py scripts/spatial_gnn_probe.py
git commit -m "feat(graph): frozen-EffNet embedding extraction + PCA-128 reduction"
```

- [ ] **Step 7: [CONTROLLER RUN] Extract embeddings (GPU)**

First check memory: `nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv` (need ~3 GB free for batch 256; reduce batch if tight). Estimated host disk for `embeddings_f16.npy`: ~8.1 GB.
Run: `uv run python scripts/spatial_gnn_probe.py --phase embed`
Expected: writes `embeddings_f16.npy` (2277877, 1792) and `embeddings_pca128.npy` (2277877, 128); ~30–40 min.

---

### Task 6: Stage-2 model units (CNN node encoder + scatter-mean GraphSAGE)

**Files:**
- Create: `src/dapidl/graph/gnn.py`
- Test: `tests/test_graph_gnn.py`

PyG is **not installed**; aggregation is hand-rolled `scatter_mean`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_gnn.py
import torch
from dapidl.graph.gnn import NucleusNodeCNN, scatter_mean, SageCellTyper


def test_scatter_mean_matches_hand_average():
    src = torch.tensor([0, 0, 1])         # node 0 has 2 neighbours, node 1 has 1
    dst = torch.tensor([1, 2, 2])
    x = torch.tensor([[1.0], [3.0], [5.0]])
    out = scatter_mean(x[dst], src, num_nodes=3)
    assert torch.allclose(out[0], torch.tensor([4.0]))   # mean(x[1], x[2]) = mean(3,5)
    assert torch.allclose(out[1], torch.tensor([5.0]))   # mean(x[2]) = 5
    assert torch.allclose(out[2], torch.tensor([0.0]))   # node 2 has no out-edges


def test_node_cnn_output_shape():
    cnn = NucleusNodeCNN(out_dim=128)
    y = cnn(torch.randn(4, 1, 40, 40))
    assert y.shape == (4, 128)


def test_sage_celltyper_forward_shape():
    model = SageCellTyper(node_dim=128, hidden=64, num_classes=4, layers=2)
    crops = torch.randn(6, 1, 40, 40)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    logits = model(crops, edge_index)
    assert logits.shape == (6, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph_gnn.py -v`
Expected: FAIL — `ModuleNotFoundError: ... gnn`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/graph/gnn.py
"""Stage-2 clean learned-BANKSY: a small CNN on a tight nucleus crop (context-poor
nodes) + hand-rolled scatter-mean GraphSAGE (PyG is not a dependency)."""
from __future__ import annotations

import torch
from torch import nn


def scatter_mean(messages: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean of ``messages`` grouped by destination node ``index`` (here: aggregate
    each source node's neighbour messages). Nodes with no messages -> zeros."""
    d = messages.shape[1]
    out = torch.zeros(num_nodes, d, device=messages.device, dtype=messages.dtype)
    cnt = torch.zeros(num_nodes, 1, device=messages.device, dtype=messages.dtype)
    out.index_add_(0, index, messages)
    cnt.index_add_(0, index, torch.ones(index.shape[0], 1, device=messages.device, dtype=messages.dtype))
    return out / cnt.clamp_min(1.0)


class NucleusNodeCNN(nn.Module):
    """3 stride-2 conv blocks on a 1x40x40 nucleus crop -> GAP -> out_dim embedding."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.net = nn.Sequential(block(1, 32), block(32, 64), block(64, out_dim),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class SageLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h, edge_index):
        src, dst = edge_index
        agg = scatter_mean(h[dst], src, num_nodes=h.shape[0])
        return self.lin(torch.cat([h, agg], dim=1))


class SageCellTyper(nn.Module):
    def __init__(self, node_dim=128, hidden=64, num_classes=4, layers=2):
        super().__init__()
        self.encoder = NucleusNodeCNN(out_dim=node_dim)
        dims = [node_dim] + [hidden] * layers
        self.sage = nn.ModuleList(SageLayer(dims[i], dims[i + 1]) for i in range(layers))
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, crops, edge_index):
        h = self.encoder(crops)
        for layer in self.sage:
            h = torch.relu(layer(h, edge_index))
        return self.head(h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph_gnn.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/gnn.py tests/test_graph_gnn.py
git commit -m "feat(graph): nucleus-local CNN + scatter-mean GraphSAGE (no PyG)"
```

---

### Task 7: Stage 1 + gate + Stage 2 drivers (integration, controller-run)

**Files:**
- Modify: `scripts/spatial_gnn_probe.py` (add `phase_stage1`, `phase_stage2`)
- Test: `tests/test_graph_probe_eval.py` already covers the pure gate/eval; the drivers are integration-run.

- [ ] **Step 1: Add `phase_stage1` (BANKSY + LightGBM + λ=0 ablation)**

```python
def phase_stage1() -> None:
    import numpy as np
    import polars as pl
    import lightgbm as lgb
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from dapidl.graph.knn_graph import build_within_slide_knn
    from dapidl.graph.banksy_features import banksy_augment

    CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    pca = np.load(OUT / "embeddings_pca128.npy")            # (N,128), row-aligned
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()

    edge_index = build_within_slide_knn(coords, src, k=8)
    test_mask = src == "xenium_rep2"
    val_mask = src == "sthelar_breast_s3"
    train_mask = ~test_mask & ~val_mask

    results = {}
    for lam in [0.0, 0.2, 0.5, 0.8]:
        feats = banksy_augment(pca, edge_index, coords, lambda_=lam)
        dtr = lgb.Dataset(feats[train_mask], labels[train_mask])
        dval = lgb.Dataset(feats[val_mask], labels[val_mask], reference=dtr)
        booster = lgb.train(
            {"objective": "multiclass", "num_class": 4, "learning_rate": 0.05,
             "num_leaves": 63, "metric": "multi_logloss", "is_unbalance": True,
             "verbose": -1, "seed": 0},
            dtr, num_boost_round=600, valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = booster.predict(feats[test_mask]).argmax(1)
        truth = labels[test_mask]
        pre, rec, f1, sup = precision_recall_fscore_support(
            truth, pred, labels=[0, 1, 2, 3], zero_division=0)
        results[f"lambda_{lam}"] = {
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "per_class": {CLASSES[k]: {"f1": float(f1[k]), "support": int(sup[k])} for k in range(4)},
            "best_iteration": int(booster.best_iteration)}
        # persist rep2 predictions for the best non-zero lambda for McNemar (Step 3)
        np.save(OUT / f"stage1_pred_lambda{lam}.npy", pred)
    np.save(OUT / "stage1_truth.npy", labels[test_mask])
    (OUT / "stage1_metrics.json").write_text(json.dumps(
        {"baseline_effnet_macro_f1": 0.619, "results": results}, indent=2))
    logger.info("stage1 done -> stage1_metrics.json")
```

- [ ] **Step 2: [CONTROLLER RUN] Stage 1**

Run: `uv run python scripts/spatial_gnn_probe.py --phase stage1`
Expected: `stage1_metrics.json` with per-λ macro-F1 + per-class on rep2, including the **λ=0 ablation** (no graph) as the within-pipeline baseline. RAM: BANKSY matrix ≈ 3.3 GB.

- [ ] **Step 3: Compute the gate + McNemar + bootstrap from Stage 1 outputs**

Add `phase_gate` to the driver:

```python
def phase_gate() -> None:
    import numpy as np
    from dapidl.graph.probe_eval import gate_decision, mcnemar_test, bootstrap_macro_f1_ci
    m = json.loads((OUT / "stage1_metrics.json").read_text())["results"]
    best_lam = max([0.2, 0.5, 0.8], key=lambda L: m[f"lambda_{L}"]["macro_f1"])
    base = m["lambda_0.0"]; best = m[f"lambda_{best_lam}"]
    truth = np.load(OUT / "stage1_truth.npy")
    pred0 = np.load(OUT / "stage1_pred_lambda0.0.npy")
    predB = np.load(OUT / f"stage1_pred_lambda{best_lam}.npy")
    gate = gate_decision(
        macro_delta=best["macro_f1"] - base["macro_f1"],
        endo_delta=best["per_class"]["Endothelial"]["f1"] - base["per_class"]["Endothelial"]["f1"],
        stromal_delta=best["per_class"]["Stromal"]["f1"] - base["per_class"]["Stromal"]["f1"])
    lo, point, hi = bootstrap_macro_f1_ci(truth, predB, n_boot=1000, seed=0)
    out = {"best_lambda": best_lam, "gate": gate,
           "mcnemar_graph_vs_nograph": mcnemar_test(truth, pred0, predB),
           "macro_f1_ci": {"lo": lo, "point": point, "hi": hi}}
    (OUT / "stage1_gate.json").write_text(json.dumps(out, indent=2))
    logger.info(f"gate: proceed={gate['proceed']} ({gate['reason']}); best_lambda={best_lam}")
```

Run: `uv run python scripts/spatial_gnn_probe.py --phase gate` (add `"gate": phase_gate` to the dispatch dict).
Expected: `stage1_gate.json`; the `proceed` flag decides whether Step 4 runs.

- [ ] **Step 4: Add `phase_stage2` (gated)** — nucleus-local CNN + GraphSAGE, full-slide-graph training, rep2 eval

```python
def phase_stage2() -> None:
    import numpy as np, polars as pl, struct, lmdb, torch
    from torch import nn
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from dapidl.graph.knn_graph import build_within_slide_knn
    from dapidl.graph.gnn import SageCellTyper
    from dapidl.graph.embed import decode_record

    CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy(); coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = torch.from_numpy(reg["coarse_idx"].to_numpy()).long()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # nucleus-local 40x40 crops (centre of each 128 patch) read once into a uint8/16 cache
    n = len(reg); crop = 40; off = (128 - crop) // 2
    crops = np.empty((n, crop, crop), dtype=np.uint16)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for i in range(n):
            _, p = decode_record(txn.get(struct.pack(">Q", i)), 128)
            crops[i] = p[off:off + crop, off:off + crop]
    env.close()

    def to_tensor(rows):
        x = crops[rows].astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        return torch.from_numpy(x)[:, None].to(device)

    test_mask = src == "xenium_rep2"; val_mask = src == "sthelar_breast_s3"
    train_slides = [s for s in np.unique(src) if s not in ("xenium_rep2", "sthelar_breast_s3")]

    model = SageCellTyper(node_dim=128, hidden=64, num_classes=4, layers=2).to(device)
    from breast_pooled_train import class_weights  # reuse existing helper
    import sys; sys.path.insert(0, "scripts")
    w = class_weights(reg["coarse_idx"].to_numpy(), 4, max_ratio=10.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    lossf = nn.CrossEntropyLoss(weight=w)

    def slide_graph(mask):
        idx = np.where(mask)[0]
        remap = {g: l for l, g in enumerate(idx)}
        ei = build_within_slide_knn(coords[idx], src[idx], k=8)
        ei = np.vectorize(remap.get)(ei) if ei.size else ei
        return idx, torch.from_numpy(ei).long().to(device)

    best_val = -1.0; patience = 0
    for epoch in range(40):
        model.train()
        for s in train_slides:                       # one slide-graph per step
            idx, ei = slide_graph(src == s)
            opt.zero_grad()
            logits = model(to_tensor(idx), ei)
            loss = lossf(logits, labels[idx].to(device))
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vidx, vei = slide_graph(val_mask)
            vpred = model(to_tensor(vidx), vei).argmax(1).cpu().numpy()
        vf1 = f1_score(labels[vidx].numpy(), vpred, average="macro", zero_division=0)
        if vf1 > best_val:
            best_val = vf1; patience = 0; torch.save(model.state_dict(), OUT / "stage2_best.pt")
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(torch.load(OUT / "stage2_best.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        tidx, tei = slide_graph(test_mask)
        tpred = model(to_tensor(tidx), tei).argmax(1).cpu().numpy()
    truth = labels[tidx].numpy()
    pre, rec, f1, sup = precision_recall_fscore_support(truth, tpred, labels=[0,1,2,3], zero_division=0)
    np.save(OUT / "stage2_pred.npy", tpred); np.save(OUT / "stage2_truth.npy", truth)
    (OUT / "stage2_metrics.json").write_text(json.dumps({
        "macro_f1": float(f1_score(truth, tpred, average="macro", zero_division=0)),
        "per_class": {CLASSES[k]: {"f1": float(f1[k]), "support": int(sup[k])} for k in range(4)},
        "baseline_effnet_macro_f1": 0.619}, indent=2))
    logger.info("stage2 done -> stage2_metrics.json")
```

- [ ] **Step 5: [CONTROLLER RUN] Stage 2 (only if gate.proceed)**

Check `nvidia-smi` first. Run: `uv run python scripts/spatial_gnn_probe.py --phase stage2`
Expected: `stage2_metrics.json` + `stage2_pred.npy`. RAM for the 40×40 uint16 crop cache: 2.28M×40×40×2 B ≈ 7.3 GB (fits 62 GB; if tight, stream per-slide instead of caching all).

- [ ] **Step 6: Commit the drivers**

```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): stage1 BANKSY+LightGBM, gate, stage2 GraphSAGE drivers"
```

- [ ] **Step 7: [CONTROLLER RUN] Write the readout**

Synthesise `pipeline_output/spatial_gnn_probe_2026_06/readout.md` from `stage1_metrics.json`, `stage1_gate.json`, and (if run) `stage2_metrics.json`: did the spatial graph beat **0.619**, and did it specifically lift **Endothelial / Stromal** vs the λ=0 ablation and the 0.497 handcrafted floor? Include the McNemar p-value and the bootstrap CI. Commit the readout.

---

## Self-Review

**Spec coverage:** D1 probe goal → Tasks 5/7 (A/B vs 0.619). D2 staged → Task 7 (stage1, gate, stage2). D3 split → registry + masks in `phase_stage1`/`phase_stage2`. D4 within-slide k-NN k=8 → Task 1. D5 PCA-128 / 40px nucleus crop → Tasks 5/6/7. D6 integrity (McNemar, bootstrap, λ=0 ablation, floor baseline, batch-probe) → Task 3 + Task 7 readout. Phase 0 registry replay+verify → Task 4. RAM budget (PCA fix) → Tasks 5/7. **Gap noted:** the **batch-identity probe** from §Integrity is described in the spec but has no dedicated task; it is a small addition — fold a `phase_batch_probe` (train LightGBM on PCA-128 → predict `source`; report accuracy) into the Task 7 readout step, or accept it as an optional follow-up. (Left optional to keep the probe minimal; the slide-grouped split already prevents leakage.)

**Placeholder scan:** no TBD/TODO; every code step has complete, runnable code; commands are explicit with expected output.

**Type consistency:** `edge_index` is `(2, E)` int throughout (Tasks 1, 2, 6, 7); `decode_record(value, patch_size) -> (int, uint16[H,W])` used identically in Tasks 5 and 7; `build_within_slide_knn(coords, slide_ids, k)`, `banksy_augment(feats, edge_index, coords, lambda_)`, `gate_decision(macro_delta, endo_delta, stromal_delta)`, `SageCellTyper(node_dim, hidden, num_classes, layers)` signatures match between definition and call sites. `model.head.in_features` (Task 5) matches `DapiClassifier.head = nn.Linear(feat_dim, num_classes)`.

One fixed inconsistency: Stage 2 remaps global→local node indices before building each slice's edge_index (`slide_graph`), because `SageCellTyper` indexes `h[dst]` on the local crop batch, not global rows.
