# Graph-Arm Stage 3 — Pluggable Probe Harness + Run-First Pair (LOSO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Stage-2-proper spatial-GNN harness into a pluggable `(encoder × aggregator × splitter)` evaluator and run the two "run-first" experiments — frozen-EffNet features in the learned graph (E1) and spatial smoothing of production probabilities (E2) — under full leave-one-slide-out.

**Architecture:** Extract the four seams already implicit in `scripts/spatial_gnn_probe.py::phase_stage2_proper` into focused modules (`splits.py`, `smooth.py`, `encoders.py`, extended `gnn.py` + `knn_graph.py`, new `harness.py`). The two-arm ablation becomes "swap the `Aggregator`, hold encoder+head identical." A characterization test pins the refactor by reproducing the committed 0.537/0.628. The run-first experiments are thin driver phases that configure the harness; they are controller-run (GPU + full 2.28M-cell data), not subagent-TDD'd.

**Tech Stack:** Python 3.12 (`uv run`), numpy, scipy.sparse, polars, torch, scikit-learn, pytest. Pure-numpy cores (splits, smooth) and torch modules (encoders, aggregators, harness) are unit-tested; driver phases reuse the existing probe outputs in `pipeline_output/spatial_gnn_probe_2026_06/`.

**Spec:** `docs/superpowers/specs/2026-06-10-graph-arm-stage3-loso-design.md`

**Branch:** `feat/spatial-gnn-probe` (already checked out at `/mnt/work/git/dapidl`). Commit on this branch only. No AI attribution in commit messages.

**YAGNI refinements of the spec (intentional):**
- `normalized_adjacency` is realized as a **row-stochastic random-walk transition matrix with self-loops** (`transition_matrix`), so smoothing provably preserves the probability simplex and isolated nodes are well-defined. (The spec's "normalized adjacency" role is unchanged; this is the concrete, testable form.)
- The **PostHoc seam = the `smooth.py` module**, exercised directly by `phase_cands_loso`. No `PostHoc` Protocol wrapper class is added, because no polymorphic consumer (`run_ablation`) needs to swap post-hocs yet. Add it when a consumer does.
- `run_ablation`/`train_arm` do **not** take a `k` argument — `k` is implicit in the `(n, k)` neighbour table's second dimension.

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `src/dapidl/graph/splits.py` | `LOSOSplit`, `Stage2ProperSplit` — yield (name, train, val, test) index arrays; pure numpy | Create |
| `src/dapidl/graph/smooth.py` | `transition_matrix`, `smooth`, `correct_and_smooth` — graph label smoothing; pure numpy/scipy | Create |
| `src/dapidl/graph/knn_graph.py` | add `build_within_slide_nbr_table` — `(n,k)` neighbour table for the GNN minibatch loop | Modify |
| `src/dapidl/graph/gnn.py` | add `NoGraphAggregator`, `MeanAggregator` (thin `nn.Module`s) | Modify |
| `src/dapidl/graph/encoders.py` | `FrozenFeatureEncoder`, `CropCNNEncoder` — `encode(rows)->[B,out_dim]` | Create |
| `src/dapidl/graph/harness.py` | `GraphArmModel`, `ArmResult`, `train_arm`, `run_ablation` | Create |
| `scripts/spatial_gnn_probe.py` | `phase_logits`, `phase_stage3_loso`, `phase_cands_loso`, `phase_stage2_proper_harness`, `phase_stage3_readout`; extend `--phase` choices | Modify |
| `tests/test_graph_splits.py` | unit tests for splits | Create |
| `tests/test_graph_smooth.py` | unit tests for smoothing | Create |
| `tests/test_graph_knn.py` | add tests for the nbr table | Modify |
| `tests/test_graph_gnn.py` | add tests for aggregators | Modify |
| `tests/test_graph_encoders.py` | unit tests for encoders | Create |
| `tests/test_graph_harness.py` | unit tests for the harness | Create |

Tasks 1–6 are pure/unit TDD (subagent-implementable). Tasks 7–11 are `[CONTROLLER RUN]` driver wiring (GPU + full data) — write + commit the code, then the controller runs them and checks the outputs; do not attempt to unit-test the full-data phases.

---

### Task 1: `splits.py` — LOSO and Stage-2-proper splitters

**Files:**
- Create: `src/dapidl/graph/splits.py`
- Test: `tests/test_graph_splits.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_splits.py
import numpy as np
from dapidl.graph.splits import LOSOSplit, Stage2ProperSplit


def _toy():
    # 3 slides, 4 cells each; y in [0,1]; one -1 (unlabeled) cell per slide
    src = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    coords = np.tile(np.array([[0, 0.1], [0, 0.4], [0, 0.7], [0, 0.95]], dtype=float), (3, 1))
    labels = np.array([0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2, -1])
    return src, coords, labels


def test_loso_one_fold_per_slide_test_is_single_slide():
    src, coords, labels = _toy()
    folds = list(LOSOSplit(src, coords, labels, val_frac=0.20).folds())
    assert len(folds) == 3
    for name, tr, va, te in folds:
        assert set(src[te]) == {name}                       # test = exactly the held-out slide
        assert np.all(labels[te] != -1)                     # no unlabeled in test


def test_loso_sets_disjoint_and_exclude_unlabeled():
    src, coords, labels = _toy()
    for name, tr, va, te in LOSOSplit(src, coords, labels).folds():
        assert len(np.intersect1d(tr, va)) == 0
        assert len(np.intersect1d(tr, te)) == 0
        assert len(np.intersect1d(va, te)) == 0
        for s in (tr, va, te):
            assert np.all(labels[s] != -1)                  # -1 excluded everywhere
        assert name not in set(src[va])                     # val drawn from TRAINING slides only


def test_stage2proper_matches_inline_split():
    src = np.array(["xenium_rep1"] * 5 + ["xenium_rep2"] * 3 + ["sthelar_breast_s0"] * 2)
    coords = np.zeros((10, 2)); coords[:5, 1] = [0.1, 0.2, 0.5, 0.85, 0.95]  # rep1 y
    labels = np.array([0, 1, 2, 3, 0, 0, 1, 2, 0, 1])
    (name, tr, va, te), = list(Stage2ProperSplit(src, coords, labels, val_frac=0.20).folds())
    assert name == "xenium_rep2"
    assert set(src[te]) == {"xenium_rep2"}
    assert set(src[va]) == {"xenium_rep1"}                  # val is a rep1 stripe
    assert "sthelar_breast_s0" in set(src[tr])              # other slides are train
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_splits.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.graph.splits'`

- [ ] **Step 3: Implement `splits.py`**

```python
# src/dapidl/graph/splits.py
"""Train/val/test index splitters over the spatial registry. Pure numpy: no torch,
no IO. label == -1 (unlabeled context cells) are excluded from train/val/test
(they remain in the graph as context only)."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _ystripe_val(train_pool: np.ndarray, source: np.ndarray, coords: np.ndarray,
                 val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Within each slide present in `train_pool`, move its top-`val_frac` y-stripe to
    val. Returns (train_idx, val_idx), spatially separating val from train per slide."""
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    pool_src = source[train_pool]
    for s in np.unique(pool_src):
        sl = train_pool[pool_src == s]
        ythr = np.quantile(coords[sl, 1], 1.0 - val_frac)
        is_val = coords[sl, 1] > ythr
        val_parts.append(sl[is_val])
        train_parts.append(sl[~is_val])
    train = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    return np.sort(train), np.sort(val)


class LOSOSplit:
    """Leave-one-slide-out. `folds()` yields (held_out_slide, train_idx, val_idx,
    test_idx) once per unique slide: test = all labeled cells of the held-out slide;
    val = top `val_frac` y-stripe of each TRAINING slide; train = the rest. -1 excluded."""

    def __init__(self, source, coords, labels, val_frac: float = 0.20):
        self.source = np.asarray(source)
        self.coords = np.asarray(coords, dtype=float)
        self.labels = np.asarray(labels)
        self.val_frac = val_frac

    def folds(self) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        labeled = np.where(self.labels != -1)[0]
        lab_src = self.source[labeled]
        for held in np.unique(lab_src):
            test_idx = np.sort(labeled[lab_src == held])
            train_pool = labeled[lab_src != held]
            train_idx, val_idx = _ystripe_val(train_pool, self.source, self.coords, self.val_frac)
            yield str(held), train_idx, val_idx, test_idx


class Stage2ProperSplit:
    """The original Stage-2-proper split: val = top `val_frac` y-stripe of `val_slide`;
    test = `test_slide`; train = rest of `val_slide` + all other non-test slides.
    -1 excluded. Single fold (one yield)."""

    def __init__(self, source, coords, labels, val_frac: float = 0.20,
                 val_slide: str = "xenium_rep1", test_slide: str = "xenium_rep2"):
        self.source = np.asarray(source)
        self.coords = np.asarray(coords, dtype=float)
        self.labels = np.asarray(labels)
        self.val_frac = val_frac
        self.val_slide = val_slide
        self.test_slide = test_slide

    def folds(self) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        labeled = self.labels != -1
        vslide = (self.source == self.val_slide) & labeled
        ythr = np.quantile(self.coords[vslide, 1], 1.0 - self.val_frac)
        val_idx = np.where(vslide & (self.coords[:, 1] > ythr))[0]
        train_idx = np.where(
            (vslide & (self.coords[:, 1] <= ythr))
            | ((self.source != self.val_slide) & (self.source != self.test_slide) & labeled)
        )[0]
        test_idx = np.where((self.source == self.test_slide) & labeled)[0]
        yield self.test_slide, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_splits.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/splits.py tests/test_graph_splits.py
git commit -m "feat(graph): LOSO + Stage2Proper index splitters"
```

---

### Task 2: `smooth.py` — graph label smoothing (Correct-and-Smooth)

**Files:**
- Create: `src/dapidl/graph/smooth.py`
- Test: `tests/test_graph_smooth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_smooth.py
import numpy as np
from dapidl.graph.smooth import transition_matrix, smooth, correct_and_smooth


def _line_graph():
    # 0-1-2 path + isolated node 3 ; directed edges both ways along the path
    edge = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    return edge, 4


def test_transition_matrix_row_stochastic_incl_isolated():
    edge, n = _line_graph()
    T = transition_matrix(edge, n).toarray()
    assert np.allclose(T.sum(1), 1.0)            # every row (incl isolated node 3) sums to 1
    assert T[3, 3] == 1.0                         # isolated node maps to itself (self-loop)


def test_smooth_alpha_zero_is_identity():
    edge, n = _line_graph()
    T = transition_matrix(edge, n)
    p = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    assert np.allclose(smooth(p, T, alpha=0.0, iters=10), p)


def test_smooth_preserves_simplex():
    edge, n = _line_graph()
    T = transition_matrix(edge, n)
    p = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.3, 0.7]])
    out = smooth(p, T, alpha=0.8, iters=25)
    assert np.allclose(out.sum(1), 1.0)          # convex combination keeps rows on the simplex


def test_correct_step_inert_when_no_train_labels():
    # held-out-slide case: train_idx empty -> C&S reduces to smoothing-only
    edge, n = _line_graph()
    T = transition_matrix(edge, n)
    p = np.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
    cs = correct_and_smooth(p, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                            T, num_classes=2, iters=20)
    assert np.allclose(cs, smooth(p, T, alpha=0.8, iters=20))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_smooth.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.graph.smooth'`

- [ ] **Step 3: Implement `smooth.py`**

```python
# src/dapidl/graph/smooth.py
"""Post-hoc label smoothing over the spatial graph: a row-stochastic random-walk
transition matrix, PPR smoothing, and Correct-and-Smooth (Huang et al. 2020). Pure
numpy/scipy. Smoothing of a probability matrix preserves the simplex (convex update)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def transition_matrix(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    """Row-stochastic random-walk transition D^-1 (A_sym + I) with self-loops, built
    from a directed edge_index (2, E) of global node indices. Self-loops make every
    node (including isolated ones) a defined, simplex-preserving update."""
    src, dst = edge_index
    A = sp.coo_matrix((np.ones(src.shape[0]), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)        # symmetrize, binary
    A = (A + sp.identity(n, format="coo")).tocsr()  # self-loops
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0 / deg)
    return (Dinv @ A).tocsr()


def smooth(probs: np.ndarray, transition: sp.csr_matrix, alpha: float, iters: int) -> np.ndarray:
    """PPR diffusion p <- (1-alpha) p0 + alpha (T @ p), `iters` times. alpha=0 is the
    identity; with row-stochastic T and probs rows on the simplex, rows stay on it."""
    p0 = np.asarray(probs, dtype=np.float64)
    p = p0.copy()
    for _ in range(iters):
        p = (1.0 - alpha) * p0 + alpha * (transition @ p)
    return p


def correct_and_smooth(probs, train_idx, train_labels, transition, num_classes: int = 4,
                       alpha_correct: float = 0.8, alpha_smooth: float = 0.8,
                       iters: int = 30) -> np.ndarray:
    """Correct-and-Smooth. Correct: diffuse the train residual (one-hot truth - prob)
    and add it back. Smooth: diffuse the corrected probs. When `train_idx` is empty
    (held-out slide shares no labels with its within-slide graph) the Correct step is a
    no-op and this reduces to smoothing-only."""
    probs = np.asarray(probs, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    resid = np.zeros_like(probs)
    if len(train_idx) > 0:
        onehot = np.zeros((len(train_idx), num_classes))
        onehot[np.arange(len(train_idx)), np.asarray(train_labels, dtype=np.int64)] = 1.0
        resid[train_idx] = onehot - probs[train_idx]
    corrected = probs + smooth(resid, transition, alpha_correct, iters)
    corrected = np.clip(corrected, 0.0, None)
    rs = corrected.sum(1, keepdims=True)
    corrected = np.divide(corrected, rs, out=np.zeros_like(corrected), where=rs > 0)
    return smooth(corrected, transition, alpha_smooth, iters)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_smooth.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/smooth.py tests/test_graph_smooth.py
git commit -m "feat(graph): row-stochastic transition + PPR smooth + Correct-and-Smooth"
```

---

### Task 3: `knn_graph.py` — `(n,k)` within-slide neighbour table

**Files:**
- Modify: `src/dapidl/graph/knn_graph.py`
- Test: `tests/test_graph_knn.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_graph_knn.py`)

```python
# tests/test_graph_knn.py  (append)
import numpy as np
from dapidl.graph.knn_graph import build_within_slide_nbr_table


def test_nbr_table_shape_and_padding():
    # slide "a": 3 cells; slide "b": 1 cell -> all -1 padded
    coords = np.array([[0, 0], [0, 1], [0, 2], [5, 5]], dtype=float)
    src = np.array(["a", "a", "a", "b"])
    nbr = build_within_slide_nbr_table(coords, src, k=8)
    assert nbr.shape == (4, 8)
    assert np.all(nbr[3] == -1)                       # singleton slide -> no neighbours
    assert (nbr[0] >= 0).sum() == 2                   # cell 0 has 2 same-slide neighbours


def test_nbr_within_slide_only_and_excludes_self():
    coords = np.array([[0, 0], [0, 1], [10, 0], [10, 1]], dtype=float)
    src = np.array(["a", "a", "b", "b"])
    nbr = build_within_slide_nbr_table(coords, src, k=8)
    for i in range(4):
        for j in nbr[i]:
            if j >= 0:
                assert src[j] == src[i]               # neighbour on same slide
                assert j != i                          # never self
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_knn.py -q -k nbr`
Expected: FAIL with `ImportError: cannot import name 'build_within_slide_nbr_table'`

- [ ] **Step 3: Implement** (append to `src/dapidl/graph/knn_graph.py`)

```python
# src/dapidl/graph/knn_graph.py  (append; cKDTree already imported at top)

def build_within_slide_nbr_table(coords, slide_ids, k: int = 8) -> np.ndarray:
    """(n, k) int64 table of within-slide neighbour GLOBAL indices, -1 padded. Row i
    holds up to k nearest same-slide neighbours of node i (excluding itself); slides
    smaller than k+1 are padded with -1. This is the per-row form the GNN minibatch
    loop gathers (complementary to build_within_slide_knn's (2,E) edge_index)."""
    coords = np.asarray(coords, dtype=float)
    slide_ids = np.asarray(slide_ids)
    n = len(coords)
    nbr = np.full((n, k), -1, dtype=np.int64)
    for s in np.unique(slide_ids):
        idx = np.where(slide_ids == s)[0]
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        _, nn = cKDTree(coords[idx]).query(coords[idx], k=kk + 1)  # col 0 is the node itself
        nn = np.atleast_2d(nn)
        for col in range(kk):
            nbr[idx, col] = idx[nn[:, col + 1]]
    return nbr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_knn.py -q`
Expected: PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/knn_graph.py tests/test_graph_knn.py
git commit -m "feat(graph): (n,k) within-slide neighbour table for the GNN minibatch loop"
```

---

### Task 4: `gnn.py` — `NoGraphAggregator` and `MeanAggregator`

**Files:**
- Modify: `src/dapidl/graph/gnn.py`
- Test: `tests/test_graph_gnn.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_graph_gnn.py`)

```python
# tests/test_graph_gnn.py  (append)
import torch
from dapidl.graph.gnn import NoGraphAggregator, MeanAggregator


def test_nograph_aggregator_returns_zeros():
    agg = NoGraphAggregator()
    se = torch.randn(3, 5)
    ne = torch.randn(3, 4, 5)
    valid = torch.ones(3, 4)
    out = agg(se, ne, valid)
    assert out.shape == se.shape
    assert torch.all(out == 0)
    assert agg.needs_neighbours is False


def test_mean_aggregator_masked_mean():
    agg = MeanAggregator()
    se = torch.zeros(1, 2)
    ne = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])   # 3 neighbours
    valid = torch.tensor([[1.0, 1.0, 0.0]])                      # 3rd is padding -> ignored
    out = agg(se, ne, valid)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))       # mean of first two
    assert agg.needs_neighbours is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_gnn.py -q -k aggregator`
Expected: FAIL with `ImportError: cannot import name 'NoGraphAggregator'`

- [ ] **Step 3: Implement** (append to `src/dapidl/graph/gnn.py`)

```python
# src/dapidl/graph/gnn.py  (append; torch + nn already imported at top)

class NoGraphAggregator(nn.Module):
    """No-graph arm: ignores neighbours, returns zeros so the model sees only `se`."""
    needs_neighbours = False

    def forward(self, se, ne, valid):
        return torch.zeros_like(se)


class MeanAggregator(nn.Module):
    """Masked mean of neighbour features (the GraphSAGE-mean aggregation). `valid` is
    1.0 where a neighbour exists, 0.0 for -1 padding."""
    needs_neighbours = True

    def forward(self, se, ne, valid):
        cnt = valid.sum(1, keepdim=True).clamp_min(1.0)
        return (ne * valid[:, :, None]).sum(1) / cnt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_gnn.py -q`
Expected: PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/gnn.py tests/test_graph_gnn.py
git commit -m "feat(graph): NoGraph + Mean aggregators (the two-arm swap point)"
```

---

### Task 5: `encoders.py` — frozen-feature and crop-CNN node encoders

**Files:**
- Create: `src/dapidl/graph/encoders.py`
- Test: `tests/test_graph_encoders.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_encoders.py
import numpy as np
import torch
from dapidl.graph.encoders import FrozenFeatureEncoder, CropCNNEncoder


def test_frozen_encoder_lookup_values_and_shape():
    feats = np.arange(20, dtype=np.float32).reshape(5, 4)
    enc = FrozenFeatureEncoder(feats, device="cpu")
    assert enc.out_dim == 4
    out = enc.encode(np.array([0, 2, 4]))
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.tensor(feats[[0, 2, 4]]))


def test_frozen_encoder_projection_changes_out_dim():
    feats = np.random.RandomState(0).randn(6, 8).astype(np.float32)
    enc = FrozenFeatureEncoder(feats, device="cpu", proj_dim=3)
    assert enc.out_dim == 3
    assert enc.encode(np.array([1, 5])).shape == (2, 3)


def test_cropcnn_encoder_shape():
    crops = (np.random.RandomState(0).rand(7, 40, 40) * 65535).astype(np.uint16)
    enc = CropCNNEncoder(crops, device="cpu", out_dim=128)
    assert enc.out_dim == 128
    assert enc.encode(np.array([0, 3, 6])).shape == (3, 128)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_encoders.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.graph.encoders'`

- [ ] **Step 3: Implement `encoders.py`**

```python
# src/dapidl/graph/encoders.py
"""Node-feature encoders for the probe harness. Each exposes encode(rows) ->
[len(rows), out_dim] on `device`, hiding whether features come from a trainable conv
on 40px crops or a lookup into cached frozen embeddings."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dapidl.graph.gnn import NucleusNodeCNN


class FrozenFeatureEncoder(nn.Module):
    """Lookup into a cached (N, d) frozen-embedding array (numpy OR a device tensor —
    the caller may preload it once to avoid recopying per arm/fold). encode(rows)
    returns the rows on device; if proj_dim is set, a trainable LayerNorm->Linear
    projects d -> proj_dim. The frozen features themselves are never back-propagated."""

    def __init__(self, features, device: str, proj_dim: int | None = None):
        super().__init__()
        if isinstance(features, torch.Tensor):
            self._feats = features.to(device)
        else:
            self._feats = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
        self.device = device
        in_dim = int(self._feats.shape[1])
        if proj_dim is None:
            self.proj = None
            self.out_dim = in_dim
        else:
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, proj_dim)).to(device)
            self.out_dim = proj_dim

    def encode(self, rows: np.ndarray) -> torch.Tensor:
        x = self._feats[torch.as_tensor(rows, device=self.device, dtype=torch.long)]
        return self.proj(x) if self.proj is not None else x


class CropCNNEncoder(nn.Module):
    """Trainable conv encoder over 40px nucleus crops (reproduces Stage-2-proper's
    NucleusNodeCNN path). `crops` is the (N, crop, crop) uint16 array."""

    def __init__(self, crops: np.ndarray, device: str, out_dim: int = 128):
        super().__init__()
        self._crops = crops
        self.device = device
        self.out_dim = out_dim
        self.cnn = NucleusNodeCNN(out_dim=out_dim).to(device)

    def encode(self, rows: np.ndarray) -> torch.Tensor:
        x = self._crops[rows].astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        t = torch.from_numpy(x)[:, None].to(self.device)
        return self.cnn(t)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_encoders.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/encoders.py tests/test_graph_encoders.py
git commit -m "feat(graph): frozen-feature + crop-CNN node encoders (encode(rows) interface)"
```

---

### Task 6: `harness.py` — `GraphArmModel`, `train_arm`, `run_ablation`

**Files:**
- Create: `src/dapidl/graph/harness.py`
- Test: `tests/test_graph_harness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_harness.py
import numpy as np
import torch
from dapidl.graph.encoders import FrozenFeatureEncoder
from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
from dapidl.graph.harness import GraphArmModel, run_ablation


def test_graph_arm_model_forward_shape_and_aggregator_swap():
    feats = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    enc = FrozenFeatureEncoder(feats, device="cpu")
    rows = np.array([0, 1, 2])
    nbr_rows = np.array([[1, 2], [0, 2], [0, 1]])
    valid = torch.ones(3, 2)
    m_graph = GraphArmModel(enc, MeanAggregator(), node_dim=4, num_classes=4)
    m_nograph = GraphArmModel(enc, NoGraphAggregator(), node_dim=4, num_classes=4)
    assert m_graph(rows, nbr_rows, valid).shape == (3, 4)
    assert m_nograph(rows, nbr_rows, valid).shape == (3, 4)


def test_run_ablation_two_arms_on_synthetic_separable_graph():
    # Build a tiny but learnable problem: 2 slides, class == sign of feature[0].
    rng = np.random.RandomState(0)
    n = 200
    src = np.array(["a"] * 100 + ["b"] * 100)
    feats = rng.randn(n, 4).astype(np.float32)
    labels = (feats[:, 0] > 0).astype(np.int64)            # classes 0 and 1 (both present)
    coords = rng.rand(n, 2)
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    nbr = build_within_slide_nbr_table(coords, src, k=4)

    class _Split:
        def folds(self):
            tr = np.arange(0, 80)
            va = np.arange(80, 100)
            te = np.arange(100, 200)
            yield "b", tr, va, te

    enc_feats = feats
    res = run_ablation(lambda: FrozenFeatureEncoder(enc_feats, "cpu"),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=15, patience=5)
    assert set(res["folds"]["b"].keys()) >= {"nograph", "graph", "delta_macro",
                                             "mcnemar_graph_vs_nograph"}
    assert res["folds"]["b"]["nograph"]["macro_f1"] > 0.6   # learns the separable signal
    assert "pooled" in res and "macro_graph" in res["pooled"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_harness.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'dapidl.graph.harness'`

- [ ] **Step 3: Implement `harness.py`**

```python
# src/dapidl/graph/harness.py
"""Pluggable probe harness: GraphArmModel (encoder + aggregator + head) and the
train/eval/ablation loop extracted from phase_stage2_proper. The two arms of an
ablation differ ONLY in the aggregator. `k` is implicit in the (n, k) nbr table."""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn

CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]


class GraphArmModel(nn.Module):
    """encoder + aggregator + head. The two arms differ ONLY in `aggregator`. Neighbour
    features are encoded only when the aggregator needs them (no-graph skips the work)."""

    def __init__(self, encoder, aggregator, node_dim: int, hidden: int = 64, num_classes: int = 4):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.lin = nn.Linear(node_dim * 2, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, self_rows, nbr_rows, valid):
        se = self.encoder.encode(self_rows)                              # [B, d]
        if getattr(self.aggregator, "needs_neighbours", True):
            b, k = nbr_rows.shape
            ne = self.encoder.encode(nbr_rows.reshape(-1)).reshape(b, k, -1)
        else:
            ne = se[:, None, :]                                          # placeholder, unused
        agg = self.aggregator(se, ne, valid)                            # [B, d]
        return self.head(torch.relu(self.lin(torch.cat([se, agg], 1))))


@dataclass
class ArmResult:
    macro_f1: float
    per_class: dict
    val_macro_f1: float
    pred: np.ndarray


def train_arm(encoder_factory: Callable[[], nn.Module], aggregator, *, nbr, labels,
              train_idx, val_idx, test_idx, num_classes: int = 4, device: str = "cpu",
              epochs: int = 40, patience: int = 5, seed: int = 0, batch: int = 256) -> ArmResult:
    """Train one arm with early stopping on val macro-F1, evaluate on test. Reproduces
    phase_stage2_proper's loop (Adam 3e-4, cosine T_max=epochs, weighted CE with
    class_weights max_ratio=10.0)."""
    sys.path.insert(0, "scripts")
    from breast_pooled_train import class_weights

    torch.manual_seed(seed)
    encoder = encoder_factory()
    model = GraphArmModel(encoder, aggregator, node_dim=encoder.out_dim, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(labels[train_idx], num_classes, max_ratio=10.0).to(device)
    lossf = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed)

    def step(rows, train):
        nb = nbr[rows]
        valid = torch.from_numpy((nb >= 0).astype(np.float32)).to(device)
        safe = np.where(nb >= 0, nb, rows[:, None])
        logits = model(rows, safe, valid)
        if train:
            y = torch.from_numpy(labels[rows]).long().to(device)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        return logits.argmax(1).detach().cpu().numpy()

    @torch.no_grad()
    def evaluate(idx):
        if len(idx) == 0:
            return 0.0, np.empty(0, dtype=np.int64)
        model.eval()
        pred = np.concatenate([step(idx[i:i + batch], False) for i in range(0, len(idx), batch)])
        model.train()
        return f1_score(labels[idx], pred, average="macro", zero_division=0), pred

    best_val, wait, best_state = -1.0, 0, None
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), batch):
            step(order[i:i + batch], True)
        sched.step()
        vf1, _ = evaluate(val_idx)
        if vf1 > best_val:
            best_val, wait = vf1, 0
            best_state = {kk: v.cpu().clone() for kk, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, tpred = evaluate(test_idx)
    truth = labels[test_idx]
    _, _, f1, sup = precision_recall_fscore_support(
        truth, tpred, labels=list(range(num_classes)), zero_division=0)
    per_class = {CLASSES[c]: {"f1": float(f1[c]), "support": int(sup[c])} for c in range(num_classes)}
    return ArmResult(float(f1_score(truth, tpred, average="macro", zero_division=0)),
                     per_class, float(best_val), tpred)


def run_ablation(encoder_factory, aggregators: dict, splitter, *, nbr, labels,
                 num_classes: int = 4, device: str = "cpu", **train_kw) -> dict:
    """Loop folds × arms; per fold record each arm's metrics, the graph-nograph delta,
    and per-fold McNemar; then pool predictions across folds for a pooled macro, pooled
    per-class, and pooled McNemar. Requires arm tags 'nograph' and 'graph' for the delta."""
    from dapidl.graph.probe_eval import mcnemar_test

    out: dict = {"folds": {}, "pooled": {}}
    pooled_pred = {tag: [] for tag in aggregators}
    pooled_truth: list[np.ndarray] = []
    for name, tr, va, te in splitter.folds():
        fold: dict = {}
        preds: dict = {}
        for tag, agg in aggregators.items():
            res = train_arm(encoder_factory, agg, nbr=nbr, labels=labels,
                            train_idx=tr, val_idx=va, test_idx=te,
                            num_classes=num_classes, device=device, **train_kw)
            fold[tag] = {"macro_f1": res.macro_f1, "per_class": res.per_class,
                         "val_macro_f1": res.val_macro_f1}
            preds[tag] = res.pred
            pooled_pred[tag].append(res.pred)
        truth = labels[te]
        pooled_truth.append(truth)
        if "graph" in preds and "nograph" in preds:
            fold["delta_macro"] = round(fold["graph"]["macro_f1"] - fold["nograph"]["macro_f1"], 4)
            fold["mcnemar_graph_vs_nograph"] = mcnemar_test(truth, preds["nograph"], preds["graph"])
        out["folds"][name] = fold

    if "graph" in aggregators and "nograph" in aggregators and pooled_truth:
        T = np.concatenate(pooled_truth)
        G = np.concatenate(pooled_pred["graph"])
        N = np.concatenate(pooled_pred["nograph"])
        _, _, f1g, _ = precision_recall_fscore_support(T, G, labels=list(range(num_classes)), zero_division=0)
        out["pooled"] = {
            "macro_graph": float(f1_score(T, G, average="macro", zero_division=0)),
            "macro_nograph": float(f1_score(T, N, average="macro", zero_division=0)),
            "per_class_graph": {CLASSES[c]: float(f1g[c]) for c in range(num_classes)},
            "mcnemar_graph_vs_nograph": mcnemar_test(T, N, G),
        }
        out["pooled"]["delta_macro"] = round(out["pooled"]["macro_graph"] - out["pooled"]["macro_nograph"], 4)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_harness.py -q`
Expected: PASS (2 passed). If the separable-signal assertion is flaky, it indicates a real training bug — do not loosen the threshold without investigating.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/harness.py tests/test_graph_harness.py
git commit -m "feat(graph): pluggable probe harness (GraphArmModel + train_arm + run_ablation)"
```

---

### Task 7: `[CONTROLLER RUN]` `phase_logits` — dump production EffNet probabilities

**Files:**
- Modify: `scripts/spatial_gnn_probe.py` (add `phase_logits`, extend `--phase` choices)

- [ ] **Step 1: Add `phase_logits`** (insert after `phase_embed`)

```python
def phase_logits() -> None:
    """[GPU] Dump the production EffNet's softmax class probabilities per cell -> (N,4).
    The honest Correct-and-Smooth base predictor (pca128 is lossy and cannot reconstruct
    logits). Mirrors embed.extract_embeddings but applies model.head + softmax."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import torch

    from dapidl.graph.embed import decode_record
    sys.path.insert(0, "scripts")
    from breast_pooled_train import DapiClassifier

    n = int(np.load(LMDB_DIR / "labels.npy").shape[0])
    ckpt = Path("pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DapiClassifier(num_classes=4, backbone="efficientnetv2_rw_s")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict") or state.get("model") or state)
    model.eval().to(device)

    probs = np.empty((n, 4), dtype=np.float32)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    buf: list[np.ndarray] = []
    rows: list[int] = []

    def flush():
        if not rows:
            return
        x = np.stack(buf).astype(np.float32) / 65535.0
        x = (x - 0.485) / 0.229
        t = torch.from_numpy(x)[:, None, :, :].to(device)
        with torch.no_grad():
            p = torch.softmax(model.head(model.backbone(t.expand(-1, 3, -1, -1))), dim=1)
        probs[rows] = p.cpu().numpy().astype(np.float32)
        buf.clear(); rows.clear()

    with env.begin() as txn:
        for i in range(n):
            _, patch = decode_record(txn.get(struct.pack(">Q", i)), 128)
            buf.append(patch); rows.append(i)
            if len(rows) == 256:
                flush()
        flush()
    env.close()
    np.save(OUT / "probs_production.npy", probs)
    logger.info(f"logits: production softmax probs {probs.shape} -> probs_production.npy")
```

- [ ] **Step 2: Register the phase** in `main()` (lines 395–399)

Dispatch is a `phases` dict keyed by the `--phase` choices. Add `"logits"` to BOTH the `choices` list and the `phases` dict (register each phase in the task that defines it — a dict entry referencing an undefined function breaks import):

```python
    ap.add_argument("--phase", required=True,
                    choices=["registry", "embed", "stage1", "gate", "stage2", "stage2_proper",
                             "logits"])
    args = ap.parse_args()
    phases = {"registry": phase_registry, "embed": phase_embed, "stage1": phase_stage1,
              "gate": phase_gate, "stage2": phase_stage2, "stage2_proper": phase_stage2_proper,
              "logits": phase_logits}
```

- [ ] **Step 3: Controller runs it** (check GPU first)

```bash
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
uv run python scripts/spatial_gnn_probe.py --phase logits
```
Expected: writes `pipeline_output/spatial_gnn_probe_2026_06/probs_production.npy` of shape `(N, 4)`; verify `python -c "import numpy as np; p=np.load('pipeline_output/spatial_gnn_probe_2026_06/probs_production.npy'); print(p.shape, p.sum(1)[:3])"` shows rows summing to ~1.0.

- [ ] **Step 4: Commit**

```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_logits — dump production EffNet softmax probs for C&S"
```

---

### Task 8: `[CONTROLLER RUN]` `phase_stage3_loso` — E1 frozen-feature GNN ablation under LOSO

**Files:**
- Modify: `scripts/spatial_gnn_probe.py` (add `phase_stage3_loso`)

- [ ] **Step 1: Add `phase_stage3_loso`**

```python
def phase_stage3_loso() -> None:
    """[CONTROLLER RUN] E1: frozen-EffNet features into the learned graph, two-arm
    (nograph=NoGraph, graph=Mean) ablation under leave-one-slide-out over all slides.
    Features preloaded to device once and shared across arms/folds."""
    import numpy as np
    import polars as pl
    import torch

    from dapidl.graph.encoders import FrozenFeatureEncoder
    from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    from dapidl.graph.splits import LOSOSplit

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    pca = np.load(OUT / "embeddings_pca128.npy")

    nbr = build_within_slide_nbr_table(coords, src, k=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats_dev = torch.from_numpy(np.ascontiguousarray(pca, dtype=np.float32)).to(device)  # once

    res = run_ablation(lambda: FrozenFeatureEncoder(feats_dev, device),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       LOSOSplit(src, coords, labels, val_frac=0.20),
                       nbr=nbr, labels=labels, device=device)
    res["baseline_effnet_macro_f1"] = 0.619
    (OUT / "stage3_loso_metrics.json").write_text(json.dumps(res, indent=2))
    p = res.get("pooled", {})
    logger.info(f"stage3-loso pooled: nograph={p.get('macro_nograph')} "
                f"graph={p.get('macro_graph')} delta_macro={p.get('delta_macro')}")
```

Register it: add `"stage3_loso"` to the `choices` list and `"stage3_loso": phase_stage3_loso` to the `phases` dict in `main()`.

- [ ] **Step 2: Controller runs it** (depends on `embeddings_pca128.npy` + `spatial_registry.parquet` already produced)

```bash
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
uv run python scripts/spatial_gnn_probe.py --phase stage3_loso
```
Expected: writes `stage3_loso_metrics.json` with `folds` (6 entries, one per slide) each having `nograph`, `graph`, `delta_macro`, `mcnemar_graph_vs_nograph`, plus a `pooled` block. Sanity: pooled `macro_nograph` should be near Stage-1's strong baseline (~0.70), and `delta_macro` is the leakage-immune graph lift (expected small, ~+0.02, per the spec).

- [ ] **Step 3: Commit**

```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_stage3_loso — E1 frozen-feature GNN two-arm ablation (LOSO)"
```

---

### Task 9: `[CONTROLLER RUN]` `phase_cands_loso` — E2 spatial smoothing + transductive UB

**Files:**
- Modify: `scripts/spatial_gnn_probe.py` (add `phase_cands_loso`)

- [ ] **Step 1: Add `phase_cands_loso`**

```python
def phase_cands_loso() -> None:
    """[CONTROLLER RUN] E2: smooth the production EffNet probabilities over each held-out
    slide's within-slide graph (held-out => Correct step inert => smoothing-only), versus
    raw argmax; plus a within-slide TRANSDUCTIVE upper bound (reveal 20% of the held-out
    slide's labels, run full C&S, score the other 80%) — a diagnostic, not production."""
    import numpy as np
    import polars as pl
    from sklearn.metrics import f1_score

    from dapidl.graph.knn_graph import build_within_slide_knn
    from dapidl.graph.probe_eval import mcnemar_test
    from dapidl.graph.smooth import correct_and_smooth, smooth, transition_matrix
    from dapidl.graph.splits import LOSOSplit

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    probs = np.load(OUT / "probs_production.npy")
    n = len(labels)

    edge = build_within_slide_knn(coords, src, k=8)
    T = transition_matrix(edge, n)
    sm_full = smooth(probs, T, alpha=0.8, iters=30)         # global; held-out slide is its own component

    folds: dict = {}
    pooled_truth, pooled_raw, pooled_sm = [], [], []
    for name, _tr, _va, te in LOSOSplit(src, coords, labels, val_frac=0.20).folds():
        truth = labels[te]
        raw = probs[te].argmax(1)
        sm = sm_full[te].argmax(1)
        # transductive upper bound: reveal a 20% sample of THIS slide's labels, score the rest
        rng = np.random.default_rng(0)
        reveal = rng.permutation(te)[: max(1, len(te) // 5)]
        held = np.setdiff1d(te, reveal)
        cs_full = correct_and_smooth(probs, reveal, labels[reveal], T, iters=30)
        cs = cs_full[held].argmax(1)
        folds[name] = {
            "macro_raw": float(f1_score(truth, raw, average="macro", zero_division=0)),
            "macro_smooth": float(f1_score(truth, sm, average="macro", zero_division=0)),
            "macro_cs_transductive_ub": float(f1_score(labels[held], cs, average="macro", zero_division=0)),
            "mcnemar_smooth_vs_raw": mcnemar_test(truth, raw, sm),
        }
        pooled_truth.append(truth); pooled_raw.append(raw); pooled_sm.append(sm)

    PT = np.concatenate(pooled_truth); PR = np.concatenate(pooled_raw); PS = np.concatenate(pooled_sm)
    out = {"folds": folds, "pooled": {
        "macro_raw": float(f1_score(PT, PR, average="macro", zero_division=0)),
        "macro_smooth": float(f1_score(PT, PS, average="macro", zero_division=0)),
        "mcnemar_smooth_vs_raw": mcnemar_test(PT, PR, PS)}}
    (OUT / "cands_loso_metrics.json").write_text(json.dumps(out, indent=2))
    logger.info(f"cands-loso pooled: raw={out['pooled']['macro_raw']} smooth={out['pooled']['macro_smooth']}")
```

Register it: add `"cands_loso"` to the `choices` list and `"cands_loso": phase_cands_loso` to the `phases` dict in `main()`.

- [ ] **Step 2: Controller runs it** (depends on `probs_production.npy` from Task 7)

```bash
uv run python scripts/spatial_gnn_probe.py --phase cands_loso
```
Expected: writes `cands_loso_metrics.json`. Interpretation (per spec): `macro_smooth` vs `macro_raw` is the near-free held-out-slide gain (expected small); `macro_cs_transductive_ub` >> `macro_smooth` would confirm the gain needs same-slide labels (production-irrelevant) — i.e. smoothing is the production-faithful number.

- [ ] **Step 3: Commit**

```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_cands_loso — E2 spatial smoothing + transductive UB (LOSO)"
```

---

### Task 10: `[CONTROLLER RUN]` characterization — re-express Stage-2-proper via the harness

**Files:**
- Modify: `scripts/spatial_gnn_probe.py` (add `phase_stage2_proper_harness`)
- Test: `tests/test_graph_harness.py` (add the characterization gate, marked slow)

**This task pins the refactor: the harness configured as Stage-2-proper must reproduce the committed `stage2_proper_metrics.json` (no-graph 0.537 / graph 0.628) within ±0.02 macro-F1.** Init order differs from the inline `Arm`, so the gate is a tolerance, not bit-exactness.

- [ ] **Step 1: Add `phase_stage2_proper_harness`**

```python
def phase_stage2_proper_harness() -> None:
    """[CONTROLLER RUN] Characterization: reproduce phase_stage2_proper through the new
    harness (CropCNNEncoder + {NoGraph, Mean} + Stage2ProperSplit) to prove the refactor
    is faithful. Writes stage2_proper_harness_metrics.json for comparison with the
    committed stage2_proper_metrics.json."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import polars as pl
    import torch

    from dapidl.graph.embed import decode_record
    from dapidl.graph.encoders import CropCNNEncoder
    from dapidl.graph.gnn import MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    from dapidl.graph.splits import Stage2ProperSplit
    sys.path.insert(0, "scripts")

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    n, crop, off = len(reg), 40, (128 - 40) // 2

    crops = np.empty((n, crop, crop), dtype=np.uint16)
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        for i in range(n):
            _, p = decode_record(txn.get(struct.pack(">Q", i)), 128)
            crops[i] = p[off:off + crop, off:off + crop]
    env.close()

    nbr = build_within_slide_nbr_table(coords, src, k=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_ablation(lambda: CropCNNEncoder(crops, device, out_dim=128),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       Stage2ProperSplit(src, coords, labels, val_frac=0.20),
                       nbr=nbr, labels=labels, device=device)
    (OUT / "stage2_proper_harness_metrics.json").write_text(json.dumps(res, indent=2))
    f = res["folds"]["xenium_rep2"]
    logger.info(f"stage2-proper-harness: nograph={f['nograph']['macro_f1']:.4f} "
                f"graph={f['graph']['macro_f1']:.4f} (committed ref: 0.537 / 0.628)")
```

Register it: add `"stage2_proper_harness"` to the `choices` list and `"stage2_proper_harness": phase_stage2_proper_harness` to the `phases` dict in `main()`.

- [ ] **Step 2: Add the characterization gate test** (append to `tests/test_graph_harness.py`)

```python
# tests/test_graph_harness.py  (append)
import json
import os
from pathlib import Path

import pytest

_OUT = Path("pipeline_output/spatial_gnn_probe_2026_06")


@pytest.mark.skipif(not (_OUT / "stage2_proper_harness_metrics.json").exists(),
                    reason="run `--phase stage2_proper_harness` first (controller/GPU step)")
def test_characterization_reproduces_committed_stage2_proper():
    h = json.loads((_OUT / "stage2_proper_harness_metrics.json").read_text())["folds"]["xenium_rep2"]
    assert abs(h["nograph"]["macro_f1"] - 0.537) <= 0.02   # refactor faithful (no-graph arm)
    assert abs(h["graph"]["macro_f1"] - 0.628) <= 0.02     # refactor faithful (graph arm)
    assert h["graph"]["macro_f1"] > h["nograph"]["macro_f1"]
```

- [ ] **Step 3: Controller runs the phase, then the gate**

```bash
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
uv run python scripts/spatial_gnn_probe.py --phase stage2_proper_harness
uv run pytest tests/test_graph_harness.py::test_characterization_reproduces_committed_stage2_proper -q
```
Expected: phase writes `stage2_proper_harness_metrics.json`; the gate passes (both arms within ±0.02 of 0.537/0.628). **If it fails, STOP — the refactor changed behaviour; debug before proceeding (do not loosen the tolerance).**

- [ ] **Step 4: Commit**

```bash
git add scripts/spatial_gnn_probe.py tests/test_graph_harness.py
git commit -m "test(graph): characterization — harness reproduces Stage-2-proper 0.537/0.628"
```

---

### Task 11: `[CONTROLLER RUN]` `phase_stage3_readout` — the Stage-3 readout

**Files:**
- Modify: `scripts/spatial_gnn_probe.py` (add `phase_stage3_readout`)

- [ ] **Step 1: Add `phase_stage3_readout`**

```python
def phase_stage3_readout() -> None:
    """[CONTROLLER RUN] Compose stage3_readout.md from stage3_loso_metrics.json (E1) and
    cands_loso_metrics.json (E2): per-fold + pooled macro/delta, feature-clean vs Δ-clean
    tiers, and the GNN-vs-smoother comparison.

    FEATURE_CLEAN: slides the frozen extractor did NOT train on (absolute F1 honest).
    Determine from the checkpoint's training config under
    pipeline_output/h2h_2026_05_30/ (e.g. a config.json / args listing the train slides).
    If it cannot be recovered, LEAVE THIS EMPTY: the readout then reports all folds
    Δ-clean only and labels every absolute F1 'extractor-membership unverified'. The Δ
    claim never depends on this set."""
    import json as _json

    e1 = _json.loads((OUT / "stage3_loso_metrics.json").read_text())
    e2 = _json.loads((OUT / "cands_loso_metrics.json").read_text())

    FEATURE_CLEAN: set[str] = set()   # populate from h2h training config if recoverable; else empty

    lines = ["# Graph-Arm Stage 3 — Readout (LOSO)\n",
             "## E1 — Frozen-EffNet features in the learned graph (two-arm, leave-one-slide-out)\n",
             "Δ = graph − no-graph is **leakage-immune** (both arms share identical frozen features).\n",
             "| Held-out slide | tier | no-graph | graph | Δ macro | McNemar p |",
             "|---|---|---|---|---|---|"]
    for name, f in e1["folds"].items():
        tier = "feature-clean" if name in FEATURE_CLEAN else "Δ-clean (abs F1 optimistic/unverified)"
        mp = f.get("mcnemar_graph_vs_nograph", {}).get("p_value")
        lines.append(f"| {name} | {tier} | {f['nograph']['macro_f1']:.4f} | "
                     f"{f['graph']['macro_f1']:.4f} | {f.get('delta_macro'):+.4f} | {mp} |")
    p = e1.get("pooled", {})
    lines += [f"\n**Pooled:** no-graph {p.get('macro_nograph')}, graph {p.get('macro_graph')}, "
              f"Δ {p.get('delta_macro')}, pooled McNemar p="
              f"{p.get('mcnemar_graph_vs_nograph', {}).get('p_value')}. EffNet baseline 0.619.\n",
              "## E2 — Spatial smoothing of production probabilities (near-free)\n",
              "| Held-out slide | raw | smooth | C&S transductive UB |",
              "|---|---|---|---|"]
    for name, f in e2["folds"].items():
        lines.append(f"| {name} | {f['macro_raw']:.4f} | {f['macro_smooth']:.4f} | "
                     f"{f['macro_cs_transductive_ub']:.4f} |")
    pe = e2.get("pooled", {})
    lines += [f"\n**Pooled:** raw {pe.get('macro_raw')}, smooth {pe.get('macro_smooth')}, "
              f"smooth-vs-raw McNemar p={pe.get('mcnemar_smooth_vs_raw', {}).get('p_value')}.\n",
              "## Honest framing\n",
              "- Δ (graph lift) is the scientific claim; expected small (~+0.02) on strong frozen "
              "features (vs +0.091 on the from-scratch CNN) — both arms now start strong.\n",
              "- If E2's transductive UB >> smooth, the gain needs same-slide labels (not production); "
              "smoothing is the production-faithful number.\n",
              "- Ceiling ~0.68–0.73 macro; not 0.80. Flag any fold whose gain looks too good "
              "(composition leakage).\n"]
    if not FEATURE_CLEAN:
        lines.append("\n> NOTE: extractor training set unverified — all absolute F1 are "
                     "'extractor-membership unverified'; only Δ is asserted.\n")
    (OUT / "stage3_readout.md").write_text("\n".join(lines))
    logger.info("stage3 readout -> stage3_readout.md")
```

Register it: add `"stage3_readout"` to the `choices` list and `"stage3_readout": phase_stage3_readout` to the `phases` dict in `main()`.

- [ ] **Step 2: Controller runs it**

```bash
uv run python scripts/spatial_gnn_probe.py --phase stage3_readout
```
Expected: writes `pipeline_output/spatial_gnn_probe_2026_06/stage3_readout.md` with both experiments' tables and the honest framing.

- [ ] **Step 3: Commit**

```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_stage3_readout — LOSO readout (E1 ablation + E2 smoothing)"
```

---

## Final verification (controller)

- [ ] Run the full unit suite: `uv run pytest tests/test_graph_splits.py tests/test_graph_smooth.py tests/test_graph_knn.py tests/test_graph_gnn.py tests/test_graph_encoders.py tests/test_graph_harness.py -q` — all green.
- [ ] Lint: `uv run ruff check src/dapidl/graph/ scripts/spatial_gnn_probe.py` and `uv run ruff format --check src/dapidl/graph/`.
- [ ] The characterization gate (Task 10) passed (refactor faithful).
- [ ] `stage3_readout.md` exists and reads sensibly.
- [ ] Update memory: append the Stage-3 result to `project_graph_arm_improvement_roadmap` / `project_spatial_gnn_probe_result`.
- [ ] Final code review across the branch before any merge decision (subagent-driven-development's final reviewer, or a `code-reviewer` agent).

## Execution order

Tasks 1–6 (pure/unit) can be implemented and committed in dependency order (1→6), each fully green before the next. Tasks 7–11 are controller-run and require the existing probe artifacts (`spatial_registry.parquet`, `embeddings_pca128.npy`) already present in `pipeline_output/spatial_gnn_probe_2026_06/`; run them in order 7→8→9→10→11. Check `nvidia-smi` (2–4 GB free buffer) before each GPU phase (7, 8, 10).
