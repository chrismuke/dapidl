# Graph-Arm Stage 4 — Edge-Geometry GATv2 Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an edge-geometry GATv2 arm to the LOSO harness and test whether edge-conditioned attention beats plain mean-pool (and free Correct-and-Smooth) on the feature-clean rep2 fold, using true nuclear orientation (STHELAR polygons + a targeted Xenium StarDist pass).

**Architecture:** Node features stay frozen-EffNet PCA-128 (identical across arms). New per-node geometry (`node_geom.npy`) feeds rotation-invariant **edge** attributes (`build_edge_attr`) consumed by a new `EdgeGATv2Aggregator`. The aggregator threads through the Stage-3 harness via an additive `edge_attr` path; because GATv2 is parametric, the harness's aggregator argument becomes a **factory**. A 3-arm LOSO ablation (`nograph / mean / gatv2`) isolates the attention's marginal value; the characterization gate (0.537/0.628) guards the refactor.

**Tech Stack:** Python 3.12 (`uv run`), numpy, scipy, torch, scikit-image (`regionprops`), shapely/geopandas (STHELAR polygons), `starpose.qc` (StarDist, importable from `/home/chrism/git/starpose`), polars, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-graph-arm-stage4-gatv2-design.md`

**Branch:** `feat/spatial-gnn-probe` (already checked out at `/mnt/work/git/dapidl`). Commit on this branch only. No AI attribution in commit messages.

---

## Verified APIs (use exactly these)

- **Harness** (`src/dapidl/graph/harness.py`): `GraphArmModel(encoder, aggregator, node_dim, hidden=64, num_classes=4)`, `.forward(self_rows, nbr_rows, valid)`; `train_arm(encoder_factory, aggregator, *, nbr, labels, train_idx, val_idx, test_idx, num_classes=4, device="cpu", epochs=40, patience=5, seed=0, batch=256, lr=3e-4)`; `run_ablation(encoder_factory, aggregators: dict, splitter, *, nbr, labels, num_classes=4, device="cpu", **train_kw)`.
- **Aggregators** (`src/dapidl/graph/gnn.py`): `NoGraphAggregator` (`needs_neighbours=False`), `MeanAggregator` (`needs_neighbours=True`); both `forward(se, ne, valid)`.
- **StarDist** (`starpose.qc`): `s = SegmentationGroundedScorer(SegQCConfig(erode_px=1), gpu=True, pixel_size=0.2125)`; `masks, probs = s._segment(patch)` → `masks` is `int32 (128,128)` labeled (0=bg, 1..N nuclei); `masks.max()==0` when none found.
- **STHELAR polygons** (`src/dapidl/data/sthelar.py`): `load_nucleus_geometry_with_labels(slide_root, label_cols) -> GeoDataFrame` indexed by `cell_id`, column `geometry` (shapely Polygon, pixel coords). `STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")` (in `scripts/breast_dapi_lmdb.py`). Slide zarr is double-nested: resolve `outer = STHELAR_BASE / f"sdata_{name}.zarr"`, `slide_root = outer/outer.name if (outer/outer.name/"shapes").is_dir() else outer`.
- **Registry** `pipeline_output/spatial_gnn_probe_2026_06/spatial_registry.parquet`: `row_idx, source, cell_id, x_px, y_px, coarse_idx`. Sources: `xenium_rep1/rep2`, `sthelar_breast_s0/s1/s3/s6`. cell_id: Xenium `"1"`; STHELAR `"aaaaaaaa-1"`.
- **LMDB** `/mnt/work/datasets/derived/breast-6source-dapi-p128/patches.lmdb`; `from dapidl.graph.embed import decode_record` → `decode_record(value, 128) -> (label, uint16 128x128)`.

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `src/dapidl/graph/geometry.py` | `ellipse_from_points` — PCA → (angle, eccentricity) | Create |
| `src/dapidl/graph/edge_geometry.py` | `rbf`, `build_edge_attr` — rotation-invariant `(N,k,8)` edge attrs | Create |
| `src/dapidl/graph/gnn.py` | add `EdgeGATv2Aggregator` | Modify |
| `src/dapidl/graph/harness.py` | edge_attr threading + aggregator factories + `compare_pairs` | Modify |
| `scripts/spatial_gnn_probe.py` | factory-ize the 2 Stage-3 phases; add `node_geometry`, `stage4_gatv2`, `stage4_readout` | Modify |
| `tests/test_graph_geometry.py` | unit tests for geometry | Create |
| `tests/test_graph_edge_geometry.py` | unit tests for edge attrs | Create |
| `tests/test_graph_gnn.py` | add EdgeGATv2 tests | Modify |
| `tests/test_graph_harness.py` | add edge_attr/factory/compare_pairs tests | Modify |

Tasks 1–4 are pure/unit TDD (subagent-implementable). Tasks 5–8 are `[CONTROLLER RUN]` (GPU + real data) — write + commit the code, run it, check outputs.

---

### Task 1: `geometry.py` — `ellipse_from_points`

**Files:**
- Create: `src/dapidl/graph/geometry.py`
- Test: `tests/test_graph_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_geometry.py
import numpy as np
from dapidl.graph.geometry import ellipse_from_points


def _ellipse_pts(a, b, theta, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([a * np.cos(t), b * np.sin(t)], 1)        # axis-aligned, semi-axes a>b
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return xy @ R.T                                          # rotated by theta


def test_axis_aligned_angle_and_eccentricity():
    ang, ecc = ellipse_from_points(_ellipse_pts(10.0, 2.0, 0.0))
    assert abs(((ang + np.pi / 2) % np.pi) - np.pi / 2) < 0.05   # major axis ~ x-axis (angle ~0 mod pi)
    assert 0.9 < ecc < 1.0                                       # elongated -> high eccentricity


def test_rotation_equivariant_angle_invariant_ecc():
    phi = 0.7
    a0, e0 = ellipse_from_points(_ellipse_pts(10.0, 3.0, 0.0))
    a1, e1 = ellipse_from_points(_ellipse_pts(10.0, 3.0, phi))
    # angle rotates by phi (mod pi); eccentricity unchanged
    assert abs(((a1 - a0 - phi) % np.pi)) < 0.02 or abs(((a1 - a0 - phi) % np.pi) - np.pi) < 0.02
    assert abs(e1 - e0) < 0.01


def test_degenerate_returns_nan_angle():
    ang, ecc = ellipse_from_points(np.zeros((2, 2)))            # < 3 points
    assert np.isnan(ang) and ecc == 0.0
    ang2, ecc2 = ellipse_from_points(np.zeros((5, 2)))          # zero variance
    assert np.isnan(ang2) and ecc2 == 0.0
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_graph_geometry.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/dapidl/graph/geometry.py
"""Nuclear shape descriptors from a 2-D point set (polygon vertices or mask pixels).
Pure numpy. angle is the major-axis direction in the same (x, y) frame as the input
points (so it composes with edge directions atan2(dy, dx))."""
from __future__ import annotations

import numpy as np


def ellipse_from_points(points_xy: np.ndarray) -> tuple[float, float]:
    """PCA of a 2-D point cloud. Returns (angle_rad, eccentricity). angle = atan2 of the
    principal eigenvector (major-axis direction, defined mod pi); eccentricity =
    sqrt(1 - lam_min/lam_max) in [0, 1). Degenerate input (< 3 distinct points or zero
    variance) -> (nan, 0.0)."""
    p = np.asarray(points_xy, dtype=float)
    if p.shape[0] < 3:
        return float("nan"), 0.0
    p = p - p.mean(0)
    cov = (p.T @ p) / p.shape[0]
    evals, evecs = np.linalg.eigh(cov)                 # ascending
    lam_max, lam_min = float(evals[1]), float(evals[0])
    if lam_max <= 1e-12:
        return float("nan"), 0.0
    major = evecs[:, 1]                                 # eigenvector of the larger eigenvalue
    angle = float(np.arctan2(major[1], major[0]))
    ecc = float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))
    return angle, ecc
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_graph_geometry.py -q` → PASS (3). Then `uv run ruff check src/dapidl/graph/geometry.py`.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/geometry.py tests/test_graph_geometry.py
git commit -m "feat(graph): ellipse_from_points — nuclear major-axis angle + eccentricity via PCA"
```

---

### Task 2: `edge_geometry.py` — rotation-invariant edge attributes

**Files:**
- Create: `src/dapidl/graph/edge_geometry.py`
- Test: `tests/test_graph_edge_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph_edge_geometry.py
import numpy as np
from dapidl.graph.edge_geometry import build_edge_attr


def _toy():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 50, (6, 2))
    node_geom = np.column_stack([rng.uniform(-np.pi, np.pi, 6),    # angle
                                 rng.uniform(0, 1, 6),             # ecc
                                 rng.uniform(0, 4, 6)])            # log_area
    nbr = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]])
    return coords, node_geom, nbr


def test_shape_and_padding():
    coords, ng, nbr = _toy()
    nbr2 = nbr.copy(); nbr2[0, 1] = -1
    ea = build_edge_attr(coords, ng, nbr2)
    assert ea.shape == (6, 2, 8)
    assert np.all(ea[0, 1] == 0.0)                        # -1 slot -> zero row


def test_rotation_invariant():
    coords, ng, nbr = _toy()
    ea0 = build_edge_attr(coords, ng, nbr)
    phi = 0.9
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    coords_r = coords @ R.T
    ng_r = ng.copy(); ng_r[:, 0] = ng[:, 0] + phi          # rotate axes with the slide
    ea1 = build_edge_attr(coords_r, ng_r, nbr)             # same nbr (rotation preserves kNN)
    assert np.allclose(ea0, ea1, atol=1e-6)


def test_nan_axis_zeros_directional_terms():
    coords, ng, nbr = _toy()
    ng[0, 0] = np.nan                                      # node 0 has no orientation
    ea = build_edge_attr(coords, ng, nbr)
    # for any edge touching node 0, the 3 directional dims (indices 3,4,5) are zero
    assert np.all(ea[0, :, 3:6] == 0.0)                    # node 0 as source
    assert np.all(ea[1, 0, 3:6] == 0.0)                    # node 0 as neighbour of node 1
    assert np.isfinite(ea).all()
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_graph_edge_geometry.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/dapidl/graph/edge_geometry.py
"""Rotation-invariant edge attributes for the GATv2 arm. (N, k, 8) per neighbour slot:
RBF(dist) x3, cos/sin(2*(edge_angle - axis_i)), |cos(axis_i - axis_j)|, ecc_j,
|log_area_i - log_area_j|. Pure numpy. node_geom is (N,3)=[angle, ecc, log_area];
angle may be nan (no orientation) -> the 3 directional terms are zeroed for that edge."""
from __future__ import annotations

import numpy as np


def rbf(dist: np.ndarray, centers, gamma: float) -> np.ndarray:
    """Gaussian RBF expansion: exp(-gamma (dist - c)^2) for each center c. -> (..., len(centers))."""
    d = np.asarray(dist)[..., None]
    c = np.asarray(centers, dtype=float)
    return np.exp(-gamma * (d - c) ** 2)


def build_edge_attr(coords: np.ndarray, node_geom: np.ndarray, nbr: np.ndarray,
                    rbf_centers=(4.0, 8.0, 16.0), rbf_gamma: float = 0.05) -> np.ndarray:
    """coords (N,2); node_geom (N,3)=[angle,ecc,log_area]; nbr (N,k) global indices (-1 pad).
    Returns edge_attr (N, k, 8), all rotation-invariant. -1 slots -> zero rows."""
    coords = np.asarray(coords, dtype=float)
    ng = np.asarray(node_geom, dtype=float)
    nbr = np.asarray(nbr)
    n, k = nbr.shape
    valid = nbr >= 0
    j = np.where(valid, nbr, 0)                                  # safe gather index
    src = np.broadcast_to(np.arange(n)[:, None], (n, k))

    d_xy = coords[j] - coords[src]                              # (N,k,2) from i to j
    dist = np.linalg.norm(d_xy, axis=2)                        # (N,k)
    edge_angle = np.arctan2(d_xy[..., 1], d_xy[..., 0])        # (N,k)
    axis_i = ng[src, 0]; axis_j = ng[j, 0]                     # (N,k)
    ecc_j = ng[j, 1]
    dlog_area = np.abs(ng[src, 2] - ng[j, 2])

    rbf_feats = rbf(dist, rbf_centers, rbf_gamma)              # (N,k,3)
    rel = edge_angle - axis_i                                  # nan where axis_i is nan
    cos2 = np.cos(2.0 * rel); sin2 = np.sin(2.0 * rel)
    align = np.abs(np.cos(axis_i - axis_j))
    directional_nan = ~np.isfinite(cos2) | ~np.isfinite(align) # either endpoint axis nan
    cos2 = np.where(directional_nan, 0.0, cos2)
    sin2 = np.where(directional_nan, 0.0, sin2)
    align = np.where(directional_nan, 0.0, align)

    ea = np.concatenate([rbf_feats,
                         cos2[..., None], sin2[..., None], align[..., None],
                         ecc_j[..., None], dlog_area[..., None]], axis=2).astype(np.float32)
    ea[~valid] = 0.0                                          # zero padded slots
    return ea
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_graph_edge_geometry.py -q` → PASS (3). `uv run ruff check src/dapidl/graph/edge_geometry.py`.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/edge_geometry.py tests/test_graph_edge_geometry.py
git commit -m "feat(graph): build_edge_attr — rotation-invariant (N,k,8) edge geometry for GATv2"
```

---

### Task 3: `gnn.py` — `EdgeGATv2Aggregator`

**Files:**
- Modify (append): `src/dapidl/graph/gnn.py`
- Test (append): `tests/test_graph_gnn.py`

- [ ] **Step 1: Append the failing tests** to the END of `tests/test_graph_gnn.py`:

```python


def test_edge_gatv2_forward_shape_and_attention():
    import torch
    from dapidl.graph.gnn import EdgeGATv2Aggregator
    agg = EdgeGATv2Aggregator(node_dim=16, edge_dim=8, heads=4)
    se = torch.randn(5, 16)
    ne = torch.randn(5, 3, 16)
    valid = torch.tensor([[1., 1., 1.]] * 4 + [[0., 0., 0.]])     # last node: no valid neighbours
    edge_attr = torch.randn(5, 3, 8)
    out = agg(se, ne, valid, edge_attr)
    assert out.shape == (5, 16)                                   # heads*head_dim == node_dim
    assert torch.isfinite(out).all()                             # all-invalid row -> finite (zero)
    assert torch.allclose(out[4], torch.zeros(16))               # no valid neighbours -> zero agg
    assert agg.needs_neighbours and agg.needs_edge_attr


def test_edge_gatv2_edge_attr_changes_output():
    import torch
    torch.manual_seed(0)
    from dapidl.graph.gnn import EdgeGATv2Aggregator
    agg = EdgeGATv2Aggregator(node_dim=16, edge_dim=8, heads=2)
    se = torch.randn(2, 16); ne = torch.randn(2, 3, 16); valid = torch.ones(2, 3)
    o1 = agg(se, ne, valid, torch.zeros(2, 3, 8))
    o2 = agg(se, ne, valid, torch.ones(2, 3, 8))
    assert not torch.allclose(o1, o2)                            # edge features influence attention
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_graph_gnn.py -q -k gatv2` → FAIL (`ImportError`).

- [ ] **Step 3: Append the implementation** to the END of `src/dapidl/graph/gnn.py` (`torch`, `nn` already imported):

```python


class EdgeGATv2Aggregator(nn.Module):
    """GATv2 (Brody et al. 2022) attention over the k neighbour slots, with edge features
    in the score. Multi-head; heads*head_dim == node_dim so the output matches Mean/NoGraph
    width. needs_edge_attr -> GraphArmModel passes edge_attr (B,k,edge_dim)."""
    needs_neighbours = True
    needs_edge_attr = True

    def __init__(self, node_dim: int, edge_dim: int, heads: int = 4):
        super().__init__()
        assert node_dim % heads == 0, "node_dim must be divisible by heads"
        self.heads = heads
        self.hd = node_dim // heads
        self.w = nn.Linear(2 * node_dim + edge_dim, heads * self.hd)   # GATv2 shared transform
        self.a = nn.Parameter(torch.empty(heads, self.hd))             # per-head attention vector
        self.wv = nn.Linear(node_dim, heads * self.hd)                 # value projection
        nn.init.xavier_uniform_(self.a)

    def forward(self, se, ne, valid, edge_attr):
        b, k, _ = ne.shape
        se_exp = se[:, None, :].expand(-1, k, -1)                      # [B,k,nd]
        h = torch.cat([se_exp, ne, edge_attr], dim=2)                  # [B,k,2nd+ed]
        h = torch.nn.functional.leaky_relu(self.w(h)).view(b, k, self.heads, self.hd)
        score = (h * self.a).sum(-1)                                   # [B,k,heads]
        mask = valid[:, :, None] == 0
        score = score.masked_fill(mask, -1e9)
        alpha = torch.softmax(score, dim=1) * valid[:, :, None]        # zero invalid; all-invalid -> 0
        v = self.wv(ne).view(b, k, self.heads, self.hd)               # [B,k,heads,hd]
        out = (alpha[..., None] * v).sum(1)                           # [B,heads,hd]
        return out.reshape(b, self.heads * self.hd)                   # [B, node_dim]
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_graph_gnn.py -q` → PASS (existing + 2). `uv run ruff check src/dapidl/graph/gnn.py`.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/gnn.py tests/test_graph_gnn.py
git commit -m "feat(graph): EdgeGATv2Aggregator — edge-conditioned GATv2 attention"
```

---

### Task 4: `harness.py` — edge_attr threading + aggregator factories + `compare_pairs`

**Files:**
- Modify: `src/dapidl/graph/harness.py`
- Modify: `tests/test_graph_harness.py` (update the synthetic `run_ablation` call to factories; add edge_attr + compare_pairs tests)
- Modify: `scripts/spatial_gnn_probe.py:469` and `:571` (the two existing aggregator dicts → factories, so the repo stays consistent)

This is an interface change: aggregator instances → factories. After it, the whole repo must stay green.

- [ ] **Step 1: Write/adjust the failing tests** in `tests/test_graph_harness.py`. Replace the body of `test_run_ablation_two_arms_on_synthetic_separable_graph`'s `run_ablation(...)` call to pass factories (classes are zero-arg factories), and append two new tests:

Change the existing call (inside that test) from:
```python
    res = run_ablation(lambda: FrozenFeatureEncoder(enc_feats, "cpu"),
                       {"nograph": NoGraphAggregator(), "graph": MeanAggregator()},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=15, patience=5, lr=1e-3)
```
to (factories — drop the `()`):
```python
    res = run_ablation(lambda: FrozenFeatureEncoder(enc_feats, "cpu"),
                       {"nograph": NoGraphAggregator, "graph": MeanAggregator},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=15, patience=5, lr=1e-3)
```

Append:
```python


def test_run_ablation_compare_pairs_and_edge_attr():
    import numpy as np, torch
    from dapidl.graph.encoders import FrozenFeatureEncoder
    from dapidl.graph.gnn import EdgeGATv2Aggregator, MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    from dapidl.graph.edge_geometry import build_edge_attr
    rng = np.random.RandomState(0)
    n = 200
    src = np.array(["a"] * 100 + ["b"] * 100)
    feats = rng.randn(n, 16).astype(np.float32)
    labels = (feats[:, 0] > 0).astype(np.int64)
    coords = rng.rand(n, 2) * 30
    nbr = build_within_slide_nbr_table(coords, src, k=4)
    node_geom = np.column_stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(0, 1, n),
                                 rng.uniform(0, 4, n)]).astype(np.float32)
    edge_attr = build_edge_attr(coords, node_geom, nbr)

    class _Split:
        def folds(self):
            yield "b", np.arange(0, 80), np.arange(80, 100), np.arange(100, 200)

    res = run_ablation(lambda: FrozenFeatureEncoder(feats, "cpu"),
                       {"nograph": NoGraphAggregator, "mean": MeanAggregator,
                        "gatv2": lambda: EdgeGATv2Aggregator(node_dim=16, edge_dim=8, heads=4)},
                       _Split(), nbr=nbr, labels=labels, device="cpu",
                       num_classes=2, epochs=8, patience=4, lr=1e-3,
                       edge_attr=edge_attr,
                       compare_pairs=[("nograph", "mean"), ("mean", "gatv2")])
    f = res["folds"]["b"]
    assert {"nograph", "mean", "gatv2"} <= set(f)
    assert "delta_mean_vs_nograph" in f and "delta_gatv2_vs_mean" in f
    assert "mcnemar_mean_vs_nograph" in f and "mcnemar_gatv2_vs_mean" in f
    assert "delta_gatv2_vs_mean" in res["pooled"]
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_graph_harness.py -q` → FAIL (factory call + compare_pairs not supported; `edge_attr` kwarg rejected).

- [ ] **Step 3: Implement the harness changes.**

(3a) `GraphArmModel.forward` — add `edge_attr=None` and pass it when the aggregator needs it. Replace the method:
```python
    def forward(self, self_rows, nbr_rows, valid, edge_attr=None):
        se = self.encoder.encode(self_rows)                              # [B, d]
        if getattr(self.aggregator, "needs_neighbours", True):
            b, k = nbr_rows.shape
            ne = self.encoder.encode(nbr_rows.reshape(-1)).reshape(b, k, -1)
        else:
            ne = se[:, None, :]                                          # placeholder, unused
        if getattr(self.aggregator, "needs_edge_attr", False):
            agg = self.aggregator(se, ne, valid, edge_attr)
        else:
            agg = self.aggregator(se, ne, valid)
        return self.head(torch.relu(self.lin(torch.cat([se, agg], 1))))
```

(3b) `train_arm` — aggregator becomes a factory; add `edge_attr=None`; gather edge_attr per batch. Change the signature line and the two affected lines:
```python
def train_arm(encoder_factory: Callable[[], nn.Module], aggregator_factory: Callable[[], nn.Module], *,
              nbr, labels, train_idx, val_idx, test_idx, num_classes: int = 4, device: str = "cpu",
              epochs: int = 40, patience: int = 5, seed: int = 0, batch: int = 256,
              lr: float = 3e-4, edge_attr=None) -> ArmResult:
```
Inside, replace `encoder = encoder_factory()` ... `model = GraphArmModel(encoder, aggregator, ...)` with:
```python
    torch.manual_seed(seed)
    encoder = encoder_factory()
    aggregator = aggregator_factory()
    model = GraphArmModel(encoder, aggregator, node_dim=encoder.out_dim, num_classes=num_classes).to(device)
```
And replace the `step` function's `logits = model(rows, safe, valid)` with edge-attr gathering:
```python
    def step(rows, train):
        nb = nbr[rows]
        valid = torch.from_numpy((nb >= 0).astype(np.float32)).to(device)
        safe = np.where(nb >= 0, nb, rows[:, None])
        ea = torch.from_numpy(edge_attr[rows]).float().to(device) if edge_attr is not None else None
        logits = model(rows, safe, valid, ea)
        if train:
            y = torch.from_numpy(labels[rows]).long().to(device)
            loss = lossf(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return logits.argmax(1).detach().cpu().numpy()
```

(3c) `run_ablation` — factories, `edge_attr`, `compare_pairs`. Replace the whole function:
```python
def run_ablation(encoder_factory, aggregator_factories: dict, splitter, *, nbr, labels,
                 num_classes: int = 4, device: str = "cpu", compare_pairs=None, **train_kw) -> dict:
    """Loop folds x arms (each arm an aggregator FACTORY -> fresh module per fold). For each
    (baseline, candidate) in compare_pairs (default [("nograph","graph")]) record per-fold and
    pooled delta_macro + McNemar. train_kw (incl. edge_attr=...) is forwarded to train_arm."""
    from dapidl.graph.probe_eval import mcnemar_test
    pairs = compare_pairs if compare_pairs is not None else [("nograph", "graph")]

    out: dict = {"folds": {}, "pooled": {}}
    pooled_pred = {tag: [] for tag in aggregator_factories}
    pooled_truth: list[np.ndarray] = []
    for name, tr, va, te in splitter.folds():
        fold: dict = {}
        preds: dict = {}
        for tag, fac in aggregator_factories.items():
            res = train_arm(encoder_factory, fac, nbr=nbr, labels=labels,
                            train_idx=tr, val_idx=va, test_idx=te,
                            num_classes=num_classes, device=device, **train_kw)
            fold[tag] = {"macro_f1": res.macro_f1, "per_class": res.per_class,
                         "val_macro_f1": res.val_macro_f1}
            preds[tag] = res.pred
            pooled_pred[tag].append(res.pred)
        truth = labels[te]
        pooled_truth.append(truth)
        for base, cand in pairs:
            if base in preds and cand in preds:
                fold[f"delta_{cand}_vs_{base}"] = round(fold[cand]["macro_f1"] - fold[base]["macro_f1"], 4)
                fold[f"mcnemar_{cand}_vs_{base}"] = mcnemar_test(truth, preds[base], preds[cand])
                if (base, cand) == ("nograph", "graph"):            # Stage-3 backward-compat alias
                    fold["delta_macro"] = fold[f"delta_{cand}_vs_{base}"]
        out["folds"][name] = fold

    if pooled_truth:
        T = np.concatenate(pooled_truth)
        for tag in aggregator_factories:
            P = np.concatenate(pooled_pred[tag])
            _, _, f1c, _ = precision_recall_fscore_support(T, P, labels=list(range(num_classes)), zero_division=0)
            out["pooled"][f"macro_{tag}"] = float(f1_score(T, P, average="macro", zero_division=0))
            out["pooled"][f"per_class_{tag}"] = {CLASSES[c]: float(f1c[c]) for c in range(num_classes)}
        for base, cand in pairs:
            if base in aggregator_factories and cand in aggregator_factories:
                B = np.concatenate(pooled_pred[base]); C = np.concatenate(pooled_pred[cand])
                out["pooled"][f"delta_{cand}_vs_{base}"] = round(
                    out["pooled"][f"macro_{cand}"] - out["pooled"][f"macro_{base}"], 4)
                out["pooled"][f"mcnemar_{cand}_vs_{base}"] = mcnemar_test(T, B, C)
                if (base, cand) == ("nograph", "graph"):            # Stage-3 backward-compat alias
                    out["pooled"]["delta_macro"] = out["pooled"][f"delta_{cand}_vs_{base}"]
    return out
```

(3d) Update the two Stage-3 call sites so the repo stays consistent. In `scripts/spatial_gnn_probe.py`, change both occurrences of `{"nograph": NoGraphAggregator(), "graph": MeanAggregator()}` (lines ~469 and ~571) to `{"nograph": NoGraphAggregator, "graph": MeanAggregator}` (drop the `()`).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_graph_harness.py -q` → PASS (existing updated + 1 new + characterization still skipped/green). Full graph suite: `uv run pytest tests/test_graph_*.py -q`. `uv run ruff check src/dapidl/graph/harness.py`. Confirm the script still imports: `uv run python scripts/spatial_gnn_probe.py --help | head -1`.

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/graph/harness.py tests/test_graph_harness.py scripts/spatial_gnn_probe.py
git commit -m "feat(graph): harness edge_attr threading + aggregator factories + compare_pairs"
```

---

### Task 5: `[CONTROLLER RUN]` re-run the characterization gate after the factory refactor

The factory change touched `run_ablation`/`train_arm`. Prove it still reproduces Stage-2-proper.

- [ ] **Step 1:** `nvidia-smi --query-gpu=memory.free --format=csv` (need ~2 GB free).
- [ ] **Step 2:** `uv run python scripts/spatial_gnn_probe.py --phase stage2_proper_harness` — regenerates `stage2_proper_harness_metrics.json` with the factory code.
- [ ] **Step 3:** `uv run pytest tests/test_graph_harness.py::test_characterization_reproduces_committed_stage2_proper -q` → PASS (nograph ≈ 0.537, graph ≈ 0.628, within ±0.02). **If it fails, STOP and debug the refactor — do not loosen the tolerance.**
- [ ] **Step 4:** No code change; nothing to commit (verification only).

---

### Task 6: `[CONTROLLER RUN]` `phase_node_geometry` — per-cell nuclear orientation

**Files:** Modify `scripts/spatial_gnn_probe.py` (add `phase_node_geometry`, register in `main()`).

Produces `node_geom.npy (N,3)=[angle, ecc, log_area]`, row-aligned to the registry. Xenium via StarDist (~286k patches, ~2–4 h GPU); STHELAR via polygons (CPU). Estimate RAM: STHELAR GeoDataFrame loaded per-slide; node_geom ~27 MB.

- [ ] **Step 1: Add `phase_node_geometry`** (insert before `def main()`):

```python
def phase_node_geometry() -> None:
    """[GPU+CPU] Per-cell nuclear (angle, eccentricity, log_area) -> node_geom.npy (N,3).
    Xenium: StarDist the DAPI patch, central nucleus nearest the patch centre. STHELAR:
    native nucleus polygons. Cells with no nucleus -> [nan, 0, median_log_area]."""
    import struct
    import sys

    import lmdb
    import numpy as np
    import polars as pl
    from skimage.measure import regionprops

    from dapidl.graph.embed import decode_record
    from dapidl.graph.geometry import ellipse_from_points
    sys.path.insert(0, "scripts")

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    cell_id = reg["cell_id"].to_numpy()
    n = len(reg)
    node_geom = np.full((n, 3), np.nan, dtype=np.float32)
    node_geom[:, 1] = 0.0                                      # ecc default 0

    # --- Xenium rows via StarDist on the 128px patch ---
    xen_rows = np.where(np.char.startswith(src.astype(str), "xenium"))[0]
    if len(xen_rows):
        from starpose.qc import SegQCConfig, SegmentationGroundedScorer
        scorer = SegmentationGroundedScorer(SegQCConfig(erode_px=1), gpu=True, pixel_size=0.2125)
        env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False)
        ctr = np.array([64.0, 64.0])
        miss = 0
        with env.begin() as txn:
            for ri in xen_rows:
                _, patch = decode_record(txn.get(struct.pack(">Q", int(ri))), 128)
                masks, _ = scorer._segment(patch)
                if masks.max() == 0:
                    miss += 1
                    continue
                props = regionprops(masks)
                best = min(props, key=lambda p: np.hypot(p.centroid[0] - ctr[0], p.centroid[1] - ctr[1]))
                pts = np.argwhere(masks == best.label)[:, ::-1].astype(float)   # (y,x)->(x,y)
                ang, ecc = ellipse_from_points(pts)
                node_geom[ri] = (ang, ecc, float(np.log1p(best.area)))
        env.close()
        logger.info(f"node_geometry xenium: {len(xen_rows)} cells, {miss} no-nucleus")

    # --- STHELAR rows via native nucleus polygons (per slide) ---
    from dapidl.data.sthelar import load_nucleus_geometry_with_labels
    STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")
    for s in [v for v in np.unique(src) if str(v).startswith("sthelar")]:
        name = str(s).replace("sthelar_", "")                 # e.g. breast_s0
        outer = STHELAR_BASE / f"sdata_{name}.zarr"
        slide_root = outer / outer.name if (outer / outer.name / "shapes").is_dir() else outer
        gdf = load_nucleus_geometry_with_labels(slide_root, [])
        geom = gdf["geometry"]
        rows = np.where(src == s)[0]
        miss = 0
        for ri in rows:
            cid = str(cell_id[ri])
            if cid not in geom.index:
                miss += 1
                continue
            poly = geom.loc[cid]
            pts = np.asarray(poly.exterior.coords, dtype=float)
            ang, ecc = ellipse_from_points(pts)
            node_geom[ri] = (ang, ecc, float(np.log1p(poly.area)))
        logger.info(f"node_geometry {s}: {len(rows)} cells, {miss} unmatched")

    # fill log_area NaNs with the median (keep angle NaN -> directional terms zeroed downstream)
    la = node_geom[:, 2]
    med = float(np.nanmedian(la))
    la[np.isnan(la)] = med
    node_geom[:, 2] = la
    np.save(OUT / "node_geom.npy", node_geom)
    logger.info(f"node_geom {node_geom.shape} -> node_geom.npy "
                f"({int(np.isnan(node_geom[:, 0]).sum())} cells without orientation)")
```

- [ ] **Step 2: Register** — add `"node_geometry"` to the `choices` list and `"node_geometry": phase_node_geometry` to the `phases` dict in `main()`.

- [ ] **Step 3: Run** (long — ~2–4 h GPU for the Xenium StarDist pass):
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
uv run python scripts/spatial_gnn_probe.py --phase node_geometry
```
Verify: `python -c "import numpy as np; g=np.load('pipeline_output/spatial_gnn_probe_2026_06/node_geom.npy'); print(g.shape, 'nan_angle=', np.isnan(g[:,0]).sum(), 'ecc range', g[:,1].min(), g[:,1].max())"` — shape (N,3), ecc in [0,1), a minority of NaN angles.

- [ ] **Step 4: Commit** (code only):
```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_node_geometry — nuclear orientation (Xenium StarDist + STHELAR polygons)"
```

---

### Task 7: `[CONTROLLER RUN]` `phase_stage4_gatv2` — 3-arm LOSO

**Files:** Modify `scripts/spatial_gnn_probe.py` (add `phase_stage4_gatv2`, register).

- [ ] **Step 1: Add `phase_stage4_gatv2`**:

```python
def phase_stage4_gatv2() -> None:
    """[CONTROLLER RUN] 3-arm LOSO (nograph / mean / gatv2). nograph+mean ignore edge_attr
    (apples-to-apples with Stage-3 E1); gatv2 uses rotation-invariant edge geometry."""
    import numpy as np
    import polars as pl
    import torch

    from dapidl.graph.edge_geometry import build_edge_attr
    from dapidl.graph.encoders import FrozenFeatureEncoder
    from dapidl.graph.gnn import EdgeGATv2Aggregator, MeanAggregator, NoGraphAggregator
    from dapidl.graph.harness import run_ablation
    from dapidl.graph.knn_graph import build_within_slide_nbr_table
    from dapidl.graph.splits import LOSOSplit

    reg = pl.read_parquet(OUT / "spatial_registry.parquet")
    src = reg["source"].to_numpy()
    coords = reg.select(["x_px", "y_px"]).to_numpy()
    labels = reg["coarse_idx"].to_numpy()
    pca = np.load(OUT / "embeddings_pca128.npy")
    node_geom = np.load(OUT / "node_geom.npy")

    nbr = build_within_slide_nbr_table(coords, src, k=8)
    edge_attr = build_edge_attr(coords, node_geom, nbr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats_dev = torch.from_numpy(np.ascontiguousarray(pca, dtype=np.float32)).to(device)

    res = run_ablation(
        lambda: FrozenFeatureEncoder(feats_dev, device),
        {"nograph": NoGraphAggregator, "mean": MeanAggregator,
         "gatv2": lambda: EdgeGATv2Aggregator(node_dim=128, edge_dim=edge_attr.shape[2], heads=4)},
        LOSOSplit(src, coords, labels, val_frac=0.20),
        nbr=nbr, labels=labels, edge_attr=edge_attr, device=device,
        compare_pairs=[("nograph", "mean"), ("mean", "gatv2"), ("nograph", "gatv2")])
    res["baseline_effnet_macro_f1"] = 0.619
    (OUT / "stage4_gatv2_metrics.json").write_text(json.dumps(res, indent=2))
    p = res.get("pooled", {})
    logger.info(f"stage4 pooled: nograph={p.get('macro_nograph')} mean={p.get('macro_mean')} "
                f"gatv2={p.get('macro_gatv2')} d(gatv2-mean)={p.get('delta_gatv2_vs_mean')}")
```

- [ ] **Step 2: Register** `"stage4_gatv2"` in `choices` + `phases` dict.

- [ ] **Step 3: Run** (~1–2 h: 18 GNN runs on cached features):
```bash
nvidia-smi --query-gpu=memory.free --format=csv
uv run python scripts/spatial_gnn_probe.py --phase stage4_gatv2
```
Verify `stage4_gatv2_metrics.json` has 6 folds each with `nograph/mean/gatv2` + `delta_gatv2_vs_mean` + `mcnemar_gatv2_vs_mean`, and a `pooled` block. Sanity: `folds["xenium_rep2"]["delta_gatv2_vs_mean"]` is the headline.

- [ ] **Step 4: Commit** (code only):
```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_stage4_gatv2 — 3-arm LOSO (nograph/mean/gatv2) with edge geometry"
```

---

### Task 8: `[CONTROLLER RUN]` `phase_stage4_readout`

**Files:** Modify `scripts/spatial_gnn_probe.py` (add `phase_stage4_readout`, register).

- [ ] **Step 1: Add `phase_stage4_readout`**:

```python
def phase_stage4_readout() -> None:
    """[CONTROLLER RUN] stage4_readout.md: per-fold + pooled nograph/mean/gatv2, the
    gatv2-vs-mean delta (the isolation), feature-clean tiering, and the verdict vs E1/E2."""
    import numpy as np

    d = json.loads((OUT / "stage4_gatv2_metrics.json").read_text())
    FEATURE_CLEAN = {"xenium_rep2"}   # EffNet trained on the other 5 slides (h2h summary)

    lines = ["# Graph-Arm Stage 4 — Edge-Geometry GATv2 Readout (LOSO)\n",
             "Node features = frozen-EffNet PCA-128 for all arms; only `gatv2` sees edge geometry.\n",
             "| Held-out slide | tier | nograph | mean | gatv2 | d(gatv2-mean) | McNemar p |",
             "|---|---|---|---|---|---|---|"]
    for name, f in d["folds"].items():
        tier = "feature-clean" if name in FEATURE_CLEAN else "delta-clean"
        mp = f.get("mcnemar_gatv2_vs_mean", {}).get("p_value")
        lines.append(f"| {name} | {tier} | {f['nograph']['macro_f1']:.4f} | {f['mean']['macro_f1']:.4f} "
                     f"| {f['gatv2']['macro_f1']:.4f} | {f.get('delta_gatv2_vs_mean'):+.4f} | {mp} |")
    p = d.get("pooled", {})
    r2 = d["folds"]["xenium_rep2"]
    r2_endo = r2["gatv2"]["per_class"]["Endothelial"]["f1"] - r2["mean"]["per_class"]["Endothelial"]["f1"]
    lines += [
        f"\n**Pooled:** nograph {p.get('macro_nograph')}, mean {p.get('macro_mean')}, "
        f"gatv2 {p.get('macro_gatv2')}; d(gatv2-mean) {p.get('delta_gatv2_vs_mean')}, "
        f"pooled McNemar p={p.get('mcnemar_gatv2_vs_mean', {}).get('p_value')}.\n",
        "## Verdict\n",
        f"- **Feature-clean rep2:** gatv2-vs-mean = {r2.get('delta_gatv2_vs_mean'):+.4f} macro "
        f"(Endothelial {r2_endo:+.4f}). Stage-3 bar to clear: mean graph +0.0161 and free C&S +0.0159.\n",
        "- If gatv2-vs-mean materially exceeds 0 on rep2 (esp. Endothelial), edge-geometry attention "
        "is the real lever -> multi-scale follow-on justified. If ~0, the graph caps at diffusion on "
        "these features and we stop.\n",
        "\n> Only xenium_rep2 is feature-clean (EffNet trained on rep1 + sthelar s0/s1/s3/s6). "
        "The gatv2-vs-mean delta is leakage-immune in every fold (same frozen nodes).\n"]
    (OUT / "stage4_readout.md").write_text("\n".join(lines))
    logger.info("stage4 readout -> stage4_readout.md")
```

- [ ] **Step 2: Register** `"stage4_readout"` in `choices` + `phases` dict.

- [ ] **Step 3: Run**:
```bash
uv run python scripts/spatial_gnn_probe.py --phase stage4_readout
```
Verify `pipeline_output/spatial_gnn_probe_2026_06/stage4_readout.md` renders the 3-arm table + verdict.

- [ ] **Step 4: Commit** (code only):
```bash
git add scripts/spatial_gnn_probe.py
git commit -m "feat(graph): phase_stage4_readout — gatv2-vs-mean verdict vs the Stage-3 bar"
```

---

## Final verification (controller)

- [ ] Full unit suite green: `uv run pytest tests/test_graph_geometry.py tests/test_graph_edge_geometry.py tests/test_graph_gnn.py tests/test_graph_harness.py tests/test_graph_splits.py tests/test_graph_smooth.py tests/test_graph_knn.py tests/test_graph_encoders.py tests/test_graph_probe_eval.py -q`.
- [ ] Lint: `uv run ruff check src/dapidl/graph/`.
- [ ] Characterization gate passed after the factory refactor (Task 5).
- [ ] `stage4_readout.md` exists and the gatv2-vs-mean verdict reads sensibly.
- [ ] Update memory with the Stage-4 result.
- [ ] Final code review across the Stage-4 diff (the driver phases weren't subagent-reviewed).

## Execution order

Tasks 1–4 (pure/unit) in order, each green before the next; Task 4 leaves the whole repo consistent (factories). Then controller Tasks 5 (verify refactor) → 6 (node geometry, the ~2–4 h StarDist pass) → 7 (3-arm LOSO) → 8 (readout). Check `nvidia-smi` (2–4 GB free) before GPU phases (5, 6).
