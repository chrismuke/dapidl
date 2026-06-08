# Subnuclear-Structure Triangulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how much *inner subnuclear DAPI structure* contributes to the production EfficientNet's cell-type calls, via two no-retrain readouts — an information-floor (LightGBM on per-nucleus features vs the model's 0.619) and an attribution-concentration (Integrated Gradients inside vs outside the nucleus) — over the existing `breast-6source-dapi-p128` LMDB.

**Architecture:** One shared StarDist pass over the LMDB produces, per patch, a center-nucleus mask + StarDist prob. Two pure feature/attribution libraries (`src/dapidl/qc/patch_features.py`, `src/dapidl/qc/attribution.py`, both TDD, GPU-free except IG) feed three orchestration scripts (feature pass → floor → saliency) and a driver. Streaming, chunked parquet writes keep RAM well under the 62 GB host. Subsampling + a 500-patch smoke-ETA gate de-risk StarDist throughput (recorded ~3 patch/s CPU; runs on GPU here).

**Tech Stack:** Python 3.12, `uv`, polars (pandas only at the LightGBM/skimage boundary), numpy/scipy/skimage, PyTorch (existing `DapiClassifier`), `starpose.qc.segmentation_grounded` (StarDist + pure scorers, **used as a library — no starpose edits**), LightGBM (new dep), custom Integrated Gradients (no captum).

---

## Ground truth captured from source (do not re-derive)

**LMDB** `/mnt/work/datasets/derived/breast-6source-dapi-p128/`:
- `patches.lmdb`: key `struct.pack(">Q", global_idx)`; value first 8 bytes are a stale label → **skip**; rest is `np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)`.
- `labels.npy` (coarse, `COARSE_NAMES = ["Endothelial","Epithelial","Immune","Stromal"]`, label `-1` = drop), `sources.npy` (`allow_pickle=True`).
- Counts: 2,277,877 total; `-1` = 333,362. Sources: `xenium_rep1`=157,231, `sthelar_breast_s0`=542,818, `sthelar_breast_s1`=757,374, `sthelar_breast_s3`=345,805, `sthelar_breast_s6`=360,923, **`xenium_rep2`=113,726** (the held-out test source).
- Train sources (match the 0.619 model): `xenium_rep1, sthelar_breast_s0, sthelar_breast_s1, sthelar_breast_s3, sthelar_breast_s6`. Test: `xenium_rep2`.

**Checkpoint**: `pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt` — a plain `state_dict`. Load with:
```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # for new scripts in scripts/
from breast_pooled_train import DapiClassifier  # import-safe: __main__ guard at L585
model = DapiClassifier(4, backbone="efficientnetv2_rw_s")
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
```

**Eval transform (reproduce EXACTLY for IG)** — from `DapiPatchDataset.__getitem__`:
```python
img = patch_u16.astype(np.float32) / 65535.0
img = (img - 0.485) / 0.229                      # DAPI_NORM_MEAN / DAPI_NORM_STD for efficientnetv2_rw_s
x = torch.from_numpy(img)[None, None]            # (1,1,128,128); model expands 1->3 ch internally
```
**Feature/segmentation transform**: pass the **raw uint16 patch** (cast to float as needed) — the starpose scorers + texture fns percentile-normalize internally; do NOT /65535 here.

**starpose API** (`from starpose.qc.segmentation_grounded import ...`, the real home behind the deprecated `dapidl.qc.segmentation_grounded` shim):
- `SegQCConfig()` — dataclass; `erode_px=3`, `min_interior_px=20`, `prob_min=0.40`, `pixel_size` lives on the scorer not the cfg.
- `select_center_nucleus(masks, probs, cfg) -> CenterNucleus | None`; `CenterNucleus` has `.mask` (bool HxW), `.prob`, `.centroid` (y,x), `.area_px`, `.label`.
- `structure_raw(patch, mask, cfg) -> float` (erodes internally; flat-interior guard → 0.0).
- `area_um2(mask, pixel_size) -> float`; `objectness_metrics(patch, mask, prob, cfg) -> dict` (has `eccentricity`,`solidity`,`intensity_ratio`); `interior_cov(patch, interior_mask) -> float`; `brenner_focus(patch, interior_mask) -> float`; `glcm_texture(patch, interior_mask, levels=16) -> {glcm_entropy, glcm_asm}`.
- `SegmentationGroundedScorer(cfg=None, gpu=True, pixel_size=0.2125)._segment(patch) -> (masks int32, probs float[])`, label `k` → `probs[k-1]`.

**Pixel size**: use `0.2125` (Xenium; scorer default) for `area_um2` across all sources — area is a feature, not a label; consistent scaling matters more than per-source µm/px accuracy. Note this caveat in the floor report.

---

## File structure

| File | Responsibility |
|---|---|
| `src/dapidl/qc/patch_features.py` *(new, TDD)* | Pure per-nucleus + per-patch feature vector. `haralick_features`, `nucleus_feature_vector`, stable column constants. |
| `src/dapidl/qc/attribution.py` *(new, TDD)* | Custom Integrated Gradients + mask concentration. `integrated_gradients`, `fraction_in_mask`, `attribution_concentration`. |
| `scripts/subnuclear_common.py` *(new, TDD)* | Pure orchestration helpers: index selection / balanced subset / column filters. |
| `scripts/subnuclear_feature_pass.py` *(new)* | Stream LMDB → StarDist → feature rows → chunked parquet + optional packed-bit mask cache. Smoke-ETA gate. |
| `scripts/subnuclear_floor.py` *(new)* | LightGBM floor (nuc-only vs nuc+ctx) vs 0.619; per-class F1 + importances. |
| `scripts/subnuclear_saliency.py` *(new)* | IG concentration on a balanced rep2 subset + overlay PNGs. |
| `pipeline_output/subnuclear_2026_06/run_triangulation.sh` *(new)* | Driver: smoke → ETA → feature pass → floor → saliency → readout table. |
| `tests/test_patch_features.py`, `tests/test_attribution.py`, `tests/test_subnuclear_common.py` *(new)* | Unit tests for the three importable modules. |

Outputs land in `pipeline_output/subnuclear_2026_06/` (git-ignored): `seg_features.parquet`, `center_masks/chunk_*.npz`, `floor_metrics.json`, `saliency_summary.json`, `overlays/*.png`.

---

### Task 0: Dependency + module scaffolding

**Files:**
- Modify: `pyproject.toml` / `uv.lock` (via `uv add`)
- Create: `src/dapidl/qc/patch_features.py`, `src/dapidl/qc/attribution.py`, `scripts/subnuclear_common.py` (docstring-only stubs)

- [ ] **Step 1: Add LightGBM**

Run: `uv add lightgbm`

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "import lightgbm; print(lightgbm.__version__)"`
Expected: a version string (e.g. `4.x.x`), no traceback.

- [ ] **Step 3: Create three stub modules (docstring only, so tests fail on the missing *function*, not a missing module)**

`src/dapidl/qc/patch_features.py`:
```python
"""Pure per-nucleus + per-patch DAPI feature vectors for the subnuclear-structure
triangulation (no I/O, no GPU). Reuses starpose.qc scorers; adds Haralick texture."""
```
`src/dapidl/qc/attribution.py`:
```python
"""Custom Integrated Gradients + nucleus-mask attribution concentration (no captum)."""
```
`scripts/subnuclear_common.py`:
```python
"""Pure orchestration helpers for the subnuclear-triangulation scripts (LMDB index
selection, balanced subsetting, feature-column filters). No I/O, no GPU."""
```

- [ ] **Step 4: Confirm the package imports cleanly**

Run: `uv run python -c "import dapidl.qc.patch_features, dapidl.qc.attribution; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/dapidl/qc/patch_features.py src/dapidl/qc/attribution.py scripts/subnuclear_common.py
git commit -m "chore(subnuclear): add lightgbm + module scaffolding"
```

---

### Task 1: `haralick_features` — mask-aware GLCM texture

**Files:**
- Create/Modify: `src/dapidl/qc/patch_features.py`
- Test: `tests/test_patch_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_patch_features.py`:
```python
import numpy as np
from dapidl.qc.patch_features import haralick_features


def _disc(textured, size=128, r=20, seed=0):
    patch = np.full((size, size), 1000.0)
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= r ** 2
    if textured:
        rng = np.random.default_rng(seed)
        patch[mask] = rng.integers(500, 4000, size=int(mask.sum()))
    else:
        patch[mask] = 2500.0
    return patch, mask


def test_haralick_contrast_and_entropy_higher_on_textured():
    pt, m = _disc(True)
    pf, _ = _disc(False)
    ht = haralick_features(pt, m)
    hf = haralick_features(pf, m)
    assert ht["contrast"] > hf["contrast"]
    assert ht["entropy"] > hf["entropy"]


def test_haralick_returns_six_keys():
    pt, m = _disc(True)
    h = haralick_features(pt, m)
    assert set(h) == {"contrast", "homogeneity", "energy", "correlation", "asm", "entropy"}


def test_haralick_degenerate_small_mask_is_smooth():
    patch = np.ones((32, 32)) * 1000.0
    tiny = np.zeros((32, 32), bool)
    tiny[0, 0] = True
    h = haralick_features(patch, tiny)
    assert h["entropy"] == 0.0 and h["asm"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_patch_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'haralick_features'`.

- [ ] **Step 3: Implement `haralick_features`**

Append to `src/dapidl/qc/patch_features.py`:
```python
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

_SMOOTH = {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0,
           "correlation": 1.0, "asm": 1.0, "entropy": 0.0}


def haralick_features(patch: np.ndarray, interior_mask: np.ndarray,
                      levels: int = 16) -> dict[str, float]:
    """Grey-level co-occurrence texture inside ``interior_mask``.

    Quantizes over the patch's [1, 99] percentile range (brightness-robust),
    maps masked-out pixels to level 0 and drops that row/col of the GLCM — the
    starpose ``glcm_texture`` technique — then averages props over {0, 90 deg}.
    Returns contrast, homogeneity, energy, correlation, ASM, entropy. A
    degenerate region (<8 px or flat dynamic range) -> texture-less defaults.
    """
    interior_mask = np.asarray(interior_mask, dtype=bool)
    if int(interior_mask.sum()) < 8:
        return dict(_SMOOTH)
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    if hi <= lo:
        return dict(_SMOOTH)
    q = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    qg = (q * (levels - 1)).astype(np.uint8) + 1          # interior -> levels 1..levels
    qg[~interior_mask] = 0                                 # masked-out -> level 0
    glcm = graycomatrix(qg, distances=[1], angles=[0.0, np.pi / 2.0],
                        levels=levels + 1, symmetric=True)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)           # drop the masked level-0 row/col
    tot = glcm.sum(axis=(0, 1), keepdims=True)
    tot[tot == 0] = 1.0
    pr = glcm / tot
    return {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(np.nan_to_num(graycoprops(glcm, "correlation")).mean()),
        "asm": float(graycoprops(glcm, "ASM").mean()),
        "entropy": float((-pr * np.log2(pr + 1e-12)).sum(axis=(0, 1)).mean()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_patch_features.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/patch_features.py tests/test_patch_features.py
git commit -m "feat(subnuclear): mask-aware Haralick texture (haralick_features)"
```

---

### Task 2: `nucleus_feature_vector` — stable two-scope feature row

**Files:**
- Modify: `src/dapidl/qc/patch_features.py`
- Test: `tests/test_patch_features.py`

**Design decision (documented):** `nuc_*` features come from the center-nucleus mask (geometry from the full mask; texture/focus on the 3 px-eroded interior, matching starpose). `ctx_*` features cover the whole 128² patch and **omit geometry** — a fixed square has constant eccentricity/solidity/area, i.e. zero-information dead columns. `ctx_*` therefore keeps only intensity + texture/focus, where context signal actually lives. Every call returns the **same key set** (NaN where undefined) so the chunked parquet schema is stable; `ctx_*` is computed even when there is no nucleus (it needs none).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patch_features.py`:
```python
from starpose.qc.segmentation_grounded import SegQCConfig
from dapidl.qc.patch_features import nucleus_feature_vector, NUC_COLUMNS, CTX_COLUMNS


def test_feature_vector_textured_has_more_structure_than_flat():
    pt, m = _disc(True)
    pf, _ = _disc(False)
    ft = nucleus_feature_vector(pt, m, 0.9, SegQCConfig(), 0.2125)
    ff = nucleus_feature_vector(pf, m, 0.9, SegQCConfig(), 0.2125)
    assert ft["has_nucleus"] == 1.0
    assert ft["nuc_structure_raw"] > ff["nuc_structure_raw"]
    assert ft["nuc_contrast"] > ff["nuc_contrast"]


def test_feature_vector_area_fraction_exact():
    patch = np.ones((128, 128)) * 1000.0
    m = np.zeros((128, 128), bool)
    m[50:60, 50:70] = True  # 200 px
    f = nucleus_feature_vector(patch, m, 0.5, SegQCConfig(), 0.2125)
    assert abs(f["nuc_area_fraction"] - 200.0 / (128 * 128)) < 1e-9


def test_feature_vector_no_nucleus_is_nan_but_keeps_context():
    patch = np.ones((128, 128)) * 1000.0
    f = nucleus_feature_vector(patch, None, 0.0, SegQCConfig(), 0.2125)
    assert f["has_nucleus"] == 0.0
    assert np.isnan(f["nuc_area_fraction"])
    assert np.isnan(f["nuc_structure_raw"])
    assert "ctx_int_mean" in f and not np.isnan(f["ctx_int_mean"])


def test_feature_vector_stable_columns():
    pt, m = _disc(True)
    with_nuc = nucleus_feature_vector(pt, m, 0.9, SegQCConfig(), 0.2125)
    without = nucleus_feature_vector(pt, None, 0.0, SegQCConfig(), 0.2125)
    assert set(with_nuc) == set(without)
    for c in NUC_COLUMNS + CTX_COLUMNS + ["has_nucleus"]:
        assert c in with_nuc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_patch_features.py -k feature_vector -v`
Expected: FAIL — `cannot import name 'nucleus_feature_vector'`.

- [ ] **Step 3: Implement `nucleus_feature_vector` + column constants**

Append to `src/dapidl/qc/patch_features.py`:
```python
from scipy import ndimage
from skimage.measure import regionprops
from starpose.qc.segmentation_grounded import (
    SegQCConfig, area_um2, brenner_focus, interior_cov, structure_raw,
)

_GEOM = ("area_um2", "eccentricity", "solidity", "extent", "major_axis", "minor_axis")
_INT = ("int_mean", "int_std", "int_p10", "int_p50", "int_p90", "int_above_bg")
_TEX = ("structure_raw", "brenner", "interior_cov",
        "contrast", "homogeneity", "energy", "correlation", "asm", "entropy")

NUC_COLUMNS = [f"nuc_{k}" for k in (_GEOM + _INT + _TEX)] + ["nuc_stardist_prob", "nuc_area_fraction"]
CTX_COLUMNS = [f"ctx_{k}" for k in (_INT + _TEX)]


def _intensity(patch: np.ndarray, region: np.ndarray, bg: np.ndarray) -> dict[str, float]:
    v = patch[region].astype(np.float64)
    bgv = patch[bg].astype(np.float64)
    bg_med = float(np.median(bgv)) if bgv.size else 0.0
    if v.size == 0:
        return {k: np.nan for k in _INT}
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return {"int_mean": float(v.mean()), "int_std": float(v.std()),
            "int_p10": float(p10), "int_p50": float(p50), "int_p90": float(p90),
            "int_above_bg": float(v.mean() - bg_med)}


def _texture(patch: np.ndarray, region: np.ndarray, interior: np.ndarray,
             cfg: SegQCConfig) -> dict[str, float]:
    out = {"structure_raw": structure_raw(patch, region, cfg),
           "brenner": brenner_focus(patch, interior),
           "interior_cov": interior_cov(patch, interior)}
    out.update(haralick_features(patch, interior))
    return out


def _scope(patch: np.ndarray, region: np.ndarray, cfg: SegQCConfig,
           pixel_size: float, geom: bool) -> dict[str, float]:
    region = np.asarray(region, dtype=bool)
    out: dict[str, float] = {}
    if geom:
        if region.any():
            pr = regionprops(region.astype(np.int32))[0]
            out["area_um2"] = area_um2(region, pixel_size)
            out["eccentricity"] = float(pr.eccentricity)
            out["solidity"] = float(pr.solidity)
            out["extent"] = float(pr.extent)
            out["major_axis"] = float(pr.axis_major_length)
            out["minor_axis"] = float(pr.axis_minor_length)
        else:
            out.update({k: np.nan for k in _GEOM})
        interior = ndimage.binary_erosion(region, iterations=cfg.erode_px)
        bg = ~region
    else:
        interior = region
        bg = region  # ctx "background" = whole patch -> above_bg becomes mean - global median
    out.update(_intensity(patch, region, bg))
    out.update(_texture(patch, region, interior, cfg))
    return out


def nucleus_feature_vector(patch, mask, prob, cfg: SegQCConfig,
                           pixel_size: float) -> dict[str, float]:
    """Two-scope feature row. ``nuc_*`` from the center-nucleus mask (or NaN when
    ``mask is None``); ``ctx_*`` over the whole patch (always computed). Always
    returns the same key set so the parquet schema is stable."""
    patch = np.asarray(patch)
    feats: dict[str, float] = {"has_nucleus": 1.0 if mask is not None else 0.0}
    if mask is None:
        feats.update({c: np.nan for c in NUC_COLUMNS})
    else:
        mask = np.asarray(mask, dtype=bool)
        feats.update({f"nuc_{k}": v for k, v in
                      _scope(patch, mask, cfg, pixel_size, geom=True).items()})
        feats["nuc_stardist_prob"] = float(prob)
        feats["nuc_area_fraction"] = float(mask.sum()) / float(mask.size)
    whole = np.ones(patch.shape, dtype=bool)
    feats.update({f"ctx_{k}": v for k, v in
                  _scope(patch, whole, cfg, pixel_size, geom=False).items()})
    return feats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_patch_features.py -v`
Expected: PASS (all patch-feature tests).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/patch_features.py tests/test_patch_features.py
git commit -m "feat(subnuclear): two-scope nucleus_feature_vector with stable schema"
```

---

### Task 3: `integrated_gradients` — shape-agnostic IG with completeness

**Files:**
- Modify: `src/dapidl/qc/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test**

`tests/test_attribution.py`:
```python
import numpy as np
import torch

from dapidl.qc.attribution import integrated_gradients


def test_ig_completeness_on_linear_model():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 3)
    x = torch.randn(1, 10)
    baseline = torch.zeros(1, 10)
    attr = integrated_gradients(model, x, target=1, baseline=baseline, steps=64)
    delta = (model(x)[0, 1] - model(baseline)[0, 1]).item()
    assert abs(float(attr.sum()) - delta) < 1e-4
    assert attr.shape == x.shape


def test_ig_zero_when_x_equals_baseline():
    torch.manual_seed(1)
    model = torch.nn.Linear(6, 2)
    x = torch.randn(1, 6)
    attr = integrated_gradients(model, x, target=0, baseline=x.clone(), steps=16)
    assert abs(float(attr.sum())) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: FAIL — `cannot import name 'integrated_gradients'`.

- [ ] **Step 3: Implement `integrated_gradients`**

Append to `src/dapidl/qc/attribution.py`:
```python
from __future__ import annotations

import torch


def integrated_gradients(model, x: torch.Tensor, target: int,
                         baseline: torch.Tensor | None = None,
                         steps: int = 50) -> torch.Tensor:
    """Riemann Integrated Gradients of ``model(x)[:, target]`` w.r.t. ``x``.

    Shape-agnostic (works for (1,D) linear inputs and (1,1,H,W) images alike);
    returns a tensor shaped like ``x``. Completeness holds exactly for a linear
    model: ``attr.sum() == f(x) - f(baseline)``. The model may expand channels
    internally (DapiClassifier 1->3) — gradients flow back to the single input
    channel, so no channel reduction is needed.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)
    view = [steps] + [1] * (x.dim() - 1)
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype).view(*view)
    path = (baseline + alphas * (x - baseline)).detach().requires_grad_(True)
    out = model(path)
    score = out[:, target].sum()
    (grads,) = torch.autograd.grad(score, path)
    avg_grad = grads.mean(dim=0, keepdim=True)
    return ((x - baseline) * avg_grad).detach()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/attribution.py tests/test_attribution.py
git commit -m "feat(subnuclear): custom Integrated Gradients (completeness-tested)"
```

---

### Task 4: `fraction_in_mask` + `attribution_concentration`

**Files:**
- Modify: `src/dapidl/qc/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_attribution.py`:
```python
from dapidl.qc.attribution import attribution_concentration, fraction_in_mask


def test_fraction_in_mask_uses_absolute_value():
    attr = np.array([[1.0, -3.0], [0.0, 0.0]])
    mask = np.array([[True, False], [False, False]])
    assert abs(fraction_in_mask(attr, mask) - 0.25) < 1e-9  # |1| / (|1|+|-3|)


def test_fraction_in_mask_zero_attribution_is_zero():
    attr = np.zeros((2, 2))
    mask = np.ones((2, 2), bool)
    assert fraction_in_mask(attr, mask) == 0.0


def test_concentration_divides_fraction_by_area():
    attr = np.array([[2.0, 0.0], [0.0, 2.0]])
    mask = np.array([[True, False], [False, False]])  # fraction = 0.5
    assert abs(attribution_concentration(attr, mask, 0.25) - 2.0) < 1e-9


def test_concentration_eps_guard_is_finite():
    attr = np.array([[1.0, 0.0]])
    mask = np.array([[True, False]])
    assert np.isfinite(attribution_concentration(attr, mask, 0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution.py -k "fraction or concentration" -v`
Expected: FAIL — `cannot import name 'fraction_in_mask'`.

- [ ] **Step 3: Implement both functions**

Append to `src/dapidl/qc/attribution.py`:
```python
import numpy as np


def fraction_in_mask(attr, mask) -> float:
    """Share of total |attribution| that falls inside ``mask`` (∈ [0, 1])."""
    a = np.abs(np.asarray(attr, dtype=np.float64))
    m = np.asarray(mask, dtype=bool)
    total = a.sum()
    if total <= 0:
        return 0.0
    return float(a[m].sum() / total)


def attribution_concentration(attr, mask, area_fraction: float,
                              eps: float = 1e-6) -> float:
    """Headline (D) metric: ``fraction_in_mask / max(area_fraction, eps)``.

    ≈1 nucleus ignored (attribution spread evenly by area); ≫1 subnuclear-driven
    (concentrated inside the nucleus beyond its size); <1 context-driven.
    """
    return fraction_in_mask(attr, mask) / max(area_fraction, eps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: PASS (all attribution tests).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/attribution.py tests/test_attribution.py
git commit -m "feat(subnuclear): fraction_in_mask + attribution_concentration"
```

---

### Task 5: orchestration helpers (index selection / subset / columns)

**Files:**
- Modify: `scripts/subnuclear_common.py`
- Test: `tests/test_subnuclear_common.py`

- [ ] **Step 1: Write the failing test**

`tests/test_subnuclear_common.py`:
```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from subnuclear_common import (  # noqa: E402
    balanced_subset, ctx_feature_columns, nuc_feature_columns, select_pass_indices,
)


def test_select_pass_indices_stratified_cap():
    sources = np.array(["a"] * 10 + ["b"] * 5)
    labels = np.zeros(15, int)
    idx = select_pass_indices(sources, labels, ["a", "b"], max_per_source=3, seed=1)
    assert (sources[idx] == "a").sum() == 3
    assert (sources[idx] == "b").sum() == 3


def test_select_pass_indices_deterministic():
    sources = np.array(["a"] * 100)
    labels = np.zeros(100, int)
    a = select_pass_indices(sources, labels, ["a"], max_per_source=10, seed=7)
    b = select_pass_indices(sources, labels, ["a"], max_per_source=10, seed=7)
    assert np.array_equal(a, b)


def test_select_pass_indices_drop_unlabeled_and_limit():
    sources = np.array(["a"] * 6)
    labels = np.array([-1, 0, 1, 2, 3, -1])
    idx = select_pass_indices(sources, labels, ["a"], drop_unlabeled=True, limit=3)
    assert len(idx) == 3
    assert (labels[idx] != -1).all()


def test_balanced_subset_caps_per_class_drops_unlabeled():
    labels = np.array([0] * 100 + [1] * 5 + [2] * 50 + [-1] * 9)
    s = balanced_subset(labels, 10, seed=0)
    import collections
    c = collections.Counter(labels[s].tolist())
    assert c[0] == 10 and c[1] == 5 and c[2] == 10 and c[-1] == 0


def test_column_filters():
    cols = ["global_idx", "source", "label", "has_nucleus",
            "nuc_area_um2", "nuc_contrast", "ctx_int_mean", "ctx_contrast"]
    assert nuc_feature_columns(cols) == ["nuc_area_um2", "nuc_contrast"]
    assert ctx_feature_columns(cols) == ["ctx_int_mean", "ctx_contrast"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subnuclear_common.py -v`
Expected: FAIL — `cannot import name 'select_pass_indices'`.

- [ ] **Step 3: Implement the helpers**

Append to `scripts/subnuclear_common.py`:
```python
from __future__ import annotations

import numpy as np


def select_pass_indices(sources, labels, keep_sources, max_per_source=None,
                        seed=0, limit=None, drop_unlabeled=False) -> np.ndarray:
    """Global LMDB indices for the chosen sources, optionally capped per source
    (seeded), label -1 optionally dropped, optionally truncated to ``limit``.
    Returns a sorted int array."""
    sources = np.asarray(sources)
    labels = np.asarray(labels)
    keep = np.isin(sources, list(keep_sources))
    if drop_unlabeled:
        keep &= labels != -1
    idx = np.where(keep)[0]
    if max_per_source is not None:
        rng = np.random.default_rng(seed)
        picked = []
        for s in keep_sources:
            si = idx[sources[idx] == s]
            if len(si) > max_per_source:
                si = rng.choice(si, size=max_per_source, replace=False)
            picked.append(si)
        idx = np.concatenate(picked) if picked else idx
    idx = np.sort(idx)
    if limit is not None:
        idx = idx[:limit]
    return idx


def balanced_subset(labels, per_class, seed=0) -> np.ndarray:
    """Up to ``per_class`` indices per non-negative class label (seeded). Sorted."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(labels):
        if c == -1:
            continue
        ci = np.where(labels == c)[0]
        if len(ci) > per_class:
            ci = rng.choice(ci, size=per_class, replace=False)
        picked.append(ci)
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


def nuc_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("nuc_")]


def ctx_feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("ctx_")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_subnuclear_common.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/subnuclear_common.py tests/test_subnuclear_common.py
git commit -m "feat(subnuclear): pure orchestration helpers (index/subset/columns)"
```

---

### Task 6: feature-pass script (StarDist → parquet, smoke-ETA gate)

**Files:**
- Create: `scripts/subnuclear_feature_pass.py`

- [ ] **Step 1: Write the script**

`scripts/subnuclear_feature_pass.py`:
```python
"""Stream the breast-6source LMDB through StarDist + nucleus_feature_vector,
writing seg_features.parquet in chunks (+ optional packed-bit center-mask cache).

A --limit smoke prints patches/sec and a projected full-pass ETA before any long
run is launched. Pass the RAW uint16 patch to StarDist/features (they normalize
internally); do NOT /65535 here.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnuclear_common import select_pass_indices  # noqa: E402

from dapidl.qc.patch_features import nucleus_feature_vector  # noqa: E402
from starpose.qc.segmentation_grounded import (  # noqa: E402
    SegmentationGroundedScorer, SegQCConfig, select_center_nucleus,
)

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
TRAIN = ["xenium_rep1", "sthelar_breast_s0", "sthelar_breast_s1",
         "sthelar_breast_s3", "sthelar_breast_s6"]
TEST = ["xenium_rep2"]


def _read_patch(txn, idx: int) -> np.ndarray:
    value = txn.get(struct.pack(">Q", int(idx)))
    return np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="all", help="all|train|test|comma list")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: process N then stop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--save-masks", choices=["none", "all"], default="none")
    ap.add_argument("--out-dir", default="pipeline_output/subnuclear_2026_06")
    ap.add_argument("--cpu", action="store_true", help="force CPU StarDist")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "center_masks").mkdir(exist_ok=True)

    labels = np.load(LMDB_DIR / "labels.npy")
    sources = np.load(LMDB_DIR / "sources.npy", allow_pickle=True)  # trusted self-built LMDB
    keep = {"all": TRAIN + TEST, "train": TRAIN, "test": TEST}.get(
        args.sources, args.sources.split(","))
    idx = select_pass_indices(sources, labels, keep, max_per_source=args.max_per_source,
                              seed=args.seed, limit=args.limit, drop_unlabeled=False)
    print(f"[feature-pass] {len(idx)} patches from sources={keep} "
          f"max_per_source={args.max_per_source} limit={args.limit}")

    scorer = SegmentationGroundedScorer(SegQCConfig(), gpu=not args.cpu, pixel_size=0.2125)
    cfg = scorer.cfg

    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False,
                    readahead=False, meminit=False)
    writer = None
    rows: list[dict] = []
    mask_buf: list[np.ndarray] = []
    mask_idx: list[int] = []
    t0 = time.time()
    n_done = n_nuc = 0
    suffix = f"{args.sources}" + (f"_cap{args.max_per_source}" if args.max_per_source else "")
    pq_path = out / (f"seg_features_{suffix}.parquet" if suffix else "seg_features.parquet")

    def flush_rows():
        nonlocal writer, rows
        if not rows:
            return
        table = pl.DataFrame(rows).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(pq_path, table.schema)
        writer.write_table(table)
        rows = []

    def flush_masks():
        nonlocal mask_buf, mask_idx
        if args.save_masks == "none" or not mask_buf:
            mask_buf, mask_idx = [], []
            return
        packed = np.packbits(np.stack(mask_buf).reshape(len(mask_buf), -1), axis=1)
        np.savez_compressed(out / "center_masks" / f"chunk_{mask_idx[0]:09d}.npz",
                            idx=np.array(mask_idx), masks=packed, shape=np.array([128, 128]))
        mask_buf, mask_idx = [], []

    with env.begin() as txn:
        for gi in idx:
            patch = _read_patch(txn, gi)
            masks, probs = scorer._segment(patch)
            cn = select_center_nucleus(masks, probs, cfg)
            mask = cn.mask if cn is not None else None
            prob = cn.prob if cn is not None else 0.0
            feats = nucleus_feature_vector(patch, mask, prob, cfg, scorer.pixel_size)
            row = {"global_idx": int(gi), "source": str(sources[gi]),
                   "label": int(labels[gi]), **feats}
            rows.append(row)
            if cn is not None:
                n_nuc += 1
                if args.save_masks == "all":
                    mask_buf.append(mask.astype(bool))
                    mask_idx.append(int(gi))
            n_done += 1
            if n_done % args.chunk == 0:
                flush_rows()
                flush_masks()
                rate = n_done / (time.time() - t0)
                print(f"[feature-pass] {n_done}/{len(idx)} ({rate:.1f}/s) "
                      f"nucleus_coverage={n_nuc/n_done:.3f}")
    flush_rows()
    flush_masks()
    if writer is not None:
        writer.close()

    dt = time.time() - t0
    rate = n_done / dt if dt > 0 else 0.0
    print(f"[feature-pass] DONE {n_done} patches in {dt:.1f}s ({rate:.1f}/s); "
          f"nucleus_coverage={n_nuc/max(n_done,1):.3f}; wrote {pq_path}")
    if args.limit is not None:
        full = len(select_pass_indices(sources, labels, TRAIN + TEST))
        eta_h = full / rate / 3600 if rate > 0 else float("inf")
        print(f"[ETA] full {full} patches @ {rate:.1f}/s ≈ {eta_h:.1f} h "
              f"(rep2 only: {len(select_pass_indices(sources, labels, TEST))} "
              f"≈ {len(select_pass_indices(sources, labels, TEST))/rate/3600:.2f} h)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run it (this is the script's test — exact command + expected output)**

Run: `nvidia-smi --query-gpu=memory.free --format=csv` (confirm ≥4 GB free), then
Run: `uv run python scripts/subnuclear_feature_pass.py --sources test --limit 500`
Expected: prints `[feature-pass] 500 patches ...`, a `DONE ... (X.X/s)` line, an `[ETA] ...` line, and writes `pipeline_output/subnuclear_2026_06/seg_features_test.parquet`. No traceback. `nucleus_coverage` is a sane fraction (≈0.6–0.95).

- [ ] **Step 3: Verify the parquet is well-formed and schema-stable**

Run:
```bash
uv run python -c "import polars as pl; d=pl.read_parquet('pipeline_output/subnuclear_2026_06/seg_features_test.parquet'); print(d.shape); print([c for c in d.columns if c.startswith('nuc_')][:5]); print('has_nucleus mean', d['has_nucleus'].mean())"
```
Expected: 500 rows; nuc_/ctx_ columns present; `has_nucleus mean` in (0,1].

- [ ] **Step 4: Commit**

```bash
git add scripts/subnuclear_feature_pass.py
git commit -m "feat(subnuclear): streaming StarDist feature pass with smoke-ETA gate"
```

---

### Task 7: floor script (LightGBM nuc-only vs nuc+ctx vs 0.619)

**Files:**
- Create: `scripts/subnuclear_floor.py`

- [ ] **Step 1: Write the script**

`scripts/subnuclear_floor.py`:
```python
"""Information-floor readout (C): can a LightGBM tree on per-nucleus features match
the EfficientNet's 0.619 macro-F1 on xenium_rep2? Trains nuc-only and nuc+ctx,
tests on ALL rep2 rows, writes floor_metrics.json + feature importances.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnuclear_common import ctx_feature_columns, nuc_feature_columns  # noqa: E402

CLASS_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
TRAIN = ["xenium_rep1", "sthelar_breast_s0", "sthelar_breast_s1",
         "sthelar_breast_s3", "sthelar_breast_s6"]
TEST = "xenium_rep2"
EFFNET_MACRO_F1 = 0.619


def _fit_eval(df: pl.DataFrame, feat_cols: list[str], seed: int) -> dict:
    tr = df.filter(pl.col("source").is_in(TRAIN) & (pl.col("label") != -1))
    te = df.filter((pl.col("source") == TEST) & (pl.col("label") != -1))
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(tr)) < 0.1
    Xtr = tr.select(feat_cols).to_pandas()
    ytr = tr["label"].to_numpy()
    Xte = te.select(feat_cols).to_pandas()
    yte = te["label"].to_numpy()
    clf = LGBMClassifier(objective="multiclass", num_class=4, n_estimators=2000,
                         learning_rate=0.05, class_weight="balanced",
                         random_state=seed, n_jobs=-1)
    clf.fit(Xtr[~is_val], ytr[~is_val],
            eval_set=[(Xtr[is_val], ytr[is_val])],
            callbacks=[early_stopping(50), log_evaluation(0)])
    pred = clf.predict(Xte)
    per_class = f1_score(yte, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "n_features": len(feat_cols),
        "n_train": int((~is_val).sum()), "n_test": int(len(yte)),
        "best_iteration": int(clf.best_iteration_ or clf.n_estimators),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "per_class_f1": {n: float(v) for n, v in zip(CLASS_NAMES, per_class)},
        "vs_effnet_0619": float(f1_score(yte, pred, average="macro", zero_division=0)) - EFFNET_MACRO_F1,
        "importances": dict(sorted(
            zip(feat_cols, (int(i) for i in clf.feature_importances_)),
            key=lambda kv: -kv[1])[:15]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="pipeline_output/subnuclear_2026_06/seg_features.parquet")
    ap.add_argument("--out", default="pipeline_output/subnuclear_2026_06/floor_metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pl.read_parquet(args.features)
    nuc = nuc_feature_columns(df.columns)
    ctx = ctx_feature_columns(df.columns)
    result = {
        "effnet_macro_f1": EFFNET_MACRO_F1,
        "pixel_size_caveat": "area_um2 uses 0.2125 µm/px for all sources",
        "nuc_only": _fit_eval(df, nuc, args.seed),
        "nuc_plus_ctx": _fit_eval(df, nuc + ctx, args.seed),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k]["macro_f1"] for k in ("nuc_only", "nuc_plus_ctx")}, indent=2))
    print(f"[floor] wrote {args.out}  (EffNet ref = {EFFNET_MACRO_F1})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the 500-row smoke parquet (sanity, not a real result)**

Run: `uv run python scripts/subnuclear_floor.py --features pipeline_output/subnuclear_2026_06/seg_features_test.parquet --out /tmp/floor_smoke.json`
Expected: runs without traceback and prints a JSON object with `nuc_only`/`nuc_plus_ctx` macro-F1 floats. (Numbers are meaningless on 500 rows / no train sources — this only proves the code path. It may warn about empty train split; if so, the real run uses a parquet that includes train sources.)

> Note: the 500-row smoke parquet is `--sources test` only, so it has **no train rows**. For this sanity step, regenerate a tiny mixed parquet first if needed:
> `uv run python scripts/subnuclear_feature_pass.py --sources all --max-per-source 200 --out-dir /tmp/sn_smoke` then point `--features` at `/tmp/sn_smoke/seg_features_all_cap200.parquet`.

- [ ] **Step 3: Commit**

```bash
git add scripts/subnuclear_floor.py
git commit -m "feat(subnuclear): LightGBM information-floor readout vs 0.619"
```

---

### Task 8: saliency script (IG concentration + overlays)

**Files:**
- Create: `scripts/subnuclear_saliency.py`

- [ ] **Step 1: Write the script**

`scripts/subnuclear_saliency.py`:
```python
"""Attribution readout (D): Integrated-Gradients concentration inside the nucleus
on a balanced rep2 subset. Writes saliency_summary.json + a few overlay PNGs.

IG uses the EXACT training transform (/65535 -> standardize); segmentation/area
use the raw patch via the QC scorer.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnuclear_common import balanced_subset  # noqa: E402
from breast_pooled_train import DapiClassifier  # noqa: E402  (import-safe)

from dapidl.qc.attribution import (  # noqa: E402
    attribution_concentration, fraction_in_mask, integrated_gradients,
)
from starpose.qc.segmentation_grounded import (  # noqa: E402
    SegmentationGroundedScorer, SegQCConfig, select_center_nucleus,
)

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
CKPT = Path("pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt")
CLASS_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
NORM_MEAN, NORM_STD = 0.485, 0.229


def _read_patch(txn, idx: int) -> np.ndarray:
    return np.frombuffer(txn.get(struct.pack(">Q", int(idx)))[8:],
                         dtype=np.uint16).reshape(128, 128)


def _to_input(patch: np.ndarray, device) -> torch.Tensor:
    img = (patch.astype(np.float32) / 65535.0 - NORM_MEAN) / NORM_STD
    return torch.from_numpy(img)[None, None].to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=750)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-overlays", type=int, default=12)
    ap.add_argument("--out-dir", default="pipeline_output/subnuclear_2026_06")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    labels = np.load(LMDB_DIR / "labels.npy")
    sources = np.load(LMDB_DIR / "sources.npy", allow_pickle=True)  # trusted self-built LMDB
    rep2 = np.where((sources == "xenium_rep2") & (labels != -1))[0]
    sub_local = balanced_subset(labels[rep2], args.per_class, seed=args.seed)
    subset = rep2[sub_local]

    model = DapiClassifier(4, backbone="efficientnetv2_rw_s").to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    scorer = SegmentationGroundedScorer(SegQCConfig(), gpu=not args.cpu, pixel_size=0.2125)
    cfg = scorer.cfg
    env = lmdb.open(str(LMDB_DIR / "patches.lmdb"), readonly=True, lock=False,
                    readahead=False, meminit=False)

    records = []
    overlays_left = args.n_overlays
    with env.begin() as txn:
        for gi in subset:
            patch = _read_patch(txn, int(gi))
            masks, probs = scorer._segment(patch)
            cn = select_center_nucleus(masks, probs, cfg)
            if cn is None:
                continue
            x = _to_input(patch, device)
            logits = model(x)
            pred = int(logits.argmax(1).item())
            attr = integrated_gradients(model, x, pred, steps=args.steps)
            attr_hw = attr.squeeze().cpu().numpy()
            area_frac = float(cn.mask.sum()) / cn.mask.size
            frac = fraction_in_mask(attr_hw, cn.mask)
            conc = attribution_concentration(attr_hw, cn.mask, area_frac)
            true = int(labels[gi])
            records.append({"label": true, "pred": pred, "correct": pred == true,
                            "area_fraction": area_frac, "fraction_in_mask": frac,
                            "concentration": conc})
            if overlays_left > 0:
                _save_overlay(out / "overlays" / f"ex_{int(gi)}.png",
                              patch, cn.mask, attr_hw, true, pred, conc)
                overlays_left -= 1

    _summarize(records, CLASS_NAMES, out / "saliency_summary.json", args)


def _save_overlay(path, patch, mask, attr, true, pred, conc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(patch, cmap="gray"); ax[0].set_title("DAPI"); ax[0].axis("off")
    ax[0].contour(find_boundaries(mask), colors="lime", linewidths=0.6)
    ax[1].imshow(np.abs(attr), cmap="magma"); ax[1].axis("off")
    ax[1].contour(find_boundaries(mask), colors="lime", linewidths=0.6)
    ax[1].set_title(f"|IG|  conc={conc:.2f}")
    fig.suptitle(f"true={CLASS_NAMES[true]} pred={CLASS_NAMES[pred]}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _iqr(a):
    a = np.asarray(a, float)
    return [float(np.percentile(a, 25)), float(np.percentile(a, 75))] if a.size else [float("nan")] * 2


def _summarize(records, names, path, args):
    conc = np.array([r["concentration"] for r in records], float)
    summary = {
        "n": len(records), "per_class": args.per_class, "ig_steps": args.steps,
        "checkpoint": str(CKPT),
        "overall": {"mean_concentration": float(conc.mean()) if conc.size else float("nan"),
                    "iqr": _iqr(conc),
                    "mean_fraction_in_mask": float(np.mean([r["fraction_in_mask"] for r in records])) if records else float("nan")},
        "by_class": {}, "by_correct": {},
    }
    for c, n in enumerate(names):
        cc = np.array([r["concentration"] for r in records if r["label"] == c], float)
        summary["by_class"][n] = {"n": int(cc.size),
                                  "mean_concentration": float(cc.mean()) if cc.size else float("nan"),
                                  "iqr": _iqr(cc)}
    for ok in (True, False):
        cc = np.array([r["concentration"] for r in records if r["correct"] is ok], float)
        summary["by_correct"]["correct" if ok else "incorrect"] = {
            "n": int(cc.size), "mean_concentration": float(cc.mean()) if cc.size else float("nan")}
    Path(path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"[saliency] wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a tiny balanced subset (proves checkpoint load + IG + overlay path)**

Run: `nvidia-smi --query-gpu=memory.free --format=csv` (confirm ≥4 GB), then
Run: `uv run python scripts/subnuclear_saliency.py --per-class 8 --steps 16 --n-overlays 3`
Expected: loads the checkpoint, prints an `overall` JSON block with a finite `mean_concentration`, writes `saliency_summary.json` + 3 PNGs in `overlays/`. No traceback.

- [ ] **Step 3: Eyeball one overlay**

Run: `uv run python -c "import os,glob; print(sorted(glob.glob('pipeline_output/subnuclear_2026_06/overlays/*.png'))[:3])"`
Expected: lists ≥3 PNG paths (the |IG| map should visibly concentrate on/near the green nucleus contour — confirm by opening one).

- [ ] **Step 4: Commit**

```bash
git add scripts/subnuclear_saliency.py
git commit -m "feat(subnuclear): IG attribution-concentration readout + overlays"
```

---

### Task 9: driver + README, then run the experiment (controller-executed GPU pass)

**Files:**
- Create: `pipeline_output/subnuclear_2026_06/run_triangulation.sh`, `pipeline_output/subnuclear_2026_06/README.md`

- [ ] **Step 1: Write the driver**

`pipeline_output/subnuclear_2026_06/run_triangulation.sh`:
```bash
#!/usr/bin/env bash
# Subnuclear-structure triangulation driver. Run from repo root.
# Stage 1 (smoke) prints throughput + ETA and STOPS for a human go/no-go.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=pipeline_output/subnuclear_2026_06

TRAIN_CAP=${TRAIN_CAP:-40000}     # stratified train cap for the floor (feasibility)
PER_CLASS=${PER_CLASS:-750}       # saliency balanced subset per class

echo "### GPU"; nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv

echo "### Stage 1: smoke (500 rep2 patches) -> throughput + ETA"
uv run python scripts/subnuclear_feature_pass.py --sources test --limit 500

cat <<'MSG'

  >>> Review the [ETA] line above. If the full/sampled pass is acceptable, re-run
  >>> this script with RUN_FULL=1 to execute stages 2-4. Otherwise lower TRAIN_CAP.

MSG
[ "${RUN_FULL:-0}" = "1" ] || exit 0

echo "### Stage 2a: full rep2 feature pass (test set, with masks)"
uv run python scripts/subnuclear_feature_pass.py --sources test --save-masks all
echo "### Stage 2b: stratified train feature pass (cap=${TRAIN_CAP})"
uv run python scripts/subnuclear_feature_pass.py --sources train --max-per-source "${TRAIN_CAP}"

echo "### merge train+test parquet -> seg_features.parquet"
uv run python - <<PY
import polars as pl
from pathlib import Path
out = Path("${OUT}")
parts = [pl.read_parquet(p) for p in out.glob("seg_features_*.parquet")]
pl.concat(parts, how="vertical_relaxed").write_parquet(out / "seg_features.parquet")
print("merged", sum(len(p) for p in parts), "rows")
PY

echo "### Stage 3: information floor"
uv run python scripts/subnuclear_floor.py
echo "### Stage 4: attribution saliency"
uv run python scripts/subnuclear_saliency.py --per-class "${PER_CLASS}"

echo "### READOUT"
uv run python - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
floor = json.loads((out / "floor_metrics.json").read_text())
sal = json.loads((out / "saliency_summary.json").read_text())
print(f"EffNet ref macro-F1            : {floor['effnet_macro_f1']:.3f}")
print(f"(C) nuc-only floor macro-F1    : {floor['nuc_only']['macro_f1']:.3f}  (gap {floor['nuc_only']['vs_effnet_0619']:+.3f})")
print(f"(C) nuc+ctx floor  macro-F1    : {floor['nuc_plus_ctx']['macro_f1']:.3f}  (gap {floor['nuc_plus_ctx']['vs_effnet_0619']:+.3f})")
print(f"(D) IG concentration (overall) : {sal['overall']['mean_concentration']:.2f}  (≈1 nucleus-indifferent, ≫1 subnuclear-driven, <1 context-driven)")
print("    per-class concentration     :", {k: round(v['mean_concentration'], 2) for k, v in sal['by_class'].items()})
PY
```

- [ ] **Step 2: Make it executable + lint everything + full test suite**

Run:
```bash
chmod +x pipeline_output/subnuclear_2026_06/run_triangulation.sh
uv run ruff check src/dapidl/qc/patch_features.py src/dapidl/qc/attribution.py scripts/subnuclear_*.py
uv run pytest tests/test_patch_features.py tests/test_attribution.py tests/test_subnuclear_common.py -q
```
Expected: ruff clean (fix any issues), all unit tests pass.

- [ ] **Step 3: Write the README (group definitions + how to run + how to read)**

`pipeline_output/subnuclear_2026_06/README.md` — short: what the two readouts mean, the decision rubric table (copy from the spec §7), the exact commands, and the `area_um2` pixel-size caveat.

- [ ] **Step 4: Commit the code**

```bash
git add pipeline_output/subnuclear_2026_06/run_triangulation.sh pipeline_output/subnuclear_2026_06/README.md
git commit -m "feat(subnuclear): triangulation driver + README"
```

- [ ] **Step 5: CONTROLLER-EXECUTED — run the real GPU experiment**

> This is **not** a subagent task. The controller (main session) runs it so throughput/ETA and the go/no-go are visible:
> 1. `nvidia-smi` (≥4 GB free; the model + StarDist fit in ~2–4 GB).
> 2. `bash pipeline_output/subnuclear_2026_06/run_triangulation.sh` → read the `[ETA]` line.
> 3. Decide `TRAIN_CAP` from the measured rate (target a few hours, not days), then `RUN_FULL=1 TRAIN_CAP=<n> bash pipeline_output/subnuclear_2026_06/run_triangulation.sh` (background it if long).
> 4. Read the `### READOUT` block → fill the decision rubric (greenlight vs skip the phase-A retrain matrix), per class.

**Decision rubric (from spec §7):**

| Readout | Reads | Interpretation |
|---|---|---|
| (C) nuc-only floor vs 0.619 | macro-F1 gap | small gap → CNN's spatial subnuclear modelling adds little; large gap → it has value |
| (C) +context vs nuc-only | macro-F1 lift | how much context adds on top of nucleus summary stats |
| (D) concentration ratio | ÷ area-fraction | ≈1 nucleus ignored; ≫1 subnuclear-driven; <1 context-driven |

---

## Self-review (run after Task 9, before execution)

1. **Spec coverage:** (C) floor → Tasks 2,5,7; (D) attribution → Tasks 3,4,8; shared StarDist pass → Task 6; mask cache for phase A → Task 6 (`--save-masks all`, packed-bit npz); throughput gate → Task 6 smoke + Task 9 driver; Haralick → Task 1; tests → Tasks 1–5; streaming/RAM → Task 6 chunked writer; no-retrain → nothing trains the CNN. ✅ All spec sections map to a task.
2. **Placeholder scan:** every code step has complete, runnable code; the only "short" deliverable is the README prose (Task 9 Step 3), which is descriptive content, not code. ✅
3. **Type consistency:** `nucleus_feature_vector(patch, mask, prob, cfg, pixel_size)`, `NUC_COLUMNS`/`CTX_COLUMNS`, `integrated_gradients(model, x, target, baseline, steps)`, `fraction_in_mask(attr, mask)`, `attribution_concentration(attr, mask, area_fraction)`, `select_pass_indices(...)`, `balanced_subset(labels, per_class, seed)` — names/signatures identical across the scripts that import them. ✅
4. **Deviations from spec (intentional, documented):** reuse starpose's existing GLCM technique instead of a fresh skimage path; `ctx_*` omits constant square geometry; `ctx_*` computed even when no nucleus; pixel size fixed at 0.2125 for all sources (caveat surfaced in `floor_metrics.json`).
