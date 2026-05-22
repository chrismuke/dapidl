# Segmentation-Grounded Nucleus QC Scorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A StarDist-grounded, per-nucleus rejector that flags *obviously* broken DAPIDL training patches (false detection / off-center / edge-cut / flat-interior) with a reason, validated by ladders + a stratified by-class false-positive audit.

**Architecture:** Pure, GPU-free scoring functions (structure / centeredness / completeness / objectness / broken-decision) operate on a patch + a segmentation result (label mask + per-object prob). A thin `SegmentationGroundedScorer` runs StarDist to produce that segmentation, then composes the pure functions. A dapidl dataset pass scores every patch, writes a sidecar + a stratified audit, and builds montages. The pure/segmentation split makes all logic unit-testable without a GPU.

**Tech Stack:** Python, numpy, scipy.ndimage, scikit-image, polars, StarDist (`2D_versatile_fluo`, already installed), `starpose.qc.base` ABC (imported read-only), pytest, `uv run`.

**Location decision (deviation from spec, flagged for user):** the scorer lives in **dapidl** (`src/dapidl/qc/segmentation_grounded.py`), importing the `starpose.qc.base` ABC and the `stardist` package directly — **no files are created or modified in the starpose repo** (stricter than the spec's "no starpose *changes*", and keeps all work in one repo/branch). It remains swappable with `ClassicalQualityScorer` via the shared ABC.

**Branch:** `feat/nucleus-qc-scorer` (already created off `main`).

---

## File Structure

- **Create** `src/dapidl/qc/segmentation_grounded.py` — `SegQCConfig`, `CenterNucleus`, the pure scoring functions, `decide_broken`, and `SegmentationGroundedScorer(QualityScorer)`.
- **Create** `src/dapidl/pipeline/steps/quality_control_seg.py` — dataset pass (mirrors `quality_control.py`) + the stratified audit.
- **Modify** `src/dapidl/qc/montage.py` — add `build_reason_montage`.
- **Modify** `scripts/qc_validation_montage.py` — allow `--metric` to include the new sub-scores.
- **Create** `tests/test_qc_segmentation_grounded.py`, `tests/test_quality_control_seg.py`.

Reused read-only: `dapidl.qc.io.read_patches`, `quality_control._load_patch_labels` / `_slide_groups`, `starpose.qc.base.{QualityScorer,QualityScore,NormRef}`.

---

## Task 0: Setup & StarDist prob-capture verification

**Files:** none committed (verification only).

- [ ] **Step 1: Confirm branch and StarDist**

Run:
```bash
cd /mnt/work/git/dapidl && git branch --show-current
uv run python -c "from stardist.models import StarDist2D; import starpose.qc.base as b; print('ok', b.QualityScore.__dataclass_fields__.keys())"
```
Expected: prints `feat/nucleus-qc-scorer` then `ok dict_keys(['focus_score', 'detection_score', 'qc_score', 'metrics'])`.

- [ ] **Step 2: Verify `predict_instances` returns per-object probabilities and the label→prob indexing**

Run:
```bash
uv run python - <<'PY' 2>&1 | grep -vE 'oneDNN|tensorflow/core|TF_ENABLE|AVX|deprecated'
import numpy as np
from stardist.models import StarDist2D
m = StarDist2D.from_pretrained("2D_versatile_fluo")
# two bright blobs on dark background
img = np.zeros((128,128), np.float32)
for cy,cx in [(40,40),(90,90)]:
    yy,xx = np.ogrid[:128,:128]
    img[(yy-cy)**2+(xx-cx)**2 < 9**2] = 1.0
labels, details = m.predict_instances(img)
print("n_labels", labels.max(), "prob_len", len(details["prob"]), "probs", np.round(details["prob"],3))
PY
```
Expected: `n_labels 2 prob_len 2 probs [..]` — confirms `details["prob"]` has one entry per label, where label value `k` maps to `details["prob"][k-1]`. Note this mapping; the implementation relies on it.

---

## Task 1: `SegQCConfig` + center-nucleus selection

**Files:**
- Create: `src/dapidl/qc/segmentation_grounded.py`
- Test: `tests/test_qc_segmentation_grounded.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qc_segmentation_grounded.py
import numpy as np
from dapidl.qc.segmentation_grounded import SegQCConfig, select_center_nucleus


def _disk(h, w, cy, cx, r, label):
    yy, xx = np.ogrid[:h, :w]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    out = np.zeros((h, w), np.int32)
    out[m] = label
    return out


def test_select_center_nucleus_prefers_object_covering_center():
    masks = _disk(128, 128, 64, 64, 12, 1) + _disk(128, 128, 20, 20, 8, 2)
    probs = np.array([0.9, 0.8])
    cn = select_center_nucleus(masks, probs, SegQCConfig())
    assert cn is not None and cn.label == 1
    assert cn.prob == 0.9
    assert cn.mask.sum() > 0 and cn.mask[64, 64]


def test_select_center_nucleus_none_when_center_empty_and_far():
    masks = _disk(128, 128, 15, 15, 6, 1)  # only a far corner object
    cn = select_center_nucleus(masks, np.array([0.9]), SegQCConfig())
    assert cn is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/work/git/dapidl && uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: FAIL — `ModuleNotFoundError: dapidl.qc.segmentation_grounded`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/qc/segmentation_grounded.py
"""Segmentation-grounded per-nucleus QC: reject obviously broken training patches.

Pure scoring functions (GPU-free) operate on a patch + a StarDist segmentation
(label mask + per-object prob). SegmentationGroundedScorer (Task 6) supplies the
segmentation. Lives in dapidl; imports the starpose.qc.base ABC read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegQCConfig:
    """Thresholds for the broken-patch rejector. Conservative = high specificity."""

    erode_px: int = 3
    min_interior_px: int = 20
    structure_floor: float = 0.05      # absolute floor on structure_raw
    area_min_um2: float = 8.0
    area_max_um2: float = 400.0        # above => merged/touching blob
    edge_px: int = 1                   # mask within this many px of frame => cut
    center_max_dist_frac: float = 0.35  # centroid dist / half-patch
    dominant_min_frac: float = 0.5     # target's share of the central box
    central_box_frac: float = 0.5      # central box size as fraction of patch
    prob_min: float = 0.40             # StarDist objectness floor
    solidity_min: float = 0.50         # below => debris-like
    eccentricity_max: float = 0.98     # above => line-like
    intensity_ratio_min: float = 1.10  # interior mean / background median
    structure_min: float = 0.15        # for the OPTIONAL structure cut (off by default)


@dataclass(frozen=True)
class CenterNucleus:
    """The chosen target nucleus in a patch."""

    label: int
    mask: np.ndarray   # bool (H, W)
    prob: float
    centroid: tuple[float, float]  # (y, x) px
    area_px: int


def select_center_nucleus(
    masks: np.ndarray, probs: np.ndarray, cfg: SegQCConfig
) -> CenterNucleus | None:
    """Pick the object covering the patch centre; else nearest centroid within
    a small radius; else None (no nucleus at centre)."""
    h, w = masks.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    center_label = int(masks[round(cy), round(cx)])
    if center_label == 0:
        # nearest object centroid within radius = center_max_dist_frac * half-patch
        radius = cfg.center_max_dist_frac * (h / 2.0)
        best, best_d = 0, radius
        for lab in range(1, int(masks.max()) + 1):
            m = masks == lab
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            d = float(np.hypot(ys.mean() - cy, xs.mean() - cx))
            if d < best_d:
                best, best_d = lab, d
        center_label = best
    if center_label == 0:
        return None
    m = masks == center_label
    ys, xs = np.nonzero(m)
    return CenterNucleus(
        label=center_label,
        mask=m,
        prob=float(probs[center_label - 1]),
        centroid=(float(ys.mean()), float(xs.mean())),
        area_px=int(m.sum()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/segmentation_grounded.py tests/test_qc_segmentation_grounded.py
git commit -m "feat(qc): SegQCConfig + center-nucleus selection"
```

---

## Task 2: structure score (texture inside the eroded mask)

**Files:**
- Modify: `src/dapidl/qc/segmentation_grounded.py`
- Test: `tests/test_qc_segmentation_grounded.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_qc_segmentation_grounded.py
from dapidl.qc.segmentation_grounded import structure_raw, structure_score


def test_structure_raw_textured_beats_flat_same_mean():
    h = w = 128
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 < 18 ** 2
    flat = np.full((h, w), 1000.0)
    textured = flat.copy()
    textured[mask] += ((np.indices((h, w)).sum(0) % 7) * 60.0)[mask]  # high-freq detail
    cfg = SegQCConfig()
    assert structure_raw(textured, mask, cfg) > 5 * structure_raw(flat, mask, cfg)


def test_structure_raw_zero_when_interior_too_small():
    cfg = SegQCConfig()
    tiny = np.zeros((128, 128), bool)
    tiny[64, 64] = True
    assert structure_raw(np.random.rand(128, 128) * 1000, tiny, cfg) == 0.0


def test_structure_score_calibrates_with_floor():
    cfg = SegQCConfig()
    # raw below floor -> 0; raw == p90 -> 1
    assert structure_score(0.0, ref_p90=2.0, cfg=cfg) == 0.0
    assert structure_score(2.0 + cfg.structure_floor, ref_p90=2.0, cfg=cfg) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: FAIL — `ImportError: cannot import name 'structure_raw'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/dapidl/qc/segmentation_grounded.py
from scipy import ndimage


def _eroded_interior(mask: np.ndarray, cfg: SegQCConfig) -> np.ndarray:
    if cfg.erode_px > 0:
        return ndimage.binary_erosion(mask, iterations=cfg.erode_px)
    return mask


def structure_raw(patch: np.ndarray, mask: np.ndarray, cfg: SegQCConfig) -> float:
    """MAD-normalized high-frequency (LoG) energy inside the eroded nucleus mask.

    Robust-normalizing by the interior MAD makes it invariant to per-slide
    brightness/contrast. Returns 0 if the eroded interior is too small to judge.
    """
    interior_mask = _eroded_interior(mask, cfg)
    if interior_mask.sum() < cfg.min_interior_px:
        return 0.0
    vals = patch[interior_mask].astype(np.float64)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    norm = (patch.astype(np.float64) - med) / mad
    lap = ndimage.gaussian_laplace(norm, sigma=1.0)
    return float(np.mean(lap[interior_mask] ** 2))


def structure_score(raw: float, ref_p90: float, cfg: SegQCConfig) -> float:
    """Calibrate raw structure energy to [0,1] vs a per-slide p90, with an
    absolute floor so an all-flat slide cannot manufacture passing scores."""
    denom = max(ref_p90, 1e-6)
    return float(np.clip((raw - cfg.structure_floor) / denom, 0.0, 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(qc): mask-localized structure (LoG-in-eroded-mask) score"
```

---

## Task 3: centeredness & completeness

**Files:**
- Modify: `src/dapidl/qc/segmentation_grounded.py`
- Test: `tests/test_qc_segmentation_grounded.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_qc_segmentation_grounded.py
from dapidl.qc.segmentation_grounded import (
    centeredness_score, touches_edge, area_um2, dominant_central_fraction,
)


def test_centeredness_high_at_center_low_at_offset():
    cfg = SegQCConfig()
    assert centeredness_score((64.0, 64.0), (128, 128), cfg) > 0.95
    assert centeredness_score((110.0, 110.0), (128, 128), cfg) < 0.2


def test_touches_edge():
    cfg = SegQCConfig()
    m = np.zeros((128, 128), bool); m[60:70, 0:5] = True   # touches left frame
    assert touches_edge(m, cfg)
    m2 = np.zeros((128, 128), bool); m2[60:70, 60:70] = True
    assert not touches_edge(m2, cfg)


def test_area_um2():
    m = np.zeros((128, 128), bool); m[:10, :10] = True  # 100 px
    assert abs(area_um2(m, pixel_size=0.2125) - 100 * 0.2125 ** 2) < 1e-6


def test_dominant_central_fraction():
    cfg = SegQCConfig()
    big = np.zeros((128, 128), bool); big[48:80, 48:80] = True   # fills central box
    small = np.zeros((128, 128), bool); small[48:52, 48:52] = True
    f = dominant_central_fraction(target=big, all_masks=(big | small), cfg=cfg)
    assert f > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: FAIL — import error for the new names.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/dapidl/qc/segmentation_grounded.py
def centeredness_score(centroid, patch_shape, cfg: SegQCConfig) -> float:
    h, w = patch_shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = float(np.hypot(centroid[0] - cy, centroid[1] - cx))
    return float(np.clip(1.0 - dist / (cfg.center_max_dist_frac * (h / 2.0)), 0.0, 1.0))


def touches_edge(mask: np.ndarray, cfg: SegQCConfig) -> bool:
    e = cfg.edge_px
    return bool(mask[:e, :].any() or mask[-e:, :].any()
                or mask[:, :e].any() or mask[:, -e:].any())


def area_um2(mask: np.ndarray, pixel_size: float) -> float:
    return float(mask.sum()) * pixel_size * pixel_size


def dominant_central_fraction(target: np.ndarray, all_masks: np.ndarray,
                              cfg: SegQCConfig) -> float:
    """Share of the central-box foreground that belongs to the target nucleus."""
    h, w = target.shape
    bh, bw = int(h * cfg.central_box_frac), int(w * cfg.central_box_frac)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    box = (slice(y0, y0 + bh), slice(x0, x0 + bw))
    fg = all_masks[box].sum()
    if fg == 0:
        return 0.0
    return float(target[box].sum() / fg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(qc): centeredness, edge-cut, area, dominant-central fraction"
```

---

## Task 4: objectness (real nucleus)

**Files:**
- Modify: `src/dapidl/qc/segmentation_grounded.py`
- Test: `tests/test_qc_segmentation_grounded.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_qc_segmentation_grounded.py
from dapidl.qc.segmentation_grounded import objectness_metrics


def test_objectness_round_bright_blob_is_object_like():
    cfg = SegQCConfig()
    yy, xx = np.ogrid[:128, :128]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 < 14 ** 2
    patch = np.full((128, 128), 300.0); patch[mask] = 1500.0
    om = objectness_metrics(patch, mask, prob=0.9, cfg=cfg)
    assert om["solidity"] > cfg.solidity_min
    assert om["eccentricity"] < cfg.eccentricity_max
    assert om["intensity_ratio"] > cfg.intensity_ratio_min
    assert om["objectness_score"] > 0.7


def test_objectness_low_prob_scores_low():
    cfg = SegQCConfig()
    yy, xx = np.ogrid[:128, :128]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 < 14 ** 2
    patch = np.full((128, 128), 300.0); patch[mask] = 1500.0
    om = objectness_metrics(patch, mask, prob=0.05, cfg=cfg)
    assert om["objectness_score"] < 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: FAIL — `ImportError: objectness_metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/dapidl/qc/segmentation_grounded.py
from skimage.measure import regionprops


def objectness_metrics(patch: np.ndarray, mask: np.ndarray, prob: float,
                       cfg: SegQCConfig) -> dict:
    """Real-nucleus evidence: StarDist prob, lenient morphology, intensity-above-bg.

    Morphology is intentionally lenient (only extreme outliers count) so small or
    elongated-but-valid nuclei are not penalized.
    """
    props = regionprops(mask.astype(np.int32))[0]
    ecc = float(props.eccentricity)
    solidity = float(props.solidity)
    interior = patch[mask].astype(np.float64)
    bg = patch[~mask].astype(np.float64)
    bg_med = float(np.median(bg)) if bg.size else 0.0
    intensity_ratio = float(np.median(interior) / (bg_med + 1e-6))
    morph_ok = (solidity >= cfg.solidity_min) and (ecc <= cfg.eccentricity_max)
    intensity_ok = intensity_ratio >= cfg.intensity_ratio_min
    # objectness dominated by prob, gated by sanity checks
    score = float(np.clip(prob, 0.0, 1.0)) * (1.0 if morph_ok else 0.3) \
        * (1.0 if intensity_ok else 0.5)
    return {
        "objectness_score": float(np.clip(score, 0.0, 1.0)),
        "eccentricity": ecc,
        "solidity": solidity,
        "intensity_ratio": intensity_ratio,
        "morph_ok": morph_ok,
        "intensity_ok": intensity_ok,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(qc): objectness metrics (prob + lenient morphology + intensity)"
```

---

## Task 5: compose `score_from_segmentation` + `decide_broken`

**Files:**
- Modify: `src/dapidl/qc/segmentation_grounded.py`
- Test: `tests/test_qc_segmentation_grounded.py`

`broken_reason` enum (refines the spec's 4 to 5): `no_nucleus` (nothing at centre), `false_detection` (detected but low prob / weird morph / dim), `cut_at_edge`, `off_center`, `no_structure`. **`no_structure` is OFF by default** (`use_structure_cut=False`) — structure is always *scored and reported* but never the sole drop reason unless explicitly enabled and audit-gated.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_qc_segmentation_grounded.py
from dapidl.qc.segmentation_grounded import score_from_segmentation, decide_broken


def _one_disk(cy, cx, r, label=1):
    yy, xx = np.ogrid[:128, :128]
    m = np.zeros((128, 128), np.int32); m[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = label
    return m


def test_decide_broken_no_nucleus():
    cfg = SegQCConfig()
    qs = score_from_segmentation(np.zeros((128, 128)), np.zeros((128, 128), np.int32),
                                 np.array([]), ref_p90=2.0, pixel_size=0.2125, cfg=cfg)
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason == "no_nucleus"


def test_decide_broken_cut_at_edge():
    cfg = SegQCConfig()
    masks = _one_disk(64, 64, 70)             # large disk: covers centre AND touches frame
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.9]), 2.0, 0.2125, cfg)
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason == "cut_at_edge"


def test_decide_broken_off_center():
    cfg = SegQCConfig()
    masks = _one_disk(60, 60, 6, 1)           # small, near but a neighbor dominates
    masks += _one_disk(64, 80, 16, 2)
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0
    qs = score_from_segmentation(patch, masks, np.array([0.9, 0.9]), 2.0, 0.2125, cfg)
    broken, reason = decide_broken(qs, cfg)
    assert broken and reason in ("off_center", "no_nucleus")


def test_good_nucleus_not_broken_even_if_low_structure():
    cfg = SegQCConfig()
    masks = _one_disk(64, 64, 16, 1)
    patch = np.full((128, 128), 300.0); patch[masks > 0] = 1500.0   # flat interior
    qs = score_from_segmentation(patch, masks, np.array([0.95]), 2.0, 0.2125, cfg)
    broken, reason = decide_broken(qs, cfg)            # structure cut OFF by default
    assert not broken
    assert qs.focus_score < 0.5                        # structure IS reported as low
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: FAIL — import error for `score_from_segmentation` / `decide_broken`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/dapidl/qc/segmentation_grounded.py
from starpose.qc.base import QualityScore


def score_from_segmentation(patch, masks, probs, ref_p90, pixel_size,
                            cfg: SegQCConfig) -> QualityScore:
    """Compose sub-scores into a QualityScore (no GPU). All raw signals are
    stored in metrics; broken/reason is decided separately by decide_broken."""
    cn = select_center_nucleus(masks, probs, cfg)
    if cn is None:
        return QualityScore(focus_score=0.0, detection_score=0.0, qc_score=0.0,
                            metrics={"has_nucleus": 0.0})
    s_raw = structure_raw(patch, cn.mask, cfg)
    struct = structure_score(s_raw, ref_p90, cfg)
    cent = centeredness_score(cn.centroid, patch.shape, cfg)
    a_um2 = area_um2(cn.mask, pixel_size)
    edge = touches_edge(cn.mask, cfg)
    dom = dominant_central_fraction(cn.mask, masks > 0, cfg)
    obj = objectness_metrics(patch, cn.mask, cn.prob, cfg)
    completeness = float(
        (not edge) and (cfg.area_min_um2 <= a_um2 <= cfg.area_max_um2)
    )
    qc = min(struct, cent, obj["objectness_score"])  # combined headline (reporting)
    return QualityScore(
        focus_score=struct, detection_score=obj["objectness_score"], qc_score=qc,
        metrics={
            "has_nucleus": 1.0, "structure_raw": s_raw, "centeredness": cent,
            "dominant_central": dom, "completeness": completeness,
            "area_um2": a_um2, "edge_cut": float(edge),
            "stardist_prob": cn.prob, "eccentricity": obj["eccentricity"],
            "solidity": obj["solidity"], "intensity_ratio": obj["intensity_ratio"],
            "morph_ok": float(obj["morph_ok"]), "intensity_ok": float(obj["intensity_ok"]),
        },
    )


def decide_broken(qs: QualityScore, cfg: SegQCConfig,
                  use_structure_cut: bool = False) -> tuple[bool, str]:
    """High-specificity broken decision. Order = most severe first. structure is
    never the sole reason unless use_structure_cut is explicitly enabled."""
    m = qs.metrics
    if m.get("has_nucleus", 0.0) < 1.0:
        return True, "no_nucleus"
    if m["edge_cut"] >= 1.0:
        return True, "cut_at_edge"
    if (qs.metrics["centeredness"] <= 0.0) or (m["dominant_central"] < cfg.dominant_min_frac):
        return True, "off_center"
    if (m["stardist_prob"] < cfg.prob_min) or (m["morph_ok"] < 1.0) or (m["intensity_ok"] < 1.0) \
            or not (cfg.area_min_um2 <= m["area_um2"] <= cfg.area_max_um2):
        return True, "false_detection"   # low conf / weird morph / dim / sliver / merged-blob
    if use_structure_cut and qs.focus_score < cfg.structure_min:
        return True, "no_structure"
    return False, "ok"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (15 passed).

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(qc): compose QualityScore + conservative broken decision (structure never sole by default)"
```

---

## Task 6: `SegmentationGroundedScorer` (StarDist) + `fit_reference`

**Files:**
- Modify: `src/dapidl/qc/segmentation_grounded.py`
- Test: `tests/test_qc_segmentation_grounded.py`

- [ ] **Step 1: Write the failing test** (logic-only; the StarDist call is monkeypatched so the test needs no GPU/model)

```python
# append to tests/test_qc_segmentation_grounded.py
from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer


def test_scorer_score_batch_uses_injected_segmentation(monkeypatch):
    sc = SegmentationGroundedScorer()
    # fake _segment: a centered disk + prob, independent of pixels
    yy, xx = np.ogrid[:128, :128]
    masks = np.zeros((128, 128), np.int32); masks[(yy - 64) ** 2 + (xx - 64) ** 2 < 16 ** 2] = 1
    monkeypatch.setattr(sc, "_segment", lambda p: (masks, np.array([0.95])))
    patch = np.full((1, 128, 128), 300.0, np.float64); patch[0][masks > 0] = 1500.0
    from starpose.qc.base import NormRef
    out = sc.score_batch(patch.astype(np.uint16), ref=NormRef(varlap_p90=2.0))
    assert len(out) == 1 and out[0].metrics["has_nucleus"] == 1.0
    assert sc.name == "segmentation_grounded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py::test_scorer_score_batch_uses_injected_segmentation -q`
Expected: FAIL — `ImportError: SegmentationGroundedScorer`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/dapidl/qc/segmentation_grounded.py
import os

from starpose.qc.base import NormRef, QualityScorer


class SegmentationGroundedScorer(QualityScorer):
    """StarDist-grounded broken-patch rejector. v1: StarDist only."""

    def __init__(self, cfg: SegQCConfig | None = None, gpu: bool = True,
                 pixel_size: float = 0.2125):
        self.cfg = cfg or SegQCConfig()
        self.gpu = gpu
        self.pixel_size = pixel_size
        self._model = None

    @property
    def name(self) -> str:
        return "segmentation_grounded"

    def _get_model(self):
        if self._model is None:
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            from stardist.models import StarDist2D
            self._model = StarDist2D.from_pretrained("2D_versatile_fluo")
        return self._model

    def _segment(self, patch: np.ndarray):
        """Return (label mask int32, per-object prob array). Label k -> prob[k-1]."""
        model = self._get_model()
        p_low, p_high = np.percentile(patch, [1, 99.8])
        if p_high - p_low < 1e-6:
            return np.zeros(patch.shape, np.int32), np.array([])
        img = ((patch.astype(np.float32) - p_low) / (p_high - p_low)).clip(0, 1)
        labels, details = model.predict_instances(img)
        return labels.astype(np.int32), np.asarray(details["prob"], dtype=float)

    def fit_reference(self, patches: np.ndarray) -> NormRef:
        """Per-slide structure reference: p90 of structure_raw over a sample."""
        raws = []
        for p in patches:
            masks, probs = self._segment(p)
            cn = select_center_nucleus(masks, probs, self.cfg)
            if cn is not None:
                raws.append(structure_raw(p, cn.mask, self.cfg))
        p90 = float(np.percentile(raws, 90)) if raws else 1.0
        return NormRef(varlap_p90=p90)  # field reused to hold structure-raw p90

    def score_batch(self, patches: np.ndarray, ref: NormRef | None = None):
        if ref is None:
            ref = self.fit_reference(patches)
        out = []
        for p in patches:
            masks, probs = self._segment(p)
            out.append(score_from_segmentation(
                p, masks, probs, ref.varlap_p90, self.pixel_size, self.cfg))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (16 passed).

- [ ] **Step 5: GPU smoke (manual, not a unit test)**

Run:
```bash
uv run python - <<'PY' 2>&1 | grep -vE 'oneDNN|tensorflow/core|TF_ENABLE|AVX|deprecated|Loading|Found model'
import numpy as np
from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer
sc = SegmentationGroundedScorer()
yy, xx = np.ogrid[:128, :128]
patch = np.full((128, 128), 300, np.uint16); patch[(yy-64)**2+(xx-64)**2 < 14**2] = 1500
out = sc.score_batch(np.stack([patch]))
print("metrics:", {k: round(v, 3) for k, v in out[0].metrics.items()})
PY
```
Expected: prints metrics with `has_nucleus: 1.0`, a `stardist_prob` > 0.3 — confirms the real StarDist path works end-to-end.

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat(qc): SegmentationGroundedScorer (StarDist segment + fit_reference + score_batch)"
```

---

## Task 7: dapidl dataset pass + stratified audit

**Files:**
- Create: `src/dapidl/pipeline/steps/quality_control_seg.py`
- Test: `tests/test_quality_control_seg.py`

Reuse `_load_patch_labels` and `_slide_groups` from `quality_control.py`, and `read_patches` from `dapidl.qc.io`.

- [ ] **Step 1: Write the failing test** (scorer monkeypatched → no GPU)

```python
# tests/test_quality_control_seg.py
import json
import numpy as np
import polars as pl
from dapidl.pipeline.steps.quality_control_seg import stratified_audit


def test_stratified_audit_surfaces_class_concentration():
    df = pl.DataFrame({
        "source": ["a"] * 100,
        "cell_type": ["Immune"] * 50 + ["Epithelial"] * 50,
        "area_um2": [30.0] * 100,
        "broken": [True] * 40 + [False] * 10 + [False] * 50,  # broken concentrated in Immune
        "broken_reason": (["off_center"] * 40 + ["ok"] * 10 + ["ok"] * 50),
    })
    audit = stratified_audit(df, n_size_bins=2)
    imm = audit.filter((pl.col("cell_type") == "Immune"))["broken_rate"].max()
    epi = audit.filter((pl.col("cell_type") == "Epithelial"))["broken_rate"].max()
    assert imm > epi
    assert {"source", "cell_type", "size_bin", "n", "broken_rate"}.issubset(set(audit.columns))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality_control_seg.py -q`
Expected: FAIL — `ModuleNotFoundError: dapidl.pipeline.steps.quality_control_seg`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/dapidl/pipeline/steps/quality_control_seg.py
"""Segmentation-grounded QC pass over a built DAPIDL dataset.

Mirrors quality_control.py: per-slide reference, chunked scoring, sidecar write.
Adds a stratified broken-rate audit (source x cell_type x size bin) as the
anti-censoring guardrail. metadata.parquet is never modified.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from dapidl.pipeline.steps.quality_control import _load_patch_labels, _slide_groups
from dapidl.qc.io import read_patches
from dapidl.qc.segmentation_grounded import SegmentationGroundedScorer, decide_broken

REFERENCE_SAMPLE = 2000


def stratified_audit(df: pl.DataFrame, n_size_bins: int = 4) -> pl.DataFrame:
    """Broken-rate by source x cell_type x size bin (the censoring guardrail)."""
    df = df.with_columns(
        pl.col("area_um2").qcut(n_size_bins, allow_duplicates=True, labels=[
            f"q{i}" for i in range(n_size_bins)]).alias("size_bin")
    )
    return (df.group_by(["source", "cell_type", "size_bin"])
              .agg(pl.len().alias("n"),
                   pl.col("broken").mean().alias("broken_rate"))
              .sort(["source", "cell_type", "size_bin"]))


def run_quality_control_seg(dataset_path, use_structure_cut: bool = False,
                            seed: int = 42) -> Path:
    dataset_path = Path(dataset_path)
    n, cell_ids, class_names = _load_patch_labels(dataset_path)
    sources = _slide_groups(dataset_path, n)
    scorer = SegmentationGroundedScorer()
    rng = np.random.default_rng(seed)

    cols = {k: np.zeros(n) for k in (
        "structure_score", "centeredness", "dominant_central", "completeness",
        "area_um2", "stardist_prob", "eccentricity", "solidity", "intensity_ratio")}
    broken = np.zeros(n, dtype=bool)
    reason = np.empty(n, dtype=object)

    for slide in sorted(set(sources.tolist())):
        idx = np.where(sources == slide)[0]
        sample = idx if len(idx) <= REFERENCE_SAMPLE else rng.choice(idx, REFERENCE_SAMPLE, replace=False)
        ref = scorer.fit_reference(read_patches(dataset_path, sample))
        for start in range(0, len(idx), 1000):
            chunk = idx[start:start + 1000]
            scores = scorer.score_batch(read_patches(dataset_path, chunk), ref=ref)
            for j, gi in enumerate(chunk):
                s = scores[j]
                cols["structure_score"][gi] = s.focus_score
                cols["stardist_prob"][gi] = s.metrics.get("stardist_prob", 0.0)
                for k in ("centeredness", "dominant_central", "completeness",
                          "area_um2", "eccentricity", "solidity", "intensity_ratio"):
                    cols[k][gi] = s.metrics.get(k, 0.0)
                b, r = decide_broken(s, scorer.cfg, use_structure_cut=use_structure_cut)
                broken[gi] = b
                reason[gi] = r
        logger.info(f"seg-QC scored slide {slide}: {len(idx)} patches")

    out_dir = dataset_path / "qc"
    out_dir.mkdir(exist_ok=True)
    df = pl.DataFrame({"cell_id": cell_ids, "source": sources,
                       "cell_type": class_names, **cols,
                       "broken": broken, "broken_reason": list(reason)})
    df.write_parquet(out_dir / "seg_scores.parquet")
    audit = stratified_audit(df)
    audit.write_parquet(out_dir / "seg_broken_audit.parquet")
    (out_dir / "seg_scores.meta.json").write_text(json.dumps({
        "scorer": scorer.name, "cfg": scorer.cfg.__dict__,
        "use_structure_cut": use_structure_cut,
        "broken_rate": float(broken.mean()), "date": date.today().isoformat()}, indent=2))
    logger.info(f"wrote {out_dir/'seg_scores.parquet'} (broken {broken.mean():.1%})")
    return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quality_control_seg.py -q`
Expected: PASS (1 passed). (Note: `qcut` may warn on duplicate edges with constant area; `allow_duplicates=True` handles it.)

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/pipeline/steps/quality_control_seg.py tests/test_quality_control_seg.py
git commit -m "feat(qc): seg-QC dataset pass + stratified broken-rate audit"
```

---

## Task 8: per-reason montage + ladder extension

**Files:**
- Modify: `src/dapidl/qc/montage.py`
- Modify: `scripts/qc_validation_montage.py`
- Test: `tests/test_qc_segmentation_grounded.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_qc_segmentation_grounded.py
from dapidl.qc.montage import build_reason_montage


def test_build_reason_montage_returns_rgb():
    patches = (np.random.rand(20, 128, 128) * 1000).astype(np.uint16)
    reasons = np.array(["off_center"] * 10 + ["cut_at_edge"] * 10, dtype=object)
    img = build_reason_montage(patches, reasons, reason="off_center", top_n=8)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py::test_build_reason_montage_returns_rgb -q`
Expected: FAIL — `ImportError: build_reason_montage`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/dapidl/qc/montage.py
def build_reason_montage(patches: np.ndarray, reasons: np.ndarray, reason: str,
                         top_n: int = 64, cols: int = 8) -> np.ndarray:
    """Grid of up to top_n patches flagged with a given broken_reason."""
    sel = np.where(reasons == reason)[0][:top_n]
    n = len(sel)
    rows = max(1, int(np.ceil(max(n, 1) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, idx in enumerate(sel):
        p = patches[idx].astype(np.float32)
        lo, hi = np.percentile(p, [1, 99])
        axes[i].imshow(np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
    fig.suptitle(f"broken_reason = {reason} (n shown={n})", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr
```

- [ ] **Step 4: Extend the ladder script's metric choices**

In `scripts/qc_validation_montage.py`, change the `--metric` argument's `choices` and the parquet path so it can read the seg sidecar:
```python
    ap.add_argument("--metric", default="qc_score",
                    choices=["qc_score", "focus_score", "detection_score",
                             "structure_score", "centeredness", "objectness_score"])
    ap.add_argument("--scores", default="qc/qc_scores.parquet",
                    help="sidecar relative to dataset (qc/qc_scores.parquet or qc/seg_scores.parquet)")
```
and replace the parquet load line `qc = pl.read_parquet(args.dataset / "qc" / "qc_scores.parquet")...` with `qc = pl.read_parquet(args.dataset / args.scores).sort("cell_id")`.

- [ ] **Step 5: Run tests + a ladder smoke**

Run: `uv run pytest tests/test_qc_segmentation_grounded.py -q`
Expected: PASS (17 passed).

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat(qc): per-reason montage + seg-score ladders"
```

---

## Task 9 (controller-run, not a subagent): smoke → tune → full pass → readout

**Files:** none (operational). Run by the controller.

- [ ] **Step 1: 5k/source smoke pass.** Add a `--limit-per-source` (or sample indices) path in a throwaway invocation of `run_quality_control_seg` on `~/datasets/derived/breast-6source-dapi-p128`, restricted to a stratified 5k/source sample; check GPU first (`nvidia-smi`), run with `TF_FORCE_GPU_ALLOW_GROWTH=true`.
- [ ] **Step 2: Validate.** Build ladders (`scripts/qc_validation_montage.py --scores qc/seg_scores.parquet --metric structure_score` and `--metric objectness_score`) + per-reason montages; eyeball that low rows are flat/grazing/blurred/off-center/false and high rows are crisp centrally-cut nuclei. Inspect `seg_broken_audit.parquet`: confirm broken patches are **not** concentrated in a real cell type (Immune especially). If they are → trip the escalation trigger (pause; propose a small labeled calibration set).
- [ ] **Step 3: Tune thresholds** in `SegQCConfig` conservatively from the smoke ladders; re-run smoke until separation is visually clean and the audit is unbiased.
- [ ] **Step 4: Full pass** over all 2.28M patches; write `seg_scores.parquet` + `seg_broken_audit.parquet`; copy montages/ladders to GDrive (`gdrive:dapidl-qc-montages/...`).
- [ ] **Step 5: Readout** — broken-rate by source/class/size, the censoring-guardrail verdict, and a go/no-go on dropping `broken` patches before re-training. Save the write-up to Obsidian.

---

## Self-Review

**Spec coverage:** structure-in-eroded-mask (Task 2 ✓); centeredness/completeness incl. merged-blob upper bound + edge-cut (Task 3 ✓); objectness prob+lenient-morph+intensity (Task 4 ✓); broken decision + 5 reasons, structure-never-sole (Task 5 ✓); StarDist-only scorer + per-slide ref + absolute floor (Tasks 2,6 ✓); dataset pass + sidecar + `metadata.parquet` untouched (Task 7 ✓); stratified by source×class×size audit (Task 7 ✓); ladders + per-reason montages (Task 8 ✓); smoke-before-full + escalation trigger (Task 9 ✓); no clean-bias filtering (the pass only flags broken, never weights/filters quality); Cellpose/FocusExpert deferred (not in any task ✓).

**Placeholder scan:** every code step has complete code; commands have expected output; no TBD/"handle edge cases". The only deliberately-operational task is Task 9 (controller run), which lists concrete commands/criteria rather than committed code — appropriate for a GPU run.

**Type consistency:** `SegQCConfig`, `CenterNucleus`, `select_center_nucleus`, `structure_raw`/`structure_score`, `centeredness_score`/`touches_edge`/`area_um2`/`dominant_central_fraction`, `objectness_metrics`, `score_from_segmentation`/`decide_broken`, `SegmentationGroundedScorer` (`_segment`/`fit_reference`/`score_batch`/`name`) are defined once and reused with identical signatures. `QualityScore`/`NormRef`/`QualityScorer` come from `starpose.qc.base`; `NormRef.varlap_p90` is deliberately reused to hold the per-slide structure-raw p90 (documented). `decide_broken` reads only `metrics` keys that `score_from_segmentation` writes.
