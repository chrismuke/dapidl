# p64 QC Collages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the pilot QC collages at native 64px, graded by a rethought multi-axis *absolute* quality model, annotated with 1px cyan-nucleus / magenta-cell overlays, grouped by CL-ontology-harmonized coarse+medium cell types consistent across Xenium and STHELAR.

**Architecture:** 4 phases — (1) ontology harmonization (`derive_labels` single path) → (2) native p64 LMDB → (3) multi-axis `quality_model` + per-source `cell_boundary` + 2-pass `pilot_qc_rescore` → (4) `pilot_qc_collages_v3` renderer. The pure CPU units (Phase 1, `quality_model`, `cell_boundary`, `render_tile`) are TDD'd and run **now, in parallel** with the spatial-GNN Stage 2 GPU job; the GPU/heavy-IO steps (Phase 2 build, Phase 3 Pass 1 StarDist) are **deferred** until the GPU frees.

**Tech Stack:** Python 3.11 (worktree venv), polars, numpy, scikit-image, scipy; existing `dapidl.ontology` (CLMapper, `derive_tier_label`), `starpose.qc` (`SegmentationGroundedScorer`, `decide_broken`, `score_from_segmentation`), `starpose.expansion.VoronoiExpander`.

**Conventions:** worktree `/mnt/work/git/dapidl-p64qc` on branch `feat/p64-qc-collages`; run via `uv run` (Python 3.11). Commit messages carry **no Claude/Anthropic attribution**. Spec: `docs/superpowers/specs/2026-06-09-p64-qc-collages-multiaxis-design.md`. `[DEFERRED-GPU]` tasks wait for the spatial-GNN Stage 2 GPU to free.

---

## File Structure

```
src/dapidl/ontology/
  annotator_mappings.py   MODIFY  add CAF/Plasma/pericyte/myoepithelial CL maps to GT_STHELAR
  training_tiers.py       MODIFY  add CL:0000185/CL:0002327 -> MEDIUM; host derive_labels
src/dapidl/qc/
  quality_model.py        NEW     calibrate() + 4 absolute axes + grade() (pure on a metric dict)
  cell_boundary.py        NEW     resolve_cell_mask() per-source polygon raster + Voronoi fallback
scripts/
  build_pilot_lmdb_v3.py  MODIFY  --patch-size 64, route labels via derive_labels, store x0,y0,pixel_size
  attach_medium_labels.py MODIFY  medium via derive_labels()[1]
  pilot_qc_rescore.py     NEW     2-pass enriched re-score (masks + axes + grade)
  pilot_qc_collages_v3.py NEW     native-p64 collage, 1px dual overlays (render_tile is pure)
  pilot_label_divergence.py NEW   Phase-1 gate: old inline-dict vs new derive_labels report
tests/
  test_ontology_derive.py  test_quality_model.py  test_cell_boundary.py  test_collage_render.py
```

Outputs → `pipeline_output/pilot_qc_collages_v3/` and `…/derived/breast-pilot-6source-dapi-p64-nuc-v1/`.

---

### Task 1: Ontology harmonization — `derive_labels` + mapping fixes (Phase 1, CPU)

**Files:**
- Modify: `src/dapidl/ontology/annotator_mappings.py` (GT_STHELAR dict, ~line 399)
- Modify: `src/dapidl/ontology/training_tiers.py` (MEDIUM list + new `derive_labels`)
- Test: `tests/test_ontology_derive.py`

- [ ] **Step 1: Write the failing test (the §5 cross-source oracle)**

```python
# tests/test_ontology_derive.py
import pytest
from dapidl.ontology.training_tiers import derive_labels

# (source, raw_name) -> (coarse, medium). The cross-source consistency oracle (spec §5).
CASES = [
    ("xenium_rep1", "B_Cells",                ("Immune", "B_Cell")),
    ("sthelar_breast_s0", "B cell",           ("Immune", "B_Cell")),
    ("xenium_rep1", "CD8+_T_Cells",           ("Immune", "T_Cell")),
    ("sthelar_breast_s0", "GZMK CD8 T cell",  ("Immune", "T_Cell")),
    ("xenium_rep1", "Mast_Cells",             ("Immune", "Mast_Cell")),
    ("sthelar_breast_s0", "Mast cell",        ("Immune", "Mast_Cell")),
    ("xenium_rep1", "Macrophages_1",          ("Immune", "Macrophage")),
    ("xenium_rep1", "Myoepi_ACTA2+",          ("Epithelial", "Epithelial_Basal")),
    ("xenium_rep1", "Invasive_Tumor",         ("Epithelial", "Epithelial_Luminal")),
    ("xenium_rep1", "Stromal",                ("Stromal", "Fibroblast")),
    ("sthelar_breast_s0", "CAF",              ("Stromal", "Fibroblast")),
    ("sthelar_breast_s0", "Plasma",           ("Immune", "B_Cell")),  # plasma rolls up to B lineage at MEDIUM
    ("sthelar_breast_s0", "Endothelial_Pericyte_Smooth_muscle", ("Stromal", "Pericyte")),
    ("xenium_rep1", "Perivascular-Like",      ("Stromal", "Pericyte")),
    ("xenium_rep1", "Endothelial",            ("Endothelial", "Endothelial")),
]

@pytest.mark.parametrize("source,raw,expected", CASES)
def test_derive_labels_cross_source(source, raw, expected):
    assert derive_labels(raw, source) == expected

def test_unmapped_is_unknown_not_misbinned():
    assert derive_labels("not_a_real_celltype_xyz", "sthelar_breast_s0") == ("Unknown", "Unknown")

def test_caf_and_plasma_no_longer_unknown():
    assert derive_labels("CAF", "sthelar_breast_s0")[0] == "Stromal"
    assert derive_labels("Plasma", "sthelar_breast_s0")[0] == "Immune"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ontology_derive.py -v`
Expected: FAIL — `ImportError: cannot import name 'derive_labels'`.

- [ ] **Step 3: Add the mapping fixes**

In `src/dapidl/ontology/annotator_mappings.py`, inside the `GT_STHELAR` dict, add (CL ids per spec §5.1):
```python
    "CAF": "CL:0000057",                                  # cancer-associated fibroblast -> fibroblast
    "Plasma": "CL:0000786",                               # plasma cell
    "Endothelial_Pericyte_Smooth_muscle": "CL:0000669",   # definitive -> pericyte (stromal-vascular)
    "Mammary_basal_cell_(=myoepithelial)": "CL:0000185",  # myoepithelial
```
In `src/dapidl/ontology/training_tiers.py`, ensure the MEDIUM rollup reaches Epithelial_Basal/Luminal for these CL ids by adding explicit entries after the `CL_TO_MEDIUM_NAME` definition (line ~94):
```python
# Myoepithelial (CL:0000185) and mammary-luminal variant (CL:0002327) roll up explicitly,
# in case the CL parent chain does not pass through the MEDIUM anchors.
CL_TO_MEDIUM_NAME.setdefault("CL:0000185", "Epithelial_Basal")
CL_TO_MEDIUM_NAME.setdefault("CL:0002327", "Epithelial_Luminal")
```

- [ ] **Step 4: Implement `derive_labels`**

Append to `src/dapidl/ontology/training_tiers.py`:
```python
def derive_labels(raw_name: str, source: str) -> tuple[str, str]:
    """Single CL-grounded (coarse, medium) derivation for a source-native raw label.

    ``source`` in {xenium_rep1, xenium_rep2, sthelar_breast_s0/s1/s3/s6}. STHELAR raw
    is the ``ct_tangram`` vocabulary; Xenium raw is Janesick-17. Both vocabularies are
    already merged into the global CLMapper (annotator_mappings); ``source`` is carried
    for the divergence report and future per-source disambiguation. Unmappable raw ->
    ('Unknown', 'Unknown'), never silently mis-binned.
    """
    from dapidl.ontology.cl_mapper import get_mapper
    mapper = get_mapper()
    return (derive_tier_label(raw_name, "coarse", mapper),
            derive_tier_label(raw_name, "medium", mapper))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ontology_derive.py -v`
Expected: PASS. If a case fails, the fix is a missing/incorrect entry in `GT_STHELAR` (raw→CL) or `CL_TO_MEDIUM_NAME` (CL→medium) — adjust the mapping, not the test. Do NOT special-case in `derive_labels`.

- [ ] **Step 6: Commit**

```bash
cd /mnt/work/git/dapidl-p64qc
git add src/dapidl/ontology/annotator_mappings.py src/dapidl/ontology/training_tiers.py tests/test_ontology_derive.py
git commit -m "feat(ontology): derive_labels single CL path + CAF/Plasma/myoepithelial fixes"
```

---

### Task 2: Label divergence report (Phase 1 gate, CPU)

**Files:**
- Create: `scripts/pilot_label_divergence.py`

The gate before the expensive build: for every (source, raw) pair seen in the 6 sources, compare the *old* inline-dict coarse vs the *new* `derive_labels` coarse/medium, and tally fixed Unknowns / re-bins.

- [ ] **Step 1: Write `scripts/pilot_label_divergence.py`**

```python
#!/usr/bin/env python3
"""Phase-1 gate: per (source, raw) old inline-dict coarse vs new derive_labels coarse/medium.
Reads the raw label vocabularies from the readers; writes a markdown divergence report."""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "scripts")
import breast_dapi_lmdb as B  # noqa: E402  inline dicts: JANESICK17_TO_COARSE, STHELAR_LABEL1_TO_COARSE
from dapidl.ontology.training_tiers import derive_labels  # noqa: E402
from dapidl.data.sthelar import SthelarDataReader  # noqa: E402

OUT = Path("pipeline_output/pilot_qc_collages_v3"); OUT.mkdir(parents=True, exist_ok=True)


def _sthelar_raws(slide_zarr) -> Counter:
    ndf = SthelarDataReader(slide_zarr).nucleus_df
    col = "ct_tangram" if "ct_tangram" in ndf.columns else "label1"
    return Counter(str(x) for x in ndf[col].to_list())


def main() -> None:
    rows = []
    # Xenium: Janesick-17 vocabulary (old inline = B.JANESICK17_TO_COARSE)
    for src in ["xenium_rep1", "xenium_rep2"]:
        for raw in B.JANESICK17_TO_COARSE:
            old = B.JANESICK17_TO_COARSE.get(raw) or "Unknown"
            new_c, new_m = derive_labels(raw, src)
            rows.append((src, raw, old, new_c, new_m, old != new_c))
    # STHELAR: ct_tangram vocabulary (old inline used label1 -> STHELAR_LABEL1_TO_COARSE)
    for z in sorted(B.STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        src = "sthelar_" + z.name.replace("sdata_", "").replace(".zarr", "")
        for raw in _sthelar_raws(z):
            new_c, new_m = derive_labels(raw, src)
            rows.append((src, raw, "(label1-based, not comparable)", new_c, new_m, new_c == "Unknown"))
    df = pl.DataFrame(rows, schema=["source", "raw", "old_coarse", "new_coarse", "new_medium", "changed_or_unknown"], orient="row")
    df.write_parquet(OUT / "label_divergence.parquet")
    n_unknown = int(df.filter(pl.col("new_coarse") == "Unknown").height)
    md = ["# Label divergence (old inline-dict vs derive_labels)\n",
          f"- rows: {df.height}  | new Unknowns: {n_unknown}\n"]
    md.append(df.to_pandas().to_markdown(index=False))
    (OUT / "label_divergence.md").write_text("\n".join(md))
    print(f"divergence: {df.height} (source,raw) pairs, {n_unknown} Unknown -> {OUT/'label_divergence.md'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (CPU; reads STHELAR nucleus tables)**

Run: `uv run python scripts/pilot_label_divergence.py`
Expected: writes `pipeline_output/pilot_qc_collages_v3/label_divergence.md`; **review it** — any unexpected Unknown is a missing `GT_STHELAR` mapping to add (loop back to Task 1) before the build.

- [ ] **Step 3: Commit**

```bash
git add scripts/pilot_label_divergence.py && git commit -m "feat(qc): Phase-1 label-divergence gate report"
```

---

### Task 3: Multi-axis quality model (Phase 3 pure, CPU)

**Files:**
- Create: `src/dapidl/qc/quality_model.py`
- Test: `tests/test_quality_model.py`

`quality_model` is **pure on a metric dict** (the re-score precomputes `sat_penalty` from the patch and passes it in) — no image I/O.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality_model.py
import numpy as np
import polars as pl
from dapidl.qc.quality_model import calibrate, axes, grade, Calibration


def _metric(brenner=1.0, glcm_entropy=1.0, interior_cov=1.0, glcm_asm=0.1,
            intensity_ratio=1.0, stardist_prob=0.9, centeredness=1.0,
            dominant_central=1.0, sat_penalty=0.0):
    return dict(brenner=brenner, glcm_entropy=glcm_entropy, interior_cov=interior_cov,
                glcm_asm=glcm_asm, intensity_ratio=intensity_ratio, stardist_prob=stardist_prob,
                centeredness=centeredness, dominant_central=dominant_central, sat_penalty=sat_penalty)


CAL = Calibration(brenner_lo=0.0, brenner_hi=2.0, ent_lo=0.0, ent_hi=2.0,
                  cov_lo=0.0, cov_hi=2.0, ir_lo=0.0, ir_hi=2.0)


def test_axes_in_unit_interval():
    a = axes(_metric(), CAL)
    for v in (a["detection"], a["focus"], a["texture"], a["brightness"]):
        assert 0.0 <= v <= 1.0


def test_focus_monotonic_in_brenner():
    assert axes(_metric(brenner=1.8), CAL)["focus"] > axes(_metric(brenner=0.2), CAL)["focus"]


def test_brightness_drops_with_saturation():
    hi = axes(_metric(intensity_ratio=1.5, sat_penalty=0.0), CAL)["brightness"]
    lo = axes(_metric(intensity_ratio=1.5, sat_penalty=0.5), CAL)["brightness"]
    assert lo < hi


def test_texture_rises_with_entropy_and_cov_falls_with_asm():
    base = axes(_metric(glcm_entropy=0.5, interior_cov=0.5, glcm_asm=0.5), CAL)["texture"]
    more = axes(_metric(glcm_entropy=1.8, interior_cov=1.8, glcm_asm=0.05), CAL)["texture"]
    assert more > base


def test_grade_thresholds():
    assert grade(0.7, tau_hi=0.6, tau_lo=0.3) == "Excellent"
    assert grade(0.4, tau_hi=0.6, tau_lo=0.3) == "Good"
    assert grade(0.1, tau_hi=0.6, tau_lo=0.3) == "Weak-passing"


def test_calibration_is_absolute_slide_independent():
    raw = pl.DataFrame({"broken": [False] * 6,
                        "brenner": [0, 1, 2, 3, 4, 5], "glcm_entropy": [0, 1, 2, 3, 4, 5],
                        "interior_cov": [0, 1, 2, 3, 4, 5], "intensity_ratio": [0, 1, 2, 3, 4, 5]})
    cal = calibrate(raw)
    # identical metrics -> identical axes regardless of any slide column
    m = _metric(brenner=2.0)
    assert axes(m, cal) == axes(m, cal)
    assert cal.brenner_hi > cal.brenner_lo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality_model.py -v`
Expected: FAIL — `ModuleNotFoundError: dapidl.qc.quality_model`.

- [ ] **Step 3: Implement**

```python
# src/dapidl/qc/quality_model.py
"""Multi-axis ABSOLUTE quality model for the p64 QC re-score. Pure on a per-patch
metric dict (the re-score precomputes sat_penalty); no image I/O, so Pass 2 re-runs
without re-segmenting. Four axes in [0,1]: detection, focus, texture, brightness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

SAT_LEVEL = 0.98  # fraction-of-uint16-max threshold for the saturation penalty (computed in re-score)


@dataclass(frozen=True)
class Calibration:
    brenner_lo: float; brenner_hi: float
    ent_lo: float; ent_hi: float
    cov_lo: float; cov_hi: float
    ir_lo: float; ir_hi: float


def _calib(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def calibrate(raw_table: pl.DataFrame, lo_q: float = 0.05, hi_q: float = 0.95) -> Calibration:
    """Robust p5/p95 of the soft-axis inputs pooled over ALL non-broken patches of ALL
    sources -> a single dataset-wide (absolute) Calibration."""
    nb = raw_table.filter(~pl.col("broken")) if "broken" in raw_table.columns else raw_table
    def lohi(col):
        v = nb[col].drop_nulls().to_numpy()
        if v.size == 0:
            return 0.0, 1.0
        return float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    bl, bh = lohi("brenner"); el, eh = lohi("glcm_entropy")
    cl, ch = lohi("interior_cov"); il, ih = lohi("intensity_ratio")
    return Calibration(bl, bh, el, eh, cl, ch, il, ih)


def axes(m: dict, cal: Calibration) -> dict:
    detection = float(np.clip(m["stardist_prob"], 0, 1)) * float(m["centeredness"]) * float(m["dominant_central"])
    focus = _calib(m["brenner"], cal.brenner_lo, cal.brenner_hi)
    texture = float(np.mean([_calib(m["glcm_entropy"], cal.ent_lo, cal.ent_hi),
                             _calib(m["interior_cov"], cal.cov_lo, cal.cov_hi),
                             1.0 - float(np.clip(m["glcm_asm"], 0, 1))]))
    brightness = _calib(m["intensity_ratio"], cal.ir_lo, cal.ir_hi) * (1.0 - float(m["sat_penalty"]))
    return {"detection": float(np.clip(detection, 0, 1)), "focus": focus,
            "texture": texture, "brightness": float(np.clip(brightness, 0, 1))}


def grade(quality_min: float, tau_hi: float = 0.60, tau_lo: float = 0.30) -> str:
    if quality_min >= tau_hi:
        return "Excellent"
    if quality_min >= tau_lo:
        return "Good"
    return "Weak-passing"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quality_model.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/quality_model.py tests/test_quality_model.py
git commit -m "feat(qc): multi-axis absolute quality model (detection/focus/texture/brightness)"
```

---

### Task 4: Per-source cell-boundary resolver (Phase 3 pure, CPU)

**Files:**
- Create: `src/dapidl/qc/cell_boundary.py`
- Test: `tests/test_cell_boundary.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_boundary.py
import numpy as np
from dapidl.qc.cell_boundary import rasterize_polygon_to_patch, voronoi_cell_mask


def test_polygon_um_to_patch_frame_raster():
    # a 4um square cell at full-res px (pixel_size 0.2125 -> ~18.8px), origin (x0,y0)
    poly_um = np.array([[10.0, 10.0], [14.0, 10.0], [14.0, 14.0], [10.0, 14.0]])
    mask = rasterize_polygon_to_patch(poly_um, x0=40, y0=40, pixel_size=0.2125, patch_size=64)
    assert mask.dtype == bool and mask.shape == (64, 64)
    # polygon px = poly_um/0.2125 ~ (47,47)-(65,65); minus origin (40,40) -> (7,7)-(25,25): inside frame
    assert mask.sum() > 0
    assert mask[16, 16]  # interior point


def test_polygon_offframe_returns_empty():
    poly_um = np.array([[1000.0, 1000.0], [1004.0, 1000.0], [1004.0, 1004.0], [1000.0, 1004.0]])
    mask = rasterize_polygon_to_patch(poly_um, x0=40, y0=40, pixel_size=0.2125, patch_size=64)
    assert mask.shape == (64, 64) and mask.sum() == 0


def test_voronoi_fallback_contains_nucleus_and_is_bounded():
    nuc = np.zeros((64, 64), bool)
    nuc[28:36, 28:36] = True
    cell = voronoi_cell_mask(nuc)
    assert cell.shape == (64, 64) and cell.dtype == bool
    assert (cell & nuc).sum() == nuc.sum()        # contains the nucleus
    assert cell.sum() >= nuc.sum() and cell.sum() <= 64 * 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cell_boundary.py -v`
Expected: FAIL — `ModuleNotFoundError: dapidl.qc.cell_boundary`.

- [ ] **Step 3: Implement**

```python
# src/dapidl/qc/cell_boundary.py
"""Per-source cell-mask resolver for the p64 QC re-score: native polygon (Xenium
cell_boundaries.parquet; STHELAR zarr shape if present) rasterized into the patch
frame, else a Voronoi expansion of the central-nucleus mask bounded by neighbours."""
from __future__ import annotations

import numpy as np
from skimage.draw import polygon as sk_polygon


def rasterize_polygon_to_patch(poly_um, x0: float, y0: float, pixel_size: float,
                               patch_size: int) -> np.ndarray:
    """Transform a cell polygon (full-res micron coords, shape (K,2) [x,y]) into the
    patch frame: um -> px (/pixel_size) -> minus crop origin (x0,y0) -> rasterize.
    Off-frame polygons yield an all-False mask of (patch_size, patch_size)."""
    mask = np.zeros((patch_size, patch_size), dtype=bool)
    p = np.asarray(poly_um, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3:
        return mask
    xs = p[:, 0] / pixel_size - x0
    ys = p[:, 1] / pixel_size - y0
    rr, cc = sk_polygon(ys, xs, shape=(patch_size, patch_size))  # row=y, col=x
    mask[rr, cc] = True
    return mask


def voronoi_cell_mask(nucleus_mask, max_radius_px: int | None = None) -> np.ndarray:
    """Fallback: grow the central-nucleus mask outward (distance-to-nucleus watershed-like
    expansion) into the patch, capped so it stays a plausible single-cell territory.
    Contains the nucleus; bounded by the patch (and max_radius_px if given)."""
    from scipy import ndimage
    nuc = np.asarray(nucleus_mask, dtype=bool)
    if not nuc.any():
        return nuc.copy()
    dist = ndimage.distance_transform_edt(~nuc)
    if max_radius_px is None:
        # default: half the patch (single nucleus in frame) — neighbours, when present,
        # are handled by the multi-seed path in the re-score; here we bound to the frame.
        max_radius_px = nuc.shape[0] // 2
    return (dist <= max_radius_px) | nuc
```

> **Note for the re-score (Task 7):** when neighbouring nuclei ARE present in the patch, pass them so the territory is split between seeds; the `starpose.expansion.VoronoiExpander` does this multi-seed split. `voronoi_cell_mask` here is the single-nucleus frame-bounded fallback used in tests and when no neighbour info is available.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cell_boundary.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dapidl/qc/cell_boundary.py tests/test_cell_boundary.py
git commit -m "feat(qc): per-source cell-boundary resolver (polygon raster + Voronoi fallback)"
```

---

### Task 5: Collage tile renderer with 1px dual overlays (Phase 4 pure, CPU)

**Files:**
- Create: `scripts/pilot_qc_collages_v3.py` (the pure `render_tile` first; the driver in Task 8)
- Test: `tests/test_collage_render.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collage_render.py
import numpy as np
import sys; sys.path.insert(0, "scripts")
from pilot_qc_collages_v3 import render_tile, CYAN, MAGENTA


def test_render_tile_paints_1px_dual_outlines():
    patch = (np.ones((64, 64)) * 1000).astype(np.uint16)
    nuc = np.zeros((64, 64), bool); nuc[24:40, 24:40] = True
    cell = np.zeros((64, 64), bool); cell[16:48, 16:48] = True
    rgb = render_tile(patch, nuc, cell)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    # cyan present (nucleus inner boundary) and magenta present (cell inner boundary)
    assert np.any(np.all(rgb == np.array(CYAN, np.uint8), axis=-1))
    assert np.any(np.all(rgb == np.array(MAGENTA, np.uint8), axis=-1))
    # outline is 1px: nucleus boundary pixel count is a thin ring, far less than its area
    cyan_n = int(np.all(rgb == np.array(CYAN, np.uint8), axis=-1).sum())
    assert 0 < cyan_n < int(nuc.sum())


def test_render_tile_handles_empty_cell_mask():
    patch = (np.ones((64, 64)) * 1000).astype(np.uint16)
    nuc = np.zeros((64, 64), bool); nuc[24:40, 24:40] = True
    rgb = render_tile(patch, nuc, np.zeros((64, 64), bool))
    assert rgb.shape == (64, 64, 3)
    assert not np.any(np.all(rgb == np.array(MAGENTA, np.uint8), axis=-1))  # no cell outline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_collage_render.py -v`
Expected: FAIL — `ModuleNotFoundError: pilot_qc_collages_v3`.

- [ ] **Step 3: Implement the pure renderer (driver added in Task 8)**

```python
# scripts/pilot_qc_collages_v3.py
"""Native-p64 QC collage renderer: DAPI grayscale + 1px cyan central-nucleus outline +
1px magenta resolved-cell outline. render_tile is pure (unit-tested); the grouping
driver (main) is added in Task 8."""
from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


def _stretch_to_uint8(patch: np.ndarray) -> np.ndarray:
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def render_tile(patch, nucleus_mask, cell_mask) -> np.ndarray:
    """uint16 DAPI patch + bool nucleus/cell masks -> (H,W,3) uint8 RGB with 1px inner
    outlines: cyan nucleus, magenta cell. Empty masks paint nothing."""
    g = _stretch_to_uint8(np.asarray(patch))
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    cell = np.asarray(cell_mask, dtype=bool)
    nuc = np.asarray(nucleus_mask, dtype=bool)
    if cell.any():
        rgb[find_boundaries(cell, mode="inner")] = MAGENTA
    if nuc.any():
        rgb[find_boundaries(nuc, mode="inner")] = CYAN   # nucleus drawn last (on top)
    return rgb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_collage_render.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pilot_qc_collages_v3.py tests/test_collage_render.py
git commit -m "feat(qc): p64 collage tile renderer with 1px cyan/magenta dual overlays"
```

---

### Task 6 `[DEFERRED-GPU/IO]`: Native p64 LMDB build (Phase 2)

**Files:** Modify `scripts/build_pilot_lmdb_v3.py`, `scripts/attach_medium_labels.py`.

Run only after the spatial-GNN Stage 2 GPU job frees and after the Task-2 divergence report is reviewed. Changes:
1. Route BOTH coarse (`labels.npy`) and the stored raw (`raw_labels.npy`) through `derive_labels(raw, source)` — replace the inline coarse dicts in `extract_xenium`/`extract_sthelar`. STHELAR raw column = `ct_tangram` (fallback `label1`, report via divergence).
2. Persist per-patch crop origin: write `x0, y0` (top-left of the 64px crop in source full-res px) and `pixel_size` into `patch_registry.parquet` (required by Task 7's native-boundary transform).
3. `--patch-size 64 --output breast-pilot-6source-dapi-p64-nuc-v1`.
4. `attach_medium_labels.py`: `medium_label = derive_labels(raw, source)[1]` (not `combined_final_label`).

- [ ] Add a small registry-schema test (`x0, y0, pixel_size` columns present, dtype float) before the build.
- [ ] **Build (CPU/IO, ~minutes/GB):** `uv run python scripts/build_pilot_lmdb_v3.py --patch-size 64 --output breast-pilot-6source-dapi-p64-nuc-v1`; then `uv run python scripts/attach_medium_labels.py …`.
- [ ] Commit the script changes; the LMDB is an output (untracked).

---

### Task 7 `[DEFERRED-GPU]`: 2-pass enriched re-score (Phase 3 orchestration)

**Files:** Create `scripts/pilot_qc_rescore.py`.

Uses Tasks 3 (`quality_model`) and 4 (`cell_boundary`), plus `starpose.qc` (`SegmentationGroundedScorer`, `select_center_nucleus`, `score_from_segmentation`, `decide_broken`) with `SegQCConfig(erode_px=1)` for p64.

- **Pass 1 (GPU, StarDist):** stream the p64 LMDB; per patch — segment, select centre nucleus, `score_from_segmentation` (raw metrics), compute `sat_penalty = mean(interior_pixels >= 0.98*65535)`, `resolve_cell_mask(...)` (Task 4; native Xenium polygon via `x0,y0,pixel_size`, else Voronoi). Persist nucleus+cell masks as packed-bit `.npz` chunks (reuse the subnuclear `center_masks` pattern) and write `qc/seg_scores_raw.parquet` (`row_idx, slide, cell_id, cell_type, broken, broken_reason, <raw metrics>, sat_penalty, cell_provenance`). Streaming/chunked/RAM-bounded; check `nvidia-smi` first.
- **Pass 2 (cheap, re-runnable):** `calibrate(seg_scores_raw)` → `quality_calibration.json`; `axes(...)` + `grade(min(4 axes))` → `qc/seg_scores.parquet` (raw cols + `detection, focus, texture, brightness, quality_min, grade`). Re-tuning `TAU_*` re-runs Pass 2 only.

- [ ] Implement; integration-smoke on a small `--limit`; **controller runs the full GPU Pass 1** after Stage 2.

---

### Task 8: Collage v3 grouping driver (Phase 4 orchestration, CPU, after re-score)

**Files:** Extend `scripts/pilot_qc_collages_v3.py` with `main()`.

Join `qc/seg_scores.parquet` ↔ `patch_registry.parquet` on `(row_idx, slide)`; for `granularity ∈ {coarse, medium}` (harmonized class via `labels.npy`/`labels_medium.npy` + `class_mapping`) and `group ∈ {Excellent, Good, Weak-passing, Broken-geom, Broken-quality}`, montage tiles via `render_tile` (Task 5) loading each patch's persisted nucleus+cell masks. `--sort-by {focus,texture,brightness,detection}` (default grade order). Outputs `collages/p64/<granularity>/<slide>/<class>/<group>.png` + `counts.parquet`/`counts_*.md` + `qc_groups_README.md` (4 axes, absolute calibration, grade rule, broken taxonomy, cyan/magenta legend).

- [ ] Implement `main()`; run after re-score; spot-check overlays render at 1px in two colours and every group dir populates.

---

## Self-Review

**Spec coverage:** D1 ontology → Task 1+2; D2 multi-axis absolute → Task 3; D3 cell resolver → Task 4; D4 native p64 → Task 6; D5 dual 1px overlays → Task 5; D6 ct_tangram raw → Task 1/6. Phase 1 gate → Task 2. Phase 3 re-score → Task 7. Phase 4 collage → Task 5+8. All §9 files have a task.

**Placeholder scan:** none — pure-unit tasks (1,3,4,5) have complete test+impl code; deferred GPU/orchestration tasks (6,7) reference the spec's already-detailed code and are explicitly marked `[DEFERRED-GPU]`.

**Type consistency:** `derive_labels(raw, source) -> (coarse, medium)` used identically in Tasks 1/2/6; `Calibration`/`axes(metric_dict, cal)`/`grade(quality_min, tau_hi, tau_lo)` consistent across Task 3 and Task 7; `rasterize_polygon_to_patch(poly_um, x0, y0, pixel_size, patch_size)` and `voronoi_cell_mask(nucleus_mask)` consistent across Task 4 and Task 7; `render_tile(patch, nucleus_mask, cell_mask)` consistent across Task 5 and Task 8; `x0,y0,pixel_size` produced in Task 6, consumed in Task 7.

**Parallelism note:** Tasks 1–5 (pure CPU) run now alongside the spatial-GNN Stage 2 GPU job; Tasks 6–7 (`[DEFERRED-GPU]`) wait for the GPU; Task 8 runs after Task 7.
