# p64 QC Collages — Multi-Axis Quality + Dual Segmentation Overlays + Ontology Harmonization

**Date:** 2026-06-09
**Status:** approved design, pre-implementation
**Supersedes (for the pilot QC review):** the stale `pipeline_output/pilot_qc_collages_v2/` (generated 2026-05-26/27, pre-dates the 2026-05-29 `decide_broken` revision).

## 1. Goal

Regenerate the pilot QC collages so they are (a) **native 64×64**, (b) graded by a **rethought multi-axis absolute quality model**, (c) annotated with **two 1px segmentation outlines** (central nucleus + cell) in distinct colours, and (d) grouped by **CL-ontology-harmonized coarse and medium** cell types that are consistent across the Xenium (`xenium_rep1/rep2`) and STHELAR (`sthelar_breast_s0/s1/s3/s6`) sources.

This is the human visual-gating artifact for the DAPI training pipeline; the design optimises for **visual coherence per group** and **per-axis interpretability**.

## 2. Decisions locked (from brainstorming)

| # | Decision | Choice |
|---|---|---|
| D1 | Ontology depth | **Fix labels properly** — route coarse+medium through the CL ontology; `labels.npy`/`labels_medium.npy` consistent across all sources. |
| D2 | Quality model | **Multi-axis absolute** — 4 axes (detection, focus, texture, **brightness**), dataset-wide absolute calibration. |
| D3 | Cell overlay source | **Per-source resolver** — native cell boundary where available (Xenium ✓), else starpose `VoronoiExpander`. |
| D4 | Patch size | **Native 64px** — build a 6-source p64 pilot LMDB + full re-score. |
| D5 | Overlays | **Two 1px segmentation outlines**: StarDist central nucleus = **cyan**, resolved cell = **magenta**; rasterized `find_boundaries`, nearest-neighbour compositing. |
| D6 | STHELAR raw label | **`ct_tangram`** (39-class descriptive, CL-mappable) — not `label1` (too coarse) nor `combined_final_label` (lumps Mast→Myeloid). |

## 3. Scope

**In scope:** ontology harmonization (mappings + a single derive path); native p64 pilot LMDB; the multi-axis absolute quality model; an enriched re-score that persists nucleus + cell masks; the v3 collage renderer with 1px dual overlays grouped by harmonized coarse/medium × grade.

**Out of scope (follow-on):** rebuilding the *production* `breast-6source-dapi-p128` training LMDB with the harmonized labels (the ontology fix in Phase 1 makes this a later one-liner re-run); retraining any model; the full-LMDB (non-pilot) QC pass.

**Confirmed data facts (from exploration):**
- Xenium `xenium-breast-tumor-rep1/rep2` ship **native** `cell_boundaries.parquet` + `nucleus_boundaries.parquet` (10x polygons; DAPI-only morphology, no membrane stain).
- STHELAR slides are SpatialData zarrs (`images/shapes/points/tables`); nucleus seg present; cell-level shape availability resolved at implementation, else Voronoi.
- starpose `VoronoiExpander` exists at `starpose.expansion.voronoi`.
- The 2026-05-29 high-specificity `decide_broken` (commit `26327a1`) is the current gate and is deliberately lenient on faint immune/pyknotic nuclei — **kept**.
- The pilot LMDB labels currently bypass the ontology via inline dicts in `scripts/breast_dapi_lmdb.py` / `scripts/build_pilot_lmdb_v3.py`.

## 4. Architecture — 4-phase pipeline

```
Phase 1  Ontology            src/dapidl/ontology/{annotator_mappings,training_tiers}.py
  derive_labels(raw, source) -> (coarse, medium)   [single CL-grounded path; STHELAR raw = ct_tangram]
        │
Phase 2  p64 LMDB            scripts/build_pilot_lmdb_v3.py --patch-size 64  (routed through derive_labels)
  breast-pilot-6source-dapi-p64-nuc-v1/  {patches.lmdb, labels.npy, labels_medium.npy,
                                          patch_registry.parquet(+x0,y0 origin), class_mapping.json}
        │
Phase 3  Quality + re-score  src/dapidl/qc/{quality_model,cell_boundary}.py + scripts/pilot_qc_rescore.py
  Pass 1 (GPU, StarDist): raw metrics + central-nucleus mask + resolved cell mask
        -> qc/seg_scores_raw.parquet, qc/masks/{nuc,cell}_chunk_*.npz
  Pass 2 (cheap): calibrate -> quality_calibration.json -> 4 axes + grade
        -> qc/seg_scores.parquet
        │
Phase 4  Collage v3          scripts/pilot_qc_collages_v3.py
  pipeline_output/pilot_qc_collages_v3/collages/p64/<coarse|medium>/<slide>/<class>/<grade>.png
  tiles = DAPI(RGB) + 1px cyan nucleus outline + 1px magenta cell outline
```

## 5. Phase 1 — Ontology harmonization

**Single source of truth:** `src/dapidl/ontology/training_tiers.py` (COARSE = {Epithelial, Immune, Stromal, Endothelial, Neural}; MEDIUM = 12 incl. Epithelial_Luminal/Basal, T_Cell, B_Cell, Macrophage, Dendritic_Cell, Mast_Cell, Fibroblast, Pericyte, Adipocyte, Endothelial, Neural).

**Fixes (each TDD):**
1. `annotator_mappings.py` `GT_STHELAR`: add `CAF → CL:0000057` (fibroblast), `Plasma → CL:0000786` (plasma cell), `Endothelial_Pericyte_Smooth_muscle → CL:0000669` (pericyte) [definitive assignment], and aliases for `Mammary_basal_cell_(=myoepithelial) → CL:0000185`.
2. `training_tiers.py` `CL_TO_MEDIUM_NAME`: add `CL:0000185` (myoepithelial) → `Epithelial_Basal`, `CL:0002327` (mammary luminal) → `Epithelial_Luminal`.
3. **`derive_labels(raw_name: str, source: str) -> tuple[str, str]`** — the single coarse+medium derivation, wrapping `annotator_mappings` + `cl_mapper` + `derive_tier_label`. Source dispatch picks the right per-source raw vocabulary (Janesick17 for `xenium_*`, **`ct_tangram`** for `sthelar_*`). Unmapped → `("Unknown","Unknown")` and is reported, never silently mis-binned.

**Cross-source consistency target (the test oracle):**

| Biology | Xenium raw | STHELAR raw (ct_tangram) | → coarse | → medium |
|---|---|---|---|---|
| B lineage | `B_Cells` | `B cell` | Immune | B_Cell |
| T lineage | `CD4+_T_Cells`,`CD8+_T_Cells` | `CD4 T cell`,`GZMK CD8 T cell` | Immune | T_Cell |
| Mast | `Mast_Cells` | `Mast cell` | Immune | Mast_Cell |
| Macrophage | `Macrophages_1/2` | `Macrophage`,`M1 macrophage` | Immune | Macrophage |
| Myoepithelial | `Myoepi_ACTA2+`,`Myoepi_KRT15+` | `CXCL14 mammary basal cell` | Epithelial | Epithelial_Basal |
| Luminal/tumor | `Invasive_Tumor`,`DCIS_1/2` | `SFN mammary luminal progenitor` | Epithelial | Epithelial_Luminal |
| Fibroblast | `Stromal` | `CXCL+ fibroblast`,`CAF` | Stromal | Fibroblast |
| Pericyte | `Perivascular-Like` | `Pericyte` | Stromal | Pericyte |
| Endothelial | `Endothelial` | `Venous EC`,`Arterial EC` | Endothelial | Endothelial |

**Validation gate (before Phase 2):** a divergence report `pipeline_output/pilot_qc_collages_v3/label_divergence.md` — every raw name (per source) → old (inline-dict) vs new (`derive_labels`) coarse/medium, with a count of fixed `Unknown`s and re-binned cells. Reviewed before the expensive build.

## 6. Phase 2 — Native p64 LMDB

`scripts/build_pilot_lmdb_v3.py`:
- `--patch-size 64` (already propagates via `half = patch_size//2`); `--output breast-pilot-6source-dapi-p64-nuc-v1`.
- `extract_xenium` / `extract_sthelar`: replace inline coarse-dict with `derive_labels(raw, source)` for BOTH coarse (→`labels.npy`) and store the raw string (→`raw_labels.npy`). STHELAR raw = `ct_tangram`.
- **Add patch global origin**: write `x0, y0` (top-left of the crop, in source full-resolution pixels) and `pixel_size` per patch into `patch_registry.parquet` — required by Phase 3's native-boundary transform.
- `scripts/attach_medium_labels.py`: derive `medium_label` via `derive_labels(...)[1]` (NOT `combined_final_label`).

Output: `breast-pilot-6source-dapi-p64-nuc-v1/{patches.lmdb, labels.npy, labels_medium.npy, raw_labels.npy, patch_registry.parquet(+x0,y0,pixel_size), class_mapping.json, slide_stats.json}`.

## 7. Phase 3 — Multi-axis quality model + enriched re-score

### 7.1 `src/dapidl/qc/quality_model.py` (NEW, pure, TDD)

Consumes the per-patch metric dict from starpose `score_from_segmentation` (`stardist_prob, centeredness, dominant_central, brenner, glcm_entropy, glcm_asm, interior_cov, intensity_ratio, edge_cut, area_um2, …`) plus a precomputed `sat_penalty` (added by the re-score, which has the patch) and a calibration object. `quality_model` is **pure on a metric dict** — no image I/O — so it is trivially testable and Pass 2 re-runs without re-segmenting.

**Hard broken gate (unchanged):** reuse starpose `decide_broken` → `broken_reason ∈ {no_nucleus, cut_at_edge, off_center, false_detection}`. Group map: `Broken-geom = {off_center, cut_at_edge}`, `Broken-quality = {false_detection, no_nucleus}`.

**Four absolute soft axes (non-broken only), each ∈ [0,1]:**
```
detection  = clip(stardist_prob,0,1) * centeredness * dominant_central          # already absolute
focus      = calib(brenner,         cal.brenner_lo,  cal.brenner_hi)            # noise-robust sharpness
texture    = mean( calib(glcm_entropy, cal.ent_lo, cal.ent_hi),
                   calib(interior_cov,  cal.cov_lo, cal.cov_hi),
                   1.0 - clip(glcm_asm,0,1) )                                   # chromatin content
brightness = calib(intensity_ratio, cal.ir_lo, cal.ir_hi) * (1 - sat_penalty)  # platform-invariant signal
   where sat_penalty = fraction of interior pixels >= SAT_LEVEL (0.98*uint16_max)
   and   calib(x,lo,hi) = clip((x-lo)/(hi-lo), 0, 1)
```

**Calibration (absolute, dataset-wide):** `calibrate(raw_table) -> Calibration` computes robust p5/p95 of `brenner, glcm_entropy, interior_cov, intensity_ratio` pooled across **all** non-broken patches of **all** sources; persisted to `quality_calibration.json`. This replaces the old per-slide P40/P80 — "Excellent" is now comparable across slides.

**Grade (non-broken), absolute thresholds (tunable, defaults):**
```
m = min(detection, focus, texture, brightness)
Excellent     if m >= TAU_HI (0.60)
Good          if m >= TAU_LO (0.30)
Weak-passing  otherwise
```
All four sub-scores + `m` + `grade` are written, so the collage can sort/group within a class by any single axis (e.g. lowest-focus passing immune).

### 7.2 `src/dapidl/qc/cell_boundary.py` (NEW)

`resolve_cell_mask(source, slide, cell_id, x0, y0, pixel_size, patch_size, nucleus_label_mask) -> (mask: np.ndarray[bool], provenance: str)`:
- **Xenium**: load (slide-cached) `cell_boundaries.parquet`; the cell's polygon (µm) → px (`÷pixel_size`) → subtract `(x0,y0)` → patch-frame polygon → rasterize (`skimage.draw.polygon`) to `patch_size²`. `provenance="native_xenium"`.
- **STHELAR**: if the zarr exposes a cell shape for `cell_id`, transform identically (`provenance="native_sthelar"`); else `VoronoiExpander` on `nucleus_label_mask` (`provenance="voronoi"`).
- **Fallback (any source, no native / off-frame)**: `VoronoiExpander(nucleus_label_mask)` bounded by neighbouring nuclei → `provenance="voronoi"`.
Provenance is written to the scores parquet so the collage / review can see which cells used expansion.

### 7.3 `scripts/pilot_qc_rescore.py` (NEW, orchestration)

- **Pass 1 (GPU):** stream the p64 LMDB → `SegmentationGroundedScorer._segment` (StarDist, `SegQCConfig(erode_px=1)` for p64) → `select_center_nucleus` → `score_from_segmentation` (raw metrics) → compute `sat_penalty` (fraction of interior pixels ≥ SAT_LEVEL) → `resolve_cell_mask`. Persist the central-nucleus mask + the cell mask as packed-bit `.npz` (reuse the subnuclear cache pattern), and write `qc/seg_scores_raw.parquet` (`row_idx, slide, cell_id, cell_type, broken, broken_reason, <raw metrics>, sat_penalty, cell_provenance`). Streaming, chunked, RAM-bounded.
- **Pass 2 (cheap, re-runnable):** `calibrate` over `seg_scores_raw.parquet` → `quality_calibration.json` → `quality_model` axes + grade → `qc/seg_scores.parquet` (raw cols + `detection, focus, texture, brightness, quality_min, grade`). Re-tuning `TAU_*` re-runs Pass 2 only (no re-segmentation).

## 8. Phase 4 — Collage v3

`scripts/pilot_qc_collages_v3.py` (NEW; leaves `pilot_qc_collages_v2.py` untouched):
- Native 64px (no centre-crop).
- Join `qc/seg_scores.parquet` ↔ `patch_registry.parquet` on `(row_idx, slide)`; group by **granularity ∈ {coarse, medium}** (harmonized class via `labels.npy`/`labels_medium.npy` + `class_mapping`) and **group ∈ {Excellent, Good, Weak-passing, Broken-geom, Broken-quality}** (absolute grade / broken_reason).
- **Tile render** (`render_tile`): RGB = grayscale DAPI (central-stretched) replicated to 3 channels; load the patch's persisted nucleus + cell masks; `nuc_b = find_boundaries(nuc_mask, mode="inner")`, `cell_b = find_boundaries(cell_mask, mode="inner")`; paint `RGB[nuc_b] = CYAN`, `RGB[cell_b] = MAGENTA`; `imshow(RGB, interpolation="nearest")`. Keep the small patch-ID label.
- `--sort-by {focus,texture,brightness,detection}` orders tiles within a group by one axis (default: grade order).
- Output: `pipeline_output/pilot_qc_collages_v3/collages/p64/<granularity>/<slide>/<class>/<group>.png` + `counts.parquet`/`counts_*.md` + a `qc_groups_README.md` documenting the 4 axes, the absolute calibration, the grade rule, the broken taxonomy, and the cyan/magenta overlay legend.

## 9. File structure

| File | Status | Responsibility |
|---|---|---|
| `src/dapidl/ontology/annotator_mappings.py` | modify | add CAF/Plasma/myoepithelial CL mappings to `GT_STHELAR` |
| `src/dapidl/ontology/training_tiers.py` | modify | add myoepithelial/mammary CL→MEDIUM; host `derive_labels` |
| `src/dapidl/ontology/derive.py` *(or in training_tiers)* | new | `derive_labels(raw, source) -> (coarse, medium)` single path |
| `scripts/build_pilot_lmdb_v3.py` | modify | `--patch-size 64`, route labels via `derive_labels`, store `x0,y0,pixel_size` |
| `scripts/attach_medium_labels.py` | modify | medium via `derive_labels`, not `combined_final_label` |
| `src/dapidl/qc/quality_model.py` | new (TDD) | 4 axes + absolute `calibrate` + `grade` |
| `src/dapidl/qc/cell_boundary.py` | new (TDD) | per-source cell-mask resolver (+ Voronoi fallback) |
| `scripts/pilot_qc_rescore.py` | new | 2-pass enriched re-score (masks + axes + grade) |
| `scripts/pilot_qc_collages_v3.py` | new | native-p64 collage with 1px dual overlays |
| `tests/test_ontology_derive.py` | new | cross-source consistency oracle |
| `tests/test_quality_model.py` | new | axis monotonicity, absolute calibration, grade |
| `tests/test_cell_boundary.py` | new | polygon transform, Voronoi fallback, provenance |

## 10. Testing

- **`derive_labels`** — the §5 oracle table: every (source, raw) pair → expected (coarse, medium); CAF/Plasma no longer Unknown; Myoepi → Epithelial_Basal; the 4 spellings of B/T collapse identically.
- **`quality_model`** — monotonicity (↑brenner→↑focus; ↑entropy/cov & ↓asm→↑texture; ↑intensity_ratio→↑brightness; saturation_penalty drops brightness; ↑prob/centered→↑detection); `calib` clipping; grade thresholds; **absolute**: identical metrics → identical axes regardless of slide.
- **`cell_boundary`** — a known µm polygon + origin → correct patch-frame raster; Voronoi fallback returns a mask that contains the nucleus and is bounded; provenance tags.
- **Integration smokes** — build p64 (small `--per-source`), re-score (small), collage (a few tiles): overlays render at 1px in two colours; every group dir populates; calibration json written.

## 11. Sequencing

1. **Phase 1 + validation** — implement `derive_labels` + fixes, pass the consistency tests, generate `label_divergence.md`, **review before building**.
2. **Phase 2** — build `breast-pilot-6source-dapi-p64-nuc-v1`.
3. **Phase 3** — `pilot_qc_rescore.py` Pass 1 (GPU) then Pass 2 (grade).
4. **Phase 4** — `pilot_qc_collages_v3.py`; spot-check overlays; re-push to `cms` if desired.

One design doc → one plan, executed in this order. Phase 1 is the cheap, high-value, independently-verifiable gate.

## 12. Risks & mitigations

- **`erode_px=3` too aggressive at 64px** → use `SegQCConfig(erode_px=1)` for the p64 re-score; document in the calibration meta.
- **Native polygon registration (µm→px, origin)** → unit-test the transform on a known cell; if a polygon lands off-frame, fall back to Voronoi (provenance flags it).
- **STHELAR cell shape may be absent** → resolver degrades to Voronoi per cell; no hard dependency.
- **`ct_tangram` missing for some STHELAR cells** → fall back to `label1`→coarse only, medium=`Unknown` (reported in divergence).
- **1px outline subtle at small tile size** → intentional (precise); `--sort-by` + zoom for review; line width is a one-line change if more visibility is wanted later.
- **Absolute grade thresholds need tuning** → Pass 2 is cheap and re-runnable without re-segmenting; tune `TAU_*` after the first visual pass.

## 13. Outputs

`pipeline_output/pilot_qc_collages_v3/`: `collages/p64/<coarse|medium>/<slide>/<class>/<grade>.png`, `counts.parquet`, `counts_*.md`, `qc_groups_README.md`, `label_divergence.md`.
`breast-pilot-6source-dapi-p64-nuc-v1/`: the LMDB + harmonized labels + registry.
`qc/`: `seg_scores_raw.parquet`, `quality_calibration.json`, `seg_scores.parquet`, `masks/{nuc,cell}_chunk_*.npz`.
