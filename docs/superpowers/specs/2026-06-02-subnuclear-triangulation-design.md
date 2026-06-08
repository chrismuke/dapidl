# Subnuclear-Structure Triangulation — Design Spec

**Date:** 2026-06-02
**Branch:** `feat/nucleus-qc-scorer` (current)
**Status:** approved design, pre-implementation

## 1. Goal

Quantify **how much inner subnuclear DAPI structure** (chromatin texture, heterochromatin
clumping, nucleoli — the high-frequency intensity field *inside* the nucleus) contributes to
cell-type classification by the best model, using two **no-retrain** readouts. This is a *cheap
first read* to decide whether to commit GPU to a full retrain ablation matrix (phase A, deferred).

The best model is EfficientNet-V2-S at coarse 4-class (Endothelial/Epithelial/Immune/Stromal),
test macro-F1 **0.619** on held-out `xenium_rep2` (see `project_backbone_h2h_2026_05`).

### Why (motivation)

A 128 px patch (27 µm at 0.2125 µm/px) carries four separable cue families: (1) inner subnuclear
structure, (2) nuclear silhouette (size/shape), (3) integrated intensity (DNA content), (4) context
(neighbours/architecture). Our patch-size sweep showed context is a large contributor, and NuSPIRe
(a nucleus-only foundation model) lost to EfficientNet — both suggest inner structure may be a
*modest* contributor. This experiment puts numbers on it.

## 2. Scope

**In scope (this spec):** two no-retrain readouts on the existing checkpoint —
- **(C) Information floor:** can a tree on simple per-nucleus features match 0.619?
- **(D) Attribution:** does the model concentrate evidence inside the nucleus, beyond what its
  area alone would predict?

Both are fed by **one shared StarDist pass** over the LMDB.

**Out of scope (phase A, deferred):** any retraining; the masking/flattening conditions
(nucleus-flattened, nucleus-only, context-only, pixel-shuffle); the resolution/frequency sweep.
The StarDist mask cache produced here is designed to be reused by phase A.

## 3. Inputs / assets

| Asset | Path |
|---|---|
| Patch LMDB (2.28 M patches, uint16 128×128) | `/mnt/work/datasets/derived/breast-6source-dapi-p128` |
| Labels (coarse), sources | `labels.npy`, `sources.npy` in the LMDB dir |
| Best checkpoint | `pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/best_model.pt` |
| Model class | `DapiClassifier(4, backbone="efficientnetv2_rw_s")` (scripts/breast_pooled_train.py) |
| StarDist runner | `dapidl.qc.segmentation_grounded.SegmentationGroundedScorer._segment` + `select_center_nucleus` |
| Feature helpers | `segmentation_grounded`: `structure_raw`, `area_um2`, `objectness_metrics`, `interior_cov`, `brenner_focus`, `regionprops` |

Train sources: `xenium_rep1, sthelar_breast_s0, sthelar_breast_s1, sthelar_breast_s3,
sthelar_breast_s6`. Test source: `xenium_rep2`. Cells with label `-1` are dropped (matches
training). EfficientNet preprocessing (must be reproduced exactly for IG): `img/65535 → (img −
0.485)/0.229 → model expands 1→3 ch`.

## 4. Architecture

**One StarDist pass, two readouts.**

```
LMDB patch (uint16 128²) ──_segment──► label mask + per-object prob ──select_center_nucleus──► center nucleus mask
                                                          │
                  ┌───────────────────────────────────────┴────────────────────────────────────┐
        (C) nucleus_feature_vector(patch, mask, prob)                       (D) cached mask for the saliency subset
        geometry + intensity + Haralick  → seg_features.parquet            fraction of |IG attribution| ÷ area-fraction
                                                                            (on the EfficientNet checkpoint)
```

The feature pass **streams** (chunked parquet writes; never holds all patches/masks in RAM —
62 GB host). Masks are persisted compactly only for the saliency subset and (optionally) full
set as a packed-bit cache that phase A reuses.

## 5. Components

Small, single-purpose, independently testable units.

### 5.1 `src/dapidl/qc/patch_features.py` *(new, TDD)*
Pure functions; no I/O, no GPU.
- `nucleus_feature_vector(patch, mask, prob, cfg, pixel_size) -> dict[str, float]`
  Reuses `segmentation_grounded` for geometry/intensity/focus; adds **Haralick** via skimage
  `graycomatrix`/`graycoprops`. Emits **two scopes** (prefixed keys):
  - `nuc_*` — computed inside the center-nucleus mask only (interior subnuclear signal).
  - `ctx_*` — computed over the whole 128² patch (adds context/architecture).
- Helper `haralick_features(gray_u8, mask=None) -> dict` (contrast, homogeneity, energy,
  correlation, ASM; averaged over 4 directions, distance 1). `gray_u8` = the percentile-`[1,
  99.8]`-normalized patch (same normalization as `_segment`) quantized to **16 levels**; the GLCM
  is restricted to the mask region for `nuc_*` and the whole patch for `ctx_*`.
- Returns NaN-filled vector with `has_nucleus=0` when `select_center_nucleus` is None (LightGBM
  handles NaN natively).

**Feature list (per scope):** area_um2, eccentricity, solidity, extent, major/minor axis,
intensity mean/std/p10/p50/p90, intensity-above-background, Laplacian-variance (`structure_raw`),
Brenner focus, Haralick {contrast, homogeneity, energy, correlation, ASM}. `nuc_` scope only:
`stardist_prob`. Plus `nuc_area_fraction` (= mask area ÷ patch area), used by (D).

### 5.2 `src/dapidl/qc/attribution.py` *(new, TDD)*
- `integrated_gradients(model, x, target, baseline=None, steps=50) -> Tensor`
  Custom IG (no captum). `baseline` defaults to zeros in normalized space (black). Sums
  gradients over `steps` interpolations × (x − baseline). The model input is **1-channel**
  (B,1,128,128) — `DapiClassifier` expands 1→3 ch internally — so attribution is a single H×W
  map (no channel reduction needed). **Unit test:** on a single `nn.Linear`, IG completeness
  holds exactly (Σ attr = f(x) − f(baseline)).
- `fraction_in_mask(attr, mask) -> float` = Σ|attr inside mask| ÷ Σ|attr|.
- `attribution_concentration(attr, mask, area_fraction) -> float` = `fraction_in_mask /
  max(area_fraction, eps)` — the **headline (D) metric** (≈1 indifferent, ≫1 concentrates on
  subnuclear content, <1 looks at context).

### 5.3 `scripts/subnuclear_feature_pass.py` *(new, orchestration)*
Streams the LMDB → `_segment` → `select_center_nucleus` → `nucleus_feature_vector`. Writes
`pipeline_output/subnuclear_2026_06/seg_features.parquet` with columns: `global_idx, source,
label, has_nucleus, <nuc_*>, <ctx_*>`. Chunked writes (e.g. every 50 k rows via pyarrow
incremental writer). Also writes a packed-bit **center-mask cache**
(`center_masks/chunk_*.npz`) for reuse by (D) and phase A. **Throughput gate:** a `--limit N`
smoke (N=500) prints patches/sec and projected full-pass ETA *before* the full run is launched.

### 5.4 `scripts/subnuclear_floor.py` *(new, orchestration)*
Loads `seg_features.parquet`; trains **LightGBM** (multiclass, balanced class weights to match
the imbalance) on train-source rows with a small seeded held-out slice of the train sources for
**early stopping** (so n_estimators isn't a magic number), tests on **all rep2 rows**. Two
models: `nuc_*`-only and `nuc_*`+`ctx_*`. Writes `floor_metrics.json` (macro-F1, per-class F1,
vs 0.619) + LightGBM feature importances. polars throughout; `.to_pandas()` only at the LightGBM
boundary.

### 5.5 `scripts/subnuclear_saliency.py` *(new, orchestration)*
Loads the checkpoint into `DapiClassifier(4)`, eval mode. Takes a **balanced ~3 k rep2 subset**
(≈750/class, seeded), reuses cached masks (or re-segments those 3 k). For each: IG → predicted
class → `fraction_in_mask`, `attribution_concentration`. Writes `saliency_summary.json`:
mean ± IQR of concentration overall, per class, and split by correct/incorrect. Also dumps ~12
example overlays (`overlays/*.png`: patch + nucleus contour + IG heatmap) for the figure.

### 5.6 `pipeline_output/subnuclear_2026_06/run_triangulation.sh` *(new, driver)*
Sequences: feature-pass smoke (500) → [human ETA check] → full feature pass → floor → saliency →
prints the combined readout table + decision rubric.

### 5.7 Tests
- `tests/test_patch_features.py` — synthetic patches: a textured disc vs a flat disc →
  `nuc_structure_raw`/Haralick-contrast higher on textured; `has_nucleus=0` path; area_fraction
  correctness; NaN on empty mask.
- `tests/test_attribution.py` — IG completeness on `nn.Linear`; `fraction_in_mask` on a hand-set
  attribution+mask; `attribution_concentration` = fraction/area on a known case.

## 6. Outputs

`pipeline_output/subnuclear_2026_06/`:
- `seg_features.parquet` (reusable: ≈ a seg_scores table over the whole LMDB)
- `center_masks/chunk_*.npz` (packed-bit masks, reused by phase A)
- `floor_metrics.json`, `saliency_summary.json`, `overlays/*.png`

## 7. Decision rubric (what we conclude)

| Readout | Reads | Interpretation |
|---|---|---|
| (C) nuc-only floor vs 0.619 | macro-F1 gap | small gap → CNN's spatial subnuclear modelling adds little; large gap → it has value |
| (C) +context vs nuc-only | macro-F1 lift | how much context adds on top of nucleus summary stats |
| (D) concentration ratio | ÷ area-fraction | ≈1 nucleus ignored; ≫1 subnuclear-driven; <1 context-driven |

Combined → **greenlight or skip** the phase-A retrain matrix, with a per-class view (e.g. immune
may be subnuclear-driven while stromal is context-driven).

## 8. Constraints & decisions

- **No retrain** in this phase. **Full scope:** full rep2 test + full 1.5 M-train feature pass
  (user choice — yields a reusable feature/mask substrate for phase A; the throughput gate
  reports ETA before committing).
- **RAM:** streaming, chunked writes; masks not held in RAM beyond a chunk. Est. peak well under
  62 GB.
- **New dependency:** `lightgbm` (`uv add lightgbm`). Attribution is custom (no captum).
- **Standards:** `uv run`; polars-first (pandas only at LightGBM/skimage boundary); TDD on the two
  `src/dapidl/qc/` modules; feature branch only; do **not** touch instance-seg/starpose WIP.
- **StarDist consistency:** reuse `_segment` (percentile [1, 99.8] norm) so segmentation matches
  the QC scorer exactly.

## 9. Risks & mitigations

- **StarDist throughput unknown** → 500-patch smoke prints ETA before the full pass; if the full
  1.5 M pass is impractical, fall back to a stratified train sample (floor trees saturate well
  before 1.5 M; rep2 test stays full for comparability).
- **IG preprocessing mismatch** would invalidate (D) → IG reuses the exact training transform
  (÷65535 → DAPI-norm → 1→3 ch expand); unit-tested baseline/target handling.
- **StarDist misses a nucleus** (`has_nucleus=0`) → row kept with NaN features (LightGBM-native);
  reported as a coverage stat; excluded from (D).
- **Area-fraction confound** in (D) → headline metric is concentration = fraction ÷ area-fraction,
  not raw fraction.
