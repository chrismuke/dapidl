# "Good Nucleus Patches" v1 — Design Spec

**Date:** 2026-05-25
**Status:** Approved framing, spec for user review.
**Supersedes:** the narrow `2026-05-25-nucleus-recentering-design.md` (file renamed; this unified spec replaces it).
**Branch (parent for new work):** `main` (the `feat/nucleus-qc-scorer` branch is parked at `3174ae1` with the scorer code; will rebase or branch off main).
**Codex review:** background critique landed (see §10 for what it changed).

---

## 1. Goal

Produce a "good nucleus patches" training LMDB for the 6-source breast pool, where every patch:
- is **centered on the nucleus** (not the cell)
- shows **clear subnuclear chromatin structure** (in focus, not a grey blob, not an apical/basal Z-cap)
- has a **validated cell↔nucleus pairing** (the label attached to the patch belongs to the cell whose nucleus is shown)

Then measure each effect — centering, then filtering — on the cross-source classifier with a paired-attribution A/B/C re-train protocol.

## 2. Why now / problem

Three independent issues compound today:

1. **Cell-centered patches.** Xenium patches in `breast-6source-dapi-p128` are centered on `cells.parquet` cell centroids (whole-cell, includes cytoplasm), not nuclei. Measured offset: median ~1.3 µm, ~25 % > 2.7 µm, ~11 % > 4.8 µm. STHELAR is already nucleus-centered (verified, 0.000 µm offset). The current model trains on slightly off-center Xenium nuclei.

2. **No image-quality filter.** Some patches show only an apical/basal Z-cap of the nucleus (flat smooth interior, no chromatin) or a focus failure. These are mislabeled by the morphology premise of the model — "predict cell type from a centered nucleus crop" requires the nucleus to actually be visible. We built a `SegmentationGroundedScorer` (parked on `feat/nucleus-qc-scorer` at `3174ae1`) that scores patches on `structure_score`, `centeredness`, `objectness` and flags 5 broken-patch reasons — but **never validated it visually on real data** after the recentering pivot.

3. **Pairing not audited.** STHELAR ships two independent segmenters' outputs (`table_nuclei` from DAPI, `table_cells` from H&E/CellViT). We use `table_nuclei` centroids + `table_nuclei` labels but never check the nucleus polygon for cell `X` actually sits inside the cell polygon for cell `X`. Xenium is keyed by `cell_id` on both tables so it's pairing-correct by construction, but a one-time *audit* is worth doing because labels are load-bearing.

(2) and (1) compound: smaller patch sizes (p32, p64) underperform p128 in part because they clip systematically off-center nuclei AND any clipped flat-cap patches retain spurious labels.

## 3. Non-goals (explicit)

- **Re-segmenting nuclei** for sources that lack native nucleus segmentation (MERSCOPE). This is **spec C**, future. starpose already ships `instanseg`, `cellpose`, `stardist`, `cellvit`, `cellotype`, `joint` dispatchers.
- **Per-cell Z-plane picking** from Xenium's 17-plane `morphology.ome.tif`. This is **spec D**, future. It is the only "real" fix for apical/basal caps (vs detect-and-drop); we choose detect-and-drop here.
- Changing patch size, model backbone, optimizer, augmentations, label vocabulary, or train/val/test split logic. Only the LMDB pixels and (in one model) which `cell_id`s appear change.
- Generalizing to other Xenium datasets beyond rep1/rep2.

## 4. Approach: build → audit → score → filter → re-train

```
existing LMDB (cell-centered)              new LMDB (nuc-centered, unfiltered)        new LMDB (nuc-centered, filtered)
  breast-6source-dapi-p128         ──▶     breast-6source-dapi-p128-nuc       ──▶    breast-6source-dapi-p128-nuc-filt
       │                                          │                                          │
       │  re-extract Xenium on nucleus            │  run parked SegmentationGroundedScorer   │
       │    polygon centroid;                     │  visual-validate via ladders +           │
       │  byte-copy STHELAR rows                  │    per-reason montages;                  │
       │                                          │  drop broken patches                     │
       ▼                                          ▼                                          ▼
   M_cell                                      M_nuc_full                                M_nuc_filt
   (train on cell-centered)                    (train on nuc-centered, all rows)         (train on nuc-centered, filtered)
       │                                          │                                          │
       └───── per-source A/B vs M_nuc_full ──────┘                                          │
                       (centering effect)                                                   │
                                                  └─── per-source A/B on shared UNFILTERED ─┘
                                                       nuc-centered test set
                                                       (filtering effect)
```

### 4.1 Component 1 — Native re-centering

**New method on `XeniumDataReader`:**

```python
def get_nucleus_centroids_pixels(self) -> np.ndarray:
    """Per-cell nucleus polygon centroid (mean of vertices), in pixels.
    Returns (N, 2) in cell_id order matching get_centroids_pixels(),
    so existing labels and splits carry over unchanged.
    Cells missing a nucleus polygon fall back to the cell centroid
    (so the LMDB cell_id set is preserved → A/B remains paired).
    """
    nb = pl.read_parquet(self._get_outs_path() / "nucleus_boundaries.parquet")
    centroids = (nb.group_by("cell_id")
                   .agg(pl.col("vertex_x").mean().alias("xc"),
                        pl.col("vertex_y").mean().alias("yc")))
    df = self.cells_df.join(centroids, on="cell_id", how="left").with_columns(
        pl.col("xc").fill_null(pl.col("x_centroid")),
        pl.col("yc").fill_null(pl.col("y_centroid")),
    )
    return np.column_stack([
        df["xc"].to_numpy() / self.PIXEL_SIZE,
        df["yc"].to_numpy() / self.PIXEL_SIZE,
    ])
```

`PIXEL_SIZE = 0.2125` µm/px (existing constant).

**LMDB build path** (`scripts/breast_dapi_lmdb.py --nucleus-centered`):
- **Xenium rep1/rep2** → re-extract from `morphology_focus.ome.tif` via `tifffile.memmap`, using `get_nucleus_centroids_pixels()` instead of `get_centroids_pixels()`.
- **STHELAR s0/s1/s3/s6** → **byte-copy LMDB rows by cell_id** from the existing `breast-6source-dapi-p128` (already nucleus-centered, offset 0.000 µm verified). No decode/re-encode, no zarr re-read. Removes the RAM-risky step.
- Metadata: identical `metadata.parquet` schema. Add `centroid_source` and `fallback_count` per source to `metadata.json`.

### 4.2 Component 2 — Pairing audit (`scripts/pairing_audit.py`)

One script, two source families, one summary table.

**Xenium (rep1, rep2):** for each `cell_id` in `cells.parquet`, check the nucleus polygon (from `nucleus_boundaries.parquet`) sits inside the cell polygon (from `cell_boundaries.parquet`). Report:
- % of cells with a nucleus polygon at all
- % where nucleus centroid is inside cell polygon
- % where nucleus area < cell area (sanity)
- count of nucleus polygons extending outside cell polygon

**STHELAR (s0, s1, s3, s6):** for each `cell_id` in `table_nuclei`, check the nucleus polygon (from `shapes/nucleus_boundaries`) sits inside the cell polygon (from `shapes/cell_boundaries`) for the SAME `cell_id`. Same metrics.

**Pass criteria:** for all 6 sources, ≥ 90 % of cells have a paired nucleus polygon AND ≥ 95 % of paired nuclei satisfy centroid-in-cell-polygon. **Fail → investigate before building anything.**

Output: `pipeline_output/pairing_audit/audit_summary.parquet` + a one-page markdown summary.

### 4.3 Component 3 — QC scoring

Run `dapidl.pipeline.steps.quality_control_seg.run_quality_control_seg()` (already built, T7 of `feat/nucleus-qc-scorer`) on the new nuc-centered LMDB. Produces `qc/seg_scores.parquet` + `qc/seg_broken_audit.parquet`.

**No code change** — just run the existing entry point.

### 4.4 Component 4 — Visual validation + threshold selection (user-in-the-loop gate)

Generate ladders (`scripts/qc_validation_montage.py --metric structure_score`, `--metric objectness_score`, `--metric centeredness`) for the nuc-centered LMDB, plus per-reason montages from `scripts/seg_qc_smoke.py`.

**User reviews.** If the ladders show that:
- `structure_score` cleanly distinguishes sharp chromatin from flat caps → accept defaults
- `centeredness` cleanly reflects nucleus framing → accept defaults
- a specific reason flags too aggressively / too leniently → tune `SegQCConfig` for that reason only (one knob at a time, document the change)

**Filtering build path** (`scripts/breast_dapi_lmdb.py --filter qc/seg_scores.parquet --filter-broken`):
- Drop rows where `seg_scores.broken == True`
- Output: `breast-6source-dapi-p128-nuc-filt/`
- Log per-source and per-class drop rates in `metadata.json`

### 4.5 Component 5 — Three-model A/B/C re-train + paired evaluation

Three runs with `scripts/breast_pooled_train.py`, **same seed across runs**, `--patience 5`, only `--dataset` differs:

| model | training LMDB | tagged in ClearML |
|---|---|---|
| **M_cell** | `breast-6source-dapi-p128` (existing) | `gnp-v1/cell` |
| **M_nuc_full** | `breast-6source-dapi-p128-nuc` (new) | `gnp-v1/nuc-full` |
| **M_nuc_filt** | `breast-6source-dapi-p128-nuc-filt` (new) | `gnp-v1/nuc-filt` |

**Evaluation protocol** (cohort-drift mitigation, per codex critique):

| comparison | measures | both models tested on |
|---|---|---|
| M_cell vs M_nuc_full | **centering effect** | matched test sets (each on its own framing, paired by `cell_id`) |
| M_nuc_full vs M_nuc_filt | **filtering effect** | the **same unfiltered nuc-centered test set** (test split of `breast-6source-dapi-p128-nuc`) — isolates model improvement from cohort drift |
| M_cell vs M_nuc_filt | **total effect** | each on matched test set; both geometry and cohort changed (reported but not used for decision) |

Each comparison reports macro F1 **per source AND per class** in addition to pooled. Pooled is informational, not decisional.

### 4.6 Decision criteria

| outcome | conclusion | action |
|---|---|---|
| **M_nuc_full > M_cell by > 2 % Xenium-only macro F1, with STHELAR delta < 0.5 %** | centering is a real win; STHELAR placebo confirms low noise floor | nucleus-centering becomes default for Xenium |
| **M_nuc_filt > M_nuc_full by > 1 % on the shared unfiltered test set** | filtering improves the model independent of test cohort | filtering becomes default in build |
| **filtering effect concentrated on minority classes (Stromal, Endothelial)** | filtering helps where label noise hurts most | extra confidence — keep |
| **either delta ≈ 0 within STHELAR noise floor** | that piece is geometrically/QC-correct but classifier doesn't need it | ship anyway; principled extraction, no urgency |
| **either delta < 0 by > 1 %** | unexpected regression | investigate before proceeding |

"Xenium-only macro F1" = macro-averaged F1 across classes, computed on test rows with `source ∈ {rep1, rep2}`. "STHELAR noise floor" = absolute |B−A| on the analogous STHELAR-only macro F1, which should be near zero since STHELAR patches are byte-identical between M_cell and M_nuc_full.

## 5. Verification gates (pre-training, fail-fast)

| gate | check | pass criteria |
|---|---|---|
| **Recentering audit** | join old + new patch indices on cell_id, compute Δ µm per Xenium cell | median ∈ [0.5, 5] µm; fallback rate < 10 %; STHELAR Δ = 0.000 µm |
| **Pairing audit** | nucleus polygon ⊂ cell polygon per source | ≥ 90 % paired, ≥ 95 % centroid-in-polygon, all 6 sources |
| **QC sanity** | broken-rate per source from `seg_broken_audit.parquet` | < 20 % per source; no source > 35 %; if any source > 35 %, investigate before filtering |
| **Visual validation** | user reviews ladders + per-reason montages | user explicit OK or threshold tweak |

If any gate fails → fix before progressing. Training runs only start after all four gates pass.

## 6. Components & file map

| file | change |
|---|---|
| `src/dapidl/data/xenium.py` | **+ method** `get_nucleus_centroids_pixels()` |
| `scripts/breast_dapi_lmdb.py` | **+ flag** `--nucleus-centered` (Xenium re-extract + STHELAR byte-copy); **+ flag** `--filter <scores.parquet> --filter-broken` (drop flagged rows) |
| `scripts/pairing_audit.py` | **new** — Xenium + STHELAR pairing audit, writes summary parquet + markdown |
| `scripts/recentering_audit.py` | **new** — paired Δ µm histogram with pass gates |
| `tests/data/test_xenium_nucleus_centroids.py` | **new** — synthetic polygon → known centroid, ID-order preservation, missing-polygon fallback |
| `scripts/breast_pooled_train.py` | **no code change** — used as-is, just three invocations |
| `qc/seg_scores.parquet` etc. | **no new code** — run existing `quality_control_seg.run_quality_control_seg` |

**No changes** to: `data/patches.py`, `data/dataset.py`, the model, the training loop, the QC scorer (already built and tested), or any STHELAR data path.

## 7. Output deliverables

1. **`pipeline_output/pairing_audit/`** — audit summary + markdown report
2. **`~/datasets/derived/breast-6source-dapi-p128-nuc/`** — nucleus-centered LMDB
3. **`~/datasets/derived/breast-6source-dapi-p128-nuc/qc/`** — seg_scores.parquet, seg_broken_audit.parquet, ladders, per-reason montages
4. **`~/datasets/derived/breast-6source-dapi-p128-nuc-filt/`** — filtered LMDB
5. **3 ClearML runs** tagged `gnp-v1/{cell,nuc-full,nuc-filt}`, plus a one-page `gnp_v1_readout.md` with the per-source × per-class A/B/C tables and the decision call.

## 8. Budget

- Pairing audit: ~30 min
- Re-extract Xenium + byte-copy STHELAR: ~2 h (one-time, mostly Xenium tile reads)
- QC scoring on new LMDB: ~3 h (full pass, ~2.28 M patches)
- Visual review: user-in-the-loop, ~30 min
- Filter rebuild: ~30 min
- Three training runs sequentially on 3090: ~3 × 14 h ≈ 1.75 days
- Readout + writeup: ~1 h

**Total ~2.5 days end-to-end, dominated by training.**

## 9. Risks & mitigations

| risk | mitigation |
|---|---|
| Cohort drift confounds filtering A/B (codex #2) | filtering A/B evaluates both models on the *same* unfiltered nuc-centered test set |
| STHELAR `table_nuclei` vs `table_cells` pairing error invisible (codex #4) | pairing audit (4.2) checks both sources, gate is fail-fast |
| Pixel-size / unit mistake in centroid computation | unit test on synthetic polygon with known answer using `PIXEL_SIZE = 0.2125` |
| QC `structure_score` doesn't actually distinguish in-focus from flat caps | visual validation gate (4.4) — if ladders don't show clean gradient, tune or drop the metric before filtering |
| Filtering removes too much (broken-rate > 35 %) and biases training set | broken-rate sanity gate; per-source/per-class drop rate logged in `metadata.json` |
| Cells with no nucleus polygon | fall back to cell centroid; A/B paired by `cell_id`; fallback rate logged |
| Apical/basal cap patches sneak through QC | detect-and-drop is the only fix here (spec D would prevent at source); structure_score MAD-zero rule and intensity ratio together should catch most |
| All three training runs use same seed → reproducibility tied to single seed | document seed; one optional repeat-seed run on the winner to confirm |

## 10. What codex's review changed

Codex's adversarial critique (background dispatch, 2026-05-25) flagged five concerns; four were folded in:
1. **Pairing audit added (§4.2)** — was missing, treated as a C-only problem; codex right that STHELAR ships two segmenters' outputs and we never verified consistency.
2. **Cohort-drift mitigation (§4.5)** — filtering A/B evaluation uses the *same* unfiltered test set for both models, not their respective training distributions.
3. **Z-plane picking kept separate as spec D** — codex right that mixing image-source change with QC validation confounds both.
4. **Per-source × per-class always** — promoted to decision rule; pooled is informational.
5. (Single-seed concern) — addressed via one optional repeat-seed run on the winner, not by retraining everything multiple times.

## 11. Out of scope (explicit, named for traceability)

- **Spec C — Re-segmentation + cell-mapping for MERSCOPE.** Will use starpose's `instanseg.single_channel_nuclei` (Goldsborough 2024, channel-invariant, faster than StarDist on DAPI). Pairing policy: Caicedo 2019 / Sopa / 10x XOA convention — IoU ≥ 0.5 → 1:1; else centroid-in-polygon → assign max-IoU; multinucleate flag/drop; orphan exclude from supervised training.
- **Spec D — Per-cell Z-plane picking.** Use Xenium's 17-plane `morphology.ome.tif` to pick the in-focus Z slice per cell (axial focus picking). Prerequisite: registration check between the Z-stack and `morphology_focus.ome.tif`. Only spec D *prevents* apical/basal caps; we choose detect-and-drop here.
- **Validation against Yang 2018 Google focus oracle.** Optional bonus — once we have `structure_score` per patch on real data, compare against the published Hoechst-trained defocus predictor. Cheap to add later if our metric is contested.
