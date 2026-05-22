# Segmentation-Grounded Nucleus QC Scorer — Design

- **Date:** 2026-05-22
- **Status:** Approved (design + calibration choice); revised per codex review
- **Topic:** A per-nucleus quality scorer that detects *broken/mislabeled* training patches (false detection, off-center/edge-cut, tangential graze, defocus) by grounding the score in the actual nucleus segmentation, replacing the whole-patch classical scorer that does not separate good from bad on visual inspection.
- **Companion:** `2026-05-22-nucleus-qc-scorer-design_codex_review.md` (verdict GO-WITH-CHANGES; its changes are folded in here).

## 1. Problem

DAPIDL trains a CNN to classify cell type from 128×128 DAPI patches, one per source-segmented nucleus, labeled by transcriptomics. Many patches are unusable: out-of-focus, off-center (the labeled nucleus is not the centered object), tangentially-sectioned grazes (a dim flat blob with no subnuclear structure), or false detections (not a real nucleus). These corrupt the label↔patch correspondence.

The existing `ClassicalQualityScorer` (`starpose.qc.classical`) fails because it measures over the **whole 128px patch**: variance-of-Laplacian + Tenengrad reward crowded/edgy patches regardless of the target nucleus; Otsu foreground-fraction and a central-blob heuristic carry no notion of *internal chromatin detail* and no real "is this a nucleus" test. Score-ladders (patches sampled across the score range) show the score does not track visual quality, and absolute scores are not comparable across platforms.

## 2. Goals / Non-Goals

### Goals
- A per-nucleus **high-specificity rejector of *obviously* broken patches**, grounded in segmentation, with a reason, on the 128px training patches DAPIDL actually uses. When in doubt, keep.
- Three independent, interpretable sub-scores:
  - **structure**: subnuclear detail visible *inside* the nucleus — distinguishes a centrally-sectioned, in-focus nucleus from a tangential graze or defocus (flat interior). **Treated as the most conservative axis** (see §3).
  - **completeness + centeredness**: a single dominant nucleus covers the patch center, is not truncated at the frame, and has plausible area (neither sliver nor merged blob).
  - **objectness**: it is a real nucleus (StarDist confidence + lenient morphology), not debris/false detection.
- A `broken` boolean + `broken_reason`, written to a sidecar; `metadata.parquet` never modified.
- Visual validation (score-ladders + per-reason montages) **plus a stratified false-positive audit** (broken-rate by source × class × size) as the anti-censoring guardrail.

### Non-Goals
- **No clean-bias filtering / sampling-weight.** The QC experiment showed biasing training toward "clean" patches (filter or weight) *hurt* cross-platform transfer (widened the Prime gap). This scorer's sole training use is removing genuinely broken patches (label↔patch mismatch), keeping the natural quality range otherwise.
- **No change to patch size.** Sweep showed accuracy rises monotonically with size (p256 0.764 > p128 0.702 > p64 0.600); context helps, so input stays 128px. Neighbor-masking is a separate future experiment.
- **Label-free v1 with an escalation trigger.** v1 uses only segmentation-derived signals; thresholds tuned by visual ladders + the stratified audit. A small labeled calibration set (and any learned model) is added **only if** the audit flags bias or visual review is inconclusive (§8 escalation trigger).
- **No change to existing starpose segmentation code.** The scorer is a new, self-contained file.
- **Not detectable from DAPI alone, so out of scope:** transcriptomic label mismatch / doublets that look like a normal nucleus.

## 3. Decisions (brainstorming + codex review)
- **Primary use:** remove broken/mislabeled patches (label-noise), not aesthetic filtering or weighting.
- **Backend (v1): StarDist only** (mask + per-object `prob`). CellViT/CellViT++ skipped (H&E/Virchow-oriented, not DAPI). **Cellpose consensus is v2** — added only if the StarDist-only rejector demonstrably misses broken cases.
- **"Central cut":** two senses, both handled — (a) sectioned near the equator (not a tangential/polar graze; a graze is small + flat-interior), and (b) not truncated at the patch frame.
- **v1 is deliberately minimal (codex):** StarDist center-object selection + geometry/completeness + **one** conservative interior-texture metric + montages + stratified audit. The `FocusExpert` plug-in (Google-MIQ / FluoCLIP / Cellpose3) and a multi-metric texture stack are **deferred**.
- **`structure` is the most conservative axis and never the sole reason to drop a patch.** A flat interior can be real biology — small dense lymphocytes, pyknotic/apoptotic nuclei, low-texture cell types. It fires `no_structure` only on *extreme-flat* interiors and is cross-checked by the per-class audit.
- **Guardrail = stratified false-positive review:** broken-rate reported by source × class × size bin; flagged patches must not concentrate in a real cell type.

## 4. Architecture

```
starpose/qc/
  base.py                       # existing QualityScorer ABC + QualityScore + NormRef (reuse)
  classical.py                  # existing ClassicalQualityScorer (unchanged)
  segmentation_grounded.py      # NEW: SegmentationGroundedScorer (StarDist-only v1, self-contained)
dapidl/
  pipeline/steps/quality_control_seg.py   # NEW: dataset pass (mirror of quality_control.py) + stratified audit
  qc/montage.py                 # extend: per-reason "broken examples" montage
scripts/qc_validation_montage.py          # extend: ladders for the new sub-scores
```

### `SegmentationGroundedScorer(QualityScorer)` (starpose, self-contained)
Conforms to the existing ABC (`name`, `fit_reference(patches) -> NormRef`, `score_batch(patches, ref) -> list[QualityScore]`). Loads StarDist (`2D_versatile_fluo`) directly (as `starpose.methods.stardist` does), so **no existing starpose code changes**.

Per patch (128×128 uint16 DAPI, center pixel = source centroid):
1. **Segment** with StarDist → label mask + per-object `prob` (capture `details['prob']`, which the current adapter discards). *(v1: StarDist only; Cellpose consensus is v2.)*
2. **Select center nucleus**: the StarDist object whose mask covers the patch center (64,64); else the object whose centroid is nearest the center within a small radius; else → `no_nucleus`.
3. **structure_score**: erode the center mask by `STRUCT_ERODE_PX` (default 3); robust-normalize interior intensities ((x−median)/(MAD+ε)); compute **one** conservative texture metric — MAD-normalized high-frequency (LoG) energy over the eroded interior — calibrated to [0,1] against a per-slide reference (NormRef p90) **with an absolute floor** so an all-bad slide cannot manufacture passing scores. Fires `no_structure` only on extreme-flat interiors.
4. **centeredness_score / completeness**: centroid-to-center distance (÷ half-patch); dominant-nucleus coverage of the central box (a neighbor dominating → `off_center`); border-touch test (mask within `EDGE_PX`=1 of the frame → `cut_at_edge`); area in µm² (pixel_size²·pixels) within `[AREA_MIN_UM2, AREA_MAX_UM2]` (default 8–400) — below = tangential sliver, **above = merged/touching blob** (both flagged).
5. **objectness_score** (v1): StarDist `prob` of the center object + **lenient** morphology sanity (eccentricity/solidity from `skimage.measure.regionprops`, flagging only *extreme* outliers — do not penalize legitimately small or elongated nuclei) + mean interior intensity above patch background. Cellpose↔StarDist IoU agreement is v2 (`consensus_iou` column reserved, null in v1).
6. **broken decision**: per-axis thresholds tuned **conservatively** (high specificity); `broken=True` if an axis fails; `broken_reason ∈ {no_nucleus, off_center, cut_at_edge, no_structure}` (most-severe first). `structure` alone never sets `broken` without the per-class audit confirming it is not censoring a cell type.

Return `QualityScore(focus_score=structure, detection_score=objectness, qc_score=composite, metrics={centeredness, completeness, stardist_prob, consensus_iou(null v1), area_um2, eccentricity, solidity, saturated_frac, broken, broken_reason})`.

### dapidl dataset pass — `quality_control_seg.py`
Mirrors `quality_control.py`: load patch labels + per-slide grouping (`slide_stats.json`), fit per-slide NormRef from a sample, score every patch in chunks (`read_patches`), write `qc/seg_scores.parquet` + `qc/seg_scores.meta.json` (params, thresholds, model, date). `metadata.parquet` untouched. **Then compute the stratified audit**: broken-rate and per-`broken_reason` rate by source × class × size bin → `qc/seg_broken_audit.parquet`. Optional ClearML logging.

## 5. Data flow
```
breast-6source-dapi-p128 (patches.lmdb + labels + slide_stats)
  → per slide: sample N patches → fit NormRef (structure p90, with absolute floor)
  → per patch: StarDist → center nucleus → 3 sub-scores → broken/reason
  → qc/seg_scores.parquet  (+ seg_scores.meta.json)
  → stratified audit: broken-rate by source×class×size → qc/seg_broken_audit.parquet
  → ladders (composite + per-axis) + per-reason "broken examples" montages
```

## 6. Outputs / success criteria
- `qc/seg_scores.parquet`: one row per patch — `structure_score`, `centeredness_score`, `completeness_score`, `objectness_score`, `stardist_prob`, `consensus_iou`, `nucleus_area_um2`, `eccentricity`, `solidity`, `saturated_frac`, `broken`, `broken_reason`.
- `qc/seg_broken_audit.parquet`: broken-rate (overall + per reason) by source × class × size bin — **the censoring guardrail**.
- **Success = visual separation + no class censoring**: score-ladders (composite + per-axis) show low rows = flat/grazing/blurred/off-center/false and high rows = crisp, centrally-cut, internally-structured nuclei; per-`broken_reason` montages confirm each reason; and the audit shows broken patches are **not** concentrated in a real cell type.
- A written readout of broken-rate by source/class/size + a go/no-go on dropping broken patches before re-training.

## 7. Compute plan
- Bottleneck: StarDist per patch (`predict_instances` is per-image). Strategy: chunked processing (reuse `read_patches`), GPU, `TF_FORCE_GPU_ALLOW_GROWTH=true`. Expect a multi-hour one-time pass over 2.28M patches.
- **Smoke first:** run on a ~5k/source stratified sample → ladders + per-reason montages + mini-audit → tune conservative thresholds → only then the full pass.
- Fallback if per-patch is too slow: segment each slide/FOV once and look up the center nucleus per centroid (reuses seg-diagnostic loaders). Noted, not v1.
- RAM: chunked (1000 patches), per-slide NormRef from a 2000-patch sample — bounded, safe on 62 GB.

## 8. Risks / open questions
- **Biggest risk — systematic censoring (codex).** The scorer may drop real nuclei disproportionately by source, class, morphology, or size (esp. `no_structure` on small dense lymphocytes / pyknotic nuclei, or `structure` collapsing for a whole platform with different PSF/staining), improving apparent cleanliness while *biasing* the classifier and worsening transfer. **Mitigations:** conservative high-specificity thresholds; `structure` never the sole drop reason; per-slide normalization + absolute floor; and the **stratified by-class FP audit** as a hard gate before any patch is dropped.
- **Escalation trigger (when to add labels):** if the audit shows any real cell type's broken-rate exceeds ~1.5× the overall rate, or visual ladders are inconclusive, **pause and add a ~500–1000 stratified labeled calibration set** (definitely-broken / usable / ambiguous) to set thresholds and quantify false-positives before dropping anything from training.
- **Thresholds are heuristic without labels** → only a high-specificity rejector of obvious garbage, by design. Recorded in `seg_scores.meta.json`.
- **Edge cases handled:** crowded/touching nuclei (center inside a merged object → over-large area flag); valid small/elongated nuclei (lenient morphology, extreme-only); all-bad slide (absolute floor on structure); saturation/illumination (`saturated_frac` recorded); pixel-size metadata trusted from the loader (sanity-check area distribution per source).
- **StarDist on the exact 128px crop** may behave differently at borders; center-nucleus selection + edge-cut flag handle truncation — validate on smoke.

## 9. Testing
- `SegmentationGroundedScorer` on synthetic patches: centered textured disk (not broken); flat centered disk of equal mean intensity (→ `no_structure`); edge-touching disk (`cut_at_edge`); empty/noise (`no_nucleus`); off-center disk beside a larger neighbor (`off_center`); over-large merged blob (flagged). Assert correct axis + `broken_reason`.
- Center-nucleus selection: multi-object mask → center-covering object chosen; tie/none cases.
- structure metric: textured vs flat interior of identical mean intensity → textured scores higher (texture, not brightness); absolute floor prevents all-flat slide from passing.
- dataset pass: tiny fixture dataset → asserts sidecar + audit columns, `metadata.parquet` untouched, per-slide grouping sums correctly.
- audit: synthetic scores skewed to one class → audit surfaces the class concentration.
- Smoke: one source's 5k sample end-to-end (score → parquet → audit → ladder) before the full run.

## 10. Implementation notes
- Branch `feat/nucleus-qc-scorer` (off `main`; the `feat/seg-diagnostic` branch is parked at `3151f30`).
- Reuse: `dapidl.qc.io.read_patches`, `slide_stats.json` grouping from `quality_control.py`, the ladder script, `starpose.qc.base` ABC, `TF_FORCE_GPU_ALLOW_GROWTH=true`.
- Polars for all dataframes; `uv run` for execution.

Related: [[QC-Scorer-Experiment-20260521]] (why filtering/weighting hurt — informs "remove broken only"); seg diagnostic (`docs/superpowers/specs/2026-05-21-starpose-seg-diagnostic-design.md`, shared StarDist plumbing).
