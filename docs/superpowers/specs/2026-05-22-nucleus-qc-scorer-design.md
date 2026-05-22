# Segmentation-Grounded Nucleus QC Scorer — Design

- **Date:** 2026-05-22
- **Status:** Draft (awaiting user spec review)
- **Topic:** A per-nucleus quality scorer that detects *broken/mislabeled* training patches (false detection, off-center/edge-cut, tangential graze, defocus) by grounding the score in the actual nucleus segmentation, replacing the whole-patch classical scorer that does not separate good from bad on visual inspection.

## 1. Problem

DAPIDL trains a CNN to classify cell type from 128×128 DAPI patches, one per source-segmented nucleus, labeled by transcriptomics. Many patches are unusable: out-of-focus, off-center (the labeled nucleus is not the centered object), tangentially-sectioned grazes (a dim flat blob with no subnuclear structure), or false detections (not a real nucleus). These corrupt the label↔patch correspondence.

The existing `ClassicalQualityScorer` (`starpose.qc.classical`) fails because it measures over the **whole 128px patch**: variance-of-Laplacian + Tenengrad reward crowded/edgy patches regardless of the target nucleus; Otsu foreground-fraction and a central-blob heuristic carry no notion of *internal chromatin detail* and no real "is this a nucleus" test. Score-ladders (patches sampled across the score range) show the score does not track visual quality, and absolute scores are not comparable across platforms.

## 2. Goals / Non-Goals

### Goals
- A per-nucleus score, grounded in segmentation, that flags **broken/mislabeled** patches with a reason, on the 128px training patches DAPIDL actually uses.
- Three independent, interpretable sub-scores:
  - **structure** (primary): subnuclear detail visible *inside* the nucleus — distinguishes a centrally-sectioned, in-focus nucleus (textured chromatin) from a tangential graze or defocus (flat interior).
  - **completeness + centeredness**: a single dominant nucleus covers the patch center, is not truncated at the frame, and has plausible area.
  - **objectness**: it is a real nucleus (network confidence + cross-model agreement + morphology), not debris/false detection.
- A `broken` boolean + `broken_reason`, written to a sidecar; `metadata.parquet` never modified.
- Visual validation via re-run score-ladders + per-reason montages.

### Non-Goals
- **No clean-bias filtering / sampling-weight.** The QC experiment showed biasing training toward "clean" patches (filter or weight) *hurt* cross-platform transfer (widened the Prime gap). This scorer's sole training use is removing genuinely broken patches (label↔patch mismatch), keeping the natural quality range otherwise.
- **No change to patch size.** Sweep showed accuracy rises monotonically with patch size (p256 F1 0.764 > p128 0.702 > p64 0.600); context helps, so input stays 128px. Neighbor-masking is a separate future experiment, out of scope here.
- **No manual labeling / learned model in v1.** Composite of segmentation-derived signals; thresholds tuned by visual ladder inspection. (A learned model on these features is a documented future option.)
- **No change to existing starpose segmentation code.** The scorer is a new, self-contained file.

## 3. Decisions (from brainstorming)
- **Primary use:** remove broken/mislabeled patches (label-noise), not aesthetic filtering or weighting.
- **Backends:** StarDist (primary: mask + per-object `prob`) + Cellpose (optional consensus). CellViT/CellViT++ skipped (H&E/Virchow-oriented, not DAPI fluorescence).
- **"Central cut":** two senses, both handled — (a) sectioned near the equator (not a tangential/polar graze; a graze is small + flat-interior), and (b) not truncated at the patch frame.
- **Foundation focus-models are optional plug-ins, not core.** Survey findings: Google "Microscope Image Focus Quality" (DAPI/Hoechst-trained, *domain-perfect*, but TF1.x, py≤3.7, archived Feb 2025 → high integration cost); FluoCLIP (stain-aware ordinal FQA, Feb 2026, frontier/unverified on our DAPI); Cellpose3 restoration (degradation signal, indirect). All deferred behind a `FocusExpert` plug-in seam; v1 ships the mask-localized measure.

## 4. Architecture

```
starpose/qc/
  base.py                       # existing QualityScorer ABC + QualityScore + NormRef (reuse)
  classical.py                  # existing ClassicalQualityScorer (unchanged)
  segmentation_grounded.py      # NEW: SegmentationGroundedScorer
dapidl/
  pipeline/steps/quality_control_seg.py   # NEW: dataset pass (mirror of quality_control.py)
  qc/montage.py                 # extend: per-reason "broken examples" montage
scripts/qc_validation_montage.py          # extend: ladders for the new sub-scores
```

### `SegmentationGroundedScorer(QualityScorer)` (starpose, self-contained)
Conforms to the existing ABC (`name`, `fit_reference(patches) -> NormRef`, `score_batch(patches, ref) -> list[QualityScore]`). It loads StarDist (`2D_versatile_fluo`) and optionally Cellpose directly (same as `starpose.methods.*`), so **no existing starpose code changes**.

Per patch (128×128 uint16 DAPI, center pixel = source centroid):
1. **Segment** with StarDist → label mask + per-object `prob` (capture `details['prob']`, which the current adapter discards). Optionally Cellpose → second mask.
2. **Select center nucleus**: the StarDist object whose mask covers the patch center (64,64); else the object whose centroid is nearest the center within a small radius; else → no nucleus.
3. **structure_score**: erode the center mask by `STRUCT_ERODE_PX` (default 3) to drop boundary gradients; robust-normalize interior intensities ((x−median)/(MAD+ε)); compute band-pass energy (LoG response variance over the eroded interior) and chromatin heterogeneity (local-entropy mean / interior std); combine and calibrate to [0,1] against a per-slide reference (NormRef p90 of the band-pass energy, mirroring the classical scorer's per-slide approach). Tangential graze or blur → flat interior → low.
4. **centeredness_score / completeness**: centroid-to-center distance (÷ half-patch); dominant-nucleus coverage of the central box (a neighbor dominating → off-center); border-touch test (mask within `EDGE_PX`=1 of the frame → cut_at_edge); area in µm² (pixel_size²·pixels) within `[AREA_MIN_UM2, AREA_MAX_UM2]` (default 8–400; tangential slivers fall below).
5. **objectness_score**: StarDist `prob` of the center object; if Cellpose ran, IoU between the StarDist center mask and the overlapping Cellpose mask (agreement); morphology sanity (eccentricity, solidity from `skimage.measure.regionprops`); mean interior intensity above patch background (Otsu/percentile).
6. **broken decision**: per-axis hard thresholds; `broken=True` if any axis fails; `broken_reason ∈ {no_nucleus, off_center, cut_at_edge, no_structure}` (most-severe first: no_nucleus → off_center → cut_at_edge → no_structure).

Return `QualityScore(focus_score=structure, detection_score=objectness, qc_score=composite, metrics={centeredness, completeness, stardist_prob, consensus_iou, area_um2, eccentricity, solidity, broken, broken_reason})`. The rich columns are persisted dataset-side.

### dapidl dataset pass — `quality_control_seg.py`
Mirrors `quality_control.py`: load patch labels + per-slide grouping (`slide_stats.json`), fit a per-slide NormRef from a sample, score every patch in chunks (`read_patches`), write `qc/seg_scores.parquet` (cell_id + all sub-scores + flag + reason) and `qc/seg_scores.meta.json` (params, thresholds, models, date). `metadata.parquet` untouched. Optional ClearML logging of per-reason montages + sub-score histograms.

## 5. Data flow
```
breast-6source-dapi-p128 (patches.lmdb + labels + slide_stats)
  → per slide: sample N patches → fit NormRef (structure p90)
  → per patch: StarDist (+Cellpose) → center nucleus → 4 sub-scores → broken/reason
  → qc/seg_scores.parquet  (+ seg_scores.meta.json)
  → ladders (composite + per-axis) + per-reason "broken examples" montages
```

## 6. Outputs / success criteria
- `qc/seg_scores.parquet`: one row per patch — `structure_score`, `centeredness_score`, `completeness_score`, `objectness_score`, `stardist_prob`, `consensus_iou`, `nucleus_area_um2`, `eccentricity`, `solidity`, `broken`, `broken_reason`.
- **Success = visual separation**: score-ladders sorted by composite and by each axis show low rows = flat/grazing/blurred/off-center/false and high rows = crisp, centrally-cut, internally-structured nuclei — the test the classical scorer failed. Per-`broken_reason` montages confirm each reason is correctly populated.
- A written readout of the broken-rate per source/class and a recommendation on whether/how to drop broken patches before re-training.

## 7. Compute plan
- Bottleneck: StarDist per patch (`predict_instances` is per-image). Strategy: chunked processing (reuse `read_patches`), GPU. Expect a multi-hour one-time pass over 2.28M patches; Cellpose consensus optional (off by default to halve cost; enable for a validation subset).
- **Smoke first:** run on a ~5k/source stratified sample → ladders + per-reason montages → tune thresholds → only then the full pass. (Same smoke-before-full discipline as the seg diagnostic.)
- Fallback if per-patch is too slow: segment each slide/FOV once and look up the center nucleus per centroid (reuses the seg-diagnostic loaders). Noted, not v1.
- RAM: chunked (1000 patches), per-slide NormRef from a 2000-patch sample — bounded, safe on 62 GB.

## 8. Risks / open questions
- **Thresholds are heuristic without labels.** Mitigation: per-slide normalization for `structure`; defaults tuned by visual ladders; thresholds + rationale recorded in `seg_scores.meta.json`. A small optional labeled set could calibrate later.
- **StarDist on the *exact* 128px crop** (vs full slide) may behave differently at patch borders; the center-nucleus selection + edge-cut flag handle truncation. Validate on the smoke sample.
- **Cross-platform**: structure normalization is per-slide; objectness/centeredness are geometric/probabilistic → expected to be platform-robust. Confirm s6/Prime on the smoke sample.
- **Don't re-introduce clean-bias.** Removing only `broken` patches (label-noise) is by design distinct from quality-thresholding; the readout must report broken-rate by source/class and avoid correlating "broken" with biology (e.g., a cell type that is genuinely smaller must not be over-flagged as `no_structure`/`off_center`). Per-class broken-rate is a guardrail metric.

## 9. Testing
- `SegmentationGroundedScorer`: synthetic patches with known properties — a centered textured disk (high structure, not broken); a flat centered disk (low structure → no_structure); an edge-touching disk (cut_at_edge); an empty/noise patch (no_nucleus); an off-center disk with a larger neighbor (off_center). Assert the right sub-score fails and the right `broken_reason`.
- Center-nucleus selection: multi-object mask → asserts the center-covering object is chosen; tie/none cases.
- structure measure: textured vs flat interior of identical mean intensity → textured scores higher (texture, not brightness).
- dataset pass: tiny fixture dataset → asserts sidecar columns, that `metadata.parquet` is untouched, and per-slide grouping sums correctly.
- Smoke: one source's 5k sample end-to-end (score → parquet → ladder) before the full run.

## 10. Implementation notes
- New work → own branch `feat/nucleus-qc-scorer` (the `feat/seg-diagnostic` branch is parked, committed at `3151f30`).
- Reuse: `dapidl.qc.io.read_patches`, the `slide_stats.json` grouping from `quality_control.py`, the ladder script (`scripts/qc_validation_montage.py`), `starpose.qc.base` ABC, `TF_FORCE_GPU_ALLOW_GROWTH=true` for StarDist/Cellpose GPU sharing.
- Polars for all dataframes; `uv run` for execution.

Related: [[QC-Scorer-Experiment-20260521]] (why filtering/weighting hurt — informs the "remove broken only" use), the seg diagnostic (`docs/superpowers/specs/2026-05-21-starpose-seg-diagnostic-design.md`, shared StarDist/Cellpose plumbing).
