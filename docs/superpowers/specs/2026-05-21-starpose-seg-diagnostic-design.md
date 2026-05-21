# starpose Segmentation Diagnostic (Phase 1) — Design

- **Date:** 2026-05-21
- **Status:** Draft (awaiting user review)
- **Topic:** Re-segment the 6 breast sources with starpose; compare against the source segmentation; QC-score our own patches vs source patches.

## 1. Problem

The QC experiment showed the cell-type model transfers poorly to Xenium Prime (s6), and that s6's *test* patches are the lowest-QC source (mean 0.222). Open question: **is the source segmentation responsible** — i.e., would segmenting nuclei ourselves with starpose (uniform method across platforms) yield cleaner, more consistent patches and reduce the cross-platform gap?

Phase 1 answers the diagnostic part **without** the expensive re-train: segment the same 6 slides with starpose, compare its masks to the source segmentation, and compare the QC of patches built on our segmentation vs the source's.

## 2. Goals / Non-Goals

### Goals
- For each of the 6 breast sources (Xenium rep1/rep2, STHELAR s0/s1/s3/s6), on representative FOVs:
  - Run starpose `adaptive` nucleus segmentation + proseg transcript-aware cell expansion.
  - Compare to the **source** nucleus + cell segmentation: detection precision/recall/F1 (IoU matching), matched-IoU distribution, cell-count ratio, morphometric distributions (area/solidity/eccentricity) for nucleus and cell.
  - Extract 128² nucleus patches around starpose centroids vs source centroids in the same FOVs; score both with `ClassicalQualityScorer`; compare QC distributions per source.
- Produce a per-source results table + figures + disagreement montages.

### Non-Goals (Phase 1)
- No re-annotation, no patch-dataset rebuild, no re-training (that is Phase 2, gated on these results).
- No full-slide segmentation — FOV-sampled only.
- No new model backends beyond `adaptive` (+ proseg expansion).
- starpose stays dataset/platform-agnostic; all Xenium/STHELAR-specific loading lives in dapidl.

## 3. Decisions (from brainstorming)
- **Scope:** Phase 1 diagnostic only.
- **Coverage:** ~8 density-stratified FOVs (2048²) per source via starpose `fov_selector`, seeded.
- **Nucleus backend:** `adaptive` (Cellpose+StarDist consensus).
- **Cell expansion:** proseg (transcript-aware); **watershed fallback** if proseg install fails.

## 4. Architecture — Approach ① (dapidl orchestrator over starpose library)

```
scripts/seg_diagnostic.py            # orchestrator (dapidl)
src/dapidl/seg_eval/
  __init__.py
  source_masks.py                    # SourceSegmentationLoader (per-platform)
  compare.py                         # agreement + morphometric + count rollups
  qc_compare.py                      # QC(own centroids) vs QC(source centroids)
```

starpose provides (reused, no platform code added there): `benchmark.fov_selector`,
`core.segment(method="adaptive")`, `expansion.proseg` / `expansion.watershed`,
`evaluate.agreement` (instance IoU matching), `evaluate.morphometric`. QC scoring
reuses `starpose.qc.ClassicalQualityScorer`.

### Components
- **`SourceSegmentationLoader`** — given a source + FOV bbox, returns source nucleus
  and cell label masks aligned to the FOV:
  - **Xenium**: read `nucleus_boundaries.parquet` / `cell_boundaries.parquet` (µm
    polygons), convert µm→px via the morphology pixel size, crop to bbox, rasterize
    (skimage polygon → label image).
  - **STHELAR**: read sdata `shapes` (polygons) for the FOV (note the double-nested
    `sdata_*.zarr/sdata_*.zarr` layout); `tables` carry the existing cell-type labels.
- **FOV selection** — `fov_selector` on the source's DAPI; same FOVs used for starpose
  and source so the comparison is region-matched.
- **starpose seg** — `adaptive` nuclei per FOV → proseg cell expansion (transcripts
  cropped to bbox). proseg install is Step 0 (see §7); watershed fallback.
- **`compare.py`** — instance matching (IoU ≥ 0.5) → detection P/R/F1; matched-IoU
  histogram; `n_starpose / n_source` count ratio; morphometric distribution summaries
  (nucleus + cell) for both segmentations.
- **`qc_compare.py`** — crop 128² DAPI patches around starpose centroids and around
  source centroids within the FOVs; `ClassicalQualityScorer.score_batch` (per-FOV
  reference); compare `qc_score` / `focus` / `detection` distributions per source.

## 5. Data flow
```
per source:
  DAPI mosaic + transcripts + source polygons/sdata
    → fov_selector → ~8 FOV bboxes
    → per FOV:
        starpose adaptive nuclei → proseg cells
        SourceSegmentationLoader → source nuclei + cells (same bbox)
        agreement(starpose, source) + morphometric(both)
        QC patches @ starpose centroids vs @ source centroids
    → aggregate per-source rows
→ pipeline_output/seg_diagnostic_2026_05/results.parquet
   + figures/ (agreement bars, count ratios, QC own-vs-source per source)
   + montages/ (disagreement examples: starpose-only, source-only, low-IoU)
```

## 6. Outputs / success criteria
- `results.parquet`: one row per (source, FOV, segmentation) with detection P/R/F1,
  median matched IoU, count ratio, morphometric summaries, and QC means (own vs source).
- Figures: per-source agreement, count ratio, and **QC(own) vs QC(source)** — the
  headline being whether starpose patches out-QC the source, especially on Prime s6.
- A short written readout interpreting whether segmentation explains the QC/Prime gap,
  and a go/no-go recommendation for Phase 2.

## 7. Prerequisites / risks
- **proseg install** (Step 0): `cargo install proseg` (cargo 1.91 present). If it fails,
  fall back to `expansion.watershed` and note it in the readout. Cell comparison still
  proceeds with the fallback expander.
- **STHELAR sdata reading**: double-nested zarr; needs a small reader to pull the DAPI
  image array, the `shapes` polygons, and `points` transcripts for a bbox.
- **Coordinate alignment**: Xenium polygons are in µm — must convert to pixels using the
  morphology `pixel_size` (read from the OME metadata) so masks align to the DAPI raster.
  STHELAR sdata is internally consistent (image + shapes share the coordinate system).
- **Matching threshold**: IoU ≥ 0.5 for "same object"; report the matched-IoU
  distribution too so the threshold isn't load-bearing.

## 8. Compute
~8 FOVs × 6 sources × (adaptive + proseg) on 2048² crops — minutes per source on the
3090, well under an hour total. proseg build is a one-time few-minute step.

## 9. Testing
- `SourceSegmentationLoader`: synthetic polygon → known raster mask (Xenium path);
  a tiny synthetic sdata for the STHELAR path (or a real-FOV smoke read).
- `compare.py`: two synthetic label masks with known overlap → assert P/R/F1 and
  count ratio match hand-computed values; identical masks → F1 = 1.0.
- `qc_compare.py`: reuse the QC scorer's tested behavior; assert the patch crops align
  to centroids and the per-source aggregation has the expected columns.
- Smoke: one FOV of one Xenium source end-to-end (seg → compare → QC) before the full run.
