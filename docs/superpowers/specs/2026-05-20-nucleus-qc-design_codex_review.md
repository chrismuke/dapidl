# Codex Review — Nucleus QC + Inspection Montage Design

- **Date:** 2026-05-20
- **Reviewer:** codex (codex-cli 0.130.0)
- **Reviews:** `2026-05-20-nucleus-qc-design.md`
- **Note:** codex's local sandbox could not read/write files (`bwrap: loopback` namespace error), so the spec was fed inline and this file was written from codex's stdout.

## Correctness risks
- Per-slide rank/robust normalization is relative: an all-bad slide will still assign high scores to some patches, and an all-good slide will create artificial "worst" cases. Keep raw metrics plus calibrated absolute floors.
- Focus metrics are confounded by shot noise, hot pixels, saturation, JPEG/tile artifacts, chromatin density, apoptotic condensed nuclei, and small nuclei. High Laplacian != good focus.
- Detection score assumes "bright central blob" is the truth. That penalizes valid off-center centroids, elongated/lobed nuclei, mitotic/doublet nuclei, partial edge nuclei, and tissue-specific morphology.
- Otsu + connected components will be brittle under uneven background, staining gradients, saturated DAPI, low dynamic range, debris, autofluorescence, and multiple nearby nuclei.
- `min(focus, detection)` is interpretable but harsh; a noisy detection score can dominate. Store component scores prominently and avoid treating `qc_score` as the only signal.

## Missing failure modes
- Blurry but high-contrast large nucleus may still pass detection and get only mildly penalized.
- Sharp debris, dust, dead-cell fragments, bright speckles, bubbles, or stitching edges can pass focus and maybe central-blob detection.
- Empty patches with structured background/noise can have high focus.
- Overlapping nuclei or crowded fields may look "multi-blob" and be flagged despite valid centroids.
- Saturated nuclei lose texture and can look low-focus despite being real.
- Out-of-plane nuclei may have diffuse rings/halos not captured by blob fraction alone.
- False centroids near a real neighboring nucleus may pass if the neighbor sits near center after crop jitter.

## starpose vs dapidl split
- Mostly right: reusable image-only scorer in `starpose`, dataset labels/ClearML/montage in `dapidl`.
- But the proposed `starpose qc` CLI scoring directories/zarr/image+centroids may drag dataset-ish I/O into starpose. Keep starpose CLI minimal: image/patch array in, scores table out. Avoid supporting dapidl-specific stores there.
- Normalization ownership is ambiguous. If score calibration is part of scorer semantics, starpose should define it; dapidl can provide group keys/reference stats.

## Over-engineering / simpler path
- The ABC + future FM hook may be premature. A single function/class returning a DataFrame of raw metrics + scores may be enough for v1.
- `qc_scores.npy` duplicates parquet and only stores combined score. Either save all three `.npy` arrays or defer loader-optimized files until training integration exists.
- Start with raw metrics, scores, and montages; defer generic `starpose qc` CLI unless there is an immediate non-dapidl user.

## Implementation gotchas
- Score raw DAPI patches, not normalized/augmented training tensors.
- Loading a whole slide at once may blow memory; batch within slide while sharing normalization stats, likely two-pass or sampled reference stats.
- Need atomic parquet/npy writes and schema/version provenance: scorer name, parameters, code version, normalization method, date.
- "No existing columns/files modified" conflicts with rewriting `metadata.parquet`.
- Existing QC columns need overwrite policy and compatibility handling.
- `sources.npy` may not equal slide; confirm grouping key is truly acquisition/source-specific.
- Filename-safe class names are required for montage paths.
- 128×128 labels will be unreadable unless tiles are upscaled or labels are outside tiles.
- ClearML logging many per-class montages/histograms can be noisy and slow; cap or summarize.
- Tests need hard negatives: noise, saturated blobs, debris speckles, off-center valid nuclei, crowded multi-nucleus patches, and all-bad-slide normalization behavior.
