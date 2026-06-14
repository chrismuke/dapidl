# Codex Review: Joint Nucleus Instance Segmentation + Classification Design

**Reviewed doc:** `docs/superpowers/specs/2026-05-02-instance-seg-breast-design.md`  
**Review date:** 2026-05-02  
**Verdict:** Do not start Phase A as written. The direction is sound, but the current spec has several blockers around label source, cache size, geometry ingestion, and split leakage that will waste days if not corrected first.

## Findings

### 1. BLOCKER: The proposed 11-class target cannot be produced from `label1` + `label2`

The spec says the 11-class target comes from `label1` with immune split by `tables/table_nuclei.label2` (`2026-05-02-instance-seg-breast-design.md:97`). That mapping is not present in the actual breast slides. Across s0/s1/s3/s6, `label2` has values like `T`, `B`, `Plasma`, `Mast`, and `Monocyte/Macrophage`; it does not contain `T cell CD4+`, `T cell CD8+`, `NK cell`, `Monocyte`, or `Macrophage` as separate labels.

The fine immune labels in the current data appear to live in `ct_tangram`, not `label2`. That distinction is not cosmetic: using `label1`/`label2` gives `Mast=31,709`, while a simple `ct_tangram -> 11 class` mapping gives `Mast cell=261,553`. The training target and class imbalance change materially depending on which field is authoritative.

Recommended fix: make the class source explicit before any cache build. Either reduce the target to what `label1`/`label2` can support, or switch the 11-class target to `ct_tangram` with a versioned mapping table and per-slide class count validation. Add a preflight command that prints the exact label counts used for every run and fails if any expected class is absent.

### 2. BLOCKER: The tile-cache disk estimate is too low for 50% overlap

The spec budgets about 17.5k tiles and 103 GB at 1024px tiles with stride 512 (`2026-05-02-instance-seg-breast-design.md:85`, `2026-05-02-instance-seg-breast-design.md:342`). Using the actual level-0 dimensions, the full grid is 52,490 tiles. Even filtering to tiles with at least one centroid gives about 42,136 tiles. At the spec's raw 6 MiB/tile estimate, that is about 247 GiB for non-empty 50%-overlap tiles before checkpoints and eval output.

Observed tile counts from local dimensions:

| Stride | Grid tiles | Non-empty by centroid | Raw uint32 cache estimate |
|--------|------------|-----------------------|---------------------------|
| 512 | 52,490 | 42,136 | ~247 GiB |
| 768 | 23,596 | 18,776 | ~110 GiB |
| 1024 | 13,518 | 10,535 | ~62 GiB |

Recommended fix: do not use stride 512 + dense uint32 masks as the default. Start with either stride 1024, or stride 768 with uint16 masks and measured compression. Better yet, run a one-slide materialization dry run that records compressed zarr size per tile and only then decide the final cache layout. If dense masks are retained, uint16 should be the default because the per-tile instance count is nowhere near 65k.

### 3. BLOCKER: Geometry ingestion assumptions are wrong or incomplete

The spec says WKB polygons are stored in `tables/table_nuclei` and are already pixel-space (`2026-05-02-instance-seg-breast-design.md:78`). In the local data, geometries are in `shapes/nucleus_boundaries/shapes.parquet` as GeoArrow WKB, while `table_nuclei` contains labels and centroids. Polars cannot currently read that parquet file in this environment because the `geoarrow.wkb` extension type is unsupported by the installed Polars build.

There is also a join risk. For s1, the first `table_nuclei.obs.cell_id` values and the first `shapes.parquet` index values already diverge, so row-position joins are unsafe. The cache builder must join geometry to labels by cell ID/index, not by row order.

Finally, CCW orientation is not sufficient. A sample of the first 1,000 geometries found invalid polygons in s1, s3, and especially s6. The rasterizer needs an explicit geometry repair and filtering path: load with PyArrow/GeoPandas or PyArrow/Shapely, join by cell ID, validate/repair with `shapely.make_valid` or equivalent, handle `MultiPolygon`/`GeometryCollection`, drop empty geometries with accounting, and assert coordinate bounds against the DAPI image.

### 4. HIGH: `s6` is not single-channel in `images/morpho`

The cache schema assumes `images.zarr` has shape `(n_tiles, 1, 1024, 1024)` and describes STHELAR DAPI as single-channel (`2026-05-02-instance-seg-breast-design.md:65`). Locally, s0/s1/s3 have `images/morpho/0` shape `(1, H, W)`, but s6 has shape `(5, 74945, 51265)`. Its channel metadata labels channel 0 as DAPI and the other channels as marker channels.

The current reader happens to return `arr[0]` (`src/dapidl/data/sthelar.py:257`), but the design should not rely on an implicit first-channel convention. The tile builder should select the channel whose OMERO label is `DAPI`, record the selected channel in the manifest, and fail if the DAPI channel is missing or ambiguous.

### 5. HIGH: Tile-level train/val split leaks nuclei when tiles overlap

The spec says tiles, not nuclei, are the split unit and that this avoids leakage (`2026-05-02-instance-seg-breast-design.md:183`). With 50% or 25% overlap, the same nucleus can appear in multiple tiles. A random tile split within a training slide will put duplicate views of the same nucleus into both train and validation. That makes val PQ/F1 optimistic and can affect early stopping and architecture selection.

Recommended fix: split by spatial blocks or by non-overlapping parent tiles first, then generate overlapping child tiles inside each split. Keep the planned instance-ID intersection check, but make it a hard pre-training assertion rather than a reporting sanity check.

### 6. HIGH: The Phase A validation count is not meaningful with overlapping tiles

The spec says cache instance counts should match `tables/table_nuclei.shape[0]` within +/-2% despite border duplicates (`2026-05-02-instance-seg-breast-design.md:269`). With overlapping tiles, per-tile label rows will intentionally count many nuclei multiple times, so this check will fail or encourage undercounting.

Recommended fix: validate two separate numbers. First, unique source cell IDs represented in at least one non-border tile should match table IDs after documented drops. Second, per-tile rows should be allowed to exceed table count and should be summarized as a duplicate exposure distribution.

### 7. HIGH: The three architecture plans are not methodologically equivalent

The stated goal is a one-forward-pass, shared-encoder joint model (`2026-05-02-instance-seg-breast-design.md:11`). CelloType fits that goal best. The Cellpose-SAM plan is explicitly two-stage and freezes segmentation before training a classifier head (`2026-05-02-instance-seg-breast-design.md:139`), so it is not a jointly trained shared-encoder model. The StarDist plan requires modifying a TensorFlow/Keras StarDist model to add a class head (`2026-05-02-instance-seg-breast-design.md:123`), which is likely substantially more work than the design acknowledges and may not cleanly load `2D_versatile_fluo` weights after graph surgery.

Recommended fix: separate "joint models" from "segmentation plus classifier baselines" in the benchmark. If the Cellpose-SAM and StarDist variants are not truly end-to-end, report them as baselines rather than as equivalent candidates for the core claim.

### 8. MEDIUM: Required dependencies and augmentation names are not aligned with the repo environment

The current `pyproject.toml` does not list `rasterio`, `stardist`, or `pycocotools`, and the active environment could not import those packages. The installed Albumentations is 2.0.8, where `GammaCorrection`, `GaussBlur`, and `Cutout` are not valid transform names; use `RandomGamma`, `GaussianBlur`, and `CoarseDropout` or pin/adjust the dependency.

Recommended fix: add an explicit dependency and preflight section to the design before implementation. The build should fail early if optional architecture-specific packages are missing.

### 9. MEDIUM: Evaluation metrics need exact class-aware matching semantics

The metric list is directionally right (`2026-05-02-instance-seg-breast-design.md:229`), but it does not specify whether PQ/AP are class-aware or class-agnostic, how unmatched detections affect per-class F1, or whether border-windowing is applied before matching. These choices change the result, especially for rare classes.

Recommended fix: define one canonical matching protocol: crop predictions and GT to the same eval window, match masks by IoU threshold, count unmatched predictions as false positives for their predicted class, count unmatched GT as false negatives for the GT class, and report both segmentation-only PQ and class-aware panoptic quality.

### 10. MEDIUM: Xenium eval is not enough for a manuscript-level cross-source claim

The proposed Xenium evaluation is qualitative plus marker consistency (`2026-05-02-instance-seg-breast-design.md:241`). That is acceptable for a debugging gallery, but weak for a manuscript result that claims cross-source generalization. Marker overlays validate some cell labels, not nucleus boundary quality, and they will be biased toward marker-positive cases.

Recommended fix: if this will be a manuscript figure, annotate a small Xenium ROI set for nuclei and class labels, even if only 10-20 ROIs. If manual class labels are too expensive, at least separate the claim into quantitative STHELAR LOTO performance and qualitative Xenium transfer examples.

### 11. LOW: The tile sampler can overfit duplicate rare-class contexts

The proposed tile weight is the inverse frequency of the rarest non-Other class in a tile (`2026-05-02-instance-seg-breast-design.md:218`). With overlap, the same rare nucleus and local neighborhood may be sampled repeatedly. This can inflate rare-class exposure without increasing diversity.

Recommended fix: compute weights on spatial blocks or unique source instances, then sample one tile view per source instance per epoch where practical. At minimum, report duplicate exposure per class.

## Direct Answers to Section 9 Risks

**Risk 1, polygon orientation:** Orientation is not the main risk. The implementation must handle GeoParquet WKB loading, cell-ID joins, invalid polygons, `MultiPolygon`, `GeometryCollection`, empty geometries, and coordinate bounds.

**Risk 2, class mapping completeness:** The planned `label1`/`label2` mapping is not complete enough for 11 classes. Use `ct_tangram` or shrink the class set.

**Risk 3, CelloType/PanNuke mismatch:** Replacing the class head is expected, but the encoder and mask decoder can still transfer. This is acceptable if the design treats the STHELAR class head as newly initialized and logs warm-start details.

**Risk 4, Cellpose-SAM license:** Keep it out of any commercial-facing default path. Also do not let a non-commercial-weight model become the only strong manuscript result unless the license caveat is explicit.

**Risk 5, Xenium-only eval:** Qualitative Xenium eval is useful but not defensible as the only cross-source evidence for a manuscript. Add a small labeled Xenium subset if the paper will claim quantitative transfer.

**Risk 6, overlap inflation:** Account for duplicate nuclei at split time and in loss/sampling. Random tile splitting is unsafe.

**Risk 7, pilot fairness:** Equal epochs is not enough. Use either fixed wall-clock/step budgets or a two-stage pilot: smoke test all three on a small subset, then full pilot only for the two that are operational and promising.

**Risk 8, disk pressure:** Do not accept 50% overlap with dense uint32 masks under the current 161 GB free-space constraint. The corrected estimate is too high. Prefer uint16 masks and stride 768 or 1024 unless a measured one-slide cache proves compression is enough.

**Risk 9, training time:** Training time should be estimated from tile count and model throughput, not cell count. A full three-architecture pilot is only worth running after the cache and one architecture have a measured tiles/sec baseline.

## Recommended Revised Plan

1. Add a zero-training preflight script that validates slide shapes, DAPI channel selection, geometry load/join, label mapping counts, class representation, and expected cache size.

2. Materialize one slide first, preferably s3 or a spatial subset of s1, and measure actual zarr compression, rasterization throughput, invalid geometry rate, and dataloader throughput.

3. Switch the initial cache default to `stride=1024`, `uint16` instance maps, DAPI-channel-only images, and unique source cell IDs in `labels.parquet`.

4. Use spatial-block train/val splits before overlapping tiles are generated.

5. Run CelloType first as the most aligned joint instance segmentation/classification architecture. Keep StarDist and Cellpose-SAM as baselines unless their implementations are truly end-to-end and comparable.

6. Defer the full 7-training schedule until the above preflight and one-slide materialization pass.

