# Joint Nucleus Instance Segmentation + Per-Class Classification — Breast (DAPI)

**Date:** 2026-05-02 (v2 — post codex review + data verification)
**Status:** Design — ready for Phase A implementation
**Successor of:** patch-classifier work (`pipeline_output/breast_dapi_p{32,64,128,256}/`, finished 2026-05-01)
**Supersedes:** v1 (same path, 2026-05-02 morning) — incorporates codex review findings + on-disk data verification

---

## 1. Goal

Train an end-to-end model that, in **one forward pass** on a DAPI image patch, produces:

1. **Instance masks** for every nucleus (per-pixel uint16 instance map; one ID per nucleus).
2. **Per-instance class labels** at three granularity tiers — **broad / medium / fine** — emitted in dapidl's existing `VotingResult` schema (see §3.1) so predictions slot into the existing PopV-style ensemble alongside CellTypist/SingleR/scType.

Output is "this nucleus is a CD4 T cell (fine), T_Cell (medium), Immune (broad), with confidence 0.83" — a structured prediction, not just a mask + later patch classification.

The instance-seg head and classification head share the same encoder so morphology informs class and vice versa.

**Why this is the next step.** The patch classifier (current) takes pre-segmented nuclei as input. Real users want to point a model at a DAPI mosaic and get annotated nuclei out. Today that requires three steps; this design collapses it into one trained model and benchmarks candidates head-to-head.

**Cells / cytoplasm are out of scope here.** Phase 2 will use the existing starpose Voronoi/watershed expansion on top of these nucleus instances.

---

## 2. Data

### 2.1 Training source: STHELAR breast (Giraud-Sauveur et al., 2025)

Located at `/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s{0,1,3,6}.zarr/sdata_breast_s{0,1,3,6}.zarr/`.

| Slide | Cells (verified 2026-05-02 from `tables/table_nuclei`) | Notes |
|-------|--------------------------------------------------------|-------|
| s0 | **576,963** | breast tumor |
| s1 | **892,966** | breast tumor (largest) |
| s3 | **365,604** | breast tumor (smallest) |
| s6 | **692,184** | breast tumor (5-channel image; see §2.4) |
| **Total** | **2,527,717** | |

**Why STHELAR:**
- Real polygon ground truth (`shapes/nucleus_boundaries`) — not Cellpose pseudo-GT.
- DAPI + matched H&E + per-nucleus class labels.
- Same biology (breast tumor) as Xenium reps → minimal domain shift on labels.
- 4 slides → enables LOTO cross-validation.

Existing reader: `src/dapidl/data/sthelar.py` — already has `STHELAR_SCALE_FACTOR = 4.705882352941177` (1 / 0.2125 µm) and zarr structure handling. **This v2 explicitly uses that constant** for µm→px conversion (see §2.3).

### 2.2 Class label source — `ct_tangram` (BLOCKER 1 fix)

**v1 was wrong**: said 11-class target comes from `label1 + label2`. Verified 2026-05-02:

- `label2` does **not** contain `CD4 T cell`, `CD8 T cell`, `NK cell`, `Monocyte`, `Macrophage`, `Dendritic cell` as separate values on **any** of s0/s1/s3/s6. Only coarse immune labels (`T`, `B`, `Plasma`, `Mast`, `Monocyte/Macrophage`).
- `ct_tangram` has **39 fine-grained labels** including all of the above as separate values, with stable presence across all 4 slides.
- Mast cell counts differ materially (s6: 4,591 in `label2` vs 202,128 in `ct_tangram`) — the field choice changes class imbalance dramatically.

**v2 mapping pipeline:**

```
STHELAR ct_tangram  →  dapidl FINEGRAINED_CLASS_NAMES   (fine, ~17 in scope)
                  ↓
         FINE_TO_MEDIUM_MAPPING  →  dapidl MEDIUM_CLASS_NAMES   (10 classes)
                  ↓
         CELL_TYPE_HIERARCHY     →  dapidl COARSE_CLASS_NAMES   (3 + Endothelial)
```

All three lookup tables already exist:
- `src/dapidl/pipeline/components/annotators/popv_ensemble.py:53` — `MEDIUM_CLASS_NAMES`, `FINE_TO_MEDIUM_MAPPING`
- `src/dapidl/pipeline/components/annotators/mapping.py` — `COARSE_CLASS_NAMES`, `FINEGRAINED_CLASS_NAMES`, `CELL_TYPE_HIERARCHY`, `map_to_broad_category_4class()`
- `src/dapidl/pipeline/components/annotators/cell_ontology_mapping.py` — `CELL_ONTOLOGY` with `CL_DEPTH`

**Per-slide preflight gate (mandatory before Phase A):**
```bash
uv run python scripts/instance_seg/preflight_labels.py --slides s0 s1 s3 s6
```
Prints per-slide class counts at all three tiers; **fails non-zero** if any expected coarse class is missing on any slide. Surfaces the s6 outlier flagged in §9 risk #1.

### 2.3 Geometry ingestion (BLOCKER 3 fix)

**v1 was wrong on three counts:**

| v1 claim | Reality (verified 2026-05-02) |
|----------|-------------------------------|
| Polygons live in `tables/table_nuclei` | Polygons are in `shapes/nucleus_boundaries/shapes.parquet` (GeoArrow WKB) |
| Polygons are pixel-space | Polygons are in **microns** (Xenium pixel size 0.2125 µm/px) |
| Polars can read them | Polars 1.x **panics** on `geoarrow.wkb` extension type — must use PyArrow + GeoPandas |
| Row-position join is fine | s1 has cell_id misalignment between `shapes.parquet` and `table_nuclei.obs.cell_id` — must join on `cell_id` |

**v2 loader:**
```python
import geopandas as gpd
from shapely.validation import make_valid

# 1. Load geometry by cell_id
gdf = gpd.read_parquet(slide_path / "shapes/nucleus_boundaries/shapes.parquet")
# gdf.index is cell_id (str)

# 2. Load labels keyed by cell_id from table_nuclei.obs
labels = read_obs(slide_path / "tables/table_nuclei/obs", cols=["cell_id", "ct_tangram"])

# 3. Inner-join on cell_id (NOT row position)
joined = gdf.join(labels.set_index("cell_id"), how="inner")

# 4. Repair invalid polygons (s6 has 1.6% invalid; s0/s1/s3 ~0.07%)
joined.geometry = joined.geometry.apply(make_valid)
joined = joined[joined.geometry.geom_type == "Polygon"]  # drop GeometryCollection from repair

# 5. µm → px using starpose's known constant
from starpose.methods.cellvit import STHELAR_SCALE_FACTOR  # = 4.705882352941177
joined["geometry"] = joined.geometry.scale(STHELAR_SCALE_FACTOR, STHELAR_SCALE_FACTOR, origin=(0, 0))
```

Verified polygon-type distribution (no MultiPolygon/GeometryCollection in raw STHELAR breast data):
| Slide | n | invalid | empty | MultiPolygon | GeometryCollection |
|-------|---|---------|-------|--------------|---------------------|
| s0 | 576,963 | 394 (0.07%) | 0 | 0 | 0 |
| s1 | 892,966 | 618 (0.07%) | 0 | 0 | 0 |
| s3 | 365,604 | 257 (0.07%) | 0 | 0 | 0 |
| s6 | 692,184 | **11,001 (1.6%)** | 0 | 0 | 0 |

`make_valid` may produce `MultiPolygon`/`GeometryCollection` — drop those after repair (≤2% loss; track count in manifest).

### 2.4 DAPI channel selection (HIGH 4 fix)

s0/s1/s3 are single-channel `(1, H, W)`. **s6 is 5-channel `(5, 74945, 51265)`** with OMERO labels:

| Channel | Label |
|---------|-------|
| 0 | `DAPI` |
| 1 | `ATP1A1/CD45/E-Cadherin` |
| 2 | `18S` |
| 3 | `AlphaSMA/Vimentin` |
| 4 | `dummy` |

**v2 loader** reads `images/morpho/.zattrs.omero.channels`, picks the channel whose `label == "DAPI"`, records the chosen index in the cache manifest, and **fails loudly** if missing or ambiguous. Single-channel slides (where `label == "0"`) are handled via "if only 1 channel, use channel 0" branch.

### 2.5 Tile cache (revised layout — BLOCKER 2 fix)

**v1 estimate: 17.5k tiles / ~103 GB at stride 512.** Real grid measured 2026-05-02:

| Stride | Tile count | Raw uint16 mask | Raw uint32 mask |
|--------|-----------|-----------------|------------------|
| 512 (50% overlap) | 53,417 | 104.3 GiB | 208.7 GiB |
| 768 (25% overlap) | 23,730 | 46.3 GiB | 92.7 GiB |
| **1024 (no overlap)** | **13,518** | **26.4 GiB** | **52.8 GiB** |

**v2 default: stride 1024 + uint16 instance map** (max ~hundreds of cells per tile << 65k uint16 limit). Saves ~180 GiB vs v1 stride-512 + uint32 plan.

Cache layout:
```
/mnt/work/datasets/derived/sthelar_breast_tiles/
├── manifest.parquet           # tile_idx, slide, x0, y0, n_nuclei, dapi_channel, n_invalid_dropped
├── s0/
│   ├── images.zarr            # zarr v3, (n_tiles, 1024, 1024) uint16, blosc-zstd-3
│   ├── instances.zarr         # zarr v3, (n_tiles, 1024, 1024) uint16
│   └── labels.parquet         # (tile_idx, instance_id, cell_id, broad, medium, fine, area, cy, cx)
├── s1/ ...
├── s3/ ...
└── s6/ ...
```

**One-slide dry run before committing layout** (Phase 0; see §6): materialize s3 (smallest), measure compressed zarr size, decide whether to keep stride 1024 or drop to 768 with no fear.

**Estimated cache size after blosc-zstd-3 compression** (DAPI is sparse in ID space → high compression): expect ~5–8 GiB total at stride 1024. Confirmed empirically in Phase 0.

### 2.6 Eval-only source: Xenium breast reps

`~/datasets/raw/xenium/breast_tumor_rep{1,2}/` — for cross-source eval only. **No training.** See §5.4 for the hybrid eval protocol.

---

## 3. Architectures

The "joint instance seg + classification" goal has two subproblems with different existing infrastructure:

| Subproblem | Existing infrastructure | New code needed |
|-----------|------------------------|------------------|
| Instance segmentation | starpose adapters: `StarDist`, `Cellpose 4 (CPSAM)`, `InstanSeg`, `CellViT` (H&E) | Extend `Segmenter` ABC → `JointSegClassifier` ABC |
| Per-instance class head | dapidl `popv_ensemble.VotingResult`, `MEDIUM_CLASS_NAMES`, `FINE_TO_MEDIUM_MAPPING`, `CL_DEPTH` | Trainable per-instance head producing `VotingResult` |

### 3.1 Output schema (binds the two subproblems)

Every predicted instance produces a row matching dapidl's existing `VotingResult` (`popv_ensemble.py:233`):

```python
@dataclass
class VotingResult:
    cell_id: str                            # synthetic ID for predicted instance
    winner: str                             # fine label (1 of ~17)
    winner_cl_id: str | None                # CL ontology ID
    broad_category: str                     # COARSE label (1 of 3 + Endothelial)
    medium_category: str                    # MEDIUM label (1 of 10)
    n_votes: int = 1                        # 1 (single-model)
    n_agreement: int = 1
    agreement_ratio: float                  # softmax score for `winner`
    confidence_scores: list[float]          # full softmax over fine classes
    voting_strategy: str = "instance_seg_v1"
    ontology_depth: int                     # CL depth of winner
    per_method_predictions: {"instance_seg_v1": winner}
    per_method_confidences: {"instance_seg_v1": agreement_ratio}
```

This means the new model becomes "one more voter" in dapidl's PopV ensemble — no new consumer code needed.

### 3.2 Joint candidates (trained end-to-end)

Reframed from v1 (which mixed apples and oranges):

#### 3.2.1 CelloType (Apache-2.0, MaskDINO + Swin-T) — **primary joint candidate**
- The **only** architecture in our shortlist with truly shared encoder + joint mask + class output.
- Pre-trained on PanNuke; we replace the class head with a 17-fine-class head.
- Repo: https://github.com/tanlabcode/CelloType.
- **Add to starpose** as `methods/cellotype.py` implementing a new `JointSegClassifier` ABC (extends `Segmenter` with `class_logits` output).
- Pilot phase trains this first.

#### 3.2.2 starpose backbone + class head (CelloType fallback)
- If CelloType doesn't pan out: train starpose's StarDist or Cellpose 4 CPSAM seg head jointly with a small per-instance ROI-pool MLP class head.
- Two-stage option: freeze seg → train class head on extracted ROIs (degenerate joint training, but fast).

### 3.3 Inference-only baselines (segmentation quality anchor)

Run starpose adapters on the same tile cache **without classification** to get reference PQ/AJI numbers:

| Method | Adapter | Use |
|--------|---------|-----|
| StarDist 2D versatile_fluo | `starpose.methods.stardist.StarDistAdapter` | Seg-only baseline |
| Cellpose 4 CPSAM | `starpose.methods.cellpose.CellposeAdapter` | Seg-only SOTA baseline |
| InstanSeg | `starpose.methods.instanseg.InstanSegAdapter` | Seg-only embedding-based baseline |

These produce class-agnostic segmentation PQ only — no class predictions. Used to anchor "what good seg looks like" before judging joint models.

### 3.4 GPU memory budget (RTX 3090, 24 GB)

| Architecture | Tile size | Batch | Est. VRAM | Fallback |
|--------------|-----------|-------|-----------|----------|
| StarDist (seg-only baseline) | 1024² | 4 | ~8 GB | batch=2 |
| Cellpose CPSAM (seg-only baseline) | 1024² | 2 | ~12 GB | batch=1 |
| InstanSeg (seg-only baseline) | 1024² | 2 | ~10 GB | batch=1 |
| CelloType (joint) | 1024² | 1 | ~16 GB | 768²/batch=1 |
| starpose+class head (joint fallback) | 1024² | 2 | ~14 GB | batch=1 |

**Pre-flight check before every training run** (existing pattern from `breast_full_annotation.py`):
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
# require ≥4 GB headroom over estimate; abort with clear error if not
```

Trainings run **sequentially**, never two seg models at once on this 3090. OOM auto-fallback in trainer base class: catch `torch.cuda.OutOfMemoryError`, halve batch, retry, max 2.

---

## 4. Methodology

### 4.1 Pilot phase
- Train **CelloType** (primary joint candidate) on slides {s0, s1, s3}, hold out s6 as test.
- Run starpose seg-only baselines (StarDist / Cellpose-SAM / InstanSeg) on same split for anchor metrics.
- If CelloType pilot fails: add starpose+class-head fallback as second joint candidate.
- Pick winner by `0.5 * panopticPQ + 0.3 * F1_macro_medium + 0.2 * AJI+`.

### 4.2 LOTO phase (winner only)
Leave-one-tissue-out across the 4 slides (4 trainings). Mirrors `sthelar_loto_*` pattern. Reports per-slide test metrics + mean ± std.

**Total trainings:** 1 (pilot CelloType) + 3 (seg-only baselines, 1 epoch each, fast) + 4 (LOTO winner) = **8 runs**, only one of which is a full training.

### 4.3 Splits (HIGH 5 fix)

**v1 was wrong**: claimed "tile-level split avoids leak from overlapping tiles". Empirically (verified 2026-05-02 on s3): random tile-level 80/20 split with 50% overlap leaks **99.7% of val cells back into train**.

**v2 split protocol:**
1. **Spatial-block parent split first.** Divide each slide into 4096²-pixel parent blocks. Assign 80% blocks to train, 20% to val (stratified by slide).
2. **Generate child tiles inside each split.** Stride-1024 child tiles within each parent block. Optional second pass with stride 768 (25% overlap) inside train blocks only — never across the train/val parent boundary.
3. **Train/val/test:** within-slide spatial-block split (train+val) + held-out slide (test, LOTO).
4. **Hard assertion:** `train_cell_ids ∩ val_cell_ids == ∅`. Fails Phase A if violated.

### 4.4 Augmentation (Albumentations 2.0.8 — MEDIUM 8 fix)

v1 used names that don't exist in Albumentations ≥2.0. Corrected:

| v1 name (broken) | v2 name (correct) | Train prob | Notes |
|------------------|--------------------|------------|-------|
| HFlip / VFlip / Rot90 | `HorizontalFlip`, `VerticalFlip`, `RandomRotate90` | 0.5 each | safe for nuclei |
| RandomBrightnessContrast | `RandomBrightnessContrast` | 0.6 | ±40% |
| GammaCorrection | **`RandomGamma`** | 0.4 | range 0.7–1.5 |
| GaussNoise | `GaussNoise` | 0.4 | var 50–500 |
| GaussBlur | **`GaussianBlur`** | 0.2 | sigma 0.5–1.5 |
| ElasticTransform | `ElasticTransform` | 0.2 | α=30, σ=4 |
| RandomScale | `RandomScale` | 0.3 | ±15% |
| Cutout | **`CoarseDropout`** | 0.2 | 1 hole, 32×32 |

**Per-image normalization:** percentile-based (p1 → 0, p99 → 1), then standardize (validated on MERSCOPE 2024).

### 4.5 Loss

Per-architecture seg loss + class-weighted CE on the class head:
- CelloType: MaskDINO (Hungarian matching) + class CE per fine class.
- starpose+class-head fallback: native seg loss + per-instance ROI-pool → MLP → CE.

**Class weight cap:** `max_weight_ratio=10.0` (DAPIDL pattern from `losses.py`). Without this, rare classes (NK, Mast at <1%) get extreme weights and destabilize training.

**Multi-tier supervision:** loss applied at fine-class level only; medium/broad are deterministic post-hoc lookups via existing `FINE_TO_MEDIUM_MAPPING` + `map_to_broad_category_4class`. Confidence at medium/broad = sum of softmax over members.

### 4.6 Sampler weighting (LOW 11 fix)

**v1 plan:** WeightedRandomSampler over tiles, weight = inverse frequency of rarest non-Other class. Problem: with overlap, the same rare nucleus is over-sampled.

**v2 plan:**
1. **Spatial-block weighting.** Compute weights at the parent-block level (where each rare cell appears once), not per overlapping tile.
2. **Per-tile cap.** Each tile contributes at most 1 sample per epoch per parent block.
3. **Diagnostic logging.** Track per-class duplicate-exposure ratio per epoch in W&B; alert if any class is sampled >5× per source instance per epoch.

---

## 5. Evaluation

### 5.1 Metrics (MEDIUM 9 fix)

| Metric | Class-aware? | Threshold | Why |
|--------|--------------|-----------|-----|
| **Panoptic Quality (PQ)** | yes (panopticPQ) | IoU 0.5 | joint seg+class quality |
| **Segmentation PQ** | no (class-agnostic) | IoU 0.5 | seg quality only — comparable to baselines |
| **AJI+** | no | — | nuclei-specific, robust to over/under-seg |
| **AP@0.5** primary | yes | IoU 0.5 | detection accuracy at strict overlap |
| **AP@[0.5:0.95]** appendix | yes | sweep | COCO-style rigor for paper appendix |
| **Per-class F1 (medium tier)** | yes | IoU 0.5 | the metric that matters operationally |
| **Confusion matrix (medium tier)** | yes | — | diagnose Plasma↔B-cell, Mono↔Macro |

Implemented via `pycocotools` (AP) + custom PQ/AJI+ in `src/dapidl/benchmark/instance_metrics.py` (new).

### 5.2 Matching protocol (canonical, MEDIUM 9c)

1. **Slide-level stitching.** Predict on overlapping tiles, stitch predictions with NMS (IoU 0.5 keeps higher-confidence detection).
2. **Eval against full-slide GT.** Match stitched predictions to GT polygons (rasterized at slide level via the same Phase A pipeline) by IoU ≥ threshold.
3. **Unmatched predictions** = false positives for their predicted class.
4. **Unmatched GT** = false negatives for the GT class.
5. **Border handling:** none needed at slide level (only matters for tile-level metrics; reserved for smoke-test debug).

### 5.3 Sanity checks before reporting
- Test slide has all 11 fine classes represented (≥10 instances each); else exclude that class from per-class F1 (per-class numbers reported with N=).
- `train_cell_ids ∩ val_cell_ids == ∅` (asserted in Phase A).
- Per-tile rows distribution logged (with overlap, per-tile rows ≫ unique cells; verified empirically: 4× ratio at stride 512 on s3).

### 5.4 Cross-source eval — hybrid (MEDIUM 10 fix)

v1 was qualitative-only — codex flagged this as too weak for a manuscript "cross-source generalization" claim. v2 hybrid:

**Quantitative (defensible for manuscript):**
- Use **Janesick et al. 2023 supervised cell type GT** for Xenium breast rep1 (167,780 cells) — already loaded by `pipeline/components/annotators/ground_truth.py`.
- Apply trained model to rep1, compute per-class F1 at **medium-tier** (10 classes, well-supported by Janesick GT).
- Report on rep2 too if Janesick GT available; else qualitative for rep2.

**Qualitative (boundary visualization):**
- 8 representative ROIs (4 from rep1, 4 from rep2), varied tissue contexts.
- Render cascade: DAPI → predicted instance mask (colored by class) → marker overlay (CD3, CD20, EPCAM transcripts).
- Use existing `scripts/_cascade_render.py`.

**Manuscript-grade boundary GT on Xenium is deferred** to follow-up; documented in §10.

---

## 6. Implementation sequence

**Time estimate (revised v2):** ~14 days at full pilot + LOTO. Phase 0 (preflight) added; Phase B is bigger (cache + spatial-block split logic); Phase C focuses on CelloType only initially.

### Phase 0 — Preflight + one-slide dry-run (Day 1)

**Files to create:**
- `scripts/instance_seg/preflight.py` — single command:
  - validates per-slide DAPI channel selection (HIGH 4)
  - loads geometry via PyArrow + GeoPandas, joins by `cell_id` (BLOCKER 3)
  - prints per-slide class counts at all 3 tiers, fails on missing classes (BLOCKER 1)
  - estimates cache size, validates against free disk (BLOCKER 2)
  - validates polygon validity, reports invalid count per slide (HIGH s6)
- `scripts/instance_seg/dry_run_one_slide.py` — materialize s3 only, measure:
  - actual zarr compressed size per tile (decides stride 1024 vs 768)
  - rasterization throughput (tiles/sec)
  - dataloader throughput (batches/sec on dummy model)

**Validation:** preflight exits 0; dry-run produces a manifest with measured per-tile compressed size; commit the chosen stride.

### Phase A — Tile cache (Day 2–3)

**Files to create:**
- `src/dapidl/data/sthelar_tile_cache.py` — `TileCacheBuilder` class with spatial-block parent split (HIGH 5).
- `scripts/instance_seg/build_tile_cache.py` — CLI: `--slides s0 s1 s3 s6 --stride 1024 --dtype uint16 --out /mnt/work/datasets/derived/sthelar_breast_tiles/`.
- `src/dapidl/data/instance_rasterizer.py` — reusable polygon→pixel rasterizer using `rasterio.features.rasterize` with `make_valid` repair.

**Files to modify:**
- `src/dapidl/data/sthelar.py` — add `select_dapi_channel(omero_attrs)` helper (HIGH 4).

**Validation (HIGH 6 fix):**
- Two separate sanity checks:
  1. **Unique cell IDs** in non-border tiles == `table_nuclei.shape[0]` after documented drops (`make_valid` failures, less10/Unknown filter).
  2. **Per-tile row distribution** logged (mean, P95, max). Allowed to exceed table row count if overlap is enabled; document the inflation factor.
- `train_cell_ids ∩ val_cell_ids == ∅` (hard assertion).

**Output:** `/mnt/work/datasets/derived/sthelar_breast_tiles/` (estimated 5–30 GB after compression; see Phase 0 measurement).

### Phase B — Dataset + augmentation (Day 3–4)

**Files to create:**
- `src/dapidl/data/sthelar_instance_dataset.py` — `STHELARInstanceDataset(torch.utils.data.Dataset)` over the tile cache. Returns `dict(image, instance_map, fine_labels, medium_labels, broad_labels, boxes)`.
- `src/dapidl/data/instance_augment.py` — Albumentations 2.0 pipeline (corrected names, MEDIUM 8) with mask-aware transforms.

**Validation:**
- Iterate 100 batches, no crashes.
- 10 augmented samples → instance map alignment preserved (visual check).

### Phase C — Trainers (Day 4–9)

**Files to create:**
- `src/dapidl/training/instance/__init__.py`
- `src/dapidl/training/instance/base_trainer.py` — OOM auto-fallback, GPU pre-flight, file lock, checkpoint top-3.
- `src/dapidl/training/instance/cellotype_trainer.py` — primary.
- `src/dapidl/training/instance/seg_baseline_trainer.py` — wraps starpose adapters for inference-only eval.
- `src/dapidl/training/instance/starpose_class_head_trainer.py` — fallback if CelloType fails.
- `scripts/instance_seg/train_cellotype.py`
- `scripts/instance_seg/run_seg_baselines.py` — runs the three starpose adapters on the test slide; ~1 hour total.

**starpose extension:**
- Add `JointSegClassifier` ABC to starpose (`src/starpose/methods/joint.py`) with `class_logits` output.
- Add `methods/cellotype.py` adapter to starpose.
- Bump starpose to 0.3.0 (minor; backward compatible — `Segmenter` ABC unchanged).

**Pilot run command:**
```bash
uv run python scripts/instance_seg/train_cellotype.py \
  --tile-cache /mnt/work/datasets/derived/sthelar_breast_tiles \
  --train-slides s0 s1 s3 \
  --test-slide s6 \
  --epochs 50 --early-stop-patience 10 \
  --out pipeline_output/instance_seg/cellotype/pilot_v1
```

### Phase D — Eval + LOTO (Day 9–14)

**Files to create:**
- `src/dapidl/benchmark/instance_metrics.py` — PQ (panoptic + seg), AJI+, AP@0.5 + AP@[0.5:0.95].
- `scripts/instance_seg/stitch_predictions.py` — slide-level NMS stitching (MEDIUM 9c).
- `scripts/instance_seg/eval_pilot.py`.
- `scripts/instance_seg/run_loto.sh` — 4 LOTO trainings of CelloType.
- `scripts/instance_seg/eval_xenium_hybrid.py` — Janesick GT class F1 (rep1) + qualitative cascades (MEDIUM 10).
- `scripts/figures/instance_seg_figs.py` — per-arch bars + per-class confusion + cascade gallery.

**Outputs:**
- `pipeline_output/instance_seg/comparison.parquet` — pilot results (CelloType + 3 seg-only baselines)
- `pipeline_output/instance_seg/loto_results.parquet` — winner LOTO (4 folds × test metrics)
- `pipeline_output/instance_seg/cascades/` — Xenium visual gallery
- Figures synced to obsidian per usual (`scripts/sync_manuscript_to_obsidian.sh`).

---

## 7. GPU memory safety

Embedded in the trainer base class:

1. **Pre-flight:** `nvidia-smi --query-gpu=memory.free` ≥ `estimated_vram + 4 GB`. Else abort with explicit "GPU full, free up X MB" message.
2. **OOM auto-fallback:** wrap forward+backward in try/except `torch.cuda.OutOfMemoryError`; on catch: `torch.cuda.empty_cache()`, halve batch, retry. Max 2 retries.
3. **No concurrent training:** trainers acquire a file lock at `/tmp/dapidl_seg_train.lock`.
4. **Per-epoch checkpoint:** loss of any single epoch never costs more than 1 epoch of work.
5. **W&B alerts:** notify on val PQ improvement and on consecutive-fail-to-improve >5.

---

## 8. Disk + RAM budget (revised v2)

### Disk

| Item | Size |
|------|------|
| Tile cache (4 slides, stride 1024, uint16, blosc-zstd-3 compressed) | **5–30 GiB** (measured in Phase 0) |
| StarDist checkpoints (top-3 × 1 baseline run) | ~0.2 GB |
| CelloType checkpoints (top-3 × 5 runs = 1 pilot + 4 LOTO) | ~7.5 GB |
| Cellpose CPSAM baseline (1 run) | ~1 GB |
| Eval outputs (predictions, cascades, figures) | ~10 GB |
| **Total** | **~25–50 GiB / 161 GB free** |

**Headroom:** ample (>100 GB free after build) — far better than v1's 73% utilization.

### RAM (per `feedback_anticipate_ram.md`)

The system has 62 GB and systemd-oomd kills the user slice at 90% pressure. Phase A cache build is the riskiest step:

- Loading a single 893k-cell `gdf` for s1 → ~700 MB (geometry only).
- Per-slide tile materialization is streaming (tile-by-tile), so peak RAM stays <8 GB.
- **Pre-flight RAM check** before any phase (existing pattern from `breast_full_annotation.py`).

After completion: tile cache stays (reusable for follow-on work), checkpoints kept top-3 per run, eval outputs synced to obsidian then prunable.

---

## 9. Risks (v2 — updated post-verification)

1. **s6 `ct_tangram` outlier — 29% Mast cells (NEW).** Verified 2026-05-02: s6 has 202,128 Mast cells out of 692,184 (29%), which is biologically implausible for breast tumor. Likely Tangram annotation noise specific to this slide. **Mitigation:** preflight emits per-slide class fraction warning if any single class >25% of non-`less10`/`Unknown` cells. Decision point: drop s6 from pilot, or weight it down, or flag in §10.
2. **s6 invalid polygon rate (1.6%, 11k polys).** vs ~0.07% on other slides. Mitigation: `make_valid` pipeline in §2.3 with documented drop accounting in manifest.
3. **s1 cell_id row mismatch.** verified 2026-05-02. Mitigation: cell_id-based join (mandatory in §2.3 loader) — never row-position.
4. **CelloType PanNuke pretraining mismatch.** PanNuke 5-class doesn't map to our 17 fine. Encoder transfer still useful; class head re-init expected. Document warm-start details in W&B run config.
5. **Cellpose-SAM CC-BY-NC license** (only relevant if used as fallback). Document explicitly; don't make it a manuscript main result without permissive backbone alternative.
6. **Manuscript-grade Xenium boundary GT deferred** (MEDIUM 10). v2 reports quantitative class F1 via Janesick rep1 GT; boundary quality is qualitative. Annotate ROIs in follow-up if reviewer pushes back.
7. **Polars cannot read GeoArrow WKB.** Verified PanicException. Mitigation: PyArrow + GeoPandas in geometry loader; documented in §2.3.
8. **Pilot fairness** (codex risk #7). v2 narrows pilot to 1 joint candidate (CelloType) + 3 seg-only baselines (fast). Removes the wall-clock-vs-epoch fairness problem. If CelloType fails, fallback (starpose + class head) is added as 2nd joint candidate at that point.
9. **Training-time ETA.** v1 at 5–6 days assumed wrong cell counts. v2 at ~14 days, dominated by Phase C (~5 days CelloType pilot) + Phase D (~3 days × 4 LOTO).

---

## 10. Out of scope (Phase 2+)

- Cell / cytoplasm masks → use existing starpose Voronoi/watershed expansion on these nucleus instances.
- Multi-tissue (non-breast) instance seg → STHELAR has 16 tissues, easy extension once breast pilot lands.
- H&E modality fusion → STHELAR has matched H&E; deferred until DAPI-only baseline solid.
- Real-time inference / packaging → CLI for users to point at a Xenium mosaic and get annotated nuclei out.
- Manuscript-grade quantitative Xenium GT → manual ROI annotation (10–20 ROIs); deferred unless reviewer requests.

---

## 11. Sequencing relative to current work

The current `breast_finish_all.sh` orchestrator (started 2026-05-02 09:36) is on the cross-source evaluation step. Once it exits cleanly + obsidian sync done, Phase 0 (preflight) can start immediately on CPU.

Phase 0 + Phase A (tile cache) are CPU-bound → run in parallel with whatever finishing GPU work. Phase B (dataset code) similarly CPU-bound.

GPU contention check before Phase C: pause `clearml-agent-gpu-ubuntu3090` (currently idle on `gpu-local`/`gpu-training` queues) during instance seg training to avoid surprise jobs eating VRAM.

---

## Appendix A — Existing assets we reuse

### From dapidl
- `src/dapidl/data/sthelar.py` — STHELAR zarr reader, label mappings, `STHELAR_TO_DAPIDL_COARSE`, `TANGRAM_TO_COARSE`.
- `src/dapidl/pipeline/components/annotators/popv_ensemble.py` — `VotingResult`, `MEDIUM_CLASS_NAMES`, `FINE_TO_MEDIUM_MAPPING`, `CL_DEPTH`, `GranularityLevel` enum.
- `src/dapidl/pipeline/components/annotators/mapping.py` — `COARSE_CLASS_NAMES`, `FINEGRAINED_CLASS_NAMES`, `CELL_TYPE_HIERARCHY`, `map_to_broad_category_4class`.
- `src/dapidl/pipeline/components/annotators/cell_ontology_mapping.py` — `CellOntologyTerm`, `CELL_ONTOLOGY` dict.
- `src/dapidl/pipeline/components/annotators/ground_truth.py` — Janesick rep1 GT loader for §5.4 quantitative eval.
- `src/dapidl/training/losses.py` — `max_weight_ratio=10` pattern for class-weighted CE.
- `scripts/_cascade_render.py` — cascade visualization utility.
- `scripts/sync_manuscript_to_obsidian.sh` — figure sync.

### From starpose (`/home/chrism/git/starpose/`, editable install)
- `methods/{stardist,cellpose,instanseg,cellvit}.py` — `Segmenter` ABC implementations (seg-only baselines).
- `methods/cellvit.py:STHELAR_SCALE_FACTOR` — µm→px conversion (= 4.705882352941177).
- `expansion/{voronoi,watershed,proseg}.py` — `CellExpander` ABC (Phase 2 only).

### New (must build)
- `src/dapidl/data/sthelar_tile_cache.py`, `instance_rasterizer.py`, `sthelar_instance_dataset.py`, `instance_augment.py`
- `src/dapidl/training/instance/{base_trainer,cellotype_trainer,seg_baseline_trainer,starpose_class_head_trainer}.py`
- `src/dapidl/benchmark/instance_metrics.py`
- `scripts/instance_seg/{preflight,dry_run_one_slide,build_tile_cache,train_cellotype,run_seg_baselines,stitch_predictions,eval_pilot,run_loto.sh,eval_xenium_hybrid}.py`
- starpose: `methods/joint.py` (`JointSegClassifier` ABC), `methods/cellotype.py`

### New deps
- `rasterio` (polygon rasterization)
- `pycocotools` (AP metrics)
- `geopandas`, `pyarrow` (already pulled by other deps; explicit pin needed)

---

## Appendix B — Decisions log (Q1–Q5)

| Q | Question | Answer | Rationale |
|---|----------|--------|-----------|
| 1 | Nuclei first vs cells first vs both | **D — Nuclei only** | Nucleus is the universal target; expansion via existing starpose later. |
| 2 | Train data: STHELAR? Xenium? Both? | **D — STHELAR train + Xenium hybrid eval** | v2 adds Janesick rep1 GT for quantitative class F1; boundary GT deferred. |
| 3 | Architecture choice | **C → narrowed: CelloType primary; starpose seg adapters as baselines** | Codex right that v1's 3 archs weren't equivalent. Pilot just CelloType; baseline anchors via starpose. |
| 4 | Class set | **A3 → mapped via dapidl 3-tier system** | Source = `ct_tangram` (BLOCKER 1 fix), output = dapidl `VotingResult` with broad/medium/fine + CL depth. |
| 5 | Methodology depth | **C — Pilot then LOTO on winner** | Same as v1. Pilot is 1 joint + 3 seg baselines (fast). |

---

## Appendix C — Codex review responses (v2 disposition)

| # | Severity | Item | v2 disposition |
|---|----------|------|----------------|
| 1 | BLOCKER | 11-class target from `label1`+`label2` | **FIXED** §2.2: switched to `ct_tangram` → dapidl 3-tier mapping |
| 2 | BLOCKER | Cache size estimate too low | **FIXED** §2.5 + §8: stride 1024 + uint16 default; one-slide dry run before commit |
| 3 | BLOCKER | Geometry ingestion wrong | **FIXED** §2.3: PyArrow+GeoPandas, cell_id join, `make_valid`, `STHELAR_SCALE_FACTOR` µm→px |
| 4 | HIGH | s6 multi-channel | **FIXED** §2.4: OMERO label-based DAPI selection |
| 5 | HIGH | Tile-level split leaks | **FIXED** §4.3: spatial-block parent split before generating overlapping child tiles; hard assertion |
| 6 | HIGH | Cache count check vs ±2% | **FIXED** §6 Phase A validation: two separate metrics (unique cell IDs vs per-tile row distribution) |
| 7 | HIGH | Architectures not equivalent | **FIXED** §3: starpose seg-only baselines vs CelloType joint candidate; fallback only if CelloType fails |
| 8 | MEDIUM | Deps + Albumentations names | **FIXED** §4.4 + §A: corrected transform names; rasterio/pycocotools added; stardist via starpose extras |
| 9 | MEDIUM | Eval matching semantics | **FIXED** §5.1 + §5.2: panoptic-PQ + seg-PQ; IoU 0.5 primary, 0.5–0.95 appendix; slide-level NMS stitching |
| 10 | MEDIUM | Xenium eval too weak | **FIXED** §5.4: hybrid (Janesick GT class F1 + qualitative boundaries); manuscript-grade boundary GT deferred §10 |
| 11 | LOW | Sampler over-samples duplicates | **FIXED** §4.6: spatial-block weighting + diagnostic logging |

### v2-discovered issues not in codex review
- **s6 ct_tangram has 29% Mast cells** — biologically implausible, added as §9 risk #1
- **Polars panics on geoarrow.wkb** — PyArrow+GeoPandas mandatory, documented §2.3 + §9 risk #7
- **Geometry is in microns, not pixel-space** — original doc was wrong on units too, not just location

---

*End of design v2. Ready for Phase 0 implementation.*
