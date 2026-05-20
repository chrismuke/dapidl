# Nucleus Quality Control + Inspection Montage — Design

- **Date:** 2026-05-20
- **Status:** Draft (codex review incorporated; awaiting user review)
- **Topic:** Automatic QC scoring of DAPI nucleus patches, plus a ClearML-surfaced inspection montage
- **Companion:** `2026-05-20-nucleus-qc-design_codex_review.md`

## 1. Problem

DAPIDL trains a cell-type classifier on 128×128 DAPI patches centered on nucleus
centroids from spatial-transcriptomics segmentation. Many patches are low quality:
nuclei that are **out-of-focus** (blurry, real nucleus) or **falsely detected**
(a centroid where there is no well-formed nucleus). These degrade training data and
are currently invisible — there is no way to score or inspect them.

We want (a) an automatic, fast quality score per patch, and (b) a way to visually
inspect the worst patches per cell type, surfaced in ClearML.

## 2. Goals / Non-Goals

### Goals
- Per-patch quality scores capturing the failure modes separately:
  `focus_score`, `detection_score`, and a combined `qc_score`, **plus the raw metrics**.
- A **post-hoc, re-runnable** pass over already-built datasets (LMDB/Zarr +
  `metadata.parquet`) — score the data we already have without rebuilding.
- **Score + flag, keep everything.** Nothing is dropped automatically. Scores are
  written to a sidecar so we can filter/weight at train time later.
- **Per-class, worst-QC-first inspection montages** logged to ClearML, labeled with
  cell type + score.
- The scorer is a reusable **starpose** capability; the dataset pass / montage /
  ClearML logging are **dapidl** glue.
- Fast: classical scorer < 1 ms/patch on CPU, batchable. No model loading for v1.

### Non-Goals (v1)
- No automatic filtering or dropping of patches.
- No foundation-model scorer in v1 (the ABC leaves a seam for it — see §10).
- No changes to the training loop / loss / sampler.
- No extraction-time scoring hook in v1 (post-hoc only; see §10).
- No new interactive UI beyond ClearML's image viewer.
- **Known v1 limitation (codex review):** the classical scorer catches **blur** and
  **emptiness / false detection**, but NOT sharp non-nucleus artifacts (debris, dust,
  bright speckles, stitching edges) that are simultaneously in-focus and blob-like.
  Those need the FM scorer or a dedicated artifact heuristic (deferred, §10). v1
  documents this rather than pretending to catch it.

## 3. Architecture — starpose / dapidl split

**starpose owns the scorer** (reusable image science, no dataset/label concepts):

```
src/starpose/qc/
  __init__.py
  base.py       # QualityScorer ABC + QualityScore dataclass (mirrors methods/base.py)
  classical.py  # ClassicalQualityScorer (default v1)
  # embedding.py  ← future FM scorer (optional torch), not in v1
```

- CLI: a `qc` command in `src/starpose/cli.py` (typer) kept deliberately **minimal —
  patch array(s) in, scores table out.** It does NOT scan dapidl dataset stores
  (LMDB/Zarr) and knows nothing about cell types; that I/O stays in dapidl (§4.3).
  This CLI is **deferred** unless a non-dapidl caller materializes — it is not on the
  v1 critical path.

**dapidl owns the glue** (dataset formats + cell-type labels + ClearML):

```
src/dapidl/qc/
  __init__.py
  montage.py    # per-class worst-first grid builder
src/dapidl/pipeline/steps/quality_control.py  # the post-hoc pass
# + a `dapidl qc` CLI command in src/dapidl/cli.py
```

**Rationale.** starpose already owns "everything about nuclei from images"
(segmentation backends in `methods/`, mask-shape QC in `evaluate/morphometric.py`).
Image-quality scoring of nucleus crops is squarely that domain and stays reusable.
dapidl owns dataset I/O, cell-type labels, and ClearML, so the montage (needs labels)
and ClearML logging live there. The scorer is shared; orchestration is project-specific.
starpose is already a dependency of dapidl (editable install in `pyproject.toml`).

**Normalization ownership:** score *calibration* is scorer semantics → starpose defines
how raw metrics become scores (incl. absolute floors). dapidl supplies the per-slide
**group keys** and the sampled **reference stats** (§6). starpose does not read datasets.

## 4. Components

### 4.1 `starpose.qc.base`

```python
@dataclass(frozen=True)
class QualityScore:
    focus_score: float       # 0..1, higher = sharper
    detection_score: float   # 0..1, higher = clearer real nucleus
    qc_score: float          # 0..1 combined (see §6)
    metrics: dict[str, float]  # RAW named metrics, always populated

class QualityScorer(ABC):
    @abstractmethod
    def score_batch(self, patches: np.ndarray, ref: NormRef | None = None) -> list[QualityScore]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
```

Mirrors the `Segmenter`/`CellExpander` ABC pattern in `src/starpose/methods/base.py`.
`score_batch` takes `(N, H, W)` and an optional normalization reference (per-slide stats
fitted by the caller). A `score(patch)` convenience wraps the batch form. The ABC is
intentionally thin; if the FM seam never lands it collapses cleanly to one class.

### 4.2 `starpose.qc.classical.ClassicalQualityScorer`

- **focus_score**: variance-of-Laplacian (primary) + Tenengrad. Combined into a 0..1
  score using BOTH a per-slide relative position AND a calibrated absolute floor (§6).
- **detection_score**: Otsu foreground fraction + central-blob check (a real nucleus
  crop tends to have a bright, roughly-central connected component of sane size).
- Pure `numpy` / `scipy` / `scikit-image` (already starpose core deps). Operates on
  **raw DAPI patches** (uint16→float internally), never normalized/augmented training
  tensors. Target < 1 ms/patch; vectorized across the batch.
- **Caveat (codex review):** the "bright central blob" heuristic can wrongly penalize
  valid off-center centroids, elongated/lobed/mitotic/doublet nuclei, edge nuclei, and
  crowded fields; Otsu is brittle under staining gradients and saturation. Therefore
  `detection_score` is a *soft* signal — the **raw metrics are always persisted** so a
  low score can be sanity-checked in the montage rather than trusted blindly. Saturated
  (clipped) nuclei must not be auto-classed as blurry.

### 4.3 `dapidl.pipeline.steps.quality_control` (the post-hoc pass)

Reads an existing built dataset, scores it, writes a sidecar, builds montages, logs to
ClearML. Reuses dapidl's existing patch read path (the same LMDB/Zarr reader used by the
dataset classes / `compute_dataset_stats`). Exact read API confirmed at implementation
time against `src/dapidl/data/dataset.py`.

### 4.4 `dapidl.qc.montage`

Per-class grid: sort that class's patches by `qc_score` ascending, take the worst N
(default 64), contrast-normalize each tile, tile into a grid (e.g. 8×8). **Labels
(`score`, optionally `cell_id`) render in a margin strip beside/under each tile, NOT
overlaid on the 128px pixels** (overlaid text on a 128px tile is unreadable). Returns an
RGB image array. Per-class tile count is **capped** (configurable) to avoid flooding
ClearML; a per-class score histogram summarizes the full distribution.

### 4.5 CLI

- `starpose qc` — minimal, array in / scores out (deferred; §3).
- `dapidl qc --dataset <path> [--montage-top-n 64] [--no-clearml]` — the dapidl pass:
  scores the dataset, writes the sidecar, builds montages, logs to ClearML.

## 5. Data flow

```
existing dataset (patches.zarr OR patches.lmdb)
   + metadata.parquet (cell_id, predicted_type, broad_category, confidence, x/y centroid)
   + sources.npy (candidate per-slide grouping key — verify, §11)
   + class_mapping.json
        │
        ▼
dapidl qc step
   1. group patch indices by slide/source (true acquisition key)
   2. per slide: sample patches → fit NormRef → stream batches →
      starpose ClassicalQualityScorer.score_batch(patches, ref)   # never load whole slide
   3. assemble focus/detection/qc_score + raw metrics arrays (dataset order)
   4. WRITE sidecar <dataset>/qc/qc_scores.parquet  (metadata.parquet untouched)
        + provenance (scorer name/params/floors/version/date)
   5. per cell-type: worst-first montage grid (dapidl.qc.montage)
   6. ClearML: report_image(montage) per class (capped)
              + report_histogram(qc_score) per class
              + upload_artifact(qc_scores.parquet)
```

Re-runnable and idempotent (atomic overwrite of the sidecar/montages). Works for
single-slide datasets (one group) and pooled multi-source datasets.

## 6. Scoring detail

- **Combined qc_score**: `qc_score = min(focus_score, detection_score)` for v1 — a patch
  is good only if it is *both* in focus *and* contains a real nucleus. `min` is the
  conservative, interpretable default; because a noisy `detection_score` can dominate,
  the **component scores and raw metrics are always surfaced** (montage + sidecar) and
  `qc_score` is never treated as the sole signal.
- **Per-slide normalization (the #1 correctness risk).** DAPI intensity varies ~16×
  across platforms (CLAUDE.md: MERSCOPE median ~13000 vs Xenium ~800). Absolute
  thresholds do not transfer between slides → relative scoring within a slide.
- **Relative + absolute (codex review).** Pure per-slide ranking is misleading: an
  all-bad slide still produces high relative scores, and an all-good slide manufactures
  artificial "worst" cases. So we (a) always persist the **raw metrics**, and (b) combine
  the per-slide relative score with a calibrated **absolute floor** (e.g. a
  variance-of-Laplacian level below which a patch is blurry regardless of slide). Floors
  are scorer parameters recorded in provenance (§7).
- **Memory (62 GB host).** Never load a whole slide's patches at once. Fit the
  normalization reference from a **random sample** of the slide's patches, then stream
  the remainder in batches applying the fixed reference (sampled-reference, single pass).

## 7. Storage / schema

- **Sidecar `<dataset>/qc/qc_scores.parquet`** keyed by `cell_id`, columns:
  `focus_score`, `detection_score`, `qc_score`, plus raw metrics (`var_laplacian`,
  `tenengrad`, `foreground_frac`, `central_blob_frac`, ...). **`metadata.parquet` is NOT
  modified** — avoids mutating a primary artifact and sidesteps overwrite/compat issues.
  Join back by `cell_id` when needed.
- **Provenance** in the sidecar (or `qc/qc_scores.meta.json`): scorer `name`, parameters
  (incl. absolute floors + normalization method), starpose/dapidl code version, run date.
- **Re-run policy:** overwrite atomically (write temp + rename). Idempotent.
- **Defer `qc_scores.npy`** until training integration consumes it (§10) — a
  loader-optimized array now would store only the combined score and duplicate the
  parquet. The parquet suffices for scoring + montage.
- Montage PNGs → `<dataset>/qc/montage_<safe_class>.png` (filename-safe class names),
  also written when ClearML is off, so artifacts always exist on disk.
- **Nothing existing is modified or removed** — all output lives in the new `qc/` dir.

## 8. Error handling

- Missing `metadata.parquet` or patch store → explicit error naming the dataset path.
- uint16 patches → cast to float for metrics; never mutate stored patches; score the
  **raw** patches, not normalized/augmented training tensors.
- No usable per-slide grouping key on a multi-slide dataset → error (per-slide
  normalization cannot be done safely); single-slide datasets normalize as one group.
- Empty class (0 patches) → skip montage, warn, continue.
- ClearML unavailable / `--no-clearml` → still write sidecar + save montages to disk.

## 9. Testing

- **starpose unit tests** (`tests/test_qc_classical.py`): deterministic synthetic
  patches with explicit expected orderings —
  - sharp Gaussian blob → high focus + high detection
  - Gaussian-blurred blob → low focus, detection roughly unchanged
  - flat noise / empty → low detection
  - **hard negatives:** saturated/clipped blob (must NOT be classed as blurry), sharp
    debris speckle (documented false-negative — assert known behavior), off-center valid
    nucleus (must not be over-penalized), crowded multi-nucleus patch.
  - **normalization:** an all-bad batch must NOT manufacture high absolute scores (the
    absolute floor holds).
- **dapidl tests**: a tiny fixture dataset exercised through the pass; sidecar parquet
  has expected columns + provenance; montage builder returns an image of expected shape
  with labels rendered in the margin; ClearML client mocked so no network.

## 10. Future work (out of scope for v1)

- **FM scorer** (`starpose.qc.embedding`): DINOv2-small embedding + kNN/Mahalanobis
  outlier detection, registered as an alternative `QualityScorer`. Also the natural way
  to catch the sharp-non-nucleus-artifact mode v1 misses. Caveat: outlier ≠ low-quality
  (rare valid cell types look like outliers). Pluggable via the §4.1 ABC.
- **Extraction-time hook**: call the scorer in-line in `PatchExtractor.prepare_dataset`.
- **Training integration**: consume scores as a filter threshold or sample weight
  (alongside the existing Tier 1/2/3 confidence weighting); add `qc_scores.npy` then.
- **Pipeline step**: register the pass as a ClearML pipeline step so QC runs after build.

## 11. Open questions

- Exact LMDB/Zarr read API to reuse for loading patches back (confirm against
  `src/dapidl/data/dataset.py` at implementation time).
- Confirm the per-slide grouping key: `sources.npy` may be a *source* label not
  one-to-one with an acquisition/slide. Verify it is the right normalization group; if
  not, derive a true slide key.
