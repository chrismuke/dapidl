# Nucleus Quality Control + Inspection Montage — Design

- **Date:** 2026-05-20
- **Status:** Draft (awaiting user review)
- **Topic:** Automatic QC scoring of DAPI nucleus patches, plus a ClearML-surfaced inspection montage

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
- Per-patch quality scores capturing the two failure modes separately:
  `focus_score`, `detection_score`, and a combined `qc_score`.
- A **post-hoc, re-runnable** pass over already-built datasets (LMDB/Zarr +
  `metadata.parquet`) — score the data we already have without rebuilding.
- **Score + flag, keep everything.** Nothing is dropped automatically. Scores are
  written back so we can filter/weight at train time later.
- **Per-class, worst-QC-first inspection montages** logged to ClearML, each tile
  labeled with cell type + score.
- The scorer is a reusable **starpose** capability (library + CLI); the dataset
  pass / montage / ClearML logging are **dapidl** glue.
- Fast: classical scorer < 1 ms/patch on CPU, batchable. No model loading for v1.

### Non-Goals (v1)
- No automatic filtering or dropping of patches.
- No foundation-model scorer in v1 (the ABC leaves a seam for it — see §9).
- No changes to the training loop / loss / sampler.
- No extraction-time scoring hook in v1 (post-hoc only; see §9).
- No new interactive UI beyond ClearML's image viewer.

## 3. Architecture — starpose / dapidl split

**starpose owns the scorer** (reusable image science, no dataset/label concepts):

```
src/starpose/qc/
  __init__.py
  base.py       # QualityScorer ABC + QualityScore dataclass (mirrors methods/base.py)
  classical.py  # ClassicalQualityScorer (default v1)
  # embedding.py  ← future FM scorer (optional torch), not in v1
```

- CLI: add a `qc` command to `src/starpose/cli.py` (typer): score a directory of
  patches or an image+centroids, emit a scores table (parquet).

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

starpose is already a dependency of dapidl (editable install in `pyproject.toml`),
so dapidl importing `starpose.qc` requires no new wiring.

## 4. Components

### 4.1 `starpose.qc.base`

```python
@dataclass(frozen=True)
class QualityScore:
    focus_score: float       # 0..1, higher = sharper
    detection_score: float   # 0..1, higher = clearer central nucleus
    qc_score: float          # 0..1 combined (see §6)
    metrics: dict[str, float]  # raw named metrics for debugging/calibration

class QualityScorer(ABC):
    @abstractmethod
    def score_batch(self, patches: np.ndarray) -> list[QualityScore]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
```

Mirrors the existing `Segmenter`/`CellExpander` ABC pattern in
`src/starpose/methods/base.py`. `score_batch` takes `(N, H, W)` (or `(N, H, W, C)`)
and returns one `QualityScore` per patch. A `score(patch)` convenience wraps the batch
form.

### 4.2 `starpose.qc.classical.ClassicalQualityScorer`

- **focus_score**: variance-of-Laplacian (primary) and Tenengrad (gradient energy).
  Both rise with edge sharpness. Combined and squashed to 0..1 against a per-call
  reference distribution (see §6 normalization).
- **detection_score**: Otsu threshold → foreground fraction, plus a central-blob
  check (is there a single bright, roughly-central connected component, vs. empty /
  edge-only / multi-blob clutter?). A real nucleus crop has a bright central blob
  occupying a sane fraction of the patch.
- Pure `numpy` / `scipy` / `scikit-image` (already starpose core deps). uint16→float
  internally. Target < 1 ms/patch; vectorized across the batch where possible.
- Stateless except for an optional fitted normalization reference passed in by the
  caller (so per-slide stats can be supplied).

### 4.3 `dapidl.pipeline.steps.quality_control` (the post-hoc pass)

Reads an existing built dataset, scores it, writes scores back, builds montages,
logs to ClearML. Reuses dapidl's existing patch read path (the same LMDB/Zarr reader
used by the dataset classes / `compute_dataset_stats`). Exact read API to be confirmed
at implementation time against `src/dapidl/data/dataset.py`.

### 4.4 `dapidl.qc.montage`

Builds a per-class grid: sort that class's patches by `qc_score` ascending, take the
worst N (default 64), contrast-normalize each tile for visibility, tile into a grid
(e.g. 8×8), overlay `celltype + score` text per tile. Returns an RGB image array.

### 4.5 CLI

- `starpose qc --patches <dir|zarr> [--out scores.parquet]` — generic scorer.
- `dapidl qc --dataset <path> [--montage-top-n 64] [--no-clearml]` — the dapidl pass:
  scores the dataset, writes scores back, builds montages, logs to ClearML.

## 5. Data flow

```
existing dataset (patches.zarr OR patches.lmdb)
   + metadata.parquet (cell_id, predicted_type, broad_category, confidence, x/y centroid)
   + sources.npy (slide/source per patch, for pooled datasets)
   + class_mapping.json
        │
        ▼
dapidl qc step
   1. group patch indices by slide/source
   2. per slide: load patches → per-slide normalize → starpose ClassicalQualityScorer.score_batch
   3. assemble focus_score / detection_score / qc_score arrays (dataset order)
   4. WRITE BACK:
        - add focus_score, detection_score, qc_score columns to metadata.parquet
        - write qc_scores.npy (parallel to confidence.npy) for loader consumption later
   5. per cell-type: worst-first montage grid (dapidl.qc.montage)
   6. ClearML:
        - report_image(montage) per class
        - report_histogram(qc_score) per class
        - upload_artifact(scores table)
```

Re-runnable and idempotent (overwrites scores/montages). Works for single-slide
datasets (one group) and pooled multi-source datasets (group by `sources.npy`).

## 6. Scoring detail

- **Combined qc_score**: `qc_score = min(focus_score, detection_score)` for v1 — a
  patch is good only if it is *both* in focus *and* contains a real nucleus. (A
  weighted geometric mean is an alternative; min is the conservative, interpretable
  default and is easy to reason about in the montage.)
- **Per-slide normalization (the #1 correctness risk).** DAPI intensity varies ~16×
  across platforms (CLAUDE.md: MERSCOPE median ~13000 vs Xenium ~800). Absolute
  thresholds do not transfer. Within each slide we convert raw focus/detection metrics
  to robust 0..1 scores against that slide's own distribution (e.g. rank or
  median/MAD-based squashing). This is why the pass groups by slide before scoring.

## 7. Storage / schema changes

- `metadata.parquet`: add `focus_score`, `detection_score`, `qc_score` (float).
- `qc_scores.npy`: float array parallel to `confidence.npy` / `labels.npy`, so future
  dataset loaders can filter/weight by it without parsing parquet. Matches the existing
  `confidence_levels.npy` convention in `multi_tissue_dataset.py`.
- Montage PNGs are written to `<dataset>/qc/montage_<class>.png` (also when ClearML is
  off), so the artifacts always exist on disk regardless of logging.
- No existing columns/files are modified or removed.

## 8. Error handling

- Missing `metadata.parquet` or patch store → explicit error naming the dataset path.
- uint16 patches → cast to float for metrics; never mutate the stored patches.
- No slide/source column on a multi-slide dataset → error (per-slide normalization
  cannot be done safely); single-slide datasets normalize as one group.
- Empty class (0 patches) → skip its montage, log a warning, continue.
- ClearML unavailable / `--no-clearml` → still write scores + save montages to disk.

## 9. Testing

- **starpose unit tests** (`tests/test_qc_classical.py`): synthetic patches —
  a sharp Gaussian blob scores high focus + detection; a Gaussian-blurred copy scores
  low focus, similar detection; flat noise scores low detection. Deterministic, no I/O.
- **dapidl tests**: a tiny fixture dataset (a handful of patches + metadata) exercised
  through the pass; montage builder returns an image of the expected shape; ClearML
  client mocked so no network.

## 10. Future work (explicitly out of scope for v1)

- **FM scorer** (`starpose.qc.embedding`): DINOv2-small embedding + kNN/Mahalanobis
  outlier detection, registered as an alternative `QualityScorer`. Caveat: outlier ≠
  low-quality (rare valid cell types look like outliers). Pluggable via the §4.1 ABC.
- **Extraction-time hook**: call the scorer in-line in `PatchExtractor.prepare_dataset`
  so new datasets are born with QC scores.
- **Training integration**: consume `qc_scores.npy` as a filter threshold or sample
  weight (alongside the existing Tier 1/2/3 confidence weighting).
- **Pipeline step**: register the pass as a ClearML pipeline step so QC runs
  automatically after dataset build.

## 11. Open questions

- Exact LMDB/Zarr read API to reuse for loading patches back (confirm against
  `src/dapidl/data/dataset.py` at implementation time).
- Default montage tile count per class (64 proposed) and whether to upscale tiles for
  label legibility in the ClearML viewer.
