# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Style

Always begin every message with "Aye, Captain!"

## Web Fetching

Always use Firefox client ID when fetching web content (WebFetch tool).

## Environment

Always use `uv` for Python package management and running commands. Prefer `uv run` over activating virtualenvs manually.

## GPU Memory Management

**IMPORTANT**: Before running any GPU-intensive task (training, popV annotation, etc.), always check GPU memory:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv
```

Plan a buffer of at least 2-4GB free memory. If memory is tight, reduce batch size or wait for other GPU tasks to complete.

## DataFrame Library Standard

**PRIMARY**: Use `polars` for all new code and internal data processing. Ruff enforces this via TID251 (banned `pandas` import).

**BOUNDARY**: At external library interfaces (scanpy, anndata, celltypist, R), convert at the boundary:
- `df.to_pandas()` when passing to external libs
- `pl.from_pandas(pd_df)` when receiving from external libs

Files with `TID251` exemptions in `pyproject.toml` are the pandas-boundary files (annotation, cross_validation, ensemble_annotation, popv_annotation, ground_truth, azimuth).

## Project Overview

DAPIDL predicts cell types from DAPI nuclear staining using spatial transcriptomics (Xenium, MERSCOPE) as automatic training data. The core idea: use CellTypist/popV/SingleR ensemble to annotate cells from gene expression, then train a CNN to predict those labels from DAPI patches alone. This enables cell type prediction on any DAPI image without needing gene expression.

## Commands

```bash
# Install/sync dependencies
uv sync

# Lint and type check
uv run ruff check src/
uv run ruff format src/
uv run mypy src/

# Run tests
uv run pytest tests/
uv run pytest tests/test_specific.py::test_function

# List available models and backbones
uv run dapidl list-models
uv run dapidl list-backbones

# ClearML Pipeline (primary workflow — supports 1-N datasets via -t flags)
# Format: -t <tissue> <clearml_id_or_path> <platform> <confidence_tier>
uv run dapidl clearml-pipeline run -t lung bf8f913f xenium 2 --local --epochs 10
uv run dapidl clearml-pipeline run \
    -t lung bf8f913f xenium 2 \
    -t heart 482be038 xenium 2 \
    --epochs 50 --sampling sqrt

# SOTA pipeline (declarative annotation methods)
uv run dapidl clearml-pipeline sota --dataset-id <id> --tissue breast

# Run individual pipeline steps standalone
uv run dapidl step data-loader --local-path /path/to/data
uv run dapidl step annotate --annotator celltypist --strategy consensus
uv run dapidl step lmdb --patch-size 128
uv run dapidl step train --backbone efficientnetv2_rw_s --epochs 50

# Legacy local pipeline
uv run dapidl pipeline -x /path/to/xenium -o ./experiment --epochs 50
uv run dapidl prepare -x /path/to/xenium -o ./dataset
uv run dapidl train -d ./dataset --epochs 50 --batch-size 64

# HEIST graph neural network
uv run dapidl heist-prepare -x /path/to/data -o ./heist_data
uv run dapidl heist-train -d ./heist_data --epochs 50

# PopV annotation
uv run dapidl popv annotate -i /path/to/data -o ./output
uv run dapidl popv list-references

# LMDB export and cleaning
uv run dapidl export-lmdb -d /path/to/zarr -o ./lmdb_output
uv run dapidl clean-dataset -d /path/to/lmdb
```

## Architecture

### High-Level Data Flow

```
Raw Spatial Data (Xenium/MERSCOPE/STHELAR)
    → Data Reader (xenium.py / merscope.py / sthelar_reader.py)
    → Ensemble Annotation (CellTypist + SingleR + popV consensus)
    → Confidence Filtering (marker enrichment + spatial coherence)
    → CL Ontology Standardization (harmonize label vocabularies)
    → Patch Extraction + LMDB Creation (nucleus-centered patches)
    → CNN Training (DALI data loading → backbone → classification head)
```

### ClearML Pipeline System

The primary workflow runs as a ClearML pipeline with individually-cached steps:

```
data_loader → ensemble_annotation → confidence_filtering → cl_standardization → lmdb_creation → training
```

**Controllers** (in `pipeline/`): Multiple controllers exist for different pipeline modes:
- `sota_controller.py` — Current default. Declarative annotation methods via `MethodSpec`.
- `unified_controller.py` — Consolidation target. Uses `unified_config.py` (Pydantic-based).
- `universal_controller.py` — Multi-tissue training across datasets.
- `controller.py`, `enhanced_controller.py` — Older, being replaced by unified.

**Steps** (in `pipeline/steps/`): Each step is a `PipelineStep` subclass from `base.py`. Steps communicate via `StepArtifacts` (serializable paths/references). Each step has a source-code hash for per-step cache invalidation.

**Registry** (`pipeline/registry.py`): Decorator-based registry for swappable segmenters (`@register_segmenter`) and annotators (`@register_annotator`).

### Annotation System

Located in `pipeline/components/annotators/`:

| Annotator | Type | Notes |
|-----------|------|-------|
| `celltypist.py` | ML-based | Multi-model consensus |
| `singler.py` | R-based (rpy2) | HPCA + Blueprint references |
| `popv_ensemble.py` | Ensemble | popV-style unweighted voting |
| `popv.py` | ML-based | Tabula Sapiens reference |
| `scanvi.py` | Deep learning | scVI/scANVI transfer |
| `scarches.py` | Deep learning | Surgery-based transfer |
| `scina.py` | Marker-based | Semi-supervised |
| `sctype.py` | Marker-based | Pure polars implementation |
| `azimuth.py` | R-based | Seurat reference mapping |
| `ground_truth.py` | Reference | Excel-based known labels |
| `consensus.py` | Meta | Cross-method voting |
| `universal_ensemble.py` | Meta | Tissue-agnostic ensemble |
| `auto_selector.py` | Meta | Automatic method selection |

### Model Architecture

**Standard classifier** (`models/classifier.py`): `SingleChannelAdapter` (1→3 channels) → timm backbone → Dropout → Linear head.

**Hierarchical classifier** (`models/hierarchical.py`): Shared backbone → 3 heads (coarse/medium/fine). Confidence-based inference fallback: fine → medium → coarse. Supports curriculum learning.

**HEIST** (`models/heist/`): Graph neural network that combines image patches with spatial neighborhood information. Uses `graph_builder.py` for KNN spatial graphs, `blending.py` for image-graph feature fusion.

**Backbones** (`models/backbone.py`): Configurable via `BACKBONE_PRESETS`:
- ImageNet pretrained: EfficientNetV2-S (default), ResNet-18/34, ConvNeXt-Tiny/Small, ViT-Base/Large
- Custom microscopy CNNs: `microscopy_cnn`, `microscopy_cnn_deep` (native single-channel)
- Pathology foundation models: Phikon-v2, UNI (ViT-L/16)
- LoRA fine-tuning support via PEFT

### Data Layer

**Readers**: `XeniumDataReader` (xenium.py), `MERSCOPEReader` (merscope.py), `STHELARReader` (sthelar_reader.py) — each reads platform-specific formats and extracts DAPI images + cell coordinates.

**Datasets**:
- `DAPIDLDataset` — Zarr-based, original format
- `HierarchicalDataset` — Multi-level labels (coarse/medium/fine)
- `MultiTissueDataset` — Cross-tissue training with configurable sampling
- `HEISTDataset` — Patches + spatial graphs for GNN

**Storage formats**: Zarr (legacy) → LMDB (current, for NVIDIA DALI fast loading). LMDB creation in `pipeline/steps/lmdb_creation.py`.

**DALI pipeline** (`data/dali_pipeline.py`, `data/dali_native.py`): GPU-accelerated data loading via NVIDIA DALI for LMDB datasets.

### Label Harmonization

**Ontology** (`ontology/`): Maps free-text cell type labels to Cell Ontology (CL) IDs. `cl_mapper.py` uses fuzzy matching + curated mappings. `cl_database.py` loads CL OBO file. `annotator_mappings.py` has per-annotator label → CL mappings.

**Harmonization** (`harmonization/`): `hierarchy.py` defines tissue-specific cell type hierarchies (e.g., `BREAST_HIERARCHY`). `mapper.py` maps labels from different annotation sources to common hierarchy levels.

### Validation & Confidence

**Cross-modal validation** (`validation/cross_modal.py`): Leiden clustering agreement, DAPI morphology agreement, multi-method consensus.

**Annotation confidence** (`validation/annotation_confidence.py`): GT-free confidence scoring using marker enrichment, spatial coherence (KNN), cross-method consensus, proportion plausibility.

**Marker database** (`validation/marker_database.py`): Unified lookup from Cell Marker Accordion + CellMarker 2.0 databases.

**Three-pass hierarchical confidence filtering** (`pipeline/steps/confidence_filtering.py`): Filters low-confidence annotations before training by relabeling to "Unknown".

### Supporting Modules

- **HPO** (`hpo/`): ClearML + Optuna hyperparameter optimization. `search_space.py` defines parameter ranges.
- **Evaluation** (`evaluation/metrics.py`): Accuracy, macro/weighted F1, MCC, per-class metrics, confusion matrices.
- **Tracking** (`tracking/`): Reproducibility capture (git state, CLI command, env, dataset fingerprint). Backend-agnostic protocol.
- **Domain Adaptation** (`models/domain_adaptation.py`): Cross-platform transfer learning.

### Configuration

Hydra-based config system in `configs/`:
- `config.yaml` — Main config (references data, model, training defaults)
- `data/default.yaml` — Patch size, pixel size, class hierarchy, splits
- `model/efficientnet.yaml` — Backbone, head config
- `training/default.yaml` — Epochs, optimizer, scheduler, early stopping, augmentation
- `recipes.yaml` — Custom pipeline step sequences

ClearML pipeline config via `pipeline/unified_config.py` (Pydantic models).

## Critical Implementation Details

**LMDB is the current training format**: DALI reads from LMDB for GPU-accelerated data loading. Zarr is legacy.

**Transform order**: Must convert uint16 to float BEFORE applying augmentations (GaussNoise fails on uint16).

**Class imbalance**: Uses weighted loss + WeightedRandomSampler. `max_weight_ratio=10.0` caps rare class weights to prevent mode collapse.

**Unweighted voting wins**: popV-style unweighted ensemble voting consistently beats confidence-weighted by 15-22%.

**SingleR is essential for stromal**: Blueprint reference jumps Stromal F1 from 0.35 → 0.76.

**MERSCOPE intensity differs ~16x from Xenium**: Adaptive percentile normalization is required for cross-platform compatibility.

**STHELAR data**: SpatialData zarr objects with co-registered DAPI + H&E for 31 Xenium sections across 16 tissues. Group labels are noisy (Leiden misassigns); prefer `TANGRAM_TO_COARSE` mapping.

## Dataset Locations

All datasets under `~/datasets/` (symlink to `/mnt/work/datasets/`). Set `DAPIDL_LOCAL_DATA` env var for auto-detection (colon-separated paths, defaults to `/mnt/work/datasets/raw/xenium:/mnt/work/datasets/raw/merscope`).

**S3**: Bucket `dapidl` in `eu-central-1`, AWS profile `dapidl`. Always use `--output-uri s3://dapidl/...` for ClearML artifacts.

- in your obsidian brain is a file Dear Claude where I can write something to you to ingest. each note is ended by Done checkbox. when you read this inform me and mark it
