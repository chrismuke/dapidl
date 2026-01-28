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

Plan a buffer of at least 2-4GB free memory. If memory is tight:
- Reduce batch size
- Wait for other GPU tasks to complete
- Kill unnecessary background processes

## DataFrame Library Standard

**PRIMARY**: Use `polars` for all new code and internal data processing.

**BOUNDARY**: At external library interfaces (scanpy, anndata, celltypist, R), convert at the boundary:
- `df.to_pandas()` when passing to external libs
- `pl.from_pandas(pd_df)` when receiving from external libs

**LEGACY**: Some files still use pandas unnecessarily. When modifying these files, migrate to polars.

### Common Polars Patterns

```python
# Instead of: df.groupby('col').count().to_pandas().set_index('col')['count'].to_dict()
# Use:
dict(df.group_by("col").agg(pl.len().alias("count")).iter_rows())

# Instead of: pd.read_csv(path)
# Use:
pl.read_csv(path)

# Instead of: df.to_csv(path)
# Use:
df.write_csv(path)

# Instead of: pd.read_parquet(path)
# Use:
pl.read_parquet(path)
```

### Files Requiring Pandas (External Library Boundaries)

These files MUST use pandas at the boundary because external libraries require it:
- `data/annotation.py` - celltypist, anndata (AnnData.obs is pandas DataFrame)
- `pipeline/steps/annotation.py` - anndata, scanpy
- `pipeline/steps/cross_validation.py` - scanpy, anndata
- `pipeline/steps/ensemble_annotation.py` - anndata, scanpy
- `pipeline/steps/popv_annotation.py` - anndata, scanpy
- `pipeline/components/annotators/ground_truth.py` - pd.read_excel (polars Excel support is limited)
- `pipeline/components/annotators/azimuth.py` - R interface (CSV exchange)

### Files Successfully Migrated to Polars-Only

- `pipeline/components/annotators/sctype.py` - Pure polars (no pandas import)

## Project Overview

DAPIDL predicts cell types from DAPI nuclear staining using Xenium spatial transcriptomics as automatic training data. The system uses CellTypist to annotate cells from gene expression, then trains a CNN to predict those labels from DAPI patches alone.

## Commands

```bash
# Install/sync dependencies
uv sync

# Run complete pipeline (prepare + train)
uv run dapidl pipeline -x /path/to/xenium -o ./experiment --epochs 50

# Or run steps separately:

# Prepare dataset from Xenium output
uv run dapidl prepare -x /path/to/xenium -o ./dataset

# Train model
uv run dapidl train -d ./dataset --epochs 50 --batch-size 64

# List available CellTypist models
uv run dapidl list-models

# Annotate Xenium dataset for Xenium Explorer
uv run dapidl annotate -x /path/to/xenium -o ./annotated -m Cells_Adult_Breast.pkl

# Create hardlinked dataset with custom CSV
uv run dapidl create-dataset -x /path/to/xenium -o ./output -c annotations.csv

# ClearML Pipeline (unified, supports 1-N datasets via -t flags)
# Single dataset
uv run dapidl clearml-pipeline run -t lung bf8f913f xenium 2 --local --epochs 10
# Multiple datasets
uv run dapidl clearml-pipeline run \
    -t lung bf8f913f xenium 2 \
    -t heart 482be038 xenium 2 \
    --epochs 50 --sampling sqrt
# Prepare-only (no training, creates LMDB for later use)
uv run dapidl clearml-pipeline run -t lung bf8f913f xenium 2 --skip-training --local
# Remote execution (ClearML agents)
uv run dapidl clearml-pipeline run -t breast abc123 xenium 1

# Lint
uv run ruff check src/
uv run ruff format src/

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/
uv run pytest tests/test_specific.py::test_function  # single test
```

## Architecture

### Data Flow
```
Xenium Output → XeniumDataReader → CellTypeAnnotator → PatchExtractor → Zarr Dataset
                     ↓                    ↓                  ↓
              DAPI image (H,W)    CellTypist labels    128x128 patches
              cells.parquet       broad categories     labels.npy
              expression matrix                        metadata.parquet
```

### Model Pipeline
```
DAPIDLDataset → transforms (ToFloat → augment → Normalize → ToTensor)
                                        ↓
                          SingleChannelAdapter (1→3 channels)
                                        ↓
                          EfficientNetV2-S (pretrained, 1792 features)
                                        ↓
                          Dropout(0.3) → Linear(3 classes)
```

### Key Modules
- `cli.py`: Click CLI with `pipeline`, `prepare`, `train`, `annotate`, `create-dataset`, `list-models` commands
- `data/xenium.py`: XeniumDataReader loads DAPI images and cell data from Xenium output
- `data/annotation.py`: CellTypeAnnotator wraps CellTypist with multi-model support, maps 58 cell types → 3 broad categories (Epithelial, Immune, Stromal)
- `data/patches.py`: PatchExtractor uses two-pass approach for memory-efficient Zarr writing
- `data/dataset.py`: PyTorch Dataset with stratified splits and weighted sampling for class imbalance
- `data/xenium_export.py`: Create hardlinked Xenium datasets for Xenium Explorer, export cell groups CSV
- `models/classifier.py`: CellTypeClassifier combines SingleChannelAdapter + timm backbone + classification head
- `training/trainer.py`: Training loop with W&B logging (including artifacts), early stopping, checkpointing

### Critical Implementation Details

**Memory-efficient patch saving**: Patches are written in batches of 1000 to match Zarr chunk size. Do not accumulate all patches in memory.

**Transform order**: Must convert uint16 to float BEFORE applying augmentations (GaussNoise fails on uint16).

**Class imbalance**: Dataset is ~87% Epithelial, 13% Immune, 0.2% Stromal. Uses weighted loss + WeightedRandomSampler.

**Hardlink exports**: Use `create_hardlink_dataset()` for space-efficient Xenium dataset copies. Hardlinks share inodes, saving ~10GB per dataset.

**Multi-model support**: CellTypeAnnotator accepts list of model names. When using multiple models, columns are suffixed (_1, _2, etc.).

## Cross-Modal Validation (No Ground Truth Required)

DAPIDL includes a validation framework to verify CellTypist predictions using orthogonal approaches:

### Validation Methods:
1. **Leiden Clustering** - Compare unsupervised clusters with supervised labels (ARI, NMI)
2. **DAPI Morphology** - Use trained DAPI model as independent validation (agreement rate)
3. **Multi-Method Consensus** - Check agreement between multiple annotation methods

### Usage:
```bash
# Run pipeline with validation step
uv run dapidl clearml-pipeline run -t breast abc123 xenium 2 --validate

# Or run locally
uv run dapidl clearml-pipeline run -t breast /path/to/data xenium 1 --local --validate
```

### Interpretation:
- **Leiden ARI > 0.8**: Excellent agreement with transcriptomic structure
- **DAPI Agreement > 70%**: Strong cross-modal validation
- **Consensus Score >= 0.5**: Majority of methods agree

### Key Files:
- `validation/cross_modal.py`: Core validation functions
- `pipeline/steps/cross_validation.py`: ClearML pipeline step
- `docs/CROSS_MODAL_VALIDATION.md`: Full research documentation

## Cross-Platform Compatibility (Xenium ↔ MERSCOPE)

### Gene Panel Comparison (Dec 2024)

| Platform | Genes | Shared |
|----------|-------|--------|
| Xenium (breast) | 341 | 94 (27.6%) |
| MERSCOPE (breast) | 500 | 94 (18.8%) |

**Key insight**: Only ~28% gene overlap, BUT this doesn't matter for DAPIDL because:

1. **DAPIDL uses DAPI images only** - The model learns morphological features from nuclear staining, not gene expression
2. **Transcripts only used for training labels** - CellTypist generates annotations independently on each platform
3. **DAPI is platform-agnostic** - Same staining chemistry on both platforms

### Marker Gene Coverage (for CellTypist annotation quality)

| Cell Type | In Both Panels | Notes |
|-----------|----------------|-------|
| Immune | 14/15 (93%) | Excellent - CD3D/E, CD4, CD8A/B, CD14, CD68, etc. |
| Endothelial | 4/6 (67%) | Good - PECAM1, VWF, CLDN5, KDR |
| Stromal | 3/10 (30%) | Moderate - ACTA2, PDGFRA, PDGFRB |
| Epithelial | 2/9 (22%) | Limited but key markers: EPCAM, CDH1 |

### Cross-Platform Model Transfer

A model trained on MERSCOPE **should work on Xenium** (and vice versa) because:
- Same DAPI staining chemistry
- Same 128x128 patch extraction around nuclei
- Adaptive normalization handles intensity differences (~16x higher on MERSCOPE)
- CellTypist uses similar markers for both platforms

**Main considerations for transfer**:
1. Normalize images appropriately (adaptive percentile normalization)
2. Cell type distributions may differ between tissues/samples
3. Image quality/resolution differences may affect fine-grained classification

## Training Run Status (Dec 9, 2024)

### Currently Running Experiments

| Experiment | Platform | Classes | Epoch | Best F1 | Status |
|------------|----------|---------|-------|---------|--------|
| merscope_finegrained_v3 | MERSCOPE | 17 | 6/50 | 0.2019 | Training (with balanced weights) |
| vizgen_adaptive_v2 | MERSCOPE | 4 | 32/50 | 0.0468 | Training (coarse) |
| merscope_finegrained_v2 | MERSCOPE | 17 | 11/50 | 0.0011 | Training (baseline) |

### Key Findings

**Class Weight Capping (max_weight_ratio=10.0)**: Massive improvement!
- MERSCOPE v2 (uncapped): F1 = 0.0011 after 10 epochs (mode collapse to majority class)
- MERSCOPE v3 (capped): F1 = 0.2019 after 5 epochs (180x improvement!)

**Adaptive Normalization**: Essential for MERSCOPE data
- MERSCOPE intensity ~16x higher than Xenium (median ~13000 vs ~800)
- Stats: p_low=2188, p_high=24577, mean=0.298, std=0.203

### Completed Experiments (Xenium)

| Experiment | Classes | Best Val F1 | Backbone |
|------------|---------|-------------|----------|
| groundtruth | 3 | ~0.68 | EfficientNetV2-S |
| groundtruth_finegrained | ~20 | ~0.45 | EfficientNetV2-S |
| groundtruth_convnext | 3 | ~0.65 | ConvNeXt-Tiny |
| groundtruth_resnet50 | 3 | ~0.62 | ResNet50 |
| consensus_v2 | 3 | ~0.70 | EfficientNetV2-S |
| finegrained_immune | ~15 | ~0.55 | EfficientNetV2-S |

### Problem: Mode Collapse in Fine-Grained Training

Without class weight capping, rare classes (< 1% of samples) get extreme weights that destabilize training. Solution:
```python
# In losses.py and dataset.py
max_weight_ratio=10.0  # Rare classes get at most 10x the weight of common classes
```

### MERSCOPE Dataset Characteristics

- Total cells: 713,121 (all annotations)
- Training samples: 151,479 (high-confidence consensus)
- Classes after filtering (min 20 samples): 17 fine-grained types
- Filtered classes: DC1, DC3, Fibro-myofibro, pDC, NK cells (< 20 samples each)

## Available Datasets (Dec 2024)

### Dataset Organization

All datasets are organized under `~/datasets/` (symlink to `/mnt/work/datasets/`):

```
~/datasets/
├── raw/
│   ├── xenium/           # Raw Xenium spatial transcriptomics data
│   │   ├── breast_tumor_rep1/   # Main dataset with ground truth
│   │   ├── breast_tumor_rep2/   # Second replicate
│   │   ├── lung_2fov/           # Small test dataset
│   │   └── ovarian_cancer/      # Large Xenium Prime dataset
│   └── merscope/         # Raw MERSCOPE data
│       └── breast/              # MERSCOPE breast cancer
├── processed/            # Intermediate processed outputs
│   └── xenium_breast_cellpose/  # Cellpose nucleus segmentation
└── derived/              # Training-ready LMDB datasets (18 datasets, 66 GB)
    ├── xenium-breast-cellpose-{coarse,finegrained}-p{32,64,128,256}/
    └── xenium-breast-xenium-{coarse,finegrained}-p{32,64,128,256}/
```

### Local Dataset Auto-Detection (`DAPIDL_LOCAL_DATA`)

The pipeline's `data_loader` step automatically detects locally available datasets to avoid redundant S3/ClearML downloads. It scans configured directories and matches ClearML dataset names to local subdirectories.

**Environment variable:**
```bash
export DAPIDL_LOCAL_DATA="/mnt/work/datasets/raw/xenium:/mnt/work/datasets/raw/merscope"
```

- Colon-separated list of directories containing raw spatial datasets
- Each directory should contain subdirectories named like the ClearML datasets
- Automatically handles `outs/` subdirectories (Xenium convention)
- If not set, defaults to `/mnt/work/datasets/raw/xenium` and `/mnt/work/datasets/raw/merscope`

**Fallback order:**
1. Explicit `local_path` in config (highest priority)
2. Local registry lookup by S3 URI or dataset name
3. Local lookup by ClearML dataset ID (queries ClearML for name, then matches locally)
4. S3 cache directory (`~/.cache/dapidl/s3_downloads/`)
5. Download from S3 or ClearML (last resort)

**Verification:** When resolving by ClearML dataset ID, file sizes are compared against ClearML metadata. If fewer than 90% of files match, the local copy is considered stale and falls back to downloading.

### Local Xenium Datasets

| Dataset | Location | Size | Cells | Notes |
|---------|----------|------|-------|-------|
| **Breast Tumor Rep1** | `~/datasets/raw/xenium/breast_tumor_rep1/` | 19 GB | 167,780 | Ground truth (Janesick + supervised) |
| **Breast Tumor Rep2** | `~/datasets/raw/xenium/breast_tumor_rep2/` | 18 GB | 118,752 | Second replicate |
| **Lung 2fov** | `~/datasets/raw/xenium/lung_2fov/` | 537 MB | 11,898 | Small test dataset |
| **Ovarian Cancer** | `~/datasets/raw/xenium/ovarian_cancer/` | 50 GB | 407,124 | Xenium Prime, 5K panel |

### Local MERSCOPE Datasets

| Dataset | Location | Size | Cells | Notes |
|---------|----------|------|-------|-------|
| **Breast** | `~/datasets/raw/merscope/breast/` | 22 GB | 713,121 | MERSCOPE breast cancer |

### Xenium Download URLs (for additional datasets)

```bash
# Small test datasets (2 FOV)
curl -O https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip
curl -O https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip

# Full datasets
curl -O https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
wget https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip
```

### MERSCOPE Data Access

MERSCOPE/Vizgen data with DAPI images requires registration at:
- https://info.vizgen.com/merscope-ffpe-solution
- Available: 16 FFPE datasets, 8 tumor types, ~9M cells total

### S3/ClearML Storage

**iDrive e2 S3 Configuration**:
- Endpoint: `https://s3.eu-central-2.idrivee2.com`
- Region: `eu-central-2`
- Bucket: `dapidl`
- Access Key: `evkizOGyflbhx5uSi4oV`
- Secret Key: `zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9`

**S3 Dataset Paths** (28.7 GB total, uploaded Dec 2024):

| Dataset | S3 Path | Size | Cells |
|---------|---------|------|-------|
| Lung 2fov | `s3://dapidl/raw-data/xenium-lung-2fov/` | 247 MB | 11,898 |
| Breast Rep1 | `s3://dapidl/raw-data/xenium-breast-cancer-rep1/` | 8.2 GB | 167,780 |
| Ovarian Cancer | `s3://dapidl/raw-data/xenium-ovarian-cancer/` | 20.3 GB | 407,124 |

**AWS CLI Usage**:
```bash
export AWS_ACCESS_KEY_ID=evkizOGyflbhx5uSi4oV
export AWS_SECRET_ACCESS_KEY=zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9

# List bucket contents
aws s3 ls s3://dapidl/ --endpoint-url https://s3.eu-central-2.idrivee2.com --region eu-central-2

# Download a dataset
aws s3 sync s3://dapidl/raw-data/xenium-lung-2fov/ ./xenium-lung-2fov/ \
    --endpoint-url https://s3.eu-central-2.idrivee2.com --region eu-central-2
```

**Storage Strategy**:
- **Large files (datasets, models)**: Store on S3, register with ClearML using S3 URIs
- **Experiment tracking**: Use ClearML for all pipeline runs
- **Free tier limits**: ~100GB storage, 1M API calls/month
- Always use `--output-uri s3://dapidl/...` for ClearML tasks to store artifacts on S3

## Annotation Benchmark Results (Jan 2025)

### PopV-Style Ensemble - Coarse Classification (3 classes)

Comprehensive evaluation of popV-inspired ensemble voting on Xenium breast cancer (rep1 + rep2):

**Rep1 (167,780 cells) - with HPCA + Blueprint SingleR:**

| Method | Voting | Accuracy | Macro F1 | Epi | Imm | Str |
|--------|--------|----------|----------|-----|-----|-----|
| **5 CellTypist + SingleR** | unweighted | **88.9%** | **0.844** | 0.98 | 0.81 | 0.75 |
| 3 CellTypist + SingleR | unweighted | 88.2% | 0.840 | 0.97 | 0.81 | 0.74 |
| 2 CellTypist + SingleR | unweighted | 87.0% | 0.839 | 0.95 | 0.80 | 0.76 |
| 5 CellTypist only | unweighted | 84.3% | 0.737 | 0.95 | 0.75 | 0.52 |

**Rep2 (118,752 cells):**

| Method | Voting | Accuracy | Macro F1 | Epi | Imm | Str |
|--------|--------|----------|----------|-----|-----|-----|
| **5 CellTypist + SingleR** | unweighted | **86.8%** | **0.843** | 0.97 | 0.78 | 0.77 |
| 3 CellTypist + SingleR | unweighted | 86.0% | 0.837 | 0.96 | 0.78 | 0.77 |
| 2 CellTypist + SingleR | unweighted | 84.8% | 0.832 | 0.93 | 0.78 | 0.78 |

**Key Insights:**
- **Unweighted voting always wins** (popV style) - beats confidence-weighted by 15-22%
- **SingleR (HPCA + Blueprint) is essential** - adds +0.10 to +0.27 F1
- **Both replicates show consistent ~0.84 F1** - when given equal SingleR coverage
- **Stromal F1 requires Blueprint reference** - jumps from 0.35 → 0.76 (+117%)

### Cell Ontology-Normalized Evaluation (Jan 2025)

Using CL mapping to standardize vocabulary between predictions and ground truth:

| Level | Classes (T/P) | Accuracy | F1 Macro | Notes |
|-------|---------------|----------|----------|-------|
| **BROAD** | 4/4 | **85.2%** | **0.622** | 3 categories + Unknown |
| COARSE | 10/15 | 70.4% | 0.306 | ~10 cell types |
| CL_NAME | 12/25 | 67.8% | 0.192 | Semantic match |
| CL_ID | 12/31 | 67.8% | 0.158 | Exact CL match |

**Per-Class Performance (COARSE level):**
| Cell Type | F1 | Notes |
|-----------|-----|-------|
| Epithelial_Luminal | 0.922 | Excellent |
| T_Cell | 0.804 | Great |
| B_Cell | 0.769 | Good |
| Vascular_Endothelial | 0.756 | Good |
| Macrophage | 0.580 | Moderate |
| Fibroblast | 0.356 | Poor - often Unknown/Adipocyte |
| Mast_Cell | 0.353 | Moderate |
| Dendritic_Cell | 0.348 | Confused with NK |
| Epithelial_Basal | 0.004 | 73% → Luminal (myoepithelial confusion) |

**Key Confusion Patterns:**
- Fibroblast: Only 14% correct (37% Unknown, 16% Adipocyte)
- Epithelial_Basal: 73% misclassified as Luminal (myoepithelial vs luminal)
- Pericyte: 58% Unknown (stromal challenge)

**Insight**: Fine-grained stromal classification is the main challenge. Immune cells (T, B, Macrophage) perform well.

### Legacy Benchmark Results (Dec 2024)

| Method | Accuracy | Macro F1 | Notes |
|--------|----------|----------|-------|
| SingleR-Blueprint | 92.0% | 0.907 | Standalone R-based |
| scType-Markers | 87.6% | 0.854 | Marker-based |
| CellTypist-Breast | 90.7% | 0.737 | Single model |

### PopV Ensemble Annotator Usage

```python
from dapidl.pipeline.components.annotators.popv_ensemble import (
    PopVStyleEnsembleAnnotator,
    PopVEnsembleConfig,
    VotingStrategy,
)

config = PopVEnsembleConfig(
    celltypist_models=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl", "Immune_All_Low.pkl"],
    include_singler_hpca=True,
    include_singler_blueprint=True,
    voting_strategy=VotingStrategy.UNWEIGHTED,
)
annotator = PopVStyleEnsembleAnnotator(config)
result = annotator.annotate(adata)
```

- in your obsidian brain is a file Dear Claude where I can write something to you to ingest. each note is ended by Done checkbox. when you read this inform me and mark it