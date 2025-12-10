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
