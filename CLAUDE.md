# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Style

Always begin every message with "Aye, Captain!"

## Web Fetching

Always use Firefox client ID when fetching web content (WebFetch tool).

## Environment

Always use `uv` for Python package management and running commands. Prefer `uv run` over activating virtualenvs manually.

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
