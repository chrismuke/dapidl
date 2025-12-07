# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Style

Always begin every message with "Aye, Captain!"

## Environment

Always use `uv` for Python package management and running commands. Prefer `uv run` over activating virtualenvs manually.

## Project Overview

DAPIDL predicts cell types from DAPI nuclear staining using Xenium spatial transcriptomics as automatic training data. The system uses CellTypist to annotate cells from gene expression, then trains a CNN to predict those labels from DAPI patches alone.

## Commands

```bash
# Install/sync dependencies
uv sync

# Prepare dataset from Xenium output
uv run dapidl prepare --xenium-path /path/to/xenium --output ./dataset

# Train model
uv run dapidl train --data ./dataset --epochs 50 --batch-size 64

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
- `data/xenium.py`: XeniumDataReader loads DAPI images and cell data from Xenium output
- `data/annotation.py`: CellTypeAnnotator wraps CellTypist, maps 58 cell types → 3 broad categories (Epithelial, Immune, Stromal)
- `data/patches.py`: PatchExtractor uses two-pass approach for memory-efficient Zarr writing
- `data/dataset.py`: PyTorch Dataset with stratified splits and weighted sampling for class imbalance
- `models/classifier.py`: CellTypeClassifier combines SingleChannelAdapter + timm backbone + classification head
- `training/trainer.py`: Training loop with W&B logging, early stopping, checkpointing

### Critical Implementation Details

**Memory-efficient patch saving**: Patches are written in batches of 1000 to match Zarr chunk size. Do not accumulate all patches in memory.

**Transform order**: Must convert uint16 to float BEFORE applying augmentations (GaussNoise fails on uint16).

**Class imbalance**: Dataset is ~87% Epithelial, 13% Immune, 0.2% Stromal. Uses weighted loss + WeightedRandomSampler.
