# DAPIDL

**Deep learning system to predict cell types from DAPI nuclear staining using Xenium spatial transcriptomics as automatic training data.**

## Overview

DAPIDL uses Xenium spatial transcriptomics data to automatically generate training labels for a CNN classifier. The key insight: Xenium provides both DAPI images AND gene expression-based cell type labels. We use CellTypist to annotate cells from expression data, then train a deep learning model to predict those same cell types from DAPI patches alone.

```
Training:   Xenium (DAPI + RNA) → CellTypist → Cell Type Labels → Train CNN
Deployment: Any DAPI Image → Trained Model → Predicted Cell Types
```

## Features

- **Automated labeling**: CellTypist annotation from Xenium expression data
- **Multi-model support**: Use multiple CellTypist models for different tissue types
- **Efficient data pipeline**: Memory-efficient batch Zarr writing for 166K+ patches
- **EfficientNetV2-S backbone**: Pretrained ImageNet weights with single-channel adapter
- **Class imbalance handling**: Weighted loss and weighted random sampling
- **Augmentation**: Rotation, flips, elastic deformation, noise, blur
- **Logging**: Weights & Biases integration for experiment tracking with artifact versioning
- **Xenium Explorer export**: Create annotated datasets with space-efficient hardlinks

## Installation

```bash
git clone https://github.com/chrismuke/dapidl.git
cd dapidl
pip install -e .
```

## Quick Start: Complete Pipeline

Run the entire workflow with a single command:

```bash
dapidl pipeline \
  -x /path/to/xenium_output \
  -o ./experiment \
  --epochs 50
```

This will:
1. Extract DAPI patches from the Xenium dataset
2. Annotate cells using CellTypist
3. Train a CNN classifier
4. Save dataset and model to `./experiment/`

## Usage

### Commands Overview

| Command | Description |
|---------|-------------|
| `pipeline` | Run complete workflow (prepare + train) |
| `prepare` | Extract patches and generate training labels |
| `train` | Train the CNN classifier |
| `annotate` | Annotate Xenium dataset for Xenium Explorer |
| `create-dataset` | Create hardlinked dataset with custom CSV files |
| `list-models` | List available CellTypist models |
| `evaluate` | Evaluate trained model (coming soon) |
| `predict` | Run inference on new images (coming soon) |

### Complete Pipeline

```bash
# Run full pipeline with defaults (Cells_Adult_Breast.pkl model)
dapidl pipeline -x /path/to/xenium -o ./experiment

# Use different CellTypist model with more epochs
dapidl pipeline -x /path/to/xenium -o ./experiment \
    -m Immune_All_High.pkl --epochs 100 --wandb

# Only prepare dataset (skip training)
dapidl pipeline -x /path/to/xenium -o ./experiment --skip-train

# Only train (use existing dataset)
dapidl pipeline -x /path/to/xenium -o ./experiment --skip-prepare
```

**Output structure:**
```
experiment/
├── dataset/              # Prepared training data
│   ├── patches.zarr/     # (N, 128, 128) uint16 patches
│   ├── labels.npy        # (N,) int64 class labels
│   ├── metadata.parquet  # Cell metadata
│   └── class_mapping.json
└── training/             # Model outputs
    ├── checkpoints/      # Best and latest models
    └── training.log
```

### Step-by-Step Commands

#### 1. List Available CellTypist Models

```bash
# Show all available models
dapidl list-models

# Show only downloaded models
dapidl list-models --downloaded-only

# Update model list from server
dapidl list-models --update
```

#### 2. Prepare Dataset

Extract 128x128 DAPI patches centered on cell nuclei with CellTypist annotations:

```bash
dapidl prepare \
  -x /path/to/xenium_output \
  -o ./dataset \
  --confidence-threshold 0.5

# Use multiple models
dapidl prepare -x /path/to/xenium -o ./dataset \
    -m Cells_Adult_Breast.pkl \
    -m Immune_All_High.pkl
```

#### 3. Train Model

```bash
dapidl train \
  -d ./dataset \
  -o ./outputs \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.0001 \
  --wandb
```

Training uses:
- 70/15/15 train/val/test split with stratification
- Weighted random sampling for balanced batches
- Early stopping on validation loss
- W&B logging with dataset and model artifacts

### Xenium Explorer Integration

#### Annotate and Export Dataset

Create an annotated Xenium dataset for visualization in Xenium Explorer:

```bash
dapidl annotate \
  -x /path/to/xenium \
  -o /path/to/annotated \
  -m Cells_Adult_Breast.pkl

# Use multiple models
dapidl annotate -x /path/to/xenium -o /path/to/annotated \
    -m Cells_Adult_Breast.pkl \
    -m Immune_All_High.pkl \
    --output-format csv
```

This creates a **space-efficient copy** using hardlinks (saves ~10GB per dataset), adding only the annotation CSV files.

To view in Xenium Explorer:
1. Open the annotated dataset in Xenium Explorer
2. Go to: **Cells → Cell Groups → Upload**
3. Select the generated CSV file(s)

#### Create Dataset from Custom CSV

If you have pre-computed annotations in CSV format:

```bash
dapidl create-dataset \
  -x /path/to/xenium \
  -o /path/to/output \
  -c my_annotations.csv

# Multiple CSV files
dapidl create-dataset -x /path/to/xenium -o /output \
    -c celltypes.csv -c clusters.csv

# Replace a file in the dataset
dapidl create-dataset -x /path/to/xenium -o /output \
    -c annotations.csv \
    --replace cells.parquet /path/to/modified_cells.parquet
```

## Architecture

```
Input: 128x128 grayscale DAPI patch
    ↓
SingleChannelAdapter (replicate to 3 channels)
    ↓
EfficientNetV2-S (pretrained, 1792 features)
    ↓
Dropout (0.3)
    ↓
Linear → 3 classes (Epithelial, Immune, Stromal)
```

### Data Flow

```
Xenium Output → XeniumDataReader → CellTypeAnnotator → PatchExtractor → Zarr Dataset
                     ↓                    ↓                  ↓
              DAPI image (H,W)    CellTypist labels    128x128 patches
              cells.parquet       broad categories     labels.npy
              expression matrix                        metadata.parquet
```

## Dataset Statistics (Example: Xenium Breast Tumor)

| Class | Count | Percentage |
|-------|-------|------------|
| Epithelial | 145,278 | 87.2% |
| Immune | 20,980 | 12.6% |
| Stromal | 362 | 0.2% |

**Total**: 166,620 patches from 167,780 cells

## CellTypist Models

DAPIDL uses CellTypist for automatic cell type annotation. Available models include:

- `Cells_Adult_Breast.pkl` - Human adult breast tissue (default)
- `Immune_All_High.pkl` - Immune cell subtypes (high resolution)
- `Immune_All_Low.pkl` - Immune cell subtypes (low resolution)
- `Human_Lung_Atlas.pkl` - Human lung tissue
- And many more... (use `dapidl list-models` to see all)

The 58 fine-grained cell types are mapped to 3 broad categories:
- **Epithelial**: Luminal cells, basal cells, etc.
- **Immune**: T cells, B cells, macrophages, etc.
- **Stromal**: Fibroblasts, endothelial cells, etc.

## Project Structure

```
dapidl/
├── src/dapidl/
│   ├── cli.py              # Click CLI entry points
│   ├── data/
│   │   ├── xenium.py       # Xenium data reader
│   │   ├── annotation.py   # CellTypist integration
│   │   ├── patches.py      # Patch extraction
│   │   ├── dataset.py      # PyTorch Dataset
│   │   ├── transforms.py   # Albumentations augmentation
│   │   └── xenium_export.py # Xenium Explorer export
│   ├── models/
│   │   ├── backbone.py     # timm backbone wrapper
│   │   └── classifier.py   # Full classifier model
│   ├── training/
│   │   ├── trainer.py      # Training loop + W&B artifacts
│   │   └── losses.py       # Loss functions
│   └── evaluation/
│       └── metrics.py      # Evaluation metrics
├── configs/                # Hydra configuration
└── pyproject.toml          # Package definition
```

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- timm (EfficientNetV2)
- albumentations
- zarr
- polars
- celltypist
- wandb (optional)

## License

MIT

## Citation

If you use DAPIDL in your research, please cite:

```bibtex
@software{dapidl2024,
  title = {DAPIDL: Deep Learning Cell Type Classification from DAPI Nuclear Staining},
  author = {Mayer, Chris},
  year = {2024},
  url = {https://github.com/chrismuke/dapidl}
}
```
