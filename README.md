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
- **Efficient data pipeline**: Memory-efficient batch Zarr writing for 166K+ patches
- **EfficientNetV2-S backbone**: Pretrained ImageNet weights with single-channel adapter
- **Class imbalance handling**: Weighted loss and weighted random sampling
- **Augmentation**: Rotation, flips, elastic deformation, noise, blur
- **Logging**: Weights & Biases integration for experiment tracking

## Installation

```bash
git clone https://github.com/chrismuke/dapidl.git
cd dapidl
pip install -e .
```

## Usage

### 1. Prepare Dataset

Extract 128x128 DAPI patches centered on cell nuclei with CellTypist annotations:

```bash
dapidl prepare \
  --xenium-path /path/to/xenium_output \
  --output ./dataset \
  --confidence-threshold 0.0
```

This will:
- Load the DAPI image from `morphology_focus.ome.tif`
- Run CellTypist annotation using the `Cells_Adult_Breast.pkl` model
- Extract patches for all cells with valid centroids
- Save to Zarr format with labels and metadata

**Output structure:**
```
dataset/
├── patches.zarr/      # (N, 128, 128) uint16 patches
├── labels.npy         # (N,) int64 class labels
├── metadata.parquet   # Cell metadata (ID, predicted_type, confidence, etc.)
├── class_mapping.json # {"Epithelial": 0, "Immune": 1, "Stromal": 2}
└── dataset_info.json  # Statistics and configuration
```

### 2. Train Model

```bash
dapidl train \
  --data ./dataset \
  --epochs 50 \
  --batch-size 64 \
  --lr 0.0001
```

Training uses:
- 70/15/15 train/val/test split with stratification
- Weighted random sampling for balanced batches
- Early stopping on validation loss
- Checkpoints saved to `outputs/`

### 3. Evaluate

```bash
dapidl evaluate \
  --checkpoint outputs/best_model.pt \
  --data ./dataset
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

## Dataset Statistics (Xenium Breast Tumor)

| Class | Count | Percentage |
|-------|-------|------------|
| Epithelial | 145,278 | 87.2% |
| Immune | 20,980 | 12.6% |
| Stromal | 362 | 0.2% |

**Total**: 166,620 patches from 167,780 cells

## Configuration

See `configs/` for YAML configuration files:
- `config.yaml` - Main config with Hydra composition
- `data/default.yaml` - Data pipeline settings
- `model/efficientnet.yaml` - Model architecture
- `training/default.yaml` - Training hyperparameters

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
│   │   └── transforms.py   # Albumentations augmentation
│   ├── models/
│   │   ├── backbone.py     # timm backbone wrapper
│   │   └── classifier.py   # Full classifier model
│   ├── training/
│   │   ├── trainer.py      # Training loop
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
