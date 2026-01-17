# DAPIDL State-of-the-Art Pipeline

## Overview

The SOTA pipeline (`dapidl clearml-pipeline sota`) implements best practices discovered through comprehensive benchmarking (Jan 2025).

## Key Optimizations

### Annotation (F1=0.844)

| Setting | Value | Impact |
|---------|-------|--------|
| CellTypist Models | Cells_Adult_Breast, Immune_All_High, Immune_All_Low | Best for breast tissue |
| SingleR Reference | Blueprint | +117% Stromal F1 |
| Voting Strategy | **UNWEIGHTED** | Beats confidence-weighted by 15-22% |
| Min Agreement | 2 | Balance precision/recall |

### Training (F1=0.8481)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Backbone | EfficientNetV2-S | Best accuracy/speed from HPO |
| Patch Size | 256px | +20% F1 vs 32px |
| max_weight_ratio | **10.0** | CRITICAL: prevents mode collapse |
| Learning Rate | 5e-4 | HPO Trial 13 optimal |
| Batch Size | 64 | HPO optimal |
| Dropout | 0.2 | HPO optimal |

## Usage

### Quick Start

```bash
# Remote execution on ClearML agents
dapidl clearml-pipeline sota --dataset-id abc123

# Local execution with ClearML caching (RECOMMENDED for development)
dapidl clearml-pipeline sota --local-path /path/to/xenium --local-cached

# Local execution without caching (fastest, no ClearML dependency)
dapidl clearml-pipeline sota --local-path /path/to/xenium --local

# S3 data source
dapidl clearml-pipeline sota --s3-uri s3://bucket/path --local-cached

# Fine-grained classification
dapidl clearml-pipeline sota --dataset-id abc123 --fine-grained --epochs 150
```

### First Time Setup

```bash
# Create ClearML base tasks (one-time setup)
dapidl clearml-pipeline sota --dataset-id abc123 --create-tasks --local
```

### Python API

```python
from dapidl.pipeline import SOTAPipelineController, create_sota_config

# Create with SOTA defaults
controller = SOTAPipelineController()

# Or customize
config = create_sota_config(
    dataset_id="abc123",
    platform="xenium",
    fine_grained=False,
    epochs=100,
)
controller = SOTAPipelineController(config)

# Option 1: Run locally with ClearML caching (RECOMMENDED)
controller.create_pipeline()
pipeline_id = controller.run_locally_with_caching()

# Option 2: Run locally without ClearML (fastest, no tracking)
results = controller.run_locally()

# Option 3: Run on ClearML agents (production)
controller.create_pipeline()
pipeline_id = controller.run(wait=False)
```

## Execution Modes & Caching

### Execution Mode Comparison

| Mode | Command | ClearML Caching | Best For |
|------|---------|-----------------|----------|
| Remote | `sota --dataset-id X` | ✅ Full | Production, distributed |
| **Local + Cached** | `sota --local-path X --local-cached` | ✅ Full | Development (recommended) |
| Local Only | `sota --local-path X --local` | ⚠️ File-based | Quick tests, no ClearML |

### ClearML Caching Deep Dive

When `cache_executed_step=True` (default for data steps), ClearML creates a hash from:

```
Hash = MD5(
    script (repo + commit + entry point) +
    hyper_params (all parameters) +
    configs (configuration objects) +
    docker (setup script, args) +
    artifact_hashes (content hash, NOT task ID)
)
```

**Key insight**: Artifact inputs use **content hashes**, not task IDs. This means:
- Same data from different uploads → cache hit
- S3 data changes externally → cache miss (correct behavior)
- Parameter changes → cache miss

### Using `--local-cached` (Recommended for Development)

```bash
# First run: executes all steps, stores hashes in ClearML
dapidl clearml-pipeline sota --local-path /data/xenium --local-cached

# Second run: skips steps with matching hashes
dapidl clearml-pipeline sota --local-path /data/xenium --local-cached
# Output: "Skipping cached/executed step [data_loader]"
#         "Skipping cached/executed step [segmentation]"
#         etc.
```

This mode uses `start_locally(run_pipeline_steps_locally=True)`:
- Steps run as **local subprocesses** (no remote agents needed)
- Each step gets a ClearML task for **tracking and caching**
- Full `cache_executed_step` support with artifact content hashing

### Caching Configuration

```python
# In unified_config.py / SOTAConfig
execution = ExecutionConfig(
    cache_data_steps=True,   # Cache data/annotation/LMDB steps
    cache_training=False,    # Don't cache training (different seeds)
)
```

| Step | Default Caching | Rationale |
|------|-----------------|-----------|
| data_loader | ✅ Cached | Same data → same output |
| segmentation | ✅ Cached | Expensive GPU step |
| ensemble_annotation | ✅ Cached | ~10 min per run |
| lmdb_creation | ✅ Cached | Large I/O |
| training | ❌ Not cached | Different random seeds |
| cross_validation | ✅ Cached | Deterministic |

### Local-Only Skip Logic

When running with `--local` (no ClearML), file-based skip logic kicks in:

1. Check if output files exist (`centroids.parquet`, `annotations.parquet`, etc.)
2. Validate `config.json` matches current parameters
3. If both pass → skip step and reuse outputs

```python
# Saved config.json for validation
{
    "segmenter": "cellpose",
    "diameter": 40,
    "flow_threshold": 0.4,
    ...
}
```

## Web UI Configuration

All parameters are exposed to ClearML web UI with proper grouping:

- **Input Selection**: Dataset ID, S3 URI, platform
- **Annotation Configuration**: CellTypist models, SingleR, voting
- **LMDB/Dataset Configuration**: Patch sizes, normalization
- **Training Configuration**: Backbone, epochs, class weights
- **Output Configuration**: S3 upload, model registration
- **Validation**: Cross-modal validation settings

## Pipeline Steps

1. **data_loader**: Load raw Xenium/MERSCOPE data
2. **segmentation**: Nucleus detection (Cellpose)
3. **ensemble_annotation**: Multi-method cell type annotation
4. **lmdb_creation_p{size}**: Patch extraction (parallel per size)
5. **training**: Model training with SOTA settings
6. **cross_validation**: (optional) Quality assurance

## Benchmark Results

### Coarse Classification (3 classes)

| Approach | Test F1 | Stromal F1 | Notes |
|----------|---------|------------|-------|
| **SOTA Pipeline** | **0.848** | **0.813** | Best overall |
| HPO Trial 13 | 0.848 | - | Reference |
| Focal Loss Only | 0.519 | 0.067 | Mode collapse |
| Two-Stage | 0.611 | - | Binary only |

### Key Findings

1. **Unweighted voting > Confidence-weighted**: 15-22% improvement
2. **Blueprint reference essential**: +117% Stromal F1
3. **max_weight_ratio=10.0**: Prevents mode collapse
4. **256px patches > smaller**: +20% F1 vs 32px

## Files

- `src/dapidl/pipeline/sota_controller.py`: Pipeline controller
- `src/dapidl/pipeline/unified_config.py`: Configuration system
- `src/dapidl/cli.py`: CLI command (clearml-pipeline sota)
