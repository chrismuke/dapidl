# MERSCOPE Segmentation Grand Benchmark

**Date**: 2026-03-31
**Status**: Design
**Goal**: Find the best nucleus and cell segmentation method for MERSCOPE DAPI images by benchmarking 7 individual methods + 3 ensemble/consensus approaches, evaluated on biological, morphometric, and cross-method axes.

## Context

DAPIDL's downstream cell type prediction depends on segmentation quality. The current pipeline uses Cellpose (cyto3 model) but hasn't been systematically compared against newer methods. The MERSCOPE breast dataset (713K cells, 2507 FOVs, 20 GB DAPI mosaic) provides a rich testbed.

**Key constraint**: No manual ground truth annotations exist. Evaluation uses three complementary proxy metrics (biological, morphometric, cross-method agreement).

**Key constraint**: No `detected_transcripts.csv` with individual molecule coordinates. We have `cell_by_gene.csv` (cell-level expression, 594 genes × 713K cells) and `cell_metadata.csv` (centroids, bounding boxes in microns). Transcript-based evaluation is adapted to work with cell-level expression data.

## Data

### MERSCOPE Breast Dataset
- **DAPI image**: `images/mosaic_DAPI_z3.tif` — 94,805 × 110,485 px, uint16, ~20 GB
- **Cell metadata**: `cell_metadata.csv` — 713,121 cells, 2,507 FOVs
- **Expression**: `cell_by_gene.csv` — 544 genes + 50 blank controls
- **Transform**: `images/micron_to_mosaic_pixel_transform.csv` — micron-to-pixel affine (9.26 px/µm)
- **Pixel size**: 0.108 µm/pixel

### FOV Selection Strategy

Select **5 FOVs** representing diverse tissue contexts:

| FOV | Selection Criterion | Purpose |
|-----|---------------------|---------|
| FOV-dense | Highest cell density (cells/area) | Stress-test over-segmentation |
| FOV-sparse | Lowest cell density | Test sensitivity in sparse tissue |
| FOV-mixed | Moderate density, high cell-size variance | General-purpose benchmark |
| FOV-edge | Near mosaic boundary | Test boundary handling |
| FOV-immune | High immune cell fraction (small nuclei) | Test small-object detection |

**FOV selection algorithm**:
1. Load `cell_metadata.csv`, compute per-FOV: cell count, spatial extent (bounding box from min/max coordinates), mean cell volume, volume variance
2. Compute cell density = cell_count / spatial_extent_area
3. Select FOVs at density percentiles: p95 (dense), p5 (sparse), p50 (mixed)
4. Select FOV nearest mosaic edge (max pixel coordinate)
5. Select FOV with smallest mean cell volume (immune-rich proxy)
6. For each FOV, extract the bounding region from the DAPI mosaic using the affine transform + a 50-pixel padding

**Expected FOV tile size**: ~3000×3000 px per FOV (based on MERSCOPE FOV geometry at 200×200 µm ≈ 1852×1852 px, with padding).

## Methods

### Individual Segmenters (7)

| # | Method | Package | Model | Mode | Notes |
|---|--------|---------|-------|------|-------|
| 1 | **Cellpose-SAM** | `cellpose==4.0.8` | `nuclei` (SAM backbone) | Nucleus | SOTA transformer-based, already installed |
| 2 | **Cellpose cyto3** | `cellpose==4.0.8` | `cyto3` | Nucleus+cell | Current pipeline default |
| 3 | **Cellpose nuclei** | `cellpose==4.0.8` | `nuclei` (non-SAM) | Nucleus | Classic Cellpose nuclei model |
| 4 | **StarDist** | `stardist==0.9.2` | `2D_versatile_fluo` | Nucleus | Star-convex polygons, fast |
| 5 | **Mesmer/DeepCell** | `deepcell==0.12+` | `mesmer` | Nucleus+cell | Panoptic segmentation |
| 6 | **InstanSeg** | `instanseg==0.0.2` | Default | Nucleus+cell | Embedding-based, fast |
| 7 | **MERSCOPE native** | N/A | Platform output | Nucleus | Vizgen's built-in (from cell_metadata.csv centroids + volumes) |

**Note on Cellpose-SAM vs Cellpose nuclei**: Cellpose 4 includes the SAM backbone by default for the `nuclei` model. We test both the SAM-enabled path and explicitly disable SAM (`use_sam=False` or model `nuclei_cp3`) for comparison.

**Classical baseline**: We skip standalone watershed since Cellpose/StarDist already subsume it and the benchmark focuses on DL methods.

### Ensemble/Consensus Methods (3)

| # | Method | Strategy | Description |
|---|--------|----------|-------------|
| E1 | **Majority voting** | Pixel-level | For each pixel, assign the label that most methods agree on (requires label-to-instance mapping via IoU) |
| E2 | **IoU-weighted consensus** | Instance-level | Match instances across methods via IoU, weight each method by its morphometric quality score, merge via weighted centroid + boundary averaging |
| E3 | **Topological voting** | Local neighborhood | CellSampler-inspired: divide image into local patches, select the best-performing method per patch based on local quality metrics, stitch results |

## Architecture

### Module Structure

```
src/dapidl/benchmark/
├── __init__.py
├── runner.py              # BenchmarkRunner: orchestrates everything
├── fov_selector.py        # FOV selection from cell_metadata
├── segmenters/
│   ├── __init__.py
│   ├── base.py            # SegmenterAdapter ABC
│   ├── cellpose_adapter.py    # Cellpose-SAM, cyto3, nuclei
│   ├── stardist_adapter.py
│   ├── mesmer_adapter.py
│   ├── instanseg_adapter.py
│   └── native_adapter.py     # MERSCOPE platform segmentation
├── consensus/
│   ├── __init__.py
│   ├── majority_voting.py
│   ├── iou_weighted.py
│   └── topological_voting.py
├── evaluation/
│   ├── __init__.py
│   ├── biological.py      # Expression-based metrics
│   ├── morphometric.py    # Shape and size metrics
│   ├── cross_method.py    # Agreement between methods
│   └── practical.py       # Runtime, memory, detection count
└── reporting.py           # Generate comparison tables + plots
```

### Segmenter Adapter Interface

```python
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class SegmentationOutput:
    """Standardized output from any segmenter."""
    masks: np.ndarray           # (H, W) int32 label image, 0=background
    centroids: np.ndarray       # (N, 2) float64 [y, x] in pixels
    n_cells: int
    runtime_seconds: float
    peak_memory_mb: float
    method_name: str

class SegmenterAdapter(ABC):
    """Thin wrapper to standardize any segmentation method."""

    @abstractmethod
    def segment(self, image: np.ndarray, pixel_size_um: float = 0.108) -> SegmentationOutput:
        """Segment a single FOV image tile.

        Args:
            image: (H, W) uint16 DAPI image
            pixel_size_um: Physical pixel size in microns

        Returns:
            SegmentationOutput with masks, centroids, timing
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""
        ...

    @property
    def supports_cell_boundaries(self) -> bool:
        """Whether this method produces whole-cell (not just nucleus) segmentation."""
        return False
```

### Evaluation Metrics

#### 1. Biological Metrics (`evaluation/biological.py`)

Since we don't have individual transcript coordinates, we evaluate biological quality by matching new segmentation masks to MERSCOPE's native cell assignments:

- **Native cell recovery rate**: What fraction of MERSCOPE native cells have a matching segmented nucleus (IoU > 0.3 between native cell bounding box and segmented mask)?
- **Expression profile coherence**: For cells matched to native assignments, how consistent is the gene expression profile with expected cell-type markers? (Uses CellTypist annotations from prior pipeline runs.)
- **Multi-cell contamination**: Cases where one segmented cell overlaps multiple native cells (suggests under-segmentation).
- **Split-cell rate**: Cases where one native cell is split across multiple segmented cells (over-segmentation).

#### 2. Morphometric Metrics (`evaluation/morphometric.py`)

- **Nucleus area distribution**: Mean, median, std, and histogram. Compare to expected nuclear sizes for breast tissue (50-150 µm² for epithelial, 30-80 µm² for immune).
- **Eccentricity distribution**: Ratio of minor to major axis. Nuclei should be roughly round (eccentricity < 0.8).
- **Solidity**: Ratio of area to convex hull area. High solidity (>0.9) indicates clean boundaries without artifacts.
- **Size outlier rate**: Fraction of detected objects with area < 20 µm² (debris) or > 500 µm² (merged nuclei).
- **Detection density**: Cells detected per 1000 µm².

#### 3. Cross-Method Agreement (`evaluation/cross_method.py`)

- **Pairwise IoU matrix**: For each pair of methods, compute mean IoU of matched instances.
- **Consensus overlap score**: For each cell, count how many methods detected it (IoU > 0.5). Cells detected by 5+ methods are "high-confidence."
- **Method agreement heatmap**: Fraction of cells where method A and method B agree (within IoU > 0.5).
- **Unique detections**: Cells found by only one method (potential false positives or unique true positives).

#### 4. Practical Metrics (`evaluation/practical.py`)

- **Runtime**: Wall-clock seconds per FOV tile.
- **GPU memory peak**: Via `torch.cuda.max_memory_allocated()`.
- **Detection count**: Total cells detected per FOV.
- **Detection count vs native**: Ratio of detected cells to MERSCOPE native count.

### Consensus Engine

#### Majority Voting (`consensus/majority_voting.py`)

1. For each method's mask, extract instance labels
2. Build a pixel-level vote matrix: for each pixel, which instance from which method claims it
3. Use IoU-based instance matching across methods to establish correspondence
4. For each matched instance group, take the union/intersection of masks weighted by vote count
5. Unmatched instances (detected by <50% of methods) are discarded

#### IoU-Weighted Consensus (`consensus/iou_weighted.py`)

1. Match instances across all methods using Hungarian algorithm on IoU matrix
2. For each matched group, compute a weighted average boundary:
   - Weight = method's overall morphometric quality score (solidity × (1 - outlier_rate))
   - Merge via distance-transform averaging of weighted masks
3. Unmatched instances kept if from a high-scoring method

#### Topological Voting (`consensus/topological_voting.py`)

Inspired by CellSampler:
1. Divide the FOV into overlapping local patches (128×128 px, 50% overlap)
2. For each local patch, evaluate each method's local quality (morphometric scores)
3. Select the best method per patch
4. Stitch results using the overlap regions to resolve boundary conflicts
5. Re-label to ensure globally unique instance IDs

## Execution Flow

```
1. FOV Selection
   └── Load cell_metadata.csv → compute per-FOV stats → select 5 FOVs

2. FOV Extraction
   └── For each FOV: load DAPI region from mosaic, apply transform, save as tiles

3. Segmentation (parallelizable across methods)
   └── For each method × each FOV:
       ├── Run segmenter adapter
       ├── Record runtime + memory
       └── Save SegmentationOutput

4. Individual Evaluation
   └── For each method × each FOV:
       ├── Biological metrics (vs native cells)
       ├── Morphometric metrics
       └── Practical metrics

5. Cross-Method Evaluation
   └── For each FOV:
       ├── Pairwise IoU matrix
       ├── Consensus overlap scores
       └── Agreement heatmap

6. Consensus Segmentation
   └── For each FOV:
       ├── Run 3 consensus methods on all individual results
       └── Evaluate consensus results (same metrics as step 4)

7. Reporting
   └── Generate:
       ├── Summary table (method × metric)
       ├── Per-FOV breakdowns
       ├── Visualization: overlay masks on DAPI for each method
       └── Recommendation: ranked methods with rationale
```

## Package Dependencies

**Already installed:**
- `cellpose==4.0.8` (includes Cellpose-SAM)
- `segment-anything==1.0`

**To install:**
- `stardist>=0.9.2` — StarDist nucleus segmentation
- `deepcell>=0.12` — Mesmer/DeepCell (may need TensorFlow)
- `instanseg>=0.0.2` — InstanSeg embedding-based segmentation

**Potential issues:**
- **deepcell** requires TensorFlow, which may conflict with PyTorch. Mitigation: install in same env (TF and PyTorch coexist) or skip if dependency hell. Mesmer is lower priority than the other methods.
- **stardist** requires TensorFlow backend. Same consideration.
- **instanseg** is PyTorch-native, should install cleanly.

**Fallback**: If TensorFlow methods (Mesmer, StarDist) cause dependency conflicts, drop them from the benchmark. The PyTorch methods (Cellpose-SAM, cyto3, InstanSeg) plus consensus are the primary focus.

## Output Structure

```
pipeline_output/segmentation_benchmark/
├── config.json                          # Benchmark configuration
├── fovs/
│   ├── fov_dense.tif                    # Extracted FOV tiles
│   ├── fov_sparse.tif
│   ├── fov_mixed.tif
│   ├── fov_edge.tif
│   └── fov_immune.tif
├── results/
│   ├── cellpose_sam/
│   │   ├── fov_dense_masks.npy
│   │   ├── fov_dense_centroids.parquet
│   │   └── ...
│   ├── cellpose_cyto3/
│   ├── stardist/
│   ├── mesmer/
│   ├── instanseg/
│   ├── native/
│   ├── consensus_majority/
│   ├── consensus_iou_weighted/
│   └── consensus_topological/
├── evaluation/
│   ├── biological_metrics.json
│   ├── morphometric_metrics.json
│   ├── cross_method_metrics.json
│   ├── practical_metrics.json
│   └── summary.json
├── visualizations/
│   ├── overlay_comparison.png           # Side-by-side mask overlays
│   ├── metric_radar_chart.png           # Multi-axis comparison
│   ├── agreement_heatmap.png            # Method agreement matrix
│   └── per_fov_breakdown.png
└── report.md                            # Human-readable summary with recommendations
```

## Success Criteria

1. All 7 individual methods produce valid segmentation masks on at least 4/5 FOVs
2. At least one consensus method outperforms the best individual method on ≥2 evaluation axes
3. Clear ranking emerges with actionable recommendation for DAPIDL pipeline default
4. Runtime for full benchmark (5 FOVs × 10 methods × evaluation) completes in <2 hours on RTX 3090

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| TF/PyTorch dependency conflict | Try install; skip Mesmer/StarDist if broken |
| OOM on large FOV tiles | Tile within FOV if needed; FOVs are ~2000×2000 px, should fit |
| InstanSeg too new/buggy | Treat as optional; 6 methods still sufficient |
| No ground truth limits evaluation | Multi-axis evaluation compensates; biological metrics provide ground truth proxy |
| MERSCOPE native has no actual masks | Use bounding box from cell_metadata as proxy for native "segmentation" |
