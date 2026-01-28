# SingleR Algorithm Documentation

## Overview

SingleR (Single-cell Recognition) is a reference-based cell type annotation method developed by Aran et al. (2019). It assigns cell type labels by comparing single-cell expression profiles to reference datasets of known cell types.

## How SingleR Works

### Step 1: Reference Database Selection

SingleR uses curated reference databases with known cell types:

| Database | Full Name | Cell Types | Best For |
|----------|-----------|------------|----------|
| **HPCA** | Human Primary Cell Atlas | 36 main types | Broad tissue coverage, epithelial/stromal |
| **Blueprint** | Blueprint Epigenome | 25 main types | Immune cell subtypes (T cells, B cells, etc.) |

### Step 2: Gene Matching

```
Query cells (your Xenium data)      Reference database
     ↓                                    ↓
  541 genes                          ~15,000 genes
     ↓                                    ↓
          → Find overlapping genes ←
                    ↓
            Use ~200-500 shared genes
```

### Step 3: Spearman Correlation

For each query cell, SingleR:

1. **Computes correlation** between the cell's expression profile and each reference cell type
2. **Uses Spearman correlation** (rank-based, robust to outliers)

```
Cell_i expression: [EPCAM=100, CD3D=5, ACTA2=20, ...]
                            ↓
Compare to Reference:
  - Epithelial_cells:  r = 0.85  ← Highest
  - T_cells:           r = 0.42
  - Fibroblasts:       r = 0.38
  ...
```

### Step 4: Fine-Tuning (Iterative Refinement)

SingleR iteratively narrows down the label:

```
Round 1: All 36 cell types → Top 10 candidates (r > threshold)
Round 2: Recalculate with marker genes → Top 5 candidates
Round 3: Final selection → Best match
```

### Step 5: Delta Score (Confidence)

The **delta score** measures confidence:

```
delta = correlation(best match) - correlation(second best)
```

- **High delta (>0.1)**: Confident assignment
- **Low delta (<0.05)**: Ambiguous, may be pruned

### Step 6: Pruning (Quality Control)

Cells with low confidence can be "pruned" (marked as NA):
- Low delta score
- Poor correlation with all references
- Indicates unusual or low-quality cells

## Example Output

```csv
cell_id,singler_hpca_label,singler_hpca_score,singler_bp_label,singler_bp_score
1,Epithelial_cells,0.051,Epithelial cells,0.042
2,T_cells,0.035,CD4+ T-cells,0.068
3,Fibroblasts,0.089,Smooth muscle,0.045
```

## Why Use Multiple References?

| Single Reference | Multiple References |
|-----------------|---------------------|
| May miss cell types not in reference | Complementary coverage |
| Limited to one annotation scheme | Multiple perspectives |
| Faster | More accurate for diverse tissues |

### HPCA vs Blueprint Strengths

```
HPCA:
  ✓ Epithelial cells (50,456 in rep1)
  ✓ Fibroblasts/Stromal
  ✓ Endothelial cells

Blueprint:
  ✓ CD4+ T-cells (13,786 in rep1)
  ✓ CD8+ T-cells
  ✓ Macrophages (11,568 in rep1)
  ✓ B cells, NK cells
```

## References

- Aran, D. et al. (2019). "Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage." Nature Immunology, 20(2), 163-172.
- celldex R package: https://bioconductor.org/packages/celldex/
- SingleR R package: https://bioconductor.org/packages/SingleR/

## Usage in DAPIDL

```bash
# Run SingleR with both references
Rscript scripts/run_singler_dual.R \
  /path/to/cell_feature_matrix.h5 \
  output/singler_predictions.csv
```

Output is used by the PopV-style ensemble annotator for improved cell type prediction.
