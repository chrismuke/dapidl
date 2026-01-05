# Comprehensive Fine-Grained Cell Type Annotation Benchmark

**Date:** January 5, 2026
**Dataset:** Xenium Breast Cancer (Janesick et al.)
**Task:** Fine-grained classification (20 cell types)

## Executive Summary

This report presents a comprehensive benchmark of cell type annotation methods for fine-grained classification on spatial transcriptomics data. We tested **4 annotation methods** (SingleR, CellTypist, scType, SCINA) across **15 combinations** (all possible 1, 2, 3, and 4-method ensembles) on **2 replicate datasets**.

### Key Findings

1. **Best Overall Combination**: **scType + SingleR** (2-method ensemble)
   - Rep1: 55.97% accuracy, **0.2829 macro F1**
   - Rep2: 54.52% accuracy, **0.2668 macro F1**

2. **Adding More Methods Hurts Performance**: 4-method ensembles perform *worse* than the best 2-method ensemble.

3. **SingleR is Essential**: All top-performing combinations include SingleR.

4. **Cell Ontology Standardization**: Created comprehensive mapping from annotator outputs to [Cell Ontology](https://obofoundry.org/ontology/cl.html) terms.

---

## Methods Tested

| Method | Type | Reference | Implementation |
|--------|------|-----------|----------------|
| **SingleR** | Reference-based | Aran et al. 2019 | R/Bioconductor |
| **CellTypist** | Classifier-based | Dominguez Conde et al. 2022 | Python |
| **scType** | Marker-based | Ianevski et al. 2022 | Python (DAPIDL) |
| **SCINA** | EM-based | Zhang et al. 2019 | Python (DAPIDL) |

---

## Results

### Rep1 Dataset (167,780 cells)

#### Single Methods

| Method | Accuracy | Macro F1 | Cohen Kappa |
|--------|----------|----------|-------------|
| SingleR | 59.54% | 0.2716 | 0.508 |
| scType | 47.59% | 0.2204 | 0.378 |
| SCINA | 41.90% | 0.2182 | 0.331 |
| CellTypist | 44.61% | 0.2114 | 0.360 |

#### Best Combinations by Size

| Size | Best Combination | Accuracy | Macro F1 |
|------|------------------|----------|----------|
| 1 | SingleR | 59.54% | 0.2716 |
| 2 | **scType + SingleR** | 55.97% | **0.2829** |
| 3 | scina + sctype + singler | 53.08% | 0.2715 |
| 4 | celltypist + scina + sctype + singler | 53.86% | 0.2736 |

### Rep2 Dataset (118,752 cells)

#### Single Methods

| Method | Accuracy | Macro F1 | Cohen Kappa |
|--------|----------|----------|-------------|
| SingleR | 57.63% | 0.2595 | 0.498 |
| scType | 45.41% | 0.2135 | 0.369 |
| CellTypist | 43.12% | 0.2060 | 0.342 |
| SCINA | 34.83% | 0.1997 | 0.265 |

#### Best Combinations by Size

| Size | Best Combination | Accuracy | Macro F1 |
|------|------------------|----------|----------|
| 1 | SingleR | 57.63% | 0.2595 |
| 2 | **scType + SingleR** | 54.52% | **0.2668** |
| 3 | celltypist + sctype + singler | 53.37% | 0.2587 |
| 4 | celltypist + scina + sctype + singler | 51.54% | 0.2611 |

---

## Analysis

### Why scType + SingleR Works Best

1. **Complementary Strengths**:
   - SingleR excels at reference-based label transfer using transcriptomic similarity
   - scType uses curated marker gene signatures for cell type identification
   - Together they validate each other's predictions

2. **Diminishing Returns with More Methods**:
   - Adding CellTypist and SCINA introduces noise
   - These methods have lower individual accuracy, diluting the ensemble signal
   - Confidence-weighted voting can't fully compensate for weak predictors

3. **Consistency Across Replicates**:
   - The same pattern holds on both rep1 and rep2
   - scType + SingleR consistently outperforms 3 and 4-method combinations

### Performance by Cell Type Category

Fine-grained classification remains challenging because:

1. **Tumor Subtypes**: "Invasive_Tumor", "DCIS_1", "DCIS_2", "Prolif_Invasive_Tumor" are difficult to distinguish
2. **Macrophage Subtypes**: "Macrophages_1" vs "Macrophages_2" require subtle expression differences
3. **Hybrid Cells**: "Stromal_&_T_Cell_Hybrid", "T_Cell_&_Tumor_Hybrid" are inherently ambiguous
4. **Rare Populations**: "LAMP3+_DCs", "Mast_Cells" have few training examples

### Comparison: Coarse vs Fine-Grained

| Task | Best Method | Accuracy | Macro F1 |
|------|-------------|----------|----------|
| Coarse (4 classes) | SingleR alone | 90.0% | 0.850 |
| Fine-grained (20 classes) | scType + SingleR | 55.0% | 0.283 |

The ~35% accuracy drop from coarse to fine-grained classification is expected given the 5x increase in classes and presence of similar subtypes.

---

## Cell Ontology Standardization

To ensure consistent cell type naming across methods, we created a mapping through [Cell Ontology](https://obofoundry.org/ontology/cl.html) (CL):

### Mapping Strategy

```
Annotator Output → Cell Ontology ID → Ground Truth Label
```

Example mappings:
| Annotator Output | CL ID | CL Name | Ground Truth |
|-----------------|-------|---------|--------------|
| CD4+ T-cells (SingleR) | CL:0000624 | CD4-positive, alpha-beta T cell | CD4+_T_Cells |
| CD4-positive, alpha-beta T cell (CellTypist) | CL:0000624 | CD4-positive, alpha-beta T cell | CD4+_T_Cells |
| CD4+ T cells (scType) | CL:0000624 | CD4-positive, alpha-beta T cell | CD4+_T_Cells |
| CD4_T_cells (SCINA) | CL:0000624 | CD4-positive, alpha-beta T cell | CD4+_T_Cells |

Full mapping available in: `src/dapidl/pipeline/components/annotators/cell_ontology_mapping.py`

---

## Recommendations

### For Fine-Grained Annotation

1. **Use scType + SingleR ensemble** as the default combination
2. Avoid 4-method ensembles - they underperform simpler combinations
3. Consider hierarchical classification (coarse → fine-grained) for better accuracy

### For Cell Type Naming

1. **Standardize to Cell Ontology** for cross-study compatibility
2. Use [CellOntologyMapper](https://github.com/Starlitnightly/CellOntologyMapper) for automated mapping
3. Document dataset-specific labels (like tumor subtypes) explicitly

### For Future Work

1. Test additional methods: scANVI, scArches, Azimuth (when dependencies available)
2. Implement hierarchical classification pipeline
3. Add weighted ensemble with learned weights (not just confidence-based)

---

## ClearML Experiment Tracking

All results logged to ClearML:
- Project: `DAPIDL/Finegrained-Benchmark`
- Rep1 3-4 combos: https://app.clear.ml/projects/63966ad2e31447338db5c6f8a1905e77/experiments/db08540441b84b08893b4702ae88068d
- Rep2 all combos: https://app.clear.ml/projects/63966ad2e31447338db5c6f8a1905e77/experiments/5049bfde978a4dabbc0e90dd310c9572

---

## References

1. Aran D, et al. Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage. Nat Immunol. 2019.
2. Dominguez Conde C, et al. Cross-tissue immune cell analysis reveals tissue-specific features in humans. Science. 2022.
3. Ianevski A, et al. Fully-automated and ultra-fast cell-type identification using specific marker combinations from single-cell transcriptomic data. Nat Commun. 2022.
4. Zhang AW, et al. Probabilistic cell-type assignment of single-cell RNA-seq for tumor microenvironment profiling. Nat Methods. 2019.
5. Cell Ontology: https://obofoundry.org/ontology/cl.html
6. CellOntologyMapper: https://github.com/Starlitnightly/CellOntologyMapper
