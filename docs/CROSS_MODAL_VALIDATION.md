# Cross-Modal Validation for Cell Type Annotations

## Overview

This document describes methods for validating CellTypist predictions **without ground truth annotations** using orthogonal approaches that leverage different biological signals.

## The Problem

When using CellTypist (or any automated annotation tool) on new datasets:
- No ground truth available for validation
- Risk of propagating annotation errors to downstream models (like DAPIDL)
- Need confidence metrics to filter training data

## Solution: Cross-Modal Validation Framework

Three independent validation approaches that should **agree** if annotations are correct:

### 1. Leiden Clustering Consistency (Transcriptomic Self-Validation)

**Principle**: If CellTypist correctly identifies cell types, unsupervised clustering on the same gene expression data should produce similar groupings.

**Method**:
```python
from dapidl.validation import compute_leiden_metrics

ari, nmi, best_resolution = compute_leiden_metrics(adata, "celltypist_labels")
```

**Metrics**:
| Metric | Range | Interpretation |
|--------|-------|----------------|
| ARI (Adjusted Rand Index) | -1 to 1 | > 0.8 excellent, 0.5-0.8 moderate, < 0.5 investigate |
| NMI (Normalized Mutual Info) | 0 to 1 | Higher = better agreement |

**Key Papers**:
- SpatialLeiden (Genome Biology, 2025) - Spatially-aware clustering
- spARI (Biometrics, 2025) - Spatially-aware ARI metric
- ⚠️ Nature Biotech 2025: Avoid Silhouette score for single-cell data

### 2. DAPI Morphology Cross-Validation (Independent Modality)

**Principle**: Different cell types have distinct nuclear morphology. A model trained on DAPI images provides an **independent signal** from gene expression.

**Scientific Basis** (from literature review):
- Immune cells: Small, spherical, densely-stained nuclei
- Epithelial cells: Large, round nuclei with higher nucleus-to-cytoplasm ratio
- Stromal/Fibroblasts: Flat-ellipsoidal, spindle-shaped nuclei

**Literature Performance**:
| Study | Task | Accuracy |
|-------|------|----------|
| NephNet3D (2021) | Kidney cell classification from DAPI | 80.26% |
| Cancer cell lines | 8-class classification | 94.6% |
| CellCycleNet | Cell cycle staging from DAPI | AUROC 0.95 |
| **DAPIDL (our results)** | 3-class coarse | F1 = 0.73 ✓ |

**Method**:
```python
from dapidl.validation import compute_dapi_agreement, extract_morphology_embeddings

# Direct prediction comparison
agreement, conf, corr, per_class = compute_dapi_agreement(
    model, patches, celltypist_labels, celltypist_confidence, class_names
)

# Or cluster in morphology space independently
embeddings = extract_morphology_embeddings(model, patches)
morphology_clusters = cluster_morphology_embeddings(embeddings, n_clusters=3)
```

**Key Insight**:
> "Broad cell type families have distinct, essentially non-overlapping morpho-electric phenotypes" - Nature 2020

This validates why DAPIDL works and why it provides orthogonal validation.

### 3. Multi-Method Consensus (Annotation Ensemble)

**Principle**: Multiple annotation methods should agree on correct annotations. Disagreement indicates uncertainty.

**Methods Available in DAPIDL**:

| Method | Description | Speed |
|--------|-------------|-------|
| CellTypist Consensus | Vote across multiple tissue models | Fast |
| popV | Ensemble of 8 algorithms + ontology voting | Medium |
| Hierarchical | Tissue model → specialized refinement | Fast |

**Implementation**:
```python
from dapidl.data.annotation import CellTypeAnnotator, AnnotationStrategy

# Multi-model CellTypist consensus
annotator = CellTypeAnnotator(
    model_names=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
    strategy=AnnotationStrategy.CONSENSUS
)

# Or popV ensemble (8 methods)
annotator = CellTypeAnnotator(strategy=AnnotationStrategy.POPV)
```

**Consensus Score Interpretation**:
- ≥ 0.5: Majority of methods agree → High confidence
- < 0.5: Disagreement → Flag for review or exclude

## Combined Confidence Scoring

```python
from dapidl.validation import compute_confidence_tiers

tiers = compute_confidence_tiers(
    celltypist_confidence=celltypist_conf,
    dapi_confidence=dapi_conf,
    consensus_score=consensus,
    dapi_agreement=agreement_mask
)

# Use tiers for training data selection
high_confidence = tiers.filter(pl.col("confidence_tier") == "high")
```

**Tier Definitions**:
| Tier | Criteria | Action |
|------|----------|--------|
| **High** | Combined score ≥ 0.7 | Use for training |
| **Medium** | Score 0.4-0.7 | Include with caution |
| **Low** | Score < 0.4 | Exclude or manual review |

## Validation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROSS-MODAL VALIDATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CellTypist Prediction ──┬── Leiden Cluster Match? ── ARI/NMI  │
│  (Gene Expression)       │                                      │
│                          ├── DAPI Model Agreement? ── Agreement │
│                          │   (Morphology)             Score     │
│                          │                                      │
│                          └── Multi-Model Consensus? ─ Voting    │
│                              (popV/CellTypist ensemble)         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  CONFIDENCE TIERS:                                              │
│                                                                 │
│  🟢 HIGH: All methods agree, confidence ≥ 0.7                   │
│  🟡 MEDIUM: 2/3 methods agree, confidence 0.4-0.7               │
│  🔴 LOW: Disagreement or low confidence                         │
└─────────────────────────────────────────────────────────────────┘
```

## ClearML Pipeline Integration

The validation can be run as a ClearML pipeline step:

```bash
# Run validation pipeline
uv run dapidl clearml-pipeline run \
    --pipeline validation \
    --dataset-id <dataset_id> \
    --model-id <model_id>
```

## Key References

### Leiden Clustering & Metrics
- SpatialLeiden (Genome Biology, 2025)
- spARI - Spatially aware ARI (Biometrics, 2025)
- "Shortcomings of silhouette" (Nature Biotech, 2025)

### Morphology-Based Classification
- NephNet3D - 3D nuclear signatures (PMC, 2021)
- GHIST - Gene expression from histology (Nature Methods, 2025)
- SPiRiT - Vision transformer for spatial transcriptomics (2024)

### Consensus Methods
- popV - Multi-method ensemble (Nature Genetics, 2024)
- scConsensus - Supervised + unsupervised (BMC Bioinformatics, 2021)
- mLLMCelltype - Multi-LLM consensus (bioRxiv, 2025)

### DAPI & Cell Type Morphology
- "Nuclear morphologies: diversity and functional relevance" (PMC, 2017)
- "Deep-learning quantified cell-type-specific nuclear morphology" (bioRxiv, 2023)
- "Phenotypic variation of transcriptomic cell types" (Nature, 2020)

## Implementation Files

```
src/dapidl/validation/
├── __init__.py
├── cross_modal.py          # Core validation functions
└── clearml_pipeline.py     # ClearML pipeline integration

Key functions:
- compute_leiden_metrics()
- compute_dapi_agreement()
- extract_morphology_embeddings()
- cluster_morphology_embeddings()
- compute_confidence_tiers()
- quick_validate()
```
