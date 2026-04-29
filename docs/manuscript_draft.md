# DAPIDL: Predicting Cell Types from DAPI Nuclear Staining Using Spatial Transcriptomics as Automatic Training Data

## Authors
Christian Muke^1^

^1^ Independent Researcher

---

## Abstract

Cell type identification is fundamental to understanding tissue architecture, yet current methods require either expensive spatial transcriptomics or labor-intensive manual annotation. We present DAPIDL, a framework that uses spatial transcriptomics platforms (10x Xenium, MERSCOPE, STHELAR) as automatic training data sources to build deep learning models that predict cell types from **DAPI nuclear staining alone** -- a near-universal, single-channel, sub-dollar imaging modality. To characterise the practical reach of DAPI-only classification we systematically evaluated over 700 experimental configurations spanning 20+ annotation methods, 7 backbone architectures, 4 patch sizes, and 6 foundation models across 3 platforms, 16 tissue types, and 11 million cells, then benchmarked DAPI against co-registered H&E and naive 4-channel multimodal fusion on the same dataset. Our ensemble annotation pipeline combines CellTypist, SingleR, scType, SCINA, BANKSY spatial clustering, and popV consensus voting, standardized through a 9-level Cell Ontology mapping cascade; BANKSY spatial clustering (F1=0.802) outperforms all cell-level annotation methods. For DAPI classification, Cell-DINO LoRA fine-tuning (F1=0.463) outperforms supervised EfficientNetV2-S (F1=0.446) on the same data, revealing that DINO-based pretraining objectives produce superior features regardless of domain. The best DAPI-only model achieves macro F1=0.60 for coarse classification using hierarchical curriculum learning. **In a within-STHELAR head-to-head benchmark (1.26 M patches, 9 classes, 16 tissues, identical architecture, splits, and hyperparameters), DAPI alone reaches macro F1=0.478, H&E alone F1=0.524 (+0.046), and naive 4-channel DAPI+H&E fusion F1=0.530 (+0.006 over H&E). DAPI dominates per-class on epithelial cells (F1=0.894 vs H&E 0.470), while H&E dominates pericytes, macrophages, B/T cells and fibroblasts; naive early fusion cannot preserve both -- pericyte F1 collapses from 0.929 (H&E) to 0.523 (multimodal). DAPI thus captures most of the single-modality F1 H&E achieves, while H&E adds discriminative tissue-context information for stromal and lymphocyte subtypes that nuclear morphology alone cannot resolve.** GradCAM corroborates this asymmetry: the model correctly identifies kidney-shaped macrophage nuclei (100%) and spindle-shaped myoepithelial nuclei (90%), but produces diffuse attention on lymphocytes whose identity is determined by surface markers invisible to DAPI. Our GT-free confidence filtering system improves F1 by 10–16% through 4-signal quality estimation without ground truth. Crucially, leave-one-tissue-out evaluation across all 16 STHELAR tissues reveals universal tissue-shortcutting in *both* image modalities: mean accuracy collapse of 1.78× for DAPI (range 1.22×–4.79×, brain 0.956→0.200), with H&E showing the same failure mode at slightly smaller magnitude (mean LOTO acc 0.506 vs DAPI 0.459). LOTO validation should therefore be a required reporting standard for any multi-tissue DAPI- or H&E-based classifier. The fully automated pipeline processes raw spatial transcriptomics data into trained DAPI classifiers without manual intervention, offering a practical path toward inexpensive, high-throughput cell typing while honestly bounding what is achievable from nuclear staining alone.

**Keywords:** spatial transcriptomics, cell type classification, DAPI staining, deep learning, nuclear morphology, H&E benchmark, multimodal fusion, Xenium, MERSCOPE, STHELAR, BANKSY, foundation models, Cell Ontology, tissue shortcut

---

## 1. Introduction

Spatial transcriptomics has transformed our ability to study tissue architecture by simultaneously measuring gene expression and spatial location of individual cells (1, 2). Platforms such as 10x Genomics Xenium (3, 4), Vizgen MERSCOPE (34), STHELAR (27), NanoString CosMx, Resolve Biosciences Molecular Cartography, and BGI Stereo-seq now routinely profile hundreds of thousands to millions of cells per tissue section, generating rich datasets that combine imaging modalities with molecular readouts.

A key output of these platforms is cell type annotation -- the assignment of each cell to a biological category based on its gene expression profile. Computational methods including CellTypist (9), SingleR (11), scType (10), popV (13), Azimuth (12), and spatial-aware approaches like BANKSY (25) achieve high accuracy when gene panels contain sufficient marker genes. However, systematic benchmarking reveals significant method-dependent biases: no single method performs best across all cell types, with expression-based methods excelling at lymphocyte subtypes while spatial methods dominate endothelial and macrophage classification (6). These expression-based annotations require expensive spatial transcriptomics instruments and consumables ($1000+/slide), limiting their deployment to research settings.

In contrast, DAPI (4',6-diamidino-2-phenylindole) staining is a universal, inexpensive fluorescent nuclear marker (<$1/slide) used in virtually all histological workflows. DAPI images reveal nuclear morphology -- size, shape, chromatin texture, and spatial arrangement -- features that correlate with cell identity (7, 8). Previous work on nuclear morphology-based classification includes NephNet3D achieving 80.3% balanced accuracy on 3D DAPI in kidney (Woloshuk et al.), CelloType reaching mAP=0.40 on DAPI-only (achieving 90% of multimodal DAPI+transcripts performance), and MitoCheck obtaining F1=0.84 for nuclear phenotype classification. The critical observation motivating this work is that spatial transcriptomics platforms co-acquire DAPI images alongside gene expression data, creating naturally paired training datasets where expression-derived cell type labels can supervise a morphology-based classifier.

We present DAPIDL (DAPI Deep Learning), a fully automated pipeline that: (1) ingests raw spatial transcriptomics data from multiple platforms; (2) generates consensus cell type annotations using an ensemble of 11 computational methods including spatial clustering; (3) applies GT-free confidence filtering to remove low-quality labels; (4) standardizes heterogeneous cell type vocabularies via Cell Ontology mapping; (5) extracts normalized DAPI patches around each annotated nucleus; and (6) trains a convolutional neural network or foundation model to classify cell types from DAPI morphology alone.

The DAPI-only focus is deliberate. DAPI is the cheapest, most universal, and most format-stable nuclear stain in clinical and research histology — present in essentially every fluorescence-imaging workflow, requiring a single excitation/emission channel, and reusable across decades of archival tissue. In contrast, H&E (the brightfield gold standard for human pathology) carries cytoplasmic, stromal and tissue-context information but requires a different acquisition pipeline, costlier scanners, and is not co-acquired by the major spatial transcriptomics platforms by default. The natural question is therefore not "is DAPI as good as H&E?" but "*how much* of the achievable image-only single-modality cell-type signal does DAPI alone deliver, and where does it systematically fall short?". To answer this we use STHELAR (27), the only public spatial transcriptomics dataset with co-registered DAPI and H&E across 16 human tissues, to benchmark DAPI against H&E and against a naive 4-channel multimodal fusion under matched architecture, splits, and hyperparameters. The benchmark serves as an *upper-bound reference* for DAPI-only and a guide to which classes are inherently difficult from nuclear staining alone.

---

## 2. Methods

### 2.1 Data Sources and Platform Support

DAPIDL supports three spatial transcriptomics platforms with distinct data formats, imaging characteristics, and DAPI resolutions (Table 1).

**Table 1. Platform characteristics and imaging modalities.**

| Property | Xenium (10x) | MERSCOPE (Vizgen) | STHELAR | CosMx (NanoString) |
|----------|-------------|-------------------|---------|-------------------|
| DAPI pixel size | 0.2125 um/px | 0.108 um/px | 0.2125 um/px | 0.12 um/px |
| 128px patch field | 27.2 um | 13.8 um | 27.2 um | 15.4 um |
| DAPI intensity | Median ~800 | Median ~13,000 | Median ~800 | -- |
| Genes per panel | 300-8,456 | 500 | ~380 | 960-18,935 |
| Additional channels | Boundary stain, interior stain | PolyT | H&E co-registered | CD298/b2m membrane, 4 IF |
| Segmentation | Native (v1: algorithmic, v3: 3-method 93.9% stain-guided) | Watershed DAPI | CellViT from H&E | Membrane-guided |
| Cell metadata | Parquet | CSV + affine | Zarr (SpatialData) | -- |
| Expression format | HDF5 (sparse CSC) | CSV (dense) | Zarr arrays | -- |

Platform-specific data readers (`XeniumDataReader`, `MerscopeDataReader`, `SthelarDataReader`) handle format differences. MERSCOPE images exceed 19 GB per z-plane (94,805 x 110,485 uint16), requiring memory-mapped access via tifffile. STHELAR uses nested SpatialData zarr format with a 5-level DAPI image pyramid (level 0: ~54K x 48K).

DAPI resolution directly impacts classification: MERSCOPE provides 2x finer resolution than Xenium (0.108 vs 0.2125 um/px), meaning a 128px patch covers only 13.8 um on MERSCOPE versus 27.2 um on Xenium -- potentially insufficient for multi-cellular context on MERSCOPE.

**Datasets.** We utilized 73+ datasets: 42 Xenium datasets (8 tissue types, 8.6M cells), 1 MERSCOPE dataset (breast, 713K cells, 17 fine-grained classes), and 31 STHELAR datasets (16 tissue types, 11M cells) from BioStudies S-BIAD2146 (27). Additionally, the Vizgen FFPE Showcase provides 18 MERSCOPE FFPE datasets from 8 cancer types totaling 8.7M cells (freely available on Google Cloud). Key evaluation datasets include Xenium breast cancer replicate 1 (167,780 cells) with expert ground truth from Janesick et al. (3).

### 2.2 Ensemble Cell Type Annotation

Rather than relying on a single annotation method, DAPIDL employs an ensemble approach. We evaluated 20+ annotation methods across 700+ configurations on 6 datasets (2 Xenium + 4 STHELAR), spanning expression-based, marker-based, spatial, and reference-mapping approaches (Table 2).

**Table 2. Annotation methods evaluated.**

| Method | Type | Category | Reference |
|--------|------|----------|-----------|
| CellTypist | Logistic regression | Expression | Dominguez Conde et al. (9) |
| SingleR | Reference correlation | Expression | Aran et al. (11) |
| scType | Marker gene scoring | Marker | Ianevski et al. (10) |
| SCINA | EM-based markers | Marker | -- |
| BANKSY | Spatial clustering + labeling | Spatial | Singhal et al. (25) |
| popV | 8-method consensus | Ensemble | Ergen et al. (13) |
| Azimuth | Seurat reference mapping | Reference | Hao et al. (12) |
| scANVI | Semi-supervised VAE | Deep learning | De Donno et al. (30) |
| scArches | Surgical reference mapping | Transfer | -- |
| OnClass | Cell Ontology-aware | Ontology | -- |
| CellAssign | Probabilistic marker | Marker | -- |

**Tissue-specific model selection.** For CellTypist, tissue-appropriate models are selected from a curated registry of 42 dataset-specific assignments covering 20 tissue types. We tested 298 CellTypist model consensus combinations; the best 3-model combination (Liver + Nuclei_Lung + Intestinal) achieved F1=0.554, inferior to BANKSY (F1=0.802).

**Marker database comparison.** Four marker databases were evaluated: Custom markers (best for Xenium, F1=0.592), CellMarker 2.0 (26,915 markers), PanglaoDB (8,211 markers, 178 cell types), and scTypeDB (~1,200 markers, lacks breast tissue). Rankings reversed between Xenium and STHELAR, with PanglaoDB performing best on STHELAR.

**PopV architecture.** PopV combines 8 sub-methods (Random Forest, SVM, CellTypist, OnClass, scVI+kNN, scANVI, BBKNN+kNN, Scanorama+kNN) with ontology-based vote resolution via ancestor propagation. Consensus score calibration: 8/8 agreement = 98% accuracy, 7/8 = 95%, 6/8 = 90%, 5/8 = 80%, <=3 = <50%. Retraining on target-specific references is essential: popV with Tabula Sapiens retrain achieves F1=0.651 versus F1=0.110 without retraining (6x improvement).

### 2.3 BANKSY Spatial Annotation Architecture

A key finding is that spatial-aware annotation via BANKSY (25) outperforms all cell-level methods. BANKSY augments each cell's expression profile with neighborhood gene expression (radius r, mixing parameter lambda), then clusters the augmented space.

**BANKSY-first vs cell-level consensus.** We compared two architectures: (1) running BANKSY first, then labeling clusters with marker enrichment, and (2) using popV-style cell-level voting. BANKSY-first achieved F1=0.802 (coarse, 4 classes) while popV-as-voter achieved only F1=0.459. This represents a paradigm shift: spatial context before annotation, not after.

**BANKSY + LLM annotation.** BANKSY clusters annotated by an LLM achieved F1=0.822 (coarse) and F1=0.350 (medium), with the LLM providing biological explanations for 4 misclassified clusters.

**Per-class routing.** No single method wins all cell types. BANKSY excels at spatial types (Endothelial F1=0.844, Macrophage F1=0.680, B_Cell F1=0.683), while CellTypist excels at expression-defined subtypes (CD4_T F1=0.654, CD8_T F1=0.565, Fibroblast F1=0.699). A routing strategy assigning each cell type to its best method would further improve performance.

**BANKSY parameter optimization.** Optimal parameters: k=10 neighbors, lambda=0.2, r=0.3 for breast tissue. Higher r (3.0) over-smooths, reducing T_NK F1 from 0.885 to 0.024.

### 2.4 GT-Free Confidence Filtering

A critical innovation is the GT-free annotation validation module, which estimates label confidence without ground truth using 4-6 orthogonal signals:

1. **Marker enrichment score**: Expected marker gene expression for the assigned cell type (PanglaoDB, 8,211 markers)
2. **Spatial coherence**: Whether neighboring cells share the same label
3. **Cross-method consensus**: Agreement rate between independent annotation methods
4. **Proportion plausibility**: Whether observed cell type proportions match expected tissue composition (8 supported tissues)
5. **IsolationForest anomaly detection**: 6.5% anomaly rate on Epithelial predictions
6. **KNN entropy**: 70.9% pure neighborhoods; Epithelial 93%, Endothelial/Immune/Stromal ~50%

**Filtering results.** On Xenium breast rep2, confidence threshold 0.3 retains 81.5% of cells and improves F1 by +10.6% (0.758 to 0.838). On rep1, the same threshold gains +15.9% F1 (0.558 to 0.644). At threshold 0.7, accuracy reaches 95.8% but only 18.6% of cells are retained.

**Hierarchical confidence.** Evaluating agreement independently at coarse/medium/fine levels yields 93.9% cell retention versus 35.5% with binary filtering -- dramatically better coverage at equivalent quality.

**Limitation.** Filtering improves F1 monotonically when base method F1 > 0.7, but cannot fix systematic errors (e.g., BANKSY misclassifying entire regions).

### 2.5 Cell Ontology Standardization

Different annotators produce different labels for identical cell types. We address this through CLMapper, a 9-level Cell Ontology (CL) mapping cascade:

1. Curated annotator-specific mappings (170 entries, confidence=1.0)
2. Ground truth dataset mappings (110 entries covering Xenium, MERSCOPE, STHELAR)
3. Exact name match against 66 curated CL terms
4. Synonym match (130 synonyms, confidence=0.95)
5. OBO ontology name match (confidence=0.90)
6. OBO synonym match (confidence=0.85)
7. Pattern matching via 40+ regex rules for CellTypist naming conventions
8. Fuzzy string matching via RapidFuzz (threshold=0.85)
9. Keyword-based fallback (confidence=0.60)

Labels map into a three-level hierarchy: broad (4 categories), coarse (~10-15 types), and fine (50+ CL terms). Multi-granularity evaluation shows coarse F1=0.811, medium F1=0.577, fine F1=~0.05 -- confirming that fine-grained classification fails on limited gene panels.

### 2.6 Nucleus Segmentation

We benchmarked 8 segmentation methods across MERSCOPE and Xenium platforms using StarPose (v0.2.0), our adaptive multi-modal segmentation framework.

**Table 3. Segmentation benchmark (MERSCOPE breast, 5 FOV types).**

| Method | Avg Cells | Recovery | Solidity | Key Characteristic |
|--------|-----------|----------|----------|-------------------|
| Consensus topological | 128 | **0.604** | 0.939 | Best overall recovery |
| Cellpose (SAM/cyto3/nuclei) | 172 | 0.574 | 0.962 | All 3 variants identical in CP4 |
| StarDist | 243 | 0.487 | **0.965** | Most cells, highest solidity |
| InstanSeg | 65 | 0.260 | 0.933 | Poor on DAPI-only (designed for H&E) |

On Xenium, Cellpose aggressive achieved best recovery (0.56-0.93 across FOVs) but with potential over-segmentation. The adaptive consensus adds no benefit on Xenium where methods are already well-calibrated (StarDist/Cellpose ratio < 2x).

**CellViT comparison (STHELAR).** StarPose nucleus detection compared against CellViT H&E-based cell boundaries showed StarDist excelling at medium-to-dense FOVs (recovery 0.82-0.91) while Cellpose had higher precision in dense regions.

**Transcript-guided expansion.** For cell boundary definition, Proseg (18) achieves ~98% transcript-to-cell assignment versus ~80% with Voronoi tessellation, directly improving annotation quality.

### 2.7 LMDB Dataset Creation and Data Engineering

Annotated cells are converted into training datasets by extracting DAPI patches centered on each cell nucleus.

- **Patch sizes evaluated**: 32, 64, 128, 256 pixels
- **Normalization**: Adaptive percentile (1st-99.5th percentile, then z-score). Handles the 16x intensity difference between Xenium and MERSCOPE.
- **Physical size normalization**: MERSCOPE patches resized to match Xenium resolution (0.2125 um/px)
- **Edge exclusion**: Cells within 64 pixels of image boundaries excluded

**Storage format performance.** Zarr format achieves ~60% GPU utilization during training. LMDB format with NVIDIA DALI achieves 97-98% GPU utilization -- a critical engineering optimization for the 474K+ patch datasets.

**Data leakage prevention.** Patient-level splitting ensures no cell from the same tissue section appears in both training and test sets. Dataset hashing (SHA256 of file sizes + first/last 1MB) enables deterministic reproducibility.

### 2.8 Model Architecture

**Transfer learning classifier.** The primary architecture:
```
DAPI patch (1, 128, 128) -> SingleChannelAdapter (3, 128, 128)
    -> Backbone (ImageNet pretrained) -> features
    -> Dropout(0.3) -> Linear(features, N_classes)
```

**Table 4. Backbone architectures evaluated.**

| Backbone | Params | Features | Best F1 | Pretraining |
|----------|--------|----------|---------|-------------|
| EfficientNetV2-S | 20M | 1792 | 0.596 | ImageNet |
| ConvNeXt-Tiny | 28M | 768 | 0.500 | ImageNet |
| ResNet-50 | 25M | 2048 | 0.490 | ImageNet |
| Phikon-v2 ViT-L | 304M | 1024 | 0.412 | H&E pathology |

**Foundation model LoRA fine-tuning (STHELAR, 2.5M cells, 4 classes).**

| Model | Pretraining | Domain | LoRA F1 | Frozen F1 |
|-------|------------|--------|---------|-----------|
| **Cell-DINO ViT-L/16** | DINO | Fluorescence | **0.463** | 0.407 |
| DINOv2 ViT-S/14 | DINO | Natural images | 0.453 | 0.388 |
| OpenPhenom ViT-S/16 | MAE | Fluorescence | 0.393 | 0.374 |
| NuSPIRe ViT-B/16 | MAE | DAPI nuclei (15.5M) | 0.339 | 0.305 |

**Critical finding: DINO >> MAE.** The pretraining objective matters more than domain, model size, or input resolution. DINOv2 (trained on natural images via DINO) at 112x112 achieves F1=0.447, outperforming NuSPIRe (trained on 15.5M DAPI nuclear crops via MAE) at F1=0.339 -- a +32% gap with identical input. NuSPIRe fails because MAE learns reconstruction (perturbation detection), not discriminative classification boundaries.

**Context radius ablation.** Masking the patch to different radii around the nucleus:
- d25 mask (+5um perinuclear context): F1=0.380 (optimal)
- d0 nucleus-only: F1=0.368
- No mask (full 128px): F1=0.347
Perinuclear context improves in-domain F1, but hurts cross-domain transfer (ranking flips: foundation models generalize better than supervised despite lower in-domain F1).

**Class imbalance handling.** Three combined strategies:
- Focal loss (22) with gamma=2.0 and per-class alpha weights
- WeightedRandomSampler for balanced mini-batches
- **Class weight capping** (max_weight_ratio=10.0): uncapped weights caused F1=0.001 (mode collapse) on 17-class MERSCOPE; capping at 10x achieved F1=0.20 -- a **180x improvement**

**Curriculum learning.** For multi-tissue training, a hierarchical curriculum progressively activates classification heads:
- Phase 1 (epochs 0-20): Coarse only (4 categories)
- Phase 2 (epochs 20-50): Coarse + medium
- Phase 3 (epochs 50+): All three levels
Backbone frozen for 2 epochs at transitions, LR reduced 10x. A known phase-skip bug (Phase 1 -> Phase 3 skipping Phase 2) caused F1 to crash 83% at epoch 12 when triggered.

**Confidence-tier weighting.** Multi-dataset training weights labels by quality: tier 1 (ground truth, weight=1.0), tier 2 (ensemble consensus, weight=0.8), tier 3 (single annotator, weight=0.5). Tissue-balanced sampling via `sqrt(N)` prevents large-dataset dominance.

### 2.9 Reproducibility

Every training run logs: git commit hash, full environment, dataset hash (SHA256), all hyperparameters, and 5 artifact types to Weights & Biases. A 5-step reproduction protocol enables exact replication. ClearML tracks pipeline lineage with parent-child dataset relationships. The codebase spans 55,823 LOC across 218 files with 229 commits.

---

## 3. Results

### 3.1 Annotation Method Benchmarking

We systematically evaluated 12+ annotation methods on STHELAR breast_s0 (574,869 cells, 380 genes, 7 broad classes) against STHELAR reference labels derived from Tangram deconvolution (27, 39).

**Table 5. Annotation method comparison on STHELAR breast (ranked by macro F1).**

| Method | F1 Macro | Accuracy | Best Class | Worst Class | Runtime |
|--------|----------|----------|------------|-------------|---------|
| popV retrain (Tabula Sapiens) | **0.651** | 87.96% | Epi 0.973 | Specialized 0.0 | ~2,000s |
| Combined Tangram+CellTypist | 0.623 | 89.53% | Epi 0.967 | B_Plasma 0.0 | ~3,400s |
| popV DISCO retrain | 0.582 | 88.36% | Epi 0.983 | Adipocyte 0.0 | ~2,023s |
| BANKSY finegrained | 0.486 | 84.52% | **T_NK 0.885** | Mast 0.0 | ~6,039s |
| Tangram+markers | 0.473 | 79.57% | Epi 0.949 | B_Plasma 0.0 | ~2,100s |
| Complete pipeline (6-step) | 0.425 | 73.35% | Epi 0.925 | Fibro 0.0 | ~8,335s |
| popV Hub (no retrain) | 0.416 | 80.27% | Epi 0.946 | Myeloid 0.002 | -- |
| BANKSY r=3.0 | 0.386 | 72.58% | Epi 0.929 | B_Plasma 0.0 | ~12,368s |
| OnClass DISCO | 0.110 | 58.14% | Epi 0.761 | Adipocyte 0.0 | -- |
| popV DISCO (no retrain) | 0.110 | 58.79% | Epi 0.758 | Adipocyte 0.0 | -- |

**Method complementarity.** BANKSY achieves the best T_NK F1 (0.885, +20% vs popV) while popV dominates Blood_vessel (F1=0.891, +43% vs BANKSY). The optimal per-class assignment would exceed any single method.

**Xenium benchmarks.** On Xenium breast with Janesick ground truth (3), the best ensemble (5 CellTypist + 2 SingleR) achieves F1=0.844 for 3 broad categories, consistent across replicates (rep2: F1=0.843). SingleR references (HPCA + Blueprint) boost Stromal F1 from 0.35 to 0.76 (+117%).

**Endothelial detection failure.** Endothelial F1=0.000 across ALL 23+ non-spatial expression-based methods on 313-gene Xenium panels (only 4/9 endothelial markers present). BANKSY spatial clustering is the only method achieving Endothelial F1=0.844 by leveraging spatial gene expression patterns. 92% of endothelial cells are misclassified as Stromal without spatial context.

**Immune subtype accuracy.** At the broad level (Immune vs not), individual annotator accuracy: CD4 T cells 98%, CD8 T cells 98%, B cells 94%, Macrophages 91-96%, Dendritic cells 96%, Mast cells 92%.

**Foundation models vs traditional annotation.** Foundation models (scGPT, Geneformer) do NOT outperform traditional methods in zero-shot annotation on 300-500 gene panels, because these panels represent 1.5-2.5% of the full transcriptome that foundation models were trained on.

### 3.2 Confidence Filtering and Quality Control

**Table 6. GT-free confidence filtering on Xenium breast.**

| Threshold | Rep1 F1 | Rep1 Retention | Rep2 F1 | Rep2 Retention |
|-----------|---------|----------------|---------|----------------|
| None | 0.558 | 100% | 0.758 | 100% |
| 0.3 | 0.644 (+15.9%) | 83.4% | 0.838 (+10.6%) | 81.5% |
| 0.5 | -- | 56.5% | 0.456 | 56.5% |
| 0.7 | -- | 18.6% | **0.918** | 21.1% |

**Per-type confidence scores.**
- CellTypist: Epithelial 0.639, Immune 0.478, Stromal 0.129, Endothelial 0.002
- BANKSY: Epithelial 0.677, Immune 0.556, Stromal 0.508
- Cross-method consensus: Epithelial agreement 0.883, Immune 0.563, Stromal 0.091

**scVI validation.** Independent scVI clustering achieves ARI=0.281, NMI=0.442, mean cluster purity=0.880 against pipeline annotations.

### 3.3 DAPI Classification: Model Comparison

We trained and evaluated 50+ DAPI models (subset of 700+ total configurations) across 7 architectures, 4 patch sizes, and 3 platforms.

**Table 7. DAPI model performance (ranked by validation F1).**

| Configuration | Backbone | Patch | Classes | Val F1 | Test F1 | Platform |
|---------------|----------|-------|---------|--------|---------|----------|
| Hierarchical curriculum | EfficientNetV2-S | 128 | 3 coarse | **0.596** | -- | Xenium |
| Universal 4-class (23 tissues) | EfficientNetV2-S | 128 | 4 coarse | 0.610 | -- | Xenium |
| Cross-platform | EfficientNetV2-S | 128 | 3 coarse | 0.548 | 0.516 | MERSCOPE |
| **Cell-DINO LoRA** | **ViT-L/16** | 128 | 4 coarse | **0.463** | -- | STHELAR |
| DINOv2 LoRA | ViT-S/14 | 128 | 4 coarse | 0.453 | -- | STHELAR |
| Larger patches | EfficientNetV2-S | 256 | fine | 0.472 | -- | Xenium |
| STHELAR DAPI (6 classes) | EfficientNetV2-S | 128 | 6 fine | 0.455* | -- | STHELAR |
| EfficientNet baseline | EfficientNetV2-S | 128 | fine | 0.413 | -- | Xenium |
| Phikon-v2 (frozen) | ViT-L | 128 | fine | 0.412 | -- | Xenium |
| Cell-DINO frozen | ViT-L/16 | 128 | 4 coarse | 0.407 | -- | STHELAR |
| OpenPhenom LoRA | ViT-S/16 | 128 | 4 coarse | 0.393 | -- | STHELAR |
| NuSPIRe LoRA | ViT-B/16 | 128 | 4 coarse | 0.339 | -- | STHELAR |
| Heavy augmentation | EfficientNetV2-S | 128 | 3 coarse | 0.329 | 0.333 | Xenium |

*Training in progress at time of writing (epoch 3/50).

**Expression-based upper bound.** HEIST (28), a graph neural network using gene expression and spatial context, achieves F1=0.913 on 17 fine-grained types, establishing a 0.31 F1 modality gap.

### 3.4 Per-Class Analysis and GradCAM Interpretability

**Table 8. Per-class F1 and GradCAM-derived morphological basis.**

| Cell Type | F1 | GradCAM Finding | Morphological Signal |
|-----------|-----|-----------------|---------------------|
| Epithelial Luminal | 0.956 | Sharp nuclear border focus | Large, round nuclei, distinct chromatin |
| Myoepithelial (ACTA2+) | 0.900* | Elongated shape attention | Spindle-shaped, aligned with basement membrane |
| Invasive Tumor | 0.867 | Irregular boundary attention | High pleomorphism, mitotic figures visible |
| T Cell | 0.804 | Diffuse, uncertain attention | Small, round, dense -- identity via surface markers |
| B Cell | 0.769 | Diffuse attention (like T) | Small round, tend to cluster |
| Vascular Endothelial | 0.756 | Vessel-aligned attention | Elongated nuclei along vessel walls |
| Macrophage | 0.580 | **Sharp kidney-shaped focus (100% accuracy)** | Distinctive kidney/bean-shaped nuclei |
| Dendritic Cell | 0.348 | Larger irregular nuclei | Confused with NK cells |
| Fibroblast | 0.356 | Variable, low confidence | Morphologically diverse across tissues |
| Mast Cell | 0.353 | Too few samples | Insufficient training data |
| Epithelial Basal | 0.004 | 73% misclassified as Luminal | Myoepithelial vs luminal confusion |
| Pericyte | 0.003 | Indistinguishable from fibroblast | No nuclear morphological signal |

*From 6-class epithelial sub-classification experiment (256x256 patches, test F1=0.72).

**GradCAM key insight.** Lymphocyte subtypes (CD4 T, CD8 T, B cells) are distinguished by surface markers (CD3, CD4, CD8, CD19) invisible to DAPI. The model's diffuse attention pattern on lymphocytes confirms no discriminative nuclear signal exists -- this is a fundamental biological limitation, not a model failure.

### 3.5 Cross-Platform and Multi-Tissue Results

**Cross-platform transfer.** Direct Xenium-to-MERSCOPE transfer without normalization achieves only 2.99% accuracy (catastrophic collapse to single class). With adaptive percentile normalization and physical size correction, F1 recovers to 0.548 -- demonstrating normalization is essential, not optional.

**Multi-tissue universal model.** Training on 23 Xenium datasets (3.3M cells, 4 classes) achieves overall F1=0.610 but reveals tissue-identity learning:

**Table 9. Per-tissue performance of universal 4-class model.**

| Tissue | Endo F1 | Epi F1 | Immune F1 | Stromal F1 | Dominant Prediction |
|--------|---------|--------|-----------|------------|-------------------|
| Overall | 0.54 | 0.55 | **0.83** | 0.52 | -- |
| Breast rep1 | 0.01 | 0.83 | 0.49 | 0.04 | Epithelial |
| Breast rep2 | 0.00 | 0.80 | 0.50 | 0.06 | Epithelial |
| Colon cancer | 0.71 | 0.03 | 0.45 | 0.06 | Endothelial |
| Liver normal | 0.02 | 0.01 | **0.95** | 0.20 | Immune |
| Tonsil | -- | 0.00 | **1.00** | -- | Immune |

Immune is the only universally reliable class (F1=0.83), consistent with the cross-tissue generalization literature showing lymphocyte nuclear morphology is most conserved across tissues. Additionally, 37.5% of all 7.5M cells across tissues were classified as "Unknown" due to CL mapper gaps from inappropriate CellTypist model selection (7/13 tissues used generic Immune_All_High.pkl instead of tissue-specific models).

**Cross-modal validation.** DAPI-CellTypist agreement: 6.5%. Leiden ARI: 0.004. These near-zero cross-modal agreement scores indicate that DAPI morphology and gene expression capture largely orthogonal information at fine granularity.

**STHELAR DAPI training (first-ever).** We performed the first DAPI-only cell-type classification on STHELAR data, ultimately at full multi-tissue scale: 1,296,694 patches across 31 slides spanning 16 human tissues, mapped to 9 Cell Ontology classes (epithelial, T cell, fibroblast, endothelial, B cell, macrophage, pericyte, mast cell, adipocyte). All prior STHELAR work used H&E images (CellViT Dice=0.883, F1=0.855); ours is the first DAPI-only result and the first multi-tissue characterization at this scale.

The baseline EfficientNetV2-S model trained for 21 epochs (early-stopped, best at epoch 6) achieves test accuracy 0.755, weighted F1 0.764, macro F1 0.522 on a stratified 70/15/15 split (194,505 test patches). Per-class F1 ranges from 0.925 (epithelial cell, n=88,495) to 0.000 (adipocyte, n=18 test patches) and 0.196 (mast cell). The information-theoretic ceiling suggested by [§3.4 GradCAM analysis] is reproduced here: classes separable by nucleus shape (epithelial F1=0.925, endothelial 0.675, T cell 0.673, fibroblast 0.642) cluster well, while classes whose discriminative signal is cytoplasmic (mast cells, adipocytes) or surface-marker-defined (T-vs-B distinction) plateau below F1=0.25.

**Five-experiment ablation.** We tested five targeted modifications against the baseline at fixed 10-epoch budget, all on the same 1.3M-patch LMDB (Table 11b). Only one — dropping the 2 minority classes (Exp 5) — improved macro F1, and that gain is partly bookkeeping (averaging over 7 classes instead of 9). Heavy augmentation, hierarchical auxiliary head, and DINOv2/ViT-S backbone all regressed below the baseline at this training budget; the heavy-augmentation result (test F1=0.467) likely reflects under-convergence (final train_acc=0.516 vs baseline's 0.762) and uniquely lifted adipocyte from F1=0 to 0.207, suggesting it would benefit from extended training.

**Table 11b. STHELAR multi-tissue 5-experiment ablation (10 epochs each, same architecture except where noted).**

| Experiment | Test acc | Test macro F1 | Δ vs baseline |
|---|---:|---:|---:|
| Baseline (EfficientNetV2-S, 9-class) | 0.755 | 0.522 | — |
| Exp 5 — drop adipocyte+mast (7-class) | 0.770 | 0.647 | +0.125 |
| Exp 1 — hierarchical-lite (aux coarse head) | 0.725 | 0.477 | −0.045 |
| Exp 2 — heavy augmentation (CLAHE, GaussNoise, elastic) | 0.624 | 0.467 | −0.055 |
| Exp 4 — ViT-S / DINOv2 backbone | 0.628 | 0.384 | −0.138 |

DINOv2 ViT-S underperforms ImageNet-pretrained EfficientNetV2-S by ΔF1=0.138 on every cell type and every tissue, confirming that for small-patch (128×128) single-channel microscopy, ImageNet-pretrained CNNs remain superior to vision-transformer backbones.

**Cross-tissue generalisation: leave-one-tissue-out across all 16 tissues.** The strongest finding from STHELAR concerns cross-tissue generalisation. Because per-tissue accuracy in multi-tissue models can be inflated by the model learning *tissue identity from global DAPI texture* rather than per-cell morphology, we systematically performed leave-one-tissue-out (LOTO) training: 16 separate runs, each excluding one tissue from training entirely and evaluating on that tissue's held-out patches.

**Table 11c. LOTO accuracy collapse across all 16 STHELAR tissues, sorted by collapse magnitude.** "Baseline acc" is the per-tissue accuracy of the multi-tissue baseline (which had the tissue in training); "LOTO acc" is the corresponding accuracy when that tissue is held out.

| Tissue | n_test | Baseline acc | LOTO acc | Δ acc | Collapse ratio |
|---|---:|---:|---:|---:|---:|
| brain | 39,989 | 0.956 | 0.200 | 0.757 | **4.79×** |
| bone_marrow | 54,534 | 0.516 | 0.163 | 0.353 | **3.16×** |
| heart | 8,437 | 0.536 | 0.289 | 0.247 | 1.85× |
| bone | 2,132 | 0.460 | 0.254 | 0.207 | 1.81× |
| kidney | 80,410 | 0.865 | 0.511 | 0.354 | 1.69× |
| cervix | 49,993 | 0.846 | 0.507 | 0.339 | 1.67× |
| breast | 199,998 | 0.694 | 0.440 | 0.254 | 1.58× |
| liver | 100,000 | 0.825 | 0.537 | 0.288 | 1.54× |
| lymph_node | 50,000 | 0.553 | 0.376 | 0.177 | 1.47× |
| colon | 99,946 | 0.854 | 0.597 | 0.257 | 1.43× |
| skin | 111,323 | 0.780 | 0.545 | 0.235 | 1.43× |
| pancreatic | 149,976 | 0.778 | 0.566 | 0.212 | 1.37× |
| lung | 99,986 | 0.651 | 0.491 | 0.160 | 1.33× |
| tonsil | 99,986 | 0.603 | 0.463 | 0.140 | 1.30× |
| prostate | 49,984 | 0.874 | 0.670 | 0.203 | 1.30× |
| ovary | 100,000 | 0.894 | 0.734 | 0.160 | 1.22× |
| **mean** | | **0.722** | **0.434** | **0.288** | **1.78×** |

**Every one of 16 tissues exhibits measurable tissue-shortcutting** when held out, with mean accuracy collapse of 1.78× (range 1.22×–4.79×). The pattern is consistent with shortcut-learning failures observed in other domain-generalisation settings (50, 52): the model exploits whichever feature minimises training loss, and global DAPI texture is a more learnable correlate of cell type than per-cell morphology when the per-tissue cell-type composition is non-uniform. The two extreme cases — brain (4.79×) and bone marrow (3.16×) — are tissues with the most distinctive global DAPI texture (neuronal/glial parenchyma; densely packed haematopoietic cells) and most uniform cell-type composition (brain is ~95% endothelial in our patches). Eight tissues collapse 1.3×–1.5×, mostly heterogeneous epithelial/lymphoid tissues whose cells morphologically resemble cells from other tissues.

The most striking single result is brain: the baseline classifier achieves **0.956 accuracy on brain** when brain is in training, but only **0.200 accuracy when brain is held out**. The endothelial recall on brain drops from approximately 1.0 to 0.168 (precision remains 0.989). This empirically demonstrates that the model learned "looks-like-brain → predict endothelial" as a shortcut rather than learning brain endothelial morphology. The pattern generalises across tissues: in-distribution per-tissue accuracy substantially over-estimates true cell-classification capability when the test tissue is in the training distribution.

We propose that **leave-one-tissue-out evaluation should be a required reporting standard** for any multi-tissue DAPI classifier. The mean LOTO accuracy collapse ratio is a single number that captures cross-tissue generalisation in a way the in-distribution per-tissue table cannot. For DAPIDL with 16 tissues this number is 1.78×; we expect comparable or larger collapse ratios in published multi-tissue models that have not been LOTO-validated.

### 3.6 H&E Benchmark and Multimodal Fusion (STHELAR)

To bound what is achievable from image-only single-modality cell typing — and to identify which classes DAPI alone is fundamentally limited on — we trained matched classifiers on three input variants over the same STHELAR HE-intersection (cells where both DAPI and co-registered H&E patches exist after inverting the per-slide SpatialData affine transforms). All three runs share an identical EfficientNetV2-S backbone (ImageNet pretrained), weighted-CE loss with 10× capped class weights and label smoothing 0.1, AdamW with cosine warm restarts (T_0=10), 21-epoch budget with patience 8, and a 70/15/15 stratified split (seed=42). Only the input adapter differs: DAPI uses a zero-parameter 1→3 channel replication preserving the ImageNet stem; H&E uses the native 3-channel stem; multimodal concatenates DAPI and RGB into a 4-channel tensor and passes it through a learned 1×1 conv (4→3, init: identity-on-RGB + 0.1× DAPI mixing). Splits: train 885,117, val 189,668, test 189,668.

**Headline. DAPI captures most of the H&E single-modality macro F1, but the gap is class-specific.**

**Table 11d. Within-STHELAR three-way modality benchmark (1.26 M patches, 9 classes, 16 tissues, single seed).**

| Modality | accuracy | macro F1 | weighted F1 | best epoch | wall-clock |
|---|---:|---:|---:|---:|---:|
| DAPI alone | 0.715 | 0.478 | 0.717 | 9 | 5h 07m |
| H&E alone | **0.758** | 0.524 | **0.757** | 5 | 4h 00m |
| DAPI + H&E (4-ch, 1×1 conv) | 0.749 | **0.530** | 0.749 | 10 | 5h 51m |

H&E alone improves macro F1 by +0.046 over DAPI alone (+9.6% relative); naive multimodal fusion adds only +0.006 macro F1 over H&E (effectively within seed-noise; we did not bootstrap). H&E reaches its best validation F1 in epoch 5; DAPI takes nine. Acquisition cost, channel count and sample-prep flexibility favour DAPI; per-class informativeness favours H&E for stromal and lymphocyte subtypes (see below).

**Per-class complementarity.** The two modalities are not redundant — they are *complementary*, but the naive 4-channel adapter is unable to combine them losslessly (Table 11e).

**Table 11e. Per-class macro F1 across modalities.**

| Class | n_test | DAPI | H&E | DAPI+H&E | Δ(H&E − DAPI) | Best modality |
|---|---:|---:|---:|---:|---:|---|
| epithelial cell | 86,035 | **0.894** | 0.470 | **0.916** | −0.425 | DAPI / multimodal |
| T cell | 35,346 | 0.658 | 0.688 | 0.672 | +0.030 | H&E |
| fibroblast | 21,505 | 0.555 | 0.654 | 0.608 | +0.100 | H&E |
| endothelial cell | 19,061 | 0.601 | 0.514 | **0.673** | −0.087 | multimodal |
| B cell | 14,869 | 0.457 | 0.541 | 0.501 | +0.084 | H&E |
| macrophage | 9,727 | 0.486 | **0.619** | 0.518 | +0.133 | H&E |
| pericyte | 1,731 | 0.413 | **0.929** | 0.523 | +0.516 | H&E |
| mast cell | 1,375 | 0.129 | 0.165 | 0.135 | +0.037 | H&E |
| adipocyte | 19 | 0.111 | 0.136 | **0.227** | +0.025 | multimodal (n.b. n=19) |

DAPI is uniquely the best modality only for the most abundant class (epithelial cell, 86 k test patches, F1=0.894), where nuclear shape — large, round, distinctive chromatin — is highly discriminative. H&E is the best modality on six of nine classes, with the largest gap on pericyte (+0.516 F1) — a class where the discriminative signal is the cell's elongated cytoplasmic relationship to vessel walls, not nuclear morphology. The naive multimodal model wins on three classes (epithelial, endothelial, adipocyte) but underperforms H&E alone on six (pericyte −0.406, macrophage −0.101, fibroblast −0.046, B cell −0.040, mast cell −0.030, T cell −0.016). The pericyte case is the clearest signal of architectural failure: H&E alone reaches F1=0.929, but adding the DAPI channel through a 1×1 conv collapses it to 0.523. Early fusion at the input layer cannot preserve modality-specific feature subspaces.

**Cross-tissue generalisation also under H&E.** We repeated the 16-tissue leave-one-tissue-out sweep on H&E (6 epochs per holdout) to test whether shortcutting is a DAPI-specific artefact (Table 11f). It is not.

**Table 11f. LOTO accuracy under DAPI vs H&E across 16 STHELAR tissues, sorted by H&E LOTO accuracy.**

| Tissue | Baseline acc | DAPI LOTO acc | H&E LOTO acc | Δ(H&E − DAPI) LOTO |
|---|---:|---:|---:|---:|
| ovary | 0.894 | 0.734 | 0.816 | +0.082 |
| colon | 0.854 | 0.597 | 0.765 | +0.168 |
| prostate | 0.874 | 0.670 | 0.688 | +0.018 |
| kidney | 0.865 | 0.511 | 0.670 | +0.159 |
| pancreatic | 0.778 | 0.566 | 0.619 | +0.053 |
| cervix | 0.846 | 0.507 | 0.587 | +0.080 |
| liver | 0.825 | 0.537 | 0.559 | +0.022 |
| skin | 0.780 | 0.545 | 0.534 | −0.011 |
| lung | 0.651 | 0.491 | 0.487 | −0.004 |
| tonsil | 0.603 | 0.463 | 0.442 | −0.021 |
| breast | 0.694 | 0.440 | 0.425 | −0.015 |
| lymph_node | 0.553 | 0.376 | 0.334 | −0.042 |
| bone_marrow | 0.516 | 0.163 | 0.310 | +0.147 |
| bone | 0.460 | 0.254 | 0.301 | +0.047 |
| brain | 0.956 | 0.200 | 0.288 | +0.088 |
| heart | 0.536 | 0.289 | 0.269 | −0.020 |
| **mean** | **0.730** | **0.459** | **0.506** | **+0.047** |

H&E generalises slightly better than DAPI (+0.047 mean LOTO acc, better in 10/16 tissues), but the failure mode is identical: large in-distribution-vs-LOTO collapses on tissues with distinctive global texture (brain 0.956→0.288 under H&E, vs 0.956→0.200 under DAPI), and uniformly low LOTO accuracy on lymphoid/marrow tissues. Tissue shortcutting is therefore a property of multi-tissue training on tissue-imbalanced cell-type compositions, not of the DAPI channel specifically.

**Interpretation.** The H&E benchmark establishes three things relevant to DAPI-only research. (i) The image-only single-modality reference for STHELAR multi-tissue 9-class classification is macro F1 ≈ 0.524; DAPI alone reaches 0.478, i.e. about nine-tenths of that reference (with the caveat that this is a ratio of macro F1 values, not a clean biological decomposition). (ii) DAPI's class-specific blind spots are predictable from biology — they coincide with classes whose discriminative features are cytoplasmic, surface-marker-defined or organisation-defined (pericytes, macrophages, lymphocyte subtypes), exactly the classes where H&E carries extra signal. (iii) Naive early-fusion multimodal (4-channel concat + 1×1 conv) is *not* a free upgrade: it adds 0.006 macro F1 over H&E in aggregate but erases H&E's strongest per-class signals. This is consistent with reports that early-fusion multimodal classifiers under-perform their best uni-modal counterpart unless the optimisation rate of each modality is balanced (51); separate-stem late fusion and cross-attention are the standard mitigations. We report this as an architectural negative result rather than as a biological one.

**External context (not directly comparable).** Expression-based methods that use full transcript counts and spatial graphs reach far higher F1 on related tasks — e.g., HEIST (28) achieves F1=0.913 on Xenium breast 17-class (a *different* dataset, panel and class set). We list this only as orientation: molecularly informed methods substantially outperform any image-only modality, but a directly matched comparison to DAPIDL on STHELAR-9-class is not available and we do not claim "molecular ceiling" without a matched task.

### 3.7 Patch Size Impact

**Table 10. Effect of patch size on DAPI classification.**

| Patch (px) | Field (um, Xenium) | Field (um, MERSCOPE) | Val F1 (Xenium fine) | Val F1 (MERSCOPE immune) |
|------------|-------------------|---------------------|---------------------|-------------------------|
| 32 | 6.8 | 3.5 | -- | -- |
| 64 | 13.6 | 6.9 | 0.086 | -- |
| 128 | 27.2 | 13.8 | 0.413 | 0.097 |
| 160 | -- | 17.3 | -- | 0.097 |
| 252 | -- | 27.2 | -- | 0.106 |
| 256 | 54.4 | -- | **0.472** | -- |
| 350 | -- | 37.8 | -- | 0.116 |

Larger patches consistently help: 256px achieves +14% F1 over 128px on Xenium. At 64px, the model fails (F1=0.086) because single-nucleus patches lack perinuclear context. On MERSCOPE, even 350px (37.8 um) achieves only F1=0.116 for 10-class immune subtyping -- confirming that DAPI cannot distinguish immune subtypes regardless of patch size.

### 3.8 Segmentation Benchmarking

**Table 11. Xenium segmentation comparison (5 FOV types).**

| Method | Dense Rec. | Sparse Rec. | Mixed Rec. | Edge Rec. | Immune Rec. |
|--------|-----------|-------------|------------|-----------|-------------|
| Cellpose aggressive | **0.932** | **0.556** | **0.824** | **0.909** | **0.565** |
| Cellpose default | 0.824 | 0.422 | 0.725 | 0.825 | 0.418 |
| Adaptive consensus | 0.824 | 0.467 | 0.725 | 0.825 | 0.418 |
| StarDist | 0.814 | 0.467 | 0.724 | 0.793 | 0.394 |

The adaptive dispatcher correctly identifies that Xenium tissues do not benefit from ensemble segmentation (StarDist/Cellpose density ratio < 2x). On MERSCOPE dense tissue, topological consensus provides 5.2% recovery improvement (0.604 vs 0.574).

### 3.9 Negative Results

1. **Heavy augmentation degraded performance** (F1=0.33 vs 0.60 baseline). Standard augmentation (flip, rotate, brightness) is sufficient; aggressive perturbation destroys discriminative nuclear features.

2. **Dual-scale architecture (Nu-Class) failed.** DINOv2-S with global context (1024px patches): gate weight g=0.013, the model learns to ignore global DAPI context entirely. Endothelial F1=0.002. DAPI lacks tissue architecture information present in H&E.

3. **MERSCOPE immune fine-grained classification failed.** F1=0.097-0.116 across all patch sizes and 2 datasets (melanoma 178K cells, breast 505K cells). Near-random for 10 classes. DAPI cannot distinguish immune subtypes.

4. **Pathology foundation models underperform ImageNet (frozen).** Phikon-v2 (ViT-L, H&E-pretrained, frozen) F1=0.41 vs EfficientNetV2-S (ImageNet) F1=0.60. However, with LoRA fine-tuning, fluorescence-domain Cell-DINO (F1=0.463) outperforms supervised EfficientNet (F1=0.446) on the same STHELAR dataset -- the frozen-vs-LoRA distinction is critical.

5. **NuSPIRe failure.** The most domain-relevant foundation model (15.5M DAPI nuclear crops, 15 organs) achieves only F1=0.339. MAE pretraining learns reconstruction, not classification boundaries.

6. **Fine-grained classification (>10 types) fails universally.** popV fine-grained on 22 CL types: F1=0.045. scVI fine-grained 17 types: F1=0.028. Medium granularity (~7 types) is the practical ceiling.

7. **Cross-platform transfer without normalization: catastrophic failure.** Xenium-to-MERSCOPE: 2.99% accuracy (mode collapse to single class). Adaptive normalization recovers to F1=0.548.

8. **Naive 4-channel multimodal fusion is not a free upgrade.** Concatenating DAPI and RGB H&E into a 4-channel input passed through a 1×1 conv (4→3, identity-on-RGB initialisation) adds only +0.006 macro F1 over H&E alone (0.530 vs 0.524) and *erases* H&E's strongest per-class signal — pericyte F1 collapses from 0.929 to 0.523. Six of nine classes underperform H&E alone. The result is an architectural negative result, not evidence that multimodal information is unhelpful: separate-stem or cross-attention fusion is required to preserve modality-specific feature subspaces (§4.7).

---

## 4. Discussion

### 4.1 What DAPI Captures, Benchmarked Against H&E

Our systematic evaluation consistently converges on macro F1≈0.60 for coarse (3-4 class) DAPI prediction on Xenium and MERSCOPE, and macro F1=0.478 on the harder 9-class STHELAR multi-tissue task. Two questions follow: how much of the achievable image-only single-modality signal does DAPI deliver, and where does it systematically fall short?

The within-STHELAR benchmark (§3.6) provides the cleanest answer to the first question. Under matched architecture, splits and hyperparameters, H&E alone reaches macro F1=0.524 — 0.046 above DAPI, or approximately 91% of the H&E F1 captured by DAPI. We frame this as a *single-modality reference*, not as a "ceiling": macro F1 is a class-balanced average that mixes very different per-class behaviours together, and the +0.046 aggregate gap is therefore best read alongside the per-class table. Our reading is that DAPI captures the bulk of what is achievable from image-only single-modality inputs *on average*, but with class-specific blind spots that are predictable from biology.

The per-class story (Table 11e) makes those blind spots concrete. DAPI is the best modality on the most numerous class — epithelial cells (F1=0.894) — where nuclear shape is highly discriminative. H&E is the best modality on six of nine classes, with the largest H&E−DAPI gap on pericytes (+0.516 F1, attributable to elongated cytoplasmic alignment with vessel walls), macrophages (+0.133, kidney-bean morphology compounded by abundant cytoplasm and tissue context), fibroblasts (+0.100, mostly cytoplasmic) and B cells (+0.084, organisational context). The general pattern: DAPI excels where nuclear shape is discriminative; H&E excels where the discriminative signal is cytoplasmic, surface-marker-defined or organisation-defined. None of this is fundamentally about gene expression — the shortfall reflects information that is geometrically present in the cytoplasm or stroma and therefore visible to brightfield H&E but absent from a single-channel nuclear stain.

GradCAM analysis on the Xenium-trained models corroborates the asymmetry: DAPI attention focuses sharply on kidney-shaped macrophage nuclei (correctly classified ~100% of the time) and spindle-shaped myoepithelial cells (~90%), but produces diffuse, low-confidence attention on lymphocyte subtypes whose identity is determined by surface markers (CD3, CD4, CD8, CD19) invisible to any nuclear stain. This is a real information-theoretic boundary — but it is a boundary on *single-channel-nuclear* images, not on imaging in general. Comparable work supports the same picture: Woloshuk et al. report ~60% balanced accuracy for 2D DAPI improving to 80.3% with 3D nuclear imaging, and CelloType achieves mAP=0.40 on DAPI-only (90% of its DAPI+transcript multimodal performance) — i.e. independent groups converge on a single-channel DAPI single-modality bound at the same order of magnitude.

We therefore avoid a quantitative "DAPI vs molecular" ceiling claim across datasets. Expression-based methods such as HEIST achieve much higher F1 (0.913 on Xenium breast 17-class) but on different data, panel and class structure; that gap is large but is not a matched comparison and we report it only as external context (§3.6).

### 4.2 BANKSY: Spatial Context Before Annotation

Our most impactful annotation finding is that BANKSY spatial clustering followed by marker-based cluster labeling (F1=0.802) dramatically outperforms cell-level consensus methods (popV F1=0.459). This BANKSY-first architecture represents an unpublished combination: BANKSY was designed for tissue domain segmentation, not cell type annotation. By augmenting each cell's expression with neighborhood context before clustering, BANKSY captures spatial organization that purely expression-based methods miss -- critically enabling endothelial detection (F1=0.844 vs F1=0.000 without spatial context).

### 4.3 DINO vs MAE: Pretraining Objective Matters More Than Domain

The foundation model comparison reveals a counter-intuitive hierarchy: Cell-DINO (DINO, fluorescence, F1=0.463) > DINOv2 (DINO, natural images, F1=0.453) >> OpenPhenom (MAE, fluorescence, F1=0.393) > NuSPIRe (MAE, DAPI, F1=0.339). The pretraining objective (DINO vs MAE) accounts for more variance than domain (fluorescence vs natural images). DINO produces instance-discriminative features optimized for classification, while MAE learns reconstruction features that detect perturbations but not category boundaries.

Furthermore, foundation models exhibit superior cross-domain generalization: DINOv2 frozen drops only 13% when transferring STHELAR-to-Xenium (F1=0.336), while supervised EfficientNet drops 45% (F1=0.245). The in-domain vs cross-domain ranking flips, suggesting that foundation models learn more transferable nuclear representations.

### 4.4 Annotation Quality as the Primary Bottleneck

Annotation quality dominates model architecture: the same EfficientNetV2-S backbone achieves F1=0.33 (heavy augmentation, poor labels) to F1=0.60 (curriculum learning, filtered labels). The GT-free confidence filtering system provides +10-16% F1 improvement without any ground truth, using orthogonal signals (marker enrichment, spatial coherence, cross-method consensus, proportion plausibility). This is particularly valuable because ground truth labels are computationally derived, introducing circular evaluation risk.

### 4.5 Practical Value

Despite the morphology ceiling, DAPIDL offers practical value: (1) Rapid tissue screening at <$1/slide versus >$1000 for spatial transcriptomics; (2) Retrospective analysis of existing DAPI-stained archives; (3) Quality control flagging regions of interest for targeted spatial transcriptomics; (4) Zero-cost post-hoc improvement via spatial neighborhood averaging (Kalinin 2018: aggregating softmax outputs from cells within 50um radius improved classification from 63% to 98% in a comparable setting).

### 4.6 Limitations

1. All DAPI data is 2D; 3D DAPI achieves 80.3% (Woloshuk et al.)
2. Ground truth labels are computationally derived (circular evaluation risk)
3. **Multi-tissue models exhibit universal tissue-shortcutting in both image modalities.** Across 16 STHELAR tissues, leave-one-tissue-out evaluation reveals mean accuracy collapse of 1.78× for DAPI (range 1.22×–4.79×, brain 0.956→0.200) and an essentially identical pattern for H&E (mean LOTO acc 0.506 vs DAPI 0.459). In-distribution per-tissue accuracy therefore over-estimates true cell-classification capability by ~1.8× regardless of imaging modality. Any reported per-tissue accuracy in a multi-tissue model should be paired with a LOTO accuracy on the same tissue (§3.5–3.6).
4. **Single-seed modality benchmarks.** The H&E and multimodal results in §3.6 are single-seed, with no bootstrapped per-tissue confidence intervals. Aggregate gaps of +0.046 macro F1 (H&E−DAPI) are well outside seed-noise on this scale of dataset, but the +0.006 multimodal-over-H&E delta is small enough to be plausibly within seed-noise; we report it as such and do not claim multimodal beats H&E.
5. 37.5% "Unknown" cells in multi-tissue training from inappropriate model selection
6. Endothelial annotation requires spatial methods (BANKSY), unavailable for all tissues
7. MERSCOPE images (~20 GB) require memory-mapped access
8. DAPI-CellTypist cross-modal agreement is only 6.5% at fine granularity

### 4.7 Future Directions

1. **Proper multimodal fusion as a benchmark upper-bound.** Naive 4-channel concat-and-1×1-conv adds only +0.006 macro F1 over H&E and erases per-class strengths (pericyte 0.929→0.523). Separate-stem late fusion, cross-attention between DAPI and H&E feature maps, or learned channel-mixing with a residual back to the H&E channels are obvious next steps; we treat these as a way to *bound* the multimodal benchmark above DAPI, not as a competitor to DAPI-only.
2. **LOTO as a standard evaluation protocol.** Based on the universal 1.3×–4.8× LOTO collapse observed here under both DAPI and H&E, we recommend that any multi-tissue image-based cell classification paper report at least one held-out-tissue accuracy alongside any per-tissue accuracy claim. Where compute allows, the mean LOTO collapse ratio across all training tissues is the most informative single number for cross-tissue generalisation.
3. **Tissue-conditioning ablations.** Training with an explicit tissue token vs. without — and then comparing LOTO collapse — would directly quantify how much of the shortcutting is the model's representation choice vs. how much is inherent to multi-tissue training.
4. **Multi-seed, bootstrapped confidence intervals.** The within-STHELAR benchmark is currently single-seed. Multi-seed runs and per-tissue bootstraps would let us put confidence intervals on H&E−DAPI and multimodal−H&E deltas, and on per-tissue LOTO collapse ratios.
5. **Multi-channel DAPI-adjacent inputs (still nuclear-only).** Xenium provides a boundary stain (E-Cad/CD45) and an interior stain (18S rRNA) alongside DAPI. A 3-channel nuclear-region input is the closest extension to DAPI-only that does *not* require an H&E pipeline and could close part of the H&E-vs-DAPI gap on stromal classes while remaining single-modality.
6. **Spatial neighbourhood averaging.** Zero-cost post-hoc improvement: aggregating softmax outputs from cells within a fixed radius (e.g. 50 µm) is reported to recover substantial F1 in comparable settings without retraining.
7. **Test-time adaptation (FUSION).** Fully unsupervised batch normalisation fusion for platform transfer.
8. **CelloType integration.** Using DAPI+transcript labels instead of CellTypist-only for better ground truth.
9. **Contrastive alignment (H&Enium).** Learning shared H&E + Xenium embedding spaces.
10. **3D DAPI.** Leveraging MERSCOPE z-stacks (7 z-planes per dataset).

---

## 5. Data and Code Availability

DAPIDL is implemented in Python using PyTorch with NVIDIA DALI for GPU data loading. The pipeline supports local execution and distributed processing via ClearML. Source code: project repository. Trained models and LMDB datasets: S3 (`s3://dapidl/`). STHELAR data: BioStudies S-BIAD2146. MERSCOPE FFPE data: Google Cloud (`gs://vz-ffpe-showcase/`).

---

## References

1. Stahl PL, et al. (2016). Visualization and analysis of gene expression in tissue sections by spatial transcriptomics. *Science*, 353(6294), 78-82.
2. Chen KH, et al. (2015). Spatially resolved, highly multiplexed RNA profiling in single cells. *Science*, 348(6233), aaa6090.
3. Janesick A, et al. (2023). High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis. *Nat Commun*, 14(1), 8353.
4. Marco Salas S, et al. (2025). Optimizing Xenium In Situ data utility by quality assessment and best-practice analysis workflows. *Nat Methods*.
5. Erickson A, et al. (2022). Spatially resolved clonal copy number alterations in benign and malignant tissue. *Nature*, 608(7922), 360-367.
6. Abdelaal T, et al. (2019). A comparison of automatic cell identification methods for single-cell RNA sequencing data. *Genome Biol*, 20(1), 194.
7. Abel JF, et al. (2023). Deep-learning quantified cell-type-specific nuclear morphology predicts genomic instability. *bioRxiv*.
8. Mohammad S, et al. (2024). Deep Learning Powered Identification of Differentiated Early Mesoderm Cells from Pluripotent Stem Cells. *Cells*, 13(6), 534.
9. Dominguez Conde C, et al. (2022). Cross-tissue immune cell analysis reveals tissue-specific features in humans. *Science*, 376(6594), eabl5197. [CellTypist]
10. Ianevski A, et al. (2022). Fully-automated and ultra-fast cell-type identification using specific marker combinations. *Nat Commun*, 13(1), 1246. [scType]
11. Aran D, et al. (2019). Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage. *Nat Immunol*, 20(2), 163-172. [SingleR]
12. Hao Y, et al. (2021). Integrated analysis of multimodal single-cell data. *Cell*, 184(13), 3573-3587. [Seurat v4/Azimuth]
13. Ergen C, et al. (2023). Consensus prediction of cell type labels with popV. *bioRxiv*.
14. Schmidt U, et al. (2018). Cell Detection with Star-Convex Polygons. *MICCAI 2018*, pp. 265-273. [StarDist]
15. Stringer C, et al. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nat Methods*, 18(1), 100-106.
16. Horst F, et al. (2024). CellViT -- Cell Segmentation and Classification in Pathology Images Using Vision Transformers. *Med Image Anal*, 94, 103143.
17. Petukhov V, et al. (2022). Cell segmentation in imaging-based spatial transcriptomics. *Nat Biotechnol*, 40(3), 345-354. [Baysor]
18. Jones DC, et al. (2024). Cell Simulation as Cell Segmentation. *bioRxiv*. [Proseg]
19. Tan M, Le QV. (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML 2021*.
20. Liu Z, et al. (2022). A ConvNet for the 2020s. *CVPR 2022*. [ConvNeXt]
21. Wightman R. (2019). PyTorch Image Models (timm). GitHub.
22. Lin T-Y, et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.
23. Singh J, et al. (2023). Batch-balanced focal loss: a hybrid solution to class imbalance. *J Med Imaging*, 10(5), 051809.
24. Masci AM, et al. (2009). An improved ontological representation of dendritic cells as a paradigm for all cell types. *BMC Bioinformatics*, 10, 70. [Cell Ontology]
25. Singhal V, et al. (2024). BANKSY unifies cell typing and tissue domain segmentation. *Nat Genet*, 56(3), 431-441.
26. Brbic M, et al. (2022). Annotation of spatially resolved single-cell data with STELLAR. *Nat Methods*, 19(11), 1411-1418.
27. Giraud-Sauveur F, et al. (2025). STHELAR, a multi-tissue dataset linking spatial transcriptomics and histology. *bioRxiv*.
28. Madhu H, et al. (2025). HEIST: A Graph Foundation Model for Spatial Transcriptomics. *arXiv*.
29. Wang CX, et al. (2025). scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics. *bioRxiv*.
30. De Donno C, et al. (2023). Population-level integration of single-cell datasets enables multi-scale analysis. *Nat Methods*, 20(11), 1683-1692.
31. Russell AJC, et al. (2024). Slide-tags enables single-nucleus barcoding for multimodal spatial genomics. *Nature*, 625(7993), 101-109.
32. Chen R, et al. (2025). A comprehensive benchmarking for spatially resolved transcriptomics clustering methods. *iMeta*, 4(6).
33. Yu K-H, et al. (2020). Classifying non-small cell lung cancer types and transcriptomic subtypes using convolutional neural networks. *JAMIA*, 27(5), 757-769.
34. Moffitt JR, et al. (2018). Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region. *Science*, 362(6416), eaau5324. [MERFISH]
35. Wolf FA, et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biol*, 19, 15.
36. Diehl AD, et al. (2016). The Cell Ontology 2016: enhanced content, modularization, and ontology interoperability. *J Biomed Semantics*, 7, 44.
37. He K, et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. [ResNet]
38. McInnes L, et al. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
39. Biancalani T, et al. (2021). Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram. *Nat Methods*, 18(11), 1352-1362.
40. Moses L, Bhatt S. (2022). Museum of spatial transcriptomics. *Nat Methods*, 19, 534-546.
41. Woloshuk A, et al. (2021). In situ classification of cell types in human kidney tissue using 3-D nuclear staining. *Cytometry Part A*, 99(7), 707-721. [NephNet3D]
42. CelloType. (2024). DAPI-only cell typing in spatial transcriptomics. *Nature Methods*.
43. Pfaendler R, et al. (2023). scDINO: self-supervised learning for single-cell image analysis. *bioRxiv*.
44. Glettig M, et al. (2025). H&Enium: contrastive alignment of H&E and Xenium. *bioRxiv*.
45. Kalinin AA, et al. (2018). 3D cell nuclear morphology: analysis and classification. *PLoS Comput Biol*.
46. CASSIA. (2025). Context-aware spatial cell type annotation. *Nat Commun*.
47. TRACERx-PHLEX. (2024). Spatial transcriptomics analysis framework. *Nat Commun*.
48. Xu J, et al. (2025). Nu-Class: dual-scale nuclear classification. *bioRxiv*.
49. Xu J, et al. (2025). TAND: tissue-aware nuclear decomposition. *bioRxiv*.
50. Geirhos R, Jacobsen J-H, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. (2020). Shortcut Learning in Deep Neural Networks. *Nat Mach Intell*, 2(11), 665–673.
51. Wang W, Tran D, Feiszli M. (2020). What Makes Training Multi-Modal Classification Networks Hard? *CVPR 2020*.
52. Beery S, Van Horn G, Perona P. (2018). Recognition in Terra Incognita. *ECCV 2018*.

---

## Supplementary Information

### Figure Descriptions

**Figure 1.** DAPIDL pipeline overview. (A) Data flow from raw spatial transcriptomics (Xenium/MERSCOPE/STHELAR) through ensemble annotation, CL standardization, GT-free filtering, LMDB creation, to CNN training. (B) Platform-specific data reader architecture. (C) Deployed DAPI-only inference mode.

**Figure 2.** Annotation method comparison. (A) Macro F1 bar chart across 12+ methods on STHELAR breast. (B) Per-class F1 heatmap showing method complementarity (BANKSY wins spatial types, CellTypist wins expression types). (C) BANKSY-first vs cell-level consensus architecture comparison.

**Figure 3.** GT-free confidence filtering. (A) Four-signal confidence estimation schematic. (B) F1 vs retention curve at different thresholds for rep1 and rep2. (C) Per-type confidence scores comparing CellTypist vs BANKSY.

**Figure 4.** DAPI classification results. (A) Validation F1 curves across architectures and configurations. (B) Per-class F1 bar chart with morphological predictability hierarchy. (C) GradCAM heatmaps for macrophage (sharp kidney-shape focus), epithelial (nuclear border), and T cell (diffuse/uncertain).

**Figure 5.** Foundation model comparison. (A) DINO vs MAE pretraining objective comparison. (B) LoRA vs frozen performance. (C) Cross-domain generalization: foundation models generalize better despite lower in-domain F1. (D) Context radius ablation (d0/d25/nomask).

**Figure 6.** Modality gap analysis. (A) Expression-based F1 (y) vs DAPI F1 (x) per cell type. (B) HEIST (expression+spatial) vs DAPIDL (DAPI) training curves. (C) Class weight capping ablation.

**Figure 7.** Cross-platform transfer. (A) DAPI intensity distributions before/after normalization. (B) Catastrophic failure without normalization (2.99% accuracy) vs with (54.8%). (C) Physical size normalization effect.

**Figure 8.** Multi-tissue analysis. (A) Per-tissue F1 heatmap showing tissue-identity bias. (B) 37.5% Unknown cell problem. (C) Immune as the only universally reliable class.

**Figure 8b.** STHELAR multi-tissue baseline characterisation. (A) Class distribution across the 1.3M-patch 9-class LMDB (log-scale). (B) Tissue distribution across 16 tissues / 31 slides. (C) Tissue × cell-type composition heatmap showing the highly non-uniform per-tissue cell-type mixes (brain ~95% endothelial, lymph node T/B-dominant). (D) Per-class F1 with support; epithelial F1=0.925 dominates while mast cell F1=0.196 and adipocyte F1=0 sit at the DAPI-information ceiling.

**Figure 8c.** STHELAR 5-experiment ablation. Side-by-side comparison of baseline (9-class, EfficientNetV2-S, DALI) against four targeted modifications: drop-2-rare-classes (Exp 5, +0.125 macro F1), hierarchical-lite auxiliary head (Exp 1, −0.045), heavy augmentation (Exp 2, −0.055), DINOv2/ViT-S backbone (Exp 4, −0.138). Panel (A) shows overall macro/weighted F1 and accuracy; panel (B) shows per-class F1 deltas; panel (C) shows per-tissue macro F1 across all 5 experiments as a heatmap.

**Figure 8d.** Leave-one-tissue-out across all 16 STHELAR tissues. (A) Side-by-side bars of in-distribution (baseline) vs LOTO accuracy per tissue, with collapse ratios annotated. (B) Accuracy drop magnitude per tissue, color-mapped — brain (0.757 absolute drop) dominates, bone marrow second. (C) Collapse ratio chart showing brain at 4.79× and bone_marrow at 3.16× as extreme outliers, mean across tissues 1.78×. (D) Weighted F1 cross-check confirming the pattern is metric-robust.

**Figure 8e.** STHELAR within-dataset modality benchmark (DAPI vs H&E vs naive 4-channel multimodal). (A) Overall accuracy / macro F1 / weighted F1 bar chart for the three modalities; H&E reaches its best validation F1 in 5 epochs, DAPI in 9, multimodal in 10. (B) Per-class F1 grouped bar chart over the 9 STHELAR classes, with H&E−DAPI deltas annotated; epithelial dominated by DAPI/multimodal (F1=0.916), pericyte and macrophage dominated by H&E (F1=0.929 and 0.619 respectively), naive multimodal collapses pericyte to 0.523. (C) 16-tissue LOTO comparison under DAPI vs H&E (Table 11f) showing identical shortcutting failure mode under both modalities, H&E mean LOTO acc 0.506 vs DAPI 0.459. (D) Per-class winners-by-modality bar chart highlighting the asymmetry (DAPI strong: 1 class; H&E strong: 6; multimodal strong: 3 including epithelial which DAPI also wins).

**Figure 9.** Patch size ablation. (A) F1 vs patch size for Xenium and MERSCOPE. (B) Example patches at 32/64/128/256px showing context differences. (C) Physical field of view comparison across platforms.

**Figure 10.** Segmentation benchmark. (A) Per-method recovery across FOV types (dense/sparse/mixed/edge/immune). (B) StarPose vs CellViT on STHELAR. (C) Adaptive dispatcher decision tree.

### Supplementary Tables

**Table S1.** Complete inventory of 50+ DAPI training experiments with all metrics.
**Table S2.** All 298 CellTypist model consensus combinations tested.
**Table S3.** Marker database comparison across platforms (Custom, CellMarker 2.0, PanglaoDB, scTypeDB).
**Table S4.** Complete STHELAR dataset inventory (31 slides, 16 tissues, per-slide cell counts).
**Table S5.** Cross-method IoU matrices for segmentation benchmarks.
**Table S6.** popV 8-method consensus score calibration table.
**Table S7.** BANKSY parameter optimization (k, lambda, r combinations).
**Table S8.** Foundation model inventory with rationale for inclusion/exclusion.
**Table S9.** STHELAR 5-experiment ablation full per-class metrics (precision, recall, F1, support per class × experiment).
**Table S10.** STHELAR 16-tissue DAPI-LOTO complete results: per-tissue baseline acc, LOTO acc, drop, ratio, weighted F1, n_test, train/val split sizes, best epoch, best val macro F1.
**Table S11.** STHELAR three-way modality benchmark — full per-class precision / recall / F1 / support for DAPI, H&E and naive multimodal under matched architecture, splits and hyperparameters.
**Table S12.** STHELAR 16-tissue H&E-LOTO complete results paralleling S10: per-tissue baseline acc, H&E LOTO acc, weighted F1, n_train/val/test, best epoch, best val macro F1.
