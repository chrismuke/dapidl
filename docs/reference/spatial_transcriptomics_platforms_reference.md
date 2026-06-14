# Comprehensive Reference: Spatial Transcriptomics Platforms

**Last updated:** 2026-03-30
**Purpose:** Exhaustive technical reference for all spatial transcriptomics (ST) platforms

---

## Table of Contents

1. [Platform Classification Overview](#1-platform-classification-overview)
2. [Imaging-Based Platforms (Subcellular Resolution)](#2-imaging-based-platforms-subcellular-resolution)
3. [Sequencing-Based Platforms (Spot/Capture)](#3-sequencing-based-platforms-spotcapture)
4. [In-Situ Hybridization Methods (Academic)](#4-in-situ-hybridization-methods-academic)
5. [Emerging / Newer Platforms](#5-emerging--newer-platforms)
6. [ROI-Based / Digital Spatial Profiling](#6-roi-based--digital-spatial-profiling)
7. [Specialty / Unique Modality Platforms](#7-specialty--unique-modality-platforms)
8. [Cross-Platform Comparison Tables](#8-cross-platform-comparison-tables)
9. [Benchmarking Results (2024-2026)](#9-benchmarking-results-2024-2026)
10. [Platform Adoption and Usage](#10-platform-adoption-and-usage)

---

## 1. Platform Classification Overview

| Category | Approach | Resolution | Gene Coverage | Key Platforms |
|----------|----------|-----------|---------------|---------------|
| **Imaging-based** | Fluorescent probe detection in situ | Subcellular (~100-300 nm) | Targeted (100-19,000 genes) | Xenium, MERSCOPE, CosMx, Resolve MC |
| **Sequencing-based** | Capture + NGS readout | Spot-level (0.5-55 um) | Whole transcriptome | Visium, Visium HD, Stereo-seq, Slide-seq |
| **ISH (academic)** | Sequential FISH/ISS | Subcellular | Targeted (30-10,000 genes) | seqFISH+, MERFISH, STARmap, osmFISH |
| **ROI-based** | Region-of-interest profiling | ROI (10+ um) | WTA or targeted | GeoMx DSP |
| **Proximity-based** | Barcoded beads + scRNA-seq | Single-cell (~10 um) | Whole transcriptome | Slide-tags, Curio Seeker |

---

## 2. Imaging-Based Platforms (Subcellular Resolution)

### 2.1 10x Genomics Xenium

| Specification | Details |
|--------------|---------|
| **Company** | 10x Genomics (Pleasanton, CA, USA) |
| **Platform name** | Xenium In Situ / Xenium Analyzer |
| **Method type** | Imaging-based (padlock probes + rolling circle amplification) |
| **DAPI staining** | YES -- 0.2125 um/pixel native resolution; 3D Z-stack OME-TIFF; used for nucleus segmentation |
| **H&E staining** | YES -- post-assay H&E on same section; co-registered via image alignment; protocol published for integrating IHC + H&E with Xenium data |
| **Other stainings** | Cell boundary markers: ATP1A1, CD45, E-Cadherin (Multimodal Cell Segmentation Kit); 18S rRNA interior stain; protein sub-panels (customizable add-on); immunofluorescence morphology markers |
| **Gene panel size** | Pre-designed: 300-500 genes (tissue-specific panels); Custom: up to 480 genes (RNA v1); Xenium Prime: ~5,000 genes; Custom add-on: up to 100 genes on top of base panel |
| **Maximum genes** | ~5,100 (5K Prime + 100 custom add-on) |
| **Spatial resolution** | Subcellular; ~200 nm per pixel; individual transcript localization at nanometer precision |
| **Cell segmentation** | Built-in: DAPI-based nucleus segmentation (default); Multimodal Cell Segmentation algorithm (3-step: boundary + interior + nuclear expansion); also compatible with external tools (Cellpose, StarDist) |
| **Typical cell count** | 100K-500K+ cells per run; depends on tissue area (up to 472 mm^2 per run, 2 slides) |
| **Tissue area** | Up to 472 mm^2 per run (2 slides, each 6.5 x ~36 mm usable) |
| **Run time** | <3 days (480 genes) or <6 days (5K genes) |
| **Data format** | Transcript coordinates: Parquet; Cell feature matrix: H5; Morphology images: OME-TIFF (pyramidal, JPEG-2000 compressed, 16-bit grayscale); Cell polygons: Parquet; Metadata: JSON; cells.parquet, transcripts.parquet |
| **Public datasets** | [10x Genomics Datasets](https://www.10xgenomics.com/datasets) -- breast cancer (Rep1/Rep2), mouse brain, pancreas, lung, heart, ovarian cancer; [Bioconductor TENxXeniumData](https://bioconductor.org/packages/release/data/experiment/html/TENxXeniumData.html); CC BY 4.0 license |
| **Cost estimate** | Instrument: ~$350K-400K (list); Reagents: ~$3,000-7,500/slide (panel-dependent); Xenium Prime 5K: ~$12,000/slide (through core facilities); 10x offers periodic 33% reagent discounts |
| **Key limitations** | Targeted panel (not whole transcriptome for standard panels); 5K panel requires 6-day run; DAPI-only segmentation has limitations in dense tissue; gene panel design requires upfront planning; 2-slide per run constraint |

### 2.2 Vizgen MERSCOPE / MERSCOPE Ultra (MERFISH)

| Specification | Details |
|--------------|---------|
| **Company** | Vizgen (Cambridge, MA, USA) |
| **Platform name** | MERSCOPE (original) / MERSCOPE Ultra (2024+) |
| **Method type** | Imaging-based (MERFISH -- Multiplexed Error-Robust FISH; direct probe hybridization with error-correcting barcodes) |
| **DAPI staining** | YES -- ~100 nm transcript resolution; DAPI imaged automatically in every run; 7-plane Z-stack (0.7 um between Z-positions); mosaic TIFF output |
| **H&E staining** | NO native H&E -- not part of standard workflow; post-run H&E possible but not co-registered by default |
| **Other stainings** | PolyT stain (always imaged); Cell Boundary Stain Kit (3 protein markers: Cellbound1/2/3); Protein Co-Detection Kits (up to 6-plex standard, up to 9-plex custom through Lab Services); Protein Stain Verification Kit |
| **Gene panel size** | Standard MERSCOPE: 140 / 300 / 500 genes (18-bit / 21-bit / 27-bit encoding); MERSCOPE Ultra: up to 1,000 genes; MERFISH 2.0 chemistry for enhanced sensitivity |
| **Maximum genes** | ~1,000 (MERSCOPE Ultra with 27-bit encoding) |
| **Spatial resolution** | Subcellular; ~100 nm transcript detection resolution |
| **Cell segmentation** | Built-in Cellpose-based segmentation; uses DAPI + cell boundary stains; Vizgen Post-Processing Tool (VPT) for re-segmentation; compatible with external tools |
| **Typical cell count** | 100K-700K+ cells per run depending on tissue area |
| **Tissue area** | MERSCOPE: 1.25 cm^2 (FCX-S) or 3.0 cm^2 (FCX-L); MERSCOPE Ultra: up to 3.0 cm^2 per slide; ~9.0 cm^2/week throughput |
| **Run time** | ~2-3 days per run (MERSCOPE Ultra is >2x faster imaging) |
| **Data format** | Transcript positions: CSV (x,y,z); Cell metadata: CSV; Cell boundaries: Parquet (v232+) or HDF5 (older); Mosaic images: TIFF; Experiment metadata: JSON; VZG binary for Vizualizer; compatible with Parquet/CSV/HDF5 open formats |
| **Public datasets** | [Vizgen Data Release Program](https://vizgen.com/data-release-program/) -- mouse brain receptor map (483 genes, 734K cells, 554M transcripts); MERFISH 2.0 datasets (breast, colon, brain) |
| **Cost estimate** | Instrument: ~$300K-400K (list); Reagents: ~$2,000-5,000/slide (panel and encoding dependent); pricing varies by gene panel size |
| **Key limitations** | Lower sensitivity than Xenium per benchmark studies; gene panel maxes at 1,000 (no whole transcriptome); no native H&E; cell boundary stain kit is separate purchase; large DAPI mosaics can be >20 GB (memory-intensive) |

### 2.3 NanoString CosMx SMI (now Bruker Spatial Biology)

| Specification | Details |
|--------------|---------|
| **Company** | NanoString Technologies / Bruker Spatial Biology (acquired 2024) |
| **Platform name** | CosMx Spatial Molecular Imager (SMI) |
| **Method type** | Imaging-based (cyclic FISH with branch-chain hybridization amplification) |
| **DAPI staining** | YES -- 0.12028 um/pixel resolution (~120 nm/pixel); 5-channel fluorescence imaging (DAPI, FITC, TRITC, Texas Red, Cy5); used for segmentation |
| **H&E staining** | Post-run H&E compatible; not natively co-registered but can be aligned |
| **Other stainings** | Morphology markers: PanCK, CD45, CD68 (immunofluorescence, standard panel); up to 64-76 validated proteins alongside RNA (immuno-oncology panel); cell membrane staining for segmentation |
| **Gene panel size** | 100-plex RNA panel; 1,000-plex RNA panel (Human Universal Cell Characterization); 6K Discovery panel (6,175 RNA targets); Human Whole Transcriptome: ~18,935 genes (CosMx 2.0) |
| **Maximum genes** | ~18,935 genes (whole transcriptome) + up to 76 proteins simultaneously |
| **Spatial resolution** | Subcellular; ~120 nm/pixel; FOV size 0.51 x 0.51 mm; custom water immersion optics 1.1 NA, 22.78x magnification |
| **Cell segmentation** | Built-in ML-augmented algorithm using cell membrane proteins + nuclei + RNA transcript positions; CosMx 2.0 introduces best-in-class segmentation |
| **Typical cell count** | 50K-200K+ cells per run (FOV-dependent); scan area smaller than Xenium |
| **Tissue area** | FOV-based scanning; smaller coverage than Xenium per run |
| **Run time** | ~3-7 days depending on panel size and number of FOVs |
| **Data format** | Counts matrices; Cell metadata; Decoded transcripts; Morphology scans: OME-TIFF; Seurat objects (.rds); AtoMx SIP (cloud analysis platform) |
| **Public datasets** | [Bruker Spatial Biology datasets](https://brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/) -- human pancreas (WTX), liver (1K), frontal cortex (6K), mouse brain (1K), NSCLC, colon; [Zenodo archives](https://zenodo.org/records/14330691) |
| **Cost estimate** | Instrument: ~$295K; 1,000-plex RNA panel: ~$3,300/slide; 100-plex RNA panel: ~$1,850/slide; Custom panel: ~$2/target/slide; Whole transcriptome: pricing TBD (2025 launch) |
| **Key limitations** | Smaller scan area than Xenium (fewer cells per run); lower on-target fraction in benchmarks vs Xenium; CosMx 2.0 hardware needed for WTX; higher cost per cell; slower imaging throughput; NanoString bankruptcy/Bruker acquisition caused market uncertainty |

### 2.4 Resolve Biosciences Molecular Cartography

| Specification | Details |
|--------------|---------|
| **Company** | Resolve Biosciences (Monheim am Rhein, Germany) |
| **Platform name** | Molecular Cartography (MC) / MC2 (next-gen) |
| **Method type** | Imaging-based (combinatorial smFISH; single-molecule FISH with iterative decoding) |
| **DAPI staining** | YES -- DAPI images provided for each target region; TIFF format |
| **H&E staining** | YES -- post-assay H&E on same section (non-destructive assay); bridges pathology and molecular analysis |
| **Other stainings** | Additional staining images (TIFF) can be provided alongside DAPI |
| **Gene panel size** | Standard: up to 100 RNA targets; Advanced RNA assay: up to 330 RNA targets |
| **Maximum genes** | ~330 genes per tissue section |
| **Spatial resolution** | Subcellular; ~300 nm in x, y, z; single-molecule detection |
| **Cell segmentation** | External -- user-provided (Cellpose, nf-core/molkart pipeline); transcript coordinates + DAPI provided for segmentation |
| **Typical cell count** | Variable; depends on tissue section size |
| **Data format** | Transcript coordinates: text file (x,y,z,gene); DAPI images: TIFF; Additional stain images: TIFF; Data summary report |
| **Public datasets** | [Resolve public datasets](https://resolvebiosciences.com/datasets/) -- various tissue types available |
| **Cost estimate** | Service-based model (no instrument purchase); turnaround 6-8 weeks from probe synthesis; pricing on request |
| **Key limitations** | Service-only model (no self-run instrument until MC2); limited to 330 genes; 6-8 week turnaround; smaller user community; limited throughput |

### 2.5 BGI STOmics Stereo-seq

| Specification | Details |
|--------------|---------|
| **Company** | BGI Genomics / STOmics (Shenzhen, China) |
| **Platform name** | Stereo-seq (SpaTial Enhanced REsolution Omics-sequencing) |
| **Method type** | Sequencing-based with subcellular resolution (DNA nanoball barcode array + NGS on DNBSEQ) |
| **DAPI staining** | YES -- compatible with DAPI and ssDNA nuclear staining; used for cell segmentation |
| **H&E staining** | YES -- Stereo-seq H&E Solution available; H&E images used for segmentation and tissue annotation |
| **Other stainings** | ssDNA staining (standard); multiplex immunofluorescence (ST-FFPE-mIF); compatible with various histological stains |
| **Gene panel size** | Whole transcriptome (unbiased; no pre-designed probes for FF; V2 uses random primers for FFPE) |
| **Maximum genes** | Whole transcriptome (~20,000+ genes) |
| **Spatial resolution** | 500 nm spot spacing (center-to-center); effective subcellular resolution (~400 spots per 10 um cell); Stereo-seq V2 maintains subcellular resolution on FFPE |
| **Cell segmentation** | External tools recommended: Cellpose (1/3), DeepCell, MEDIAR for DAPI/ssDNA; HoverNet, StarDist, SAM for H&E; StereoCell (custom); mask import (.tif) supported |
| **Typical cell count** | Highly variable -- standard 1 cm x 1 cm chip has 400M capture spots; can capture millions of cells from large tissues |
| **Tissue area** | Standard chip: 1 cm x 1 cm; expandable up to 13 cm x 13 cm; largest FOV of any ST platform |
| **Run time** | Library prep + sequencing; ~25 working days turnaround for 4 samples through BGI service |
| **Data format** | Gene expression matrix (GEF format); spatial barcode maps; segmentation masks (.tif); SAW analysis pipeline output; compatible with Scanpy/Seurat via conversion |
| **Public datasets** | [STOmics Database](https://db.cngb.org/stomics/project/) -- mouse brain, embryo atlases, various tissues; Spatiotemporal Omics Consortium data |
| **Cost estimate** | ~$56/mm^2 (first run), ~$49/mm^2 (second run) for smaller chips; ~$3,000-5,000 per 1 cm^2 chip assay (including sequencing); lower than Western competitors |
| **Key limitations** | Requires DNBSEQ sequencing platform (BGI ecosystem); limited availability outside Asia until recently; fresh frozen samples preferred (V2 adds FFPE); complex data analysis; large data files; brand recognition lower in Western markets |

### 2.6 Rebus Biosystems Esper

| Specification | Details |
|--------------|---------|
| **Company** | Rebus Biosystems (Santa Clara, CA, USA) |
| **Platform name** | Esper |
| **Method type** | Imaging-based (cyclic smFISH + Synthetic Aperture Optics; EEL FISH compatible) |
| **DAPI staining** | YES -- fluorescence imaging included |
| **H&E staining** | Not documented in available sources |
| **Other stainings** | Standard fluorescence channels |
| **Gene panel size** | High Fidelity assay: up to 30 custom genes (smFISH); EEL assay: scales to 5,000+ genes |
| **Maximum genes** | ~5,000+ (with EEL FISH assay) |
| **Spatial resolution** | Subcellular; 81 nm pixel resolution (Synthetic Aperture Optics with 20x objective) |
| **Cell segmentation** | Built-in processing; integrated software |
| **Typical cell count** | 100K+ cells per run |
| **Run time** | <1 hour hands-on time; automated run |
| **Data format** | Processed single-cell data; spatial maps; standard formats |
| **Public datasets** | Limited public data |
| **Cost estimate** | Not publicly disclosed |
| **Key limitations** | Small company; limited adoption; smFISH assay limited to 30 genes; EEL assay still emerging; limited public validation data |

---

## 3. Sequencing-Based Platforms (Spot/Capture)

### 3.1 10x Genomics Visium (v1 / v2 / CytAssist)

| Specification | Details |
|--------------|---------|
| **Company** | 10x Genomics (Pleasanton, CA, USA) |
| **Platform name** | Visium Spatial Gene Expression (v1, v2, CytAssist) |
| **Method type** | Sequencing-based (spatially barcoded poly(dT) capture on slide + NGS) |
| **DAPI staining** | Not part of standard workflow (brightfield H&E is standard) |
| **H&E staining** | YES -- H&E is integral to the workflow; tissue section is H&E stained and imaged before permeabilization and capture; co-registered natively |
| **Other stainings** | IF (immunofluorescence) compatible on Visium CytAssist; brightfield H&E standard |
| **Gene panel size** | Whole transcriptome (~18,000 human / ~20,000 mouse genes via probe-based v2); or unbiased polyA capture (v1) |
| **Maximum genes** | Whole transcriptome (unbiased) |
| **Spatial resolution** | 55 um spot diameter; 100 um center-to-center; each spot covers 1-10 cells; NOT single-cell |
| **Cell segmentation** | None built-in (spot-level, not cell-level); deconvolution methods (cell2location, RCTD, Tangram) used to infer cell types per spot |
| **Typical cell count** | ~5,000 spots per 6.5x6.5 mm capture area; ~14,000 spots per XL (11x11 mm); represents mixed cell populations |
| **Data format** | Space Ranger output: feature-barcode matrix (H5/MEX); spatial coordinates; tissue images (TIFF); Loupe Browser compatible; Seurat/Scanpy import |
| **Public datasets** | [10x Genomics Datasets](https://www.10xgenomics.com/datasets) -- extensive collection across tissue types; [Bioconductor TENxVisiumData](https://github.com/HelenaLC/TENxVisiumData); Illumina BaseSpace |
| **Cost estimate** | ~$1,000-2,000/sample (reagents + sequencing); instrument: CytAssist ~$50K; standard Visium slides + reagents ~$300-500/slide |
| **Key limitations** | NOT single-cell resolution (55 um spots); gaps between spots (not continuous); requires deconvolution; being superseded by Visium HD; limited spatial resolution for cell-cell interaction studies |

### 3.2 10x Genomics Visium HD

| Specification | Details |
|--------------|---------|
| **Company** | 10x Genomics (Pleasanton, CA, USA) |
| **Platform name** | Visium HD Spatial Gene Expression |
| **Method type** | Sequencing-based (continuous barcoded array + probe-based capture + NGS) |
| **DAPI staining** | Not standard (H&E is the primary stain) |
| **H&E staining** | YES -- H&E natively integrated; tissue stained and imaged as part of workflow |
| **Other stainings** | IF compatible (CytAssist workflow) |
| **Gene panel size** | Whole transcriptome (probe-based) |
| **Maximum genes** | Whole transcriptome (~18,000+ genes) |
| **Spatial resolution** | 2 x 2 um barcoded squares (native); continuous lawn with no gaps; recommended analysis bin: 8 x 8 um; single cell-scale resolution |
| **Cell segmentation** | Not built-in at native 2 um level; binning approach (8x8 um recommended starting point); external tools for cell-level analysis |
| **Typical cell count** | Millions of 2 um bins per 6.5 x 6.5 mm capture area; single cell-scale at 8 um bins |
| **Data format** | Space Ranger output: binned feature matrices at multiple resolutions (2, 8, 16 um); spatial coordinates; H&E images; H5/MEX format |
| **Public datasets** | [10x Datasets](https://www.10xgenomics.com/datasets) -- human CRC, mouse intestine, mouse brain (tiny); Bioconductor STexampleData |
| **Cost estimate** | ~$2,000-4,000/sample (reagents + sequencing); higher sequencing depth needed than standard Visium |
| **Key limitations** | High sequencing cost (deep sequencing needed for 2 um resolution); data processing intensive; 8 um bins may still contain 1-2 cells; limited gene detection per bin at native 2 um; FFPE only currently |

### 3.3 Slide-seq / Slide-seqV2 (Curio Seeker)

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Macosko & Chen labs, Broad Institute); commercialized as Curio Seeker by Curio Bioscience (now Takara Bio USA) |
| **Platform name** | Slide-seq / Slide-seqV2 / Curio Seeker |
| **Method type** | Sequencing-based (randomly deposited barcoded beads + NGS) |
| **DAPI staining** | NO (bead-based capture, no imaging) |
| **H&E staining** | NO (tissue is lysed for RNA capture; histology on adjacent section only) |
| **Other stainings** | None on capture slide |
| **Gene panel size** | Whole transcriptome (unbiased polyA capture) |
| **Maximum genes** | Whole transcriptome |
| **Spatial resolution** | 10 um (bead diameter); near-cellular resolution |
| **Cell segmentation** | No built-in segmentation (bead-level analysis); computational deconvolution |
| **Typical cell count** | Tens of thousands of beads; variable capture efficiency |
| **Data format** | Bead-by-gene count matrix; bead spatial coordinates; standard scRNA-seq formats |
| **Curio Seeker kits** | 3x3 mm tiles (8 per kit) or 10x10 mm tiles (4 per kit); 8.5 hour library prep; ~2.5 hours hands-on |
| **Public datasets** | [Broad Single Cell Portal SCP815](https://singlecell.broadinstitute.org/single_cell/study/SCP815/) |
| **Cost estimate** | Curio Seeker kit: ~$500-1,000 + sequencing costs; no specialized instrument needed |
| **Key limitations** | 10 um resolution (not truly subcellular); variable RNA capture efficiency (V2 ~50% of scRNA-seq); FF only (no FFPE for Curio Seeker); no morphology images from capture slide; bead synthesis variability |

### 3.4 HDST (High-Definition Spatial Transcriptomics)

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Stahl/Lundeberg labs, SciLifeLab/KTH + Broad Institute) |
| **Platform name** | HDST |
| **Method type** | Sequencing-based (2 um bead array + spatial indexing + NGS) |
| **DAPI staining** | NO |
| **H&E staining** | YES -- tissue H&E imaged before capture |
| **Other stainings** | None standard |
| **Gene panel size** | Whole transcriptome |
| **Maximum genes** | Whole transcriptome |
| **Spatial resolution** | 2 um well diameter (bead array); subcellular |
| **Cell segmentation** | External |
| **Typical cell count** | Hundreds of thousands of spatial barcodes per experiment |
| **Data format** | Standard spatial count matrices; bead coordinates |
| **Public datasets** | [Single Cell Portal SCP420](https://singlecell.broadinstitute.org/single_cell/study/SCP420/hdst) -- mouse brain, breast cancer |
| **Cost estimate** | Academic method; no commercial product; cost of bead fabrication + sequencing |
| **Key limitations** | Academic-only; low capture efficiency; complex bead fabrication; not commercialized; superseded by Visium HD and other methods |

### 3.5 DBiT-seq

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Fan lab, Yale University) |
| **Platform name** | DBiT-seq (Deterministic Barcoding in Tissue) |
| **Method type** | Sequencing-based (microfluidic crossflow barcoding + NGS) |
| **DAPI staining** | NO |
| **H&E staining** | Compatible (tissue sections can be H&E stained before/after) |
| **Other stainings** | Antibody co-detection (multi-omics: protein + RNA simultaneously) |
| **Gene panel size** | Whole transcriptome (mRNA + multi-omics) |
| **Maximum genes** | Whole transcriptome |
| **Spatial resolution** | Adjustable: 10, 25, or 50 um pixel size (microfluidic channel width); theoretical limit ~5 um |
| **Cell segmentation** | Pixel-level; at 10 um approaches single-cell |
| **Typical cell count** | 50x50 = 2,500 pixels per section (standard); variable with channel design |
| **Data format** | Standard count matrices; microfluidic grid coordinates |
| **Public datasets** | GEO repositories (mouse embryo datasets) |
| **Cost estimate** | Academic method; PDMS chip fabrication + sequencing; relatively low cost |
| **Key limitations** | Academic-only; requires microfluidic expertise; limited pixel count per section (2,500); PDMS chip fabrication; not commercially available; Patho-DBiT variant for FFPE is newer |

### 3.6 Seq-Scope / Seq-Scope-X

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Jun Hee Lee lab, University of Michigan) |
| **Platform name** | Seq-Scope / Seq-Scope-eXpanded (Seq-Scope-X) |
| **Method type** | Sequencing-based (repurposed Illumina flow cells for spatial capture) |
| **DAPI staining** | NO (sequencing-based) |
| **H&E staining** | Adjacent section H&E registration possible |
| **Other stainings** | None on capture surface |
| **Gene panel size** | Whole transcriptome |
| **Maximum genes** | Whole transcriptome |
| **Spatial resolution** | 0.5-0.7 um center-to-center (standard); 0.2 um effective (with Seq-Scope-X tissue expansion); ~3M pixels/mm^2 (standard), ~27M pixels/mm^2 (expanded) |
| **Cell segmentation** | Computational; nuclear vs cytoplasmic compartments resolvable in Seq-Scope-X |
| **Typical cell count** | Variable; >23 UMI/um^2 demonstrated in colon |
| **Data format** | Standard spatial count matrices; hexagonal pixel array |
| **Public datasets** | Published datasets in associated papers (colon, liver) |
| **Cost estimate** | Low (repurposes standard Illumina flow cells); academic method |
| **Key limitations** | Academic-only; not commercialized; Seq-Scope-X requires tissue expansion (labor-intensive); dependent on Illumina flow cell availability; complex fabrication |

---

## 4. In-Situ Hybridization Methods (Academic)

### 4.1 seqFISH / seqFISH+

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Long Cai lab, Caltech); licensed to Spatial Genomics |
| **Platform name** | seqFISH / seqFISH+ |
| **Method type** | ISH (sequential barcoded FISH with pseudocolor encoding) |
| **DAPI staining** | YES (standard fluorescence microscopy) |
| **H&E staining** | Not standard |
| **Gene panel size** | seqFISH: hundreds of genes; seqFISH+: ~10,000 genes |
| **Maximum genes** | ~10,000 (seqFISH+; transcriptome-scale) |
| **Spatial resolution** | Subcellular; sub-diffraction-limit; single-molecule quantitation |
| **Cell segmentation** | Image-based (DAPI + fluorescence); external tools |
| **Key limitations** | Very slow (many imaging rounds); small FOV; limited throughput; Spatial Genomics attempting commercialization; complex experimental setup; optical crowding management needed |

### 4.2 osmFISH

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Linnarsson lab, Karolinska Institute) |
| **Platform name** | osmFISH (ouroboros smFISH) |
| **Method type** | ISH (non-barcoding cyclic smFISH) |
| **DAPI staining** | YES |
| **H&E staining** | Not standard |
| **Gene panel size** | ~33 genes demonstrated; theoretical ~8,000+ in cultured cells |
| **Maximum genes** | Limited by imaging rounds (non-barcoding); ~30-50 per tissue section typical |
| **Spatial resolution** | Subcellular; single-molecule |
| **Cell segmentation** | Image-based |
| **Key limitations** | Very low gene multiplexing (non-barcoding approach); extremely slow; academic-only; superseded by barcoded methods (MERFISH, seqFISH+) |

### 4.3 STARmap / STARmap PLUS

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Wang lab, Stanford/MIT) |
| **Platform name** | STARmap / STARmap PLUS |
| **Method type** | In-situ sequencing (SNAIL probes + hydrogel embedding + in situ sequencing) |
| **DAPI staining** | YES (tissue cleared and imaged) |
| **H&E staining** | Not standard (tissue is hydrogel-embedded) |
| **Gene panel size** | STARmap: 160-1,020 genes; STARmap PLUS: ~2,766 genes + protein co-detection |
| **Maximum genes** | ~2,766 (STARmap PLUS) |
| **Spatial resolution** | Subcellular; voxel size 95 x 95 x 350 nm (~100 nm); 3D volumetric |
| **Cell segmentation** | 3D volume-based segmentation |
| **Key limitations** | Complex protocol (hydrogel embedding, tissue clearing); 3D processing requirements; academic-only; limited throughput; not commercially available |

### 4.4 EEL FISH

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Nilsson/Mats lab); licensed to Rebus Biosystems for Esper platform |
| **Platform name** | EEL FISH (Enhanced ELectric FISH) |
| **Method type** | ISH (electrophoretic RNA transfer + cyclic FISH decoding) |
| **DAPI staining** | YES |
| **H&E staining** | Not standard |
| **Gene panel size** | ~448 genes per color channel; ~900 genes (2 channels); routinely 3 colors now |
| **Maximum genes** | ~1,350 (3 color channels); theoretical: thousands |
| **Spatial resolution** | Single-cell (RNA transferred electrostatically preserving spatial info) |
| **Cell segmentation** | Post-transfer decoding; computational |
| **Throughput** | 1 cm^2 of tissue in 58 hours (448 genes); nearly fully automated (4 hours hands-on) |
| **Key limitations** | RNA transfer step adds noise; requires ITO-coated slides; signal may be weaker than direct FISH; early-stage adoption |

### 4.5 HybISS (Hybridization-based In Situ Sequencing)

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Nilsson lab, Stockholm University / SciLifeLab) |
| **Platform name** | HybISS |
| **Method type** | ISH (padlock probes + RCA + sequence-by-hybridization readout) |
| **DAPI staining** | YES |
| **H&E staining** | Adjacent section compatible |
| **Gene panel size** | Hundreds to thousands of genes (improved over original ISS) |
| **Maximum genes** | ~1,000+ (improved combinatorial barcoding vs ISS) |
| **Spatial resolution** | Cellular; single-molecule detection via amplified signals |
| **Cell segmentation** | Image-based; epifluorescence microscopy compatible |
| **Key limitations** | Academic-only; padlock probe design complexity; lower multiplexing than MERFISH/seqFISH+; standard microscope compatible but slower; limited adoption |

### 4.6 ExSeq (Expansion Sequencing)

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Boyden lab, MIT + Church lab, Harvard) |
| **Platform name** | ExSeq |
| **Method type** | In-situ sequencing (tissue expansion + FISSEQ in situ sequencing) |
| **DAPI staining** | Compatible (after expansion) |
| **H&E staining** | Not compatible (tissue is physically expanded) |
| **Gene panel size** | Untargeted: thousands of genes (whole transcriptome); Targeted: custom panels |
| **Maximum genes** | Whole transcriptome (untargeted mode) |
| **Spatial resolution** | Nanoscale after expansion (tissue swells up to 100x); subcellular compartments (dendrites, spines) resolvable |
| **Cell segmentation** | 3D volume-based; post-expansion |
| **Key limitations** | Extremely complex protocol; tissue expansion distortion; very low throughput; long experiment time; academic-only; specialized equipment needed; not scalable |

---

## 5. Emerging / Newer Platforms

### 5.1 Open-ST

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Rajewsky lab, MDC Berlin) |
| **Platform name** | Open-ST |
| **Method type** | Sequencing-based (repurposed Illumina flow cells; open-source) |
| **Resolution** | ~0.6 um capture spot resolution; subcellular |
| **Gene coverage** | Whole transcriptome |
| **H&E** | YES -- H&E stained cryosections used; 2D/3D reconstruction with histology |
| **Cost** | <130 EUR per 12 mm^2 capture area (very low cost) |
| **Status** | Published in Cell (2024); protocol in STAR Protocols (2025) |
| **Key limitation** | Academic; requires Illumina flow cells; fresh frozen only |

### 5.2 Nova-ST

| Specification | Details |
|--------------|---------|
| **Company** | Academic |
| **Platform name** | Nova-ST |
| **Method type** | Sequencing-based (nano-patterned barcoded Illumina flow cells) |
| **Resolution** | High resolution (Illumina flow cell density) |
| **Gene coverage** | Whole transcriptome |
| **Cost** | Low cost; open-source |
| **Status** | Published 2024; protocol on protocols.io |
| **Key limitation** | Academic; requires optimization per tissue |

### 5.3 MAGIC-seq

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Chinese institutions) |
| **Platform name** | MAGIC-seq (Microfluidics-Assisted Grid Chips) |
| **Method type** | Sequencing-based (carbodiimide chemistry + spatial combinatorial indexing + microfluidics) |
| **Resolution** | Near single-cell |
| **Gene coverage** | Whole transcriptome |
| **Throughput** | 8x increase over standard methods; minimal cost and reduced batch effects |
| **Status** | Published in Nature Genetics (2024) |
| **Key limitation** | Academic; custom microfluidics fabrication |

### 5.4 Decoder-seq

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Yang lab, Xiamen University) |
| **Platform name** | Decoder-seq / Decoder-FFPE-seq |
| **Method type** | Sequencing-based (dendrimeric DNA nanosubstrates + microfluidic barcoding) |
| **Resolution** | Near-cellular; ~10x higher DNA density than previous methods |
| **Gene coverage** | Whole transcriptome |
| **Capture efficiency** | 20-30% (high for ST); Decoder-FFPE-seq: 2.5-4.25x improvement over similar assays |
| **Status** | Published in Nature Biotechnology (2024) |
| **Key limitation** | Academic; complex nanosubstrate fabrication |

### 5.5 PIXEL-seq

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Gu lab, University of Washington) |
| **Platform name** | PIXEL-seq |
| **Method type** | Sequencing-based (polony gel arrays; ~1 um clonal DNA clusters) |
| **Resolution** | ~1 um feature diameter; 0.5-0.8M features/mm^2 |
| **Gene coverage** | Whole transcriptome + proteomics |
| **Performance** | 58.9 UMIs at 2 um resolution; 1,199 at 10 um; 25,618 at 50 um |
| **Cost** | 35x lower cost than sequencing-dependent array fabrication; ~7 hours fabrication |
| **Status** | Published in Cell (2022) |
| **Key limitation** | Academic; polony gel fabrication expertise needed |

### 5.6 Slide-tags

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Bhatt/Macosko labs, Broad Institute) |
| **Platform name** | Slide-tags |
| **Method type** | Proximity-based (photocleavable spatial barcodes diffuse into nuclei + snRNA-seq) |
| **Resolution** | <10 um spatial resolution; single-nucleus |
| **Gene coverage** | Whole transcriptome (snRNA-seq quality) |
| **Multimodal** | Compatible with ATAC-seq, protein, and other -omics modalities |
| **Status** | Published in Nature (2024) |
| **Key limitation** | 20 um tissue sections; fresh frozen only; requires bead array fabrication and spatial indexing |

### 5.7 sci-Space

| Specification | Details |
|--------------|---------|
| **Company** | Academic (Trapnell/Shendure labs, UW) |
| **Platform name** | sci-Space |
| **Method type** | Combinatorial indexing with spatial hashing |
| **Resolution** | ~200 um (80 um-radius spots); low spatial resolution |
| **Gene coverage** | Whole transcriptome (~1,200 genes/cell) |
| **Scale** | ~120,000 nuclei per experiment |
| **Status** | Published in Science (2021) |
| **Key limitation** | Very low spatial resolution (200 um); cannot resolve cell-cell interactions; superseded by higher-resolution methods |

### 5.8 Stereo-CITE-seq

| Specification | Details |
|--------------|---------|
| **Company** | BGI |
| **Platform name** | Stereo-CITE-seq |
| **Method type** | Multi-omics (Stereo-seq + CITE-seq antibody barcoding) |
| **Resolution** | 500 nm (Stereo-seq resolution) |
| **RNA coverage** | Whole transcriptome |
| **Protein detection** | 17-plex ADTs (surface proteins) |
| **Status** | Published; early stage |
| **Key limitation** | Only 17 surface proteins; surface proteins only; early-stage technology |

---

## 6. ROI-Based / Digital Spatial Profiling

### 6.1 NanoString GeoMx DSP (now Bruker)

| Specification | Details |
|--------------|---------|
| **Company** | NanoString Technologies / Bruker Spatial Biology |
| **Platform name** | GeoMx Digital Spatial Profiler (DSP) |
| **Method type** | ROI-based (UV-cleavable oligo tags on tissue + NGS or nCounter readout) |
| **DAPI staining** | YES (used for morphology visualization and ROI selection) |
| **H&E staining** | YES -- compatible; non-destructive profiling preserves tissue for subsequent H&E |
| **Other stainings** | IF morphology markers for ROI selection (up to 4 channels); compatible with standard IHC/IF |
| **Gene panel size** | Cancer Transcriptome Atlas (CTA): ~1,800 genes; Whole Transcriptome Atlas (WTA): ~18,000 genes; Protein panels: 570+ targets |
| **Maximum genes** | ~18,000 (WTA) + 570+ proteins (separately or simultaneously) |
| **Spatial resolution** | ROI-level: down to 10 um (individual cell illumination possible); NOT single-cell resolution (pools cells within ROI) |
| **Cell segmentation** | None (ROI-based, not cell-based); segmentation-free approach using tissue compartments |
| **ROIs per slide** | Up to 90 ROIs per tissue slide; flexible geometric shapes |
| **Typical throughput** | Up to 4 tissue slides per day |
| **Data format** | Gene count matrices per ROI; DCC/PKC files; nCounter or NGS readout; Seurat/Scanpy compatible |
| **Public datasets** | GEO repositories; NanoString/Bruker data portal |
| **Cost estimate** | Instrument: ~$295K; ~$500/sample (translational research); library prep: ~$1,475/plate (95 AOIs) |
| **Key limitations** | NOT single-cell (pools cells within ROI); destructive UV cleavage consumes material; ROI selection is manual/biased; throughput limited by ROI count; lower spatial resolution than imaging-based platforms; requires user to define regions of interest |

---

## 7. Specialty / Unique Modality Platforms

### 7.1 Pixelgen Molecular Pixelation (MPX)

| Specification | Details |
|--------------|---------|
| **Company** | Pixelgen Technologies (Stockholm, Sweden) |
| **Platform name** | Molecular Pixelation (MPX) |
| **Method type** | Proximity-based sequencing (DNA nanopixels link cell surface proteins; optics-free) |
| **DAPI staining** | NO (optics-free method; no imaging) |
| **H&E staining** | NO |
| **Target type** | Cell surface PROTEINS (not RNA transcriptomics) |
| **Panel size** | 80 protein targets (Immunology Panel 2, including 4 controls) |
| **Spatial resolution** | ~50 nm average (protein-protein proximity on cell surface); 3D cell surface mapping |
| **Cell count** | 100-1,000 cells per sample |
| **Readout** | NGS (sequencing-based spatial inference) |
| **Data format** | Pixelator analysis tool output; 3D protein proximity networks; co-localization scores |
| **Status** | Published in Nature Methods (2024); commercial kits available |
| **Key limitations** | PROTEINS ONLY (not transcriptomics); cell surface proteins only; low cell count (100-1,000); requires PFA fixation; complex workflow (~2 days); niche application |

---

## 8. Cross-Platform Comparison Tables

### 8.1 Key Specifications Comparison (Commercial Platforms)

| Platform | Company | Resolution | Max Genes | WTA? | DAPI? | H&E? | Protein? | Cost/Sample |
|----------|---------|-----------|-----------|------|-------|------|----------|-------------|
| **Xenium** | 10x Genomics | 200 nm/px | 5,100 | No (targeted) | Yes | Post-run | Add-on panels | $3K-12K |
| **MERSCOPE** | Vizgen | 100 nm | 1,000 | No (targeted) | Yes | No | 6-9 plex | $2K-5K |
| **CosMx SMI** | Bruker/NanoString | 120 nm/px | 18,935 | Yes (2.0) | Yes | Post-run | 76 plex | $2K-5K |
| **Stereo-seq** | BGI/STOmics | 500 nm | WTA | Yes | Yes | Yes | 17 (CITE) | $3K-5K |
| **Visium** | 10x Genomics | 55 um | WTA | Yes | No (H&E) | Yes | IF option | $1K-2K |
| **Visium HD** | 10x Genomics | 2 um (bin 8 um) | WTA | Yes | No (H&E) | Yes | IF option | $2K-4K |
| **GeoMx DSP** | Bruker/NanoString | 10 um (ROI) | 18,000 | Yes | Yes | Yes | 570+ | $500+ |
| **Curio Seeker** | Takara Bio | 10 um | WTA | Yes | No | No | No | $500-1K + seq |
| **Resolve MC** | Resolve Bio | 300 nm | 330 | No | Yes | Post-run | No | Service-based |

### 8.2 DAPI Specifications Comparison

| Platform | DAPI Available | Pixel Size (um) | Bit Depth | Z-stack | Format |
|----------|---------------|-----------------|-----------|---------|--------|
| **Xenium** | Yes | 0.2125 | 16-bit | Yes (3D) | OME-TIFF (pyramidal) |
| **MERSCOPE** | Yes | ~0.108 | 16-bit | Yes (7 planes, 0.7 um spacing) | TIFF mosaic |
| **CosMx SMI** | Yes | 0.120 | 16-bit | Not standard | OME-TIFF |
| **Stereo-seq** | Yes | Variable | Variable | No (2D) | TIFF |
| **Resolve MC** | Yes | ~0.300 | Variable | Optional Z | TIFF |
| **GeoMx DSP** | Yes | Variable | - | No | - |
| **Visium/HD** | No (H&E) | - | - | - | TIFF (brightfield) |
| **MERSCOPE Ultra** | Yes | ~0.108 | 16-bit | Yes | TIFF mosaic |

### 8.3 Cell Segmentation Comparison

| Platform | Built-in Segmentation | Method | Stains Used | External Tool Support |
|----------|----------------------|--------|-------------|----------------------|
| **Xenium** | Yes (Multimodal) | Deep learning (3-step) | DAPI + ATP1A1/CD45/E-Cad + 18S | Cellpose, StarDist |
| **MERSCOPE** | Yes (Cellpose) | Cellpose-based | DAPI + PolyT + Cell Boundary Kit | VPT, custom Cellpose |
| **CosMx** | Yes (ML-augmented) | Membrane + nuclei + RNA | DAPI + PanCK/CD45/CD68 + PolyT | External pipelines |
| **Stereo-seq** | Optional (StereoCell) | Multiple algorithms | DAPI/ssDNA or H&E | Cellpose, DeepCell, MEDIAR, SAM |
| **Visium/HD** | No (spot-level) | N/A (deconvolution) | H&E | cell2location, RCTD, Tangram |
| **Resolve MC** | No | N/A | DAPI (provided) | Cellpose, nf-core/molkart |

---

## 9. Benchmarking Results (2024-2026)

### 9.1 Imaging Platform Comparison: Xenium vs MERSCOPE vs CosMx

**Source:** "Systematic benchmarking of imaging spatial transcriptomics platforms in FFPE tissues" (Nature Communications, 2025). Benchmarked on serial sections from tissue microarrays (17 tumor + 16 normal tissue types).

| Metric | Xenium | MERSCOPE | CosMx |
|--------|--------|----------|-------|
| **Transcript counts per gene** | Highest | Lower | Moderate |
| **Sensitivity (shared genes)** | Reference | ~0.5x (2x less) | ~0.08x (12x less) |
| **On-target fraction** | Highest | Moderate | Lowest |
| **False discovery rate** | Lowest | Moderate | Higher |
| **Cell types resolved (breast)** | 9/9 expected | Not tested in this study | 6/9 expected |
| **Concordance with scRNA-seq** | High | Not measured | High |

**Key finding:** Xenium outperforms both CosMx and MERSCOPE in sensitivity and specificity. The chemistry difference: Xenium uses padlock probes + RCA, CosMx uses branch-chain hybridization, MERSCOPE uses direct hybridization with many tiled probes.

### 9.2 Subcellular Platform Comparison (2025)

**Source:** "Systematic benchmarking of high-throughput subcellular spatial transcriptomics platforms across human tumors" (Nature Communications, 2025). Compared Stereo-seq v1.3, Visium HD FFPE, CosMx 6K, and Xenium 5K.

| Metric | Xenium 5K | CosMx 6K | Visium HD | Stereo-seq v1.3 |
|--------|-----------|----------|-----------|-----------------|
| **Background noise** | Lower | Higher | N/A | N/A |
| **Spatial signal** | Stronger | Comparable | Superior fidelity | Variable |
| **Cell type classification** | Comparable | Comparable | Via binning | Via segmentation |

### 9.3 Sequencing Platform Comparison (2025-2026)

**Source:** "A technical comparison of spatial transcriptomics platforms across six cancer types" (Genome Biology, 2026). Compared Visium v1, Visium v2/CytAssist, Visium HD, Xenium, CosMx.

- Visium v2 improves over v1 in gene detection and spatial coherence
- Visium HD achieves superior spatial fidelity among sequencing-based platforms
- Xenium and CosMx yield comparable cell type classifications when panels overlap
- Xenium shows lower background noise and stronger spatial signal than CosMx

### 9.4 Method of the Year

Spatial Transcriptomics was recognized as Nature's **Method of the Year 2024**, reflecting the maturation and adoption of these technologies.

---

## 10. Platform Adoption and Usage

### 10.1 Most Commonly Used Platforms (2024-2026)

**For whole-transcriptome discovery (hypothesis generation):**
1. 10x Genomics Visium (v1/v2/CytAssist) -- most published datasets
2. Stereo-seq -- growing rapidly, especially in Asia
3. Visium HD -- rapidly replacing standard Visium

**For targeted single-cell analysis (hypothesis testing):**
1. 10x Genomics Xenium -- market leader; highest sensitivity/specificity
2. NanoString/Bruker CosMx -- whole transcriptome advantage (2.0)
3. Vizgen MERSCOPE -- strong MERFISH legacy; MERSCOPE Ultra for larger panels

**For spatial multi-omics:**
1. CosMx (RNA + protein on same slide)
2. GeoMx DSP (WTA + 570 protein targets)
3. Stereo-CITE-seq (limited protein plex)

### 10.2 Resolution Progression (2016-2026)

| Year | Platform | Resolution |
|------|----------|-----------|
| 2016 | ST (original) | 100 um |
| 2019 | Visium | 55 um |
| 2019 | Slide-seq | 10 um |
| 2022 | Stereo-seq | 500 nm |
| 2022 | PIXEL-seq | ~1 um |
| 2023 | Visium HD | 2 um |
| 2023 | Seq-Scope | 0.5-0.7 um |
| 2024 | Open-ST | 0.6 um |
| 2024 | Stereo-seq V2 | 0.22 um (claimed) |
| 2025 | Seq-Scope-X | 0.2 um (with expansion) |

This represents a **450x resolution improvement** over ~5 years (55 um to 0.12 um).

### 10.3 Practical Guide for Platform Selection

| Research Goal | Recommended Platform(s) | Rationale |
|--------------|------------------------|-----------|
| Whole transcriptome + morphology | Visium HD, Stereo-seq | WTA + H&E co-registered |
| Single-cell targeted profiling | Xenium, CosMx | Highest sensitivity at subcellular resolution |
| Maximum gene panel (imaging) | CosMx 2.0 (WTX) | ~19K genes at single-cell resolution |
| Largest tissue area | Stereo-seq (up to 13x13 cm) | Unmatched FOV |
| Lowest cost per sample | Open-ST, Curio Seeker | <130 EUR or ~$500 + sequencing |
| FFPE clinical samples | Xenium, CosMx, Visium HD | All optimized for FFPE |
| Protein + RNA same slide | CosMx, GeoMx DSP | Up to 76 or 570+ proteins |
| Fresh frozen only | Curio Seeker, Slide-tags, Open-ST | No FFPE support |
| Multi-omics (epigenome) | Spatial-CUT&Tag, DBiT-seq | Chromatin + RNA |

---

## Key Reference Papers

1. "Systematic benchmarking of imaging spatial transcriptomics platforms in FFPE tissues" -- Nature Communications, 2025
2. "Systematic benchmarking of high-throughput subcellular spatial transcriptomics platforms across human tumors" -- Nature Communications, 2025
3. "A technical comparison of spatial transcriptomics platforms across six cancer types" -- Genome Biology, 2026
4. "Comparison of imaging-based single-cell resolution spatial transcriptomics profiling platforms using FFPE tumor samples" -- Nature Communications, 2025
5. "A practical guide for choosing an optimal spatial transcriptomics technology from seven major commercially available options" -- BMC Genomics, 2025
6. "Museum of spatial transcriptomics" -- Nature Methods, 2022 (historical overview)
7. "Systematic comparison of sequencing-based spatial transcriptomic methods" -- Nature Methods, 2024

---

## Notes for DAPIDL Project Relevance

For the DAPIDL pipeline (DAPI-based cell type prediction), the key considerations are:

| Platform | DAPIDL Compatibility | Notes |
|----------|---------------------|-------|
| **Xenium** | PRIMARY -- native DAPI (0.2125 um/px, 16-bit, OME-TIFF) | Current production platform |
| **MERSCOPE** | COMPATIBLE -- native DAPI (~0.108 um/px); ~16x higher intensity than Xenium | Confirmed working; requires adaptive normalization |
| **CosMx** | COMPATIBLE -- native DAPI (0.120 um/px) | Not yet tested; similar to MERSCOPE pipeline |
| **Stereo-seq** | COMPATIBLE -- native DAPI available | Different data format (GEF); would need adapter |
| **Resolve MC** | COMPATIBLE -- DAPI TIFF provided | Service-only limits utility |
| **Visium/HD** | NOT COMPATIBLE -- H&E only, no DAPI | Would need separate staining |
| **GeoMx** | PARTIAL -- DAPI for ROI selection but ROI-level only | Not suitable for cell-level DAPIDL |
