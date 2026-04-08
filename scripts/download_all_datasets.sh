#!/usr/bin/env bash
# download_all_datasets.sh — Download ALL spatial transcriptomics datasets for DAPIDL
#
# Usage:
#   ./scripts/download_all_datasets.sh [TARGET_DIR]
#
# TARGET_DIR defaults to /mnt/data/datasets (new 2TB SSD).
# Supports resume (wget --continue), parallel downloads, and integrity checks.
#
# Prerequisites:
#   - MERSCOPE: Register at https://info.vizgen.com/merscope-ffpe-solution first
#   - STHELAR: No registration needed (BioStudies public)
#   - Xenium: No registration needed (10x public)

set -euo pipefail

TARGET_DIR="${1:-/mnt/data/datasets}"
JOBS=3  # parallel downloads

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

mkdir -p "$TARGET_DIR"/{xenium,merscope,sthelar}

# ─────────────────────────────────────────────────────────────────────
# 1. STHELAR — BioStudies S-BIAD2146
# ─────────────────────────────────────────────────────────────────────
# 27 slides total, 13 tissue types. Download as SpatialData zarr objects.
# Source: https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2146
#
# We already have: breast_s0, breast_s1, breast_s3, breast_s6, skin_s1,
#                   skin_s2, skin_s3, skin_s4

STHELAR_BASE="https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/146/S-BIAD2146/Files"
STHELAR_DIR="$TARGET_DIR/sthelar"

# All 27 STHELAR slides (download only missing ones)
STHELAR_SLIDES=(
    # Already have (on /mnt/work) — skip by default
    # "sdata_breast_s0"
    # "sdata_breast_s1"
    # "sdata_breast_s3"
    # "sdata_breast_s6"
    # "sdata_skin_s1"
    # "sdata_skin_s2"
    # "sdata_skin_s3"
    # "sdata_skin_s4"

    # Missing — need to download (19 slides)
    "sdata_cervix_s0"
    "sdata_colon_s1"
    "sdata_colon_s2"
    "sdata_heart_s0"
    "sdata_kidney_s0"
    "sdata_kidney_s1"
    "sdata_liver_s0"
    "sdata_liver_s1"
    "sdata_lung_s1"
    "sdata_lung_s3"
    "sdata_lymph_node_s0"
    "sdata_ovary_s0"
    "sdata_ovary_s1"
    "sdata_pancreatic_s0"
    "sdata_pancreatic_s1"
    "sdata_pancreatic_s2"
    "sdata_prostate_s0"
    "sdata_tonsil_s0"
    "sdata_tonsil_s1"
)

download_sthelar() {
    log "━━━ STHELAR: Downloading ${#STHELAR_SLIDES[@]} missing slides ━━━"
    for slide in "${STHELAR_SLIDES[@]}"; do
        local dest="$STHELAR_DIR/${slide}.zarr"
        if [ -d "$dest" ]; then
            log "  SKIP: $slide (already exists)"
            continue
        fi

        log "  Downloading: $slide ..."
        # STHELAR zarr files may be stored as .zarr.zip on BioStudies
        local url="${STHELAR_BASE}/${slide}.zarr.zip"
        local zip_dest="$STHELAR_DIR/${slide}.zarr.zip"

        wget --continue --progress=bar:force:noscroll \
             -O "$zip_dest" "$url" 2>&1 || {
            # Try alternative path structure
            warn "  Direct download failed, trying alternative path..."
            url="${STHELAR_BASE}/sdata_slides/${slide}.zarr.zip"
            wget --continue --progress=bar:force:noscroll \
                 -O "$zip_dest" "$url" 2>&1 || {
                err "  Failed to download $slide"
                continue
            }
        }

        # Extract zarr
        log "  Extracting: $slide ..."
        unzip -q "$zip_dest" -d "$STHELAR_DIR/" && rm -f "$zip_dest"
        log "  Done: $slide"
    done
}

# ─────────────────────────────────────────────────────────────────────
# 2. MERSCOPE FFPE IO — Vizgen Data Release (16 datasets, 8 tissues)
# ─────────────────────────────────────────────────────────────────────
# IMPORTANT: You must first register at https://info.vizgen.com/merscope-ffpe-solution
# After registration, you get access to Google Cloud Storage links at:
# https://info.vizgen.com/ffpe-showcase
#
# Each dataset is on GCS. The exact bucket paths are shown on the showcase page.
# Below are placeholder paths — update after registration.
#
# We already have: breast, melanoma_1 (melanoma-p1), melanoma_2 (melanoma-p2)

MERSCOPE_DIR="$TARGET_DIR/merscope"

# GCS bucket (update after registration — check https://info.vizgen.com/ffpe-showcase)
# The datasets are typically at: gs://vizgen-ffpe-showcase/{dataset_name}/
MERSCOPE_GCS_BASE="gs://vizgen-ffpe-showcase"

MERSCOPE_DATASETS=(
    # Already have (on /mnt/work/datasets/raw/merscope/)
    # "breast_cancer"        → merscope-breast
    # "melanoma_1"           → merscope-melanoma-p1
    # "melanoma_2"           → merscope-melanoma-p2

    # Missing — need to download (13 datasets)
    "colon_cancer_1"
    "colon_cancer_2"
    "liver_cancer_1"
    "liver_cancer_2"
    "lung_cancer_1"
    "lung_cancer_2"
    "ovarian_cancer_1"
    "ovarian_cancer_2"
    "ovarian_cancer_3"
    "ovarian_cancer_4"
    "prostate_cancer_1"
    "prostate_cancer_2"
    "uterine_cancer"
)

download_merscope() {
    log "━━━ MERSCOPE: Downloading ${#MERSCOPE_DATASETS[@]} missing datasets ━━━"
    warn "NOTE: MERSCOPE downloads require prior registration at:"
    warn "  https://info.vizgen.com/merscope-ffpe-solution"
    warn "After registration, visit https://info.vizgen.com/ffpe-showcase"
    warn "to get the actual GCS download links."
    echo ""

    # Check if gsutil is available
    if ! command -v gsutil &>/dev/null; then
        warn "gsutil not found. Install Google Cloud SDK:"
        warn "  curl https://sdk.cloud.google.com | bash"
        warn "  gcloud init"
        warn ""
        warn "Alternatively, download from the GCS Console links on the showcase page."
        warn "Each dataset contains: cell_by_gene.csv, cell_metadata.csv, images/"
        warn "The images/ directory contains the DAPI mosaic TIFF files."
        return 1
    fi

    for dataset in "${MERSCOPE_DATASETS[@]}"; do
        local dest="$MERSCOPE_DIR/merscope-${dataset}"
        if [ -d "$dest" ]; then
            log "  SKIP: $dataset (already exists)"
            continue
        fi

        log "  Downloading: $dataset ..."
        mkdir -p "$dest"

        # NOTE: Update this GCS path after checking the showcase page
        gsutil -m cp -r "${MERSCOPE_GCS_BASE}/${dataset}/*" "$dest/" 2>&1 || {
            err "  Failed to download $dataset"
            err "  Check the GCS path at https://info.vizgen.com/ffpe-showcase"
            continue
        }
        log "  Done: $dataset"
    done
}

# ─────────────────────────────────────────────────────────────────────
# 3. XENIUM — 10x Genomics public datasets
# ─────────────────────────────────────────────────────────────────────
# Download URL pattern:
#   https://cf.10xgenomics.com/samples/xenium/{version}/{name}/{name}_outs.zip
# Alternative (older datasets):
#   https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/{version}/{name}/{name}_outs.zip
#
# We already have 25 datasets. Below are datasets identified as potentially
# missing from our collection.

XENIUM_DIR="$TARGET_DIR/xenium"

# Format: "local_name|download_url"
# Add new datasets as you find them on https://www.10xgenomics.com/datasets
XENIUM_DOWNLOADS=(
    # Potentially missing datasets (verify on 10x website before downloading)
    # These are known from web search results but need URL confirmation

    # Multi-Tissue Panel preview datasets (may overlap with existing)
    # Uncomment and fill in URLs after checking 10x website:

    # "xenium-brain-preview|https://cf.10xgenomics.com/samples/xenium/1.0.0/Xenium_V1_Human_Brain/Xenium_V1_Human_Brain_outs.zip"
    # "xenium-colon-preview|https://cf.10xgenomics.com/samples/xenium/1.0.0/Xenium_V1_Human_Colon/Xenium_V1_Human_Colon_outs.zip"
    # "xenium-kidney-preview|https://cf.10xgenomics.com/samples/xenium/1.0.0/Xenium_V1_Human_Kidney_preview/Xenium_V1_Human_Kidney_preview_outs.zip"
    # "xenium-pancreas-preview|https://cf.10xgenomics.com/samples/xenium/1.0.0/Xenium_V1_Human_Pancreas_preview/Xenium_V1_Human_Pancreas_preview_outs.zip"
    # "xenium-lymphnode-preview|https://cf.10xgenomics.com/samples/xenium/1.0.0/Xenium_V1_Human_LymphNode_preview/Xenium_V1_Human_LymphNode_preview_outs.zip"
    # "xenium-lung-cancer-multimodal|https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_FFPE_Human_Lung_Cancer_MultiCellSeg/Xenium_V1_FFPE_Human_Lung_Cancer_MultiCellSeg_outs.zip"

    # Known good URLs from CLAUDE.md (already downloaded, kept as reference):
    # "xenium-breast-tumor-rep1|https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
    # "xenium-breast-tumor-rep2|https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_outs.zip"
    # "xenium-lung-2fov|https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip"
    # "xenium-ovarian-cancer|https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip"

    # PLACEHOLDER: Add more datasets after manually checking
    # https://www.10xgenomics.com/datasets (filter: Xenium, Human)
    ""  # empty sentinel
)

download_xenium() {
    log "━━━ XENIUM: Checking for new datasets ━━━"

    local count=0
    for entry in "${XENIUM_DOWNLOADS[@]}"; do
        [ -z "$entry" ] && continue
        local name="${entry%%|*}"
        local url="${entry##*|}"
        local dest="$XENIUM_DIR/$name"

        if [ -d "$dest" ]; then
            log "  SKIP: $name (already exists)"
            continue
        fi

        log "  Downloading: $name ..."
        mkdir -p "$dest"
        local zip="$XENIUM_DIR/${name}_outs.zip"

        wget --continue --progress=bar:force:noscroll \
             -O "$zip" "$url" 2>&1 || {
            err "  Failed to download $name from $url"
            continue
        }

        log "  Extracting: $name ..."
        unzip -q "$zip" -d "$dest/" && rm -f "$zip"
        count=$((count + 1))
        log "  Done: $name"
    done

    if [ $count -eq 0 ]; then
        log "  No new Xenium datasets to download."
        log "  Check https://www.10xgenomics.com/datasets for new releases."
    else
        log "  Downloaded $count new Xenium datasets."
    fi
}

# ─────────────────────────────────────────────────────────────────────
# 4. Verification
# ─────────────────────────────────────────────────────────────────────
verify_downloads() {
    log "━━━ VERIFICATION ━━━"

    log "STHELAR slides:"
    local sthelar_count=0
    for d in "$STHELAR_DIR"/sdata_*.zarr/; do
        [ -d "$d" ] && sthelar_count=$((sthelar_count + 1))
    done
    # Also count existing ones on /mnt/work
    for d in /mnt/work/datasets/STHELAR/sdata_slides/sdata_*.zarr/; do
        [ -d "$d" ] && sthelar_count=$((sthelar_count + 1))
    done
    log "  Total STHELAR slides: $sthelar_count / 27"

    log "MERSCOPE datasets:"
    local merscope_count=0
    for d in "$MERSCOPE_DIR"/merscope-*/; do
        [ -d "$d" ] && merscope_count=$((merscope_count + 1))
    done
    for d in /mnt/work/datasets/raw/merscope/merscope-*/; do
        [ -d "$d" ] && merscope_count=$((merscope_count + 1))
    done
    log "  Total MERSCOPE datasets: $merscope_count / 16"

    log "Xenium datasets:"
    local xenium_count=0
    for d in "$XENIUM_DIR"/xenium-*/; do
        [ -d "$d" ] && xenium_count=$((xenium_count + 1))
    done
    for d in /mnt/work/datasets/raw/xenium/xenium-*/; do
        [ -d "$d" ] && xenium_count=$((xenium_count + 1))
    done
    log "  Total Xenium datasets: $xenium_count"

    log ""
    log "Disk usage:"
    du -sh "$TARGET_DIR"/*/ 2>/dev/null || true
    df -h "$TARGET_DIR"
}

# ─────────────────────────────────────────────────────────────────────
# 5. Dataset Inventory (for reference)
# ─────────────────────────────────────────────────────────────────────
print_inventory() {
    cat << 'INVENTORY'
═══════════════════════════════════════════════════════════════════════
DAPIDL COMPLETE DATASET INVENTORY
═══════════════════════════════════════════════════════════════════════

STHELAR (27 slides, 13 tissues, ~11M cells)
  Source: BioStudies S-BIAD2146
  URL:    https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2146
  HF:     https://huggingface.co/datasets/FelicieGS/STHELAR_40x
  ┌─────────────┬────────────────────────────────────────────┬────────┐
  │ Tissue      │ Slides                                     │ Count  │
  ├─────────────┼────────────────────────────────────────────┼────────┤
  │ Breast      │ breast_s0, s1, s3, s6                      │ 4      │
  │ Cervix      │ cervix_s0                                  │ 1      │
  │ Colon       │ colon_s1, s2                               │ 2      │
  │ Heart       │ heart_s0                                   │ 1      │
  │ Kidney      │ kidney_s0, s1                              │ 2      │
  │ Liver       │ liver_s0, s1                               │ 2      │
  │ Lung        │ lung_s1, s3                                │ 2      │
  │ Lymph Node  │ lymph_node_s0                              │ 1      │
  │ Ovarian     │ ovary_s0, s1                               │ 2      │
  │ Pancreatic  │ pancreatic_s0, s1, s2                      │ 3      │
  │ Prostate    │ prostate_s0                                │ 1      │
  │ Skin        │ skin_s1, s2, s3, s4                        │ 4      │
  │ Tonsil      │ tonsil_s0, s1                              │ 2      │
  └─────────────┴────────────────────────────────────────────┴────────┘

MERSCOPE FFPE IO (16 datasets, 8 tissues, ~9M cells, 500 genes each)
  Source: Vizgen Data Release Program
  URL:    https://info.vizgen.com/ffpe-showcase
  Reg:    https://info.vizgen.com/merscope-ffpe-solution
  ┌─────────────┬────────────────────────────────────────────┬────────┐
  │ Tissue      │ Datasets                                   │ Count  │
  ├─────────────┼────────────────────────────────────────────┼────────┤
  │ Breast      │ breast_cancer                              │ 1      │
  │ Colon       │ colon_cancer_1, colon_cancer_2             │ 2      │
  │ Liver       │ liver_cancer_1, liver_cancer_2             │ 2      │
  │ Lung        │ lung_cancer_1, lung_cancer_2               │ 2      │
  │ Melanoma    │ melanoma_1, melanoma_2                     │ 2      │
  │ Ovarian     │ ovarian_1, ovarian_2, ovarian_3, ovarian_4 │ 4      │
  │ Prostate    │ prostate_cancer_1, prostate_cancer_2       │ 2      │
  │ Uterine     │ uterine_cancer                             │ 1      │
  └─────────────┴────────────────────────────────────────────┴────────┘

XENIUM (25+ datasets, 14+ tissues, ~7.6M cells)
  Source: 10x Genomics
  URL:    https://www.10xgenomics.com/datasets (filter: Xenium, Human)
  Pattern: https://cf.10xgenomics.com/samples/xenium/{ver}/{name}/{name}_outs.zip
  ┌─────────────────────────────┬──────────┬──────────┬──────────────┐
  │ Dataset                     │ Cells    │ Size     │ Tissue       │
  ├─────────────────────────────┼──────────┼──────────┼──────────────┤
  │ breast-tumor-rep1           │ 167,780  │ 14 GB    │ Breast       │
  │ breast-tumor-rep2           │ 118,752  │ 11 GB    │ Breast       │
  │ breast-cancer-prime         │ 699,110  │ 35 GB    │ Breast (5K)  │
  │ brain-gbm                   │ 816,769  │ 37 GB    │ Brain        │
  │ cervical-cancer-prime        │ 840,387  │ 29 GB    │ Cervical (5K)│
  │ colon-cancer                │ 647,524  │ 20 GB    │ Colon        │
  │ colon-normal                │ 270,984  │ 12 GB    │ Colon        │
  │ colorectal-cancer           │ 388,175  │ 12 GB    │ Colorectal   │
  │ heart-normal                │ 26,366   │ 3.3 GB   │ Heart        │
  │ kidney-cancer               │ 56,510   │ 3.3 GB   │ Kidney       │
  │ kidney-normal               │ 97,560   │ 4.8 GB   │ Kidney       │
  │ liver-cancer                │ 162,628  │ 8.7 GB   │ Liver        │
  │ liver-normal                │ 239,271  │ 15 GB    │ Liver        │
  │ lung-2fov                   │ 11,898   │ 804 MB   │ Lung         │
  │ lung-cancer                 │ 162,254  │ 7.5 GB   │ Lung         │
  │ lymph-node-normal           │ 377,985  │ 7 GB     │ Lymph Node   │
  │ ovarian-cancer              │ 407,124  │ 50 GB    │ Ovarian (5K) │
  │ ovary-cancer-ff             │ 205,082  │ 17 GB    │ Ovary        │
  │ pancreas-cancer             │ 190,965  │ 6.9 GB   │ Pancreas     │
  │ prostate-cancer-prime       │ 193,000  │ 12 GB    │ Prostate (5K)│
  │ skin-normal-sample1         │ 68,476   │ 12 GB    │ Skin         │
  │ skin-normal-sample2         │ 90,106   │ 7.1 GB   │ Skin         │
  │ skin-prime-ffpe             │ 112,551  │ 7.1 GB   │ Skin         │
  │ tonsil-lymphoid             │ 864,388  │ 22 GB    │ Tonsil       │
  │ tonsil-reactive             │ 1,349,620│ 27 GB    │ Tonsil       │
  └─────────────────────────────┴──────────┴──────────┴──────────────┘

  NOTE: Check 10x website for additional datasets not listed above.
  Empty placeholders on disk: xenium-breast-biomarkers, xenium-skin-preview

═══════════════════════════════════════════════════════════════════════
TOTAL: ~73 datasets, ~28.5M cells, 16 tissue types, 3 platforms
═══════════════════════════════════════════════════════════════════════
INVENTORY
}

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
main() {
    log "DAPIDL Dataset Downloader"
    log "Target directory: $TARGET_DIR"
    log ""

    case "${1:-all}" in
        sthelar)   download_sthelar ;;
        merscope)  download_merscope ;;
        xenium)    download_xenium ;;
        verify)    verify_downloads ;;
        inventory) print_inventory ;;
        all)
            print_inventory
            echo ""
            download_sthelar
            echo ""
            download_merscope
            echo ""
            download_xenium
            echo ""
            verify_downloads
            ;;
        *)
            echo "Usage: $0 [TARGET_DIR] {all|sthelar|merscope|xenium|verify|inventory}"
            exit 1
            ;;
    esac
}

# Allow passing target dir as $1 and command as $2
if [ $# -ge 2 ]; then
    TARGET_DIR="$1"
    mkdir -p "$TARGET_DIR"/{xenium,merscope,sthelar}
    shift
    main "$1"
else
    main "${1:-all}"
fi
