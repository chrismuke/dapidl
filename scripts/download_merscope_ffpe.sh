#!/usr/bin/env bash
# Download all MERSCOPE FFPE IO datasets from Vizgen GCS bucket.
# Only downloads files needed for DAPIDL: cell_by_gene.csv, cell_metadata.csv,
# DAPI z3 (middle z-plane), and the coordinate transform.
#
# Usage: ./scripts/download_merscope_ffpe.sh [TARGET_DIR]
# Default: ~/datasets/raw/merscope

set -euo pipefail

TARGET="${1:-$HOME/datasets/raw/merscope}"
BUCKET="gs://vz-ffpe-showcase"
LOG_DIR="/tmp/merscope_downloads"
mkdir -p "$TARGET" "$LOG_DIR"

# Map GCS folder → local folder name
declare -A DATASETS=(
    ["HumanColonCancerPatient1"]="merscope-colon-cancer-1"
    ["HumanColonCancerPatient2"]="merscope-colon-cancer-2"
    ["HumanLiverCancerPatient1"]="merscope-liver-cancer-1"
    ["HumanLiverCancerPatient2"]="merscope-liver-cancer-2"
    ["HumanLungCancerPatient1"]="merscope-lung-cancer-1"
    ["HumanLungCancerPatient2"]="merscope-lung-cancer-2"
    ["HumanOvarianCancerPatient1"]="merscope-ovarian-cancer-1"
    ["HumanOvarianCancerPatient2Slice1"]="merscope-ovarian-cancer-2"
    ["HumanOvarianCancerPatient2Slice2"]="merscope-ovarian-cancer-3"
    ["HumanOvarianCancerPatient2Slice3"]="merscope-ovarian-cancer-4"
    ["HumanProstateCancerPatient1"]="merscope-prostate-cancer-1"
    ["HumanProstateCancerPatient2"]="merscope-prostate-cancer-2"
    ["HumanUterineCancerPatient1"]="merscope-uterine-cancer"
)

download_one() {
    local gcs_name="$1"
    local local_name="$2"
    local dest="$TARGET/$local_name"
    local log="$LOG_DIR/${local_name}.log"

    if [ -f "$dest/cell_metadata.csv" ] && [ -f "$dest/images/mosaic_DAPI_z3.tif" ]; then
        echo "[SKIP] $local_name — already complete"
        return 0
    fi

    mkdir -p "$dest/images"
    echo "[START] $local_name → $dest" | tee "$log"

    # Download metadata files (small, fast)
    gsutil -q cp "$BUCKET/$gcs_name/cell_metadata.csv" "$dest/" 2>>"$log"
    gsutil -q cp "$BUCKET/$gcs_name/cell_by_gene.csv" "$dest/" 2>>"$log"
    gsutil -q cp "$BUCKET/$gcs_name/images/micron_to_mosaic_pixel_transform.csv" "$dest/images/" 2>>"$log" || true
    echo "[META] $local_name — metadata downloaded" >> "$log"

    # Download DAPI z3 (middle z-plane, ~17GB)
    gsutil -q cp "$BUCKET/$gcs_name/images/mosaic_DAPI_z3.tif" "$dest/images/" 2>>"$log"
    echo "[DAPI] $local_name — DAPI z3 downloaded" >> "$log"

    # Download cell boundaries if available
    gsutil -q -m cp -r "$BUCKET/$gcs_name/cell_boundaries/" "$dest/cell_boundaries/" 2>>"$log" || true

    echo "[DONE] $local_name" | tee -a "$log"
}

echo "═══════════════════════════════════════════════════════════════"
echo "MERSCOPE FFPE IO Downloader — 13 datasets from GCS"
echo "Target: $TARGET"
echo "Bucket: $BUCKET"
echo "═══════════════════════════════════════════════════════════════"

# Launch all downloads in parallel (3 at a time)
PIDS=()
NAMES=()
ACTIVE=0
MAX_PARALLEL=3

for gcs_name in "${!DATASETS[@]}"; do
    local_name="${DATASETS[$gcs_name]}"

    # Skip if already complete
    if [ -f "$TARGET/$local_name/cell_metadata.csv" ] && [ -f "$TARGET/$local_name/images/mosaic_DAPI_z3.tif" ]; then
        echo "[SKIP] $local_name"
        continue
    fi

    # Wait if max parallel reached
    while [ $ACTIVE -ge $MAX_PARALLEL ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" || true
                unset "PIDS[$i]"
                unset "NAMES[$i]"
                ACTIVE=$((ACTIVE - 1))
            fi
        done
        [ $ACTIVE -ge $MAX_PARALLEL ] && sleep 5
    done

    download_one "$gcs_name" "$local_name" &
    PIDS+=($!)
    NAMES+=("$local_name")
    ACTIVE=$((ACTIVE + 1))
    echo "[QUEUED] $local_name (PID $!)"
done

# Wait for all remaining
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All downloads complete. Verifying..."
echo "═══════════════════════════════════════════════════════════════"

# Verify
for gcs_name in "${!DATASETS[@]}"; do
    local_name="${DATASETS[$gcs_name]}"
    dest="$TARGET/$local_name"
    if [ -f "$dest/cell_metadata.csv" ] && [ -f "$dest/images/mosaic_DAPI_z3.tif" ]; then
        size=$(du -sh "$dest" | cut -f1)
        echo "[OK] $local_name ($size)"
    else
        echo "[MISSING] $local_name"
    fi
done
