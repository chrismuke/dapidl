#!/usr/bin/env bash
# Skin training orchestrator. Waits for breast C trainings to land their
# summary.json files, then runs skin Coarse + Medium with internal split,
# then re-launches the breast orchestrators to pick up D.
#
# Usage:
#   bash scripts/skin_orchestrator.sh > /tmp/dapidl_logs/skin_orchestrator.log 2>&1 &

set -e
set -o pipefail   # so `uv run ... | tee log` propagates Python's exit code
cd /mnt/work/git/dapidl

LMDB="/mnt/work/datasets/derived/skin-4source-dapi-p128"
OUT_BASE="pipeline_output/skin_pooled_2026_05"
LOG_BASE="/tmp/dapidl_logs"
BREAST_C_COARSE="pipeline_output/breast_pooled_2026_05/C_sthelar_prime_to_all/summary.json"
BREAST_C_MEDIUM="pipeline_output/breast_pooled_2026_05/C_sthelar_prime_to_all_medium/summary.json"

# Source names in sources.npy are 'sthelar_skin_s*' (built by skin_dapi_lmdb.py)
TRAIN_SOURCES_OVERRIDE="sthelar_skin_s1,sthelar_skin_s2,sthelar_skin_s3,sthelar_skin_s4"

mkdir -p "$OUT_BASE" "$LOG_BASE"

echo "[$(date)] skin_orchestrator: waiting for breast C summaries..."
while ! [ -f "$BREAST_C_COARSE" ] || ! [ -f "$BREAST_C_MEDIUM" ]; do
    sleep 30
done
echo "[$(date)] both C summaries present. Proceeding."

# Confirm GPU is free (no breast train processes hogging it)
echo "[$(date)] Waiting until D training processes release the GPU (if any)..."
while pgrep -f "breast_pooled_train.py.*D_all_sthelar" > /dev/null 2>&1; do
    echo "[$(date)] D training still alive, waiting..."
    sleep 30
done
echo "[$(date)] GPU should be free now."

# Skin training — pooled all 4 skin sources, INTERNAL held-out test split
TRAIN_SOURCES="$TRAIN_SOURCES_OVERRIDE"

run_skin() {
    local NAME=$1
    local TIER=$2
    local OUT="$OUT_BASE/$NAME"
    echo ""
    echo "============================================================"
    echo "[$(date)] $NAME ($TIER): pooled $TRAIN_SOURCES, INTERNAL split"
    echo "============================================================"
    if [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: summary.json exists, skipping"
        return
    fi
    mkdir -p "$OUT"
    uv run python scripts/pooled_train.py \
        --lmdb-dir "$LMDB" \
        --train-sources "$TRAIN_SOURCES" \
        --test-sources INTERNAL \
        --tier "$TIER" \
        --output "$OUT" --epochs 30 --patience 8 \
        2>&1 | tee "$LOG_BASE/${NAME}.log"
    # Defensive: orchestrator must fail if training did not produce a summary
    if ! [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: ERROR — no summary.json written, aborting orchestrator"
        exit 1
    fi
}

# Coarse first (quickest, fewer classes)
run_skin "skin_pooled_coarse" "coarse"
# Then Medium
run_skin "skin_pooled_medium" "medium"

echo ""
echo "============================================================"
echo "[$(date)] Skin trainings complete. Re-launching breast orchestrators for D."
echo "============================================================"

# Re-launch the breast orchestrators in background — they'll skip A/B/C summaries
# and run only D_*.
nohup bash scripts/breast_pooled_orchestrator.sh \
    > "$LOG_BASE/coarse_orchestrator_resume.log" 2>&1 &
nohup bash scripts/breast_pooled_orchestrator_medium.sh \
    > "$LOG_BASE/medium_orchestrator_resume.log" 2>&1 &

echo "[$(date)] Breast D orchestrators relaunched. skin_orchestrator done."
ls -la "$OUT_BASE"/*/summary.json 2>/dev/null
