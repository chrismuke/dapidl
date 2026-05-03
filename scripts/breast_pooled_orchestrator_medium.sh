#!/usr/bin/env bash
# Medium-tier orchestrator. Runs the same 4 train/test pairs as coarse, but at
# CL Medium granularity (~12 classes). Designed to run IN PARALLEL with
# breast_pooled_orchestrator.sh — both share the GPU so each training takes
# ~1.5x longer but you get coarse + medium results around the same time.

set -e
cd /mnt/work/git/dapidl

LMDB="/mnt/work/datasets/derived/breast-6source-dapi-p128"
OUT_BASE="pipeline_output/breast_pooled_2026_05"
LOG_BASE="/tmp/dapidl_logs"
mkdir -p "$OUT_BASE" "$LOG_BASE"

# Wait for sources.npy (built by coarse orchestrator) AND labels.npy
echo "[$(date)] Medium orchestrator waiting for LMDB + sources.npy..."
while ! [ -f "$LMDB/sources.npy" ] || ! [ -f "$LMDB/labels.npy" ]; do
    sleep 60
done

# Derive labels_medium.npy if not present
if ! [ -f "$LMDB/labels_medium.npy" ]; then
    echo "[$(date)] Deriving labels_medium.npy..."
    uv run python scripts/derive_tier_labels.py --tier medium 2>&1 | tee "$LOG_BASE/derive_medium.log"
fi

# Wait until coarse orchestrator has at least started its first training
# (so we don't fight for GPU during model warmup)
echo "[$(date)] Waiting for coarse training A to begin..."
while ! [ -d "$OUT_BASE/A_janesick_to_sthelar" ]; do
    sleep 30
done
sleep 60  # give coarse a head start

run_training() {
    local NAME=$1
    local TRAIN=$2
    local TEST=$3
    local OUT="$OUT_BASE/${NAME}_medium"
    echo ""
    echo "============================================================"
    echo "[$(date)] MEDIUM $NAME: train=$TRAIN  test=$TEST"
    echo "============================================================"
    if [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: summary.json already exists, skipping"
        return
    fi
    mkdir -p "$OUT"
    uv run python scripts/breast_pooled_train.py \
        --tier medium \
        --train-sources "$TRAIN" --test-sources "$TEST" \
        --output "$OUT" --epochs 30 --patience 8 \
        2>&1 | tee "$LOG_BASE/${NAME}_medium.log"
}

# A: Janesick → STHELAR
run_training "A_janesick_to_sthelar" \
    "xenium_rep1,xenium_rep2" \
    "sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6"

# B: STHELAR standard → Janesick + Prime
run_training "B_sthelar_std_to_janesick_prime" \
    "sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3" \
    "xenium_rep1,xenium_rep2,sthelar_breast_s6"

# C: STHELAR Prime → everything else
run_training "C_sthelar_prime_to_all" \
    "sthelar_breast_s6" \
    "xenium_rep1,xenium_rep2,sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3"

# D: All STHELAR → Janesick
run_training "D_all_sthelar_to_janesick" \
    "sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6" \
    "xenium_rep1,xenium_rep2"

echo ""
echo "============================================================"
echo "[$(date)] All 4 MEDIUM trainings complete"
echo "============================================================"
ls -la "$OUT_BASE"/*_medium/summary.json
