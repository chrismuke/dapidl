#!/usr/bin/env bash
# Scenario E: "max-pool minus Prime" — overnight wrapper, long mode.
#
# Train: Janesick rep1+rep2 + STHELAR s0+s1+s3 (5 sources, 2 platforms)
# Test:  STHELAR s6 (Xenium Prime, fully held out)
#
# Extends the A/B/C/D matrix with a held-out-platform-generation test —
# directly answers "what if we never saw Prime at training time?"
#
# Long mode: --epochs 30 --patience 8 (matches Scenario A).
# Runs coarse then medium sequentially on the local 3090 (~6h total).
set -e
cd /mnt/work/git/dapidl

OUT_BASE="pipeline_output/breast_pooled_2026_05"
LOG_BASE="/tmp/dapidl_logs"
mkdir -p "$OUT_BASE" "$LOG_BASE"

TRAIN="xenium_rep1,xenium_rep2,sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3"
TEST="sthelar_breast_s6"

run_e() {
    local NAME=$1
    local TIER_FLAG=$2
    local OUT="$OUT_BASE/$NAME"
    echo ""
    echo "============================================================"
    echo "[$(date)] $NAME: train=$TRAIN  test=$TEST  $TIER_FLAG"
    echo "============================================================"
    if [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: summary.json already exists, skipping"
        return
    fi
    mkdir -p "$OUT"
    uv run python scripts/breast_pooled_train.py \
        $TIER_FLAG \
        --train-sources "$TRAIN" --test-sources "$TEST" \
        --output "$OUT" --epochs 30 --patience 5 \
        2>&1 | tee "$LOG_BASE/${NAME}.log"
}

run_e "E_full_pool_to_prime"        ""
run_e "E_full_pool_to_prime_medium" "--tier medium"

echo ""
echo "============================================================"
echo "[$(date)] Scenario E complete"
echo "============================================================"
ls -la "$OUT_BASE/E_"*/summary.json 2>/dev/null
