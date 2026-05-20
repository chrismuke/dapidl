#!/usr/bin/env bash
# QC-threshold experiment: does filtering low-QC TRAINING patches help generalization?
#
# Scenario (matches baseline B): train STHELAR std (s0+s1+s3) -> test Janesick
# rep1/rep2 + Prime s6. The TEST set is never filtered, so cross-source transfer
# is comparable across runs. Config matches baseline B: --epochs 18 --patience 5.
#
# Four-way comparison:
#   - unfiltered  : existing pipeline_output/breast_pooled_2026_05/B_sthelar_std_to_janesick_prime
#                   (val 0.630; rep1 0.604 rep2 0.578 s6 0.317)
#   - qc>=0.15    : keep 89.2% of train (drop the worst ~11%)
#   - qc>=0.25    : keep 64.6% of train (drop the worst ~35%) -> QUALITY arm
#   - random 0.6464: keep a random 64.6% of train -> QUANTITY control (isolates
#                    whether any qc>=0.25 effect is from removing BAD patches vs
#                    just FEWER patches)
set -e
set -o pipefail
cd /mnt/work/git/dapidl

OUT_BASE="pipeline_output/qc_threshold_2026_05"
LOG_BASE="/tmp/dapidl_logs"
mkdir -p "$OUT_BASE" "$LOG_BASE"

TRAIN="sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3"
TEST="xenium_rep1,xenium_rep2,sthelar_breast_s6"

run() {
    local NAME=$1; shift
    local OUT="$OUT_BASE/$NAME"
    echo ""
    echo "============================================================"
    echo "[$(date)] $NAME  args: $*"
    echo "============================================================"
    if [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: summary.json exists, skipping"
        return
    fi
    mkdir -p "$OUT"
    uv run python scripts/breast_pooled_train.py \
        --train-sources "$TRAIN" --test-sources "$TEST" \
        --output "$OUT" --epochs 18 --patience 5 "$@" \
        2>&1 | tee "$LOG_BASE/${NAME}.log"
}

run "B_qc015"      --qc-threshold 0.15
run "B_qc025"      --qc-threshold 0.25
run "B_random065"  --random-keep-frac 0.6464

echo ""
echo "============================================================"
echo "[$(date)] QC-threshold experiment complete"
echo "============================================================"
ls -la "$OUT_BASE"/*/summary.json 2>/dev/null
