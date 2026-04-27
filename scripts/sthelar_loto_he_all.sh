#!/bin/bash
# H&E LOTO sweep — 16 tissues, 2 concurrent.
# Mirrors the DAPI LOTO sweep settings: 6 epochs, batch 64, 4 workers per run.

set -o pipefail
cd /mnt/work/git/dapidl

TISSUES=(
    "brain" "bone" "bone_marrow" "breast" "cervix" "colon"
    "heart" "kidney" "liver" "lung" "lymph_node"
    "ovary" "pancreatic" "prostate" "skin" "tonsil"
)

EPOCHS=6
MAX_CONCURRENT=2
NUM_WORKERS=4
OUT=/mnt/work/git/dapidl/pipeline_output
LOGS=/tmp

launched=0
pids=()

wait_slot() {
    while [ ${#pids[@]} -ge $MAX_CONCURRENT ]; do
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        if [ ${#pids[@]} -ge $MAX_CONCURRENT ]; then
            sleep 60
        fi
    done
}

for tissue in "${TISSUES[@]}"; do
    wait_slot

    out_dir="$OUT/sthelar_loto_he_${tissue}"
    log="$LOGS/sthelar_loto_he_${tissue}.log"

    if [ -f "$out_dir/analysis/summary.json" ]; then
        echo "[$(date '+%F %T')] SKIP loto_he $tissue (summary.json exists)"
        continue
    fi

    launched=$((launched+1))
    echo "[$(date '+%F %T')] START ($launched/16) loto_he $tissue -> $log"
    mkdir -p "$out_dir"

    (
        uv run python scripts/sthelar_loto_he.py \
            --output "$out_dir" \
            --holdout-tissue "$tissue" \
            --epochs $EPOCHS \
            --num-workers $NUM_WORKERS \
            2>&1 | tee "$log"
        echo "[$(date '+%F %T')] DONE loto_he $tissue (rc=$?)"
    ) &
    pids+=($!)

    # stagger launches 45s — same as DAPI sweep, avoids GPU-mem spikes
    sleep 45
done

# wait for all remaining
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "=================================================="
echo "[$(date '+%F %T')] ALL 16 H&E LOTO RUNS DONE"
echo "=================================================="
