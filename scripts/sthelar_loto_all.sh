#!/bin/bash
# Run LOTO (leave-one-tissue-out) for all 15 remaining tissues.
# brain already done as sthelar_exp3_loto_brain (10 epochs).
#
# 2 concurrent on local GPU. Each run is 6 epochs (baseline LOTO brain
# plateaued around epoch 7–8). 4 DataLoader workers per run to avoid
# I/O thrash with 2 concurrent jobs.

set -o pipefail
cd /mnt/work/git/dapidl

# 15 tissues (brain already done separately)
TISSUES=(
    "bone" "bone_marrow" "breast" "cervix" "colon"
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

    out_dir="$OUT/sthelar_loto_${tissue}"
    log="$LOGS/sthelar_loto_${tissue}.log"

    if [ -f "$out_dir/analysis/summary.json" ]; then
        echo "[$(date '+%F %T')] SKIP loto $tissue (summary.json exists)"
        continue
    fi

    launched=$((launched+1))
    echo "[$(date '+%F %T')] START ($launched/15) loto $tissue -> $log"
    mkdir -p "$out_dir"

    (
        uv run python scripts/sthelar_exp3_loto.py \
            --output "$out_dir" \
            --holdout-tissue "$tissue" \
            --epochs $EPOCHS \
            --num-workers $NUM_WORKERS \
            2>&1 | tee "$log"
        echo "[$(date '+%F %T')] DONE loto $tissue (rc=$?)"
    ) &
    pids+=($!)

    # stagger launches to spread startup GPU-memory spikes
    sleep 45
done

# wait for all remaining
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo "=================================================="
echo "[$(date '+%F %T')] ALL 15 LOTO RUNS DONE"
echo "=================================================="
