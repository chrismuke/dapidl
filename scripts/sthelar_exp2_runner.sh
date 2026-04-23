#!/bin/bash
# Wait for GPU idle, then run Exp 2 (heavy aug, LMDB-compatible).
# Queued after the main driver which runs Exp 1 and Exp 3.

set -o pipefail
cd /mnt/work/git/dapidl

LOGS=/tmp
OUT=/mnt/work/git/dapidl/pipeline_output/sthelar_exp2_heavy_aug

# Wait until main driver has moved past Exp 3 (i.e. until GPU is fully idle)
echo "[$(date '+%F %T')] waiting for GPU idle..."
while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    running=$(ps aux | grep -E "dapidl train|sthelar_exp[0-9]" | grep -v grep | grep -v exp2_runner | wc -l)
    if [ "${free:-0}" -ge 22000 ] && [ "${running:-0}" -eq 0 ]; then
        echo "[$(date '+%F %T')] GPU idle (${free} MiB free, 0 active training), starting Exp 2"
        break
    fi
    sleep 120
done

uv run python scripts/sthelar_exp2_heavy_aug.py \
    --output "$OUT" \
    --epochs 10 --batch-size 64 --lr 1e-4 2>&1 | tee "$LOGS/sthelar_exp2_heavy_aug.log"
