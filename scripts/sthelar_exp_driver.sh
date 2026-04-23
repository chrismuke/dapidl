#!/bin/bash
# Sequential driver for Exp 4 -> 2 -> 1 -> 3.
# Each experiment runs to completion before the next starts.
# Assumes Exp 5 is already running (or done).

set -o pipefail
cd /mnt/work/git/dapidl

OUT=/mnt/work/git/dapidl/pipeline_output
LOGS=/tmp
DATA=/mnt/work/datasets/derived/sthelar-multitissue-p128

wait_for_gpu() {
    # Poll until GPU has >= 20 GB free (i.e. no other training is running)
    echo "[$(date '+%F %T')] waiting for GPU to be idle..."
    while true; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
        if [ "${free:-0}" -ge 20000 ]; then
            echo "[$(date '+%F %T')] GPU idle (${free} MiB free), proceeding"
            return 0
        fi
        sleep 60
    done
}

log_and_run() {
    local name=$1; shift
    wait_for_gpu
    echo "=================================================="
    echo "[$(date '+%F %T')] START $name"
    echo "CMD: $*"
    echo "=================================================="
    "$@" 2>&1 | tee "$LOGS/sthelar_$name.log"
    local rc=$?
    echo "[$(date '+%F %T')] END $name (exit=$rc)"
    return $rc
}

# ---------- Exp 4: ViT/DINOv2 backbone swap ----------
log_and_run "exp4_vit" \
    uv run dapidl train \
        -d "$DATA" \
        -o "$OUT/sthelar_exp4_vit" \
        --epochs 10 --batch-size 64 --lr 1e-4 \
        --backbone vit_small_patch16_224.dino \
        --backend dali-lmdb --max-weight-ratio 10.0 --no-wandb

log_and_run "exp4_vit_analysis" \
    uv run python scripts/sthelar_exp_analysis.py \
        --model-dir "$OUT/sthelar_exp4_vit"

# ---------- Exp 2: Heavy augmentation ----------
log_and_run "exp2_heavy_aug" \
    uv run dapidl train \
        -d "$DATA" \
        -o "$OUT/sthelar_exp2_heavy_aug" \
        --epochs 10 --batch-size 64 --lr 1e-4 \
        --backbone efficientnetv2_rw_s \
        --backend pytorch --heavy-aug \
        --max-weight-ratio 10.0 --no-wandb

log_and_run "exp2_heavy_aug_analysis" \
    uv run python scripts/sthelar_exp_analysis.py \
        --model-dir "$OUT/sthelar_exp2_heavy_aug"

# ---------- Exp 1: Hierarchical-lite ----------
log_and_run "exp1_hierarchical" \
    uv run python scripts/sthelar_exp1_hierarchical.py \
        --output "$OUT/sthelar_exp1_hierarchical" \
        --epochs 10 --batch-size 64 --lr 1e-4 --coarse-weight 0.3

# (Exp 1 self-saves metrics, no separate analysis step)

# ---------- Exp 3: LOTO (brain held out) ----------
log_and_run "exp3_loto_brain" \
    uv run python scripts/sthelar_exp3_loto.py \
        --output "$OUT/sthelar_exp3_loto_brain" \
        --holdout-tissue brain \
        --epochs 10 --batch-size 64 --lr 1e-4

# (Exp 3 self-saves metrics, no separate analysis step)

echo "=================================================="
echo "[$(date '+%F %T')] ALL DONE"
echo "=================================================="
