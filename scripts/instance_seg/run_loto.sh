#!/usr/bin/env bash
# Phase D LOTO: leave-one-tissue-out for the chosen architecture.
# Trains 4× on 3 train slides + 1 test slide, reports per-fold metrics.
set -euo pipefail

CACHE_ROOT=${CACHE_ROOT:-/mnt/work/datasets/derived/sthelar_breast_tiles}
ARCH=${ARCH:-cellotype}
OUT_ROOT=${OUT_ROOT:-pipeline_output/instance_seg/loto}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}

SLIDES=(breast_s0 breast_s1 breast_s3 breast_s6)

mkdir -p "$OUT_ROOT"

for held_out in "${SLIDES[@]}"; do
    train=()
    for s in "${SLIDES[@]}"; do
        if [[ "$s" != "$held_out" ]]; then
            train+=("$s")
        fi
    done
    run_name="loto_${held_out}"
    echo ">>> LOTO fold: held-out = ${held_out}, train = ${train[*]}"
    echo ">>> output → $OUT_ROOT/$run_name"
    uv run python "scripts/instance_seg/train_${ARCH}.py" \
        --tile-cache "$CACHE_ROOT" \
        --train-slides "${train[@]}" \
        --test-slide "$held_out" \
        --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" \
        --out "$OUT_ROOT/$run_name" \
        --run-name "$run_name" \
        --wandb-group loto \
        || { echo "FAIL on fold $held_out"; exit 1; }
done

echo "all 4 LOTO folds complete"
