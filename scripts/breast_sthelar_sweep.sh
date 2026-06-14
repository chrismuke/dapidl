#!/usr/bin/env bash
# STHELAR-breast-only DAPI sweep at all 4 patch sizes.
# Builds LMDBs (capped to ~70K cells/slide for fair comparison with Xenium-only).
# Trains 4 EfficientNetV2-S models. Sequential to share GPU.

set -euo pipefail

cd /mnt/work/git/dapidl

PATCH_SIZES=(32 64 128 256)
EPOCHS=21
MAX_CELLS_PER_SOURCE=70000

# Build LMDBs first (CPU only, ~3-5 min each)
for SIZE in "${PATCH_SIZES[@]}"; do
    LMDB_DIR="/mnt/work/datasets/derived/breast-sthelar-only-dapi-p${SIZE}"
    if [ -f "${LMDB_DIR}/labels.npy" ]; then
        echo "===== p${SIZE} STHELAR-only LMDB exists, SKIP build ====="
        continue
    fi
    MAP_GB=10
    [ "${SIZE}" -ge 256 ] && MAP_GB=50
    [ "${SIZE}" -eq 128 ] && MAP_GB=20
    echo "===== Building STHELAR-only LMDB p${SIZE} (cap ${MAX_CELLS_PER_SOURCE}/slide) ====="
    uv run python scripts/breast_dapi_lmdb.py \
        --patch-size "${SIZE}" \
        --no-xenium \
        --output "breast-sthelar-only-dapi-p${SIZE}" \
        --max-cells-per-source "${MAX_CELLS_PER_SOURCE}" \
        --map-size-gb "${MAP_GB}"
done

# Train one model per size on STHELAR-only LMDB
for SIZE in "${PATCH_SIZES[@]}"; do
    OUT_DIR="pipeline_output/breast_dapi_sthelar_p${SIZE}"
    LMDB_NAME="breast-sthelar-only-dapi-p${SIZE}"

    if [ -f "${OUT_DIR}/best_model.pt" ]; then
        echo "===== STHELAR p${SIZE}: SKIP (best_model.pt already exists) ====="
        continue
    fi

    # Auto-pick batch size based on patch size to avoid OOM on RTX 3090 24GB
    if [ "${SIZE}" -le 64 ]; then
        BATCH=128
    elif [ "${SIZE}" -le 128 ]; then
        BATCH=64
    else
        BATCH=32
    fi
    echo "===== STHELAR p${SIZE}: training (batch=${BATCH}) in ${OUT_DIR} ====="
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv | head -2
    uv run python scripts/breast_dapi_train.py \
        --patch-size "${SIZE}" \
        --classes 4 \
        --lmdb "${LMDB_NAME}" \
        --output "${OUT_DIR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH}" \
        --num-workers 4
    echo "===== STHELAR p${SIZE}: done ====="
done

echo "All STHELAR sizes trained."
