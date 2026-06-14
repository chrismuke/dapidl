#!/usr/bin/env bash
# Run the breast-DAPI patch-size sweep (sequential to share the GPU).
#
# Prereqs:
#   - LMDBs built via scripts/breast_dapi_lmdb.py at p32, p64, p128, p256
#   - GPU free (check nvidia-smi)
#
# Trains 4 EfficientNetV2-S models, identical settings, only patch_size differs.

set -euo pipefail

cd /mnt/work/git/dapidl

PATCH_SIZES=(32 64 128 256)
EPOCHS=21

for SIZE in "${PATCH_SIZES[@]}"; do
    OUT_DIR="pipeline_output/breast_dapi_p${SIZE}"
    LMDB_NAME="breast-multisource-dapi-p${SIZE}"

    if [ -f "${OUT_DIR}/best_model.pt" ]; then
        echo "===== p${SIZE}: SKIP (best_model.pt already exists) ====="
        continue
    fi

    echo "===== p${SIZE}: training in ${OUT_DIR} ====="
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv | head -2
    uv run python scripts/breast_dapi_train.py \
        --patch-size "${SIZE}" \
        --classes 4 \
        --lmdb "${LMDB_NAME}" \
        --output "${OUT_DIR}" \
        --epochs "${EPOCHS}"
    echo "===== p${SIZE}: done ====="
done

echo "All sizes trained. Run scripts/breast_size_compare.py to aggregate."
