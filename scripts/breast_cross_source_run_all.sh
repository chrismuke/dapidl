#!/usr/bin/env bash
# Run all 8 cross-source evaluations (4 sizes × 2 directions).
#
# Direction A: STHELAR-trained (breast_dapi_sthelar_p<size>) → tested on Xenium LMDB
# Direction B: Xenium-trained  (breast_dapi_p<size>)         → tested on STHELAR LMDB

set -euo pipefail

cd /mnt/work/git/dapidl

PATCH_SIZES=(32 64 128 256)

for SIZE in "${PATCH_SIZES[@]}"; do
    XENIUM_CKPT="pipeline_output/breast_dapi_p${SIZE}/best_model.pt"
    STHELAR_CKPT="pipeline_output/breast_dapi_sthelar_p${SIZE}/best_model.pt"
    XENIUM_LMDB="breast-multisource-dapi-p${SIZE}"
    STHELAR_LMDB="breast-sthelar-only-dapi-p${SIZE}"

    # Direction A: STHELAR → Xenium
    if [ -f "${STHELAR_CKPT}" ]; then
        echo "===== A: STHELAR-trained p${SIZE} → Xenium rep1+rep2 ====="
        uv run python scripts/breast_cross_source_eval.py \
            --train-source sthelar --test-source xenium \
            --patch-size "${SIZE}" \
            --ckpt "${STHELAR_CKPT}" \
            --eval-lmdb "${XENIUM_LMDB}"
    else
        echo "  SKIP A p${SIZE}: ${STHELAR_CKPT} missing"
    fi

    # Direction B: Xenium → STHELAR
    if [ -f "${XENIUM_CKPT}" ]; then
        echo "===== B: Xenium-trained p${SIZE} → STHELAR breast ====="
        uv run python scripts/breast_cross_source_eval.py \
            --train-source xenium --test-source sthelar \
            --patch-size "${SIZE}" \
            --ckpt "${XENIUM_CKPT}" \
            --eval-lmdb "${STHELAR_LMDB}"
    else
        echo "  SKIP B p${SIZE}: ${XENIUM_CKPT} missing"
    fi
done

echo "Cross-source sweep complete."
