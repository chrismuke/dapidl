#!/bin/bash
# Wait for the H&E LMDB build to complete, then sequentially train:
#   1. H&E-only model
#   2. DAPI+H&E multimodal model
# Same architecture/hyperparameters as the DAPI baseline (which is already
# trained — we'll compare against pipeline_output/sthelar_multitissue_9class).

set -o pipefail
cd /mnt/work/git/dapidl

OUT=/mnt/work/git/dapidl/pipeline_output
LOGS=/tmp
HE_LMDB=/mnt/work/datasets/derived/sthelar-multitissue-p128-he

# 1. Wait for H&E LMDB build (script writes metadata.json at completion)
echo "[$(date '+%F %T')] waiting for H&E LMDB metadata.json..."
while [ ! -f "$HE_LMDB/metadata.json" ]; do
    sleep 60
done
echo "[$(date '+%F %T')] H&E LMDB ready ($(du -sh $HE_LMDB | cut -f1))"

# 2. Wait for GPU to be idle
wait_gpu() {
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
        running=$(ps aux | grep -E "dapidl train|sthelar.*\.py" | grep -v grep | grep -v modality_runner | grep -v he_lmdb_build | wc -l)
        if [ "${free:-0}" -ge 20000 ] && [ "${running:-0}" -eq 0 ]; then
            echo "[$(date '+%F %T')] GPU idle (${free} MiB), proceeding"
            return 0
        fi
        sleep 60
    done
}

# 3. Train H&E only
wait_gpu
echo "=================================================="
echo "[$(date '+%F %T')] START H&E-only training"
echo "=================================================="
uv run python scripts/sthelar_modality_train.py \
    --mode he \
    --output "$OUT/sthelar_modality_he" \
    --epochs 21 --patience 8 --batch-size 64 --lr 1e-4 --num-workers 6 \
    2>&1 | tee "$LOGS/sthelar_modality_he.log"
echo "[$(date '+%F %T')] END H&E-only (rc=$?)"

# 4. Train multimodal DAPI+H&E
wait_gpu
echo "=================================================="
echo "[$(date '+%F %T')] START DAPI+H&E multimodal training"
echo "=================================================="
uv run python scripts/sthelar_modality_train.py \
    --mode both \
    --output "$OUT/sthelar_modality_both" \
    --epochs 21 --patience 8 --batch-size 64 --lr 1e-4 --num-workers 6 \
    2>&1 | tee "$LOGS/sthelar_modality_both.log"
echo "[$(date '+%F %T')] END DAPI+H&E (rc=$?)"

# 5. Run comparison
echo "=================================================="
echo "[$(date '+%F %T')] Running 3-way comparison"
echo "=================================================="
uv run python scripts/sthelar_modality_compare.py 2>&1 | tee "$LOGS/sthelar_modality_compare.log"

echo "=================================================="
echo "[$(date '+%F %T')] ALL DONE"
echo "=================================================="
