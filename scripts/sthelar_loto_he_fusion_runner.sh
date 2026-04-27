#!/bin/bash
# Master runner for the H&E LOTO sweep + proper multimodal fusion.
#
# Sequence:
#   1. H&E LOTO sweep — 16 tissues, 2 concurrent (~30h wall)
#   2. Update DAPI vs H&E LOTO comparison
#   3. Fusion model — single run, full HE-intersection (~10h)
#   4. Update modality 4-way comparison (DAPI / H&E / both-naive / fusion)
#
# Logs go to /tmp/sthelar_loto_he_*.log and /tmp/sthelar_modality_fusion.log.

set -o pipefail
cd /mnt/work/git/dapidl

OUT=/mnt/work/git/dapidl/pipeline_output
LOGS=/tmp

# 1. H&E LOTO sweep (16 tissues, 2 concurrent)
echo "=================================================="
echo "[$(date '+%F %T')] PHASE 1: H&E LOTO sweep"
echo "=================================================="
bash scripts/sthelar_loto_he_all.sh 2>&1 | tee "$LOGS/sthelar_loto_he_all.log"

# 2. Comparison: DAPI vs H&E LOTO
echo "=================================================="
echo "[$(date '+%F %T')] PHASE 2: LOTO modality comparison"
echo "=================================================="
uv run python scripts/sthelar_loto_modality_compare.py 2>&1 | tee "$LOGS/sthelar_loto_modality_compare.log"

# 3. Wait for any stragglers + GPU to clear
sleep 30
while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    running=$(ps aux | grep -E "sthelar.*\.py" | grep -v grep | grep -v fusion_runner | grep -v loto_modality_compare | wc -l)
    if [ "${free:-0}" -ge 18000 ] && [ "${running:-0}" -eq 0 ]; then
        echo "[$(date '+%F %T')] GPU idle (${free} MiB), proceeding to fusion"
        break
    fi
    sleep 60
done

# 4. Fusion model training
echo "=================================================="
echo "[$(date '+%F %T')] PHASE 3: Multimodal fusion training"
echo "=================================================="
uv run python scripts/sthelar_modality_fusion.py \
    --output "$OUT/sthelar_modality_fusion" \
    --epochs 21 --patience 8 --batch-size 48 --lr 1e-4 --num-workers 6 \
    --num-fusion-layers 2 --num-heads 8 \
    2>&1 | tee "$LOGS/sthelar_modality_fusion.log"

# 5. Update modality comparison (DAPI / H&E / both / fusion)
echo "=================================================="
echo "[$(date '+%F %T')] PHASE 4: Update modality comparison"
echo "=================================================="
uv run python scripts/sthelar_modality_compare.py 2>&1 | tee "$LOGS/sthelar_modality_compare_v2.log"

echo "=================================================="
echo "[$(date '+%F %T')] ALL PHASES COMPLETE"
echo "=================================================="
