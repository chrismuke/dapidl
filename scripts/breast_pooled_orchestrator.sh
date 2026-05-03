#!/usr/bin/env bash
# Orchestrate the 4 cross-source breast trainings.
# Waits for the LMDB build to complete, derives sources.npy from slide_stats,
# then runs the 4 trainings sequentially on the local GPU.

set -e
cd /mnt/work/git/dapidl

LMDB="/mnt/work/datasets/derived/breast-6source-dapi-p128"
OUT_BASE="pipeline_output/breast_pooled_2026_05"
LOG_BASE="/tmp/dapidl_logs"
mkdir -p "$OUT_BASE" "$LOG_BASE"

# 1. Wait for LMDB build to finish
echo "[$(date)] Waiting for LMDB build at $LMDB..."
while ! [ -f "$LMDB/metadata.json" ]; do
    sleep 60
done
echo "[$(date)] LMDB metadata appeared. Verifying..."
if ! [ -f "$LMDB/labels.npy" ]; then
    echo "[$(date)] ERROR: labels.npy missing"
    exit 1
fi

# 2. Derive sources.npy if not present
if ! [ -f "$LMDB/sources.npy" ]; then
    echo "[$(date)] Deriving sources.npy from slide_stats.json..."
    uv run python -c "
import json, numpy as np
from pathlib import Path
d = Path('$LMDB')
stats = json.load(open(d / 'slide_stats.json'))
labels = np.load(d / 'labels.npy')
sources = np.empty(len(labels), dtype=object)
i = 0
for s in stats:
    n = s['n_written']
    sources[i:i+n] = s['source']
    i += n
assert i == len(labels), f'mismatch: {i} vs {len(labels)}'
np.save(d / 'sources.npy', sources)
print(f'wrote {d / \"sources.npy\"}: {len(sources)} cells, {len(set(sources))} unique sources')
print('counts:', dict([(s, int((sources==s).sum())) for s in sorted(set(sources))]))
"
fi

# 3. Run the 4 trainings sequentially
ALL_SLIDES="xenium_rep1,xenium_rep2,sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6"

run_training() {
    local NAME=$1
    local TRAIN=$2
    local TEST=$3
    local OUT="$OUT_BASE/$NAME"
    echo ""
    echo "============================================================"
    echo "[$(date)] $NAME: train=$TRAIN  test=$TEST"
    echo "============================================================"
    if [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: summary.json already exists, skipping"
        return
    fi
    mkdir -p "$OUT"
    uv run python scripts/breast_pooled_train.py \
        --train-sources "$TRAIN" --test-sources "$TEST" \
        --output "$OUT" --epochs 30 --patience 8 \
        2>&1 | tee "$LOG_BASE/${NAME}.log"
}

# A: Janesick → STHELAR (train rep1+rep2, test all 4 STHELAR)
run_training "A_janesick_to_sthelar" \
    "xenium_rep1,xenium_rep2" \
    "sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6"

# For B/C/D: tighter epoch budget so we fit more pairs in the deck window.
# Override via run_training_short.
run_training_short() {
    local NAME=$1
    local TRAIN=$2
    local TEST=$3
    local OUT="$OUT_BASE/$NAME"
    echo ""
    echo "============================================================"
    echo "[$(date)] $NAME (short): train=$TRAIN  test=$TEST"
    echo "============================================================"
    if [ -f "$OUT/summary.json" ]; then
        echo "[$(date)] $NAME: summary.json already exists, skipping"
        return
    fi
    mkdir -p "$OUT"
    uv run python scripts/breast_pooled_train.py \
        --train-sources "$TRAIN" --test-sources "$TEST" \
        --output "$OUT" --epochs 18 --patience 5 \
        2>&1 | tee "$LOG_BASE/${NAME}.log"
}

# B: STHELAR standard → Janesick + Prime (train s0+s1+s3, test rep1/rep2/s6)
run_training_short "B_sthelar_std_to_janesick_prime" \
    "sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3" \
    "xenium_rep1,xenium_rep2,sthelar_breast_s6"

# C: STHELAR Prime alone → everything else (train s6, test rep1/rep2/s0/s1/s3)
run_training_short "C_sthelar_prime_to_all" \
    "sthelar_breast_s6" \
    "xenium_rep1,xenium_rep2,sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3"

# D: All STHELAR → Janesick (train s0+s1+s3+s6, test rep1+rep2)
run_training_short "D_all_sthelar_to_janesick" \
    "sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6" \
    "xenium_rep1,xenium_rep2"

echo ""
echo "============================================================"
echo "[$(date)] All 4 trainings complete"
echo "============================================================"
ls -la "$OUT_BASE"/*/summary.json
