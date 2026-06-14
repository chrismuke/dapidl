#!/usr/bin/env bash
# Master orchestrator: wait for the current Xenium-only sweep, then STHELAR sweep + cross-source eval.
#
# Run this in background:
#   nohup bash scripts/breast_full_pipeline.sh > /tmp/breast_full.log 2>&1 &

set -euo pipefail
cd /mnt/work/git/dapidl

SWEEP_LOG=/tmp/breast_sweep.log

echo "[$(date)] Waiting for Xenium-only sweep (p32/p64/p128/p256) to finish..."
# Wait until all 4 sizes have a "Final:" line (one per size)
while true; do
    n_final=$(grep -c "^Final:" "${SWEEP_LOG}" 2>/dev/null || echo 0)
    if [ "${n_final}" -ge 4 ]; then
        echo "[$(date)] Xenium sweep complete (${n_final} Final lines)"
        break
    fi
    echo "[$(date)] Still waiting... ${n_final}/4 finished"
    sleep 120
done

echo "[$(date)] Starting STHELAR-only LMDB builds + training sweep..."
bash scripts/breast_sthelar_sweep.sh

echo "[$(date)] Starting cross-source eval (8 directions)..."
bash scripts/breast_cross_source_run_all.sh

echo "[$(date)] Aggregating + rendering figures..."
uv run python scripts/breast_size_compare.py
uv run python scripts/figures/breast_size_figs.py
uv run python scripts/figures/breast_cross_source_figs.py

echo "[$(date)] FULL PIPELINE COMPLETE"
df -h /mnt/work | tail -2
