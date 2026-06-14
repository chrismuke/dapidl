#!/usr/bin/env bash
# Resume the breast pipeline after p256 OOM recovery.
# Waits for: 1) p256 training, 2) STHELAR LMDB builds, then runs:
#   - STHELAR training sweep
#   - Cross-source eval (8 directions)
#   - Figures
set -euo pipefail
cd /mnt/work/git/dapidl

P256_LOG=/tmp/breast_p256.log
STHELAR_LMDB_LOG=/tmp/sthelar_lmdb_build.log

echo "[$(date)] Waiting for p256 training to finish..."
while ! grep -qE "^Final:" "${P256_LOG}" 2>/dev/null; do
    sleep 120
    if ! ps -p $(pgrep -f "patch-size 256.*breast_dapi_train" 2>/dev/null | head -1) > /dev/null 2>&1; then
        if ! grep -qE "^Final:" "${P256_LOG}" 2>/dev/null; then
            echo "[$(date)] p256 process exited without Final: line — investigate /tmp/breast_p256.log"
            tail -30 "${P256_LOG}"
            exit 2
        fi
    fi
done
echo "[$(date)] p256 done"
grep "^Final:" "${P256_LOG}"

echo "[$(date)] Waiting for STHELAR LMDB builds to finish..."
while ! grep -q "ALL STHELAR LMDBs BUILT" "${STHELAR_LMDB_LOG}" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] STHELAR LMDBs built"

echo "[$(date)] Running STHELAR training sweep..."
bash scripts/breast_sthelar_sweep.sh

echo "[$(date)] Running cross-source eval..."
bash scripts/breast_cross_source_run_all.sh

echo "[$(date)] Aggregating + rendering figures..."
uv run python scripts/breast_size_compare.py
uv run python scripts/figures/breast_size_figs.py
uv run python scripts/figures/breast_cross_source_figs.py

echo "[$(date)] FULL PIPELINE COMPLETE"
df -h /mnt/work | tail -2
