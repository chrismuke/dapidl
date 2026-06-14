#!/usr/bin/env bash
# Resume orchestrator v2 — fixed pgrep check.
# Polls /tmp/breast_p256.log for "Final:" line. If log mtime stops advancing
# for 30+ minutes, declare training dead and abort.

set -uo pipefail
cd /mnt/work/git/dapidl

P256_LOG=/tmp/breast_p256.log
STHELAR_LMDB_LOG=/tmp/sthelar_lmdb_build.log
STALL_THRESHOLD=1800   # 30 min

echo "[$(date)] Waiting for p256 training to finish..."
while true; do
    if grep -qE "^Final:" "${P256_LOG}" 2>/dev/null; then
        echo "[$(date)] p256 done"
        grep "^Final:" "${P256_LOG}"
        break
    fi
    # Stall check
    if [ -f "${P256_LOG}" ]; then
        mtime=$(stat -c %Y "${P256_LOG}")
        now=$(date +%s)
        age=$((now - mtime))
        if [ "${age}" -gt "${STALL_THRESHOLD}" ]; then
            echo "[$(date)] p256 log stalled for ${age}s — aborting"
            exit 2
        fi
    fi
    sleep 180
done

echo "[$(date)] Verifying STHELAR LMDB builds..."
if ! grep -q "ALL STHELAR LMDBs BUILT" "${STHELAR_LMDB_LOG}" 2>/dev/null; then
    echo "[$(date)] WARN: STHELAR LMDB build hasn't logged completion — checking files..."
    for sz in 32 64 128 256; do
        if [ ! -f "/mnt/work/datasets/derived/breast-sthelar-only-dapi-p${sz}/labels.npy" ]; then
            echo "  MISSING p${sz}!"
            exit 3
        fi
    done
    echo "  All 4 STHELAR LMDBs present despite missing log line — proceeding"
fi

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
