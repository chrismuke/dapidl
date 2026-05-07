#!/usr/bin/env bash
# BANKSY subprocess-isolated orchestrator for the 6 breast slides.
# Calls banksy_breast_worker.py once per slide in a fresh python process.
# After all slides complete, integrates results into annotation_run_2026_05/
# by appending to per_slide/{slide}.json (extra method) and refreshing the
# consensus parquet.
#
# Memory guard (post-2026-05-05 OOM):
#   - Worker estimates BANKSY RAM up front and either subsamples (when
#     --max-cells is set in MAX_CELLS map below) or aborts (exit 3) before
#     calling initialize_banksy.
#   - Optional second layer: systemd-run --user --scope -p MemoryMax=30G
#     wrapping (only when USE_SCOPE=1 AND user systemd bus is healthy).
#     Disabled by default because the user systemd was in State=closing
#     after the prior crash; not all sessions can spawn user scopes.
#
# Usage:
#   bash scripts/banksy_breast_orchestrator.sh > /tmp/dapidl_logs/banksy_breast.log 2>&1 &
#
#   USE_SCOPE=1 bash scripts/banksy_breast_orchestrator.sh   # add systemd MemoryMax wrap
#   FORCE=1     bash scripts/banksy_breast_orchestrator.sh   # re-run slides that already have output

set -e
set -o pipefail
cd /mnt/work/git/dapidl

LOG_BASE="/tmp/dapidl_logs"
PER_SLIDE="pipeline_output/annotation_run_2026_05/per_slide"
mkdir -p "$LOG_BASE" "$PER_SLIDE"

SLIDES=("rep1" "rep2" "breast_s0" "breast_s1" "breast_s3" "breast_s6")

# BANKSY worker now mirrors the main annotation runner's subsample logic
# (cells*genes > 1e9 → 100k cells, seed=42). This keeps cell IDs aligned with
# the saved GT JSONs so the integrate step can join by row index.
#
# Post-mirror RAM estimates:
#   rep1       158k×313  →  6.3 GB ✓
#   rep2       114k×313  →  4.6 GB ✓
#   breast_s0  577k×541  → 23.7 GB ✓ (just under 25 GB)
#   breast_s1  893k×541  → 36.6 GB ✗ (main runner doesn't subsample → BANKSY won't either)
#   breast_s3  366k×541  → 15.0 GB ✓
#   breast_s6  100k×8232 →  7.8 GB ✓ (main runner subsamples)
#
# breast_s1 will pre-flight ABORT (exit 3). To force-run it, pass
# `MAX_CELLS[breast_s1]=N` here — but the resulting predictions will NOT
# align with GT and won't integrate into the metrics parquet.
declare -A MAX_CELLS=()

# Probe whether `systemd-run --user --scope` actually works in this session.
# After an OOMD kill of init.scope, the user manager may be State=closing
# and reject new scopes. Fall back to bare `uv run` when it can't.
SCOPE_CMD=""
if [ "${USE_SCOPE:-0}" = "1" ]; then
    if systemd-run --user --scope --quiet -p MemoryMax=200M /bin/true 2>/dev/null; then
        SCOPE_CMD="systemd-run --user --scope --quiet -p MemoryMax=30G --collect"
        echo "[$(date)] systemd-run --user --scope OK — enabling MemoryMax=30G wrap"
    else
        echo "[$(date)] systemd-run --user unavailable (user bus closing?) — relying on Python-side guard"
    fi
fi

echo "[$(date)] BANKSY breast orchestrator: starting ${#SLIDES[@]} slides"

OK=()
SKIPPED=()
FAILED_OOM=()
FAILED_OTHER=()
PREFLIGHT_ABORT=()

for slide in "${SLIDES[@]}"; do
    out_json="$PER_SLIDE/${slide}_banksy.json"
    if [ -f "$out_json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[$(date)] $slide: already done ($out_json)"
        SKIPPED+=("$slide")
        continue
    fi
    echo ""
    echo "============================================================"
    echo "[$(date)] BANKSY $slide -> $out_json"
    echo "============================================================"

    extra_args=""
    if [ -n "${MAX_CELLS[$slide]:-}" ]; then
        extra_args="--max-cells ${MAX_CELLS[$slide]}"
        echo "[$(date)]   max-cells cap: ${MAX_CELLS[$slide]}"
    fi

    # Each slide gets a fresh python process — no state leak across slides.
    # `set +e` around the call so we capture the exit code without aborting.
    set +e
    $SCOPE_CMD uv run python scripts/banksy_breast_worker.py \
        "$slide" "$out_json" $extra_args \
        2>&1 | tee "$LOG_BASE/banksy_${slide}.log"
    rc=${PIPESTATUS[0]}
    set -e

    case "$rc" in
        0)
            echo "[$(date)] $slide: OK (exit 0)"
            OK+=("$slide")
            ;;
        3)
            echo "[$(date)] $slide: PRE-FLIGHT ABORT — set MAX_CELLS[$slide] to subsample"
            PREFLIGHT_ABORT+=("$slide")
            ;;
        137|139)
            echo "[$(date)] $slide: KILLED (exit $rc — likely OOM or scope MemoryMax kill)"
            FAILED_OOM+=("$slide")
            ;;
        *)
            echo "[$(date)] $slide: FAILED (exit $rc) — continuing"
            FAILED_OTHER+=("$slide")
            ;;
    esac
done

echo ""
echo "============================================================"
echo "[$(date)] All BANKSY slides attempted"
echo "  OK              : ${OK[*]:-(none)}"
echo "  skipped (cached): ${SKIPPED[*]:-(none)}"
echo "  pre-flight abort: ${PREFLIGHT_ABORT[*]:-(none)}"
echo "  OOM-killed      : ${FAILED_OOM[*]:-(none)}"
echo "  other failures  : ${FAILED_OTHER[*]:-(none)}"
echo "============================================================"
ls -la "$PER_SLIDE"/*_banksy.json 2>/dev/null

# Merge BANKSY results into the existing annotation_run_2026_05 outputs.
# Skip integration if no slides produced output (avoid clobbering the parquet
# with an empty merge).
if [ "${#OK[@]}" -eq 0 ] && [ "${#SKIPPED[@]}" -eq 0 ]; then
    echo "[$(date)] No BANKSY outputs to integrate, skipping merge."
    exit 0
fi
echo "[$(date)] Integrating BANKSY into existing annotation results..."
uv run python scripts/banksy_integrate_results.py \
    2>&1 | tee "$LOG_BASE/banksy_integrate.log"

echo "[$(date)] Done."
