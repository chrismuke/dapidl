#!/usr/bin/env bash
# Run the 5-point data-scale curve in sequence on the local GPU.
set -euo pipefail

cd /mnt/work/git/dapidl
OUT=pipeline_output/data_scale_2026_05
mkdir -p "$OUT"

for FRAC in 0.05 0.10 0.25 0.50 1.00; do
    LABEL="frac_$(printf '%03d' "$(python -c "print(int($FRAC*100))")")"
    DEST="$OUT/$LABEL"
    if [[ -f "$DEST/summary.json" ]]; then
        echo "[skip] $DEST already complete"
        continue
    fi
    echo "============================================================"
    echo "[run]  fraction=$FRAC → $DEST"
    echo "============================================================"
    uv run python scripts/sthelar_data_scale.py \
        --fraction "$FRAC" --output "$DEST" \
        --epochs 20 --patience 5 \
        2>&1 | tee "$DEST.log"
done

echo "============================================================"
echo "[done] all fractions written to $OUT"
ls -la "$OUT"/*/summary.json
