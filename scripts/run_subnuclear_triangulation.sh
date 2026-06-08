#!/usr/bin/env bash
# Subnuclear-structure triangulation driver.
#
# Stage 1 (smoke) measures StarDist throughput on 500 rep2 patches, prints an ETA,
# and STOPS for a human go/no-go. Re-run with RUN_FULL=1 to execute stages 2-4.
#
# Measured throughput on this box is ~3.5 patch/s (StarDist-bound), so a FULL
# 2.28M pass is infeasible (~182 h). Defaults below target an overnight run:
#   REP2_CAP   (default empty = FULL rep2, 113k, ~9 h) -- per-source cap on the rep2 TEST pass.
#                                                          Full rep2 keeps (C) directly comparable
#                                                          to the model's 0.619-on-full-rep2.
#   TRAIN_CAP  (default 10000 -> 50k train total, ~4 h) -- per-source cap on the TRAIN pass
#                                                          (LightGBM saturates well before this).
#   PER_CLASS  (default 750 -> 3k patches, ~15 min)     -- saliency balanced rep2 subset per class.
# For a faster ~6 h first-read set REP2_CAP=20000 (the floor's rep2 test is then a 20k
# sample, slightly less directly comparable to 0.619 -- noted in the readout).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=pipeline_output/subnuclear_2026_06
REP2_CAP="${REP2_CAP:-}"
TRAIN_CAP="${TRAIN_CAP:-10000}"
PER_CLASS="${PER_CLASS:-750}"

echo "### GPU"; nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv

echo "### Stage 1: smoke (500 rep2 patches) -> throughput + ETA"
uv run python scripts/subnuclear_feature_pass.py --sources test --limit 500

if [ "${RUN_FULL:-0}" != "1" ]; then
  cat <<'MSG'

  >>> Smoke complete. Review the [ETA] line above, then launch the full run:
  >>>   RUN_FULL=1 bash scripts/run_subnuclear_triangulation.sh                 # full rep2 (~13 h)
  >>>   RUN_FULL=1 REP2_CAP=20000 bash scripts/run_subnuclear_triangulation.sh  # ~6 h first-read
  >>> Background it:  RUN_FULL=1 nohup bash scripts/run_subnuclear_triangulation.sh > sn.log 2>&1 &

MSG
  exit 0
fi

# Fresh merge: drop any prior per-source parquets (incl. the 500-row smoke file) so
# the concat below cannot double-count or mix a stale rep2 cap with a new one.
echo "### cleaning previous feature parquets for a fresh merge"
rm -f "${OUT}"/seg_features_*.parquet "${OUT}/seg_features.parquet"

REP2_ARG=""
[ -n "$REP2_CAP" ] && REP2_ARG="--max-per-source $REP2_CAP"
echo "### Stage 2a: rep2 feature pass (test set, with masks) cap='${REP2_CAP:-full}'"
uv run python scripts/subnuclear_feature_pass.py --sources test $REP2_ARG --save-masks all
echo "### Stage 2b: train feature pass cap=${TRAIN_CAP}"
uv run python scripts/subnuclear_feature_pass.py --sources train --max-per-source "${TRAIN_CAP}"

echo "### merge per-source parquets -> seg_features.parquet"
uv run python - <<PY
import polars as pl
from pathlib import Path
out = Path("${OUT}")
parts = [pl.read_parquet(p) for p in sorted(out.glob("seg_features_*.parquet"))]
pl.concat(parts, how="vertical_relaxed").write_parquet(out / "seg_features.parquet")
print("merged", sum(len(p) for p in parts), "rows ->", out / "seg_features.parquet")
PY

echo "### Stage 3: information floor (C)"
uv run python scripts/subnuclear_floor.py
echo "### Stage 4: attribution saliency (D)"
uv run python scripts/subnuclear_saliency.py --per-class "${PER_CLASS}"

echo "### READOUT"
uv run python - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
floor = json.loads((out / "floor_metrics.json").read_text())
sal = json.loads((out / "saliency_summary.json").read_text())
ref = floor["effnet_macro_f1"]
print(f"EffNet ref macro-F1            : {ref:.3f}")
print(f"(C) nuc-only floor macro-F1    : {floor['nuc_only']['macro_f1']:.3f}  "
      f"(gap {floor['nuc_only']['vs_effnet_0619']:+.3f}, n_test={floor['nuc_only']['n_test']})")
print(f"(C) nuc+ctx floor  macro-F1    : {floor['nuc_plus_ctx']['macro_f1']:.3f}  "
      f"(gap {floor['nuc_plus_ctx']['vs_effnet_0619']:+.3f})")
print(f"(D) IG concentration (overall) : {sal['overall']['mean_concentration']:.2f}  "
      f"(1=area-proportional, >1 subnuclear-driven, <1 context-driven; n={sal['overall'].get('n', '?')})")
print("    per-class concentration     :",
      {k: round(v["mean_concentration"], 2) for k, v in sal["by_class"].items()})
PY
echo "### Decision rubric: small (C) gap => CNN's subnuclear spatial modelling adds little;"
echo "###   (D)>>1 => subnuclear-driven, ~1 => nucleus ignored, <1 => context-driven."
