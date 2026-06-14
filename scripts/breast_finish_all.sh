#!/usr/bin/env bash
# One-shot finisher: complete every remaining piece sequentially in a single log.
# - p256 STHELAR training is considered DONE at ep09 (best_model.pt already saved).
# - Steps continue on individual failures (set -u, no -e).

set -uo pipefail
cd /mnt/work/git/dapidl

LOG_DIR=/tmp
STAMP() { date '+[%Y-%m-%d %H:%M:%S]'; }
banner() { echo; echo "$(STAMP) ===== $* ====="; echo; }

banner "STEP 1/8: SKIPPED — breast_s6 runs in separate detached job (see breast_s6_only.sh)"

banner "STEP 2/8: rebuild master_metrics from existing metrics_*.json (5 datasets)"
uv run python scripts/breast_remerge_master_metrics.py \
    || echo "$(STAMP) WARN: remerge failed"

banner "STEP 3/8: cross-source eval (8 directions: STHELAR↔Xenium × p32/64/128/256)"
bash scripts/breast_cross_source_run_all.sh \
    || echo "$(STAMP) WARN: cross-source sweep had failures"

banner "STEP 4/8: aggregate breast patch-size metrics"
uv run python scripts/breast_size_compare.py \
    || echo "$(STAMP) WARN: size compare failed"

banner "STEP 5/8: consensus combinations (k=2..8 over top-10 methods per dataset; 5 datasets — s6 added later by breast_s6_only.sh)"
uv run python scripts/breast_consensus_combinations.py \
    --datasets rep1,rep2,breast_s0,breast_s1,breast_s3 \
    --top-n 10 --max-k 8 \
    || echo "$(STAMP) WARN: consensus combos failed"

banner "STEP 6/8: render size sweep figures (S1-S4)"
uv run python scripts/figures/breast_size_figs.py \
    || echo "$(STAMP) WARN: size figs failed"

banner "STEP 7/8: render cross-source + annotation figures (C1-C3, A1-A5)"
uv run python scripts/figures/breast_cross_source_figs.py \
    || echo "$(STAMP) WARN: cross-source figs failed"
uv run python scripts/figures/annotation_eval_figs.py \
    || echo "$(STAMP) WARN: annotation figs failed"

banner "STEP 8/8: sync to obsidian vault"
bash scripts/sync_manuscript_to_obsidian.sh \
    || echo "$(STAMP) WARN: obsidian sync failed"

banner "ALL DONE"
df -h /mnt/work | tail -2
echo "$(STAMP) Outputs:"
echo "  - pipeline_output/breast_annotation_full/  (master_metrics.{parquet,csv,md})"
echo "  - pipeline_output/breast_cross_source/     (8 cross-source eval JSONs)"
echo "  - pipeline_output/breast_size_sweep/       (size_metrics.{parquet,md})"
echo "  - pipeline_output/breast_annotation_full/consensus_top50.md"
echo "  - ~/obsidian/llmbrain/DAPIDL/Pipeline-Deep-Eval-20260501/"
