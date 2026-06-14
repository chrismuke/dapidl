#!/usr/bin/env bash
# Detached follow-up: run breast_s6 annotation in isolation (it can grind on
# scType's KNN graph for hours on 692K cells), then re-run downstream consensus
# + figure rendering with all 6 datasets so master tables include s6.

set -uo pipefail
cd /mnt/work/git/dapidl

LOG_DIR=/tmp
STAMP() { date '+[%Y-%m-%d %H:%M:%S]'; }
banner() { echo; echo "$(STAMP) ===== $* ====="; echo; }

banner "S6-1/4: breast_s6 annotation (subsampled to 100k for 62 GB RAM safety)"
uv run python scripts/breast_full_annotation.py \
    --datasets breast_s6 --max-cells 100000 \
    || echo "$(STAMP) WARN: breast_s6 annotation failed"

banner "S6-2/4: rebuild master_metrics from all 6 metrics_*.json"
uv run python scripts/breast_remerge_master_metrics.py \
    || echo "$(STAMP) WARN: remerge failed"

banner "S6-3/4: re-run consensus combinations on all 6 datasets"
uv run python scripts/breast_consensus_combinations.py \
    --datasets rep1,rep2,breast_s0,breast_s1,breast_s3,breast_s6 \
    --top-n 10 --max-k 8 \
    || echo "$(STAMP) WARN: consensus combos failed"

banner "S6-4/4: re-render annotation figures (now includes s6)"
uv run python scripts/figures/annotation_eval_figs.py \
    || echo "$(STAMP) WARN: annotation figs failed"
bash scripts/sync_manuscript_to_obsidian.sh \
    || echo "$(STAMP) WARN: obsidian sync failed"

banner "BREAST_S6 FOLLOW-UP DONE"
