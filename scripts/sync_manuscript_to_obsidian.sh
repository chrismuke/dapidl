#!/usr/bin/env bash
# Sync the manuscript draft + figures into the Obsidian DAPIDL vault.
#
# Why this exists:
#   The canonical manuscript lives under git at docs/manuscript_draft.md.
#   The Obsidian copy under ~/obsidian/llmbrain/DAPIDL/manuscript/ is a
#   *snapshot* synced via Obsidian Sync to other devices. Symlinks would
#   dangle on every device except this one — Obsidian Sync uploads files,
#   not symlink targets — so we copy.
#
# Run after manuscript edits:
#   bash scripts/sync_manuscript_to_obsidian.sh
#
# What it does:
#   1. Copies docs/figures/*.png -> obsidian/.../manuscript/figures/
#   2. Copies docs/manuscript_draft.md -> obsidian/.../manuscript/manuscript.md,
#      prepending Obsidian YAML frontmatter (tags, project, status, source).
#   3. Updates the modified: timestamp in the frontmatter on every run.
#   4. Reports a one-line diff stat at the end.
#
# Safe to run repeatedly — fully idempotent, overwrites destination.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_MD="$REPO_ROOT/docs/manuscript_draft.md"
SRC_FIG_DIR="$REPO_ROOT/docs/figures"
DST_DIR="$HOME/obsidian/llmbrain/DAPIDL/manuscript"
DST_MD="$DST_DIR/manuscript.md"
DST_FIG_DIR="$DST_DIR/figures"

if [[ ! -f "$SRC_MD" ]]; then
  echo "ERROR: source manuscript not found: $SRC_MD" >&2
  exit 1
fi
if [[ ! -d "$SRC_FIG_DIR" ]]; then
  echo "ERROR: source figures dir not found: $SRC_FIG_DIR" >&2
  exit 1
fi

mkdir -p "$DST_FIG_DIR"

# Sync figures (overwrites; deletes orphans so Obsidian doesn't accumulate stale PNGs)
rsync -a --delete --include="*.png" --exclude="*" "$SRC_FIG_DIR/" "$DST_FIG_DIR/"

# Build the destination markdown: frontmatter + source body
NOW="$(date -Iseconds)"
GIT_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
SRC_LINES="$(wc -l < "$SRC_MD")"

{
  cat <<EOF
---
title: "DAPIDL Manuscript Draft"
note_type: manuscript
project: DAPIDL
tags: [dapidl, manuscript, draft, sthelar]
status: working-draft
source_repo: dapidl
source_path: docs/manuscript_draft.md
source_commit: ${GIT_REV}
synced_from: ${SRC_MD}
synced_at: "${NOW}"
modified: "${NOW}"
---

> [!note] This file is a synced snapshot
> Canonical source: \`dapidl/docs/manuscript_draft.md\` @ commit \`${GIT_REV}\`.
> Edit the source, then re-run \`bash scripts/sync_manuscript_to_obsidian.sh\` from the dapidl repo root.
> Obsidian Sync handles propagation to other devices once this file is updated.

EOF
  cat "$SRC_MD"
} > "$DST_MD"

DST_LINES="$(wc -l < "$DST_MD")"
N_FIGS="$(ls -1 "$DST_FIG_DIR"/*.png 2>/dev/null | wc -l)"
DST_SIZE_KB="$(du -sk "$DST_DIR" | awk '{print $1}')"

echo "✓ Synced manuscript to Obsidian"
echo "  Source:  $SRC_MD ($SRC_LINES lines @ $GIT_REV)"
echo "  Dest:    $DST_MD ($DST_LINES lines incl. frontmatter)"
echo "  Figures: $N_FIGS PNG(s) in $DST_FIG_DIR"
echo "  Total:   ${DST_SIZE_KB} KiB"
