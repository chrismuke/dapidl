"""Pull p64 QC review verdicts from Label Studio (DAPIDL project) and summarize.

Fetches Henrik's annotations via the LS API (JSON-MIN), joins by `row_idx`, and writes
verdicts.parquet + prints a grade-correction confusion + flag histogram + comments.

Usage (run from the p64 worktree):
    source ~/.zshrc.local
    uv run --with requests python scripts/qc_review_ls_pull.py --project-id 5
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import polars as pl
import requests

B = os.environ["LABEL_STUDIO_URL"].rstrip("/")
H = {"Authorization": f"Token {os.environ['LABEL_STUDIO_DAPIDL_API_TOKEN']}"}
OUTDIR = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1/qc/review_ls")


def _text(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", type=int, required=True)
    ap.add_argument("--out", type=Path, default=OUTDIR / "verdicts.parquet")
    a = ap.parse_args()

    r = requests.get(f"{B}/api/projects/{a.project_id}/export", headers=H,
                     params={"exportType": "JSON_MIN"})
    r.raise_for_status()
    recs = r.json()

    rows = []
    for d in recs:
        flag = d.get("flag") or []
        if isinstance(flag, str):
            flag = [flag]
        rows.append({
            "row_idx": d.get("row_idx"), "slide": d.get("slide"),
            "cell_class": d.get("cell_class"), "assigned_group": d.get("assigned_group"),
            "flags": json.dumps(flag), "correct_group": _text(d.get("correct_group")),
            "comment": _text(d.get("comment")), "reviewer": d.get("annotator"),
        })

    if not rows:
        print("[pull] no annotations yet — Henrik has not reviewed anything.")
        return

    df = pl.DataFrame(rows)
    df.write_parquet(a.out)
    flagged = df.filter(pl.col("flags") != "[]")
    print(f"[pull] reviewed {df.height} tasks; flagged {flagged.height} -> {a.out}")

    counts = Counter(f for fl in df["flags"] for f in json.loads(fl))
    print(f"[pull] flag counts: {dict(counts)}")

    wrong = df.filter(pl.col("correct_group").is_not_null())
    if wrong.height:
        print("[pull] grade corrections (grader assigned -> Henrik says):")
        print(wrong.group_by(["assigned_group", "correct_group"]).len().sort("len", descending=True))

    cm = df.filter(pl.col("comment").is_not_null() & (pl.col("comment") != ""))
    print(f"[pull] {cm.height} comments:")
    for row in cm.select(["row_idx", "assigned_group", "comment"]).head(40).iter_rows(named=True):
        print(f"    row {row['row_idx']} [{row['assigned_group']}]: {row['comment']}")


if __name__ == "__main__":
    main()
