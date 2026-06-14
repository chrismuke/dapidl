#!/usr/bin/env python3
"""Aggregate breast-DAPI patch-size sweep results into a unified table.

Reads pipeline_output/breast_dapi_p{32,64,128,256}/analysis/{summary,per_class}.json
and writes pipeline_output/breast_size_sweep/{size_metrics.parquet, size_metrics.md}.

Usage:
    uv run python scripts/breast_size_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

PIPELINE = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = PIPELINE / "breast_size_sweep"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [32, 64, 128, 256]


def main() -> None:
    rows = []
    pc_rows = []
    for sz in SIZES:
        run_dir = PIPELINE / f"breast_dapi_p{sz}"
        summary = run_dir / "analysis" / "summary.json"
        per_class = run_dir / "analysis" / "per_class_metrics.json"
        if not summary.exists():
            print(f"  p{sz}: missing {summary}")
            continue
        d = json.loads(summary.read_text())
        rows.append({
            "patch_size": sz,
            "best_epoch": d.get("best_epoch"),
            "best_val_macro_f1": d.get("best_val_macro_f1"),
            "test_accuracy": d.get("test_accuracy"),
            "test_macro_f1": d.get("test_macro_f1"),
            "test_weighted_f1": d.get("test_weighted_f1"),
            "n_train": d.get("n_train"),
            "n_val": d.get("n_val"),
            "n_test": d.get("n_test"),
        })
        if per_class.exists():
            pcd = json.loads(per_class.read_text())
            for cls, m in pcd.get("per_class", {}).items():
                pc_rows.append({
                    "patch_size": sz,
                    "class": cls,
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "support": m.get("support"),
                })

    if not rows:
        print("No size sweep results found yet.")
        return

    main_df = pl.DataFrame(rows)
    pc_df = pl.DataFrame(pc_rows) if pc_rows else None

    main_df.write_parquet(OUT / "size_metrics.parquet")
    if pc_df is not None:
        pc_df.write_parquet(OUT / "size_per_class.parquet")

    md_lines = [
        "# Breast DAPI — Patch Size Sweep Results",
        "",
        "| Patch size | Best epoch | Val macro F1 | Test macro F1 | Test weighted F1 | Test acc | n_test |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in main_df.sort("patch_size").iter_rows(named=True):
        md_lines.append(
            f"| {r['patch_size']} | {r['best_epoch']} | "
            f"{r['best_val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | "
            f"{r['test_weighted_f1']:.4f} | {r['test_accuracy']:.4f} | "
            f"{r['n_test']:,} |"
        )

    if pc_df is not None and len(pc_df):
        md_lines.append("")
        md_lines.append("## Per-class F1 by patch size")
        md_lines.append("")
        pivot = pc_df.pivot(values="f1", index="class", on="patch_size")
        cols = [c for c in pivot.columns if c != "class"]
        md_lines.append("| Class | " + " | ".join(f"p{c}" for c in cols) + " |")
        md_lines.append("|---" + "|---:" * len(cols) + "|")
        for r in pivot.iter_rows(named=True):
            md_lines.append(
                f"| {r['class']} | "
                + " | ".join(
                    (f"{r[c]:.3f}" if r[c] is not None else "—") for c in cols
                )
                + " |"
            )

    (OUT / "size_metrics.md").write_text("\n".join(md_lines))
    print(f"Wrote {OUT / 'size_metrics.parquet'}")
    print(f"Wrote {OUT / 'size_metrics.md'}")
    print(main_df)


if __name__ == "__main__":
    main()
