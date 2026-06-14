#!/usr/bin/env python3
"""Rebuild master_metrics.{parquet,csv,md} from per-dataset metrics_*.json files.

Used after running breast_full_annotation.py with --datasets <subset>: reads every
metrics_<ds>.json that exists in pipeline_output/breast_annotation_full/ and emits
the unified master table.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

DIR = Path("/mnt/work/git/dapidl/pipeline_output/breast_annotation_full")
COARSE = ["Endothelial", "Epithelial", "Immune", "Stromal"]


def main() -> None:
    rows = []
    for jf in sorted(DIR.glob("metrics_*.json")):
        ds = jf.stem.replace("metrics_", "")
        method_metrics = json.loads(jf.read_text())
        for method, m in method_metrics.items():
            if "error" in m:
                rows.append({
                    "dataset": ds, "method": method,
                    "macro_f1": None, "accuracy": None,
                    "weighted_f1": None, "runtime_s": None,
                    "error": m["error"],
                    **{f"f1_{c}": None for c in COARSE},
                })
                continue
            row = {
                "dataset": ds, "method": method,
                "macro_f1": m.get("f1_macro"),
                "accuracy": m.get("accuracy"),
                "weighted_f1": m.get("f1_weighted"),
                "runtime_s": m.get("runtime_s"),
                "error": None,
            }
            for c in COARSE:
                row[f"f1_{c}"] = m.get("per_class", {}).get(c, {}).get("f1")
            rows.append(row)

    if not rows:
        print("No metrics_*.json files found.")
        return

    df = pl.DataFrame(rows)
    df.write_parquet(DIR / "master_metrics.parquet")
    df.write_csv(DIR / "master_metrics.csv")

    lines = [
        "# Breast Annotation — Master Metrics (all methods × all breast datasets)",
        "",
        f"**Datasets**: {df['dataset'].n_unique()} | "
        f"**Methods**: {df['method'].n_unique()} | "
        f"**Runs**: {len(df)}",
        "",
        "## Top by macro F1 per dataset",
        "",
    ]
    for ds in sorted(df["dataset"].unique().to_list()):
        sub = df.filter(
            (pl.col("dataset") == ds) & pl.col("macro_f1").is_not_null()
        ).sort("macro_f1", descending=True)
        if not len(sub):
            continue
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Method | Macro F1 | Accuracy | Endo | Epi | Imm | Str | s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sub.iter_rows(named=True):
            def _f(v):
                return f"{v:.3f}" if v is not None else "—"
            lines.append(
                f"| `{r['method']}` | {_f(r['macro_f1'])} | {_f(r['accuracy'])} | "
                f"{_f(r['f1_Endothelial'])} | {_f(r['f1_Epithelial'])} | "
                f"{_f(r['f1_Immune'])} | {_f(r['f1_Stromal'])} | "
                f"{r['runtime_s'] if r['runtime_s'] else '—'} |"
            )
        lines.append("")

    (DIR / "master_metrics.md").write_text("\n".join(lines))
    print(f"Wrote master_metrics.{{parquet,csv,md}} from {len(rows)} rows "
          f"({df['dataset'].n_unique()} datasets × {df['method'].n_unique()} methods)")


if __name__ == "__main__":
    main()
