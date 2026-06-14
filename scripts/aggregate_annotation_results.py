#!/usr/bin/env python3
"""Aggregate every existing annotation-method result into one parquet table.

Walks pipeline_output/sthelar_pipeline/ for popV / popV-DISCO / OnClass /
BANKSY / Tangram / scANVI results on STHELAR breast_s0, plus any
finegrained_3method_comparison.json files. Produces a unified
{by_method,by_dataset,by_class}.parquet schema for figure scripts to consume.

Output: pipeline_output/annotation_eval_2026_05/{by_method,by_dataset,by_class}.parquet
        pipeline_output/annotation_eval_2026_05/summary.md

Usage:
    uv run python scripts/aggregate_annotation_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "annotation_eval_2026_05"
OUT.mkdir(parents=True, exist_ok=True)

PIPELINE_DIR = ROOT / "sthelar_pipeline"

# Files in sthelar_pipeline/ named like <method>_<slide>.json
SINGLE_RESULT_FILES = {
    "popv_breast_s0": ("popV", "breast_s0", "label1"),
    "popv_finegrained_breast_s0": ("popV", "breast_s0", "fine"),
    "popv_full_retrain_breast_s0": ("popV-retrained", "breast_s0", "label1"),
    "tangram_disco_breast_s0": ("Tangram-DISCO", "breast_s0", "label1"),
    "banksy_r2.0_breast_s0": ("BANKSY-r2.0", "breast_s0", "label1"),
    "banksy_r3.0_breast_s0": ("BANKSY-r3.0", "breast_s0", "label1"),
    "banksy_r2.0_combined_breast_s0": ("BANKSY-r2.0+sctype", "breast_s0", "label1"),
    "banksy_r3.0_combined_breast_s0": ("BANKSY-r3.0+sctype", "breast_s0", "label1"),
    "combined_breast_s0": ("combined", "breast_s0", "label1"),
    "results_breast_s0": ("pipeline-results", "breast_s0", "label1"),
}


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _extract_metrics_block(payload: dict, level: str) -> dict | None:
    """Find a `results_label1` / `results_label2` / `metrics` style block."""
    candidates = [
        f"results_{level}",
        "results_label1" if level == "label1" else "results_label2",
        "metrics",
    ]
    for k in candidates:
        if k in payload and isinstance(payload[k], dict):
            return payload[k]
    return payload  # fallback: top-level may BE the metrics block


def _scan_breast_s0_singles() -> tuple[list[dict], list[dict]]:
    by_method: list[dict] = []
    by_class: list[dict] = []
    if not PIPELINE_DIR.exists():
        return by_method, by_class

    for stem, (method, slide, level) in SINGLE_RESULT_FILES.items():
        p = PIPELINE_DIR / f"{stem}.json"
        if not p.exists():
            continue
        payload = _read_json(p)
        if payload is None:
            continue
        block = _extract_metrics_block(payload, level)
        if not block or "accuracy" not in block:
            # results_breast_s0 has nested metrics under cluster_annotations
            if "metrics" in payload:
                block = payload["metrics"]
            else:
                continue
        by_method.append({
            "method": method,
            "dataset": f"sthelar_{slide}",
            "level": level,
            "n_cells": block.get("n_cells") or payload.get("n_cells"),
            "accuracy": block.get("accuracy"),
            "f1_macro": block.get("f1_macro") or block.get("macro_f1"),
            "f1_weighted": block.get("f1_weighted") or block.get("weighted_f1"),
            "elapsed_s": payload.get("elapsed_s"),
            "source_file": str(p.relative_to(ROOT)),
        })
        for cls, m in (block.get("per_class") or {}).items():
            if not isinstance(m, dict):
                continue
            by_class.append({
                "method": method,
                "dataset": f"sthelar_{slide}",
                "level": level,
                "class": cls,
                "f1": m.get("f1"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "support": m.get("support"),
            })
    return by_method, by_class


def _scan_3method_comparison() -> tuple[list[dict], list[dict]]:
    by_method: list[dict] = []
    by_class: list[dict] = []
    p = PIPELINE_DIR / "finegrained_3method_comparison.json"
    if not p.exists():
        return by_method, by_class
    d = _read_json(p)
    if d is None:
        return by_method, by_class

    for key, payload in d.items():
        method = payload.get("method", key)
        for level in ("results_label1", "results_label2"):
            block = payload.get(level)
            if not block:
                # Some payloads have label1 metrics at top level
                if level == "results_label1" and "f1_macro" in payload:
                    block = payload
                else:
                    continue
            level_short = level.replace("results_", "")
            by_method.append({
                "method": method,
                "dataset": "sthelar_breast_s0",
                "level": level_short,
                "n_cells": block.get("n_cells"),
                "accuracy": block.get("accuracy"),
                "f1_macro": block.get("f1_macro"),
                "f1_weighted": block.get("f1_weighted"),
                "elapsed_s": payload.get("elapsed_s"),
                "source_file": str(p.relative_to(ROOT)),
            })
            for cls, m in (block.get("per_class") or {}).items():
                if not isinstance(m, dict):
                    continue
                by_class.append({
                    "method": method,
                    "dataset": "sthelar_breast_s0",
                    "level": level_short,
                    "class": cls,
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "support": m.get("support"),
                })
    return by_method, by_class


def _scan_popv_disco_results() -> tuple[list[dict], list[dict]]:
    """popv_disco_fresh_results.json + popv_disco_results.json keep multiple methods inside."""
    by_method: list[dict] = []
    by_class: list[dict] = []
    for fname in ("popv_disco_fresh_results.json", "popv_disco_results.json"):
        p = PIPELINE_DIR / fname
        if not p.exists():
            continue
        payload = _read_json(p)
        if payload is None:
            continue
        # The file structure looks like a top-level dict of {method_key: payload_dict}
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, dict) and ("results_label1" in v or "f1_macro" in v):
                    method = v.get("method", k)
                    block = v.get("results_label1") or v
                    by_method.append({
                        "method": method,
                        "dataset": "sthelar_breast_s0",
                        "level": "label1",
                        "n_cells": block.get("n_cells"),
                        "accuracy": block.get("accuracy"),
                        "f1_macro": block.get("f1_macro"),
                        "f1_weighted": block.get("f1_weighted"),
                        "elapsed_s": v.get("elapsed_s"),
                        "source_file": str(p.relative_to(ROOT)),
                    })
                    for cls, m in (block.get("per_class") or {}).items():
                        if not isinstance(m, dict):
                            continue
                        by_class.append({
                            "method": method,
                            "dataset": "sthelar_breast_s0",
                            "level": "label1",
                            "class": cls,
                            "f1": m.get("f1"),
                            "precision": m.get("precision"),
                            "recall": m.get("recall"),
                            "support": m.get("support"),
                        })
    return by_method, by_class


def write_summary_md(by_method: pl.DataFrame, by_class: pl.DataFrame) -> None:
    out = OUT / "summary.md"
    lines = [
        "# Annotation Methods — Aggregated Results",
        "",
        f"Aggregated from {ROOT}/sthelar_pipeline/ on `aggregate_annotation_results.py`.",
        "",
        f"- **Methods**: {by_method['method'].n_unique()}",
        f"- **Datasets**: {by_method['dataset'].n_unique()}",
        f"- **Total method-runs**: {len(by_method)}",
        f"- **Per-class rows**: {len(by_class)}",
        "",
        "## Top methods on STHELAR breast_s0 (label1, 9-class CL)",
        "",
    ]

    label1 = by_method.filter(
        (pl.col("level") == "label1") & pl.col("f1_macro").is_not_null()
    ).sort("f1_macro", descending=True)
    cols = ["method", "dataset", "n_cells", "accuracy", "f1_macro",
            "f1_weighted", "elapsed_s"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}"
        return str(v)

    for r in label1.head(20).iter_rows(named=True):
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("## Per-class F1 leaders (top method per class on label1)")
    lines.append("")

    pc = by_class.filter(pl.col("level") == "label1").sort("f1", descending=True)
    leaders = pc.group_by("class").agg(
        pl.col("method").first().alias("best_method"),
        pl.col("f1").first(),
        pl.col("support").first(),
    )
    cols_pc = ["class", "best_method", "f1", "support"]
    lines.append("| " + " | ".join(cols_pc) + " |")
    lines.append("|" + "|".join(["---"] * len(cols_pc)) + "|")
    for r in leaders.sort("f1", descending=True).iter_rows(named=True):
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols_pc) + " |")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> None:
    rows_m: list[dict] = []
    rows_c: list[dict] = []

    for collector in (
        _scan_breast_s0_singles,
        _scan_3method_comparison,
        _scan_popv_disco_results,
    ):
        m, c = collector()
        rows_m.extend(m)
        rows_c.extend(c)

    if not rows_m:
        print("No annotation results found in pipeline_output/sthelar_pipeline/.")
        return

    by_method = pl.DataFrame(rows_m).unique(subset=["method", "dataset", "level"])
    by_class = pl.DataFrame(rows_c).unique(subset=["method", "dataset", "level", "class"])

    by_method.write_parquet(OUT / "by_method.parquet")
    by_class.write_parquet(OUT / "by_class.parquet")
    by_method.write_csv(OUT / "by_method.csv")
    by_class.write_csv(OUT / "by_class.csv")

    print(f"By method: {len(by_method)} rows  →  {OUT / 'by_method.parquet'}")
    print(f"By class: {len(by_class)} rows  →  {OUT / 'by_class.parquet'}")
    print()
    print("Top label1 methods:")
    label1 = by_method.filter(pl.col("level") == "label1").sort(
        "f1_macro", descending=True
    ).head(10).select(["method", "dataset", "f1_macro", "accuracy"])
    print(label1)

    write_summary_md(by_method, by_class)


if __name__ == "__main__":
    main()
