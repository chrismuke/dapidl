#!/usr/bin/env python3
"""Inventory existing breast-DAPI training artifacts.

Queries master_metrics.parquet and by_tissue.parquet (produced by
aggregate_trained_models.py) and writes a markdown summary listing every
checkpoint that has been evaluated on breast tissue, plus disk locations and
F1 scores. This is the "Step 0" check for the patch-size sweep.

Output: pipeline_output/model_eval_2026_05/breast_dapi_inventory.md
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

BASE = Path("/mnt/work/git/dapidl/pipeline_output/model_eval_2026_05")
MASTER = BASE / "master_metrics.parquet"
BY_TISSUE = BASE / "by_tissue.parquet"
OUT = BASE / "breast_dapi_inventory.md"


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def main() -> None:
    master = pl.read_parquet(MASTER)
    by_t = pl.read_parquet(BY_TISSUE)

    # Multi-tissue runs that include a 'breast' row
    breast_in_mt = by_t.filter(pl.col("tissue") == "breast").sort(
        "macro_f1", descending=True
    )

    # Runs with 'breast' in the run name (LOTO + dedicated)
    breast_runs = master.filter(
        pl.col("run").str.contains("breast")
    ).sort("test_macro_f1", descending=True, nulls_last=True)

    lines: list[str] = [
        "# Breast-DAPI Training Inventory",
        "",
        "Audit before launching the patch-size sweep. Lists every existing checkpoint",
        "that has been evaluated on breast tissue, ordered by macro F1.",
        "",
        "## A. Multi-tissue models — breast-tissue test slice",
        "",
        "These models were trained on all 16 STHELAR tissues and evaluated per-tissue.",
        "The 'breast' macro F1 is the in-domain breast performance.",
        "",
        "| Run | Modality | n_test | Accuracy | Breast macro F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in breast_in_mt.iter_rows(named=True):
        modality = (
            "DAPI+HE" if "fusion" in r["run"] or "both" in r["run"]
            else ("HE" if "_he_" in r["run"] or r["run"].endswith("_he") else "DAPI")
        )
        lines.append(
            f"| `{r['run']}` | {modality} | {_fmt(r['n'])} | "
            f"{_fmt(r['accuracy'])} | {_fmt(r['macro_f1'])} |"
        )

    lines += [
        "",
        "## B. Breast-named runs (LOTO + dedicated)",
        "",
        "| Run | Modality | n_classes | Test F1 (overall) | Notes |",
        "|---|---|---:|---:|---|",
    ]
    note_map = {
        "sthelar_loto_breast": "DAPI multi-tissue model with breast HELD OUT",
        "sthelar_loto_he_breast": "HE multi-tissue model with breast HELD OUT",
        "sthelar_dapi_breast_s0": "Single-slide DAPI on STHELAR breast_s0 (legacy)",
        "merscope_immune_merscope_breast": "MERSCOPE breast immune-subtype 10-class",
    }
    for r in breast_runs.iter_rows(named=True):
        note = note_map.get(r["run"], "")
        lines.append(
            f"| `{r['run']}` | {r['modality']} | {_fmt(r['n_classes'])} | "
            f"{_fmt(r['test_macro_f1'])} | {note} |"
        )

    lines += [
        "",
        "## C. Available LMDBs",
        "",
        "From `/mnt/work/datasets/derived/`:",
        "",
        "| LMDB | Patch size | Source | Approx cells |",
        "|---|---:|---|---|",
        "| `sthelar-multitissue-p128` | 128 | STHELAR all 16 tissues | ~1.26M |",
        "| `sthelar-multitissue-p128-he` | 128 | STHELAR all (HE) | ~1.26M (HE-intersection) |",
        "| `sthelar-breast_s0-finegrained-p128` | 128 | STHELAR breast s0 | ~570K |",
        "| `xenium-breast-tumor-rep1-local-finegrained-p128` | 128 | Xenium rep1 | ~167K |",
        "| `xenium-breast-tumor-rep2-local-finegrained-p128` | 128 | Xenium rep2 | ~118K |",
        "",
        "## D. Existing checkpoints (`best_model.pt`)",
        "",
    ]

    pipeline_root = Path("/mnt/work/git/dapidl/pipeline_output")
    ckpt_lines = []
    for run_dir in sorted(pipeline_root.iterdir()):
        ckpt = run_dir / "best_model.pt"
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1024 / 1024
            ckpt_lines.append(
                f"- `{run_dir.name}/best_model.pt` ({size_mb:.0f} MB)"
            )
    lines.extend(ckpt_lines)

    lines += [
        "",
        "## E. Implications for the patch-size sweep",
        "",
        "**What's already established for breast at p128:**",
        "- DAPI multi-tissue model on breast = macro F1 0.332 (rep1/rep2 not yet tested)",
        "- 9-class baseline on breast = 0.387",
        "- 7-class (rare classes dropped) on breast = 0.493",
        "- Fusion on breast = 0.414",
        "- LOTO (breast unseen) DAPI = 0.218 → tissue identity carries +0.17 F1",
        "",
        "**What we still need for the size story:**",
        "- DAPI breast-only training at p32, p64, p256 (have p128 multi-tissue baseline)",
        "- Cross-platform eval: STHELAR-trained model → Xenium rep1/rep2 (ZERO existing measurements)",
        "- Per-class F1 vs patch size to see which classes need bigger context",
        "",
        "**Strategy: don't retrain at p128**, just evaluate the existing `sthelar_modality_dapi`",
        "checkpoint on the new breast-multisource LMDB at p128 to anchor the curve, then train",
        "at p32, p64, p256 from scratch.",
    ]

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    print(f"  {len(breast_in_mt)} multi-tissue runs × breast tissue")
    print(f"  {len(breast_runs)} breast-named runs")
    print(f"  {len(ckpt_lines)} checkpoints on disk")


if __name__ == "__main__":
    main()
