#!/usr/bin/env python3
"""Cross-experiment master table for every trained DAPIDL model.

Walks pipeline_output/ for STHELAR experiments + LOTO + modality + MERSCOPE
runs, reads each summary.json / per_class_metrics.json / per_tissue_metrics.json,
and writes a unified parquet + markdown master table.

Output: pipeline_output/model_eval_2026_05/{master_metrics.parquet, by_class.parquet,
        by_tissue.parquet, master_metrics.md}

Usage:
    uv run python scripts/aggregate_trained_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "model_eval_2026_05"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENT_GROUPS = {
    "modality": ["sthelar_modality_dapi", "sthelar_modality_he",
                 "sthelar_modality_both", "sthelar_modality_fusion"],
    "exp": [f"sthelar_exp{i}_*" for i in range(1, 6)],
    "loto_dapi": ["sthelar_loto_*"],
    "loto_he": ["sthelar_loto_he_*"],
    "single": ["sthelar_dapi_breast_s0", "sthelar_multitissue_9class",
               "merscope_immune"],
}

CHECKPOINT_BACKBONES = {
    "sthelar_modality_dapi": ("EfficientNetV2-S", "DAPI", 9),
    "sthelar_modality_he": ("EfficientNetV2-S", "HE", 9),
    "sthelar_modality_both": ("EfficientNetV2-S+1x1conv", "DAPI+HE", 9),
    "sthelar_modality_fusion": ("EfficientNetV2-S x2 + cross-attn", "DAPI+HE", 9),
    "sthelar_exp1_hierarchical": ("EfficientNetV2-S+aux", "DAPI", 9),
    "sthelar_exp2_heavy_aug": ("EfficientNetV2-S", "DAPI", 9),
    "sthelar_exp3_loto_brain": ("EfficientNetV2-S", "DAPI", 9),
    "sthelar_exp4_vit": ("ViT-S DINO", "DAPI", 9),
    "sthelar_exp5_7class": ("EfficientNetV2-S", "DAPI", 7),
    "sthelar_dapi_breast_s0": ("EfficientNetV2-S", "DAPI", 9),
    "sthelar_multitissue_9class": ("EfficientNetV2-S", "DAPI", 9),
    "merscope_immune": ("DenseNet-121", "DAPI", 10),
}


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _classify_run(name: str) -> tuple[str, str]:
    if name.startswith("sthelar_loto_he_"):
        return "loto_he", name.replace("sthelar_loto_he_", "")
    if name.startswith("sthelar_loto_"):
        return "loto_dapi", name.replace("sthelar_loto_", "")
    if name.startswith("sthelar_modality_"):
        return "modality", name.replace("sthelar_modality_", "")
    if name.startswith("sthelar_exp"):
        return "exp", name
    return "single", name


def _checkpoint_size_mb(run_dir: Path) -> float | None:
    p = run_dir / "best_model.pt"
    return round(p.stat().st_size / 1024 / 1024, 1) if p.exists() else None


def collect_master_rows() -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(
            ("sthelar_", "merscope_")
        ):
            continue
        analysis = run_dir / "analysis"
        summary = _read_json(analysis / "summary.json")
        if summary is None:
            # MERSCOPE result is at top level
            alt = _read_json(run_dir / "merscope_results.json")
            if alt is None:
                continue
            for entry in alt:
                rows.append({
                    "run": run_dir.name + "_" + entry["dataset"],
                    "group": "single",
                    "subset": entry["dataset"],
                    "modality": "DAPI",
                    "n_classes": entry.get("n_classes"),
                    "backbone": entry.get("model"),
                    "n_params": None,
                    "best_epoch": None,
                    "best_val_macro_f1": entry.get("best_val_f1"),
                    "test_acc": None,
                    "test_macro_f1": entry.get("test_f1"),
                    "test_weighted_f1": None,
                    "n_train": None,
                    "n_val": None,
                    "n_test": entry.get("n_immune"),
                    "ckpt_mb": None,
                })
            continue

        group, subset = _classify_run(run_dir.name)
        backbone, modality, n_classes = CHECKPOINT_BACKBONES.get(
            run_dir.name,
            ("EfficientNetV2-S", "DAPI" if "_he_" not in run_dir.name else "HE",
             len(summary.get("class_names") or []) or None),
        )
        rows.append({
            "run": run_dir.name,
            "group": group,
            "subset": subset,
            "modality": modality,
            "n_classes": n_classes,
            "backbone": backbone,
            "n_params": summary.get("n_params"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_macro_f1": summary.get("best_val_macro_f1"),
            "test_acc": summary.get("test_accuracy") or summary.get("test_acc"),
            "test_macro_f1": summary.get("test_macro_f1"),
            "test_weighted_f1": summary.get("test_weighted_f1"),
            "n_train": summary.get("n_train"),
            "n_val": summary.get("n_val"),
            "n_test": summary.get("n_test"),
            "ckpt_mb": _checkpoint_size_mb(run_dir),
        })
    return rows


def collect_per_class_rows() -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(ROOT.iterdir()):
        pc = run_dir / "analysis" / "per_class_metrics.json"
        if not pc.exists():
            continue
        d = _read_json(pc)
        if d is None or "per_class" not in d:
            continue
        group, subset = _classify_run(run_dir.name)
        for cls, m in d["per_class"].items():
            rows.append({
                "run": run_dir.name,
                "group": group,
                "subset": subset,
                "class": cls,
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "support": m.get("support"),
            })
    return rows


def collect_per_tissue_rows() -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(ROOT.iterdir()):
        pt = run_dir / "analysis" / "per_tissue_metrics.json"
        if not pt.exists():
            continue
        d = _read_json(pt)
        if d is None:
            continue
        group, subset = _classify_run(run_dir.name)
        for tissue, m in d.items():
            if not isinstance(m, dict):
                continue
            rows.append({
                "run": run_dir.name,
                "group": group,
                "subset": subset,
                "tissue": tissue,
                "n": m.get("n"),
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "weighted_f1": m.get("weighted_f1"),
            })
    return rows


def write_master_md(master: pl.DataFrame) -> None:
    out = OUT / "master_metrics.md"
    lines = [
        "# DAPIDL Trained Models — Master Metrics",
        "",
        f"Aggregated from `{ROOT}` on {Path(__file__).name}.",
        "",
        f"**{len(master)} runs total** across {master['group'].n_unique()} groups.",
        "",
    ]
    # Sort by group then by test_macro_f1 desc
    sorted_df = master.sort(["group", "test_macro_f1"], descending=[False, True])
    cols = ["run", "modality", "n_classes", "backbone", "n_params",
            "best_epoch", "best_val_macro_f1", "test_acc", "test_macro_f1",
            "test_weighted_f1", "n_test"]

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}"
        return str(v)

    for group in sorted_df["group"].unique().to_list():
        sub = sorted_df.filter(pl.col("group") == group)
        lines.append(f"## Group: `{group}` ({len(sub)} runs)")
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in sub.iter_rows(named=True):
            lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> None:
    master_rows = collect_master_rows()
    pc_rows = collect_per_class_rows()
    pt_rows = collect_per_tissue_rows()

    if not master_rows:
        print("No experiments found.")
        return

    master = pl.DataFrame(master_rows)
    pc = pl.DataFrame(pc_rows)
    pt = pl.DataFrame(pt_rows)

    master.write_parquet(OUT / "master_metrics.parquet")
    pc.write_parquet(OUT / "by_class.parquet")
    pt.write_parquet(OUT / "by_tissue.parquet")

    print(f"Master: {len(master)} runs  →  {OUT / 'master_metrics.parquet'}")
    print(f"By class: {len(pc)} rows  →  {OUT / 'by_class.parquet'}")
    print(f"By tissue: {len(pt)} rows  →  {OUT / 'by_tissue.parquet'}")
    print()

    # Quick CLI summary
    top = master.filter(pl.col("test_macro_f1").is_not_null()).sort(
        "test_macro_f1", descending=True
    ).head(10).select(["run", "modality", "n_classes", "test_macro_f1"])
    print("Top-10 by test macro F1:")
    print(top)

    write_master_md(master)


if __name__ == "__main__":
    main()
