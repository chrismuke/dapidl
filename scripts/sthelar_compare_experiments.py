#!/usr/bin/env python3
"""Aggregate and compare results from all 5 STHELAR experiments.

Reads per-class + per-tissue metrics from each experiment's analysis/ folder,
builds comparison tables and figures, and saves everything ready for the
Obsidian write-up.

Outputs:
- comparison/all_results.json     — unified metrics dict
- comparison/comparison_table.md  — pretty markdown tables
- comparison/figures/*.png        — per-class delta, per-tissue delta, overall bar chart
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BASELINE_NAME = "baseline"
EXPERIMENTS = [
    ("baseline", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_multitissue_9class")),
    ("exp5_7class", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_exp5_7class")),
    ("exp4_vit", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_exp4_vit")),
    ("exp2_heavy_aug", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_exp2_heavy_aug")),
    ("exp1_hierarchical", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_exp1_hierarchical")),
    ("exp3_loto_brain", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_exp3_loto_brain")),
]

PRETTY_NAMES = {
    "baseline": "Baseline (9-class EfficientNetV2-S, DALI)",
    "exp5_7class": "Exp 5 — Drop adipocyte+mast (7-class)",
    "exp4_vit": "Exp 4 — ViT-S / DINO backbone",
    "exp2_heavy_aug": "Exp 2 — Heavy augmentation",
    "exp1_hierarchical": "Exp 1 — Hierarchical-lite (aux coarse head)",
    "exp3_loto_brain": "Exp 3 — LOTO (brain held out)",
}


def load_exp(path: Path) -> dict | None:
    """Load the per-class and per-tissue metrics dict for one experiment."""
    a = path / "analysis"
    if not a.exists():
        return None
    out = {"output_dir": str(path)}
    pc = a / "per_class_metrics.json"
    if pc.exists():
        with open(pc) as f:
            out["per_class_metrics"] = json.load(f)
    pt = a / "per_tissue_metrics.json"
    if pt.exists():
        with open(pt) as f:
            out["per_tissue_metrics"] = json.load(f)
    s = a / "summary.json"
    if s.exists():
        with open(s) as f:
            out["summary"] = json.load(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("/mnt/work/git/dapidl/pipeline_output/sthelar_comparison"),
    )
    ap.add_argument(
        "--obsidian-dir", type=Path,
        default=Path("/home/chrism/obsidian/llmbrain/DAPIDL/Multi-Tissue-STHELAR-20260422/figures-comparison"),
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.obsidian_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all
    results: dict[str, dict] = {}
    for name, path in EXPERIMENTS:
        r = load_exp(path)
        if r is None:
            print(f"[WARN] missing analysis for {name} ({path})")
            continue
        results[name] = r
    with open(args.output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if not results:
        print("no results found")
        return

    # ----------- overall comparison table ---------------
    lines = []
    lines.append("# Experiment Overall Comparison\n")
    lines.append("| Experiment | n_test | accuracy | macro F1 | weighted F1 | Δ macro F1 vs baseline |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    baseline_f1 = None
    for name, _ in EXPERIMENTS:
        if name not in results:
            continue
        m = results[name].get("per_class_metrics", {})
        acc = m.get("accuracy")
        mf1 = m.get("macro_f1")
        wf1 = m.get("weighted_f1")
        n = sum(d["support"] for d in m.get("per_class", {}).values()) if m else 0
        if name == BASELINE_NAME and mf1 is not None:
            baseline_f1 = mf1
        delta = (mf1 - baseline_f1) if (baseline_f1 is not None and mf1 is not None and name != BASELINE_NAME) else None
        if acc is None:
            lines.append(f"| {PRETTY_NAMES[name]} | — | — | — | — | — |")
            continue
        delta_str = f"{delta:+.4f}" if delta is not None else "—"
        lines.append(f"| {PRETTY_NAMES[name]} | {n:,} | {acc:.4f} | {mf1:.4f} | {wf1:.4f} | {delta_str} |")

    # ----------- per-class F1 table ----------
    lines.append("\n# Per-Class F1 Across Experiments\n")
    classes = list(results[BASELINE_NAME]["per_class_metrics"]["per_class"].keys()) if BASELINE_NAME in results else []
    if classes:
        hdr_cells = ["class", "support"] + [PRETTY_NAMES[n] for n, _ in EXPERIMENTS if n in results]
        lines.append("| " + " | ".join(hdr_cells) + " |")
        lines.append("|---|---:|" + "|".join([":---:"] * (len(hdr_cells) - 2)) + "|")
        for c in classes:
            sup = results[BASELINE_NAME]["per_class_metrics"]["per_class"][c]["support"]
            row = [c, f"{sup:,}"]
            for name, _ in EXPERIMENTS:
                if name not in results:
                    continue
                pc = results[name].get("per_class_metrics", {}).get("per_class", {}).get(c)
                row.append(f"{pc['f1']:.3f}" if pc else "—")
            lines.append("| " + " | ".join(row) + " |")

    # ----------- per-tissue F1 table ----------
    if results[BASELINE_NAME].get("per_tissue_metrics"):
        lines.append("\n# Per-Tissue Macro F1 Across Experiments\n")
        tissues = list(results[BASELINE_NAME]["per_tissue_metrics"].keys())
        hdr = ["tissue", "n"] + [PRETTY_NAMES[n] for n, _ in EXPERIMENTS if n in results]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|---|---:|" + "|".join([":---:"] * (len(hdr) - 2)) + "|")
        for t in tissues:
            n_ = results[BASELINE_NAME]["per_tissue_metrics"][t]["n"]
            row = [t, f"{n_:,}"]
            for name, _ in EXPERIMENTS:
                if name not in results:
                    continue
                pt = results[name].get("per_tissue_metrics", {}).get(t)
                row.append(f"{pt['macro_f1']:.3f}" if pt else "—")
            lines.append("| " + " | ".join(row) + " |")

    with open(args.output_dir / "comparison_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"table written to {args.output_dir / 'comparison_table.md'}")

    # ----------- figures ---------------
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

    present = [n for n, _ in EXPERIMENTS if n in results]

    # Figure A: Overall macro/weighted F1 bars
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(present))
    macro = [results[n]["per_class_metrics"]["macro_f1"] for n in present]
    weighted = [results[n]["per_class_metrics"]["weighted_f1"] for n in present]
    acc = [results[n]["per_class_metrics"]["accuracy"] for n in present]
    w = 0.28
    ax.bar(x - w, macro, w, label="macro F1", color="#1f77b4")
    ax.bar(x, weighted, w, label="weighted F1", color="#2ca02c")
    ax.bar(x + w, acc, w, label="accuracy", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAMES[n].split(" — ")[0] for n in present], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Overall test performance across experiments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "comparison_01_overall.png", bbox_inches="tight")
    plt.close(fig)

    # Figure B: Per-class F1 delta vs baseline
    if BASELINE_NAME in results and classes:
        base_pc = results[BASELINE_NAME]["per_class_metrics"]["per_class"]
        others = [n for n in present if n != BASELINE_NAME]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classes))
        w = 0.8 / max(len(others), 1)
        for i, name in enumerate(others):
            deltas = []
            for c in classes:
                pc = results[name].get("per_class_metrics", {}).get("per_class", {}).get(c)
                if pc is None:
                    deltas.append(0)
                else:
                    deltas.append(pc["f1"] - base_pc[c]["f1"])
            ax.bar(x + (i - (len(others) - 1) / 2) * w, deltas, w,
                   label=PRETTY_NAMES[name].split(" — ")[0])
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylabel("Δ F1 vs baseline")
        ax.set_title("Per-class F1 change relative to baseline")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison_02_per_class_delta.png", bbox_inches="tight")
        plt.close(fig)

    # Figure C: Per-tissue macro F1 heatmap
    if results[BASELINE_NAME].get("per_tissue_metrics"):
        tissues = list(results[BASELINE_NAME]["per_tissue_metrics"].keys())
        M = np.full((len(tissues), len(present)), np.nan)
        for j, name in enumerate(present):
            pts = results[name].get("per_tissue_metrics", {})
            for i, t in enumerate(tissues):
                if t in pts:
                    M[i, j] = pts[t]["macro_f1"]
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            M, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=0.7,
            xticklabels=[PRETTY_NAMES[n].split(" — ")[0] for n in present],
            yticklabels=tissues, ax=ax,
        )
        ax.set_title("Per-tissue macro F1 across experiments")
        fig.tight_layout()
        fig.savefig(fig_dir / "comparison_03_per_tissue_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
