#!/usr/bin/env python3
"""3-way modality comparison: DAPI / H&E / DAPI+H&E on STHELAR multi-tissue.

Reads results from:
- DAPI baseline:       pipeline_output/sthelar_multitissue_9class
- H&E only:            pipeline_output/sthelar_modality_he
- DAPI+H&E multimodal: pipeline_output/sthelar_modality_both

Produces:
- comparison/modality_3way.json
- comparison/modality_3way.md
- figures-comparison/modality_*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


EXP_DIRS = [
    ("DAPI", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_modality_dapi")),
    ("H&E", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_modality_he")),
    ("DAPI+H&E", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_modality_both")),
]


def load_exp(d: Path) -> dict | None:
    a = d / "analysis"
    if not a.exists():
        return None
    out = {}
    for fname in ["per_class_metrics.json", "per_tissue_metrics.json", "summary.json"]:
        fp = a / fname
        if fp.exists():
            out[fname.replace(".json", "")] = json.load(open(fp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/mnt/work/git/dapidl/pipeline_output/sthelar_comparison"))
    ap.add_argument("--figures-dir", type=Path,
                    default=Path("/home/chrism/obsidian/llmbrain/DAPIDL/Multi-Tissue-STHELAR-20260422/figures-comparison"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, d in EXP_DIRS:
        r = load_exp(d)
        if r is None:
            print(f"[WARN] missing {name} ({d})")
            continue
        results[name] = r
    with open(args.output_dir / "modality_3way.json", "w") as f:
        json.dump(results, f, indent=2)
    if not results:
        return

    # ---------- overall ----------
    lines = ["# 3-Way Modality Comparison (DAPI / H&E / DAPI+H&E)\n"]
    lines.append("| Modality | accuracy | macro F1 | weighted F1 |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in EXP_DIRS:
        if name not in results:
            lines.append(f"| {name} | — | — | — |")
            continue
        m = results[name]["per_class_metrics"]
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |")

    # ---------- per-class ----------
    classes = []
    if "DAPI" in results:
        classes = list(results["DAPI"]["per_class_metrics"]["per_class"].keys())
    if classes:
        lines.append("\n# Per-class F1 across modalities\n")
        lines.append("| Class | Support | DAPI | H&E | DAPI+H&E | Δ(H&E−DAPI) | Δ(DAPI+H&E−DAPI) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for c in classes:
            sup = results["DAPI"]["per_class_metrics"]["per_class"][c]["support"]
            row = [c, f"{sup:,}"]
            f1s = {}
            for name, _ in EXP_DIRS:
                if name in results:
                    pc = results[name]["per_class_metrics"]["per_class"].get(c)
                    f1s[name] = pc["f1"] if pc else None
                    row.append(f"{pc['f1']:.3f}" if pc else "—")
                else:
                    row.append("—")
            d_he = (f1s.get("H&E", 0) - f1s.get("DAPI", 0)) if all(k in f1s and f1s[k] is not None for k in ("DAPI", "H&E")) else None
            d_both = (f1s.get("DAPI+H&E", 0) - f1s.get("DAPI", 0)) if all(k in f1s and f1s[k] is not None for k in ("DAPI", "DAPI+H&E")) else None
            row.append(f"{d_he:+.3f}" if d_he is not None else "—")
            row.append(f"{d_both:+.3f}" if d_both is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

    # ---------- per-tissue ----------
    if "DAPI" in results and "per_tissue_metrics" in results["DAPI"]:
        tissues = list(results["DAPI"]["per_tissue_metrics"].keys())
        lines.append("\n# Per-tissue macro F1 across modalities\n")
        lines.append("| Tissue | n_test | DAPI | H&E | DAPI+H&E | Δ(H&E−DAPI) | Δ(DAPI+H&E−DAPI) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for t in tissues:
            n = results["DAPI"]["per_tissue_metrics"][t]["n"]
            row = [t, f"{n:,}"]
            f1s = {}
            for name, _ in EXP_DIRS:
                if name in results and t in results[name].get("per_tissue_metrics", {}):
                    f1 = results[name]["per_tissue_metrics"][t]["macro_f1"]
                    f1s[name] = f1
                    row.append(f"{f1:.3f}")
                else:
                    row.append("—")
            d_he = (f1s.get("H&E", 0) - f1s.get("DAPI", 0)) if all(k in f1s for k in ("DAPI", "H&E")) else None
            d_both = (f1s.get("DAPI+H&E", 0) - f1s.get("DAPI", 0)) if all(k in f1s for k in ("DAPI", "DAPI+H&E")) else None
            row.append(f"{d_he:+.3f}" if d_he is not None else "—")
            row.append(f"{d_both:+.3f}" if d_both is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

    md_path = args.output_dir / "modality_3way.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"table: {md_path}")

    # ---------- figures ----------
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

    present = [n for n, _ in EXP_DIRS if n in results]
    colors = {"DAPI": "#1f77b4", "H&E": "#d62728", "DAPI+H&E": "#9467bd"}

    # Fig 1: overall bars
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(present))
    w = 0.27
    for i, metric in enumerate(["macro_f1", "weighted_f1", "accuracy"]):
        vals = [results[n]["per_class_metrics"][metric] for n in present]
        ax.bar(x + (i - 1) * w, vals, w, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("DAPI vs H&E vs DAPI+H&E — overall test metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "modality_01_overall.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: per-class F1
    if classes:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(classes))
        w = 0.27
        for i, name in enumerate(present):
            vals = [results[name]["per_class_metrics"]["per_class"].get(c, {}).get("f1", 0) for c in classes]
            ax.bar(x + (i - len(present) / 2 + 0.5) * w, vals, w, label=name, color=colors.get(name))
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("test F1")
        ax.set_title("Per-class F1 across modalities")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.figures_dir / "modality_02_per_class.png", bbox_inches="tight")
        plt.close(fig)

    # Fig 3: per-tissue heatmap
    if "DAPI" in results and "per_tissue_metrics" in results["DAPI"]:
        tissues = list(results["DAPI"]["per_tissue_metrics"].keys())
        M = np.full((len(tissues), len(present)), np.nan)
        for j, name in enumerate(present):
            for i, t in enumerate(tissues):
                pt = results[name].get("per_tissue_metrics", {}).get(t)
                if pt is not None:
                    M[i, j] = pt["macro_f1"]
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=0.7,
                    xticklabels=present, yticklabels=tissues, ax=ax)
        ax.set_title("Per-tissue macro F1 across modalities")
        fig.tight_layout()
        fig.savefig(args.figures_dir / "modality_03_per_tissue.png", bbox_inches="tight")
        plt.close(fig)

    print(f"figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
