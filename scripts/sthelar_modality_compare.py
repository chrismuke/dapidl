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
    ("Fusion", Path("/mnt/work/git/dapidl/pipeline_output/sthelar_modality_fusion")),
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
    present_names = [n for n, _ in EXP_DIRS if n in results]
    lines = [f"# {len(present_names)}-Way Modality Comparison ({' / '.join(present_names)})\n"]
    lines.append("| Modality | accuracy | macro F1 | weighted F1 |")
    lines.append("|---|---:|---:|---:|")
    for name, _ in EXP_DIRS:
        if name not in results:
            lines.append(f"| {name} | — | — | — |")
            continue
        m = results[name]["per_class_metrics"]
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |")

    # Helper for delta columns vs DAPI baseline
    delta_modalities = [n for n in present_names if n != "DAPI"]
    delta_header = " | ".join(f"Δ({n}−DAPI)" for n in delta_modalities)

    def _build_align_row(n_modalities: int, n_deltas: int) -> str:
        # 1 left-aligned (Class/Tissue) + 1 right (Support/n_test) + n right (modalities) + n right (deltas)
        cols = ["---"] + ["---:"] * (1 + n_modalities + n_deltas)
        return "| " + " | ".join(cols) + " |"

    # ---------- per-class ----------
    classes = []
    if "DAPI" in results:
        classes = list(results["DAPI"]["per_class_metrics"]["per_class"].keys())
    if classes:
        lines.append("\n# Per-class F1 across modalities\n")
        modality_cols = " | ".join(present_names)
        lines.append(f"| Class | Support | {modality_cols} | {delta_header} |")
        lines.append(_build_align_row(len(present_names), len(delta_modalities)))
        for c in classes:
            sup = results["DAPI"]["per_class_metrics"]["per_class"][c]["support"]
            row = [c, f"{sup:,}"]
            f1s = {}
            for name in present_names:
                pc = results[name]["per_class_metrics"]["per_class"].get(c)
                f1s[name] = pc["f1"] if pc else None
                row.append(f"{pc['f1']:.3f}" if pc else "—")
            for n in delta_modalities:
                d = (f1s[n] - f1s["DAPI"]) if (f1s.get(n) is not None and f1s.get("DAPI") is not None) else None
                row.append(f"{d:+.3f}" if d is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

    # ---------- per-tissue ----------
    if "DAPI" in results and "per_tissue_metrics" in results["DAPI"]:
        tissues = list(results["DAPI"]["per_tissue_metrics"].keys())
        lines.append("\n# Per-tissue macro F1 across modalities\n")
        modality_cols = " | ".join(present_names)
        lines.append(f"| Tissue | n_test | {modality_cols} | {delta_header} |")
        lines.append(_build_align_row(len(present_names), len(delta_modalities)))
        for t in tissues:
            n = results["DAPI"]["per_tissue_metrics"][t]["n"]
            row = [t, f"{n:,}"]
            f1s = {}
            for name in present_names:
                pt = results[name].get("per_tissue_metrics", {}).get(t)
                if pt is not None:
                    f1s[name] = pt["macro_f1"]
                    row.append(f"{f1s[name]:.3f}")
                else:
                    f1s[name] = None
                    row.append("—")
            for nm in delta_modalities:
                d = (f1s[nm] - f1s["DAPI"]) if (f1s.get(nm) is not None and f1s.get("DAPI") is not None) else None
                row.append(f"{d:+.3f}" if d is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

    md_path = args.output_dir / "modality_3way.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"table: {md_path}")

    # ---------- figures ----------
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

    present = [n for n, _ in EXP_DIRS if n in results]
    colors = {"DAPI": "#1f77b4", "H&E": "#d62728", "DAPI+H&E": "#9467bd", "Fusion": "#2ca02c"}

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
    ax.set_title("Modality comparison — overall test metrics")
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
