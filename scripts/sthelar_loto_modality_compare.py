#!/usr/bin/env python3
"""DAPI vs H&E leave-one-tissue-out comparison across 16 tissues.

Reads:
- DAPI LOTO: pipeline_output/sthelar_loto_{tissue}/analysis/summary.json
            (and sthelar_exp3_loto_brain for brain)
- H&E  LOTO: pipeline_output/sthelar_loto_he_{tissue}/analysis/summary.json

Produces:
- pipeline_output/sthelar_comparison/loto_modality.json
- pipeline_output/sthelar_comparison/loto_modality.md
- {figures-comparison}/loto_modality_per_tissue.png
- {figures-comparison}/loto_modality_delta.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


TISSUES = [
    "bone", "bone_marrow", "brain", "breast", "cervix", "colon",
    "heart", "kidney", "liver", "lung", "lymph_node", "ovary",
    "pancreatic", "prostate", "skin", "tonsil",
]

PO = Path("/mnt/work/git/dapidl/pipeline_output")


def load_dapi_loto(tissue: str) -> dict | None:
    # Brain DAPI LOTO is in sthelar_exp3_loto_brain (10 epochs);
    # the rest are in sthelar_loto_{tissue} (6 epochs).
    candidates = [
        PO / f"sthelar_loto_{tissue}" / "analysis" / "summary.json",
        PO / f"sthelar_exp3_loto_{tissue}" / "analysis" / "summary.json",
    ]
    for p in candidates:
        if p.exists():
            return json.load(open(p))
    return None


def load_he_loto(tissue: str) -> dict | None:
    p = PO / f"sthelar_loto_he_{tissue}" / "analysis" / "summary.json"
    return json.load(open(p)) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=PO / "sthelar_comparison")
    ap.add_argument("--figures-dir", type=Path,
                    default=Path("/home/chrism/obsidian/llmbrain/DAPIDL/Multi-Tissue-STHELAR-20260422/figures-comparison"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in TISSUES:
        d = load_dapi_loto(t)
        h = load_he_loto(t)
        rows.append({
            "tissue": t,
            "dapi": d, "he": h,
            "dapi_test_f1": d["test_macro_f1"] if d else None,
            "he_test_f1": h["test_macro_f1"] if h else None,
            "dapi_test_acc": d["test_accuracy"] if d else None,
            "he_test_acc": h["test_accuracy"] if h else None,
            "n_test": (d or h or {}).get("n_test"),
        })

    # ---------- markdown table ----------
    lines = ["# DAPI vs H&E — Leave-One-Tissue-Out\n"]
    lines.append("Held-out tissue is unseen during training. Test set = all HE-intersection patches "
                 "of that tissue. Both DAPI and H&E sweeps use the same 6-epoch budget, batch 64, "
                 "and same train/val split (85/15 stratified, seed 42).\n")

    lines.append("\n## Per-tissue test macro F1\n")
    lines.append("| Tissue | n_test | DAPI | H&E | Δ(H&E−DAPI) |")
    lines.append("|---|---:|---:|---:|---:|")
    deltas = []
    mean_d = 0.0
    med_d = 0.0
    wins_he = 0
    for r in rows:
        d_f1 = r["dapi_test_f1"]
        h_f1 = r["he_test_f1"]
        delta = (h_f1 - d_f1) if (d_f1 is not None and h_f1 is not None) else None
        if delta is not None:
            deltas.append(delta)
        d_str = f"{d_f1:.3f}" if d_f1 is not None else "—"
        h_str = f"{h_f1:.3f}" if h_f1 is not None else "—"
        delta_str = f"{delta:+.3f}" if delta is not None else "—"
        n = r["n_test"]
        n_str = f"{n:,}" if n is not None else "—"
        lines.append(f"| {r['tissue']} | {n_str} | {d_str} | {h_str} | {delta_str} |")

    if deltas:
        mean_d = float(np.mean(deltas))
        med_d = float(np.median(deltas))
        wins_he = sum(1 for d in deltas if d > 0)
        lines.append(f"| **mean** | | | | **{mean_d:+.3f}** |")
        lines.append(f"| **median** | | | | **{med_d:+.3f}** |")

    lines.append(f"\nH&E wins on **{wins_he}/{len(deltas)}** tissues "
                 f"(mean delta = **{mean_d:+.3f}**, median = **{med_d:+.3f}**).\n")

    md_path = args.output_dir / "loto_modality.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"table: {md_path}")

    with open(args.output_dir / "loto_modality.json", "w") as f:
        json.dump({"rows": [{k: r[k] for k in
                              ("tissue", "dapi_test_f1", "he_test_f1",
                               "dapi_test_acc", "he_test_acc", "n_test")}
                            for r in rows],
                   "summary": {
                       "mean_delta_f1": float(np.mean(deltas)) if deltas else None,
                       "median_delta_f1": float(np.median(deltas)) if deltas else None,
                       "he_wins": wins_he, "n_tissues": len(deltas),
                   }}, f, indent=2)

    # ---------- figures ----------
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

    # Fig 1: side-by-side bars
    fig, ax = plt.subplots(figsize=(11, 5))
    tissues_x = [r["tissue"] for r in rows
                 if r["dapi_test_f1"] is not None and r["he_test_f1"] is not None]
    d_vals = [r["dapi_test_f1"] for r in rows
              if r["dapi_test_f1"] is not None and r["he_test_f1"] is not None]
    h_vals = [r["he_test_f1"] for r in rows
              if r["dapi_test_f1"] is not None and r["he_test_f1"] is not None]
    x = np.arange(len(tissues_x))
    w = 0.4
    ax.bar(x - w/2, d_vals, w, label="DAPI", color="#1f77b4")
    ax.bar(x + w/2, h_vals, w, label="H&E", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(tissues_x, rotation=30, ha="right")
    ax.set_ylim(0, max(max(d_vals), max(h_vals)) * 1.15)
    ax.set_ylabel("test macro F1 (held-out tissue)")
    ax.set_title("Cross-tissue generalisation: DAPI vs H&E LOTO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "loto_modality_per_tissue.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: deltas
    fig, ax = plt.subplots(figsize=(10, 4))
    delta_rows = [(r["tissue"], r["he_test_f1"] - r["dapi_test_f1"]) for r in rows
                  if r["dapi_test_f1"] is not None and r["he_test_f1"] is not None]
    delta_rows.sort(key=lambda x: x[1])
    tissues_d = [t for t, _ in delta_rows]
    delta_v = [d for _, d in delta_rows]
    colors = ["#d62728" if d > 0 else "#1f77b4" for d in delta_v]
    ax.bar(np.arange(len(tissues_d)), delta_v, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(tissues_d)))
    ax.set_xticklabels(tissues_d, rotation=30, ha="right")
    ax.set_ylabel("Δ macro F1 (H&E − DAPI)")
    ax.set_title("Per-tissue modality advantage (positive = H&E better)")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "loto_modality_delta.png", bbox_inches="tight")
    plt.close(fig)

    print(f"figures: {args.figures_dir}")
    print(f"H&E wins: {wins_he}/{len(deltas)} | mean delta: {mean_d:+.4f}")


if __name__ == "__main__":
    main()
