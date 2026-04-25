#!/usr/bin/env python3
"""Aggregate LOTO results across all 16 tissues, compare to baseline.

For each tissue T:
- baseline_acc/F1 on T: from baseline's per_tissue_metrics
- loto_acc/F1 on T: from LOTO (tissue T held out)
- shortcutting_gap = baseline_acc - loto_acc
- shortcutting_ratio = baseline_acc / loto_acc

Produces:
- comparison/loto_all_results.json
- comparison/loto_comparison.md
- figures-comparison/loto_*.png (bar chart, collapse ratio, etc.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


BASELINE_PT = Path("/mnt/work/git/dapidl/pipeline_output/sthelar_multitissue_9class/analysis/per_tissue_metrics.json")

# LOTO output directories; brain is the old Exp 3 run, others are the new loto_all sweep
LOTO_DIRS = {
    "brain":       Path("/mnt/work/git/dapidl/pipeline_output/sthelar_exp3_loto_brain"),
    "bone":        Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_bone"),
    "bone_marrow": Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_bone_marrow"),
    "breast":      Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_breast"),
    "cervix":      Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_cervix"),
    "colon":       Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_colon"),
    "heart":       Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_heart"),
    "kidney":      Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_kidney"),
    "liver":       Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_liver"),
    "lung":        Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_lung"),
    "lymph_node":  Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_lymph_node"),
    "ovary":       Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_ovary"),
    "pancreatic":  Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_pancreatic"),
    "prostate":    Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_prostate"),
    "skin":        Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_skin"),
    "tonsil":      Path("/mnt/work/git/dapidl/pipeline_output/sthelar_loto_tonsil"),
}


def load_loto(dir_path: Path) -> dict | None:
    sp = dir_path / "analysis" / "summary.json"
    if not sp.exists():
        return None
    with open(sp) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("/mnt/work/git/dapidl/pipeline_output/sthelar_comparison"),
    )
    ap.add_argument(
        "--figures-dir", type=Path,
        default=Path("/home/chrism/obsidian/llmbrain/DAPIDL/Multi-Tissue-STHELAR-20260422/figures-comparison"),
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    with open(BASELINE_PT) as f:
        baseline_pt = json.load(f)

    results = {}
    missing = []
    for tissue, d in LOTO_DIRS.items():
        loto = load_loto(d)
        if loto is None:
            missing.append(tissue)
            continue
        if tissue not in baseline_pt:
            continue
        b = baseline_pt[tissue]
        results[tissue] = {
            "n_test": int(loto.get("n_test", b.get("n", 0))),
            "baseline_acc": b["accuracy"],
            "baseline_macro_f1": b["macro_f1"],
            "baseline_weighted_f1": b["weighted_f1"],
            "loto_acc": loto["test_accuracy"],
            "loto_macro_f1": loto["test_macro_f1"],
            "loto_weighted_f1": loto["test_weighted_f1"],
            "acc_drop": b["accuracy"] - loto["test_accuracy"],
            "acc_ratio": b["accuracy"] / max(loto["test_accuracy"], 1e-6),
            "weighted_f1_drop": b["weighted_f1"] - loto["test_weighted_f1"],
            "weighted_f1_ratio": b["weighted_f1"] / max(loto["test_weighted_f1"], 1e-6),
            "best_epoch": loto.get("best_epoch", "?"),
            "best_val_macro_f1": loto.get("best_val_macro_f1", "?"),
        }
    with open(args.output_dir / "loto_all_results.json", "w") as f:
        json.dump({"results": results, "missing": missing}, f, indent=2)

    if missing:
        print(f"[WARN] missing {len(missing)} tissues: {missing}")
    if not results:
        print("no complete runs yet")
        return

    # Sort by baseline_acc descending — so we see which tissues had the biggest drop
    sorted_t = sorted(results.items(), key=lambda x: -x[1]["baseline_acc"])

    lines = []
    lines.append("# LOTO vs Baseline — Tissue-Shortcutting Magnitude\n")
    lines.append(
        "| Tissue | n_test | Baseline acc | LOTO acc | acc drop | ratio | "
        "Baseline weighted F1 | LOTO weighted F1 | wF1 drop |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t, r in sorted_t:
        lines.append(
            f"| {t} | {r['n_test']:,} | "
            f"{r['baseline_acc']:.3f} | {r['loto_acc']:.3f} | "
            f"**{r['acc_drop']:.3f}** | **{r['acc_ratio']:.2f}×** | "
            f"{r['baseline_weighted_f1']:.3f} | {r['loto_weighted_f1']:.3f} | "
            f"{r['weighted_f1_drop']:.3f} |"
        )
    with open(args.output_dir / "loto_comparison.md", "w") as f:
        f.write("\n".join(lines))
    print(f"table: {args.output_dir / 'loto_comparison.md'}")

    # Figures
    sns.set_style("whitegrid")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 10})

    tissues = [t for t, _ in sorted_t]
    b_acc = np.array([results[t]["baseline_acc"] for t in tissues])
    l_acc = np.array([results[t]["loto_acc"] for t in tissues])
    b_wf1 = np.array([results[t]["baseline_weighted_f1"] for t in tissues])
    l_wf1 = np.array([results[t]["loto_weighted_f1"] for t in tissues])

    # Fig 1: Baseline vs LOTO accuracy, side-by-side bars
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tissues))
    w = 0.4
    ax.bar(x - w / 2, b_acc, w, label="baseline (in-dist)", color="#2ca02c")
    ax.bar(x + w / 2, l_acc, w, label="LOTO (held out)", color="#d62728")
    for i, (ba, la) in enumerate(zip(b_acc, l_acc)):
        ax.annotate(f"{la / max(ba, 1e-6):.2f}×", (i, min(ba, la) + 0.02),
                    ha="center", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("LOTO vs baseline accuracy — each tissue held out individually")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "loto_01_accuracy_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Accuracy drop (bar chart) — the direct shortcutting measure
    drops = np.array([results[t]["acc_drop"] for t in tissues])
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Reds(0.3 + 0.7 * (drops / max(drops.max(), 0.1)))
    ax.bar(range(len(tissues)), drops, color=colors)
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_ylabel("baseline acc − LOTO acc")
    ax.set_title("Accuracy collapse when each tissue is held out (tissue-shortcutting magnitude)")
    for i, d in enumerate(drops):
        ax.text(i, d + 0.01, f"{d:+.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(args.figures_dir / "loto_02_accuracy_drop.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Collapse ratio on log scale (baseline/LOTO acc)
    ratios = np.array([results[t]["acc_ratio"] for t in tissues])
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Reds(0.3 + 0.7 * (np.log(np.maximum(ratios, 1)) / np.log(max(ratios.max(), 1.1))))
    ax.bar(range(len(tissues)), ratios, color=colors)
    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_ylabel("baseline_acc / LOTO_acc")
    ax.set_title("Tissue-shortcutting ratio (higher = more tissue identity shortcut)")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.05, f"{r:.2f}×", ha="center", fontsize=8)
    ax.axhline(1, color="k", linewidth=0.5, linestyle="--", alpha=0.4,
               label="no shortcut")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "loto_03_shortcut_ratio.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Weighted F1 comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, b_wf1, w, label="baseline (in-dist)", color="#1f77b4")
    ax.bar(x + w / 2, l_wf1, w, label="LOTO (held out)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("weighted F1")
    ax.set_title("LOTO vs baseline weighted F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "loto_04_weighted_f1.png", bbox_inches="tight")
    plt.close(fig)

    print(f"figures: {args.figures_dir}")
    print(f"{len(results)}/16 tissues complete" + (f" (missing: {missing})" if missing else ""))


if __name__ == "__main__":
    main()
