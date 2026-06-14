#!/usr/bin/env python3
"""Render cross-source breast DAPI figures.

Reads:
- pipeline_output/breast_dapi_p{32,64,128,256}/analysis/summary.json    (Xenium in-domain)
- pipeline_output/breast_dapi_sthelar_p{32,64,128,256}/analysis/summary.json (STHELAR in-domain)
- pipeline_output/breast_cross_source/{sthelar,xenium}_to_*_p{N}.json   (cross-source)

Writes:
- C1: 2x2 matrix (in-domain vs cross-source) per patch size
- C2: F1 vs patch size, 4 lines (Xen→Xen, Xen→STHELAR, STHELAR→STHELAR, STHELAR→Xen)
- C3: Per-class cross-source delta heatmap

Usage:
    uv run python scripts/figures/breast_cross_source_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PIPELINE = Path("/mnt/work/git/dapidl/pipeline_output")
CROSS = PIPELINE / "breast_cross_source"
OUT = Path("/home/chrism/obsidian/llmbrain/DAPIDL/Pipeline-Deep-Eval-20260501/figures")
OUT.mkdir(parents=True, exist_ok=True)

PATCH_SIZES = [32, 64, 128, 256]
COARSE = ["Endothelial", "Epithelial", "Immune", "Stromal"]


def _read_in_domain(family: str) -> dict[int, dict]:
    """family ∈ {'p', 'sthelar_p'} → {patch_size: summary_dict}."""
    out: dict[int, dict] = {}
    for sz in PATCH_SIZES:
        prefix = "breast_dapi" if family == "xenium" else "breast_dapi_sthelar"
        f = PIPELINE / f"{prefix}_p{sz}" / "analysis" / "summary.json"
        if f.exists():
            out[sz] = json.loads(f.read_text())
    return out


def _read_cross(direction: str) -> dict[int, dict]:
    """direction ∈ {'sthelar_to_xenium', 'xenium_to_sthelar'} → {size: dict}."""
    out: dict[int, dict] = {}
    for sz in PATCH_SIZES:
        f = CROSS / f"{direction}_p{sz}.json"
        if f.exists():
            out[sz] = json.loads(f.read_text())
    return out


def figure_C1() -> None:
    """4-line plot: in-domain Xenium, in-domain STHELAR, X→S, S→X across patch sizes."""
    xx = _read_in_domain("xenium")    # Xenium → Xenium (in-domain)
    ss = _read_in_domain("sthelar")   # STHELAR → STHELAR (in-domain)
    sx = _read_cross("sthelar_to_xenium")
    xs = _read_cross("xenium_to_sthelar")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = [
        ("Xenium → Xenium (in-domain)", xx, "#1f77b4", "test_macro_f1", "o"),
        ("STHELAR → STHELAR (in-domain)", ss, "#2ca02c", "test_macro_f1", "s"),
        ("Xenium → STHELAR (cross)", xs, "#9467bd", "macro_f1", "^"),
        ("STHELAR → Xenium (cross)", sx, "#d62728", "macro_f1", "v"),
    ]
    for label, data, color, key, marker in series:
        if not data:
            continue
        sizes = sorted(data.keys())
        ys = [data[s].get(key) for s in sizes]
        ax.plot(sizes, ys, marker=marker, label=label, color=color,
                linewidth=1.4, markersize=7)
        for x, y in zip(sizes, ys):
            if y is not None:
                ax.annotate(f"{y:.3f}", (x, float(y)),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(PATCH_SIZES)
    ax.set_xticklabels([str(s) for s in PATCH_SIZES])
    ax.set_xlabel("Patch size (px, log2)")
    ax.set_ylabel("Macro F1 (4-class)")
    ax.set_title("C1 — In-domain vs cross-source DAPI breast performance vs patch size")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    out = OUT / "C1_in_domain_vs_cross_source.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_C2() -> None:
    """4x4 transfer matrix at one patch size — replicate per size."""
    xx = _read_in_domain("xenium")
    ss = _read_in_domain("sthelar")
    sx = _read_cross("sthelar_to_xenium")
    xs = _read_cross("xenium_to_sthelar")

    fig, axes = plt.subplots(1, len(PATCH_SIZES), figsize=(3 * len(PATCH_SIZES), 3.5))
    if len(PATCH_SIZES) == 1:
        axes = [axes]

    for ax, sz in zip(axes, PATCH_SIZES):
        # matrix[train_idx][test_idx]: 0=Xenium, 1=STHELAR
        m = np.full((2, 2), np.nan)
        if sz in xx:
            m[0, 0] = xx[sz].get("test_macro_f1", np.nan)
        if sz in xs:
            m[0, 1] = xs[sz].get("macro_f1", np.nan)
        if sz in sx:
            m[1, 0] = sx[sz].get("macro_f1", np.nan)
        if sz in ss:
            m[1, 1] = ss[sz].get("test_macro_f1", np.nan)
        im = ax.imshow(m, vmin=0, vmax=0.85, cmap="viridis", aspect="equal")
        for i in range(2):
            for j in range(2):
                v = m[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Xenium", "STHELAR"])
        ax.set_yticklabels(["Xenium", "STHELAR"])
        ax.set_xlabel("Test on")
        ax.set_ylabel("Train on")
        ax.set_title(f"p{sz}", fontsize=9)
    fig.suptitle("C2 — Train×Test transfer matrix per patch size (macro F1)", y=1.02)
    fig.tight_layout()
    out = OUT / "C2_transfer_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_C3() -> None:
    """Cross-source per-class F1 vs patch size — 4 panels (one per class)."""
    sx = _read_cross("sthelar_to_xenium")
    xs = _read_cross("xenium_to_sthelar")
    xx = _read_in_domain("xenium")
    ss = _read_in_domain("sthelar")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes_f = axes.flatten()
    for ai, cls in enumerate(COARSE):
        ax = axes_f[ai]
        for label, data, color in [
            ("Xenium → Xenium", xx, "#1f77b4"),
            ("STHELAR → STHELAR", ss, "#2ca02c"),
            ("Xenium → STHELAR", xs, "#9467bd"),
            ("STHELAR → Xenium", sx, "#d62728"),
        ]:
            sizes, ys = [], []
            for sz in PATCH_SIZES:
                if sz in data:
                    pc = data[sz].get("per_class", {})
                    if cls in pc and pc[cls] is not None:
                        sizes.append(sz)
                        ys.append(pc[cls].get("f1"))
            if sizes:
                ax.plot(sizes, ys, marker="o", label=label, color=color,
                        linewidth=1.2, markersize=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(PATCH_SIZES)
        ax.set_xticklabels([str(s) for s in PATCH_SIZES])
        ax.set_title(cls, fontsize=9)
        ax.set_ylabel("F1")
        ax.set_xlabel("Patch size")
        ax.set_ylim(0, 1)
        if ai == 0:
            ax.legend(loc="upper left", fontsize=6)
    fig.suptitle("C3 — Per-class F1 vs patch size, by train/test pair", y=1.0)
    fig.tight_layout()
    out = OUT / "C3_per_class_cross_source.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    figure_C1()
    figure_C2()
    figure_C3()


if __name__ == "__main__":
    main()
