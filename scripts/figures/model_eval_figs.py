#!/usr/bin/env python3
"""Render figures M1, M3, M4, M6 for the trained-model evaluation thread.

Reads from pipeline_output/model_eval_2026_05/{master_metrics,by_class,by_tissue}.parquet
and writes manuscript-grade PNGs (DPI 300) to:
    ~/obsidian/llmbrain/DAPIDL/Pipeline-Deep-Eval-20260501/figures/

Figures:
- M1: Cross-experiment master barchart (test macro F1 sorted, modality-colored)
- M3: Per-tissue × experiment heatmap (16 tissues × selected experiments)
- M4: DAPI vs HE LOTO test-F1 paired barchart by tissue
- M6: Per-class F1 trajectory across the 5 sthelar_exp* experiments + modality runs

Usage:
    uv run python scripts/figures/model_eval_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")

EVAL = Path("/mnt/work/git/dapidl/pipeline_output/model_eval_2026_05")
OUT = Path("/home/chrism/obsidian/llmbrain/DAPIDL/Pipeline-Deep-Eval-20260501/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Cross-figure colour contract
MOD_COLOR = {
    "DAPI": "#1f77b4",
    "HE": "#d62728",
    "DAPI+HE": "#9467bd",
    "FUSION": "#2ca02c",
}
CLASS_COLOR = {
    "epithelial cell": "#1f77b4",
    "T cell": "#d62728",
    "fibroblast": "#2ca02c",
    "endothelial cell": "#9467bd",
    "B cell": "#ff7f0e",
    "macrophage": "#8c564b",
    "pericyte": "#e377c2",
    "mast cell": "#7f7f7f",
    "adipocyte": "#bcbd22",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _modality_color(modality: str, run: str) -> str:
    if "fusion" in run:
        return MOD_COLOR["FUSION"]
    return MOD_COLOR.get(modality, "#888888")


def figure_M1(master: pl.DataFrame) -> None:
    """Cross-experiment master barchart — sorted by test macro F1."""
    df = master.filter(
        pl.col("test_macro_f1").is_not_null() & ~pl.col("group").is_in(["loto_he"])
    ).sort("test_macro_f1", descending=False)

    # For non-LOTO, keep distinct experiments. Truncate LOTO to two anchor points (best/worst)
    keep_loto = df.filter(pl.col("group") == "loto_dapi").sort(
        "test_macro_f1", descending=True
    )
    head_tail = pl.concat([keep_loto.head(3), keep_loto.tail(3)])
    non_loto = df.filter(pl.col("group") != "loto_dapi")
    plot_df = pl.concat([non_loto, head_tail]).sort("test_macro_f1", descending=False)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * len(plot_df))))
    ys = np.arange(len(plot_df))
    f1s = plot_df["test_macro_f1"].to_list()
    colors = [
        _modality_color(m, r)
        for m, r in zip(plot_df["modality"].to_list(), plot_df["run"].to_list())
    ]
    ax.barh(ys, f1s, color=colors, edgecolor="black", linewidth=0.4, height=0.7)
    for y, f1 in zip(ys, f1s):
        ax.text(float(f1) + 0.005, float(y), f"{f1:.3f}", va="center", fontsize=6)
    ax.set_yticks(ys)
    ax.set_yticklabels(plot_df["run"].to_list(), fontsize=6)
    ax.set_xlabel("Test macro F1")
    ax.set_xlim(0, max(f1s) * 1.18)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.set_title(
        "M1 — All trained DAPIDL models, sorted by test macro F1 (LOTO truncated to 3 best + 3 worst)"
    )
    handles = [
        mpatches.Patch(color=c, label=k)
        for k, c in MOD_COLOR.items()
    ]
    ax.legend(handles=handles, loc="lower right", title="Modality / role")
    fig.tight_layout()
    out = OUT / "M1_master_barchart.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_M3(by_t: pl.DataFrame) -> None:
    """Per-tissue × experiment heatmap of macro F1."""
    keep_runs = [
        "sthelar_modality_dapi",
        "sthelar_modality_he",
        "sthelar_modality_both",
        "sthelar_modality_fusion",
        "sthelar_exp1_hierarchical",
        "sthelar_exp2_heavy_aug",
        "sthelar_exp4_vit",
        "sthelar_exp5_7class",
        "sthelar_multitissue_9class",
    ]
    sub = by_t.filter(pl.col("run").is_in(keep_runs))

    pivot = sub.pivot(values="macro_f1", index="tissue", on="run").sort("tissue")
    cols = [c for c in keep_runs if c in pivot.columns]
    matrix = np.array(
        [[pivot[col][i] if col in pivot.columns else np.nan for col in cols]
         for i in range(len(pivot))],
        dtype=float,
    )
    tissues = pivot["tissue"].to_list()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols) + 2),
                                     max(4, 0.35 * len(tissues) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=0.7)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("sthelar_", "") for c in cols],
                       rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tissues)))
    ax.set_yticklabels(tissues)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.4 else "black", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("macro F1")
    ax.set_title("M3 — Per-tissue macro F1 across experiments")
    fig.tight_layout()
    out = OUT / "M3_per_tissue_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_M4() -> None:
    """DAPI vs HE LOTO paired barchart per held-out tissue."""
    # The OVERALL test_macro_f1 from the master == LOTO held-out tissue F1.
    # Use master instead of by_t (skipped per-tissue tables for LOTO runs).
    master = pl.read_parquet(EVAL / "master_metrics.parquet")
    md = master.filter(pl.col("group") == "loto_dapi").select(
        ["subset", "test_macro_f1"]
    ).rename({"test_macro_f1": "f1_dapi"})
    mh = master.filter(pl.col("group") == "loto_he").select(
        ["subset", "test_macro_f1"]
    ).rename({"test_macro_f1": "f1_he"})
    paired = md.join(mh, on="subset", how="full").drop_nulls("subset").sort(
        "f1_dapi", descending=True
    )

    tissues = paired["subset"].to_list()
    f1d = paired["f1_dapi"].to_list()
    f1h = paired["f1_he"].to_list()
    n = len(tissues)
    x = np.arange(n)
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * n + 2), 4))
    ax.bar(x - w / 2, f1d, w, color=MOD_COLOR["DAPI"], label="DAPI", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, [v if v is not None else 0 for v in f1h], w,
           color=MOD_COLOR["HE"], label="HE", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=45, ha="right")
    ax.set_ylabel("Test macro F1 (held-out tissue)")
    ax.set_title("M4 — Leave-One-Tissue-Out: DAPI vs HE per held-out tissue")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max([v or 0 for v in f1d + f1h]) * 1.15)
    fig.tight_layout()
    out = OUT / "M4_loto_dapi_vs_he.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_M6(by_class: pl.DataFrame) -> None:
    """Per-class F1 trajectory across experiments — line plot."""
    track_runs = [
        "sthelar_modality_dapi",
        "sthelar_modality_he",
        "sthelar_modality_both",
        "sthelar_modality_fusion",
        "sthelar_exp1_hierarchical",
        "sthelar_exp2_heavy_aug",
        "sthelar_exp4_vit",
        "sthelar_exp5_7class",
    ]
    sub = by_class.filter(pl.col("run").is_in(track_runs))

    pivot = sub.pivot(values="f1", index="class", on="run")
    cls_order = list(CLASS_COLOR.keys())
    pivot = pivot.with_columns(
        pl.col("class").map_elements(lambda c: cls_order.index(c) if c in cls_order else 99,
                                     return_dtype=pl.Int64).alias("ord")
    ).sort("ord").drop("ord")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(track_runs))
    for cls in pivot["class"].to_list():
        ys = []
        for run in track_runs:
            if run in pivot.columns:
                row = pivot.filter(pl.col("class") == cls)[run].item()
                ys.append(row if row is not None else np.nan)
            else:
                ys.append(np.nan)
        color = CLASS_COLOR.get(cls, "#888888")
        ax.plot(xs, ys, marker="o", color=color, label=cls,
                linewidth=1.2, markersize=4, alpha=0.85)

    ax.set_xticks(xs)
    ax.set_xticklabels([r.replace("sthelar_", "") for r in track_runs],
                       rotation=45, ha="right")
    ax.set_ylabel("Per-class F1")
    ax.set_title("M6 — Per-class F1 trajectory across experiments")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, frameon=False)
    fig.tight_layout()
    out = OUT / "M6_per_class_trajectory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_M2() -> None:
    """4x4 coarse-class confusion matrices for STHELAR-trained models on rep1/rep2."""
    import json
    rep1 = json.loads((EVAL / "cross_platform_rep1.json").read_text())
    rep2 = json.loads((EVAL / "cross_platform_rep2.json").read_text())
    if not rep1 and not rep2:
        print("M2 skipped: no cross-platform results")
        return
    coarse = ["Endo.", "Epi.", "Imm.", "Strom."]

    # Pick the same model from both reps for paired comparison
    models = sorted({r["model"] for r in rep1 + rep2})
    if not models:
        return
    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.4 * n), squeeze=False)
    for i, model_key in enumerate(models):
        for j, (label, results) in enumerate([("rep1", rep1), ("rep2", rep2)]):
            cm = next((r["confusion_matrix"] for r in results
                       if r["model"] == model_key), None)
            ax = axes[i, j]
            if cm is None:
                ax.axis("off")
                continue
            mat = np.asarray(cm, dtype=float)
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            mat_norm = mat / row_sums  # row-normalized recall view
            im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(coarse, fontsize=6)
            ax.set_yticklabels(coarse, fontsize=6)
            for ii in range(4):
                for jj in range(4):
                    v = mat_norm[ii, jj]
                    ax.text(jj, ii, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > 0.5 else "black", fontsize=6)
            short = model_key.replace("sthelar_", "")
            ax.set_title(f"{short} → {label}", fontsize=7)
            if j == 0:
                ax.set_ylabel("True", fontsize=7)
            if i == n - 1:
                ax.set_xlabel("Predicted", fontsize=7)
    fig.suptitle("M2 — Coarse-class confusion (row-normalised) | STHELAR DAPI → Xenium",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    out = OUT / "M2_cross_platform_confusion.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_M5() -> None:
    """Cross-platform transfer barchart: in-domain vs rep1 vs rep2."""
    import json
    rep1 = json.loads((EVAL / "cross_platform_rep1.json").read_text())
    rep2 = json.loads((EVAL / "cross_platform_rep2.json").read_text())
    master = pl.read_parquet(EVAL / "master_metrics.parquet")

    if not rep1 and not rep2:
        print("M5 skipped: no cross-platform results")
        return

    models = sorted({r["model"] for r in rep1 + rep2})
    rows = []
    for m in models:
        sub = master.filter(pl.col("run") == m).select("test_macro_f1")
        in_domain = sub.item(0, 0) if len(sub) else None
        r1 = next((r["macro_f1"] for r in rep1 if r["model"] == m), None)
        r2 = next((r["macro_f1"] for r in rep2 if r["model"] == m), None)
        rows.append({"model": m, "in_domain": in_domain, "rep1": r1, "rep2": r2})

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(rows) + 2), 4))
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, [r["in_domain"] or 0 for r in rows], w, color="#2ca02c",
           label="STHELAR (in-domain)", edgecolor="black", linewidth=0.4)
    ax.bar(x, [r["rep1"] or 0 for r in rows], w, color="#1f77b4",
           label="Xenium rep1 (cross-platform)", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, [r["rep2"] or 0 for r in rows], w, color="#aec7e8",
           label="Xenium rep2 (cross-platform)", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["model"].replace("sthelar_", "") for r in rows],
        rotation=20, ha="right",
    )
    ax.set_ylabel("Macro F1 (coarse 4-class)")
    ax.set_title("M5 — STHELAR-trained DAPI → cross-platform inference on Xenium breast")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right")
    # Add delta callouts
    for i, r in enumerate(rows):
        if r["in_domain"] is not None and r["rep1"] is not None:
            drop = r["in_domain"] - r["rep1"]
            ax.annotate(
                f"−{drop:.2f}", (i, r["rep1"] + 0.02),
                ha="center", fontsize=6, color="#b22222",
            )
    fig.tight_layout()
    out = OUT / "M5_cross_platform_transfer.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    master = pl.read_parquet(EVAL / "master_metrics.parquet")
    by_class = pl.read_parquet(EVAL / "by_class.parquet")
    by_tissue = pl.read_parquet(EVAL / "by_tissue.parquet")
    figure_M1(master)
    figure_M2()
    figure_M3(by_tissue)
    figure_M4()
    figure_M5()
    figure_M6(by_class)


if __name__ == "__main__":
    main()
