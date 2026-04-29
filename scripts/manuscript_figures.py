"""Generate publication-ready figures for the DAPIDL manuscript.

Produces 5 figures in docs/figures/:
  fig1_modality_benchmark.png    — three-way modality benchmark + training curves
  fig2_per_class_modality.png    — 9-class F1 across DAPI / H&E / multimodal
  fig3_loto_16tissue.png         — DAPI vs H&E LOTO per-tissue + collapse ratios
  fig4_annotation_vs_janesick.png — annotation methods on Xenium breast rep1 vs GT
  fig5_backbone_comparison.png   — DAPI backbone & foundation-model F1 sweep

Numbers are pulled directly from pipeline_output/ JSON for the modality + LOTO
figures; static numbers from the manuscript are used for the annotation-method
and backbone figures (no JSON exists in the repo for those legacy benchmarks).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/work/git/dapidl")
PO = ROOT / "pipeline_output"
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent palette: DAPI = blue, H&E = pink, Multimodal = purple, Janesick GT = grey-green
C_DAPI = "#1f77b4"
C_HE = "#e377c2"
C_MULTI = "#9467bd"
C_GT = "#2ca02c"
C_GREY = "#7f7f7f"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_modality_summary(mode: str) -> dict:
    return json.load(open(PO / f"sthelar_modality_{mode}/analysis/summary.json"))


def load_modality_per_class(mode: str) -> dict:
    return json.load(open(PO / f"sthelar_modality_{mode}/analysis/per_class_metrics.json"))


def load_modality_history(mode: str) -> list[dict]:
    return json.load(open(PO / f"sthelar_modality_{mode}/history.json"))


def load_loto(modality: str) -> dict[str, dict]:
    """{tissue -> summary} for the per-tissue LOTO sweep on a given modality.

    DAPI brain LOTO lives at sthelar_exp3_loto_brain (it was the first LOTO,
    pre-dating the 16-tissue sweep); the rest live at sthelar_loto_<tissue>.
    H&E LOTO lives at sthelar_loto_he_<tissue> for all 16 tissues.
    """
    out = {}
    if modality == "dapi":
        for d in sorted(PO.glob("sthelar_loto_*/analysis/summary.json")):
            if "_he_" in str(d):
                continue
            s = json.load(open(d))
            out[s["holdout_tissue"]] = s
        brain = PO / "sthelar_exp3_loto_brain/analysis/summary.json"
        if brain.exists() and "brain" not in out:
            s = json.load(open(brain))
            out["brain"] = s
    else:
        for d in sorted(PO.glob(f"sthelar_loto_{modality}_*/analysis/summary.json")):
            s = json.load(open(d))
            out[s["holdout_tissue"]] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Three-way modality benchmark + training curves
# ─────────────────────────────────────────────────────────────────────────────
def fig1_modality_benchmark():
    summaries = {m: load_modality_summary(m) for m in ["dapi", "he", "both"]}
    histories = {m: load_modality_history(m) for m in ["dapi", "he", "both"]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1.0, 1.3]})

    # Panel A — overall metrics
    ax = axes[0]
    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    metric_labels = ["accuracy", "macro F1", "weighted F1"]
    modes = [("dapi", "DAPI", C_DAPI),
             ("he", "H&E", C_HE),
             ("both", "DAPI + H&E", C_MULTI)]
    x = np.arange(len(metrics))
    w = 0.27
    for i, (m, lbl, c) in enumerate(modes):
        vals = [summaries[m][k] for k in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=c, label=lbl, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("score")
    ax.set_title("A. Three-way modality benchmark on STHELAR\n(1.26 M patches, 9 classes, 16 tissues)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # Panel B — val macro F1 training curves
    ax = axes[1]
    for m, lbl, c in modes:
        h = histories[m]
        epochs = [r["epoch"] + 1 for r in h]
        f1 = [r["val_macro_f1"] for r in h]
        ax.plot(epochs, f1, "-o", color=c, label=lbl, ms=4, lw=1.6)
        # mark best epoch
        best_idx = int(np.argmax(f1))
        ax.scatter([epochs[best_idx]], [f1[best_idx]], marker="*",
                   s=180, color=c, edgecolor="black", linewidth=0.7, zorder=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro F1")
    ax.set_title("B. Training trajectories\n(★ best epoch — H&E in 5, DAPI in 9, multimodal in 10)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_ylim(0.30, 0.58)

    fig.suptitle("Figure 1 · Within-STHELAR modality benchmark (matched architecture, splits, hyperparameters)",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig1_modality_benchmark.png")
    plt.close(fig)
    print("✓ fig1_modality_benchmark.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Per-class F1 across modalities
# ─────────────────────────────────────────────────────────────────────────────
def fig2_per_class_modality():
    # Per-class metrics live under per_class_metrics.json -> per_class dict.
    pc = {m: load_modality_per_class(m)["per_class"] for m in ["dapi", "he", "both"]}
    classes_all = list(pc["dapi"].keys())
    supports = {c: pc["dapi"][c]["support"] for c in classes_all}
    classes = sorted(classes_all, key=lambda c: -supports[c])

    def f1_of(mode: str, cls: str) -> float:
        return pc[mode][cls]["f1"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [2.5, 1.0]})

    # Panel A — grouped F1 bars
    ax = axes[0]
    x = np.arange(len(classes))
    w = 0.27
    for i, (m, lbl, c) in enumerate([("dapi", "DAPI", C_DAPI),
                                     ("he", "H&E", C_HE),
                                     ("both", "DAPI + H&E", C_MULTI)]):
        f1s = [f1_of(m, cls) for cls in classes]
        ax.bar(x + (i - 1) * w, f1s, w, color=c, label=lbl, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("A. Per-class test F1 across modalities (sorted by support, descending)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    # Annotate support
    for i, c in enumerate(classes):
        n = supports[c]
        ax.text(i, -0.06, f"n={n:,}", ha="center", va="top", fontsize=7.5, color=C_GREY)

    # Panel B — winner-by-class summary
    ax = axes[1]
    winner_counts = {"DAPI": 0, "H&E": 0, "DAPI + H&E": 0, "tie": 0}
    rows = []
    for cls in classes:
        f1 = {m: f1_of(m, cls) for m in ["dapi", "he", "both"]}
        best = max(f1, key=lambda k: f1[k])
        delta = f1[best] - sorted(f1.values())[-2]
        if delta < 0.005:
            winner_counts["tie"] += 1
        elif best == "dapi":
            winner_counts["DAPI"] += 1
        elif best == "he":
            winner_counts["H&E"] += 1
        else:
            winner_counts["DAPI + H&E"] += 1
        rows.append((cls, best, delta))

    cats = list(winner_counts.keys())
    vals = [winner_counts[k] for k in cats]
    colors = [C_DAPI, C_HE, C_MULTI, C_GREY]
    bars = ax.barh(cats, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.07, b.get_y() + b.get_height() / 2,
                f"{int(v)}", va="center", fontsize=10)
    ax.set_xlabel("# classes won")
    ax.set_xlim(0, max(vals) + 1.5)
    ax.set_title("B. Winner-by-class summary\n(of 9 classes, ≥0.005 F1 lead)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    fig.suptitle("Figure 2 · Per-class complementarity — DAPI vs H&E vs naive multimodal",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig2_per_class_modality.png")
    plt.close(fig)
    print("✓ fig2_per_class_modality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — 16-tissue LOTO DAPI vs H&E
# ─────────────────────────────────────────────────────────────────────────────
def fig3_loto_16tissue():
    dapi_loto = load_loto("dapi")
    he_loto = load_loto("he")

    # Manuscript baseline (from Table 11c) — in-distribution per-tissue accuracy
    baseline = {
        'brain': 0.956, 'bone_marrow': 0.516, 'heart': 0.536, 'bone': 0.460,
        'kidney': 0.865, 'cervix': 0.846, 'breast': 0.694, 'liver': 0.825,
        'lymph_node': 0.553, 'colon': 0.854, 'skin': 0.780, 'pancreatic': 0.778,
        'lung': 0.651, 'tonsil': 0.603, 'prostate': 0.874, 'ovary': 0.894,
    }

    tissues = sorted(set(dapi_loto) & set(he_loto) & set(baseline),
                     key=lambda t: -baseline[t])

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.8),
                             gridspec_kw={"height_ratios": [1.4, 1.0]})

    # Panel A — per-tissue accuracy: baseline / DAPI LOTO / H&E LOTO
    ax = axes[0]
    x = np.arange(len(tissues))
    w = 0.27
    base_v = [baseline[t] for t in tissues]
    dapi_v = [dapi_loto[t]["test_accuracy"] for t in tissues]
    he_v = [he_loto[t]["test_accuracy"] for t in tissues]
    ax.bar(x - w, base_v, w, color=C_GT, label="in-distribution baseline", edgecolor="white")
    ax.bar(x, dapi_v, w, color=C_DAPI, label="DAPI LOTO", edgecolor="white")
    ax.bar(x + w, he_v, w, color=C_HE, label="H&E LOTO", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=35, ha="right")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("A. Per-tissue accuracy: in-distribution vs leave-one-tissue-out (DAPI vs H&E)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    # Means as horizontal lines
    base_mean = np.mean(base_v)
    dapi_mean = np.mean(dapi_v)
    he_mean = np.mean(he_v)
    for v, c, lbl in [(base_mean, C_GT, f"baseline μ={base_mean:.2f}"),
                      (dapi_mean, C_DAPI, f"DAPI LOTO μ={dapi_mean:.2f}"),
                      (he_mean, C_HE, f"H&E LOTO μ={he_mean:.2f}")]:
        ax.axhline(v, color=c, linestyle=":", alpha=0.7, lw=1.4)
        ax.text(len(tissues) - 0.4, v + 0.012, lbl, ha="right", color=c, fontsize=8)

    # Panel B — collapse ratio (baseline / LOTO)
    ax = axes[1]
    dapi_ratio = np.array([baseline[t] / max(dapi_loto[t]["test_accuracy"], 1e-3) for t in tissues])
    he_ratio = np.array([baseline[t] / max(he_loto[t]["test_accuracy"], 1e-3) for t in tissues])
    ax.bar(x - w / 2, dapi_ratio, w, color=C_DAPI, label=f"DAPI (μ={dapi_ratio.mean():.2f}×)", edgecolor="white")
    ax.bar(x + w / 2, he_ratio, w, color=C_HE, label=f"H&E (μ={he_ratio.mean():.2f}×)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, rotation=35, ha="right")
    ax.set_ylabel("collapse ratio  (baseline / LOTO acc)")
    ax.axhline(1.0, color="black", lw=0.7, linestyle="-")
    ax.set_title("B. Tissue-shortcut collapse ratio per modality (1.0 = no shortcut)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    # Annotate brain ratio (the extreme)
    brain_idx = tissues.index("brain")
    ax.annotate(f"brain {dapi_ratio[brain_idx]:.1f}×",
                xy=(brain_idx - w / 2, dapi_ratio[brain_idx]),
                xytext=(brain_idx - 1.5, dapi_ratio[brain_idx] - 0.6),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.7},
                fontsize=9)

    fig.suptitle("Figure 3 · Universal tissue-shortcutting — both image modalities collapse under LOTO",
                 y=1.0, fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig3_loto_16tissue.png")
    plt.close(fig)
    print("✓ fig3_loto_16tissue.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Annotation methods vs Janesick expert GT (Xenium breast rep1)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_annotation_vs_janesick():
    # Numbers from manuscript §3.1 + Annotation Benchmark Results obsidian doc.
    # These are macro F1 against expert-pathologist Janesick GT (3 broad classes).
    data = [
        # (method, accuracy, macro_F1, epi_F1, imm_F1, str_F1, type)
        ("popV ensemble (5 CT + 2 SR)", 0.889, 0.844, 0.98, 0.81, 0.75, "ensemble"),
        ("popV (3 CT + 2 SR)",          0.882, 0.840, 0.97, 0.81, 0.74, "ensemble"),
        ("popV (2 CT + 2 SR)",          0.870, 0.839, 0.95, 0.80, 0.76, "ensemble"),
        ("5 CellTypist consensus",      0.843, 0.737, 0.95, 0.75, 0.52, "consensus"),
        ("SingleR-Blueprint",           0.920, 0.907, None, None, None, "single"),
        ("scType + markers",            0.876, 0.854, None, None, None, "single"),
        ("CellTypist (Cells_Adult_Breast)", 0.907, 0.737, None, None, None, "single"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})

    # Panel A — overall macro F1 ranking
    ax = axes[0]
    methods = [d[0] for d in data]
    f1 = [d[2] for d in data]
    types = [d[6] for d in data]
    color_map = {"ensemble": C_MULTI, "consensus": C_DAPI, "single": C_GREY}
    colors = [color_map[t] for t in types]

    order = np.argsort(f1)
    methods_o = [methods[i] for i in order]
    f1_o = [f1[i] for i in order]
    colors_o = [colors[i] for i in order]
    bars = ax.barh(methods_o, f1_o, color=colors_o, edgecolor="white")
    for b, v in zip(bars, f1_o):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.70, 0.96)
    ax.set_xlabel("macro F1 vs Janesick expert GT")
    ax.set_title("A. Annotation methods on Xenium breast rep1 (167,780 cells, 3 broad)\nGT = Janesick et al. 2023 expert pathologist labels")
    # Manual legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [C_MULTI, C_DAPI, C_GREY]]
    ax.legend(handles, ["popV ensemble", "CellTypist consensus", "single method"],
              loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    # Panel B — per-class F1 for ensembles with per-class data
    ax = axes[1]
    ensembles = [d for d in data if d[3] is not None]
    class_labels = ["Epithelial", "Immune", "Stromal"]
    x = np.arange(len(class_labels))
    n_e = len(ensembles)
    w = 0.8 / n_e
    palette = ["#5a3093", C_MULTI, "#ad6cd1", "#c193dc"][:n_e]
    for i, row in enumerate(ensembles):
        name, _, _, e, im, st, _ = row
        vals = [e, im, st]
        offset = (i - (n_e - 1) / 2) * w
        ax.bar(x + offset, vals, w, color=palette[i],
               label=name, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("per-class F1")
    ax.set_title("B. Per-class F1 (ensembles)\nEpi excellent · Imm strong · Str hardest")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle("Figure 4 · Automatic annotation pipeline benchmarked against expert GT",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(OUT / "fig4_annotation_vs_janesick.png")
    plt.close(fig)
    print("✓ fig4_annotation_vs_janesick.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Backbone & foundation model comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig5_backbone_comparison():
    # From manuscript Table 4 + Table 7 + foundation-model section
    data = [
        # (model, params_M, F1, family, note)
        ("EfficientNetV2-S", 20, 0.596, "supervised CNN", "Xenium 3-class hierarchical"),
        ("ConvNeXt-Tiny",    28, 0.500, "supervised CNN", "Xenium 3-class"),
        ("ResNet-50",        25, 0.490, "supervised CNN", "Xenium 3-class"),
        ("Phikon-v2 ViT-L (frozen)", 304, 0.412, "frozen ViT", "H&E pathology pretrain"),
        ("Cell-DINO ViT-L (LoRA)",   304, 0.463, "LoRA ViT",   "Fluorescence DINO"),
        ("DINOv2 ViT-S (LoRA)",       22, 0.453, "LoRA ViT",   "Natural-image DINO"),
        ("OpenPhenom ViT-S (LoRA)",   22, 0.393, "LoRA ViT",   "Fluorescence MAE"),
        ("NuSPIRe ViT-B (LoRA)",      86, 0.339, "LoRA ViT",   "DAPI MAE (15.5M)"),
        ("Cell-DINO ViT-L (frozen)", 304, 0.407, "frozen ViT", "Fluorescence DINO"),
    ]

    fam_color = {
        "supervised CNN": C_DAPI,
        "frozen ViT":     C_GREY,
        "LoRA ViT":       C_MULTI,
    }

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # Sort by F1 descending
    data.sort(key=lambda r: -r[2])
    names = [d[0] for d in data]
    f1 = [d[2] for d in data]
    colors = [fam_color[d[3]] for d in data]
    bars = ax.barh(range(len(names)), f1, color=colors, edgecolor="white")
    for i, (b, v, d) in enumerate(zip(bars, f1, data)):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}  ({d[1]} M params · {d[4]})",
                va="center", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("DAPI test macro F1")
    ax.set_title("Figure 5 · DAPI backbone & foundation-model comparison\n(coarse classification — Xenium 3-class for CNNs, STHELAR 4-class for foundation models)",
                 fontsize=11, fontweight="bold", loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    ax.legend(handles, list(fam_color.keys()), loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    fig.savefig(OUT / "fig5_backbone_comparison.png")
    plt.close(fig)
    print("✓ fig5_backbone_comparison.png")


if __name__ == "__main__":
    fig1_modality_benchmark()
    fig2_per_class_modality()
    fig3_loto_16tissue()
    fig4_annotation_vs_janesick()
    fig5_backbone_comparison()
    print(f"\nAll figures written to {OUT}")
