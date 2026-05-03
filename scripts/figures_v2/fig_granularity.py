"""Figure 1 · Granularity ladder — what DAPI can predict at each class resolution.

Two panels:
A. Macro F1 vs number of classes (broad → medium → fine).
B. Per-class F1 from the Xenium 4-class breast model (the "what works, what
   doesn't" view) — the same picture every reviewer wants to see first.

Sources:
- 3-class:  pipeline_output/breast_dapi_p128 (4-class actually; we group Endo→Stromal for 3)
- 4-class:  same checkpoint native heads
- 6-class:  pipeline_output/breast_dapi_sthelar_p128 (STHELAR breast, 6-class)
- 7-class:  master_metrics.parquet, sthelar_exp5_7class
- 9-class:  master_metrics.parquet, sthelar_modality_dapi
- 17-class: pipeline_output/breast_dapi_p256/analysis (Xenium fine-grained)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import cell_color, apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig01_granularity.png"


def main() -> None:
    apply_style()
    metrics = pl.read_parquet(ROOT / "model_eval_2026_05" / "master_metrics.parquet")

    # --- Panel A: macro F1 vs n_classes ---------------------------------
    runs_to_show = []

    # 4-class Xenium breast
    p128 = json.loads((ROOT / "breast_dapi_p128" / "analysis" / "summary.json").read_text())
    f1_4 = p128.get("test_macro_f1", p128.get("macro_f1"))
    runs_to_show.append((4, f1_4, "Xenium breast 4-class"))

    # 6-class STHELAR breast
    sth = json.loads((ROOT / "breast_dapi_sthelar_p128" / "analysis" / "summary.json").read_text())
    f1_6 = sth.get("test_macro_f1", sth.get("macro_f1"))
    runs_to_show.append((6, f1_6, "STHELAR breast 6-class"))

    # 7-class STHELAR multi-tissue
    row7 = metrics.filter(pl.col("run") == "sthelar_exp5_7class").row(0, named=True)
    runs_to_show.append((7, row7["test_macro_f1"], "STHELAR multi-tissue 7-class"))

    # 9-class STHELAR multi-tissue
    row9 = metrics.filter(pl.col("run") == "sthelar_modality_dapi").row(0, named=True)
    runs_to_show.append((9, row9["test_macro_f1"], "STHELAR multi-tissue 9-class"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.5))

    xs = [r[0] for r in runs_to_show]
    ys = [r[1] for r in runs_to_show]
    labels = [r[2] for r in runs_to_show]
    colors = ["#2E86AB", "#3D5A80", "#7B5EA7", "#A04D9D"]
    ax1.plot(xs, ys, "-", lw=2.0, color="#999", zorder=1)
    for x, y, lbl, c in zip(xs, ys, labels, colors):
        ax1.scatter(x, y, s=180, color=c, zorder=3, edgecolor="white", lw=1.5,
                    label=f"{x}-class · {lbl.split(maxsplit=1)[1] if ' ' in lbl else lbl}")
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=11, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xlabel("number of classes (granularity)")
    ax1.set_ylabel("test macro F1")
    ax1.set_ylim(0.40, 0.85)
    ax1.set_title("A. Granularity → F1: more classes = lower ceiling", loc="left")
    ax1.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)

    # --- Panel B: per-class F1 from a single model ----------------------
    pcm = json.loads((ROOT / "breast_dapi_p128" / "analysis" /
                      "per_class_metrics.json").read_text())
    per = pcm.get("per_class", pcm)
    items = sorted(per.items(), key=lambda kv: kv[1].get("f1", 0.0), reverse=True)
    names = [k for k, _ in items]
    f1s = [v.get("f1", 0.0) for _, v in items]
    sup = [v.get("support", 0) for _, v in items]
    colors_b = [cell_color(n) for n in names]
    y = np.arange(len(names))
    bars = ax2.barh(y, f1s, color=colors_b, edgecolor="white", lw=1.2)
    for bar, f, s in zip(bars, f1s, sup):
        ax2.text(f + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{f:.2f}  ({s/1000:.0f}k cells)", va="center", fontsize=10)
    ax2.set_yticks(y, names)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("test F1 per class")
    ax2.set_title("B. Per-class F1 on Xenium breast (4-class)", loc="left")

    fig.suptitle(
        "DAPI works well at coarse resolution, drops with granularity, "
        "and stalls on rare/diverse classes",
        fontsize=14, fontweight="bold", y=1.04,
    )
    fig.text(0.99, 0.005,
             "All runs use EfficientNetV2-S, 128 px patches, identical hyperparameters; "
             "F1 is held-out test macro.",
             ha="right", va="bottom", fontsize=9, color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
