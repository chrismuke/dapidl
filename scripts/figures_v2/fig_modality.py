"""Figure 4 · Modality comparison — DAPI vs H&E vs DAPI+H&E.

Source: master_metrics.parquet, group=modality. Same 9-class STHELAR test set
(189,668 cells), same EfficientNetV2-S backbone, same train/val/test splits.

Story: H&E alone slightly beats DAPI; concat ≈ HE; cross-attention fusion
adds a real but modest +5 percentage points over DAPI alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import MODALITY_COLORS, apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig04_modality.png"


# Display label / variant tag / source modality (for color)
PLOT_ORDER = [
    ("DAPI",            "1-channel adapter",     "DAPI"),
    ("H&E",             "ImageNet RGB",          "HE"),
    ("DAPI + H&E",      "early concat → 1×1",    "DAPI+HE"),
    ("DAPI + H&E",      "late cross-attention",  "Fusion"),
]


def main() -> None:
    apply_style()
    df = pl.read_parquet(ROOT / "model_eval_2026_05" / "master_metrics.parquet")
    df = df.filter(pl.col("group") == "modality")

    runs = {row["run"]: row for row in df.iter_rows(named=True)}
    rows = [
        ("DAPI",         "1ch",     runs["sthelar_modality_dapi"],   "DAPI"),
        ("H&E",          "RGB",     runs["sthelar_modality_he"],     "HE"),
        ("D + H concat", "1×1 conv", runs["sthelar_modality_both"],   "DAPI+HE"),
        ("D + H fusion", "x-attn",  runs["sthelar_modality_fusion"], "Fusion"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    labels = [name for name, _, *_ in rows]
    sublabels = [tag for _, tag, *_ in rows]
    f1_vals = [r["test_macro_f1"] for *_, r, _ in rows]
    acc_vals = [r["test_acc"] for *_, r, _ in rows]
    colors = [MODALITY_COLORS[mod] for *_, mod in rows]

    x_pos = np.arange(len(labels))
    combined = [f"{n}\n({s})" for n, s in zip(labels, sublabels)]

    bars = ax1.bar(x_pos, f1_vals, color=colors, edgecolor="white", lw=1.2)
    for b, v in zip(bars, f1_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(combined, fontsize=10)
    ax1.set_ylabel("test macro F1")
    ax1.set_ylim(0.40, 0.62)
    ax1.set_title("A. Macro F1 across modalities (9-class)", loc="left")

    bars2 = ax2.bar(x_pos, acc_vals, color=colors, edgecolor="white", lw=1.2, alpha=0.85)
    for b, v in zip(bars2, acc_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(combined, fontsize=10)
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0.65, 0.82)
    ax2.set_title("B. Accuracy (less informative under imbalance)", loc="left")

    fig.suptitle(
        "Adding H&E to DAPI helps, but only with cross-attention fusion does it pay off",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.text(
        0.99, 0.01,
        "STHELAR breast s0 + multi-tissue, same 9-class test split (n=189,668), EfficientNetV2-S",
        ha="right", va="bottom", fontsize=9, color="#555",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
