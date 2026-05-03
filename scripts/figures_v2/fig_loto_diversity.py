"""Figure 3 · LOTO generalisation + tissue compositional imbalance.

Two-panel:
A. Per-held-out-tissue test macro F1 (DAPI and H&E), sorted by class diversity.
B. The "brain paradox" — accuracy and macro F1 diverge as class diversity drops.

Class diversity = Shannon entropy of the test split's true label distribution,
normalised by log2(num classes). 0 = single-class tissue; 1 = uniform.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _style import MODALITY_COLORS, apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig03_loto_diversity.png"
LMDB_DIR = Path("/mnt/work/datasets/derived/sthelar-multitissue-p128")


def compute_tissue_class_distribution() -> pl.DataFrame:
    """Per-tissue class proportions and entropy (cached on first call)."""
    cache = ROOT / "figures_v2" / "_tissue_class_dist.parquet"
    if cache.exists():
        return pl.read_parquet(cache)

    with open(LMDB_DIR / "slide_stats.json") as f:
        slides = json.load(f)
    with open(LMDB_DIR / "class_mapping.json") as f:
        cls_map = json.load(f)
    labels = np.load(LMDB_DIR / "labels.npy")

    n_classes = len(cls_map)
    cls_names = [k for k, _ in sorted(cls_map.items(), key=lambda x: x[1])]

    rows = []
    pos = 0
    for slide_name, info in slides.items():
        n = int(info["patches_written"])
        slice_ = labels[pos:pos + n]
        pos += n
        tissue = info["tissue"]
        rows.append({
            "tissue": tissue,
            "slide": slide_name,
            "n": n,
            "labels": slice_,
        })

    by_tissue: dict[str, list[np.ndarray]] = {}
    for r in rows:
        by_tissue.setdefault(r["tissue"], []).append(r["labels"])

    out_rows = []
    for tissue, parts in by_tissue.items():
        all_lab = np.concatenate(parts)
        counts = np.bincount(all_lab, minlength=n_classes).astype(np.float64)
        n_total = float(counts.sum())
        p = counts / max(1.0, n_total)
        nz = p[p > 0]
        H_bits = float(-(nz * np.log2(nz)).sum())
        out_rows.append({
            "tissue": tissue,
            "n_cells": int(n_total),
            "H_bits": H_bits,
            "H_norm": H_bits / float(np.log2(n_classes)),
            **{f"frac_{cls_names[c]}": float(p[c]) for c in range(n_classes)},
        })
    df = pl.DataFrame(out_rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def main() -> None:
    apply_style()
    metrics = pl.read_parquet(ROOT / "model_eval_2026_05" / "master_metrics.parquet")
    loto_main = metrics.filter(pl.col("group").is_in(["loto_dapi", "loto_he"])).with_columns(
        tissue=pl.col("run").str.replace("sthelar_loto_he_", "").str.replace("sthelar_loto_", "")
    )
    # Brain LOTO lives in the `exp` group as `sthelar_exp3_loto_brain`. Pull it
    # in and tag as DAPI so the figure shows the most extreme case.
    brain = (
        metrics.filter(pl.col("run") == "sthelar_exp3_loto_brain")
        .with_columns(
            tissue=pl.lit("brain"),
            modality=pl.lit("DAPI"),
        )
    )
    loto = pl.concat([loto_main, brain], how="diagonal_relaxed")

    div = compute_tissue_class_distribution()
    j = loto.join(div, on="tissue", how="left").sort("H_norm", descending=False)

    dapi = j.filter(pl.col("modality") == "DAPI").sort("H_norm")
    he = j.filter(pl.col("modality") == "HE").sort("H_norm")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.5),
                                   gridspec_kw={"width_ratios": [1.6, 1.0]})

    # Panel A — bar chart of LOTO test F1 (DAPI vs HE) ordered by diversity
    tissues = dapi["tissue"].to_list()
    y = np.arange(len(tissues))
    width = 0.4

    f1_dapi = dapi["test_macro_f1"].to_list()
    he_lookup = {row["tissue"]: row["test_macro_f1"] for row in he.iter_rows(named=True)}
    f1_he = [he_lookup.get(t, np.nan) for t in tissues]

    ax1.barh(y - width / 2, f1_dapi, width, color=MODALITY_COLORS["DAPI"],
             label="DAPI", edgecolor="white", lw=0.8)
    ax1.barh(y + width / 2, f1_he, width, color=MODALITY_COLORS["HE"],
             label="H&E", edgecolor="white", lw=0.8)

    div_lookup = {row["tissue"]: row["H_norm"] for row in div.iter_rows(named=True)}
    labels_y = [f"{t}  (H={div_lookup.get(t, 0):.2f})" for t in tissues]
    ax1.set_yticks(y, labels_y)
    ax1.set_xlabel("LOTO test macro F1 (held-out tissue)")
    ax1.set_xlim(0, 0.4)
    ax1.set_title("A. Cross-tissue generalisation by held-out tissue\n"
                  "(sorted by class diversity, low → high)", loc="left")
    ax1.legend(loc="lower right")
    ax1.invert_yaxis()

    # Panel B — accuracy vs macro F1 colored by diversity
    h_norm = dapi["H_norm"].to_list()
    acc = dapi["test_acc"].to_list()
    sc = ax2.scatter(h_norm, f1_dapi, c=h_norm, cmap="viridis",
                     s=130, edgecolor="black", lw=0.8, label="macro F1")
    ax2.scatter(h_norm, acc, c=h_norm, cmap="viridis",
                s=130, marker="^", edgecolor="black", lw=0.8, label="accuracy")
    for h, f, a, t in zip(h_norm, f1_dapi, acc, tissues):
        if abs(a - f) > 0.4 or t in ("brain", "ovary", "lung", "skin"):
            ax2.annotate(t, (h, max(a, f) + 0.02), ha="center", fontsize=9, color="#222")
    ax2.set_xlabel("class diversity  (H / log₂9)")
    ax2.set_ylabel("DAPI metric value")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("B. The brain paradox\n"
                  "(accuracy ≫ macro F1 when one class dominates)", loc="left")
    ax2.legend(loc="lower right", title="metric")

    fig.suptitle(
        "LOTO performance is bounded by tissue class diversity, not by model capacity",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.text(0.99, 0.005,
             "Each row = one tissue held out from training; F1 is computed on that tissue alone. "
             "Diversity = entropy of test labels normalised to [0,1].",
             ha="right", va="bottom", fontsize=9, color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
