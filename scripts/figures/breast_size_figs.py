#!/usr/bin/env python3
"""Render figures S1-S4 for the breast-DAPI patch-size sweep.

S1: Macro F1 vs patch size (single curve)
S2: Per-class F1 vs patch size (4 lines)
S3: Cascade-style visual reference (DAPI patches at 4 sizes around 1 example cell)
S4: Inference throughput vs patch size (synthetic / measured)

Reads from pipeline_output/breast_size_sweep/size_metrics.parquet (and
size_per_class.parquet if available). Writes to:
    ~/obsidian/llmbrain/DAPIDL/Pipeline-Deep-Eval-20260501/figures/

Usage:
    uv run python scripts/figures/breast_size_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")

SWEEP = Path("/mnt/work/git/dapidl/pipeline_output/breast_size_sweep")
OUT = Path("/home/chrism/obsidian/llmbrain/DAPIDL/Pipeline-Deep-Eval-20260501/figures")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_COLOR = {
    "Endothelial": "#9467bd",
    "Epithelial": "#1f77b4",
    "Immune": "#d62728",
    "Stromal": "#2ca02c",
}


def figure_S1() -> None:
    p = SWEEP / "size_metrics.parquet"
    if not p.exists():
        print("S1 skipped: no size_metrics.parquet (run scripts/breast_size_compare.py)")
        return
    df = pl.read_parquet(p).sort("patch_size")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["patch_size"], df["test_macro_f1"], marker="o",
            color="#1f77b4", linewidth=1.4, markersize=7,
            label="Test macro F1 (4-class)")
    ax.plot(df["patch_size"], df["test_weighted_f1"], marker="s",
            color="#aec7e8", linewidth=1.0, markersize=6,
            linestyle="--", label="Test weighted F1")
    for x, y in zip(df["patch_size"].to_list(), df["test_macro_f1"].to_list()):
        ax.annotate(f"{y:.3f}", (x, float(y)), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["patch_size"].to_list())
    ax.set_xticklabels([str(x) for x in df["patch_size"].to_list()])
    ax.set_xlabel("Patch size (px, log2 scale)")
    ax.set_ylabel("Test F1")
    ax.set_title("S1 — Breast DAPI macro F1 vs patch size (4-class coarse)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path = OUT / "S1_size_vs_macro_f1.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def figure_S2() -> None:
    p = SWEEP / "size_per_class.parquet"
    if not p.exists():
        print("S2 skipped: no per-class parquet")
        return
    df = pl.read_parquet(p).sort(["class", "patch_size"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in df["class"].unique().to_list():
        sub = df.filter(pl.col("class") == cls).sort("patch_size")
        color = CLASS_COLOR.get(cls, "#888888")
        ax.plot(sub["patch_size"], sub["f1"], marker="o", color=color,
                linewidth=1.2, label=cls, markersize=5)
    ax.set_xscale("log", base=2)
    sizes = sorted(df["patch_size"].unique().to_list())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Patch size (px, log2)")
    ax.set_ylabel("Per-class F1")
    ax.set_title("S2 — Per-class F1 vs patch size (breast DAPI 4-class)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    out_path = OUT / "S2_per_class_vs_size.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def figure_S3_cascade_reference() -> None:
    """Pick one DAPI patch from the existing rep1 LMDB and show it at 4 sizes."""
    import lmdb
    import struct

    rep1_lmdb = Path("/mnt/work/datasets/derived/xenium-breast-tumor-rep1-local-finegrained-p128") / "patches.lmdb"
    if not rep1_lmdb.exists():
        print("S3 skipped: no rep1 p128 LMDB")
        return

    env = lmdb.open(str(rep1_lmdb), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        value = txn.get(struct.pack(">Q", 1234))  # arbitrary cell
    if value is None:
        return
    # Skip 8-byte label, then 128*128*2 bytes uint16
    patch128 = np.frombuffer(value[8:], dtype=np.uint16).reshape(128, 128)
    env.close()

    # Simulate p32, p64 by center-crop. p256 isn't available from this LMDB.
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    sizes = [32, 64, 128, 256]
    for ax, sz in zip(axes, sizes):
        if sz <= 128:
            half = sz // 2
            cy, cx = 64, 64
            crop = patch128[cy - half:cy + half, cx - half:cx + half]
        else:
            # Pad p128 to p256 with zeros — illustrative only
            crop = np.zeros((sz, sz), dtype=np.uint16)
            off = (sz - 128) // 2
            crop[off:off + 128, off:off + 128] = patch128
        # Adaptive normalize for display
        lo, hi = np.percentile(crop, [2, 98])
        crop_disp = np.clip((crop.astype(float) - lo) / max(hi - lo, 1), 0, 1)
        ax.imshow(crop_disp ** 0.7, cmap="gray", aspect="equal")
        ax.set_title(f"p{sz}", fontsize=9)
        ax.axis("off")
        if sz > 128:
            ax.text(
                sz // 2, sz - 10, "(padded — actual p256 needs new LMDB)",
                ha="center", color="red", fontsize=6,
            )
    fig.suptitle("S3 — Same rep1 nucleus visualised at 4 patch sizes", y=1.02)
    fig.tight_layout()
    out_path = OUT / "S3_cascade_reference.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def figure_S4_throughput() -> None:
    """Synthetic throughput curve (will be replaced with measured)."""
    sizes = np.array([32, 64, 128, 256, 512])
    # Modeled: throughput scales as 1 / patch_size^2 roughly, with constant overhead
    base_fps = 8000  # at p32
    fps = base_fps / (sizes / 32) ** 2
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sizes, fps, marker="o", color="#9467bd", linewidth=1.3,
            label="Modeled (1/size²)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Patch size (px)")
    ax.set_ylabel("Throughput (cells/s, single GPU, log)")
    ax.set_title("S4 — Inference throughput vs patch size (modeled, batch=256)")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    out_path = OUT / "S4_throughput.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  (model — replace with measured after sweep)")


def main() -> None:
    figure_S1()
    figure_S2()
    figure_S3_cascade_reference()
    figure_S4_throughput()


if __name__ == "__main__":
    main()
