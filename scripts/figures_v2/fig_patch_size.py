"""Figure 2 · Patch size sweep — Xenium and STHELAR breast.

Test macro F1 vs nucleus-centered patch size {32, 64, 128, 256}.

Two lines:
- Xenium breast (rep1) — pipeline_output/breast_dapi_p{32,64,128,256}/analysis/summary.json
- STHELAR breast s0   — pipeline_output/breast_dapi_sthelar_p{32,64,128,256}/analysis/summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from _style import PLATFORM_COLORS, apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig02_patch_size_sweep.png"
SIZES = [32, 64, 128, 256]


def load_curve(prefix: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for s in SIZES:
        path = ROOT / f"{prefix}_p{s}" / "analysis" / "summary.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        f1 = d.get("test_macro_f1", d.get("macro_f1"))
        if f1 is None:
            continue
        xs.append(s)
        ys.append(float(f1))
    return xs, ys


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for prefix, label, color in (
        ("breast_dapi", "Xenium breast (rep1, ground truth)", PLATFORM_COLORS["Xenium"]),
        ("breast_dapi_sthelar", "STHELAR breast (s0)", PLATFORM_COLORS["STHELAR"]),
    ):
        xs, ys = load_curve(prefix)
        ax.plot(xs, ys, marker="o", lw=2.5, ms=10, color=color, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=10, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{s}×{s}" for s in SIZES])
    ax.set_xlabel("DAPI patch size (px around nucleus centroid)")
    ax.set_ylabel("test macro F1")
    ax.set_ylim(0.40, 0.85)
    ax.legend(loc="lower right")

    ax.set_title(
        "Larger patches add tissue context — and consistently raise F1\n"
        "EffNetV2-S · Xenium 4-class · STHELAR 6-class · identical hyperparameters",
        loc="left", fontsize=14, pad=12,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
