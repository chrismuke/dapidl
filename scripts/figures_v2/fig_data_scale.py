"""Figure 6 · Training-data scale curve.

Source: pipeline_output/data_scale_2026_05/frac_*/summary.json (5 fractions).

Two panels:
A. Macro F1 vs fraction of train pool (log-scaled x).
B. Per-class F1 at each fraction (small multiples) — shows that rare classes
   need *much* more data than common classes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import cell_color, apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
SCALE_DIR = ROOT / "data_scale_2026_05"
OUT = ROOT / "figures_v2" / "fig06_data_scale.png"


def load_summaries() -> list[dict]:
    rows = []
    for d in sorted(SCALE_DIR.glob("frac_*/")):
        s = d / "summary.json"
        if not s.exists():
            continue
        rows.append(json.loads(s.read_text()))
    rows.sort(key=lambda r: r["fraction"])
    return rows


def main() -> None:
    apply_style()
    rows = load_summaries()
    if not rows:
        raise SystemExit(f"no completed runs under {SCALE_DIR}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.4]})
    fig.subplots_adjust(top=0.85, wspace=0.28)

    fractions = [r["fraction"] for r in rows]
    n_train = [r["n_train"] for r in rows]
    f1 = [r["test_macro_f1"] for r in rows]
    acc = [r["test_accuracy"] for r in rows]

    # Panel A — main learning curve
    ax1.plot(n_train, f1, "o-", lw=2.5, ms=12, color="#3D5A80",
             label="test macro F1")
    ax1.plot(n_train, acc, "s--", lw=1.5, ms=9, color="#999999",
             label="test accuracy")
    for n, y in zip(n_train, f1):
        ax1.annotate(f"{y:.3f}", (n, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlabel("training cells (log scale)")
    ax1.set_ylabel("test metric")
    ax1.set_ylim(0.20, 1.00)
    ax1.set_title("A. F1 climbs · accuracy stays flat", loc="left", pad=10)
    ax1.legend(loc="lower right")

    # Panel B — per-class F1 small multiples
    class_names = rows[-1]["class_names"]
    for cls_idx, cls in enumerate(class_names):
        ys = []
        for r in rows:
            entry = next(e for e in r["per_class"] if e["class_name"] == cls)
            ys.append(entry["f1"])
        ax2.plot(n_train, ys, "o-", lw=1.8, ms=8,
                 color=cell_color(cls), label=cls)
    ax2.set_xscale("log")
    ax2.set_xlabel("training cells (log scale)")
    ax2.set_ylabel("per-class test F1")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("B. Rare classes benefit most from extra data",
                  loc="left", pad=10)
    ax2.legend(loc="center right", fontsize=9, ncol=1)

    fig.suptitle(
        "Training-data scale: more data still helps, but rare classes benefit most",
        fontsize=14, fontweight="bold", y=0.98,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}  (n_runs={len(rows)})")


if __name__ == "__main__":
    main()
