"""Figure 5 · Cross-source transfer — STHELAR ↔ Xenium.

Source files:
  pipeline_output/breast_cross_source/{sthelar,xenium}_to_{xenium,sthelar}_p{32,64,128}.json

Both are breast tissue; the ONLY thing that changes between train and test is
the platform (different microscope, different cohort). Drop in F1 = domain
shift cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import PLATFORM_COLORS, apply_style


ROOT = Path("/mnt/work/git/dapidl/pipeline_output")
OUT = ROOT / "figures_v2" / "fig05_cross_source.png"
SIZES = [32, 64, 128]


def load_run(direction: str, p: int) -> dict:
    path = ROOT / "breast_cross_source" / f"{direction}_p{p}.json"
    return json.loads(path.read_text())


def main() -> None:
    apply_style()

    # In-domain reference (test on same source as training)
    xen_in = []
    sth_in = []
    for p in SIZES:
        xen = json.loads((ROOT / f"breast_dapi_p{p}" / "analysis" / "summary.json").read_text())
        sth = json.loads((ROOT / f"breast_dapi_sthelar_p{p}" / "analysis" / "summary.json").read_text())
        xen_in.append(xen.get("test_macro_f1", xen.get("macro_f1")))
        sth_in.append(sth.get("test_macro_f1", sth.get("macro_f1")))

    s2x = [load_run("sthelar_to_xenium", p)["macro_f1"] for p in SIZES]
    x2s = [load_run("xenium_to_sthelar", p)["macro_f1"] for p in SIZES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)

    x = np.arange(len(SIZES))
    width = 0.35

    # Panel A — train on Xenium
    ax1.bar(x - width / 2, xen_in, width, label="test = Xenium (in-domain)",
            color=PLATFORM_COLORS["Xenium"], edgecolor="white", lw=1.2)
    ax1.bar(x + width / 2, x2s, width, label="test = STHELAR (out-of-domain)",
            color=PLATFORM_COLORS["Xenium"], alpha=0.45, edgecolor="white", lw=1.2)
    for i, (a, b) in enumerate(zip(xen_in, x2s)):
        ax1.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=10)
        ax1.text(i + width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
        ax1.annotate(f"−{(a-b)*100:.0f} pp",
                     xy=(i, max(a, b) + 0.04), ha="center",
                     fontsize=10, color="#A33", fontweight="bold")
    ax1.set_xticks(x, [f"p{s}" for s in SIZES])
    ax1.set_xlabel("patch size")
    ax1.set_ylabel("test macro F1")
    ax1.set_title("A. Train on Xenium →", loc="left")
    ax1.legend(loc="upper left")

    # Panel B — train on STHELAR
    ax2.bar(x - width / 2, sth_in, width, label="test = STHELAR (in-domain)",
            color=PLATFORM_COLORS["STHELAR"], edgecolor="white", lw=1.2)
    ax2.bar(x + width / 2, s2x, width, label="test = Xenium (out-of-domain)",
            color=PLATFORM_COLORS["STHELAR"], alpha=0.45, edgecolor="white", lw=1.2)
    for i, (a, b) in enumerate(zip(sth_in, s2x)):
        ax2.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=10)
        ax2.text(i + width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
        ax2.annotate(f"−{(a-b)*100:.0f} pp",
                     xy=(i, max(a, b) + 0.04), ha="center",
                     fontsize=10, color="#A33", fontweight="bold")
    ax2.set_xticks(x, [f"p{s}" for s in SIZES])
    ax2.set_xlabel("patch size")
    ax2.set_title("B. Train on STHELAR →", loc="left")
    ax2.legend(loc="upper left")

    ax1.set_ylim(0.30, 0.95)

    fig.suptitle(
        "Cross-platform domain shift costs ~10–20 F1 points; bigger patches narrow the gap",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(0.99, 0.005,
             "Both panels are breast tissue. In-domain = held-out test split of the SAME platform; "
             "out-of-domain = full opposite platform.",
             ha="right", va="bottom", fontsize=9, color="#555")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
