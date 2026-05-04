"""Training dynamics — Coarse + Medium A pair.

Side-by-side panels showing:
  - train loss (axis L, light line)
  - val macro F1 (axis R, bold line, primary metric)
  - val accuracy (axis R, dashed)
  - early-stop point marked

For Coarse A and Medium A.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style

ROOT = Path("/mnt/work/git/dapidl/pipeline_output/breast_pooled_2026_05")
OUT = Path("/mnt/work/git/dapidl/pipeline_output/figures_v2/fig_a5_training_curves.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def render_panel(ax, name: str, color: str, history: list, summary: dict,
                 ylim_f1: tuple[float, float]):
    epochs = np.array([h["epoch"] for h in history])
    train_loss = np.array([h["train_loss"] for h in history])
    val_f1 = np.array([h["macro_f1"] for h in history])
    val_acc = np.array([h["accuracy"] for h in history])

    # Right axis = F1 + acc
    ax_r = ax.twinx()
    line_f1, = ax_r.plot(epochs, val_f1, "-", lw=2.6, color=color,
                         label="val macro F1", zorder=4)
    ax_r.plot(epochs, val_acc, "--", lw=1.4, color=color, alpha=0.6,
              label="val accuracy")
    # Mark best epoch
    best_ep = summary["best_epoch"]
    best_f1 = summary["best_val_macro_f1"]
    ax_r.scatter([best_ep], [best_f1], s=160, color="gold",
                 edgecolor=color, lw=2.5, zorder=10,
                 label=f"best ep {best_ep} F1={best_f1:.3f}")
    ax_r.axvline(best_ep, color="gold", lw=1, ls=":", alpha=0.7, zorder=1)
    ax_r.set_ylim(*ylim_f1)
    ax_r.set_ylabel("validation F1 / accuracy", color=color, fontsize=11)
    ax_r.tick_params(axis="y", labelcolor=color)

    # Left axis = train loss
    line_loss, = ax.plot(epochs, train_loss, "-", lw=1.5, color="#888",
                         alpha=0.7, label="train loss")
    ax.set_ylim(0, max(train_loss.max() * 1.1, 0.6))
    ax.set_ylabel("train loss", color="#666", fontsize=10)
    ax.tick_params(axis="y", labelcolor="#666")
    ax.set_xlabel("epoch", fontsize=11)

    # Mark early-stop epoch (last epoch in history)
    last_ep = int(epochs[-1])
    if last_ep < 30:  # early-stopped
        ax_r.axvline(last_ep, color="#A33", lw=1.5, ls=":", alpha=0.8)
        ax_r.annotate(f"early stop ep {last_ep}",
                      xy=(last_ep, ylim_f1[0] + (ylim_f1[1] - ylim_f1[0]) * 0.08),
                      xytext=(last_ep - 4, ylim_f1[0] + (ylim_f1[1] - ylim_f1[0]) * 0.18),
                      arrowprops=dict(arrowstyle="->", color="#A33", lw=1),
                      fontsize=10, color="#A33", style="italic", fontweight="bold")

    ax.set_title(name, loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Combined legend
    lines = [line_loss, line_f1]
    labels = [l.get_label() for l in lines]
    ax_r.legend(lines + [ax_r.collections[0]],
                labels + [f"best ep {best_ep}"],
                loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)
    return ax_r


def main() -> None:
    apply_style()
    history_c = json.loads((ROOT / "A_janesick_to_sthelar" / "history.json").read_text())
    summary_c = json.loads((ROOT / "A_janesick_to_sthelar" / "summary.json").read_text())
    history_m = json.loads((ROOT / "A_janesick_to_sthelar_medium" / "history.json").read_text())
    summary_m = json.loads((ROOT / "A_janesick_to_sthelar_medium" / "summary.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.5))
    render_panel(ax1, "A. Coarse A (4-class) — Janesick → STHELAR",
                 "#1A4B7A", history_c, summary_c, ylim_f1=(0.4, 0.85))
    render_panel(ax2, "B. Medium A (12-class) — Janesick → STHELAR",
                 "#A33", history_m, summary_m, ylim_f1=(0.0, 0.6))

    fig.suptitle(
        "Training dynamics — A pair (Coarse 4 + Medium 12)",
        fontsize=14, fontweight="bold", y=1.005,
    )
    fig.text(0.99, 0.005,
             f"Coarse: {len(history_c)} ep, train_time={summary_c['train_time_s']/60:.1f} min, "
             f"early-stopped at ep {len(history_c)-1}.  "
             f"Medium: {len(history_m)} ep, {summary_m['train_time_s']/60:.1f} min.",
             ha="right", va="bottom", fontsize=8.5, color="#666")

    plt.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
