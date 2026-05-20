"""Per-class worst-first montage of nucleus patches for visual QC."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def build_class_montage(
    patches: np.ndarray,
    scores: np.ndarray,
    cell_type: str,
    top_n: int = 64,
    cols: int = 8,
) -> np.ndarray:
    """Grid of the worst-scoring patches for one cell type.

    Patches are sorted by score ascending (worst first). Each tile's score is
    rendered as a title (margin), never overlaid on the 128px pixels. Returns an
    (H, W, 3) uint8 RGB image.
    """
    scores = np.asarray(scores)
    order = np.argsort(scores)[: min(top_n, len(scores))]
    n = len(order)
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, idx in enumerate(order):
        p = patches[idx].astype(np.float32)
        lo, hi = np.percentile(p, [1, 99])
        norm = np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1)
        axes[i].imshow(norm, cmap="gray")
        axes[i].set_title(f"{scores[idx]:.2f}", fontsize=7)
    fig.suptitle(f"{cell_type} — worst {n} by qc_score", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr
