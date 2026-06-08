"""10x10 QC collages with patch IDs, central-box stretch, coarse + medium labels,
both 128 and 64 patch sizes (64 = center-crop of 128).

Five QC groups (per-slide structure_score percentiles):
  Excellent      not broken AND structure_score >= P80 AND objectness >= 0.7
  Good           not broken AND structure_score in [P40, P80)
  Weak-passing   not broken AND structure_score < P40
  Broken-geom    broken_reason in {off_center, cut_at_edge}
  Broken-quality broken_reason in {false_detection, no_nucleus}

Each tile shows its global patch ID (LMDB row index, also the key in
patch_registry.parquet) so the user can refer to specific patches by ID.

Display normalization is a central-box percentile stretch (p2/p99 of the inner
32x32 for 128px, inner 16x16 for 64px) -> focuses the dynamic range on the
target nucleus instead of being washed out by bright neighbors or empty bg.

Outputs:
    pipeline_output/pilot_qc_collages_v2/
        counts_coarse.md, counts_medium.md, counts.parquet
        collages/<size>/<granularity>/<slide>/<class>/<NN>_<group>.png

Usage:
    uv run python scripts/pilot_qc_collages_v2.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402

from dapidl.qc.io import read_patches  # noqa: E402

DEFAULT_DATASET = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p128-nuc-v3")
DEFAULT_OUT = Path("/mnt/work/git/dapidl/pipeline_output/pilot_qc_collages_v2")

GROUP_DEFS = [
    ("01_Excellent", "Excellent"),
    ("02_Good", "Good"),
    ("03_Weak-passing", "Weak-passing"),
    ("04_Broken-geom", "Broken-geom"),
    ("05_Broken-quality", "Broken-quality"),
]
GEOM_REASONS = {"off_center", "cut_at_edge", "edge_cut", "merged_blob"}
QUALITY_REASONS = {"false_detection", "no_nucleus", "flat_interior"}


def assign_groups(df: pl.DataFrame, p80: float, p40: float) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("broken_reason").is_in(list(GEOM_REASONS)))
          .then(pl.lit("Broken-geom"))
        .when(pl.col("broken_reason").is_in(list(QUALITY_REASONS)))
          .then(pl.lit("Broken-quality"))
        .when(pl.col("broken"))
          .then(pl.lit("Broken-other"))
        .when((pl.col("structure_score") >= p80) & (pl.col("objectness_score") >= 0.7))
          .then(pl.lit("Excellent"))
        .when(pl.col("structure_score") >= p40)
          .then(pl.lit("Good"))
        .otherwise(pl.lit("Weak-passing"))
        .alias("qc_group")
    )


def central_stretch(patch: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """p2/p99 stretch from the central frac*frac box. Falls back to full-patch."""
    h, w = patch.shape
    box = max(8, int(min(h, w) * frac))
    y0, x0 = (h - box) // 2, (w - box) // 2
    center = patch[y0:y0 + box, x0:x0 + box].astype(np.float32)
    lo, hi = np.percentile(center, [2, 99])
    if hi <= lo:
        lo, hi = np.percentile(patch.astype(np.float32), [1, 99])
        if hi <= lo:
            return np.zeros_like(patch, dtype=np.float32)
    return np.clip((patch.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def render_collage(patches: np.ndarray, ids: np.ndarray, title: str,
                   out_path: Path, grid: int = 10, size_px: int = 128) -> None:
    """grid x grid montage. Each tile carries its global patch ID (row_idx)."""
    n = min(len(patches), grid * grid)
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.2, grid * 1.25))
    for i in range(grid * grid):
        ax = axes[i // grid, i % grid]
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i < n:
            p = patches[i]
            if size_px != p.shape[0]:
                # 64-mode: center-crop 128 -> 64
                pad = (p.shape[0] - size_px) // 2
                p = p[pad:pad + size_px, pad:pad + size_px]
            disp = central_stretch(p)
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            ax.text(0.03, 0.97, str(int(ids[i])), transform=ax.transAxes,
                    fontsize=5, color="lime", ha="left", va="top",
                    bbox=dict(facecolor="black", alpha=0.5, pad=0.5, linewidth=0))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_for_granularity(qc: pl.DataFrame, reg: pl.DataFrame, label_col: str,
                          collage_root: Path, dataset: Path, grid: int,
                          size_px: int, rng: np.random.Generator,
                          counts_rows: list[dict]) -> None:
    # Join QC scores with the registry on cell_id (LMDB cell_id = row index).
    # qc parquet uses 'source' for the slide name; rename to align with registry.
    joined = (qc.with_columns(pl.col("cell_id").cast(pl.Int64).alias("row_idx"))
                .rename({"source": "slide"})
                .join(reg.select(["row_idx", label_col, "slide"]),
                      on=["row_idx", "slide"], how="left")
                .rename({label_col: "_cls"}))

    granularity = "coarse" if label_col == "coarse_label_name" else "medium"
    out_root = collage_root / f"p{size_px}" / granularity
    out_root.mkdir(parents=True, exist_ok=True)

    for slide in sorted(joined["slide"].unique().to_list()):
        sub = joined.filter(pl.col("slide") == slide)
        passing = sub.filter(~pl.col("broken"))["structure_score"].to_numpy()
        struct_all = sub["structure_score"].to_numpy()
        if len(passing) >= 2:
            p80 = float(np.percentile(passing, 80))
            p40 = float(np.percentile(passing, 40))
        else:
            p80 = float(np.percentile(struct_all, 80)) if len(struct_all) else 1.0
            p40 = float(np.percentile(struct_all, 40)) if len(struct_all) else 0.0
        logger.info(f"  {slide}: P40={p40:.3f} P80={p80:.3f}")
        sub = assign_groups(sub, p80=p80, p40=p40)
        slide_dir = out_root / slide
        slide_dir.mkdir(exist_ok=True)

        for cls in sorted(c for c in sub["_cls"].unique().to_list() if c is not None):
            cls_dir = slide_dir / cls.replace("/", "_").replace(" ", "_")
            cls_dir.mkdir(exist_ok=True)
            cls_sub = sub.filter(pl.col("_cls") == cls)
            for fname, group in GROUP_DEFS:
                g = cls_sub.filter(pl.col("qc_group") == group)
                n_total = g.height
                if n_total == 0:
                    counts_rows.append(dict(size=size_px, granularity=granularity,
                                            slide=slide, cell_type=cls,
                                            qc_group=group, n_total=0, n_shown=0))
                    continue
                k = min(grid * grid, n_total)
                row_idxs = g["row_idx"].to_numpy()
                idxs = rng.choice(row_idxs, size=k, replace=False)
                patches = read_patches(dataset, idxs)
                title = (f"{slide} | {cls} | {group}  "
                         f"(n={n_total:,}, shown {k}, p{size_px})")
                render_collage(patches, idxs, title, cls_dir / f"{fname}.png",
                               grid=grid, size_px=size_px)
                counts_rows.append(dict(size=size_px, granularity=granularity,
                                        slide=slide, cell_type=cls,
                                        qc_group=group, n_total=int(n_total),
                                        n_shown=int(k)))


def write_counts_md(counts: pl.DataFrame, granularity: str, size_px: int,
                    out_path: Path) -> None:
    sub = counts.filter((pl.col("granularity") == granularity) &
                        (pl.col("size") == size_px))
    pivot = (sub.pivot(values="n_total", index=["slide", "cell_type"],
                       on="qc_group", aggregate_function="first")
                .fill_null(0).sort(["slide", "cell_type"]))
    md = [f"# Pilot v2 counts — {granularity} labels, p{size_px}", ""]
    md.append("| slide | cell_type | " + " | ".join([g for _, g in GROUP_DEFS]) + " |")
    md.append("|" + "---|" * (2 + len(GROUP_DEFS)))
    for r in pivot.iter_rows(named=True):
        cells = [r["slide"], r["cell_type"]] + [str(r.get(g, 0)) for _, g in GROUP_DEFS]
        md.append("| " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scores", default="qc/seg_scores.parquet")
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qc = pl.read_parquet(args.dataset / args.scores)
    reg = pl.read_parquet(args.dataset / "patch_registry.parquet")
    if "medium_label" not in reg.columns:
        raise SystemExit("patch_registry.parquet missing medium_label. "
                         "Run scripts/attach_medium_labels.py first.")

    # Add a coarse-name column to the registry for symmetric labelling.
    import json
    cm = json.loads((args.dataset / "class_mapping.json").read_text())
    idx_to_name = {v: k for k, v in cm.items()}
    reg = reg.with_columns(pl.col("coarse_idx")
                             .map_elements(lambda i: idx_to_name.get(int(i), "Unknown"),
                                           return_dtype=pl.Utf8)
                             .alias("coarse_label_name"))

    args.out.mkdir(parents=True, exist_ok=True)
    collage_root = args.out / "collages"
    collage_root.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    counts_rows: list[dict] = []

    for size_px in (128, 64):
        for label_col in ("coarse_label_name", "medium_label"):
            logger.info(f"=== p{size_px} {label_col} ===")
            build_for_granularity(qc, reg, label_col, collage_root, args.dataset,
                                  args.grid, size_px, rng, counts_rows)

    counts = pl.DataFrame(counts_rows)
    counts.write_parquet(args.out / "counts.parquet")
    for size_px in (128, 64):
        for gran in ("coarse", "medium"):
            write_counts_md(counts, gran, size_px,
                            args.out / f"counts_{gran}_p{size_px}.md")
    logger.info(f"wrote {len(counts_rows)} cells to counts; collages under {collage_root}")


if __name__ == "__main__":
    main()
