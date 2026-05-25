"""Generate per-(slide x celltype x qc_group) 6x6 patch collages + counts table.

Five QC groups, defined per-slide so per-source intensity/contrast differences
don't bias the stratification:

  1. Excellent      not broken AND structure_score >= P80 AND objectness >= 0.7
  2. Good           not broken AND structure_score in [P40, P80)
  3. Weak-passing   not broken AND structure_score < P40
  4. Broken-geom    broken_reason in {off_center, edge_cut, merged_blob}
  5. Broken-quality broken_reason in {false_detection, flat_interior}

Output:
    pipeline_output/pilot_qc_collages/
        counts.parquet            (slide x class x qc_group -> n_total, n_shown)
        counts.md                 markdown rendering of counts table
        collages/<slide>/<class>/<NN>_<group>.png

Usage:
    uv run python scripts/pilot_qc_collages.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402

from dapidl.qc.io import read_patches  # noqa: E402

DEFAULT_DATASET = Path("/mnt/work/datasets/derived/breast-pilot-3source-dapi-p128-nuc")
DEFAULT_OUT = Path("/mnt/work/git/dapidl/pipeline_output/pilot_qc_collages")

GROUP_DEFS = [
    ("01_Excellent", "Excellent"),
    ("02_Good", "Good"),
    ("03_Weak-passing", "Weak-passing"),
    ("04_Broken-geom", "Broken-geom"),
    ("05_Broken-quality", "Broken-quality"),
]
GEOM_REASONS = {"off_center", "edge_cut", "merged_blob"}
QUALITY_REASONS = {"false_detection", "flat_interior"}


def assign_groups(df: pl.DataFrame, p80: float, p40: float) -> pl.DataFrame:
    """Add a 'qc_group' column (string) to a per-slide df using per-slide cutoffs."""
    return df.with_columns(
        pl.when(pl.col("broken_reason").is_in(list(GEOM_REASONS)))
          .then(pl.lit("Broken-geom"))
        .when(pl.col("broken_reason").is_in(list(QUALITY_REASONS)))
          .then(pl.lit("Broken-quality"))
        .when(pl.col("broken") & ~pl.col("broken_reason").is_in(list(GEOM_REASONS | QUALITY_REASONS)))
          .then(pl.lit("Broken-other"))
        .when((pl.col("structure_score") >= p80) & (pl.col("objectness_score") >= 0.7))
          .then(pl.lit("Excellent"))
        .when(pl.col("structure_score") >= p40)
          .then(pl.lit("Good"))
        .otherwise(pl.lit("Weak-passing"))
        .alias("qc_group")
    )


def render_collage(patches: np.ndarray, title: str, out_path: Path,
                   grid: int = 6) -> None:
    """6x6 grid (or smaller if patches < 36); per-patch 1-99% contrast stretch."""
    n = min(len(patches), grid * grid)
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.3, grid * 1.3))
    for i in range(grid * grid):
        ax = axes[i // grid, i % grid]
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i < n:
            p = patches[i].astype(np.float32)
            lo, hi = np.percentile(p, [1, 99])
            ax.imshow(np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scores", default="qc/seg_scores.parquet")
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qc_path = args.dataset / args.scores
    if not qc_path.exists():
        raise SystemExit(f"QC scores not found: {qc_path} (run quality_control_seg first)")
    qc = pl.read_parquet(qc_path)
    logger.info(f"loaded {qc.height:,} rows from {qc_path}")

    class_mapping = json.loads((args.dataset / "class_mapping.json").read_text())
    idx_to_class = {v: k for k, v in class_mapping.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    collage_root = args.out / "collages"
    collage_root.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    counts_rows: list[dict] = []

    for slide in sorted(qc["source"].unique().to_list()):
        sub = qc.filter(pl.col("source") == slide)
        struct_vals = sub["structure_score"].to_numpy()
        passing = sub.filter(~pl.col("broken"))["structure_score"].to_numpy()
        if len(passing) >= 2:
            p80 = float(np.percentile(passing, 80))
            p40 = float(np.percentile(passing, 40))
        else:
            p80 = float(np.percentile(struct_vals, 80)) if len(struct_vals) else 1.0
            p40 = float(np.percentile(struct_vals, 40)) if len(struct_vals) else 0.0
        logger.info(f"{slide}: n={sub.height:,}  P40={p40:.3f}  P80={p80:.3f}")

        sub = assign_groups(sub, p80=p80, p40=p40)
        slide_dir = collage_root / slide
        slide_dir.mkdir(exist_ok=True)

        for cls_idx in sorted(idx_to_class.keys()):
            cls_name = idx_to_class[cls_idx]
            cls_dir = slide_dir / cls_name
            cls_dir.mkdir(exist_ok=True)
            cls_sub = sub.filter(pl.col("cell_type") == cls_name)
            for fname, group in GROUP_DEFS:
                g = cls_sub.filter(pl.col("qc_group") == group)
                n_total = g.height
                if n_total == 0:
                    counts_rows.append(dict(slide=slide, cell_type=cls_name,
                                            qc_group=group, n_total=0, n_shown=0))
                    continue
                k = min(args.grid * args.grid, n_total)
                idxs = rng.choice(g["cell_id"].to_numpy(), size=k, replace=False)
                patches = read_patches(args.dataset, idxs)
                title = f"{slide} | {cls_name} | {group}  (n={n_total:,}, shown {k})"
                render_collage(patches, title, cls_dir / f"{fname}.png", grid=args.grid)
                counts_rows.append(dict(slide=slide, cell_type=cls_name,
                                        qc_group=group, n_total=int(n_total), n_shown=int(k)))

    counts = pl.DataFrame(counts_rows)
    counts.write_parquet(args.out / "counts.parquet")

    # Markdown rendering — wide table indexed by (slide, cell_type) x qc_group.
    pivot = (counts.pivot(values="n_total", index=["slide", "cell_type"],
                          on="qc_group", aggregate_function="first")
                   .fill_null(0)
                   .sort(["slide", "cell_type"]))
    md_lines = ["# Pilot QC collage counts (n per group)", ""]
    md_lines.append("| slide | cell_type | " + " | ".join([g for _, g in GROUP_DEFS]) + " |")
    md_lines.append("|" + "---|" * (2 + len(GROUP_DEFS)))
    for row in pivot.iter_rows(named=True):
        cells = [row["slide"], row["cell_type"]] + [
            str(row.get(g, 0)) for _, g in GROUP_DEFS
        ]
        md_lines.append("| " + " | ".join(cells) + " |")
    (args.out / "counts.md").write_text("\n".join(md_lines) + "\n")
    logger.info(f"wrote counts.md and {len(counts_rows)} cells of counts.parquet")
    logger.info(f"collages under {collage_root}")


if __name__ == "__main__":
    main()
