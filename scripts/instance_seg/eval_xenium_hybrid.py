"""Phase D cross-source eval (hybrid).

Two parts (per design doc §5.4):

1. **Quantitative**: apply trained model to Xenium breast rep1, compute
   per-class F1 at MEDIUM tier vs Janesick supervised cell-type GT.

2. **Qualitative**: render 8 ROI cascades (4 from rep1, 4 from rep2) using
   the existing `scripts/_cascade_render.py` infrastructure.

Manuscript-grade boundary GT on Xenium is deferred (see design doc §10).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def quantitative_class_f1(
    pred_predictions_npz: Path,
    rep1_path: Path,
    out_dir: Path,
) -> dict:
    """Compute per-class F1 vs Janesick GT at MEDIUM tier."""
    from dapidl.pipeline.components.annotators.ground_truth import (
        load_janesick_ground_truth,
    )
    from dapidl.pipeline.components.annotators.popv_ensemble import (
        FINE_TO_MEDIUM_MAPPING,
        MEDIUM_CLASS_NAMES,
    )
    from sklearn.metrics import (
        confusion_matrix as _cm,
        f1_score,
    )

    pred = np.load(pred_predictions_npz, allow_pickle=False)
    pred_df = pl.DataFrame(
        {
            "cell_id": [str(c) for c in pred["cell_id"]],
            "pred_medium": [str(m) for m in pred["medium"]],
        }
    )

    gt = load_janesick_ground_truth(rep1_path)
    gt_df = pl.DataFrame(
        {
            "cell_id": gt["cell_id"],
            "gt_fine": gt["cell_type"],
        }
    ).with_columns(
        pl.col("gt_fine")
        .map_elements(
            lambda f: FINE_TO_MEDIUM_MAPPING.get(f, "Unknown"),
            return_dtype=pl.Utf8,
        )
        .alias("gt_medium")
    )

    joined = pred_df.join(gt_df, on="cell_id", how="inner")
    logger.info(f"matched {len(joined):,} cells (pred ∩ gt)")

    valid = joined.filter(pl.col("gt_medium") != "Unknown")
    y_true = valid["gt_medium"].to_list()
    y_pred = valid["pred_medium"].to_list()
    labels = MEDIUM_CLASS_NAMES

    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_macro = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    cm = _cm(y_true, y_pred, labels=labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rep1_confusion_medium.npy", cm)
    summary = {
        "n_cells": int(len(valid)),
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(labels, [float(v) for v in f1_per])),
    }
    (out_dir / "rep1_class_f1.json").write_text(json.dumps(summary, indent=2))
    return summary


def qualitative_cascades(
    pred_predictions_npz: Path,
    rep_paths: list[Path],
    out_dir: Path,
    n_per_rep: int = 4,
) -> None:
    """Render N representative ROI cascades per replicate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_cascade_render",
            Path(__file__).parent.parent / "_cascade_render.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("scripts/_cascade_render.py not loadable")
        cr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cr)
    except Exception as e:
        logger.warning(
            f"cannot import _cascade_render.py ({e}); writing TODO stubs instead"
        )
        for rep_path in rep_paths:
            (out_dir / f"{rep_path.name}_TODO.txt").write_text(
                f"Render {n_per_rep} ROI cascades for {rep_path}\n"
                f"(predictions={pred_predictions_npz})\n"
            )
        return

    for rep_path in rep_paths:
        rep_name = rep_path.name
        logger.info(f"rendering {n_per_rep} ROIs for {rep_name}")
        try:
            cr.render_cascades(  # type: ignore[attr-defined]
                pred_predictions_npz=pred_predictions_npz,
                rep_path=rep_path,
                out_dir=out_dir / rep_name,
                n_rois=n_per_rep,
            )
        except AttributeError:
            (out_dir / f"{rep_name}_TODO.txt").write_text(
                f"_cascade_render.render_cascades() missing — render manually for {rep_path}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-npz", type=Path, required=True)
    ap.add_argument(
        "--rep1-path",
        type=Path,
        default=Path("/home/chrism/datasets/raw/xenium/breast_tumor_rep1"),
    )
    ap.add_argument(
        "--rep2-path",
        type=Path,
        default=Path("/home/chrism/datasets/raw/xenium/breast_tumor_rep2"),
    )
    ap.add_argument(
        "--out", type=Path, default=Path("pipeline_output/instance_seg/eval_xenium")
    )
    args = ap.parse_args()

    logger.info("=== quantitative class F1 (rep1 vs Janesick GT) ===")
    quant = quantitative_class_f1(args.predictions_npz, args.rep1_path, args.out)
    logger.success(
        f"rep1 medium F1 macro = {quant['f1_macro']:.4f} on {quant['n_cells']:,} cells"
    )

    logger.info("=== qualitative cascades (rep1 + rep2) ===")
    qualitative_cascades(
        args.predictions_npz,
        rep_paths=[args.rep1_path, args.rep2_path],
        out_dir=args.out / "cascades",
    )


if __name__ == "__main__":
    main()
