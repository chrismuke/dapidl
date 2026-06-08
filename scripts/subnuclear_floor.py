"""Information-floor readout (C): can a LightGBM tree on per-nucleus features match
the EfficientNet's 0.619 macro-F1 on xenium_rep2? Trains nuc-only and nuc+ctx,
tests on ALL rep2 rows, writes floor_metrics.json + feature importances.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subnuclear_common import ctx_feature_columns, nuc_feature_columns  # noqa: E402

CLASS_NAMES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
TRAIN = ["xenium_rep1", "sthelar_breast_s0", "sthelar_breast_s1",
         "sthelar_breast_s3", "sthelar_breast_s6"]
TEST = "xenium_rep2"
EFFNET_MACRO_F1 = 0.619


def _fit_eval(df: pl.DataFrame, feat_cols: list[str], seed: int) -> dict:
    tr = df.filter(pl.col("source").is_in(TRAIN) & (pl.col("label") != -1))
    te = df.filter((pl.col("source") == TEST) & (pl.col("label") != -1))
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(tr)) < 0.1
    Xtr = tr.select(feat_cols).to_pandas()
    ytr = tr["label"].to_numpy()
    Xte = te.select(feat_cols).to_pandas()
    yte = te["label"].to_numpy()
    clf = LGBMClassifier(objective="multiclass", n_estimators=2000,
                         learning_rate=0.05, class_weight="balanced",
                         random_state=seed, n_jobs=-1)
    clf.fit(Xtr[~is_val], ytr[~is_val],
            eval_set=[(Xtr[is_val], ytr[is_val])],
            callbacks=[early_stopping(50), log_evaluation(0)])
    pred = clf.predict(Xte)
    macro = float(f1_score(yte, pred, average="macro", zero_division=0))
    per_class = f1_score(yte, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "n_features": len(feat_cols),
        "n_train": int((~is_val).sum()), "n_test": int(len(yte)),
        "best_iteration": int(clf.best_iteration_ or clf.n_estimators),
        "macro_f1": macro,
        "per_class_f1": {n: float(v) for n, v in zip(CLASS_NAMES, per_class, strict=True)},
        "vs_effnet_0619": macro - EFFNET_MACRO_F1,
        "importances": dict(sorted(
            zip(feat_cols, (int(i) for i in clf.feature_importances_), strict=True),
            key=lambda kv: -kv[1])[:15]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="pipeline_output/subnuclear_2026_06/seg_features.parquet")
    ap.add_argument("--out", default="pipeline_output/subnuclear_2026_06/floor_metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pl.read_parquet(args.features)
    nuc = nuc_feature_columns(df.columns)
    ctx = ctx_feature_columns(df.columns)
    result = {
        "effnet_macro_f1": EFFNET_MACRO_F1,
        "pixel_size_caveat": "area_um2 uses 0.2125 µm/px for all sources",
        "nuc_only": _fit_eval(df, nuc, args.seed),
        "nuc_plus_ctx": _fit_eval(df, nuc + ctx, args.seed),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k]["macro_f1"] for k in ("nuc_only", "nuc_plus_ctx")}, indent=2))
    print(f"[floor] wrote {args.out}  (EffNet ref = {EFFNET_MACRO_F1})")


if __name__ == "__main__":
    main()
