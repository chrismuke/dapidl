"""Re-score trained models on QC-FILTERED test sets.

Separates "model can't generalize" from "the test set has low-quality nuclei":
for each trained model, score the held-out test sources (Janesick rep1/rep2 +
Prime s6) at increasing QC thresholds. If F1 rises as we keep only higher-QC
test nuclei, much of the apparent generalization gap was bad test patches /
segmentation, not the model.

Run AFTER the qc_threshold_experiment.sh training completes (inference only).
"""

import sys
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from breast_pooled_train import (  # noqa: E402
    LMDB_DIR,
    DapiClassifier,
    DapiPatchDataset,
    load_class_names,
    load_indices_for_sources,
    load_labels,
    score_loader,
)

MODELS = {
    "baseline_B": "pipeline_output/breast_pooled_2026_05/B_sthelar_std_to_janesick_prime",
    "qc015": "pipeline_output/qc_threshold_2026_05/B_qc015",
    "qc025": "pipeline_output/qc_threshold_2026_05/B_qc025",
    "random065": "pipeline_output/qc_threshold_2026_05/B_random065",
}
TEST_SOURCES = ["xenium_rep1", "xenium_rep2", "sthelar_breast_s6"]
TEST_THRESHOLDS = [0.0, 0.15, 0.25]  # 0.0 == drop -1 only (matches existing summaries)


def main() -> None:
    tier = "coarse"
    class_names = load_class_names(tier)
    labels_full = load_labels(tier)
    qc = pl.read_parquet(LMDB_DIR / "qc" / "qc_scores.parquet").sort("cell_id")["qc_score"].to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for mname, mdir in MODELS.items():
        ckpt = Path(mdir) / "best_model.pt"
        if not ckpt.exists():
            logger.warning(f"skip {mname}: no checkpoint at {ckpt}")
            continue
        model = DapiClassifier(len(class_names)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.train(False)  # evaluation mode (codebase style; avoids builtin name)
        for thr in TEST_THRESHOLDS:
            for src in TEST_SOURCES:
                idx = load_indices_for_sources([src])
                idx = idx[labels_full[idx] != -1]
                idx = idx[qc[idx] >= thr]
                if len(idx) == 0:
                    continue
                loader = DataLoader(
                    DapiPatchDataset(idx, labels_full, augment=False),
                    batch_size=256, shuffle=False, num_workers=6, pin_memory=True,
                )
                m = score_loader(model, loader, device, class_names)
                rows.append(dict(model=mname, test_thresh=thr, source=src,
                                 n=int(len(idx)), macro_f1=float(m["macro_f1"]),
                                 accuracy=float(m["accuracy"])))
                logger.info(f"{mname:12s} test_qc>={thr:.2f} {src:20s} "
                            f"n={len(idx):>7d} F1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}")

    df = pl.DataFrame(rows)
    out = Path("pipeline_output/qc_threshold_2026_05/test_qc_eval.parquet")
    df.write_parquet(out)
    logger.info(f"wrote {out}")
    summary = (df.group_by(["model", "test_thresh"])
                 .agg(pl.col("macro_f1").mean().round(3).alias("mean_f1"))
                 .sort(["model", "test_thresh"]))
    print(summary)


if __name__ == "__main__":
    main()
