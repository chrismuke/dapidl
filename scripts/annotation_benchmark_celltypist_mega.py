#!/usr/bin/env python3
"""Mega CellTypist + Consensus Benchmark.

Tests ALL 47 human CellTypist models individually, plus:
- Every 2-model combination of top performers
- Every 3-model combination of top performers
- Tiered ensembles (tissue-specific + immune + general)
- CellTypist + SingleR hybrid ensembles
- CellTypist + SCINA hybrid ensembles
- Universal ensemble (all models at once)
- Weighted consensus with different strategies

Usage:
    uv run python scripts/annotation_benchmark_celltypist_mega.py
"""

import gc
import itertools
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import celltypist
import numpy as np
import scanpy as sc
from celltypist import models as ct_models
from loguru import logger
from scipy.sparse import issparse
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotation_benchmark_2026_03 import (
    COARSE_CLASSES,
    compute_metrics,
    load_xenium_adata,
    map_predictions_to_coarse,
    preprocess_adata,
)

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")

# All 47 human CellTypist models
ALL_HUMAN_MODELS = [
    "Adult_COVID19_PBMC.pkl", "Adult_Human_MTG.pkl", "Adult_Human_PancreaticIslet.pkl",
    "Adult_Human_PrefrontalCortex.pkl", "Adult_Human_Skin.pkl", "Adult_Human_Vascular.pkl",
    "Adult_cHSPCs_Illumina.pkl", "Adult_cHSPCs_Ultima.pkl", "Autopsy_COVID19_Lung.pkl",
    "COVID19_HumanChallenge_Blood.pkl", "COVID19_Immune_Landscape.pkl",
    "Cells_Adult_Breast.pkl", "Cells_Fetal_Lung.pkl", "Cells_Human_Tonsil.pkl",
    "Cells_Intestinal_Tract.pkl", "Cells_Lung_Airway.pkl",
    "Developing_Human_Brain.pkl", "Developing_Human_Gonads.pkl",
    "Developing_Human_Hippocampus.pkl", "Developing_Human_Organs.pkl",
    "Developing_Human_Thymus.pkl", "Fetal_Human_AdrenalGlands.pkl",
    "Fetal_Human_Pancreas.pkl", "Fetal_Human_Pituitary.pkl", "Fetal_Human_Retina.pkl",
    "Fetal_Human_Skin.pkl", "Healthy_Adult_Heart.pkl", "Healthy_COVID19_PBMC.pkl",
    "Healthy_Human_Liver.pkl", "Human_AdultAged_Hippocampus.pkl",
    "Human_Colorectal_Cancer.pkl", "Human_Developmental_Retina.pkl",
    "Human_Embryonic_YolkSac.pkl", "Human_Endometrium_Atlas.pkl",
    "Human_IPF_Lung.pkl", "Human_Longitudinal_Hippocampus.pkl",
    "Human_Lung_Atlas.pkl", "Human_PF_Lung.pkl", "Human_Placenta_Decidua.pkl",
    "Immune_All_High.pkl", "Immune_All_Low.pkl", "Lethal_COVID19_Lung.pkl",
    "Nuclei_Human_InnerEar.pkl", "Nuclei_Lung_Airway.pkl",
    "PaediatricAdult_COVID19_Airway.pkl", "PaediatricAdult_COVID19_PBMC.pkl",
    "Pan_Fetal_Human.pkl",
]

# Semantic groupings for smart ensembles
BREAST_RELEVANT = ["Cells_Adult_Breast.pkl", "Adult_Human_Vascular.pkl", "Adult_Human_Skin.pkl",
                    "Human_Endometrium_Atlas.pkl"]
IMMUNE_MODELS = ["Immune_All_High.pkl", "Immune_All_Low.pkl", "COVID19_Immune_Landscape.pkl",
                  "Adult_COVID19_PBMC.pkl", "Healthy_COVID19_PBMC.pkl", "COVID19_HumanChallenge_Blood.pkl",
                  "PaediatricAdult_COVID19_PBMC.pkl", "Developing_Human_Thymus.pkl", "Cells_Human_Tonsil.pkl"]
PAN_TISSUE = ["Pan_Fetal_Human.pkl", "Developing_Human_Organs.pkl"]
LUNG_MODELS = ["Human_Lung_Atlas.pkl", "Cells_Lung_Airway.pkl", "Nuclei_Lung_Airway.pkl",
               "Human_IPF_Lung.pkl", "Human_PF_Lung.pkl", "Cells_Fetal_Lung.pkl",
               "Autopsy_COVID19_Lung.pkl", "Lethal_COVID19_Lung.pkl",
               "PaediatricAdult_COVID19_Airway.pkl"]
GENERAL_MODELS = ["Healthy_Human_Liver.pkl", "Healthy_Adult_Heart.pkl",
                   "Human_Colorectal_Cancer.pkl", "Cells_Intestinal_Tract.pkl",
                   "Adult_Human_PancreaticIslet.pkl"]


def run_single_ct(adata, model_name, use_majority_voting=False):
    """Run a single CellTypist model, return coarse predictions."""
    try:
        model = ct_models.Model.load(model_name)
    except Exception:
        ct_models.download_models(model=model_name, force_update=False)
        model = ct_models.Model.load(model_name)

    preds_obj = celltypist.annotate(adata, model=model, majority_voting=use_majority_voting)
    result = preds_obj.to_adata()

    if use_majority_voting and "majority_voting" in result.obs.columns:
        fine_preds = result.obs["majority_voting"].astype(str).values
    else:
        fine_preds = result.obs["predicted_labels"].astype(str).values

    coarse = map_predictions_to_coarse(fine_preds)
    conf = result.obs["conf_score"].values if "conf_score" in result.obs.columns else np.ones(len(result))
    return coarse, np.array(conf), fine_preds


def consensus_vote(predictions_list, weights=None):
    """Majority vote across multiple prediction arrays. Returns coarse predictions."""
    n_cells = len(predictions_list[0])
    n_methods = len(predictions_list)
    if weights is None:
        weights = [1.0] * n_methods

    result = []
    confs = []
    for i in range(n_cells):
        votes = Counter()
        for j, preds in enumerate(predictions_list):
            label = preds[i]
            if label and label != "Unknown":
                votes[label] += weights[j]
        if votes:
            winner, count = votes.most_common(1)[0]
            result.append(winner)
            confs.append(count / sum(votes.values()))
        else:
            result.append("Unknown")
            confs.append(0.0)

    return np.array(result), np.array(confs)


def confidence_weighted_vote(predictions_list, confidences_list):
    """Confidence-weighted voting."""
    n_cells = len(predictions_list[0])
    result = []
    confs = []
    for i in range(n_cells):
        scores = Counter()
        for preds, conf in zip(predictions_list, confidences_list):
            label = preds[i]
            if label and label != "Unknown":
                scores[label] += conf[i]
        if scores:
            winner, score = scores.most_common(1)[0]
            total = sum(scores.values())
            result.append(winner)
            confs.append(score / total if total > 0 else 0)
        else:
            result.append("Unknown")
            confs.append(0.0)

    return np.array(result), np.array(confs)


def main():
    logger.info("=" * 80)
    logger.info("MEGA CELLTYPIST + CONSENSUS BENCHMARK")
    logger.info("=" * 80)

    ds_name = os.environ.get("BENCH_DATASETS", "rep1")
    logger.info(f"Dataset: {ds_name}")

    adata_raw = load_xenium_adata(ds_name)
    adata_pp = preprocess_adata(adata_raw)
    gt = np.array(adata_pp.obs["gt_coarse"].values)
    logger.info(f"Loaded: {len(adata_pp)} cells, {adata_pp.n_vars} genes")

    all_results = {}
    model_preds = {}  # Cache predictions for consensus
    model_confs = {}

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Run ALL 47 human CellTypist models individually
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 1: Individual models ({len(ALL_HUMAN_MODELS)} models)")
    logger.info(f"{'='*60}")

    for i, model_name in enumerate(ALL_HUMAN_MODELS):
        short = model_name.replace(".pkl", "")
        logger.info(f"  [{i+1}/{len(ALL_HUMAN_MODELS)}] {short}...", )
        try:
            t0 = time.time()
            coarse, conf, fine = run_single_ct(adata_pp, model_name, use_majority_voting=False)
            f1 = f1_score(gt, coarse, average="macro", zero_division=0, labels=COARSE_CLASSES)
            elapsed = time.time() - t0

            model_preds[short] = coarse
            model_confs[short] = conf

            metrics = compute_metrics(gt, coarse, COARSE_CLASSES)
            metrics["runtime_s"] = round(elapsed, 1)
            all_results[f"ct_{short}"] = metrics
            logger.info(f"    → F1={f1:.3f} ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"    → FAILED: {e}")
            all_results[f"ct_{short}"] = {"error": str(e)}

    # Rank models
    ranked = sorted(
        [(k, v) for k, v in all_results.items() if "f1_macro" in v],
        key=lambda x: -x[1]["f1_macro"]
    )
    logger.info(f"\nTop 15 individual models:")
    for k, v in ranked[:15]:
        logger.info(f"  {k:<50s} F1={v['f1_macro']:.3f}")

    top_models = [k.replace("ct_", "") for k, _ in ranked[:15]]

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: All 2-model and 3-model combinations of top 10
    # ══════════════════════════════════════════════════════════════════════
    top10 = top_models[:10]
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2: Pairwise + Triple consensus (top 10)")
    logger.info(f"  2-combos: {len(list(itertools.combinations(top10, 2)))}")
    logger.info(f"  3-combos: {len(list(itertools.combinations(top10, 3)))}")
    logger.info(f"{'='*60}")

    best_pair = ("", "", 0)
    for m1, m2 in itertools.combinations(top10, 2):
        if m1 not in model_preds or m2 not in model_preds:
            continue
        preds, confs = consensus_vote([model_preds[m1], model_preds[m2]])
        f1 = f1_score(gt, preds, average="macro", zero_division=0, labels=COARSE_CLASSES)
        key = f"cons2_{m1}__{m2}"
        all_results[key] = {"f1_macro": round(float(f1), 4), "type": "consensus_2"}
        if f1 > best_pair[2]:
            best_pair = (m1, m2, f1)

    logger.info(f"Best 2-model: {best_pair[0]} + {best_pair[1]} → F1={best_pair[2]:.4f}")

    best_triple = ("", "", "", 0)
    for m1, m2, m3 in itertools.combinations(top10, 3):
        if not all(m in model_preds for m in [m1, m2, m3]):
            continue
        preds, confs = consensus_vote([model_preds[m1], model_preds[m2], model_preds[m3]])
        f1 = f1_score(gt, preds, average="macro", zero_division=0, labels=COARSE_CLASSES)
        key = f"cons3_{m1}__{m2}__{m3}"
        all_results[key] = {"f1_macro": round(float(f1), 4), "type": "consensus_3"}
        if f1 > best_triple[3]:
            best_triple = (m1, m2, m3, f1)

    logger.info(f"Best 3-model: {best_triple[0]} + {best_triple[1]} + {best_triple[2]} → F1={best_triple[3]:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Semantic ensembles
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 3: Semantic ensembles")
    logger.info(f"{'='*60}")

    semantic_combos = {
        "breast_only": BREAST_RELEVANT,
        "immune_only": IMMUNE_MODELS,
        "pan_tissue": PAN_TISSUE,
        "lung_models": LUNG_MODELS,
        "general_organs": GENERAL_MODELS,
        "breast_immune": BREAST_RELEVANT + IMMUNE_MODELS[:3],
        "breast_immune_pan": BREAST_RELEVANT + IMMUNE_MODELS[:2] + PAN_TISSUE,
        "breast_lung_immune": BREAST_RELEVANT + LUNG_MODELS[:2] + IMMUNE_MODELS[:2],
        "all_immune": IMMUNE_MODELS,
        "top5_plus_immune": [m + ".pkl" for m in top_models[:5]] + IMMUNE_MODELS[:3],
        "top5_plus_breast": [m + ".pkl" for m in top_models[:5]] + BREAST_RELEVANT,
        "top10_unweighted": [m + ".pkl" for m in top_models[:10]],
        "top15_unweighted": [m + ".pkl" for m in top_models[:15]],
        "top20_unweighted": [m + ".pkl" for m in top_models[:20]],
        "all_47_universal": ALL_HUMAN_MODELS,
    }

    for name, model_list in semantic_combos.items():
        shorts = [m.replace(".pkl", "") for m in model_list]
        valid = [s for s in shorts if s in model_preds]
        if len(valid) < 2:
            logger.warning(f"  {name}: only {len(valid)} valid models, skipping")
            continue

        # Unweighted
        preds_list = [model_preds[s] for s in valid]
        preds, confs = consensus_vote(preds_list)
        f1 = f1_score(gt, preds, average="macro", zero_division=0, labels=COARSE_CLASSES)
        key = f"sem_uw_{name}"
        metrics = compute_metrics(gt, preds, COARSE_CLASSES)
        all_results[key] = metrics
        logger.info(f"  {name} (unweighted, {len(valid)} models): F1={f1:.3f}")

        # Confidence-weighted
        confs_list = [model_confs[s] for s in valid]
        preds_cw, confs_cw = confidence_weighted_vote(preds_list, confs_list)
        f1_cw = f1_score(gt, preds_cw, average="macro", zero_division=0, labels=COARSE_CLASSES)
        key_cw = f"sem_cw_{name}"
        metrics_cw = compute_metrics(gt, preds_cw, COARSE_CLASSES)
        all_results[key_cw] = metrics_cw
        logger.info(f"  {name} (conf-weighted, {len(valid)} models): F1={f1_cw:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Hybrid ensembles (CellTypist + SCINA + SingleR)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 4: Hybrid CellTypist + SCINA + SingleR")
    logger.info(f"{'='*60}")

    # Run SCINA to get its predictions
    try:
        from scripts.annotation_benchmark_round2 import run_scina, run_singler
        scina_result = run_scina(adata_pp)
        scina_coarse = map_predictions_to_coarse(scina_result["predictions"])
        scina_conf = scina_result["confidence"]
        model_preds["SCINA"] = scina_coarse
        model_confs["SCINA"] = scina_conf
        logger.info(f"  SCINA loaded: F1={f1_score(gt, scina_coarse, average='macro', zero_division=0, labels=COARSE_CLASSES):.3f}")
    except Exception as e:
        logger.error(f"  SCINA failed: {e}")

    # Run SingleR references
    for ref in ["blueprint", "hpca"]:
        try:
            from scripts.annotation_benchmark_2026_03 import run_singler
            sr = run_singler(adata_pp, ref)
            if sr.get("predictions") is not None:
                sr_coarse = map_predictions_to_coarse(sr["predictions"])
                model_preds[f"SingleR_{ref}"] = sr_coarse
                model_confs[f"SingleR_{ref}"] = sr["confidence"]
                logger.info(f"  SingleR {ref}: F1={f1_score(gt, sr_coarse, average='macro', zero_division=0, labels=COARSE_CLASSES):.3f}")
        except Exception as e:
            logger.error(f"  SingleR {ref}: {e}")

    # Build hybrid combos
    hybrid_combos = {}
    for n_top in [3, 5, 8]:
        top_n = top_models[:n_top]
        for extra in [["SCINA"], ["SingleR_blueprint"], ["SingleR_hpca"],
                      ["SCINA", "SingleR_blueprint"], ["SCINA", "SingleR_hpca"],
                      ["SCINA", "SingleR_blueprint", "SingleR_hpca"]]:
            valid_extra = [e for e in extra if e in model_preds]
            if not valid_extra:
                continue
            name = f"top{n_top}ct_{'_'.join(valid_extra)}"
            members = top_n + valid_extra
            hybrid_combos[name] = members

    for name, members in hybrid_combos.items():
        valid = [m for m in members if m in model_preds]
        if len(valid) < 2:
            continue
        preds_list = [model_preds[m] for m in valid]
        confs_list = [model_confs[m] for m in valid]

        # Unweighted
        preds, _ = consensus_vote(preds_list)
        f1 = f1_score(gt, preds, average="macro", zero_division=0, labels=COARSE_CLASSES)
        metrics = compute_metrics(gt, preds, COARSE_CLASSES)
        all_results[f"hybrid_uw_{name}"] = metrics

        # Confidence-weighted
        preds_cw, _ = confidence_weighted_vote(preds_list, confs_list)
        f1_cw = f1_score(gt, preds_cw, average="macro", zero_division=0, labels=COARSE_CLASSES)
        metrics_cw = compute_metrics(gt, preds_cw, COARSE_CLASSES)
        all_results[f"hybrid_cw_{name}"] = metrics_cw

        logger.info(f"  {name} ({len(valid)}): UW={f1:.3f} CW={f1_cw:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: Supermajority voting (require 2/3 or 3/4 agreement)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 5: Supermajority voting thresholds")
    logger.info(f"{'='*60}")

    for n_top in [5, 10, 15, 20]:
        top_n = top_models[:n_top]
        valid = [m for m in top_n if m in model_preds]
        if len(valid) < 3:
            continue
        preds_list = [model_preds[m] for m in valid]

        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            n_cells = len(preds_list[0])
            result_preds = []
            result_confs = []
            for i in range(n_cells):
                votes = Counter()
                for p in preds_list:
                    if p[i] != "Unknown":
                        votes[p[i]] += 1
                if votes:
                    winner, count = votes.most_common(1)[0]
                    agreement = count / len(valid)
                    if agreement >= threshold:
                        result_preds.append(winner)
                        result_confs.append(agreement)
                    else:
                        result_preds.append("Unknown")
                        result_confs.append(0.0)
                else:
                    result_preds.append("Unknown")
                    result_confs.append(0.0)

            result_preds = np.array(result_preds)
            n_classified = np.sum(result_preds != "Unknown")
            pct = n_classified / n_cells * 100

            # Compute metrics on classified cells only
            mask = result_preds != "Unknown"
            if mask.sum() > 0:
                f1 = f1_score(gt[mask], result_preds[mask], average="macro", zero_division=0, labels=COARSE_CLASSES)
                acc = np.mean(gt[mask] == result_preds[mask])
            else:
                f1, acc = 0, 0

            key = f"supermaj_top{n_top}_t{int(threshold*100)}"
            all_results[key] = {
                "f1_macro": round(float(f1), 4),
                "accuracy": round(float(acc), 4),
                "pct_classified": round(pct, 1),
                "n_classified": int(n_classified),
                "threshold": threshold,
            }
            logger.info(f"  top{n_top} t={threshold}: F1={f1:.3f} Acc={acc:.3f} ({pct:.1f}% classified)")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'='*80}")
    logger.info("MEGA BENCHMARK FINAL RANKINGS")
    logger.info(f"{'='*80}")
    logger.info(f"Total configurations tested: {len(all_results)}")

    # Top 30 overall
    ranked_all = sorted(
        [(k, v) for k, v in all_results.items() if "f1_macro" in v],
        key=lambda x: -x[1]["f1_macro"]
    )

    logger.info(f"\n{'Method':<65s} {'F1':>6s} {'Acc':>6s}")
    logger.info("-" * 80)
    for k, v in ranked_all[:30]:
        pct = f" ({v['pct_classified']:.0f}%)" if "pct_classified" in v else ""
        logger.info(f"{k:<65s} {v['f1_macro']:>6.3f} {v.get('accuracy', 0):>6.3f}{pct}")

    # Save
    out_file = OUTPUT_DIR / f"celltypist_mega_{ds_name}.json"
    # Clean for JSON (remove numpy types)
    clean = {}
    for k, v in all_results.items():
        clean[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
    with open(out_file, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"\nSaved {len(all_results)} results to {out_file}")

    # Merge key results into main results file
    r1_path = OUTPUT_DIR / f"results_{ds_name}.json"
    if r1_path.exists():
        existing = json.load(open(r1_path))
        # Add top 10 new methods
        for k, v in ranked_all[:10]:
            existing[k] = v
        with open(r1_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Merged top 10 into {r1_path}")

    logger.info("\nMEGA BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
