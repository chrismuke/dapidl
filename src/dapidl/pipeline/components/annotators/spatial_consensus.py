"""BANKSY + Expression Consensus Annotation Pipeline.

Combines spatial clustering (BANKSY) with expression-based methods (CellTypist, SingleR, SCINA)
using a 5-voter hybrid consensus with spatial refinement.

Architecture:
  Phase 1: Cell-level expression annotation (4 voters)
    - CellTypist tissue-specific model (1 vote)
    - CellTypist Immune_All_High (1 vote)
    - SingleR HPCA (1 vote)
    - SingleR Blueprint (1 vote)

  Phase 2: BANKSY spatial clustering (1 voter)
    - k=15, lambda=0.2, resolution tuned per tissue
    - Cluster labeling via marker enrichment

  Phase 3: Consensus voting (5 voters)
    - Unweighted majority vote (popV-style)
    - Confidence = n_agreeing / 5

  Phase 4: Spatial refinement
    - Override isolated disagreements using BANKSY cluster majority
    - KNN spatial smoothing (k=15)

Reference: Inspired by popV (Ergen 2024) + BANKSY (Singhal 2024)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse
from scipy.spatial import cKDTree

from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

# Tissue-specific CellTypist model selection
TISSUE_MODELS = {
    "breast": "Cells_Adult_Breast.pkl",
    "skin": "Adult_Human_Skin.pkl",
    "lung": "Human_Lung_Atlas.pkl",
    "liver": "Healthy_Human_Liver.pkl",
    "colon": "Cells_Intestinal_Tract.pkl",
    "heart": "Healthy_Adult_Heart.pkl",
    "brain": "Developing_Human_Brain.pkl",
    "kidney": "Adult_Human_PancreaticIslet.pkl",
    "tonsil": "Cells_Human_Tonsil.pkl",
    "pancreas": "Adult_Human_PancreaticIslet.pkl",
}

# Tissue-specific marker databases for BANKSY cluster labeling
COARSE_MARKERS = {
    "Epithelial": {"positive": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "MUC1"],
                   "negative": ["PTPRC", "VIM", "PECAM1"]},
    "Immune": {"positive": ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD14", "CD68", "MS4A1", "NKG7", "TRAC"],
               "negative": ["EPCAM", "KRT8", "COL1A1"]},
    "Stromal": {"positive": ["COL1A1", "COL1A2", "ACTA2", "VIM", "FAP", "DCN", "PDGFRA", "PDGFRB"],
                "negative": ["EPCAM", "PTPRC", "PECAM1"]},
    "Endothelial": {"positive": ["PECAM1", "VWF", "CLDN5", "KDR"],
                    "negative": ["EPCAM", "PTPRC", "COL1A1"]},
}

SKIN_MARKERS = {
    **COARSE_MARKERS,
    "Keratinocyte": {"positive": ["KRT14", "KRT1", "KRT10", "KRT5", "DSC1", "DSG1"],
                     "negative": ["PTPRC", "COL1A1"]},
    "Melanocyte": {"positive": ["MLANA", "PMEL", "TYR", "DCT"],
                   "negative": ["PTPRC", "KRT14"]},
}


@dataclass
class SpatialConsensusConfig:
    """Configuration for BANKSY + expression consensus."""
    tissue: str = "breast"
    # BANKSY params
    banksy_k: int = 15
    banksy_lambda: float = 0.2
    banksy_resolution: float = 0.5
    # Consensus params
    min_agreement: float = 0.6  # 3/5 for medium confidence
    high_agreement: float = 0.8  # 4/5 for high confidence
    # Spatial refinement
    spatial_override_threshold: float = 0.85  # override if >85% of cluster agrees
    knn_smooth_k: int = 15
    # Method selection
    use_singler: bool = True
    use_banksy: bool = True


def run_celltypist_voter(adata, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run CellTypist and return coarse predictions + confidence."""
    import celltypist
    from celltypist import models as ct_models

    try:
        model = ct_models.Model.load(model_name)
    except Exception:
        ct_models.download_models(model=model_name, force_update=False)
        model = ct_models.Model.load(model_name)

    result = celltypist.annotate(adata, model=model, majority_voting=False).to_adata()
    fine_preds = result.obs["predicted_labels"].astype(str).values
    coarse = np.array([map_to_broad_category(p) for p in fine_preds])
    conf = result.obs["conf_score"].values if "conf_score" in result.obs.columns else np.ones(len(result))
    return coarse, conf, fine_preds


def run_singler_voter(adata, reference: str = "hpca") -> tuple[np.ndarray, np.ndarray]:
    """Run SingleR and return coarse predictions + confidence."""
    from dapidl.pipeline.components.annotators.singler import (
        SINGLER_REFERENCES,
        _fix_libstdcxx,
        is_singler_available,
    )

    if not is_singler_available():
        return None, None

    _fix_libstdcxx()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    r = ro.r
    singler_r = importr("SingleR")
    importr("celldex")

    ref_name = SINGLER_REFERENCES[reference]
    ref_data = r(f"celldex::{ref_name}()")
    ref_labels = r("function(x) colData(x)$label.main")(ref_data)

    expr = adata.X.toarray().T if issparse(adata.X) else np.asarray(adata.X).T
    expr_r = r["matrix"](
        ro.FloatVector(expr.flatten().tolist()),
        nrow=expr.shape[0], ncol=expr.shape[1],
    )
    expr_r.rownames = ro.StrVector(list(adata.var_names))
    expr_r.colnames = ro.StrVector(list(adata.obs_names))

    results = singler_r.SingleR(test=expr_r, ref=ref_data, labels=ref_labels)
    preds = np.array(list(r("function(x) as.character(x$labels)")(results)))
    scores = np.array(r("function(x) as.matrix(x$scores)")(results)).reshape(len(preds), -1)
    conf = (scores.max(axis=1) + 1) / 2

    coarse = np.array([map_to_broad_category(p) for p in preds])
    return coarse, conf


def run_banksy_voter(adata, config: SpatialConsensusConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BANKSY spatial clustering and label clusters."""
    from banksy.cluster_methods import run_Leiden_partition
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.initialize_banksy import initialize_banksy
    from banksy_utils.umap_pca import pca_umap

    a = adata.copy()
    if issparse(a.X):
        a.X = a.X.toarray()

    # HVG
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars), subset=False)
    a_hvg = a[:, a.var["highly_variable"]].copy()

    # Setup coordinates
    if "x_centroid" not in a.obs.columns:
        logger.warning("No spatial coordinates — skipping BANKSY")
        return None, None, None

    a_hvg.obs["xcoord"] = a.obs["x_centroid"].values
    a_hvg.obs["ycoord"] = a.obs["y_centroid"].values
    a_hvg.obsm["xy_coord"] = np.column_stack([
        a_hvg.obs["xcoord"].values.astype(float),
        a_hvg.obs["ycoord"].values.astype(float),
    ])

    # BANKSY
    banksy_dict = initialize_banksy(
        adata=a_hvg, coord_keys=("xcoord", "ycoord", "xy_coord"),
        num_neighbours=config.banksy_k, nbr_weight_decay="scaled_gaussian", max_m=1,
        plt_edge_hist=False, plt_nbr_weights=False, plt_theta=False,
    )
    banksy_dict, _ = generate_banksy_matrix(
        adata=a_hvg, banksy_dict=banksy_dict, lambda_list=[config.banksy_lambda],
        max_m=1, plot_std=False, verbose=False,
    )
    pca_umap(banksy_dict=banksy_dict, pca_dims=[20], plt_remaining_var=False, add_umap=False)

    results_df, _ = run_Leiden_partition(
        banksy_dict=banksy_dict, resolutions=[config.banksy_resolution],
        num_nn=50, num_iterations=-1, partition_seed=1234,
        match_labels=False, verbose=False,
    )

    cluster_ids = results_df.iloc[0]["labels"].dense

    # Label clusters with markers
    markers = SKIN_MARKERS if config.tissue == "skin" else COARSE_MARKERS
    gene_names = list(a.var_names)
    expr = np.asarray(a.X)

    cluster_labels = {}
    for cl in set(cluster_ids.astype(str)):
        mask = cluster_ids.astype(str) == cl
        cl_expr = expr[mask]
        best, best_sc = "Unknown", -999
        for ct, m in markers.items():
            pos = [g for g in m["positive"] if g in gene_names]
            neg = [g for g in m.get("negative", []) if g in gene_names]
            if not pos:
                continue
            sc_ = cl_expr[:, [gene_names.index(g) for g in pos]].mean()
            if neg:
                sc_ -= 0.5 * cl_expr[:, [gene_names.index(g) for g in neg]].mean()
            if sc_ > best_sc:
                best_sc, best = sc_, ct
        cluster_labels[cl] = best

    coarse = np.array([cluster_labels.get(str(c), "Unknown") for c in cluster_ids])
    conf = np.ones(len(cluster_ids)) * 0.8  # BANKSY clusters are spatially coherent
    return coarse, conf, cluster_ids


def spatial_consensus(
    adata: Any,
    config: SpatialConsensusConfig | None = None,
) -> pl.DataFrame:
    """Run the full BANKSY + expression consensus pipeline.

    Returns a polars DataFrame with columns:
      cell_id, consensus_label, consensus_score, confidence_tier,
      n_voters, per_voter predictions
    """
    if config is None:
        config = SpatialConsensusConfig()

    n_cells = len(adata)
    logger.info(f"Spatial Consensus: {n_cells} cells, tissue={config.tissue}")

    # Normalize if not already
    a = adata.copy()
    if issparse(a.X):
        a.X = a.X.toarray()
    a.layers["raw"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    voters = {}
    voter_conf = {}
    fine_preds = {}

    # ── Voter 1: CellTypist tissue-specific ──────────────────────────
    tissue_model = TISSUE_MODELS.get(config.tissue, "Cells_Adult_Breast.pkl")
    logger.info(f"  Voter 1: CellTypist {tissue_model}")
    try:
        coarse, conf, fine = run_celltypist_voter(a, tissue_model)
        voters["ct_tissue"] = coarse
        voter_conf["ct_tissue"] = conf
        fine_preds["ct_tissue"] = fine
    except Exception as e:
        logger.error(f"  CellTypist tissue failed: {e}")

    # ── Voter 2: CellTypist Immune_All_High ──────────────────────────
    logger.info("  Voter 2: CellTypist Immune_All_High")
    try:
        coarse, conf, fine = run_celltypist_voter(a, "Immune_All_High.pkl")
        voters["ct_immune"] = coarse
        voter_conf["ct_immune"] = conf
        fine_preds["ct_immune"] = fine
    except Exception as e:
        logger.error(f"  CellTypist immune failed: {e}")

    # ── Voter 3+4: SingleR ───────────────────────────────────────────
    if config.use_singler:
        for ref in ["hpca", "blueprint"]:
            logger.info(f"  Voter: SingleR {ref}")
            try:
                coarse, conf = run_singler_voter(a, ref)
                if coarse is not None:
                    voters[f"sr_{ref}"] = coarse
                    voter_conf[f"sr_{ref}"] = conf
            except Exception as e:
                logger.error(f"  SingleR {ref} failed: {e}")

    # ── Voter 5: BANKSY ──────────────────────────────────────────────
    cluster_ids = None
    if config.use_banksy:
        logger.info(f"  Voter: BANKSY k={config.banksy_k} λ={config.banksy_lambda}")
        try:
            coarse, conf, cluster_ids = run_banksy_voter(a, config)
            if coarse is not None:
                voters["banksy"] = coarse
                voter_conf["banksy"] = conf
        except Exception as e:
            logger.error(f"  BANKSY failed: {e}")

    n_voters = len(voters)
    logger.info(f"  Active voters: {n_voters} ({list(voters.keys())})")

    if n_voters == 0:
        raise RuntimeError("No voters succeeded")

    # ── Phase 3: Consensus voting ────────────────────────────────────
    logger.info("  Computing consensus...")
    consensus_labels = []
    consensus_scores = []
    for i in range(n_cells):
        votes = Counter()
        for _name, preds in voters.items():
            label = preds[i]
            if label and label != "Unknown":
                votes[label] += 1
        if votes:
            winner, count = votes.most_common(1)[0]
            consensus_labels.append(winner)
            consensus_scores.append(count / n_voters)
        else:
            consensus_labels.append("Unknown")
            consensus_scores.append(0.0)

    consensus_labels = np.array(consensus_labels)
    consensus_scores = np.array(consensus_scores)

    # ── Phase 4: Spatial refinement ──────────────────────────────────
    if cluster_ids is not None and config.spatial_override_threshold > 0:
        logger.info("  Spatial refinement (cluster override)...")
        n_overridden = 0
        for cl in set(cluster_ids.astype(str)):
            mask = cluster_ids.astype(str) == cl
            cluster_consensus = consensus_labels[mask]
            counts = Counter(l for l in cluster_consensus if l != "Unknown")
            if not counts:
                continue
            majority, majority_count = counts.most_common(1)[0]
            agreement = majority_count / mask.sum()
            if agreement >= config.spatial_override_threshold:
                disagree = mask & (consensus_labels != majority)
                n_overridden += disagree.sum()
                consensus_labels[disagree] = majority

        logger.info(f"    Overridden {n_overridden} cells ({n_overridden/n_cells*100:.1f}%)")

    # KNN smoothing
    if config.knn_smooth_k > 0 and "x_centroid" in adata.obs.columns:
        logger.info(f"  KNN smoothing (k={config.knn_smooth_k})...")
        coords = np.column_stack([
            adata.obs["x_centroid"].values.astype(float),
            adata.obs["y_centroid"].values.astype(float),
        ])
        tree = cKDTree(coords)
        _, indices = tree.query(coords, k=config.knn_smooth_k)
        smoothed = []
        for i in range(n_cells):
            nbr_labels = consensus_labels[indices[i]]
            counts = Counter(l for l in nbr_labels if l != "Unknown")
            smoothed.append(counts.most_common(1)[0][0] if counts else consensus_labels[i])
        consensus_labels = np.array(smoothed)

    # ── Build result DataFrame ───────────────────────────────────────
    confidence_tier = np.where(
        consensus_scores >= config.high_agreement, "HIGH",
        np.where(consensus_scores >= config.min_agreement, "MEDIUM", "LOW")
    )

    result = {
        "cell_id": list(adata.obs_names),
        "consensus_label": consensus_labels.tolist(),
        "consensus_score": consensus_scores.tolist(),
        "confidence_tier": confidence_tier.tolist(),
        "n_voters": [n_voters] * n_cells,
    }
    for name, preds in voters.items():
        result[f"voter_{name}"] = preds.tolist()
    for name, fine in fine_preds.items():
        result[f"fine_{name}"] = fine.tolist()

    df = pl.DataFrame(result)

    # Summary
    tier_counts = Counter(confidence_tier)
    logger.info("\n  Consensus summary:")
    logger.info(f"    HIGH confidence: {tier_counts.get('HIGH', 0)} ({tier_counts.get('HIGH', 0)/n_cells*100:.1f}%)")
    logger.info(f"    MEDIUM: {tier_counts.get('MEDIUM', 0)} ({tier_counts.get('MEDIUM', 0)/n_cells*100:.1f}%)")
    logger.info(f"    LOW: {tier_counts.get('LOW', 0)} ({tier_counts.get('LOW', 0)/n_cells*100:.1f}%)")
    label_counts = Counter(consensus_labels)
    for label, count in label_counts.most_common():
        logger.info(f"    {label}: {count} ({count/n_cells*100:.1f}%)")

    return df
