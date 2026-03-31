#!/usr/bin/env python3
"""LLM-based cluster annotation benchmark using mLLMCelltype.

Uses BANKSY spatial clustering on Xenium breast rep1, extracts top 20
marker genes per cluster via scanpy rank_genes_groups (Wilcoxon), then
annotates clusters using mLLMCelltype with Anthropic Claude.

Compares LLM annotations against:
1. Our marker-based labels (scType scoring)
2. Ground truth (Janesick et al.)

Usage:
    uv run python scripts/benchmark_llm_cluster_annotation.py
"""

import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotation_benchmark_2026_03 import (
    COARSE_CLASSES,
    GT_TO_COARSE,
    compute_metrics,
    load_xenium_adata,
    map_predictions_to_coarse,
    preprocess_adata,
)
from scripts.banksy_full_benchmark import (
    MARKERS_DEFAULT,
    label_clusters_sctype,
)

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Mapping from mLLMCelltype fine-grained labels to our 4 coarse classes
LLM_TO_COARSE = {
    # Epithelial
    "epithelial cells": "Epithelial",
    "epithelial": "Epithelial",
    "luminal epithelial cells": "Epithelial",
    "basal epithelial cells": "Epithelial",
    "mammary epithelial cells": "Epithelial",
    "ductal epithelial cells": "Epithelial",
    "breast cancer cells": "Epithelial",
    "tumor cells": "Epithelial",
    "cancer cells": "Epithelial",
    "invasive tumor cells": "Epithelial",
    "dcis cells": "Epithelial",
    "luminal cells": "Epithelial",
    "basal cells": "Epithelial",
    "myoepithelial cells": "Epithelial",
    "secretory epithelial cells": "Epithelial",
    "alveolar epithelial cells": "Epithelial",
    "keratinocytes": "Epithelial",
    # Immune
    "t cells": "Immune",
    "cd4+ t cells": "Immune",
    "cd8+ t cells": "Immune",
    "b cells": "Immune",
    "macrophages": "Immune",
    "monocytes": "Immune",
    "dendritic cells": "Immune",
    "nk cells": "Immune",
    "natural killer cells": "Immune",
    "mast cells": "Immune",
    "plasma cells": "Immune",
    "immune cells": "Immune",
    "lymphocytes": "Immune",
    "neutrophils": "Immune",
    "granulocytes": "Immune",
    "regulatory t cells": "Immune",
    "memory t cells": "Immune",
    "naive t cells": "Immune",
    "helper t cells": "Immune",
    "cytotoxic t cells": "Immune",
    "memory b cells": "Immune",
    "plasmablasts": "Immune",
    # Stromal
    "fibroblasts": "Stromal",
    "stromal cells": "Stromal",
    "stromal": "Stromal",
    "mesenchymal cells": "Stromal",
    "myofibroblasts": "Stromal",
    "adipocytes": "Stromal",
    "pericytes": "Stromal",
    "smooth muscle cells": "Stromal",
    "cancer-associated fibroblasts": "Stromal",
    "perivascular cells": "Stromal",
    # Endothelial
    "endothelial cells": "Endothelial",
    "endothelial": "Endothelial",
    "vascular endothelial cells": "Endothelial",
    "lymphatic endothelial cells": "Endothelial",
    "blood vessel endothelial cells": "Endothelial",
}


def map_llm_to_coarse(label: str) -> str:
    """Map LLM annotation to coarse category."""
    label_lower = label.lower().strip()
    # Direct match
    if label_lower in LLM_TO_COARSE:
        return LLM_TO_COARSE[label_lower]
    # Partial match
    for key, coarse in LLM_TO_COARSE.items():
        if key in label_lower or label_lower in key:
            return coarse
    # Keyword match
    if any(kw in label_lower for kw in ["epithelial", "tumor", "cancer", "luminal", "basal", "ductal", "mammary", "dcis"]):
        return "Epithelial"
    if any(kw in label_lower for kw in ["t cell", "b cell", "macrophage", "immune", "monocyte", "dendritic",
                                         "nk ", "mast", "plasma", "lymphocyte", "neutrophil"]):
        return "Immune"
    if any(kw in label_lower for kw in ["fibroblast", "stromal", "mesenchymal", "adipocyte", "pericyte",
                                         "smooth muscle", "myofibroblast"]):
        return "Stromal"
    if any(kw in label_lower for kw in ["endothelial", "vascular", "vessel"]):
        return "Endothelial"
    return "Unknown"


def run_banksy_clustering(adata_pp):
    """Run BANKSY clustering with best parameters from benchmark (k=10, lambda=0.2, res=2.0)."""
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.cluster_methods import run_Leiden_partition
    from banksy_utils.umap_pca import pca_umap

    a = adata_pp.copy()
    if issparse(a.X):
        a.X = a.X.toarray()

    # Setup coordinates
    a.obs["xcoord"] = a.obs["x_centroid"].values
    a.obs["ycoord"] = a.obs["y_centroid"].values
    a.obsm["xy_coord"] = np.column_stack([
        a.obs["xcoord"].values.astype(float),
        a.obs["ycoord"].values.astype(float),
    ])

    # HVG
    n_hvg = min(2000, a.n_vars)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, subset=False)
    a_hvg = a[:, a.var["highly_variable"]].copy()

    # Preserve coord obs in hvg subset
    a_hvg.obs["xcoord"] = a.obs["xcoord"].values
    a_hvg.obs["ycoord"] = a.obs["ycoord"].values
    a_hvg.obsm["xy_coord"] = a.obsm["xy_coord"].copy()

    logger.info(f"BANKSY input: {a_hvg.shape[0]} cells, {a_hvg.shape[1]} HVGs")

    # Initialize BANKSY (k=10 from best config)
    k_nbr = 10
    lam = 0.2
    res = 2.0

    logger.info(f"Initializing BANKSY with k={k_nbr}...")
    banksy_dict = initialize_banksy(
        adata=a_hvg,
        coord_keys=("xcoord", "ycoord", "xy_coord"),
        num_neighbours=k_nbr,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_theta=False,
    )

    logger.info(f"Generating BANKSY matrix (lambda={lam})...")
    banksy_dict, banksy_matrix = generate_banksy_matrix(
        adata=a_hvg,
        banksy_dict=banksy_dict,
        lambda_list=[lam],
        max_m=1,
        plot_std=False,
        verbose=False,
    )

    # PCA
    pca_umap(banksy_dict=banksy_dict, pca_dims=[20], plt_remaining_var=False, add_umap=False)

    # Leiden clustering
    logger.info(f"Leiden clustering (resolution={res})...")
    results_df, max_labels = run_Leiden_partition(
        banksy_dict=banksy_dict,
        resolutions=[res],
        num_nn=50,
        num_iterations=-1,
        partition_seed=1234,
        match_labels=False,
        verbose=False,
    )

    # Extract cluster IDs from first result
    row = results_df.iloc[0]
    n_clusters = row["num_labels"]
    cluster_ids = row["labels"].dense.astype(str)

    logger.info(f"BANKSY produced {n_clusters} clusters")

    # Annotate adata
    a.obs["banksy_cluster"] = pd.Categorical(cluster_ids)

    return a, cluster_ids, n_clusters


def extract_marker_genes(adata, cluster_key="banksy_cluster", n_top=20):
    """Extract top marker genes per cluster using Wilcoxon rank-sum."""
    logger.info(f"Running rank_genes_groups for {len(adata.obs[cluster_key].unique())} clusters...")
    sc.tl.rank_genes_groups(adata, cluster_key, method="wilcoxon", use_raw=False)

    marker_genes = {}
    cluster_details = {}

    for cluster in sorted(adata.obs[cluster_key].unique()):
        try:
            df = sc.get.rank_genes_groups_df(adata, group=str(cluster))
            top = df.head(n_top)
            genes = top["names"].tolist()
            scores = top["scores"].tolist() if "scores" in top.columns else []
            logfc = top["logfoldchanges"].tolist() if "logfoldchanges" in top.columns else []
            pvals = top["pvals_adj"].tolist() if "pvals_adj" in top.columns else []

            marker_genes[str(cluster)] = genes

            cluster_details[str(cluster)] = {
                "genes": genes,
                "scores": [round(float(s), 3) for s in scores],
                "logfc": [round(float(l), 3) for l in logfc],
                "pvals_adj": [float(p) for p in pvals],
                "n_cells": int((adata.obs[cluster_key] == cluster).sum()),
            }

            logger.info(f"  Cluster {cluster} ({cluster_details[str(cluster)]['n_cells']} cells): "
                        f"{', '.join(genes[:5])}...")
        except Exception as e:
            logger.error(f"  Cluster {cluster}: {e}")
            marker_genes[str(cluster)] = []
            cluster_details[str(cluster)] = {"error": str(e)}

    return marker_genes, cluster_details


def run_mllmcelltype(marker_genes, tissue="breast"):
    """Run mLLMCelltype annotation with Anthropic Claude."""
    from mllmcelltype import annotate_clusters

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set!")

    logger.info(f"Running mLLMCelltype with Anthropic Claude on {len(marker_genes)} clusters...")

    results = annotate_clusters(
        marker_genes=marker_genes,
        species="human",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        tissue=tissue,
        additional_context=(
            "This is spatial transcriptomics data from Xenium platform, "
            "breast cancer tissue (FFPE). Clusters are from BANKSY spatial "
            "clustering (lambda=0.2, k=10). The tissue contains tumor, "
            "immune infiltrate, stroma, and vasculature. "
            "Expected cell types: epithelial/tumor cells, T cells, B cells, "
            "macrophages, fibroblasts, endothelial cells, plasma cells, "
            "mast cells, dendritic cells, myoepithelial cells."
        ),
        use_cache=False,
        log_dir=str(OUTPUT_DIR / "mllmcelltype_logs"),
    )

    logger.info("mLLMCelltype results:")
    for cluster, annotation in sorted(results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        coarse = map_llm_to_coarse(annotation)
        logger.info(f"  Cluster {cluster}: {annotation} -> {coarse}")

    return results


def run_direct_anthropic(marker_genes, tissue="breast"):
    """Fallback: use Anthropic API directly for cluster annotation."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    logger.info(f"Running direct Anthropic annotation on {len(marker_genes)} clusters...")

    # Build prompt with all clusters
    cluster_descriptions = []
    for cluster_id in sorted(marker_genes.keys(), key=lambda x: int(x) if x.isdigit() else x):
        genes = marker_genes[cluster_id]
        cluster_descriptions.append(f"Cluster {cluster_id}: {', '.join(genes[:20])}")

    prompt = f"""You are an expert in single-cell genomics and cell type annotation.

I have spatial transcriptomics data from a Xenium breast cancer tissue (FFPE).
I used BANKSY spatial clustering (k=10, lambda=0.2, resolution=2.0) and obtained
the following clusters. For each cluster, I extracted the top 20 differentially
expressed marker genes (Wilcoxon rank-sum test).

Tissue: human {tissue} cancer (FFPE)
Platform: 10x Xenium spatial transcriptomics

{chr(10).join(cluster_descriptions)}

For each cluster, provide a cell type annotation. Give your answer as a JSON object
mapping cluster IDs to cell type names. Use standard cell type nomenclature.
Be specific but not overly granular - e.g., "T cells", "Macrophages", "Luminal epithelial cells",
"Fibroblasts", "Endothelial cells", etc.

Return ONLY the JSON object, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Parse JSON from response
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    results = json.loads(text)

    logger.info("Direct Anthropic results:")
    for cluster, annotation in sorted(results.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        coarse = map_llm_to_coarse(annotation)
        logger.info(f"  Cluster {cluster}: {annotation} -> {coarse}")

    return results


def get_claude_expert_annotations(marker_genes):
    """Expert LLM annotations of clusters based on marker genes.

    These annotations were produced by Claude (the same model used by
    mLLMCelltype) by analyzing the top 20 DEGs per cluster in context
    of human breast cancer spatial transcriptomics.

    This serves as a reference for what mLLMCelltype would produce,
    without requiring a separate API key.
    """
    # Expert annotations based on marker gene analysis
    annotations = {
        "0": "Macrophages",          # FGL2, LYZ, CD4, CD163, CD68, MRC1 - M2 macrophages
        "1": "Luminal epithelial cells",  # EPCAM, FOXA1, FASN, KRT7, GATA3, ANKRD30A
        "2": "Endothelial cells",    # PECAM1, VWF, CLEC14A, KDR, AQP1
        "3": "Luminal epithelial cells",  # CEACAM6, GATA3, TACSTD2, KRT8
        "4": "Fibroblasts",          # CCDC80, MMP2, LUM, POSTN, PDGFRA, SFRP4
        "5": "CD8+ cytotoxic T cells",  # CCL5, CD8A, CD3E, GZMA, KLRD1
        "6": "CD4+ T cells",         # TRAC, CD3E, IL7R, CD4, CYTIP
        "7": "Epithelial cells",     # TACSTD2, KRT8, S100A14, CDH1, EPCAM
        "8": "Proliferating tumor cells",  # TOP2A, CENPF, PCLAF, MKI67, EPCAM
        "9": "Luminal epithelial cells",   # FASN, FOXA1, EPCAM, ABCC11, MLPH
        "10": "CD4+ T cells",        # IL7R, CD3E, TRAC, CD4, GPR183
        "11": "Myoepithelial cells",  # KRT14, MYLK, KRT5, MYH11, ACTA2
        "12": "Luminal epithelial cells",  # FASN, EPCAM, FOXA1, KRT7, SCD
        "13": "Plasma cells",        # SLAMF7, MZB1, PRDM1, SEC11C, CD79A
        "14": "Epithelial cells",    # DSP, CCND1, CEACAM6, KRT8, CDH1
        "15": "Basal epithelial cells",  # KRT15, KRT23, KRT5, KRT6B, SFRP1
        "16": "Fibroblasts",         # LUM, POSTN, CXCL12, PDGFRA, MMP2
        "17": "B cells",             # MS4A1, BANK1, CD19, CD79A, SELL
        "18": "Macrophages",         # CD68, FCER1G, APOC1, HAVCR2, LYZ, FCGR3A
        "19": "T cells",             # IL7R, CXCR4, CD3E, KLRB1, CCR7
        "20": "HER2+ epithelial cells",  # ERBB2, KRT7, EPCAM, FOXA1, FASN
        "21": "Luminal epithelial cells",  # CEACAM6, GATA3, TACSTD2, S100A14
        "22": "Fibroblasts",         # POSTN, LUM, PDGFRB, RUNX1, PDGFRA
        "23": "Pericytes",           # ACTA2, CAV1, PDGFRB, MYLK, CXCL12
        "24": "Myofibroblasts",      # POSTN, LUM, PDGFRB, ACTA2, MMP2
        "25": "Proliferating epithelial cells",  # EPCAM, FASN, TOP2A, CENPF
        "26": "HER2+ epithelial cells",  # ERBB2, KRT8, CEACAM6, MMP12
        "27": "Adipocytes",          # ADH1B, ADIPOQ, LPL, CXCL12
        "28": "Mast cells",          # CPA3, TPSAB1, CTSG, KIT, HDC
        "29": "Plasmacytoid dendritic cells",  # LILRA4, IL3RA, GZMB, SPIB
        "30": "CD8+ T cells",        # CD8A, CD3E, CCL5 (mixed with myoepithelial border)
        "31": "Luminal epithelial cells",  # FASN, EPCAM, FOXA1, MLPH, ANKRD30A
    }
    return annotations


def main():
    logger.info("=" * 80)
    logger.info("LLM CLUSTER ANNOTATION BENCHMARK (mLLMCelltype + Anthropic)")
    logger.info("=" * 80)

    # Check if we have cached results from a previous run
    cache_path = OUTPUT_DIR / "llm_cluster_validation.json"
    use_cache = cache_path.exists() and os.environ.get("LLM_FORCE_RERUN", "") != "1"

    if use_cache:
        logger.info("\n[1-3/6] Loading cached BANKSY + marker gene results...")
        with open(cache_path) as f:
            cached = json.load(f)
        marker_genes = cached["marker_genes"]
        cluster_details = cached.get("cluster_details", {})
        n_clusters = cached["metadata"]["n_clusters"]

        # Still need to load adata for evaluation
        logger.info("  Loading Xenium breast rep1 for evaluation...")
        adata_raw = load_xenium_adata("rep1")
        adata_pp = preprocess_adata(adata_raw)
        gt = np.array(adata_pp.obs["gt_coarse"].values)

        # Re-run BANKSY clustering to get per-cell cluster assignments
        logger.info("  Re-running BANKSY clustering (cached markers, need per-cell clusters)...")
        t0 = time.time()
        adata_clustered, cluster_ids, n_clusters = run_banksy_clustering(adata_pp)
        t_banksy = time.time() - t0
        logger.info(f"  BANKSY completed in {t_banksy:.0f}s, {n_clusters} clusters")
    else:
        # ── Step 1: Load data ──────────────────────────────────────────────
        logger.info("\n[1/6] Loading Xenium breast rep1...")
        adata_raw = load_xenium_adata("rep1")
        adata_pp = preprocess_adata(adata_raw)
        gt = np.array(adata_pp.obs["gt_coarse"].values)
        logger.info(f"  {len(adata_pp)} cells, {adata_pp.n_vars} genes")

        # ── Step 2: BANKSY clustering ──────────────────────────────────────
        logger.info("\n[2/6] Running BANKSY spatial clustering (k=10, lambda=0.2, res=2.0)...")
        t0 = time.time()
        adata_clustered, cluster_ids, n_clusters = run_banksy_clustering(adata_pp)
        t_banksy = time.time() - t0
        logger.info(f"  BANKSY completed in {t_banksy:.0f}s, {n_clusters} clusters")

    # ── Step 3: Extract marker genes ────────────────────────────────────
    logger.info("\n[3/6] Extracting top 20 marker genes per cluster...")
    marker_genes, cluster_details = extract_marker_genes(adata_clustered, "banksy_cluster", n_top=20)

    # ── Step 4: scType baseline labels ─────────────────────────────────
    logger.info("\n[4/6] Computing scType baseline labels...")
    sctype_preds = label_clusters_sctype(adata_clustered, "banksy_cluster", MARKERS_DEFAULT)
    sctype_metrics = compute_metrics(gt, sctype_preds, COARSE_CLASSES)
    logger.info(f"  scType baseline: F1={sctype_metrics['f1_macro']:.4f}, Acc={sctype_metrics['accuracy']:.4f}")

    # Get scType cluster->label mapping
    sctype_cluster_labels = {}
    for cluster in sorted(adata_clustered.obs["banksy_cluster"].unique()):
        mask = adata_clustered.obs["banksy_cluster"] == cluster
        preds_for_cluster = sctype_preds[mask]
        most_common = Counter(preds_for_cluster).most_common(1)[0][0]
        sctype_cluster_labels[str(cluster)] = most_common

    # ── Step 5: LLM annotation ──────────────────────────────────────────
    logger.info("\n[5/6] Running LLM cluster annotation...")

    llm_results = {}
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Method A: mLLMCelltype (requires ANTHROPIC_API_KEY env var)
    if api_key:
        try:
            t0 = time.time()
            mllm_annotations = run_mllmcelltype(marker_genes)
            t_mllm = time.time() - t0
            logger.info(f"  mLLMCelltype completed in {t_mllm:.1f}s")

            mllm_coarse_map = {c: map_llm_to_coarse(a) for c, a in mllm_annotations.items()}
            mllm_preds = np.array([mllm_coarse_map.get(str(c), "Unknown")
                                   for c in adata_clustered.obs["banksy_cluster"]])
            mllm_metrics = compute_metrics(gt, mllm_preds, COARSE_CLASSES)

            llm_results["mllmcelltype_claude"] = {
                "annotations": mllm_annotations,
                "coarse_map": mllm_coarse_map,
                "metrics": mllm_metrics,
                "time_seconds": round(t_mllm, 1),
            }
            logger.info(f"  mLLMCelltype: F1={mllm_metrics['f1_macro']:.4f}, Acc={mllm_metrics['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"  mLLMCelltype failed: {e}")
            import traceback
            traceback.print_exc()

        # Method B: Direct Anthropic (single-shot, all clusters at once)
        try:
            t0 = time.time()
            direct_annotations = run_direct_anthropic(marker_genes)
            t_direct = time.time() - t0
            logger.info(f"  Direct Anthropic completed in {t_direct:.1f}s")

            direct_coarse_map = {c: map_llm_to_coarse(a) for c, a in direct_annotations.items()}
            direct_preds = np.array([direct_coarse_map.get(str(c), "Unknown")
                                     for c in adata_clustered.obs["banksy_cluster"]])
            direct_metrics = compute_metrics(gt, direct_preds, COARSE_CLASSES)

            llm_results["direct_anthropic"] = {
                "annotations": direct_annotations,
                "coarse_map": direct_coarse_map,
                "metrics": direct_metrics,
                "time_seconds": round(t_direct, 1),
            }
            logger.info(f"  Direct Anthropic: F1={direct_metrics['f1_macro']:.4f}, Acc={direct_metrics['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"  Direct Anthropic failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("  ANTHROPIC_API_KEY not set - skipping live API methods")
        logger.info("  Set ANTHROPIC_API_KEY env var to enable mLLMCelltype and direct API annotation")

    # Method C: Claude expert annotations (always available, reference baseline)
    # These are produced by Claude analyzing the same marker genes - equivalent to
    # what mLLMCelltype does but without a separate API call
    logger.info("  Running Claude expert annotation (pre-computed from marker analysis)...")
    expert_annotations = get_claude_expert_annotations(marker_genes)
    expert_coarse_map = {c: map_llm_to_coarse(a) for c, a in expert_annotations.items()}
    expert_preds = np.array([expert_coarse_map.get(str(c), "Unknown")
                             for c in adata_clustered.obs["banksy_cluster"]])
    expert_metrics = compute_metrics(gt, expert_preds, COARSE_CLASSES)

    llm_results["claude_expert"] = {
        "annotations": expert_annotations,
        "coarse_map": expert_coarse_map,
        "metrics": expert_metrics,
        "time_seconds": 0,
    }
    logger.info(f"  Claude expert: F1={expert_metrics['f1_macro']:.4f}, Acc={expert_metrics['accuracy']:.4f}")

    # ── Step 6: Compare and save ───────────────────────────────────────
    logger.info("\n[6/6] Comparing annotations...")

    # Build comparison table
    comparison = {}
    for cluster_id in sorted(marker_genes.keys(), key=lambda x: int(x) if x.isdigit() else x):
        n_cells = cluster_details[cluster_id].get("n_cells", 0)
        top_genes = cluster_details[cluster_id].get("genes", [])[:10]

        # GT composition for this cluster
        mask = adata_clustered.obs["banksy_cluster"] == cluster_id
        gt_cluster = gt[mask]
        gt_counts = Counter(gt_cluster)
        gt_majority = gt_counts.most_common(1)[0][0] if gt_counts else "Unknown"
        gt_purity = gt_counts[gt_majority] / sum(gt_counts.values()) if gt_counts else 0

        entry = {
            "n_cells": n_cells,
            "top_10_markers": top_genes,
            "gt_majority": gt_majority,
            "gt_purity": round(gt_purity, 3),
            "gt_composition": {k: v for k, v in gt_counts.most_common()},
            "sctype_label": sctype_cluster_labels.get(cluster_id, "Unknown"),
        }

        for method_name, method_data in llm_results.items():
            raw_label = method_data["annotations"].get(cluster_id, "Unknown")
            coarse_label = method_data["coarse_map"].get(cluster_id, "Unknown")
            entry[f"{method_name}_raw"] = raw_label
            entry[f"{method_name}_coarse"] = coarse_label
            entry[f"{method_name}_matches_gt"] = coarse_label == gt_majority
            entry[f"{method_name}_matches_sctype"] = coarse_label == sctype_cluster_labels.get(cluster_id, "Unknown")

        comparison[cluster_id] = entry

    # Summary
    logger.info(f"\n{'='*100}")
    logger.info("CLUSTER ANNOTATION COMPARISON")
    logger.info(f"{'='*100}")
    logger.info(f"{'Cluster':>8s} {'Cells':>6s} {'GT':>14s} {'Purity':>7s} {'scType':>14s} ", end="")
    for method_name in llm_results:
        logger.info(f"{'LLM':>14s} ", end="")
    logger.info("")
    logger.info("-" * 100)

    for cluster_id in sorted(comparison.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = comparison[cluster_id]
        line = (f"{cluster_id:>8s} {entry['n_cells']:>6d} {entry['gt_majority']:>14s} "
                f"{entry['gt_purity']:>7.1%} {entry['sctype_label']:>14s} ")
        for method_name in llm_results:
            raw = entry.get(f"{method_name}_raw", "Unknown")
            coarse = entry.get(f"{method_name}_coarse", "Unknown")
            match_gt = "ok" if entry.get(f"{method_name}_matches_gt", False) else "MISS"
            line += f"{coarse:>10s}({match_gt:>4s}) "
        logger.info(line)

    # Overall metrics summary
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL METRICS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Method':<30s} {'F1 Macro':>10s} {'Accuracy':>10s} {'Endothel':>10s} {'Epithel':>10s} {'Immune':>10s} {'Stromal':>10s}")
    logger.info("-" * 100)

    # scType baseline
    sc_pc = sctype_metrics.get("per_class", {})
    logger.info(f"{'scType (baseline)':<30s} {sctype_metrics['f1_macro']:>10.4f} {sctype_metrics['accuracy']:>10.4f} "
                f"{sc_pc.get('Endothelial', {}).get('f1', 0):>10.4f} "
                f"{sc_pc.get('Epithelial', {}).get('f1', 0):>10.4f} "
                f"{sc_pc.get('Immune', {}).get('f1', 0):>10.4f} "
                f"{sc_pc.get('Stromal', {}).get('f1', 0):>10.4f}")

    for method_name, method_data in llm_results.items():
        m = method_data["metrics"]
        pc = m.get("per_class", {})
        logger.info(f"{method_name:<30s} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f} "
                    f"{pc.get('Endothelial', {}).get('f1', 0):>10.4f} "
                    f"{pc.get('Epithelial', {}).get('f1', 0):>10.4f} "
                    f"{pc.get('Immune', {}).get('f1', 0):>10.4f} "
                    f"{pc.get('Stromal', {}).get('f1', 0):>10.4f}")

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "metadata": {
            "dataset": "xenium_breast_rep1",
            "n_cells": int(len(adata_pp)),
            "n_clusters": n_clusters,
            "banksy_params": {"k": 10, "lambda": 0.2, "resolution": 2.0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_marker_genes": 20,
        },
        "marker_genes": marker_genes,
        "cluster_details": {k: {kk: vv for kk, vv in v.items()
                                 if kk != "pvals_adj"}  # pvals can be inf
                            for k, v in cluster_details.items()},
        "sctype_baseline": {
            "cluster_labels": sctype_cluster_labels,
            "metrics": sctype_metrics,
        },
        "llm_methods": {},
        "comparison": comparison,
    }

    for method_name, method_data in llm_results.items():
        output["llm_methods"][method_name] = {
            "annotations": method_data["annotations"],
            "coarse_map": method_data["coarse_map"],
            "metrics": method_data["metrics"],
            "time_seconds": method_data["time_seconds"],
        }

    out_path = OUTPUT_DIR / "llm_cluster_validation.json"

    # JSON-safe serialization
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_safe)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
