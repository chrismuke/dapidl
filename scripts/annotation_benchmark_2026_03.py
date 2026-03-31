#!/usr/bin/env python3
"""Comprehensive Cell Type Annotation Benchmark (March 2026).

Evaluates annotation methods (individual + ensemble) on:
- Xenium breast rep1 (167K cells, 313 genes, expert GT)
- Xenium breast rep2 (118K cells, 313 genes, expert GT)
- Xenium rep1+rep2 combined
- STHELAR breast slides s0,s1,s3,s6 (577K-893K cells, 573 genes, Tangram GT)

Methods tested:
- CellTypist (5+ models)
- SingleR (4 references)
- scType (multiple marker DBs: custom, ScTypeDB, PanglaoDB, CellMarker2.0)
- popV (full ensemble)
- scANVI (transfer learning)
- decoupler (enrichment)
- Ensembles (PopV-style, Universal)

Usage:
    uv run python scripts/annotation_benchmark_2026_03.py [--datasets rep1,rep2,sthelar_s0] [--methods all]
"""

import gc
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from loguru import logger
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="celltypist")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("pipeline_output/annotation_benchmark_2026_03")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")

COARSE_CLASSES = ["Endothelial", "Epithelial", "Immune", "Stromal"]
COARSE_ORDER = {c: i for i, c in enumerate(COARSE_CLASSES)}

# CellTypist models to test
CELLTYPIST_MODELS = [
    "Cells_Adult_Breast.pkl",
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Pan_Fetal_Human.pkl",
    "Adult_Human_Vascular.pkl",
    "Developing_Human_Organs.pkl",
]

# SingleR references
SINGLER_REFS = ["blueprint", "hpca", "monaco", "novershtern"]

# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH LOADING
# ──────────────────────────────────────────────────────────────────────────────

# Janesick et al. ground truth mapping (17 types → 4 coarse)
GT_TO_COARSE = {
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Endothelial": "Endothelial",
    "IRF7+_DCs": "Immune",
    "Invasive_Tumor": "Epithelial",
    "LAMP3+_DCs": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "Mast_Cells": "Immune",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "Perivascular-Like": "Stromal",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Stromal": "Stromal",
    "Unlabeled": "Unknown",
    "Hybrid": "Unknown",
}


def load_xenium_adata(rep: str) -> ad.AnnData:
    """Load Xenium breast cancer replicate with ground truth."""
    if rep == "rep1":
        base = XENIUM_BASE / "xenium-breast-tumor-rep1"
        gt_file = base / "celltypes_ground_truth_rep1_supervised.xlsx"
    elif rep == "rep2":
        base = XENIUM_BASE / "xenium-breast-tumor-rep2"
        gt_file = base / "celltypes_ground_truth_rep2_supervised.xlsx"
    else:
        raise ValueError(f"Unknown rep: {rep}")

    outs = base / "outs" if (base / "outs").exists() else base

    # Load expression
    h5_path = outs / "cell_feature_matrix.h5"
    logger.info(f"Loading {rep} from {h5_path}")
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    # Load cell metadata
    cells_path = outs / "cells.parquet"
    if cells_path.exists():
        cells_df = pd.read_parquet(cells_path)
        cells_df["cell_id"] = cells_df["cell_id"].astype(str)
        cells_df.index = cells_df["cell_id"]
        # Align with adata
        common = adata.obs_names.intersection(cells_df.index)
        adata = adata[common].copy()
        for col in ["x_centroid", "y_centroid"]:
            if col in cells_df.columns:
                adata.obs[col] = cells_df.loc[adata.obs_names, col].values

    # Load ground truth
    if gt_file.exists():
        logger.info(f"Loading ground truth from {gt_file}")
        gt_df = pd.read_excel(gt_file)
        # Find barcode and cluster columns (case-insensitive)
        barcode_col = next((c for c in gt_df.columns if "barcode" in c.lower()), gt_df.columns[0])
        cluster_col = next(
            (c for c in gt_df.columns if c.lower() in ("cluster", "cell_type", "celltype", "ground_truth")),
            gt_df.columns[1],
        )
        gt_df[barcode_col] = gt_df[barcode_col].astype(str)
        gt_df = gt_df.set_index(barcode_col)
        common_gt = adata.obs_names.intersection(gt_df.index)
        adata = adata[common_gt].copy()
        adata.obs["gt_fine"] = gt_df.loc[adata.obs_names, cluster_col].values
        adata.obs["gt_coarse"] = adata.obs["gt_fine"].map(GT_TO_COARSE).fillna("Unknown")
        # Filter unknown
        mask = adata.obs["gt_coarse"] != "Unknown"
        n_before = len(adata)
        adata = adata[mask].copy()
        logger.info(f"  {rep}: {n_before} → {len(adata)} cells (removed {n_before - len(adata)} Unknown/Hybrid)")
    else:
        logger.warning(f"No ground truth found at {gt_file}")
        adata.obs["gt_fine"] = "Unknown"
        adata.obs["gt_coarse"] = "Unknown"

    adata.obs["dataset"] = rep
    return adata


def load_sthelar_adata(slide_name: str) -> ad.AnnData:
    """Load STHELAR breast slide with Tangram ground truth."""
    from dapidl.data.sthelar_reader import TANGRAM_TO_COARSE

    zarr_path = STHELAR_BASE / f"sdata_{slide_name}.zarr" / f"sdata_{slide_name}.zarr"
    table_path = zarr_path / "tables" / "table_combined"

    logger.info(f"Loading STHELAR {slide_name} from {table_path}")
    adata = ad.read_zarr(str(table_path))

    # Map Tangram labels to coarse
    if "ct_tangram" in adata.obs.columns:
        adata.obs["gt_fine"] = adata.obs["ct_tangram"].astype(str)
        adata.obs["gt_coarse"] = adata.obs["gt_fine"].map(TANGRAM_TO_COARSE).fillna("Unknown")
        mask = adata.obs["gt_coarse"] != "Unknown"
        n_before = len(adata)
        adata = adata[mask].copy()
        logger.info(f"  {slide_name}: {n_before} → {len(adata)} cells (removed {n_before - len(adata)} Unknown)")
    else:
        logger.warning(f"No ct_tangram column in {slide_name}")
        adata.obs["gt_fine"] = "Unknown"
        adata.obs["gt_coarse"] = "Unknown"

    adata.obs["dataset"] = slide_name
    return adata


# ──────────────────────────────────────────────────────────────────────────────
# ANNOTATION METHODS
# ──────────────────────────────────────────────────────────────────────────────


def preprocess_adata(adata: ad.AnnData) -> ad.AnnData:
    """Standard preprocessing for annotation methods."""
    a = adata.copy()
    if issparse(a.X):
        a.X = a.X.toarray()
    # Store raw counts
    a.layers["raw"] = a.X.copy()
    # Normalize
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def run_celltypist(adata: ad.AnnData, model_name: str) -> dict:
    """Run a single CellTypist model."""
    import celltypist
    from celltypist import models as ct_models

    try:
        model = ct_models.Model.load(model_name)
    except Exception:
        ct_models.download_models(model=model_name, force_update=False)
        model = ct_models.Model.load(model_name)

    predictions = celltypist.annotate(adata, model=model, majority_voting=True)
    result = predictions.to_adata()

    preds = result.obs["majority_voting"].astype(str).values
    conf = result.obs["conf_score"].values if "conf_score" in result.obs.columns else np.ones(len(result))

    return {"predictions": preds, "confidence": conf, "method": f"celltypist_{model_name.replace('.pkl', '')}"}


def run_singler(adata: ad.AnnData, reference: str) -> dict:
    """Run SingleR with a specific reference."""
    from dapidl.pipeline.components.annotators.singler import (
        SINGLER_REFERENCES,
        _fix_libstdcxx,
        is_singler_available,
    )

    if not is_singler_available():
        return {"predictions": None, "error": "SingleR not available"}

    _fix_libstdcxx()
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    r = ro.r
    singler_r = importr("SingleR")
    celldex_r = importr("celldex")  # noqa: F841

    # Get reference data
    ref_name = SINGLER_REFERENCES[reference]
    ref_data = r(f"celldex::{ref_name}()")

    # Prepare expression matrix (genes x cells) — NO auto-converter context
    expr = adata.X.toarray().T if issparse(adata.X) else np.asarray(adata.X).T
    expr_r = r["matrix"](
        ro.FloatVector(expr.flatten().tolist()),
        nrow=expr.shape[0],
        ncol=expr.shape[1],
    )
    expr_r.rownames = ro.StrVector(list(adata.var_names))
    expr_r.colnames = ro.StrVector(list(adata.obs_names))

    # Get reference labels using R $ accessor (rpy2 3.6+ doesn't have rx2 on RS4)
    ref_labels = r("function(x) colData(x)$label.main")(ref_data)

    # Run SingleR
    results = singler_r.SingleR(test=expr_r, ref=ref_data, labels=ref_labels)

    # Extract results using R functions (RS4 slots not directly accessible in rpy2 3.6+)
    preds = np.array(list(r("function(x) as.character(x$labels)")(results)))
    scores_r = r("function(x) as.matrix(x$scores)")(results)
    scores = np.array(scores_r).reshape(len(preds), -1)

    conf = scores.max(axis=1)
    conf = (conf + 1) / 2  # Normalize correlation [-1,1] → [0,1]

    return {"predictions": preds, "confidence": conf, "method": f"singler_{reference}"}


def run_sctype(adata: ad.AnnData, markers: dict, marker_name: str) -> dict:
    """Run scType with given marker gene sets."""
    # Build scores per cell type
    gene_names = list(adata.var_names)
    # Always convert to dense numpy array
    if issparse(adata.X):
        expr = np.asarray(adata.X.toarray())
    else:
        expr = np.asarray(adata.X)
    n_cells = len(adata)

    scores = {}
    for ct, m in markers.items():
        pos_genes = [g for g in m.get("positive", []) if g in gene_names]
        neg_genes = [g for g in m.get("negative", []) if g in gene_names]

        if not pos_genes:
            scores[ct] = np.zeros(n_cells)
            continue

        pos_idx = [gene_names.index(g) for g in pos_genes]
        pos_mean = np.asarray(expr[:, pos_idx].mean(axis=1)).ravel()

        if neg_genes:
            neg_idx = [gene_names.index(g) for g in neg_genes]
            neg_mean = np.asarray(expr[:, neg_idx].mean(axis=1)).ravel()
            scores[ct] = pos_mean - 0.5 * neg_mean
        else:
            scores[ct] = pos_mean

    # Debug logging
    n_scored = sum(1 for s in scores.values() if np.any(s > 0))
    logger.info(f"    sctype debug: {len(scores)} types, {n_scored} with>0, n_cells={n_cells}, expr={expr.shape}")

    # Filter out cell types with zero scores everywhere
    valid_cts = {ct: s for ct, s in scores.items() if np.any(s > 0)}
    if not valid_cts:
        # Fallback: use all scores even if zero
        valid_cts = scores

    score_matrix = np.column_stack([valid_cts[ct] for ct in valid_cts])
    ct_names = list(valid_cts.keys())

    if score_matrix.shape[1] == 0:
        return {"predictions": np.full(n_cells, "Unknown"), "confidence": np.zeros(n_cells),
                "method": f"sctype_{marker_name}"}

    best_idx = score_matrix.argmax(axis=1)
    preds = np.array([ct_names[i] for i in best_idx])

    # Confidence: difference between best and second-best score
    if score_matrix.shape[1] >= 2:
        sorted_scores = np.sort(score_matrix, axis=1)
        conf = sorted_scores[:, -1] - sorted_scores[:, -2]
        max_conf = conf.max()
        conf = np.clip(conf / (max_conf + 1e-8), 0, 1) if max_conf > 0 else np.zeros(n_cells)
    else:
        conf = np.ones(n_cells) * 0.5

    return {"predictions": preds, "confidence": conf, "method": f"sctype_{marker_name}"}


def run_decoupler(adata: ad.AnnData, marker_source: str = "panglaodb") -> dict:
    """Run decoupler ULM enrichment for cell type annotation."""
    try:
        import decoupler as dc
    except ImportError:
        return {"predictions": None, "error": "decoupler not installed"}

    if marker_source == "panglaodb":
        try:
            markers = dc.op.resource("PanglaoDB")
            if "organism" in markers.columns:
                markers = markers[markers["organism"] == "Hs"]
        except Exception as e:
            return {"predictions": None, "error": f"Failed to get PanglaoDB: {e}"}
    else:
        return {"predictions": None, "error": f"Unknown marker source: {marker_source}"}

    try:
        # Build network: source=gene, target=cell_type, weight=1
        net = markers[["genesymbol", "cell_type"]].copy()
        net.columns = ["source", "target"]
        net["weight"] = 1.0
        net = net.drop_duplicates()

        # Run ULM (univariate linear model) per cell — returns (estimate_df, pvals_df)
        estimate, pvals = dc.mt.ulm(adata, net, source="source", target="target", weight="weight",
                                     use_raw=False, tmin=1)

        ct_names = list(estimate.columns) if hasattr(estimate, 'columns') else [f"ct_{i}" for i in range(estimate.shape[1])]
        scores = estimate.values if hasattr(estimate, 'values') else np.asarray(estimate)

        best_idx = scores.argmax(axis=1)
        preds = np.array([ct_names[i] for i in best_idx])
        conf = np.clip(scores.max(axis=1) / (np.abs(scores).max() + 1e-8), 0, 1)

        return {"predictions": preds, "confidence": conf, "method": f"decoupler_{marker_source}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"predictions": None, "error": f"decoupler failed: {e}"}


def run_popv(adata_raw: ad.AnnData, organ: str = "Mammary") -> dict:
    """Run full popV pipeline via HubModel API (popV 0.6.0+)."""
    try:
        from popv.hub import HubModel
    except ImportError:
        return {"predictions": None, "error": "popV not installed"}

    try:
        a = adata_raw.copy()
        # popV needs raw counts
        if "raw" in a.layers:
            a.X = a.layers["raw"].copy()

        hub_name = f"popV/tabula_sapiens_{organ}"
        logger.info(f"  Loading HubModel: {hub_name}")
        hub = HubModel.pull_from_huggingface_hub(hub_name)

        # Check gene format
        sample_genes = list(a.var_names[:5])
        genes_are_ensembl = all(str(g).startswith("ENSG") for g in sample_genes)
        gene_symbols_param = None if genes_are_ensembl else "feature_name"

        logger.info(f"  Running annotate_data ({a.n_obs} cells, gene_symbols={gene_symbols_param})...")
        result = hub.annotate_data(a, gene_symbols=gene_symbols_param)

        if "popv_prediction" in result.obs.columns:
            # popV may subsample — align back to original cells
            n_orig = len(a)
            n_result = len(result)
            if n_result < n_orig:
                logger.info(f"  popV subsampled {n_orig} → {n_result} cells, aligning...")
                # Create full-size arrays with "Unknown" for missing cells
                preds_full = np.full(n_orig, "Unknown", dtype=object)
                conf_full = np.zeros(n_orig)
                # Match by obs_names
                orig_names = list(a.obs_names)
                result_names = list(result.obs_names)
                name_to_idx = {name: i for i, name in enumerate(orig_names)}
                for j, rname in enumerate(result_names):
                    if rname in name_to_idx:
                        preds_full[name_to_idx[rname]] = result.obs["popv_prediction"].iloc[j]
                        if "popv_prediction_score" in result.obs.columns:
                            conf_full[name_to_idx[rname]] = result.obs["popv_prediction_score"].iloc[j]
                        else:
                            conf_full[name_to_idx[rname]] = 1.0
                preds = preds_full
                conf = conf_full
            else:
                preds = np.array(result.obs["popv_prediction"].values)
                conf = np.array(result.obs["popv_prediction_score"].values) if "popv_prediction_score" in result.obs.columns else np.ones(n_result)

            # Also collect per-method predictions for analysis
            method_cols = [c for c in result.obs.columns if c.endswith("_prediction") and c != "popv_prediction"]
            per_method = {c: result.obs[c].values for c in method_cols}

            return {
                "predictions": preds,
                "confidence": conf,
                "method": "popv_full",
                "per_method": per_method,
                "n_methods": len(method_cols),
            }
        else:
            # Fallback: check majority vote
            for col in ["popv_majority_vote_prediction", "popv_Mammary_prediction"]:
                if col in result.obs.columns:
                    return {
                        "predictions": np.array(result.obs[col].values),
                        "confidence": np.ones(len(result)),
                        "method": "popv_full",
                    }
            return {"predictions": None, "error": f"No prediction column. Available: {list(result.obs.columns)}"}

    except Exception as e:
        logger.error(f"popV failed: {e}")
        import traceback

        traceback.print_exc()
        return {"predictions": None, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# MARKER GENE DATABASES
# ──────────────────────────────────────────────────────────────────────────────


def get_default_markers() -> dict:
    """Get the custom DEFAULT_MARKERS from sctype.py (hand-curated)."""
    from dapidl.pipeline.components.annotators.sctype import DEFAULT_MARKERS

    return DEFAULT_MARKERS


def get_sctype_db_markers() -> dict:
    """Get ScTypeDB markers for breast tissue (Immune system)."""
    from dapidl.pipeline.components.annotators.sctype_db import get_tissue_markers

    return get_tissue_markers("breast", include_immune=True)


def get_panglaodb_markers() -> dict:
    """Get PanglaoDB markers via decoupler."""
    try:
        import decoupler as dc

        markers = dc.op.resource("PanglaoDB")
        # PanglaoDB uses organism column with "Hs" for human
        if "organism" in markers.columns:
            markers = markers[markers["organism"] == "Hs"]
        # Filter for canonical markers if available
        if "canonical_marker" in markers.columns:
            markers = markers[markers["canonical_marker"] == True]  # noqa: E712
        genesets = {}
        for ct, grp in markers.groupby("cell_type"):
            genes = grp["genesymbol"].tolist()
            genesets[str(ct)] = {"positive": genes, "negative": []}
        return genesets
    except Exception as e:
        logger.error(f"Failed to load PanglaoDB: {e}")
        return {}


def get_cellmarker2_markers() -> dict:
    """Get CellMarker 2.0 markers (via GSEApy Enrichr or local cache)."""
    cache_path = OUTPUT_DIR / "cellmarker2_human_breast.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    # Try to download CellMarker 2.0
    try:
        import urllib.request

        url = "http://bio-bigdata.hrbmu.edu.cn/CellMarker/CellMarker_download_files/file/Cell_marker_Human.xlsx"
        local_path = OUTPUT_DIR / "CellMarker2_Human.xlsx"
        if not local_path.exists():
            logger.info("Downloading CellMarker 2.0...")
            urllib.request.urlretrieve(url, local_path)

        cm = pd.read_excel(local_path)
        # Filter for breast/mammary tissue
        breast_mask = cm["tissue_type"].str.contains("Breast|Mammary|Milk", case=False, na=False)
        # Also get general immune/stromal markers
        immune_mask = cm["tissue_type"].str.contains("Blood|Immune|Lymph", case=False, na=False)
        cm_filtered = cm[breast_mask | immune_mask]

        genesets = {}
        for ct, grp in cm_filtered.groupby("cell_name"):
            genes = []
            for _, row in grp.iterrows():
                marker = str(row.get("marker", ""))
                genes.extend([g.strip() for g in marker.split(",") if g.strip()])
            if genes:
                genesets[ct] = {"positive": list(set(genes)), "negative": []}

        cache_path.write_text(json.dumps(genesets, indent=2))
        return genesets
    except Exception as e:
        logger.error(f"Failed to get CellMarker 2.0: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# BROAD CATEGORY MAPPING
# ──────────────────────────────────────────────────────────────────────────────


def map_predictions_to_coarse(predictions: np.ndarray) -> np.ndarray:
    """Map fine-grained predictions to coarse categories."""
    from dapidl.pipeline.components.annotators.mapping import map_to_broad_category

    return np.array([map_to_broad_category(str(p)) for p in predictions])


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str] | None = None) -> dict:
    """Compute comprehensive metrics."""
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    classes = [c for c in classes if c != "Unknown"]

    # Filter out Unknown
    mask = (y_true != "Unknown") & (y_pred != "Unknown")
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"error": "No valid predictions"}

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=classes)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=classes)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0

    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)

    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "cohen_kappa": round(float(kappa), 4),
        "mcc": round(float(mcc), 4),
        "n_cells": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": classes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE METHODS
# ──────────────────────────────────────────────────────────────────────────────


def run_ensemble_unweighted(method_results: dict[str, dict], n_cells: int) -> dict:
    """Unweighted majority voting (popV-style)."""
    votes = np.empty((n_cells, len(method_results)), dtype=object)
    valid_methods = []
    for i, (name, result) in enumerate(method_results.items()):
        if result.get("predictions") is not None and result.get("coarse_predictions") is not None:
            votes[:, len(valid_methods)] = result["coarse_predictions"]
            valid_methods.append(name)

    if not valid_methods:
        return {"predictions": None, "error": "No valid methods for ensemble"}

    votes = votes[:, : len(valid_methods)]
    preds = []
    confs = []
    for i in range(n_cells):
        cell_votes = [v for v in votes[i, :] if v and v != "Unknown"]
        if not cell_votes:
            preds.append("Unknown")
            confs.append(0.0)
        else:
            from collections import Counter

            counts = Counter(cell_votes)
            winner = counts.most_common(1)[0]
            preds.append(winner[0])
            confs.append(winner[1] / len(cell_votes))

    return {
        "predictions": np.array(preds),
        "confidence": np.array(confs),
        "method": f"ensemble_unweighted_{len(valid_methods)}methods",
        "n_methods": len(valid_methods),
        "methods_used": valid_methods,
    }


def run_ensemble_confidence_weighted(method_results: dict[str, dict], n_cells: int) -> dict:
    """Confidence-weighted voting."""
    class_scores = defaultdict(lambda: np.zeros(n_cells))
    valid_methods = []
    for name, result in method_results.items():
        if result.get("predictions") is not None and result.get("coarse_predictions") is not None:
            coarse = result["coarse_predictions"]
            conf = result.get("confidence", np.ones(n_cells))
            for i in range(n_cells):
                if coarse[i] and coarse[i] != "Unknown":
                    class_scores[coarse[i]][i] += conf[i]
            valid_methods.append(name)

    if not valid_methods:
        return {"predictions": None, "error": "No valid methods"}

    # Pick class with highest weighted score per cell
    all_classes = list(class_scores.keys())
    score_matrix = np.column_stack([class_scores[c] for c in all_classes])
    best_idx = score_matrix.argmax(axis=1)
    preds = np.array([all_classes[i] for i in best_idx])
    total_scores = score_matrix.sum(axis=1)
    confs = np.where(total_scores > 0, score_matrix.max(axis=1) / total_scores, 0)

    return {
        "predictions": preds,
        "confidence": confs,
        "method": f"ensemble_confweighted_{len(valid_methods)}methods",
        "n_methods": len(valid_methods),
        "methods_used": valid_methods,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────


def plot_method_comparison(all_results: dict, output_dir: Path) -> None:
    """Create comprehensive comparison plots."""
    # 1. Bar chart: F1 macro across methods and datasets
    fig, ax = plt.subplots(figsize=(16, 8))
    datasets = sorted(set(ds for ds, _ in all_results.keys()))
    methods = sorted(set(m for _, m in all_results.keys()))

    x = np.arange(len(methods))
    width = 0.8 / max(len(datasets), 1)

    for i, ds in enumerate(datasets):
        f1s = []
        for m in methods:
            r = all_results.get((ds, m), {})
            f1s.append(r.get("f1_macro", 0))
        ax.bar(x + i * width, f1s, width, label=ds, alpha=0.8)

    ax.set_xlabel("Method")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Cell Type Annotation: F1 Macro by Method and Dataset")
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "01_method_comparison.png", dpi=150)
    plt.close()

    # 2. Per-class F1 heatmap
    for ds in datasets:
        ds_methods = [(m, all_results[(ds, m)]) for _, m in all_results.keys() if _ == ds and "per_class" in all_results.get((ds, m), {})]
        if not ds_methods:
            continue

        method_names = [m for m, _ in ds_methods]
        class_names = COARSE_CLASSES
        f1_matrix = np.zeros((len(method_names), len(class_names)))
        for i, (m, r) in enumerate(ds_methods):
            for j, c in enumerate(class_names):
                f1_matrix[i, j] = r.get("per_class", {}).get(c, {}).get("f1", 0)

        fig, ax = plt.subplots(figsize=(10, max(6, len(method_names) * 0.4)))
        im = ax.imshow(f1_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticks(range(len(method_names)))
        ax.set_yticklabels(method_names, fontsize=8)
        for i in range(len(method_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{f1_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if f1_matrix[i, j] < 0.5 else "black")
        plt.colorbar(im, ax=ax, label="F1 Score")
        ax.set_title(f"Per-Class F1 Heatmap: {ds}")
        plt.tight_layout()
        plt.savefig(output_dir / f"02_perclass_heatmap_{ds}.png", dpi=150)
        plt.close()

    # 3. Confusion matrices for top methods
    for (ds, method), r in all_results.items():
        if "confusion_matrix" not in r:
            continue
        cm = np.array(r["confusion_matrix"])
        classes = r.get("class_names", COARSE_CLASSES)
        if cm.shape[0] != len(classes):
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        # Normalize
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=9)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
        plt.colorbar(im, ax=ax, label="Fraction")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix: {method} on {ds}")
        plt.tight_layout()
        safe_name = method.replace("/", "_").replace(" ", "_")[:50]
        plt.savefig(output_dir / f"03_cm_{ds}_{safe_name}.png", dpi=150)
        plt.close()

    # 4. Marker database comparison
    marker_methods = [m for _, m in all_results.keys() if "sctype_" in m]
    if marker_methods:
        marker_methods = sorted(set(marker_methods))
        fig, ax = plt.subplots(figsize=(12, 6))
        for ds in datasets:
            f1s = [all_results.get((ds, m), {}).get("f1_macro", 0) for m in marker_methods]
            ax.plot(marker_methods, f1s, "o-", label=ds, markersize=8)
        ax.set_ylabel("F1 Macro")
        ax.set_title("Marker Gene Database Comparison (scType)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "04_marker_db_comparison.png", dpi=150)
        plt.close()

    # 5. Ensemble vs Individual
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ds in zip(axes, datasets[:2]):
        individual = [(m, all_results[(ds, m)].get("f1_macro", 0)) for _, m in all_results.keys()
                       if _ == ds and "ensemble" not in m]
        ensemble = [(m, all_results[(ds, m)].get("f1_macro", 0)) for _, m in all_results.keys()
                     if _ == ds and "ensemble" in m]
        if individual:
            names, vals = zip(*sorted(individual, key=lambda x: -x[1]))
            y = range(len(names))
            ax.barh(y, vals, color="steelblue", alpha=0.7, label="Individual")
        if ensemble:
            names_e, vals_e = zip(*sorted(ensemble, key=lambda x: -x[1]))
            y_e = range(len(names), len(names) + len(names_e))
            ax.barh(y_e, vals_e, color="coral", alpha=0.7, label="Ensemble")
            names = list(names) + list(names_e)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("F1 Macro")
        ax.set_title(f"Individual vs Ensemble: {ds}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "05_ensemble_vs_individual.png", dpi=150)
    plt.close()

    logger.info(f"All plots saved to {output_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_dataset(dataset_name: str, adata_raw: ad.AnnData, adata_pp: ad.AnnData) -> dict:
    """Run all methods on a single dataset and return results."""
    results = {}
    gt_coarse = adata_pp.obs["gt_coarse"].values

    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATING: {dataset_name} ({len(adata_pp)} cells)")
    logger.info(f"{'='*80}")
    logger.info(f"GT distribution: {dict(pd.Series(gt_coarse).value_counts())}")

    # ── CellTypist models ────────────────────────────────────────────────
    for model_name in CELLTYPIST_MODELS:
        key = f"celltypist_{model_name.replace('.pkl', '')}"
        logger.info(f"  Running {key}...")
        try:
            t0 = time.time()
            result = run_celltypist(adata_pp, model_name)
            result["coarse_predictions"] = map_predictions_to_coarse(result["predictions"])
            metrics = compute_metrics(gt_coarse, result["coarse_predictions"], COARSE_CLASSES)
            metrics["runtime_s"] = round(time.time() - t0, 1)
            results[key] = {**metrics, "raw_result": result}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f} ({metrics['runtime_s']}s)")
        except Exception as e:
            logger.error(f"    → FAILED: {e}")
            results[key] = {"error": str(e)}

    # ── SingleR references ───────────────────────────────────────────────
    for ref in SINGLER_REFS:
        key = f"singler_{ref}"
        logger.info(f"  Running {key}...")
        try:
            t0 = time.time()
            result = run_singler(adata_pp, ref)
            if result.get("predictions") is None:
                results[key] = {"error": result.get("error", "Unknown")}
                logger.warning(f"    → SKIPPED: {result.get('error')}")
                continue
            result["coarse_predictions"] = map_predictions_to_coarse(result["predictions"])
            metrics = compute_metrics(gt_coarse, result["coarse_predictions"], COARSE_CLASSES)
            metrics["runtime_s"] = round(time.time() - t0, 1)
            results[key] = {**metrics, "raw_result": result}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f} ({metrics['runtime_s']}s)")
        except Exception as e:
            logger.error(f"    → FAILED: {e}")
            results[key] = {"error": str(e)}

    # ── scType with different marker databases ───────────────────────────
    marker_dbs = {
        "custom_default": get_default_markers(),
        "sctype_db_immune": get_sctype_db_markers(),
        "panglaodb": get_panglaodb_markers(),
        "cellmarker2": get_cellmarker2_markers(),
    }
    for db_name, markers in marker_dbs.items():
        if not markers:
            logger.warning(f"  Skipping sctype_{db_name}: empty marker set")
            continue
        key = f"sctype_{db_name}"
        logger.info(f"  Running {key} ({len(markers)} cell types)...")
        try:
            t0 = time.time()
            result = run_sctype(adata_pp, markers, db_name)
            result["coarse_predictions"] = map_predictions_to_coarse(result["predictions"])
            metrics = compute_metrics(gt_coarse, result["coarse_predictions"], COARSE_CLASSES)
            metrics["runtime_s"] = round(time.time() - t0, 1)
            metrics["n_marker_types"] = len(markers)
            results[key] = {**metrics, "raw_result": result}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f} ({metrics['runtime_s']}s)")
        except Exception as e:
            logger.error(f"    → FAILED: {e}")
            results[key] = {"error": str(e)}

    # ── decoupler (PanglaoDB) ────────────────────────────────────────────
    logger.info("  Running decoupler_panglaodb...")
    try:
        t0 = time.time()
        result = run_decoupler(adata_pp.copy(), "panglaodb")
        if result.get("predictions") is not None:
            result["coarse_predictions"] = map_predictions_to_coarse(result["predictions"])
            metrics = compute_metrics(gt_coarse, result["coarse_predictions"], COARSE_CLASSES)
            metrics["runtime_s"] = round(time.time() - t0, 1)
            results["decoupler_panglaodb"] = {**metrics, "raw_result": result}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f}")
        else:
            results["decoupler_panglaodb"] = {"error": result.get("error")}
            logger.warning(f"    → SKIPPED: {result.get('error')}")
    except Exception as e:
        logger.error(f"    → FAILED: {e}")
        results["decoupler_panglaodb"] = {"error": str(e)}

    # ── popV ─────────────────────────────────────────────────────────────
    logger.info("  Running popv_full...")
    try:
        t0 = time.time()
        result = run_popv(adata_raw)
        if result.get("predictions") is not None:
            result["coarse_predictions"] = map_predictions_to_coarse(result["predictions"])
            metrics = compute_metrics(gt_coarse, result["coarse_predictions"], COARSE_CLASSES)
            metrics["runtime_s"] = round(time.time() - t0, 1)
            results["popv_full"] = {**metrics, "raw_result": result}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f}")
        else:
            results["popv_full"] = {"error": result.get("error")}
            logger.warning(f"    → SKIPPED: {result.get('error')}")
    except Exception as e:
        logger.error(f"    → FAILED: {e}")
        results["popv_full"] = {"error": str(e)}

    # ── Ensembles ────────────────────────────────────────────────────────
    # Collect all successful method results
    method_pool = {}
    for key, r in results.items():
        if "raw_result" in r and r["raw_result"].get("coarse_predictions") is not None:
            method_pool[key] = r["raw_result"]

    if len(method_pool) >= 2:
        # Unweighted ensemble (all methods)
        logger.info(f"  Running ensemble_unweighted ({len(method_pool)} methods)...")
        ens_result = run_ensemble_unweighted(method_pool, len(adata_pp))
        if ens_result.get("predictions") is not None:
            metrics = compute_metrics(gt_coarse, ens_result["predictions"], COARSE_CLASSES)
            results[ens_result["method"]] = {**metrics, "raw_result": ens_result}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f}")

        # Confidence-weighted ensemble (all methods)
        logger.info(f"  Running ensemble_confweighted ({len(method_pool)} methods)...")
        ens_result2 = run_ensemble_confidence_weighted(method_pool, len(adata_pp))
        if ens_result2.get("predictions") is not None:
            metrics = compute_metrics(gt_coarse, ens_result2["predictions"], COARSE_CLASSES)
            results[ens_result2["method"]] = {**metrics, "raw_result": ens_result2}
            logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f}")

        # CellTypist-only ensemble
        ct_pool = {k: v for k, v in method_pool.items() if "celltypist" in k}
        if len(ct_pool) >= 2:
            logger.info(f"  Running celltypist_ensemble ({len(ct_pool)} models)...")
            ct_ens = run_ensemble_unweighted(ct_pool, len(adata_pp))
            if ct_ens.get("predictions") is not None:
                ct_ens["method"] = f"ensemble_celltypist_{len(ct_pool)}models"
                metrics = compute_metrics(gt_coarse, ct_ens["predictions"], COARSE_CLASSES)
                results[ct_ens["method"]] = {**metrics, "raw_result": ct_ens}
                logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f}")

        # CellTypist + SingleR ensemble (popV-style)
        popv_pool = {k: v for k, v in method_pool.items() if "celltypist" in k or "singler" in k}
        if len(popv_pool) >= 3:
            logger.info(f"  Running popv_style_ensemble ({len(popv_pool)} methods)...")
            popv_ens = run_ensemble_unweighted(popv_pool, len(adata_pp))
            if popv_ens.get("predictions") is not None:
                popv_ens["method"] = f"ensemble_popv_style_{len(popv_pool)}methods"
                metrics = compute_metrics(gt_coarse, popv_ens["predictions"], COARSE_CLASSES)
                results[popv_ens["method"]] = {**metrics, "raw_result": popv_ens}
                logger.info(f"    → F1={metrics['f1_macro']:.3f} Acc={metrics['accuracy']:.3f}")

    return results


def main():
    """Run full benchmark."""
    logger.info("=" * 80)
    logger.info("CELL TYPE ANNOTATION BENCHMARK (March 2026)")
    logger.info("=" * 80)

    all_results = {}
    all_adata = {}

    # ── Load datasets ────────────────────────────────────────────────────
    datasets_to_run = os.environ.get("BENCH_DATASETS", "rep1,rep2,breast_s0").split(",")

    for ds_name in datasets_to_run:
        ds_name = ds_name.strip()
        try:
            if ds_name.startswith("rep"):
                adata_raw = load_xenium_adata(ds_name)
            elif ds_name.startswith("breast_s"):
                adata_raw = load_sthelar_adata(ds_name)
            else:
                logger.warning(f"Unknown dataset: {ds_name}")
                continue

            adata_pp = preprocess_adata(adata_raw)
            all_adata[ds_name] = (adata_raw, adata_pp)
            logger.info(f"Loaded {ds_name}: {len(adata_pp)} cells, {adata_pp.n_vars} genes")
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}")
            import traceback

            traceback.print_exc()

    # ── Run evaluations ──────────────────────────────────────────────────
    for ds_name, (adata_raw, adata_pp) in all_adata.items():
        ds_results = evaluate_dataset(ds_name, adata_raw, adata_pp)
        for method, metrics in ds_results.items():
            # Strip raw_result for JSON serialization
            clean = {k: v for k, v in metrics.items() if k != "raw_result"}
            all_results[(ds_name, method)] = clean

        # Save per-dataset results
        ds_out = {method: {k: v for k, v in metrics.items() if k != "raw_result"}
                  for method, metrics in ds_results.items()}
        with open(OUTPUT_DIR / f"results_{ds_name}.json", "w") as f:
            json.dump(ds_out, f, indent=2, default=str)

        # Free memory
        del adata_raw, adata_pp
        gc.collect()

    # ── Combined rep1+rep2 ───────────────────────────────────────────────
    if "rep1" in all_adata and "rep2" in all_adata:
        logger.info("\n=== Running on combined rep1+rep2 ===")
        rep1_raw, _ = all_adata["rep1"]
        rep2_raw, _ = all_adata["rep2"]
        combined_raw = ad.concat([rep1_raw, rep2_raw], join="inner")
        combined_pp = preprocess_adata(combined_raw)
        ds_results = evaluate_dataset("rep1_rep2", combined_raw, combined_pp)
        for method, metrics in ds_results.items():
            clean = {k: v for k, v in metrics.items() if k != "raw_result"}
            all_results[("rep1_rep2", method)] = clean
        with open(OUTPUT_DIR / "results_rep1_rep2.json", "w") as f:
            json.dump({m: {k: v for k, v in met.items() if k != "raw_result"}
                       for m, met in ds_results.items()}, f, indent=2, default=str)
        del combined_raw, combined_pp
        gc.collect()

    # ── Summary table ────────────────────────────────────────────────────
    logger.info(f"\n{'='*100}")
    logger.info("SUMMARY")
    logger.info(f"{'='*100}")

    # Build summary
    summary_rows = []
    for (ds, method), metrics in sorted(all_results.items()):
        if "error" in metrics:
            summary_rows.append({
                "dataset": ds, "method": method, "f1_macro": None,
                "accuracy": None, "error": metrics["error"]
            })
        else:
            row = {"dataset": ds, "method": method}
            row.update({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix", "class_names")})
            for c in COARSE_CLASSES:
                row[f"f1_{c}"] = metrics.get("per_class", {}).get(c, {}).get("f1", None)
            summary_rows.append(row)

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(OUTPUT_DIR / "summary.csv")

    # Print top results per dataset
    for ds in sorted(set(d for d, _ in all_results.keys())):
        logger.info(f"\n--- {ds} ---")
        ds_results = {m: r for (d, m), r in all_results.items() if d == ds and "f1_macro" in r}
        ranked = sorted(ds_results.items(), key=lambda x: -x[1].get("f1_macro", 0))
        logger.info(f"{'Method':<50s} {'F1':>6s} {'Acc':>6s} {'Endo':>6s} {'Epi':>6s} {'Imm':>6s} {'Str':>6s}")
        logger.info("-" * 92)
        for method, r in ranked:
            pc = r.get("per_class", {})
            logger.info(
                f"{method:<50s} {r['f1_macro']:>6.3f} {r['accuracy']:>6.3f} "
                f"{pc.get('Endothelial', {}).get('f1', 0):>6.3f} "
                f"{pc.get('Epithelial', {}).get('f1', 0):>6.3f} "
                f"{pc.get('Immune', {}).get('f1', 0):>6.3f} "
                f"{pc.get('Stromal', {}).get('f1', 0):>6.3f}"
            )

    # ── Visualizations ───────────────────────────────────────────────────
    logger.info("\nGenerating visualizations...")
    plot_method_comparison(all_results, OUTPUT_DIR)

    # Save full results
    serializable = {}
    for (ds, method), metrics in all_results.items():
        serializable[f"{ds}/{method}"] = metrics
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {OUTPUT_DIR}/")
    logger.info("BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
