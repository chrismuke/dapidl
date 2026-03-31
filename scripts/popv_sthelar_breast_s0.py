#!/usr/bin/env python
"""
Full popV Retrain Pipeline for STHELAR Breast S0 Dataset

Runs the real popV Process_Query + annotate_data pipeline (retrain mode) with
Tabula Sapiens Mammary reference on STHELAR breast_s0 (576,963 cells).

Strategy:
  1. Load STHELAR breast_s0 from zarr (raw counts from layers["count"])
  2. Load GT labels from table_combined (final_label_combined)
  3. Subsample 50K cells (stratified by GT label) for popV
  4. Download/load Tabula Sapiens Mammary reference (30,936 cells, 17 cell types)
  5. Map gene symbols to Ensembl IDs (reference uses Ensembl, STHELAR uses symbols)
  6. Run Process_Query (retrain mode) + annotate_data (all 8 methods)
  7. KNN transfer labels from 50K subsample back to full 577K cells
  8. Map popV Cell Ontology predictions to STHELAR 10 categories
  9. Evaluate against GT (final_label_combined)

Methods: CELLTYPIST, KNN_BBKNN, KNN_HARMONY, KNN_SCVI, ONCLASS, SCANVI_POPV,
         Support_Vector, XGboost

Outputs:
  - pipeline_output/sthelar_pipeline/popv_full_retrain_breast_s0.json
"""

import gc
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pretrained models.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path("/mnt/work/git/dapidl")
STHELAR_BASE = Path(
    "/mnt/work/datasets/STHELAR/sdata_slides/sdata_breast_s0.zarr/"
    "sdata_breast_s0.zarr/tables"
)
OUT_DIR = PROJECT / "pipeline_output" / "sthelar_pipeline"
OUT_JSON = OUT_DIR / "popv_full_retrain_breast_s0.json"
CACHE_DIR = OUT_DIR / "popv_cache"

MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--popV--tabula_sapiens_Mammary"
SNAPSHOT_DIR = MODEL_CACHE / "snapshots" / "b6dbe818af83fc4259ed8fd26361f969abc0adb0"
ONTOLOGY_DIR = (
    Path.home()
    / ".cache/huggingface/hub/datasets--popV--ontology"
    / "snapshots/2da43b6e227e76e67b4f32f028886e6308f56246"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# popV Cell Ontology -> STHELAR coarse label mapping
# ---------------------------------------------------------------------------
# STHELAR categories: Epithelial, Blood_vessel, Fibroblast_Myofibroblast,
#   Myeloid, B_Plasma, T_NK, Melanocyte, Glioblastoma, Specialized, Other
#
# Tabula Sapiens Mammary labels (17 types from metadata):
#   B cell, CD4-positive alpha-beta T cell, CD8-positive alpha-beta T cell,
#   T cell, basal cell, basophil, endothelial cell, fibroblast of breast,
#   luminal epithelial cell of mammary gland, macrophage, mast cell,
#   mature NK T cell, natural killer cell, plasma cell,
#   progenitor cell of mammary luminal epithelium, regulatory T cell,
#   vascular associated smooth muscle cell, unassigned

POPV_TO_STHELAR = {
    # Epithelial
    "luminal epithelial cell of mammary gland": "Epithelial",
    "basal cell": "Epithelial",
    "progenitor cell of mammary luminal epithelium": "Epithelial",
    "epithelial cell": "Epithelial",
    "mammary gland epithelial cell": "Epithelial",
    "luminal cell": "Epithelial",
    "keratinocyte": "Epithelial",
    "stem cell of epidermis": "Epithelial",
    "epidermal cell": "Epithelial",
    # Blood vessel (endothelial + smooth muscle/pericyte)
    "endothelial cell": "Blood_vessel",
    "vascular endothelial cell": "Blood_vessel",
    "blood vessel endothelial cell": "Blood_vessel",
    "capillary endothelial cell": "Blood_vessel",
    "endothelial cell of artery": "Blood_vessel",
    "endothelial cell of venule": "Blood_vessel",
    "vein endothelial cell": "Blood_vessel",
    "lymphatic endothelial cell": "Blood_vessel",
    "vascular associated smooth muscle cell": "Blood_vessel",
    "pericyte": "Blood_vessel",
    "pericyte cell": "Blood_vessel",
    "smooth muscle cell": "Blood_vessel",
    # Fibroblast / Myofibroblast
    "fibroblast of breast": "Fibroblast_Myofibroblast",
    "fibroblast": "Fibroblast_Myofibroblast",
    "myofibroblast cell": "Fibroblast_Myofibroblast",
    "stromal cell": "Fibroblast_Myofibroblast",
    # Myeloid (macrophage, monocyte, dendritic, mast, basophil, neutrophil)
    "macrophage": "Myeloid",
    "monocyte": "Myeloid",
    "classical monocyte": "Myeloid",
    "non-classical monocyte": "Myeloid",
    "dendritic cell": "Myeloid",
    "Langerhans cell": "Myeloid",
    "plasmacytoid dendritic cell": "Myeloid",
    "myeloid dendritic cell": "Myeloid",
    "mast cell": "Myeloid",
    "basophil": "Myeloid",
    "neutrophil": "Myeloid",
    "granulocyte": "Myeloid",
    # B / Plasma
    "B cell": "B_Plasma",
    "plasma cell": "B_Plasma",
    # T / NK
    "T cell": "T_NK",
    "CD4-positive, alpha-beta T cell": "T_NK",
    "CD8-positive, alpha-beta T cell": "T_NK",
    "regulatory T cell": "T_NK",
    "gamma-delta T cell": "T_NK",
    "natural killer cell": "T_NK",
    "mature NK T cell": "T_NK",
    "innate lymphoid cell": "T_NK",
    "activated CD4-positive, alpha-beta T cell": "T_NK",
    "nk cell": "T_NK",
    # Melanocyte
    "melanocyte": "Melanocyte",
    # Specialized (rare cell types)
    "Schwann cell": "Specialized",
    "schwann cell": "Specialized",
    "glial cell": "Specialized",
    "adipocyte": "Specialized",
    "sebum secreting cell": "Specialized",
    # Other / catch-all
    "unknown": "Other",
    "unassigned": "Other",
}


def map_to_sthelar(label: str) -> str:
    """Map a popV Cell Ontology label to STHELAR coarse category."""
    if pd.isna(label) or label in ("unknown", "unassigned"):
        return "Other"
    label_lower = label.lower().strip()
    # Direct match (case-insensitive)
    for key, val in POPV_TO_STHELAR.items():
        if label_lower == key.lower():
            return val
    # Substring match fallback
    if "endothel" in label_lower:
        return "Blood_vessel"
    if "pericyte" in label_lower or "smooth muscle" in label_lower:
        return "Blood_vessel"
    if "epithelial" in label_lower or "luminal" in label_lower or "basal" in label_lower:
        return "Epithelial"
    if (
        "fibroblast" in label_lower
        or "myofibroblast" in label_lower
        or "stromal" in label_lower
    ):
        return "Fibroblast_Myofibroblast"
    if any(
        x in label_lower
        for x in [
            "macrophage",
            "monocyte",
            "dendritic",
            "langerhans",
            "mast",
            "neutrophil",
            "basophil",
            "granulocyte",
            "myeloid",
        ]
    ):
        return "Myeloid"
    if "b cell" in label_lower or "plasma" in label_lower:
        return "B_Plasma"
    if any(
        x in label_lower
        for x in [
            "t cell",
            "natural killer",
            "nk cell",
            "lymphoid",
            "lymphocyte",
        ]
    ):
        return "T_NK"
    if "melanocyte" in label_lower:
        return "Melanocyte"
    if any(
        x in label_lower
        for x in [
            "schwann",
            "glial",
            "neural",
            "adipocyte",
            "gland",
        ]
    ):
        return "Specialized"
    return "Other"


def elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{dt / 60:.1f}min"


# ===========================================================================
# STEP 1: Load STHELAR data
# ===========================================================================
def step1_load_data():
    t0 = time.time()
    print("=" * 70)
    print("STEP 1: Load STHELAR breast_s0 data")
    print("=" * 70)

    import anndata as ad

    # Load cell data
    print("  Loading table_cells...")
    adata = ad.read_zarr(str(STHELAR_BASE / "table_cells"))
    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]} genes")

    # Use raw counts
    if "count" in adata.layers:
        print("  Setting X to layers['count'] (raw integer counts)")
        adata.X = adata.layers["count"].copy()
    else:
        raise ValueError("No 'count' layer found!")

    # Verify raw counts
    if issparse(adata.X):
        sample = adata.X.data[:20]
    else:
        sample = adata.X.ravel()[:20]
    print(f"  X sample: {sample}")
    print(f"  X dtype: {adata.X.dtype}")

    # Load GT labels
    print("\n  Loading table_combined for GT labels...")
    gt = ad.read_zarr(str(STHELAR_BASE / "table_combined"))
    gt_labels = gt.obs["final_label_combined"].values
    adata.obs["gt"] = gt_labels
    del gt
    gc.collect()

    print(f"\n  GT label distribution:")
    for lbl, cnt in adata.obs["gt"].value_counts().items():
        print(f"    {lbl}: {cnt:,}")

    print(f"\n  [{elapsed(t0)}]")
    return adata


# ===========================================================================
# STEP 2: Subsample 50K cells
# ===========================================================================
def step2_subsample(adata, target_n=50_000):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"STEP 2: Stratified subsample to {target_n:,} cells")
    print("=" * 70)

    np.random.seed(42)

    # Stratified by GT label
    group_col = adata.obs["gt"].astype(str)
    group_counts = group_col.value_counts()
    group_fracs = group_counts / group_counts.sum()
    group_targets = (group_fracs * target_n).round().astype(int)

    # Adjust to hit exact target
    diff = target_n - group_targets.sum()
    if diff != 0:
        largest_group = group_targets.idxmax()
        group_targets[largest_group] += diff

    print(f"  Subsample targets ({target_n:,} total):")
    for grp, n in group_targets.items():
        print(f"    {grp}: {n:,} / {group_counts[grp]:,}")

    # Sample indices
    sub_indices = []
    for grp, n in group_targets.items():
        mask = group_col == grp
        indices = np.where(mask)[0]
        chosen = np.random.choice(indices, size=min(n, len(indices)), replace=False)
        sub_indices.append(chosen)
    sub_indices = np.sort(np.concatenate(sub_indices))

    adata_sub = adata[sub_indices].copy()
    print(f"\n  Subsample: {adata_sub.shape[0]:,} cells")
    print(f"  [{elapsed(t0)}]")

    return adata_sub, sub_indices


# ===========================================================================
# STEP 3: Run popV (Process_Query + annotate_data)
# ===========================================================================
def step3_run_popv(adata_sub):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STEP 3: Run full popV pipeline (retrain mode)")
    print("=" * 70)

    import popv
    from popv.annotation import annotate_data
    from popv.preprocessing import Process_Query

    print(f"  popV version: {popv.__version__}")

    # --- 3a: Load Tabula Sapiens Mammary reference ---
    print("\n  3a. Loading Tabula Sapiens Mammary reference...")
    t_load = time.time()

    ref_path = SNAPSHOT_DIR / "ref_adata.h5ad"
    mini_path = SNAPSHOT_DIR / "minified_ref_adata.h5ad"

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference adata not found at {ref_path}. "
            "Download via cellxgene_census.download_source_h5ad()."
        )

    ref_adata = sc.read_h5ad(str(ref_path))
    print(f"      Ref shape: {ref_adata.shape}")
    print(f"      Ref var_names format: {ref_adata.var_names[0]}")

    # Load minified for popv_labels and batch_key metadata
    mini = sc.read_h5ad(str(mini_path))
    print(f"      Mini shape: {mini.shape}")

    # Ensure ref_adata has popv_labels and batch_key
    if "popv_labels" not in ref_adata.obs.columns:
        print("      Adding popv_labels and batch_key from minified reference...")
        ref_adata.obs["popv_labels"] = "unknown"
        ref_adata.obs["batch_key"] = "unknown"

        overlap_idx = ref_adata.obs.index.isin(mini.obs.index)
        ref_adata.obs.loc[overlap_idx, "popv_labels"] = (
            mini.obs.loc[
                ref_adata.obs.index[overlap_idx], "popv_labels"
            ]
            .astype(str)
            .values
        )
        ref_adata.obs.loc[overlap_idx, "batch_key"] = (
            mini.obs.loc[ref_adata.obs.index[overlap_idx], "batch_key"]
            .astype(str)
            .values
        )

        non_overlap = ~overlap_idx
        if non_overlap.sum() > 0:
            if "cell_type" in ref_adata.obs.columns:
                ref_adata.obs.loc[non_overlap, "popv_labels"] = (
                    ref_adata.obs.loc[non_overlap, "cell_type"].astype(str).values
                )
            if "donor_id" in ref_adata.obs.columns:
                ref_adata.obs.loc[non_overlap, "batch_key"] = (
                    ref_adata.obs.loc[non_overlap, "donor_id"].astype(str)
                    + "_"
                    + ref_adata.obs.loc[non_overlap, "assay"].astype(str)
                    + "_"
                    + ref_adata.obs.loc[non_overlap, "tissue"].astype(str)
                ).values

        print(
            f"      Labels assigned: {ref_adata.obs['popv_labels'].nunique()} types"
        )
    else:
        print(
            f"      Ref already has popv_labels: "
            f"{ref_adata.obs['popv_labels'].nunique()} types"
        )

    del mini
    gc.collect()

    # Use raw counts from reference
    if "decontXcounts" in ref_adata.layers:
        print("      Setting ref X to decontXcounts (raw counts)")
        ref_adata.X = ref_adata.layers["decontXcounts"].copy()
    elif ref_adata.raw is not None:
        print("      Setting ref X to raw.X")
        ref_adata = ref_adata.raw.to_adata()

    if hasattr(ref_adata, "raw") and ref_adata.raw is not None:
        del ref_adata.raw
    ref_adata.layers = {}

    # Verify counts
    if issparse(ref_adata.X):
        x_max = ref_adata.X.data.max() if ref_adata.X.nnz > 0 else 0
        x_min = ref_adata.X.data.min() if ref_adata.X.nnz > 0 else 0
    else:
        x_max = ref_adata.X.max()
        x_min = ref_adata.X.min()
    print(f"      Ref X range: {x_min:.0f} - {x_max:.0f}")

    print(f"      Ref label distribution:")
    for lbl, cnt in ref_adata.obs["popv_labels"].value_counts().head(20).items():
        print(f"        {lbl}: {cnt:,}")

    print(f"      [{elapsed(t_load)}]")

    # Check feature_name column
    has_feature_name = "feature_name" in ref_adata.var.columns
    print(f"\n      Ref has feature_name column: {has_feature_name}")
    if has_feature_name:
        print(
            f"      Ref feature_name sample: "
            f"{list(ref_adata.var['feature_name'][:5])}"
        )
    print(f"      Ref var_names sample: {list(ref_adata.var_names[:5])}")

    # --- 3b: Prepare query data with gene mapping ---
    print(f"\n  3b. Preparing query data with gene symbol -> Ensembl mapping...")

    adata_query = adata_sub.copy()

    # Clear unnecessary layers to save memory
    keep_layers = {"count"}
    for layer_name in list(adata_query.layers.keys()):
        if layer_name not in keep_layers:
            del adata_query.layers[layer_name]

    # Ensure X is raw counts
    if "count" in adata_query.layers:
        adata_query.X = adata_query.layers["count"].copy()
    del adata_query.layers

    # Map gene symbols to Ensembl IDs
    if has_feature_name:
        symbol_to_ensembl = dict(
            zip(ref_adata.var["feature_name"], ref_adata.var_names)
        )
    else:
        # Try gene_symbol or gene_name columns
        for col in ["gene_symbol", "gene_name", "gene_short_name"]:
            if col in ref_adata.var.columns:
                symbol_to_ensembl = dict(
                    zip(ref_adata.var[col], ref_adata.var_names)
                )
                break
        else:
            raise ValueError("Cannot find gene symbol column in reference!")

    query_genes = adata_query.var_names.tolist()
    mapped = {g: symbol_to_ensembl.get(g) for g in query_genes}
    mappable = {g: e for g, e in mapped.items() if e is not None}
    print(
        f"      Gene mapping: {len(mappable)}/{len(query_genes)} "
        f"symbols -> Ensembl IDs"
    )

    # Show some unmapped genes
    unmapped = [g for g in query_genes if g not in mappable]
    if unmapped:
        print(f"      Unmapped genes (first 10): {unmapped[:10]}")

    # Subset to mapped genes and rename
    mapped_genes = list(mappable.keys())
    adata_query = adata_query[:, mapped_genes].copy()
    adata_query.var["gene_symbol"] = adata_query.var_names.tolist()
    adata_query.var_names = pd.Index([mappable[g] for g in mapped_genes])
    adata_query.var_names_make_unique()
    print(f"      Query shape after mapping: {adata_query.shape}")

    # Clean obs (keep only essential columns)
    keep_cols = {"cell_id", "gt", "region"}
    for col in list(adata_query.obs.columns):
        if col not in keep_cols:
            del adata_query.obs[col]

    # Clear obsm/uns to avoid conflicts
    adata_query.obsm = {}
    adata_query.uns = {}

    # Verify X range
    if issparse(adata_query.X):
        qx_max = adata_query.X.data.max() if adata_query.X.nnz > 0 else 0
    else:
        qx_max = adata_query.X.max()
    print(f"      Query X max: {qx_max:.0f}")

    # --- 3c: Run Process_Query ---
    print(f"\n  3c. Running popV Process_Query (retrain mode)...")
    print(
        f"      This concatenates ref+query, normalizes, "
        f"computes HVGs, PCA, neighbors..."
    )
    t_pq = time.time()

    save_path = str(CACHE_DIR / "trained_models")
    os.makedirs(save_path, exist_ok=True)

    pq = Process_Query(
        query_adata=adata_query,
        ref_adata=ref_adata,
        ref_labels_key="popv_labels",
        ref_batch_key="batch_key",
        cl_obo_folder=str(ONTOLOGY_DIR),
        query_batch_key=None,  # No batch key for single-sample STHELAR
        prediction_mode="retrain",
        unknown_celltype_label="unknown",
        n_samples_per_label=300,
        save_path_trained_models=save_path,
        pretrained_scvi_path=None,  # Train from scratch in retrain mode
        hvg=4000,
    )
    concatenated = pq.adata
    print(f"      Preprocessed shape: {concatenated.shape}")
    print(f"      [{elapsed(t_pq)}]")

    # Free reference memory
    del ref_adata, adata_query
    gc.collect()

    # --- 3d: Run annotation (all 8 methods) ---
    print(f"\n  3d. Running popV annotation (all 8 methods)...")
    print(f"      Methods: CELLTYPIST, KNN_BBKNN, KNN_HARMONY, KNN_SCVI,")
    print(f"               ONCLASS, SCANVI_POPV, Support_Vector, XGboost")
    t_annot = time.time()

    popv_output = str(CACHE_DIR / "popv_output")
    os.makedirs(popv_output, exist_ok=True)

    annotate_data(
        concatenated,
        methods=None,  # All methods
        save_path=popv_output,
    )
    print(f"      Annotation complete [{elapsed(t_annot)}]")

    # --- 3e: Extract query cell results ---
    print(f"\n  3e. Extracting popV results for query cells...")

    if "_dataset" in concatenated.obs.columns:
        is_query = concatenated.obs["_dataset"] == "query"
        result = concatenated[is_query].copy()
        print(f"      Query cells: {result.shape[0]:,}")
    elif "_is_ref" in concatenated.obs.columns:
        is_query = ~concatenated.obs["_is_ref"].astype(bool)
        result = concatenated[is_query].copy()
        print(f"      Query cells: {result.shape[0]:,}")
    else:
        n_query = adata_sub.shape[0]
        result = concatenated[-n_query:].copy()
        print(f"      Query cells (last N): {result.shape[0]:,}")

    # Show popV columns
    popv_cols = [c for c in result.obs.columns if "popv" in c.lower()]
    print(f"      popV columns: {popv_cols}")

    if "prediction_keys" in result.uns:
        print(f"      prediction_keys: {list(result.uns['prediction_keys'])}")

    # Find consensus column
    consensus_col = None
    for candidate in [
        "popv_prediction",
        "popv_majority_vote_prediction",
        "popv_consensus",
        "popv_majority_vote",
    ]:
        if candidate in result.obs.columns:
            consensus_col = candidate
            break
    if consensus_col:
        print(f"\n      Consensus ({consensus_col}) distribution:")
        for lbl, cnt in result.obs[consensus_col].value_counts().head(20).items():
            print(f"        {lbl}: {cnt:,}")

    print(f"\n  Total Step 3 time: {elapsed(t0)}")
    return result


# ===========================================================================
# STEP 4: KNN label transfer to full dataset
# ===========================================================================
def step4_knn_transfer(adata_full, adata_sub, result_sub):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"STEP 4: KNN label transfer to full {adata_full.shape[0]:,} cells")
    print("=" * 70)

    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier

    # Find the consensus prediction column
    consensus_col = None
    for candidate in [
        "popv_prediction",
        "popv_majority_vote_prediction",
        "popv_majority_vote",
        "popv_consensus",
        "consensus_prediction",
    ]:
        if candidate in result_sub.obs.columns:
            consensus_col = candidate
            break

    if consensus_col is None:
        if "prediction_keys" in result_sub.uns:
            pkeys = result_sub.uns["prediction_keys"]
            if isinstance(pkeys, list) and len(pkeys) > 0:
                consensus_col = pkeys[-1]
                print(f"  Using prediction key from uns: {consensus_col}")

    if consensus_col is None:
        for c in result_sub.obs.columns:
            if "popv" in c.lower():
                consensus_col = c
                break

    if consensus_col is None:
        raise ValueError(
            f"Cannot find popV consensus column. "
            f"Available: {list(result_sub.obs.columns)}"
        )

    print(f"  Using consensus column: '{consensus_col}'")
    labels = result_sub.obs[consensus_col].astype(str).values
    print(f"  Unique labels: {len(set(labels))}")
    for lbl, cnt in Counter(labels).most_common(20):
        print(f"    {lbl}: {cnt:,}")

    # Find per-method prediction columns
    prediction_keys = []
    if "prediction_keys" in result_sub.uns:
        prediction_keys = list(result_sub.uns["prediction_keys"])
    else:
        for c in result_sub.obs.columns:
            if c.startswith("popv_") and c.endswith("_prediction"):
                prediction_keys.append(c)

    print(f"  Per-method prediction columns: {prediction_keys}")

    # Compute PCA on the full dataset for KNN transfer
    print(
        f"\n  Computing PCA on full dataset ({adata_full.shape[0]:,} cells)..."
    )
    t_pca = time.time()

    # Use log-normalized data for PCA
    if "log_norm" in adata_full.layers:
        print("  Using log_norm layer for PCA")
        X_for_pca = adata_full.layers["log_norm"]
    else:
        # Compute log-norm from raw counts on a copy
        print("  Computing log-norm from raw counts for PCA")
        adata_tmp = adata_full.copy()
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        X_for_pca = adata_tmp.X
        del adata_tmp

    if issparse(X_for_pca):
        X_for_pca_dense = X_for_pca.toarray()
    else:
        X_for_pca_dense = np.asarray(X_for_pca)

    # PCA with 50 components
    n_components = 50
    print(f"  Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca_full = pca.fit_transform(X_for_pca_dense)
    print(f"  PCA complete: {X_pca_full.shape} [{elapsed(t_pca)}]")

    del X_for_pca_dense
    gc.collect()

    # Get subsample PCA embeddings (using indices)
    sub_obs_names = set(adata_sub.obs_names)
    sub_mask = np.array(
        [name in sub_obs_names for name in adata_full.obs_names]
    )
    X_pca_sub = X_pca_full[sub_mask]

    # Align labels with PCA subset
    valid = None
    if X_pca_sub.shape[0] != len(labels):
        print(
            f"  Size mismatch: PCA sub={X_pca_sub.shape[0]}, "
            f"labels={len(labels)}"
        )
        sub_idx = adata_sub.obs.index.get_indexer(result_sub.obs.index)
        valid = sub_idx >= 0
        sub_full_idx = np.where(sub_mask)[0]
        X_pca_sub = X_pca_full[sub_full_idx[sub_idx[valid]]]
        labels = labels[valid]
        print(f"  After alignment: {X_pca_sub.shape[0]} training cells")

    # Train KNN
    print(
        f"\n  Training KNN (k=15) on {X_pca_sub.shape[0]:,} cells, "
        f"{n_components} dims..."
    )
    t_knn = time.time()

    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)
    knn.fit(X_pca_sub, labels)
    print(f"  KNN trained [{elapsed(t_knn)}]")

    # Predict all cells
    print(f"  Predicting labels for {X_pca_full.shape[0]:,} cells...")
    t_pred = time.time()
    full_labels = knn.predict(X_pca_full)
    full_proba = knn.predict_proba(X_pca_full)
    full_max_proba = full_proba.max(axis=1)
    print(f"  Prediction done [{elapsed(t_pred)}]")

    # Transfer per-method labels too
    method_labels = {}
    for method_col in prediction_keys:
        if method_col == consensus_col:
            continue
        if method_col in result_sub.obs.columns:
            y_method = result_sub.obs[method_col].astype(str).values
            if X_pca_sub.shape[0] != len(y_method):
                y_method = y_method[valid] if valid is not None else y_method
            knn_m = KNeighborsClassifier(
                n_neighbors=15, weights="distance", n_jobs=-1
            )
            knn_m.fit(X_pca_sub, y_method)
            method_labels[method_col] = knn_m.predict(X_pca_full)
            print(f"    Transferred {method_col}")

    # Mark direct vs transferred
    direct_mask = sub_mask.copy()

    # Consensus agreement score
    n_methods = len(prediction_keys)
    if n_methods > 1 and len(method_labels) > 0:
        print(
            f"\n  Computing consensus agreement across {n_methods} methods..."
        )
        agreement_scores = np.zeros(len(full_labels))
        for _method_col, mlabels in method_labels.items():
            agreement_scores += (mlabels == full_labels).astype(float)
        agreement_scores += 1  # consensus agrees with itself
        agreement_scores = agreement_scores / (len(method_labels) + 1)
    else:
        agreement_scores = full_max_proba

    # Free PCA matrix
    del X_pca_full
    gc.collect()

    print(f"\n  Total Step 4 time: {elapsed(t0)}")

    return {
        "labels_fine": full_labels,
        "knn_proba": full_max_proba,
        "direct_mask": direct_mask,
        "method_labels": method_labels,
        "prediction_keys": prediction_keys,
        "consensus_col": consensus_col,
        "agreement_scores": agreement_scores,
        "n_methods": n_methods,
    }


# ===========================================================================
# STEP 5: Evaluate against GT
# ===========================================================================
def step5_evaluate(adata_full, transfer_results):
    t0 = time.time()
    print("\n" + "=" * 70)
    print("STEP 5: Map to STHELAR categories and evaluate against GT")
    print("=" * 70)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    fine_labels = transfer_results["labels_fine"]
    coarse_labels = np.array([map_to_sthelar(l) for l in fine_labels])

    print(f"\n  Fine -> Coarse mapping distribution:")
    for lbl, cnt in Counter(fine_labels).most_common(20):
        coarse = map_to_sthelar(lbl)
        print(f"    {lbl} -> {coarse}: {cnt:,}")

    print(f"\n  STHELAR coarse label distribution (predicted):")
    for lbl, cnt in sorted(Counter(coarse_labels).items(), key=lambda x: -x[1]):
        print(f"    {lbl}: {cnt:,}")

    # GT labels
    gt_labels = adata_full.obs["gt"].astype(str).values

    # Filter out "Less10" GT category (too few cells, not in standard categories)
    valid_mask = gt_labels != "Less10"
    gt_valid = gt_labels[valid_mask]
    pred_valid = coarse_labels[valid_mask]

    print(
        f"\n  Evaluating on {valid_mask.sum():,} cells "
        f"(excluding {(~valid_mask).sum():,} Less10)"
    )

    # GT distribution
    print(f"\n  GT distribution:")
    for lbl, cnt in sorted(Counter(gt_valid).items(), key=lambda x: -x[1]):
        print(f"    {lbl}: {cnt:,}")

    # Get the unique labels present in both GT and predictions
    all_labels = sorted(set(gt_valid) | set(pred_valid))
    print(f"\n  All labels: {all_labels}")

    # Classification report
    report = classification_report(
        gt_valid, pred_valid, output_dict=True, zero_division=0
    )
    print(f"\n  Classification Report:")
    print(classification_report(gt_valid, pred_valid, zero_division=0))

    # Overall metrics
    f1_macro = f1_score(
        gt_valid, pred_valid, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        gt_valid, pred_valid, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(gt_valid, pred_valid)

    print(f"\n  SUMMARY:")
    print(f"    F1 Macro:    {f1_macro:.4f}")
    print(f"    F1 Weighted: {f1_weighted:.4f}")
    print(f"    Accuracy:    {accuracy:.4f}")

    # Build results dict
    per_class = {}
    gt_classes = sorted(set(gt_valid))
    for cls in gt_classes:
        if cls in report:
            per_class[cls] = {
                "p": round(report[cls]["precision"], 3),
                "r": round(report[cls]["recall"], 3),
                "f1": round(report[cls]["f1-score"], 3),
                "n": int(report[cls]["support"]),
            }

    results = {
        "method": "popv_full_retrain",
        "reference": "Tabula_Sapiens_Mammary",
        "prediction_mode": "retrain",
        "n_popv_subsample": 50_000,
        "n_evaluated": int(valid_mask.sum()),
        "n_total": int(len(gt_labels)),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "popv_fine_labels": dict(Counter(fine_labels).most_common()),
        "consensus_col": transfer_results["consensus_col"],
        "n_methods": transfer_results["n_methods"],
    }

    # Save results
    with open(str(OUT_JSON), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {OUT_JSON}")

    print(f"\n  [{elapsed(t0)}]")
    return results


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    t_total = time.time()
    print("=" * 70)
    print("Full popV Retrain Pipeline: STHELAR Breast S0")
    print(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import popv

    print(f"  popV version: {popv.__version__}")
    print(f"  Reference: Tabula Sapiens Mammary")
    print(f"  Output: {OUT_JSON}")
    print("=" * 70)

    # Step 1: Load data
    adata = step1_load_data()

    # Step 2: Subsample
    adata_sub, sub_indices = step2_subsample(adata)

    # Step 3: Run popV
    result_sub = step3_run_popv(adata_sub)

    # Step 4: KNN transfer
    transfer_results = step4_knn_transfer(adata, adata_sub, result_sub)

    # Step 5: Evaluate
    results = step5_evaluate(adata, transfer_results)

    print("\n" + "=" * 70)
    print(f"DONE! Total time: {elapsed(t_total)}")
    print(f"  F1 Macro: {results['f1_macro']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Results: {OUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
