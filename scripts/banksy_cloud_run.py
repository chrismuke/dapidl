"""Standalone copy of run_banksy_sctype for the cloud box.

Verbatim from annotation_run_2026_05.py (run_banksy_sctype), extracted so the
cloud worker doesn't import annotation_run_2026_05's heavy module-level chain
(derive_label2_labels -> breast_dapi_lmdb -> torch/lmdb). The algorithm is
byte-identical to the version that produced the other 5 slides' results.
"""
from __future__ import annotations

import numpy as np


def run_banksy_sctype(adata_raw, markers, k_nbr=15, lam=0.5, resolution=1.0):
    """BANKSY spatial clustering + scType labeling per cluster.

    Uses x_centroid/y_centroid from adata.obs. Returns per-cell predictions
    by labeling each BANKSY cluster with the scType-best cell type using
    cluster-mean expression.
    """
    import scanpy as sc
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.initialize_banksy import initialize_banksy
    from scipy.sparse import issparse
    from sklearn.decomposition import PCA

    a = adata_raw.copy()
    if issparse(a.X):
        a.X = a.X.toarray()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    n_hvg = min(2000, a.n_vars)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, subset=False)
    a_hvg = a[:, a.var["highly_variable"]].copy()

    if "x_centroid" not in a.obs.columns or "y_centroid" not in a.obs.columns:
        return {"predictions": None,
                "error": "no spatial coords (x_centroid/y_centroid)"}

    a_hvg.obs["xcoord"] = a.obs["x_centroid"].values
    a_hvg.obs["ycoord"] = a.obs["y_centroid"].values
    a_hvg.obsm["xy_coord"] = np.column_stack([
        a_hvg.obs["xcoord"].values.astype(float),
        a_hvg.obs["ycoord"].values.astype(float),
    ])

    banksy_dict = initialize_banksy(
        adata=a_hvg, coord_keys=("xcoord", "ycoord", "xy_coord"),
        num_neighbours=k_nbr, nbr_weight_decay="scaled_gaussian",
        max_m=1, plt_edge_hist=False, plt_nbr_weights=False, plt_theta=False,
    )
    banksy_dict, banksy_matrix = generate_banksy_matrix(
        adata=a_hvg, banksy_dict=banksy_dict, lambda_list=[lam],
        max_m=1, plot_std=False, verbose=False,
    )

    nbr_key = list(banksy_dict.keys())[0]  # noqa: F841
    bm = banksy_matrix
    if isinstance(bm, dict):
        bm = bm.get(lam) or bm.get(str(lam)) or next(iter(bm.values()))
    if hasattr(bm, "X"):  # AnnData-like
        bm = bm.X
    if issparse(bm):
        bm = bm.toarray()
    if bm.shape[0] != a_hvg.n_obs:
        return {"predictions": None,
                "error": f"BANKSY matrix shape {bm.shape} doesn't match "
                         f"n_obs {a_hvg.n_obs}"}
    n_pcs = min(20, bm.shape[1] - 1, bm.shape[0] - 1)
    pca_arr = PCA(n_components=n_pcs, random_state=42).fit_transform(bm)

    a_hvg.obsm["X_banksy_pca"] = pca_arr
    sc.pp.neighbors(a_hvg, use_rep="X_banksy_pca", n_neighbors=15)
    sc.tl.leiden(a_hvg, resolution=resolution, key_added="banksy_leiden")
    clusters = a_hvg.obs["banksy_leiden"].astype(str).values

    gene_means = {}
    for c in np.unique(clusters):
        mask = clusters == c
        gene_means[c] = a.X[mask].mean(axis=0)
    gene_names = list(a.var_names)
    cluster_labels = {}
    for c, mean_expr in gene_means.items():
        best_score = -np.inf
        best_ct = "Unknown"
        for ct, m_dict in markers.items():
            pos = [g for g in m_dict.get("positive", []) if g in gene_names]
            if not pos:
                continue
            idx = [gene_names.index(g) for g in pos]
            score = float(np.mean(mean_expr[idx]))
            if score > best_score:
                best_score = score
                best_ct = ct
        cluster_labels[c] = best_ct

    preds = np.array([cluster_labels.get(c, "Unknown") for c in clusters],
                     dtype=object)
    return {"predictions": preds, "confidence": np.ones(len(preds)),
            "method": f"banksy_sctype_l{lam}_r{resolution}"}
