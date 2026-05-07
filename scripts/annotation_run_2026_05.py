"""Re-run 5 annotation methods on 6 breast slides, save RAW + multi-tier predictions.

Methods (one from each family + alternates):
- Gene expression / ML reference:  CellTypist Cells_Adult_Breast (best for breast)
- Marker DB (custom):              scType custom_default
- Marker DB (alternate):           scType cellmarker2
- Reference atlas:                 SingleR Blueprint (most universal coverage)
- Reference atlas (alternate):     SingleR HPCA (immune-rich)

GROUND TRUTH:
- xenium rep1/rep2: Janesick supervised (in adata.obs.gt_coarse)
- STHELAR breast_s*: cells_label2 -> COARSE/MEDIUM via STHELAR_LABEL2_TO_*
  (joined from /mnt/work/datasets/STHELAR/summary_all_labels_per_slide/)

OUTPUTS:
    pipeline_output/annotation_run_2026_05/
        per_slide/{slide}.json          {method: {raw_predictions, conf}}
        coarse_metrics.parquet          per slide × method × COARSE-4 metrics
        medium_metrics.parquet          per slide × method × MEDIUM-12 metrics
        consensus_results.parquet       all 2-method + 3-method consensus combinations
        summary.md
"""
from __future__ import annotations
import gc
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Reuse the existing benchmark's loaders + run_* functions
from annotation_benchmark_2026_03 import (
    load_xenium_adata, load_sthelar_adata, preprocess_adata,
    run_celltypist, run_sctype, run_singler,
    get_default_markers, get_sctype_db_markers, get_cellmarker2_markers,
)
from dapidl.ontology.cl_mapper import get_mapper
from dapidl.ontology.training_tiers import (
    derive_tier_label, COARSE_NAMES, MEDIUM_NAMES,
)
from derive_label2_labels import (
    STHELAR_LABEL2_TO_COARSE,
    STHELAR_LABEL2_TO_MEDIUM,
    JANESICK17_TO_MEDIUM,
)
from dapidl.data.sthelar import SthelarDataReader

OUT_DIR = Path("/mnt/work/git/dapidl/pipeline_output/annotation_run_2026_05")
PER_SLIDE_DIR = OUT_DIR / "per_slide"
PER_SLIDE_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = Path("/mnt/work/datasets/STHELAR/summary_all_labels_per_slide")

# Janesick 17 -> COARSE 4 (in benchmark module: GT_TO_COARSE)
from annotation_benchmark_2026_03 import GT_TO_COARSE
JANESICK17_TO_COARSE = {k: v if v != "Unknown" else None
                        for k, v in GT_TO_COARSE.items()}

SLIDES = ["rep1", "rep2", "breast_s0", "breast_s1", "breast_s3", "breast_s6"]
COARSE_4 = ["Endothelial", "Epithelial", "Immune", "Stromal"]


def get_gt_for_slide(adata, slide_name):
    """Return (gt_coarse, gt_medium) arrays of length len(adata).

    For STHELAR slides: join cells_label2 from parquet by cell_id.
    For Janesick rep1/rep2: use adata.obs.gt_coarse from loader; derive medium
    from adata.obs.gt_fine (Janesick 17-class).
    """
    if slide_name.startswith("rep"):
        gt_coarse = adata.obs["gt_coarse"].astype(str).values
        gt_medium = np.array(
            [JANESICK17_TO_MEDIUM.get(v) or "Unknown"
             for v in adata.obs["gt_fine"].astype(str).values],
            dtype=object)
        return gt_coarse, gt_medium

    # STHELAR: join parquet
    pq_path = PARQUET_DIR / f"{slide_name}_cell_metadata.parquet"
    pq = pl.read_parquet(pq_path)
    # Build lookup cell_id -> cells_label2
    label2_lookup = dict(zip(pq["cell_id"].to_list(),
                             pq["cells_label2"].to_list()))
    if "cell_id" not in adata.obs.columns:
        # adata.obs_names is cell_id for STHELAR
        cids = adata.obs_names.astype(str).tolist()
    else:
        cids = adata.obs["cell_id"].astype(str).tolist()
    l2 = [label2_lookup.get(c) for c in cids]
    gt_coarse = np.array(
        [STHELAR_LABEL2_TO_COARSE.get(v) or "Unknown" for v in l2],
        dtype=object)
    gt_medium = np.array(
        [STHELAR_LABEL2_TO_MEDIUM.get(v) or "Unknown" for v in l2],
        dtype=object)
    return gt_coarse, gt_medium


def map_to_tier(raw_preds, tier, mapper):
    """Map raw cell-type predictions to a canonical tier name (or 'Unknown')."""
    return np.array(
        [derive_tier_label(str(p), tier, mapper) for p in raw_preds],
        dtype=object)


def macro_f1(gt, pred, labels):
    mask = np.isin(gt, labels)
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(gt[mask], pred[mask], labels=labels,
                          average="macro", zero_division=0))


def consensus_majority(preds_list):
    if len(preds_list) == 1:
        return preds_list[0]
    n = len(preds_list[0])
    out = np.empty(n, dtype=object)
    stacked = np.stack(preds_list, axis=0)
    for i in range(n):
        col = stacked[:, i]
        vals, counts = np.unique(col, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def consensus_confidence_weighted(preds_list, conf_list):
    """Per-cell confidence-weighted vote.

    For each cell, sum confidence per class across methods. Pick highest sum.
    `preds_list[i]` and `conf_list[i]` must align (same method).
    """
    if len(preds_list) == 1:
        return preds_list[0]
    n = len(preds_list[0])
    out = np.empty(n, dtype=object)
    for i in range(n):
        scores = {}
        for preds, conf in zip(preds_list, conf_list):
            label = preds[i]
            w = float(conf[i]) if conf is not None and i < len(conf) else 1.0
            scores[label] = scores.get(label, 0.0) + w
        out[i] = max(scores, key=scores.get)
    return out


def per_class_f1(gt, pred, labels):
    mask = np.isin(gt, labels)
    if mask.sum() == 0:
        return {c: 0.0 for c in labels}
    _, _, f1, _ = precision_recall_fscore_support(
        gt[mask], pred[mask], labels=labels, zero_division=0)
    return dict(zip(labels, [float(x) for x in f1]))


def run_celltypist_no_mv(adata, model_name):
    """CellTypist without majority_voting (skip slow Leiden over-clustering)."""
    import celltypist
    from celltypist import models as ct_models
    try:
        model = ct_models.Model.load(model_name)
    except Exception:
        ct_models.download_models(model=model_name, force_update=False)
        model = ct_models.Model.load(model_name)
    predictions = celltypist.annotate(adata, model=model, majority_voting=False)
    result = predictions.to_adata()
    preds = result.obs["predicted_labels"].astype(str).values
    conf = (result.obs["conf_score"].values
            if "conf_score" in result.obs.columns
            else np.ones(len(result)))
    return {"predictions": preds, "confidence": conf,
            "method": f"celltypist_noMV_{model_name.replace('.pkl', '')}"}


def run_banksy_sctype(adata_raw, markers, k_nbr=15, lam=0.5, resolution=1.0):
    """BANKSY spatial clustering + scType labeling per cluster.

    Uses x_centroid/y_centroid from adata.obs. Returns per-cell predictions
    by labeling each BANKSY cluster with the scType-best cell type using
    cluster-mean expression.
    """
    import scanpy as sc
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
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

    # Skip banksy_py's pca_umap helper — its internal dict layout varies across
    # versions and the result was unreachable on banksy-py 0.2.x (pca_dict
    # keys = []). Run PCA directly on the BANKSY matrix; the output is an
    # ndarray we can store in obsm immediately.
    nbr_key = list(banksy_dict.keys())[0]
    bm = banksy_matrix
    if isinstance(bm, dict):
        # generate_banksy_matrix may return {lambda: matrix} on some versions
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

    # Label each cluster via scType on cluster-mean expression
    # Build per-cluster mean expression using ORIGINAL gene set (not just HVG)
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


def run_methods_on_slide(adata_pp, adata_raw, methods_to_run):
    """Run each method, return {method_name: {raw_preds, conf}}."""
    out = {}
    n_cells = len(adata_pp)
    n_genes = adata_pp.n_vars
    # SingleR loads dense matrix to R — OOM danger if cells*genes > 2.5e9.
    # Xenium Prime breast_s6: 530k * 8232 = 4.4e9 → killed at ~58GB RSS.
    skip_singler = (n_cells * n_genes) > 2_500_000_000
    if skip_singler:
        logger.warning(f"  Skipping SingleR — slide too large "
                       f"({n_cells:,} cells × {n_genes} genes = {n_cells*n_genes/1e9:.1f}e9)")
    for kind, args in methods_to_run:
        try:
            if kind == "celltypist":
                model = args
                logger.info(f"  CellTypist no-MV {model}")
                r = run_celltypist_no_mv(adata_pp, model)
            elif kind == "sctype":
                markers, marker_name = args
                logger.info(f"  scType {marker_name}")
                r = run_sctype(adata_pp, markers, marker_name)
            elif kind == "singler":
                if skip_singler:
                    logger.info(f"  Skipping singler/{args} (too large)")
                    continue
                ref = args
                logger.info(f"  SingleR {ref}")
                r = run_singler(adata_pp, ref)
            elif kind == "banksy_sctype":
                markers, marker_name = args
                logger.info(f"  BANKSY+scType {marker_name}")
                r = run_banksy_sctype(adata_raw, markers)
            else:
                continue
            if r.get("predictions") is None:
                logger.warning(f"  {kind}/{args}: {r.get('error', 'no predictions')}")
                continue
            out[r["method"]] = {
                "raw_preds": r["predictions"],
                "conf": r.get("confidence"),
            }
        except Exception as e:
            logger.error(f"  {kind}/{args}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    return out


def main():
    mapper = get_mapper()

    # Build sctype marker dicts ONCE
    sctype_markers = {}
    for name, fn in [
        ("custom_default", get_default_markers),
        ("cellmarker2", get_cellmarker2_markers),
        ("sctype_db_immune", get_sctype_db_markers),
    ]:
        try:
            m = fn()
            if m:
                sctype_markers[name] = m
        except Exception as e:
            logger.warning(f"sctype {name} markers unavailable: {e}")

    # 3 stable methods (BANKSY died silently in initialize_banksy — see /tmp/dapidl_logs/annotation_full.log
    # at 00:52 — pulled out for now, will run separately with subprocess isolation)
    methods_to_run = []
    methods_to_run.append(("celltypist", "Cells_Adult_Breast.pkl"))
    if "custom_default" in sctype_markers:
        methods_to_run.append(("sctype",
                               (sctype_markers["custom_default"], "custom_default")))
    methods_to_run.append(("singler", "blueprint"))

    coarse_rows = []
    medium_rows = []
    consensus_rows = []

    for slide in SLIDES:
        out_path = PER_SLIDE_DIR / f"{slide}.json"
        if out_path.exists():
            logger.info(f"=== {slide}: already done, loading ===")
            with open(out_path) as f:
                slide_results = json.load(f)
            # Convert lists back to arrays (handle old caches without conf)
            for m in slide_results:
                slide_results[m]["raw_preds"] = np.array(
                    slide_results[m]["raw_preds"], dtype=object)
                if slide_results[m].get("conf") is not None:
                    slide_results[m]["conf"] = np.array(
                        slide_results[m]["conf"], dtype=np.float32)
            adata_pp = None
            gt_coarse_path = PER_SLIDE_DIR / f"{slide}_gt.json"
            if gt_coarse_path.exists():
                with open(gt_coarse_path) as f:
                    gt_data = json.load(f)
                gt_coarse = np.array(gt_data["coarse"], dtype=object)
                gt_medium = np.array(gt_data["medium"], dtype=object)
            else:
                logger.warning(f"  {slide}: cached preds but no GT — re-loading adata")
                if slide.startswith("rep"):
                    adata_raw = load_xenium_adata(slide)
                else:
                    adata_raw = load_sthelar_adata(slide)
                adata_pp = preprocess_adata(adata_raw)
                gt_coarse, gt_medium = get_gt_for_slide(adata_pp, slide)
                del adata_raw
                gc.collect()
        else:
            logger.info(f"=== {slide}: loading + running methods ===")
            try:
                if slide.startswith("rep"):
                    adata_raw = load_xenium_adata(slide)
                else:
                    adata_raw = load_sthelar_adata(slide)
                # Subsample if too big (memory protection — Xenium Prime s6 is
                # 530k × 8232 = 4.4 GB raw, 17 GB dense, 34 GB after .copy)
                if (len(adata_raw) * adata_raw.n_vars) > 1_000_000_000:
                    n_target = 100_000
                    logger.warning(f"  {slide}: subsampling {len(adata_raw):,} → {n_target:,} cells (memory protection)")
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(adata_raw), size=n_target, replace=False)
                    adata_raw = adata_raw[idx].copy()
                adata_pp = preprocess_adata(adata_raw)
                logger.info(f"  {slide}: {len(adata_pp):,} cells × {adata_pp.n_vars} genes")
                gt_coarse, gt_medium = get_gt_for_slide(adata_pp, slide)
                # save GT for restart
                with open(PER_SLIDE_DIR / f"{slide}_gt.json", "w") as f:
                    json.dump({"coarse": gt_coarse.tolist(),
                               "medium": gt_medium.tolist()}, f)
                slide_results = run_methods_on_slide(adata_pp, adata_raw, methods_to_run)
                # Save raw preds + confidence (json-friendly)
                save = {m: {"raw_preds": v["raw_preds"].tolist(),
                            "conf": (v["conf"].tolist()
                                     if v.get("conf") is not None
                                     else None)}
                        for m, v in slide_results.items()}
                with open(out_path, "w") as f:
                    json.dump(save, f)
                del adata_raw, adata_pp
                gc.collect()
            except Exception as e:
                logger.error(f"  {slide}: failed to load/run: {e}")
                continue

        # Compute COARSE + MEDIUM metrics per method
        for method_name, d in slide_results.items():
            preds_coarse = map_to_tier(d["raw_preds"], "coarse", mapper)
            preds_medium = map_to_tier(d["raw_preds"], "medium", mapper)
            row_c = {"slide": slide, "method": method_name,
                     "f1_macro": macro_f1(gt_coarse, preds_coarse, COARSE_4),
                     "n_eval": int(np.isin(gt_coarse, COARSE_4).sum())}
            row_c.update({f"f1_{c}": v for c, v in
                          per_class_f1(gt_coarse, preds_coarse, COARSE_4).items()})
            coarse_rows.append(row_c)
            row_m = {"slide": slide, "method": method_name,
                     "f1_macro": macro_f1(gt_medium, preds_medium, MEDIUM_NAMES),
                     "n_eval": int(np.isin(gt_medium, MEDIUM_NAMES).sum())}
            medium_rows.append(row_m)

        # Consensus combinations (2-method + 3-method) — confidence-weighted
        method_list = sorted(slide_results.keys())
        for k in [2, 3]:
            for combo in combinations(method_list, k):
                preds_raw_list = [map_to_tier(slide_results[m]["raw_preds"],
                                              "coarse", mapper) for m in combo]
                conf_list = [slide_results[m].get("conf") for m in combo]
                cons_c = consensus_confidence_weighted(preds_raw_list, conf_list)
                preds_med_list = [map_to_tier(slide_results[m]["raw_preds"],
                                              "medium", mapper) for m in combo]
                cons_m = consensus_confidence_weighted(preds_med_list, conf_list)
                consensus_rows.append({
                    "slide": slide, "n_methods": k, "combo": "+".join(combo),
                    "tier": "coarse",
                    "f1_macro": macro_f1(gt_coarse, cons_c, COARSE_4),
                })
                consensus_rows.append({
                    "slide": slide, "n_methods": k, "combo": "+".join(combo),
                    "tier": "medium",
                    "f1_macro": macro_f1(gt_medium, cons_m, MEDIUM_NAMES),
                })

    # Save consolidated tables
    pl.DataFrame(coarse_rows).write_parquet(OUT_DIR / "coarse_metrics.parquet")
    pl.DataFrame(coarse_rows).write_csv(OUT_DIR / "coarse_metrics.csv")
    pl.DataFrame(medium_rows).write_parquet(OUT_DIR / "medium_metrics.parquet")
    pl.DataFrame(medium_rows).write_csv(OUT_DIR / "medium_metrics.csv")
    if consensus_rows:
        pl.DataFrame(consensus_rows).write_parquet(OUT_DIR / "consensus_results.parquet")
        pl.DataFrame(consensus_rows).write_csv(OUT_DIR / "consensus_results.csv")

    # Markdown summary
    md = ["# Annotation methods × tiers (cells_label2 GT for STHELAR) — DAPIDL deck\n"]
    md.append(f"Slides: {SLIDES}\n")
    md.append("\n## COARSE-4 per-method per-slide\n")
    md.append(pl.DataFrame(coarse_rows).to_pandas().to_markdown(index=False))
    md.append("\n\n## MEDIUM-12 per-method per-slide\n")
    md.append(pl.DataFrame(medium_rows).to_pandas().to_markdown(index=False))
    md.append("\n\n## Top consensus combinations (mean F1 over slides where present)\n")
    if consensus_rows:
        agg = (pl.DataFrame(consensus_rows)
               .group_by(["combo","n_methods","tier"])
               .agg(pl.col("f1_macro").mean().alias("mean_f1"),
                    pl.col("f1_macro").std().alias("std_f1"))
               .sort(["tier","mean_f1"], descending=[False, True])
               .head(20))
        md.append(agg.to_pandas().to_markdown(index=False))
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    logger.info(f"wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
