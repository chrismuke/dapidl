#!/usr/bin/env python3
"""Analyze complementarity between annotation methods using stored predictions.

This script re-annotates a sample of cells and compares predictions at the cell level
to understand where methods agree/disagree and whether they can complement each other.
"""

from pathlib import Path

import celltypist
import numpy as np
import polars as pl
import scanpy as sc
from celltypist import models
from loguru import logger


def map_to_coarse(label: str) -> str:
    """Map fine-grained labels to coarse categories."""
    label_lower = label.lower()

    # Unknown patterns
    if "unknown" in label_lower or label == "Unassigned":
        return "Unknown"

    # Epithelial patterns
    epithelial_patterns = [
        "epithelial", "luminal", "basal", "keratinocyte", "mammary",
        "lumsec", "lumhr", "lumsec-basal", "secretory"
    ]
    if any(p in label_lower for p in epithelial_patterns):
        return "Epithelial"

    # Immune patterns
    immune_patterns = [
        "t cell", "b cell", "macrophage", "monocyte", "dendritic", "dc",
        "nk", "plasma", "mast", "neutrophil", "eosinophil", "basophil",
        "lymphocyte", "immune", "cd4", "cd8", "treg", "th1", "th2", "th17",
        "naive", "memory", "effector", "cytotoxic", "helper", "gamma-delta",
        "mait", "nkt", "innate", "adaptive", "lymph"
    ]
    if any(p in label_lower for p in immune_patterns):
        return "Immune"

    # Stromal patterns
    stromal_patterns = [
        "fibroblast", "stromal", "mesenchymal", "fibro", "myofibro",
        "smooth muscle", "adipocyte", "pericyte", "caf"
    ]
    if any(p in label_lower for p in stromal_patterns):
        return "Stromal"

    # Endothelial patterns
    endothelial_patterns = ["endothel", "vascular", "vessel", "lymphatic"]
    if any(p in label_lower for p in endothelial_patterns):
        return "Endothelial"

    return "Unknown"


def load_xenium_data(xenium_path: Path, max_cells: int = 30000) -> sc.AnnData:
    """Load Xenium data into AnnData format."""
    logger.info(f"Loading Xenium data from {xenium_path}")

    import h5py
    import scipy.sparse as sp

    h5_path = xenium_path / "cell_feature_matrix.h5"
    with h5py.File(h5_path, "r") as f:
        matrix = f["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = matrix["shape"][:]
        features = [x.decode() for x in matrix["features"]["name"][:]]

    X = sp.csr_matrix((data, indices, indptr), shape=(shape[1], shape[0]))

    # Subsample if needed
    if X.shape[0] > max_cells:
        idx = np.random.choice(X.shape[0], max_cells, replace=False)
        X = X[idx]
        cell_ids = [f"cell_{i}" for i in idx]
    else:
        cell_ids = [f"cell_{i}" for i in range(X.shape[0])]

    # Create AnnData
    adata = sc.AnnData(X=X.toarray().astype("float32"))
    adata.obs_names = cell_ids
    adata.var_names = features

    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def run_singler(adata: sc.AnnData, reference: str = "hpca") -> tuple[list[str], list[str]]:
    """Run SingleR annotation via rpy2."""
    from dapidl.pipeline.components.annotators.singler import (
        SINGLER_REFERENCES,
        _fix_libstdcxx,
    )

    _fix_libstdcxx()
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    logger.info(f"Running SingleR with {reference} reference...")

    singler = importr("SingleR")
    celldex = importr("celldex")

    # Get reference data
    ref_func_name = SINGLER_REFERENCES[reference]
    ref_func = getattr(celldex, ref_func_name)
    ref_data = ref_func()

    # Prepare expression matrix
    expr = adata.X.copy()
    genes = list(adata.var_names)

    # Normalize for SingleR
    lib_sizes = expr.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1
    expr_norm = np.log1p(expr / lib_sizes * 10000)

    # Get reference gene names and find intersection
    ref_genes = list(ro.r.rownames(ref_data))
    common_genes = list(set(genes) & set(ref_genes))
    logger.info(f"Common genes: {len(common_genes)} / {len(genes)}")

    # Subset expression to common genes
    gene_idx = [genes.index(g) for g in common_genes]
    expr_subset = expr_norm[:, gene_idx]

    # Convert to R matrix (genes as rows, cells as columns)
    cell_names = [str(cid) for cid in adata.obs_names]
    dimnames = ro.r.list(ro.StrVector(common_genes), ro.StrVector(cell_names))
    expr_r = ro.r.matrix(
        ro.FloatVector(expr_subset.T.flatten()),
        nrow=len(common_genes),
        ncol=adata.n_obs,
        dimnames=dimnames,
    )

    # Subset reference
    ref_subset = ro.r["["](ref_data, ro.StrVector(common_genes), True)

    # Get labels from reference
    labels = ro.r("function(x) x$label.main")(ref_data)

    # Run SingleR
    logger.info("Running SingleR prediction...")
    results = singler.SingleR(
        test=expr_r,
        ref=ref_subset,
        labels=labels,
        de_method="classic",
    )

    # Extract predictions
    pred_labels = list(ro.r("function(x) x$labels")(results))

    # Map to broad categories
    broad_categories = [map_to_coarse(label) for label in pred_labels]

    return pred_labels, broad_categories


def run_all_methods(adata: sc.AnnData) -> pl.DataFrame:
    """Run all annotation methods and collect predictions."""
    results = {"cell_id": list(adata.obs_names)}

    # CellTypist Immune_All_High
    logger.info("Running CellTypist Immune_All_High...")
    model = models.Model.load(model="Immune_All_High.pkl")
    predictions = celltypist.annotate(adata, model=model, majority_voting=False)
    results["ct_immune"] = predictions.predicted_labels["predicted_labels"].tolist()
    results["ct_immune_coarse"] = [map_to_coarse(l) for l in results["ct_immune"]]

    # CellTypist Cells_Adult_Breast
    logger.info("Running CellTypist Cells_Adult_Breast...")
    model = models.Model.load(model="Cells_Adult_Breast.pkl")
    predictions = celltypist.annotate(adata, model=model, majority_voting=False)
    results["ct_breast"] = predictions.predicted_labels["predicted_labels"].tolist()
    results["ct_breast_coarse"] = [map_to_coarse(l) for l in results["ct_breast"]]

    # SingleR HPCA
    results["sr_hpca"], results["sr_hpca_coarse"] = run_singler(adata, "hpca")

    # SingleR Blueprint
    results["sr_blueprint"], results["sr_blueprint_coarse"] = run_singler(adata, "blueprint")

    return pl.DataFrame(results)


def analyze_agreement(df: pl.DataFrame):
    """Analyze agreement patterns between methods."""

    print("\n" + "=" * 80)
    print("CELL-LEVEL AGREEMENT ANALYSIS")
    print("=" * 80)

    n_cells = df.height

    # Coarse-level agreement
    methods = ["ct_immune_coarse", "ct_breast_coarse", "sr_hpca_coarse", "sr_blueprint_coarse"]
    method_names = ["CT_Immune", "CT_Breast", "SR_HPCA", "SR_Blueprint"]

    print(f"\n## Coarse Category Agreement (n={n_cells:,} cells)")
    print("-" * 60)

    # Pairwise agreement
    print("\n### Pairwise Agreement Matrix")
    print(f"{'Method':<15}", end="")
    for name in method_names:
        print(f"{name:<12}", end="")
    print()

    for i, m1 in enumerate(methods):
        print(f"{method_names[i]:<15}", end="")
        for m2 in methods:
            agreement = (df[m1] == df[m2]).sum() / n_cells
            print(f"{agreement:.1%}       ", end="")
        print()

    # All methods agree
    all_agree = (
        (df["ct_immune_coarse"] == df["ct_breast_coarse"]) &
        (df["ct_immune_coarse"] == df["sr_hpca_coarse"]) &
        (df["ct_immune_coarse"] == df["sr_blueprint_coarse"])
    ).sum()
    print(f"\n### All 4 methods agree: {all_agree:,} cells ({all_agree/n_cells:.1%})")

    # Majority vote (at least 3 agree)
    def get_majority(row):
        votes = [row["ct_immune_coarse"], row["ct_breast_coarse"],
                 row["sr_hpca_coarse"], row["sr_blueprint_coarse"]]
        from collections import Counter
        # Filter out Unknown
        valid_votes = [v for v in votes if v != "Unknown"]
        if not valid_votes:
            return "Unknown", 0
        counts = Counter(valid_votes)
        most_common = counts.most_common(1)[0]
        return most_common[0], most_common[1]

    majority_labels = []
    majority_counts = []
    for row in df.iter_rows(named=True):
        label, count = get_majority(row)
        majority_labels.append(label)
        majority_counts.append(count)

    df = df.with_columns([
        pl.Series("majority_label", majority_labels),
        pl.Series("majority_count", majority_counts),
    ])

    # At least 3 agree
    at_least_3 = (df["majority_count"] >= 3).sum()
    print(f"### At least 3 methods agree: {at_least_3:,} cells ({at_least_3/n_cells:.1%})")

    # At least 2 agree
    at_least_2 = (df["majority_count"] >= 2).sum()
    print(f"### At least 2 methods agree: {at_least_2:,} cells ({at_least_2/n_cells:.1%})")

    # Cross-family agreement
    ct_agree = (df["ct_immune_coarse"] == df["ct_breast_coarse"]).sum()
    print(f"\n### Both CellTypist methods agree: {ct_agree:,} cells ({ct_agree/n_cells:.1%})")

    sr_agree = (df["sr_hpca_coarse"] == df["sr_blueprint_coarse"]).sum()
    print(f"### Both SingleR methods agree: {sr_agree:,} cells ({sr_agree/n_cells:.1%})")

    # Disagreement analysis
    print("\n\n## Disagreement Patterns")
    print("-" * 60)

    # Where CT_Immune and SingleR HPCA disagree
    disagree_mask = df["ct_immune_coarse"] != df["sr_hpca_coarse"]
    n_disagree = disagree_mask.sum()
    print(f"\n### CT_Immune vs SR_HPCA disagree: {n_disagree:,} cells ({n_disagree/n_cells:.1%})")

    if n_disagree > 0:
        disagree_df = df.filter(disagree_mask)
        crosstab = (
            disagree_df
            .group_by(["ct_immune_coarse", "sr_hpca_coarse"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(10)
        )
        print("\nTop disagreement patterns (CT_Immune vs SR_HPCA):")
        for row in crosstab.iter_rows(named=True):
            pct = row["count"] / n_disagree * 100
            print(f"  {row['ct_immune_coarse']:<15} vs {row['sr_hpca_coarse']:<15}: {row['count']:>6} ({pct:.1f}%)")

    # CellTypist agrees but SingleR disagrees
    ct_agree_sr_disagree = (
        (df["ct_immune_coarse"] == df["ct_breast_coarse"]) &
        (df["ct_immune_coarse"] != df["sr_hpca_coarse"]) &
        (df["ct_immune_coarse"] != df["sr_blueprint_coarse"])
    )
    n_ct_only = ct_agree_sr_disagree.sum()
    print(f"\n### CellTypist agrees but both SingleR disagree: {n_ct_only:,} cells ({n_ct_only/n_cells:.1%})")

    if n_ct_only > 100:
        ct_only_df = df.filter(ct_agree_sr_disagree)
        print("\nWhat CellTypist says vs what SingleR says:")
        crosstab = (
            ct_only_df
            .group_by(["ct_immune_coarse", "sr_hpca_coarse"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(8)
        )
        for row in crosstab.iter_rows(named=True):
            print(f"  CT={row['ct_immune_coarse']:<12}, SR_HPCA={row['sr_hpca_coarse']:<12}: {row['count']:,}")

    # SingleR agrees but CellTypist disagrees
    sr_agree_ct_disagree = (
        (df["sr_hpca_coarse"] == df["sr_blueprint_coarse"]) &
        (df["sr_hpca_coarse"] != df["ct_immune_coarse"]) &
        (df["sr_hpca_coarse"] != df["ct_breast_coarse"])
    )
    n_sr_only = sr_agree_ct_disagree.sum()
    print(f"\n### SingleR agrees but both CellTypist disagree: {n_sr_only:,} cells ({n_sr_only/n_cells:.1%})")

    if n_sr_only > 100:
        sr_only_df = df.filter(sr_agree_ct_disagree)
        print("\nWhat SingleR says vs what CellTypist says:")
        crosstab = (
            sr_only_df
            .group_by(["sr_hpca_coarse", "ct_immune_coarse"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(8)
        )
        for row in crosstab.iter_rows(named=True):
            print(f"  SR_HPCA={row['sr_hpca_coarse']:<12}, CT_Immune={row['ct_immune_coarse']:<12}: {row['count']:,}")

    # Unknown rate analysis
    print("\n\n## Unknown Rate Analysis")
    print("-" * 60)

    for i, col in enumerate(methods):
        unk = (df[col] == "Unknown").sum()
        print(f"{method_names[i]}: {unk:,} unknown ({unk/n_cells:.1%})")

    # Where CellTypist is Unknown but SingleR has prediction
    ct_unk_sr_not = (
        (df["ct_immune_coarse"] == "Unknown") &
        (df["sr_hpca_coarse"] != "Unknown")
    )
    n_ct_unk = ct_unk_sr_not.sum()
    print(f"\nCT_Immune Unknown but SR_HPCA has label: {n_ct_unk:,} cells")

    if n_ct_unk > 0:
        ct_unk_df = df.filter(ct_unk_sr_not)
        sr_dist = ct_unk_df.group_by("sr_hpca_coarse").agg(pl.len().alias("count")).sort("count", descending=True)
        print("  SingleR labels for these cells:")
        for row in sr_dist.iter_rows(named=True):
            print(f"    {row['sr_hpca_coarse']}: {row['count']:,}")

    # Where SingleR is Unknown but CellTypist has prediction
    sr_unk_ct_not = (
        (df["sr_hpca_coarse"] == "Unknown") &
        (df["ct_immune_coarse"] != "Unknown")
    )
    n_sr_unk = sr_unk_ct_not.sum()
    print(f"\nSR_HPCA Unknown but CT_Immune has label: {n_sr_unk:,} cells")

    if n_sr_unk > 0:
        sr_unk_df = df.filter(sr_unk_ct_not)
        ct_dist = sr_unk_df.group_by("ct_immune_coarse").agg(pl.len().alias("count")).sort("count", descending=True)
        print("  CellTypist labels for these cells:")
        for row in ct_dist.iter_rows(named=True):
            print(f"    {row['ct_immune_coarse']}: {row['count']:,}")

    # Ensemble potential
    print("\n\n## Ensemble Potential Analysis")
    print("-" * 60)

    # If we use CellTypist when confident, SingleR as backup
    # "Confident" = both CellTypist models agree
    ct_confident = df["ct_immune_coarse"] == df["ct_breast_coarse"]
    n_confident = ct_confident.sum()
    print(f"\nCellTypist confident (both models agree): {n_confident:,} cells ({n_confident/n_cells:.1%})")

    # For non-confident cells, check if SingleR can help
    ct_not_confident = ~ct_confident
    n_not_confident = ct_not_confident.sum()
    print(f"CellTypist not confident: {n_not_confident:,} cells")

    if n_not_confident > 0:
        not_conf_df = df.filter(ct_not_confident)
        sr_agree_in_not_conf = (not_conf_df["sr_hpca_coarse"] == not_conf_df["sr_blueprint_coarse"]).sum()
        print(f"  Of these, SingleR both agree: {sr_agree_in_not_conf:,} ({sr_agree_in_not_conf/n_not_confident:.1%})")

    # Category distribution comparison
    print("\n\n## Category Distribution by Method")
    print("-" * 60)

    for i, col in enumerate(methods):
        dist = df.group_by(col).agg(pl.len().alias("count")).sort("count", descending=True)
        print(f"\n{method_names[i]}:")
        for row in dist.iter_rows(named=True):
            pct = row["count"] / n_cells * 100
            print(f"  {row[col]:<15}: {row['count']:>6} ({pct:.1f}%)")

    return df


def main():
    """Main analysis."""
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "rep1"

    if dataset == "rep1":
        xenium_path = Path.home() / "datasets/raw/xenium/breast_tumor_rep1"
    elif dataset == "rep2":
        xenium_path = Path.home() / "datasets/raw/xenium/breast_tumor_rep2"
    else:
        logger.error(f"Unknown dataset: {dataset}")
        return

    # Load data
    adata = load_xenium_data(xenium_path, max_cells=30000)

    # Run all methods
    df = run_all_methods(adata)

    # Save predictions
    output_path = Path(f"benchmark_results/cell_level_{dataset}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(f"Saved predictions to {output_path}")

    # Analyze agreement
    df = analyze_agreement(df)


if __name__ == "__main__":
    main()
