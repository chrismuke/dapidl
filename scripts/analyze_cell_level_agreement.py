#!/usr/bin/env python3
"""Analyze cell-level agreement/disagreement between annotation methods."""

from pathlib import Path

import polars as pl
import scanpy as sc
from loguru import logger

# Import our annotation methods
from dapidl.pipeline.components.annotators.singler import SingleRAnnotator
from dapidl.pipeline.base import AnnotationConfig


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


def load_xenium_data(xenium_path: Path, max_cells: int = 50000) -> sc.AnnData:
    """Load Xenium data into AnnData format."""
    logger.info(f"Loading Xenium data from {xenium_path}")

    # Load expression matrix
    import scipy.sparse as sp
    import h5py

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
        import numpy as np
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


def run_all_methods(adata: sc.AnnData) -> pl.DataFrame:
    """Run all annotation methods and collect predictions."""
    import celltypist
    from celltypist import models

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
    logger.info("Running SingleR HPCA...")
    hpca_config = AnnotationConfig(singler_reference="hpca")
    singler = SingleRAnnotator(config=hpca_config)
    sr_results = singler.annotate(adata=adata)
    results["sr_hpca"] = sr_results.predictions["cell_type"].to_list()
    results["sr_hpca_coarse"] = sr_results.predictions["broad_category"].to_list()

    # SingleR Blueprint
    logger.info("Running SingleR Blueprint...")
    bp_config = AnnotationConfig(singler_reference="blueprint")
    singler_bp = SingleRAnnotator(config=bp_config)
    sr_results = singler_bp.annotate(adata=adata)
    results["sr_blueprint"] = sr_results.predictions["cell_type"].to_list()
    results["sr_blueprint_coarse"] = sr_results.predictions["broad_category"].to_list()

    return pl.DataFrame(results)


def analyze_agreement(df: pl.DataFrame):
    """Analyze agreement patterns between methods."""

    print("\n" + "=" * 80)
    print("CELL-LEVEL AGREEMENT ANALYSIS")
    print("=" * 80)

    n_cells = df.height

    # Coarse-level agreement
    methods = ["ct_immune_coarse", "ct_breast_coarse", "sr_hpca_coarse", "sr_blueprint_coarse"]

    print(f"\n## Coarse Category Agreement (n={n_cells:,} cells)")
    print("-" * 60)

    # Pairwise agreement
    print("\n### Pairwise Agreement Matrix")
    print(f"{'Method':<20}", end="")
    for m in methods:
        short = m.replace("_coarse", "")[:12]
        print(f"{short:<12}", end="")
    print()

    for m1 in methods:
        short1 = m1.replace("_coarse", "")[:12]
        print(f"{short1:<20}", end="")
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

    # CellTypist methods agree
    ct_agree = (df["ct_immune_coarse"] == df["ct_breast_coarse"]).sum()
    print(f"### Both CellTypist methods agree: {ct_agree:,} cells ({ct_agree/n_cells:.1%})")

    # SingleR methods agree
    sr_agree = (df["sr_hpca_coarse"] == df["sr_blueprint_coarse"]).sum()
    print(f"### Both SingleR methods agree: {sr_agree:,} cells ({sr_agree/n_cells:.1%})")

    # Cross-family agreement (any CellTypist == any SingleR)
    ct_sr_agree = (
        (df["ct_immune_coarse"] == df["sr_hpca_coarse"]) |
        (df["ct_immune_coarse"] == df["sr_blueprint_coarse"]) |
        (df["ct_breast_coarse"] == df["sr_hpca_coarse"]) |
        (df["ct_breast_coarse"] == df["sr_blueprint_coarse"])
    ).sum()
    print(f"### At least one CellTypist agrees with at least one SingleR: {ct_sr_agree:,} cells ({ct_sr_agree/n_cells:.1%})")

    # Disagreement analysis
    print("\n\n## Disagreement Patterns")
    print("-" * 60)

    # Where CT_Immune and SingleR disagree
    disagree_mask = df["ct_immune_coarse"] != df["sr_hpca_coarse"]
    n_disagree = disagree_mask.sum()
    print(f"\n### CT_Immune vs SR_HPCA disagree: {n_disagree:,} cells ({n_disagree/n_cells:.1%})")

    if n_disagree > 0:
        disagree_df = df.filter(disagree_mask)

        # Cross-tabulation
        crosstab = (
            disagree_df
            .group_by(["ct_immune_coarse", "sr_hpca_coarse"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(10)
        )

        print("\nTop disagreement patterns (CT_Immune -> SR_HPCA):")
        for row in crosstab.iter_rows(named=True):
            pct = row["count"] / n_disagree * 100
            print(f"  {row['ct_immune_coarse']:<15} -> {row['sr_hpca_coarse']:<15}: {row['count']:>6} ({pct:.1f}%)")

    # Where all CellTypist agrees but SingleR disagrees
    ct_agree_sr_disagree = (
        (df["ct_immune_coarse"] == df["ct_breast_coarse"]) &
        (df["ct_immune_coarse"] != df["sr_hpca_coarse"]) &
        (df["ct_immune_coarse"] != df["sr_blueprint_coarse"])
    )
    n_ct_only = ct_agree_sr_disagree.sum()
    print(f"\n### CellTypist agrees but both SingleR disagree: {n_ct_only:,} cells ({n_ct_only/n_cells:.1%})")

    if n_ct_only > 100:
        ct_only_df = df.filter(ct_agree_sr_disagree)
        ct_dist = ct_only_df.group_by("ct_immune_coarse").agg(pl.len().alias("count")).sort("count", descending=True)
        print("\nCellTypist predictions when SingleR disagrees:")
        for row in ct_dist.iter_rows(named=True):
            print(f"  {row['ct_immune_coarse']}: {row['count']:,}")

    # Where SingleR agrees but CellTypist disagrees
    sr_agree_ct_disagree = (
        (df["sr_hpca_coarse"] == df["sr_blueprint_coarse"]) &
        (df["sr_hpca_coarse"] != df["ct_immune_coarse"]) &
        (df["sr_hpca_coarse"] != df["ct_breast_coarse"])
    )
    n_sr_only = sr_agree_ct_disagree.sum()
    print(f"\n### SingleR agrees but both CellTypist disagree: {n_sr_only:,} cells ({n_sr_only/n_cells:.1%})")

    if n_sr_only > 100:
        sr_only_df = df.filter(sr_agree_ct_disagree)
        sr_dist = sr_only_df.group_by("sr_hpca_coarse").agg(pl.len().alias("count")).sort("count", descending=True)
        print("\nSingleR predictions when CellTypist disagrees:")
        for row in sr_dist.iter_rows(named=True):
            print(f"  {row['sr_hpca_coarse']}: {row['count']:,}")

    # Unknown rate analysis
    print("\n\n## Unknown Rate Analysis")
    print("-" * 60)

    for col in methods:
        unk = (df[col] == "Unknown").sum()
        print(f"{col}: {unk:,} unknown ({unk/n_cells:.1%})")

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

    # Ensemble voting simulation
    print("\n\n## Ensemble Voting Simulation")
    print("-" * 60)

    # Majority vote (3+ agree)
    def majority_vote(row):
        votes = [row["ct_immune_coarse"], row["ct_breast_coarse"],
                 row["sr_hpca_coarse"], row["sr_blueprint_coarse"]]
        # Filter out Unknown votes
        valid_votes = [v for v in votes if v != "Unknown"]
        if not valid_votes:
            return "Unknown"

        from collections import Counter
        counts = Counter(valid_votes)
        most_common = counts.most_common(1)[0]
        if most_common[1] >= 2:  # At least 2 agree
            return most_common[0]
        return valid_votes[0]  # Fallback to first valid

    ensemble_labels = [majority_vote(row) for row in df.iter_rows(named=True)]
    df = df.with_columns(pl.Series("ensemble_coarse", ensemble_labels))

    # Compare ensemble to individual methods
    print("\n### Ensemble vs Individual Methods Agreement")
    for method in methods:
        agreement = (df["ensemble_coarse"] == df[method]).sum() / n_cells
        print(f"  Ensemble vs {method}: {agreement:.1%}")

    # Ensemble category distribution
    print("\n### Ensemble Category Distribution")
    ensemble_dist = df.group_by("ensemble_coarse").agg(pl.len().alias("count")).sort("count", descending=True)
    for row in ensemble_dist.iter_rows(named=True):
        pct = row["count"] / n_cells * 100
        print(f"  {row['ensemble_coarse']}: {row['count']:,} ({pct:.1f}%)")

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
    adata = load_xenium_data(xenium_path, max_cells=50000)

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
