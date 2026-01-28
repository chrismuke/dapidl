#!/usr/bin/env python3
"""Analyze complementarity between annotation methods."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import scanpy as sc
from loguru import logger

# Load the validation results
def load_validation_results(dataset: str) -> dict:
    """Load validation JSON for a dataset."""
    path = Path(f"benchmark_results/validation_{dataset}/validation_{dataset}.json")
    with open(path) as f:
        return {r["method"]: r for r in json.load(f)}


def analyze_complementarity(dataset: str = "rep1"):
    """Analyze where methods agree/disagree and their strengths."""

    logger.info(f"Analyzing method complementarity for {dataset}")

    # Load the AnnData with predictions (we need to regenerate this)
    # For now, let's analyze the marker validation results
    results = load_validation_results(dataset)

    print(f"\n{'='*80}")
    print(f"METHOD COMPLEMENTARITY ANALYSIS: {dataset}")
    print(f"{'='*80}")

    # Compare marker scores by cell type
    methods = ["celltypist_immune", "celltypist_breast", "singler_hpca", "singler_blueprint"]

    # Get all cell types validated across methods
    all_cell_types = set()
    for method in methods:
        if method in results and "marker_details" in results[method]:
            details = results[method]["marker_details"]
            if "per_type_results" in details:
                all_cell_types.update(details["per_type_results"].keys())

    print(f"\n## Marker Validation Comparison by Cell Type")
    print(f"\nCell types validated: {len(all_cell_types)}")

    # Build comparison table
    rows = []
    for ct in sorted(all_cell_types):
        row = {"cell_type": ct}
        for method in methods:
            if method in results and "marker_details" in results[method]:
                details = results[method]["marker_details"]
                if "per_type_results" in details and ct in details["per_type_results"]:
                    score = details["per_type_results"][ct]["marker_score"]
                    n_cells = details["per_type_results"][ct]["n_cells"]
                    row[f"{method}_score"] = score
                    row[f"{method}_n"] = n_cells
                else:
                    row[f"{method}_score"] = None
                    row[f"{method}_n"] = 0
            else:
                row[f"{method}_score"] = None
                row[f"{method}_n"] = 0
        rows.append(row)

    df = pl.DataFrame(rows)

    # Find cell types where methods differ significantly
    print("\n### Cell Types with Best Scores by Method")
    print("-" * 80)

    for method in methods:
        score_col = f"{method}_score"
        n_col = f"{method}_n"

        # Get top performers for this method
        filtered = df.filter(pl.col(score_col).is_not_null() & (pl.col(n_col) > 50))
        if filtered.height > 0:
            top = filtered.sort(score_col, descending=True).head(5)
            print(f"\n**{method}** top cell types:")
            for row in top.iter_rows(named=True):
                print(f"  {row['cell_type']}: {row[score_col]:.3f} (n={row[n_col]})")

    # Compare coarse categories
    print("\n\n### Coarse Category Comparison")
    print("-" * 80)

    coarse_types = ["Epithelial", "Immune", "Stromal", "Epithelial cells", "T cells",
                    "B cells", "Macrophages", "Fibroblasts", "Endothelial cells"]

    print(f"\n{'Cell Type':<25} {'CT_Immune':<12} {'CT_Breast':<12} {'SR_HPCA':<12} {'SR_Blueprint':<12}")
    print("-" * 80)

    for ct in coarse_types:
        scores = []
        for method in methods:
            if method in results and "marker_details" in results[method]:
                details = results[method]["marker_details"]
                if "per_type_results" in details and ct in details["per_type_results"]:
                    scores.append(f"{details['per_type_results'][ct]['marker_score']:.3f}")
                else:
                    scores.append("-")
            else:
                scores.append("-")

        if any(s != "-" for s in scores):
            print(f"{ct:<25} {scores[0]:<12} {scores[1]:<12} {scores[2]:<12} {scores[3]:<12}")

    # Analyze cross-method agreement
    print("\n\n### Cross-Method Agreement Rates")
    print("-" * 80)

    for method in methods:
        if method in results:
            agreement = results[method].get("cross_method_agreement", 0)
            print(f"{method}: {agreement:.1%}")

    # Analyze proportion predictions
    print("\n\n### Cell Type Proportions by Method")
    print("-" * 80)

    print(f"\n{'Category':<15} {'CT_Immune':<12} {'CT_Breast':<12} {'SR_HPCA':<12} {'SR_Blueprint':<12}")
    print("-" * 70)

    categories = ["Epithelial", "Immune", "Stromal", "Endothelial", "Unknown"]
    for cat in categories:
        props = []
        for method in methods:
            if method in results and "proportions" in results[method]:
                prop = results[method]["proportions"].get(cat, 0)
                props.append(f"{prop:.1%}")
            else:
                props.append("-")
        print(f"{cat:<15} {props[0]:<12} {props[1]:<12} {props[2]:<12} {props[3]:<12}")

    # Unknown rate comparison
    print("\n\n### Unknown Rate Comparison")
    print("-" * 80)
    for method in methods:
        if method in results:
            unk = results[method].get("unknown_rate", 0)
            print(f"{method}: {unk:.1%}")

    # Spatial coherence comparison
    print("\n\n### Spatial Coherence Comparison")
    print("-" * 80)
    for method in methods:
        if method in results:
            spatial = results[method].get("spatial_coherence", 0)
            print(f"{method}: {spatial:.3f}")

    return results


def analyze_ensemble_potential(datasets: list[str] = ["rep1", "rep2", "merscope"]):
    """Analyze potential for ensemble methods."""

    print("\n" + "=" * 80)
    print("ENSEMBLE POTENTIAL ANALYSIS")
    print("=" * 80)

    for dataset in datasets:
        try:
            results = load_validation_results(dataset)

            print(f"\n## {dataset.upper()}")
            print("-" * 40)

            # Check if SingleR catches things CellTypist misses
            ct_immune = results.get("celltypist_immune", {})
            ct_breast = results.get("celltypist_breast", {})
            sr_hpca = results.get("singler_hpca", {})
            sr_bp = results.get("singler_blueprint", {})

            # Compare overall scores
            print(f"\nOverall scores:")
            print(f"  CellTypist Immune: {ct_immune.get('overall_score', 0):.3f}")
            print(f"  CellTypist Breast: {ct_breast.get('overall_score', 0):.3f}")
            print(f"  SingleR HPCA:      {sr_hpca.get('overall_score', 0):.3f}")
            print(f"  SingleR Blueprint: {sr_bp.get('overall_score', 0):.3f}")

            # Check marker scores for key types
            print(f"\nKey cell type marker scores:")

            key_types = {
                "Epithelial": ["Epithelial", "Epithelial cells"],
                "Immune": ["Immune", "T cells", "B cells", "Macrophages"],
                "Stromal": ["Stromal", "Fibroblasts"],
            }

            for category, types in key_types.items():
                print(f"\n  {category}:")
                for ct in types:
                    scores = {}
                    for method_name, method_data in [
                        ("CT_Imm", ct_immune),
                        ("CT_Brs", ct_breast),
                        ("SR_HPCA", sr_hpca),
                        ("SR_BP", sr_bp),
                    ]:
                        if "marker_details" in method_data:
                            per_type = method_data["marker_details"].get("per_type_results", {})
                            if ct in per_type:
                                scores[method_name] = per_type[ct]["marker_score"]

                    if scores:
                        best_method = max(scores, key=scores.get)
                        score_str = ", ".join([f"{k}={v:.3f}" for k, v in scores.items()])
                        print(f"    {ct}: {score_str} -> Best: {best_method}")

        except FileNotFoundError:
            print(f"  No validation results found for {dataset}")


if __name__ == "__main__":
    # Analyze each dataset
    for dataset in ["rep1", "rep2", "merscope"]:
        try:
            analyze_complementarity(dataset)
        except Exception as e:
            logger.error(f"Error analyzing {dataset}: {e}")

    # Analyze ensemble potential
    analyze_ensemble_potential()
