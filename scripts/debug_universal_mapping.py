#!/usr/bin/env python3
"""Debug universal annotation - find which labels fail to map.

This script identifies which CellTypist model outputs are not covered
by the current CELL_TYPE_HIERARCHY mapping.
"""

from pathlib import Path
from collections import Counter, defaultdict
import time

import anndata as ad
import numpy as np
from loguru import logger

from dapidl.pipeline.components.annotators.universal_ensemble import (
    CORE_HUMAN_MODELS,
    filter_available_models,
)
from dapidl.pipeline.components.annotators.mapping import (
    map_to_broad_category,
    CELL_TYPE_HIERARCHY,
)


def load_xenium_adata(xenium_path: Path) -> ad.AnnData:
    """Load AnnData from Xenium output directory and normalize."""
    from dapidl.data.xenium import XeniumDataReader
    import scanpy as sc

    reader = XeniumDataReader(xenium_path)
    expr_matrix, gene_names, cell_ids = reader.load_expression_matrix()
    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_names
    adata.obs_names = [str(c) for c in cell_ids]

    # CellTypist requires log1p normalized data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def run_individual_models(adata: ad.AnnData) -> dict[str, Counter]:
    """Run each CellTypist model individually and collect outputs."""
    import celltypist

    model_outputs = {}
    available_models = filter_available_models(CORE_HUMAN_MODELS)

    logger.info(f"Running {len(available_models)} models individually...")

    for model_name in available_models:
        logger.info(f"Running {model_name}...")
        try:
            model = celltypist.models.Model.load(model=model_name)
            predictions = celltypist.annotate(
                adata,
                model=model,
                majority_voting=False,
            )
            labels = predictions.predicted_labels["predicted_labels"].tolist()
            model_outputs[model_name] = Counter(labels)
            logger.info(f"  {len(set(labels))} unique cell types")
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    return model_outputs


def analyze_mapping_failures(model_outputs: dict[str, Counter]) -> dict:
    """Analyze which labels fail to map."""
    all_labels = set()
    unmapped_labels = set()
    mapped_labels = set()

    label_to_models = defaultdict(list)
    label_to_counts = Counter()

    for model_name, label_counts in model_outputs.items():
        for label, count in label_counts.items():
            all_labels.add(label)
            label_to_models[label].append(model_name)
            label_to_counts[label] += count

            broad = map_to_broad_category(label)
            if broad == "Unknown":
                unmapped_labels.add(label)
            else:
                mapped_labels.add(label)

    return {
        "total_labels": len(all_labels),
        "mapped_labels": len(mapped_labels),
        "unmapped_labels": len(unmapped_labels),
        "unmapped_list": sorted(unmapped_labels),
        "label_to_models": dict(label_to_models),
        "label_counts": dict(label_to_counts),
    }


def suggest_mappings(unmapped_labels: list[str]) -> dict[str, str]:
    """Suggest mappings for unmapped labels based on keywords."""
    suggestions = {}

    immune_keywords = [
        "nk", "natural killer", "t cell", "b cell", "lymph", "immune",
        "macro", "mono", "dendrit", "mast", "neutro", "myeloid", "leuko",
        "plasma", "erythro", "granu", "basophil", "thym", "cd4", "cd8"
    ]
    epithelial_keywords = [
        "epithelial", "epithelium", "keratin", "gland", "ductal", "acinar",
        "club", "ciliat", "alveol", "hepato", "entero", "colon", "goblet",
        "secretory", "mucous", "tubul", "basal", "luminal"
    ]
    stromal_keywords = [
        "fibro", "stroma", "endotheli", "vascular", "pericyte", "smooth",
        "mesench", "adipoc", "fat", "connective", "muscle", "interstitial"
    ]

    for label in unmapped_labels:
        label_lower = label.lower()

        # Check immune first (more specific)
        if any(kw in label_lower for kw in immune_keywords):
            suggestions[label] = "Immune"
        elif any(kw in label_lower for kw in epithelial_keywords):
            suggestions[label] = "Epithelial"
        elif any(kw in label_lower for kw in stromal_keywords):
            suggestions[label] = "Stromal"
        else:
            suggestions[label] = "Unknown (needs manual review)"

    return suggestions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug universal annotation mapping")
    parser.add_argument(
        "--dataset",
        type=str,
        default="~/datasets/raw/xenium/breast_tumor_rep1",
        help="Path to Xenium dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of cells to sample",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    # Load and sample data
    logger.info(f"Loading dataset: {dataset_path}")
    adata = load_xenium_adata(dataset_path)

    if adata.n_obs > args.sample_size:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, args.sample_size, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Sampled {args.sample_size} cells")

    # Run models
    model_outputs = run_individual_models(adata)

    # Analyze mapping failures
    analysis = analyze_mapping_failures(model_outputs)

    # Print results
    print("\n" + "=" * 70)
    print("MAPPING ANALYSIS")
    print("=" * 70)

    print(f"\nTotal unique labels: {analysis['total_labels']}")
    print(f"Successfully mapped: {analysis['mapped_labels']}")
    print(f"Failed to map (Unknown): {analysis['unmapped_labels']}")
    if analysis['total_labels'] > 0:
        print(f"Mapping coverage: {analysis['mapped_labels']/analysis['total_labels']:.1%}")
    else:
        print("Mapping coverage: N/A (no labels)")

    print("\n" + "-" * 70)
    print("UNMAPPED LABELS (sorted by frequency)")
    print("-" * 70)

    # Sort by count
    unmapped_counts = {
        label: analysis['label_counts'][label]
        for label in analysis['unmapped_list']
    }
    sorted_unmapped = sorted(unmapped_counts.items(), key=lambda x: -x[1])

    # Get suggestions
    suggestions = suggest_mappings(analysis['unmapped_list'])

    print(f"{'Label':<50} {'Count':>8} {'Suggested':>12}")
    print("-" * 70)
    for label, count in sorted_unmapped[:30]:
        suggestion = suggestions.get(label, "?")
        print(f"{label[:48]:<50} {count:>8} {suggestion:>12}")

    # Show which models produce each unmapped label
    print("\n" + "-" * 70)
    print("UNMAPPED LABELS BY MODEL")
    print("-" * 70)

    model_unmapped = defaultdict(list)
    for label in analysis['unmapped_list']:
        for model in analysis['label_to_models'][label]:
            model_unmapped[model].append(label)

    for model, labels in sorted(model_unmapped.items()):
        print(f"\n{model}:")
        for label in sorted(labels)[:10]:
            count = analysis['label_counts'][label]
            print(f"  - {label} ({count} cells)")
        if len(labels) > 10:
            print(f"  ... and {len(labels) - 10} more")

    # Generate suggested additions to CELL_TYPE_HIERARCHY
    print("\n" + "=" * 70)
    print("SUGGESTED ADDITIONS TO CELL_TYPE_HIERARCHY")
    print("=" * 70)

    for category in ["Immune", "Epithelial", "Stromal"]:
        cat_suggestions = [l for l, s in suggestions.items() if s == category]
        if cat_suggestions:
            print(f"\n# Add to {category}:")
            for label in sorted(cat_suggestions):
                print(f'        "{label}",')


if __name__ == "__main__":
    main()
