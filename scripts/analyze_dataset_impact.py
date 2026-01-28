#!/usr/bin/env python3
"""Analyze dataset impact for multi-tissue training.

This script:
1. Compares class distributions between datasets
2. Identifies which datasets have positive/negative impact
"""

import json
from pathlib import Path
from collections import Counter

from loguru import logger

from dapidl.data.multi_tissue_dataset import (
    MultiTissueConfig,
    MultiTissueDataset,
)
from dapidl.ontology.cl_mapper import CLMapper


def analyze_dataset(lmdb_path: Path, tissue: str, platform: str) -> dict:
    """Analyze a single dataset using MultiTissueDataset."""
    config = MultiTissueConfig(
        sampling_strategy="proportional",
        standardize_labels=True,
    )
    config.add_dataset(
        path=str(lmdb_path),
        tissue=tissue,
        platform=platform,
        confidence_tier=1,
        weight_multiplier=1.0,
    )

    dataset = MultiTissueDataset(config, split="all")

    # Get class distribution
    class_counts = Counter()
    coarse_counts = Counter()

    # Get hierarchy info
    hier = dataset.hierarchical_labels
    fine_names = hier.fine_names if hier else []
    coarse_names = hier.coarse_names if hier else []

    # Sample a subset for faster analysis
    sample_size = min(5000, len(dataset))
    indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))[:sample_size]

    for i, idx in enumerate(indices):
        try:
            _, labels = dataset[idx]
            fine_label = labels.get("fine", labels.get("label", 0))
            coarse_label = labels.get("coarse", 0)

            # Handle tensor vs int
            if hasattr(fine_label, "item"):
                fine_label = fine_label.item()
            if hasattr(coarse_label, "item"):
                coarse_label = coarse_label.item()

            fine_name = fine_names[fine_label] if fine_label < len(fine_names) else f"fine_{fine_label}"
            coarse_name = coarse_names[coarse_label] if coarse_label < len(coarse_names) else f"coarse_{coarse_label}"

            class_counts[fine_name] += 1
            coarse_counts[coarse_name] += 1
        except Exception as e:
            if i == 0:
                logger.warning(f"Error reading sample {idx}: {e}")
            break

    # Scale counts to full dataset size
    scale_factor = len(dataset) / len(indices) if indices else 1
    class_counts = Counter({k: int(v * scale_factor) for k, v in class_counts.items()})
    coarse_counts = Counter({k: int(v * scale_factor) for k, v in coarse_counts.items()})

    return {
        "num_samples": len(dataset),
        "fine_classes": len(class_counts),
        "coarse_classes": len(coarse_counts),
        "fine_distribution": dict(class_counts),
        "coarse_distribution": dict(coarse_counts),
        "hierarchy": {
            "fine_names": fine_names,
            "coarse_names": coarse_names,
        },
    }


def main():
    # Dataset paths
    rep1_path = Path("~/datasets/raw/xenium/breast_tumor_rep1/pipeline_outputs/patches/patches.lmdb").expanduser()
    rep2_path = Path("~/datasets/raw/xenium/breast_tumor_rep2/pipeline_outputs/patches/patches.lmdb").expanduser()

    logger.info("=" * 60)
    logger.info("Dataset Impact Analysis")
    logger.info("=" * 60)

    # 1. Analyze individual datasets
    logger.info("\n1. Analyzing Rep1...")
    rep1_stats = analyze_dataset(rep1_path, "breast", "xenium")

    logger.info("\n2. Analyzing Rep2...")
    rep2_stats = analyze_dataset(rep2_path, "breast", "xenium")

    # Print results
    print("\n" + "=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)

    print(f"\nRep1: {rep1_stats['num_samples']:,} samples, {rep1_stats['fine_classes']} fine classes, {rep1_stats['coarse_classes']} coarse classes")
    print(f"Rep2: {rep2_stats['num_samples']:,} samples, {rep2_stats['fine_classes']} fine classes, {rep2_stats['coarse_classes']} coarse classes")

    # Coarse distribution comparison
    print("\n--- COARSE DISTRIBUTION ---")
    all_coarse = set(rep1_stats['coarse_distribution'].keys()) | set(rep2_stats['coarse_distribution'].keys())
    print(f"{'Category':<20} {'Rep1':>12} {'Rep1 %':>10} {'Rep2':>12} {'Rep2 %':>10} {'Diff':>10}")
    print("-" * 74)

    for cat in sorted(all_coarse):
        r1 = rep1_stats['coarse_distribution'].get(cat, 0)
        r2 = rep2_stats['coarse_distribution'].get(cat, 0)
        r1_pct = 100 * r1 / rep1_stats['num_samples'] if rep1_stats['num_samples'] > 0 else 0
        r2_pct = 100 * r2 / rep2_stats['num_samples'] if rep2_stats['num_samples'] > 0 else 0
        diff = r1_pct - r2_pct
        print(f"{cat:<20} {r1:>12,} {r1_pct:>9.1f}% {r2:>12,} {r2_pct:>9.1f}% {diff:>+9.1f}%")

    # Fine distribution comparison
    print("\n--- FINE DISTRIBUTION (largest differences) ---")
    all_fine = set(rep1_stats['fine_distribution'].keys()) | set(rep2_stats['fine_distribution'].keys())

    fine_diff = []
    for cls in all_fine:
        r1 = rep1_stats['fine_distribution'].get(cls, 0)
        r2 = rep2_stats['fine_distribution'].get(cls, 0)
        r1_pct = 100 * r1 / rep1_stats['num_samples'] if rep1_stats['num_samples'] > 0 else 0
        r2_pct = 100 * r2 / rep2_stats['num_samples'] if rep2_stats['num_samples'] > 0 else 0
        diff_abs = abs(r1_pct - r2_pct)
        fine_diff.append((cls, r1, r1_pct, r2, r2_pct, r1_pct - r2_pct, diff_abs))

    fine_diff.sort(key=lambda x: x[6], reverse=True)

    print(f"{'Class':<40} {'Rep1 %':>10} {'Rep2 %':>10} {'Diff':>10}")
    print("-" * 70)
    for cls, r1, r1_pct, r2, r2_pct, diff, _ in fine_diff[:15]:
        print(f"{cls:<40} {r1_pct:>9.2f}% {r2_pct:>9.2f}% {diff:>+9.2f}%")

    # Identify classes only in one dataset
    only_rep1 = set(rep1_stats['fine_distribution'].keys()) - set(rep2_stats['fine_distribution'].keys())
    only_rep2 = set(rep2_stats['fine_distribution'].keys()) - set(rep1_stats['fine_distribution'].keys())

    if only_rep1:
        print(f"\n--- CLASSES ONLY IN REP1 ({len(only_rep1)}) ---")
        for cls in sorted(only_rep1):
            count = rep1_stats['fine_distribution'][cls]
            print(f"  {cls}: {count:,} samples")

    if only_rep2:
        print(f"\n--- CLASSES ONLY IN REP2 ({len(only_rep2)}) ---")
        for cls in sorted(only_rep2):
            count = rep2_stats['fine_distribution'][cls]
            print(f"  {cls}: {count:,} samples")

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("IMPACT ANALYSIS")
    print("=" * 60)

    issues = []
    recommendations = []

    # Check coarse balance
    for cat in all_coarse:
        r1_pct = 100 * rep1_stats['coarse_distribution'].get(cat, 0) / rep1_stats['num_samples'] if rep1_stats['num_samples'] > 0 else 0
        r2_pct = 100 * rep2_stats['coarse_distribution'].get(cat, 0) / rep2_stats['num_samples'] if rep2_stats['num_samples'] > 0 else 0
        if abs(r1_pct - r2_pct) > 5:
            issues.append(f"Coarse '{cat}' differs by {abs(r1_pct - r2_pct):.1f}%: Rep1={r1_pct:.1f}%, Rep2={r2_pct:.1f}%")

    # Check for dataset-specific classes
    if only_rep1:
        total_only_rep1 = sum(rep1_stats['fine_distribution'][c] for c in only_rep1)
        issues.append(f"{len(only_rep1)} classes only in Rep1 ({total_only_rep1:,} samples)")
        if total_only_rep1 > 0.01 * rep1_stats['num_samples']:
            recommendations.append("Consider filtering Rep1-only classes or using class mapping")

    if only_rep2:
        total_only_rep2 = sum(rep2_stats['fine_distribution'][c] for c in only_rep2)
        issues.append(f"{len(only_rep2)} classes only in Rep2 ({total_only_rep2:,} samples)")
        if total_only_rep2 > 0.01 * rep2_stats['num_samples']:
            recommendations.append("Consider filtering Rep2-only classes or using class mapping")

    # Check for large class differences
    for cls, r1, r1_pct, r2, r2_pct, diff, diff_abs in fine_diff:
        if diff_abs > 10:
            if diff > 0:
                issues.append(f"'{cls}' overrepresented in Rep1 (+{diff:.1f}%)")
            else:
                issues.append(f"'{cls}' overrepresented in Rep2 ({diff:.1f}%)")

    if issues:
        print("\nPotential Issues:")
        for issue in issues[:15]:
            print(f"  - {issue}")

    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

    # Overall assessment
    print("\n--- OVERALL ASSESSMENT ---")

    # Calculate overlap similarity
    common_classes = set(rep1_stats['fine_distribution'].keys()) & set(rep2_stats['fine_distribution'].keys())
    jaccard = len(common_classes) / len(all_fine) if all_fine else 0

    print(f"Class overlap (Jaccard): {jaccard:.2%}")
    print(f"Common classes: {len(common_classes)}/{len(all_fine)}")

    if jaccard > 0.9 and len(issues) < 3:
        print("\n✓ Datasets are highly compatible - combining should be beneficial")
    elif jaccard > 0.7:
        print("\n~ Datasets are moderately compatible - combining may help with care")
    else:
        print("\n⚠ Datasets have significant differences - combining may hurt performance")

    # Save results
    output = {
        "rep1": rep1_stats,
        "rep2": rep2_stats,
        "overlap": {
            "common_classes": list(common_classes),
            "only_rep1": list(only_rep1),
            "only_rep2": list(only_rep2),
            "jaccard": jaccard,
        },
        "issues": issues,
        "recommendations": recommendations,
    }

    with open("dataset_impact_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("\nResults saved to dataset_impact_analysis.json")


if __name__ == "__main__":
    main()
