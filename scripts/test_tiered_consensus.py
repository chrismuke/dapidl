"""Test tiered vs flat consensus on rep1 and rep2 at multiple granularities.

Compares:
- Old: flat consensus (equal voting between tissue + immune models)
- New: tiered consensus (tissue model primary, immune model refines)

Evaluates at:
- Broad (3 classes: Epithelial, Immune, Stromal)
- Medium (~10 classes: T_Cell, B_Cell, Macrophage, Fibroblast, etc.)
- Fine-grained (raw CellTypist labels)

Ground truth from Cell_Barcode_Type_Matrices.xlsx (breast rep1 + rep2).

Usage:
    uv run python scripts/test_tiered_consensus.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dapidl.pipeline.components.annotators.auto_selector import (
    AutoModelSelector,
    _map_to_medium,
    get_models_for_dataset,
    get_tissue_for_dataset,
    map_to_broad,
)

# Ground truth mapping from Xenium breast labels
GT_BROAD_MAP = {
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
}

# GT → medium category mapping
GT_MEDIUM_MAP = {
    "DCIS_1": "Epithelial_Luminal",
    "DCIS_2": "Epithelial_Luminal",
    "Invasive_Tumor": "Epithelial_Luminal",
    "Prolif_Invasive_Tumor": "Epithelial_Luminal",
    "Myoepi_ACTA2+": "Epithelial_Basal",
    "Myoepi_KRT15+": "Epithelial_Basal",
    "B_Cells": "B_Cell",
    "CD4+_T_Cells": "T_Cell",
    "CD8+_T_Cells": "T_Cell",
    "Macrophages_1": "Macrophage",
    "Macrophages_2": "Macrophage",
    "IRF7+_DCs": "Dendritic_Cell",
    "LAMP3+_DCs": "Dendritic_Cell",
    "Mast_Cells": "Mast_Cell",
    "Stromal": "Fibroblast",
    "Endothelial": "Endothelial",
    "Perivascular-Like": "Pericyte_SMC",
}


def load_ground_truth(data_path: Path, rep: str = "rep1") -> dict[str, str]:
    """Load ground truth cell type labels from Janesick et al. Excel file.

    The GT file is shared between rep1 and rep2 — different sheets.
    Cell IDs are numeric (matching Xenium cells.parquet cell_id).
    """
    import pandas as pd

    # Search for GT file
    gt_file = None
    for search_dir in [data_path, data_path.parent, data_path.parent.parent]:
        candidate = search_dir / "Cell_Barcode_Type_Matrices.xlsx"
        if candidate.exists():
            gt_file = candidate
            break
    # Also check rep1's directory for rep2 (shared file)
    if gt_file is None:
        sibling = data_path.parent / "xenium-breast-tumor-rep1" / "Cell_Barcode_Type_Matrices.xlsx"
        if sibling.exists():
            gt_file = sibling

    if gt_file is None:
        return {}

    # Select the right sheet
    sheet_map = {
        "rep1": "Xenium R1 Fig1-5 (supervised)",
        "rep2": "Xenium R2 Fig1-5 (supervised)",
    }
    sheet_name = sheet_map.get(rep)
    if sheet_name is None:
        return {}

    print(f"  GT file: {gt_file.name} → sheet '{sheet_name}'")
    df = pd.read_excel(gt_file, sheet_name=sheet_name)

    # Columns are 'Barcode' (numeric cell_id) and 'Cluster' (cell type)
    print(f"  GT rows: {len(df):,}")
    return dict(zip(df["Barcode"].astype(str), df["Cluster"].astype(str)))


def load_expression(data_path: Path):
    """Load expression data. Checks root and outs/ subdirectory."""
    import scanpy as sc

    for search_dir in [data_path, data_path / "outs"]:
        h5_path = search_dir / "cell_feature_matrix.h5"
        if h5_path.exists():
            return sc.read_10x_h5(h5_path)

        mtx_dir = search_dir / "cell_feature_matrix"
        if mtx_dir.exists():
            return sc.read_10x_mtx(mtx_dir)

    raise FileNotFoundError(f"No expression data in {data_path}")


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None):
    """Compute accuracy, macro F1, and per-class F1."""
    from collections import Counter

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0

    # Per-class precision, recall, F1
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(1 for t in y_true if t == label)

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = np.mean([v["f1"] for v in per_class.values()]) if per_class else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "n": n,
    }


def run_dataset(name: str, data_path: Path, gt_dict: dict[str, str], sample_size: int = 20000):
    """Run both consensus strategies on a dataset and compare against GT."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    tissue = get_tissue_for_dataset(str(data_path))
    models = get_models_for_dataset(str(data_path))
    print(f"  Tissue: {tissue}")
    print(f"  Models: {models}")

    # Load expression
    adata = load_expression(data_path)
    n_total = len(adata)
    print(f"  Cells: {n_total:,}")

    # Add cell_id to obs if not present
    if "cell_id" not in adata.obs.columns:
        adata.obs["cell_id"] = adata.obs.index.astype(str)

    # Filter to cells with GT
    gt_ids = set(gt_dict.keys())
    adata_ids = set(adata.obs["cell_id"].astype(str))
    overlap_ids = gt_ids & adata_ids
    print(f"  GT overlap: {len(overlap_ids):,} / {len(gt_ids):,} GT cells")

    if len(overlap_ids) < 100:
        print(f"  SKIP: Not enough GT overlap")
        return None

    # Filter to GT cells + sample
    gt_mask = adata.obs["cell_id"].astype(str).isin(overlap_ids)
    adata_gt = adata[gt_mask].copy()

    # Sample if too large
    if len(adata_gt) > sample_size:
        idx = np.random.choice(len(adata_gt), sample_size, replace=False)
        adata_gt = adata_gt[idx].copy()
        print(f"  Sampled: {len(adata_gt):,} GT cells")

    # Get GT labels for sampled cells
    cell_ids = adata_gt.obs["cell_id"].astype(str).values
    gt_labels = [gt_dict[cid] for cid in cell_ids]

    # Filter out Hybrid/Unlabeled from GT
    valid_mask = [
        gt in GT_BROAD_MAP for gt in gt_labels
    ]
    adata_gt = adata_gt[valid_mask].copy()
    cell_ids = [cid for cid, v in zip(cell_ids, valid_mask) if v]
    gt_labels = [gt for gt, v in zip(gt_labels, valid_mask) if v]
    print(f"  Valid GT cells: {len(gt_labels):,}")

    # Prepare GT at different granularities
    gt_broad = [GT_BROAD_MAP[gt] for gt in gt_labels]
    gt_medium = [GT_MEDIUM_MAP.get(gt, "Unknown") for gt in gt_labels]
    gt_fine = gt_labels  # Original GT labels

    # ── Run FLAT consensus ──
    print(f"\n  ── Flat Consensus (old) ──")
    t0 = time.time()
    selector = AutoModelSelector(tissue_type=tissue, candidate_models=models)
    flat_result = selector.build_consensus(
        adata_gt, models=models, min_agreement=0.5, confidence_weight=False,
    )
    flat_time = time.time() - t0
    print(f"  Time: {flat_time:.1f}s")

    flat_df = flat_result.annotations_df
    flat_broad = flat_df["consensus_broad"].to_list()
    flat_fine = flat_df["consensus_fine"].to_list()
    flat_medium = [_map_to_medium(f, b) for f, b in zip(flat_fine, flat_broad)]

    # ── Run TIERED consensus ──
    print(f"\n  ── Tiered Consensus (new) ──")
    t0 = time.time()
    tiered_result = selector.build_tiered_consensus(
        adata_gt, models=models, min_agreement=0.5,
    )
    tiered_time = time.time() - t0
    print(f"  Time: {tiered_time:.1f}s")

    tiered_df = tiered_result.annotations_df
    tiered_broad = tiered_df["consensus_broad"].to_list()
    tiered_fine = tiered_df["consensus_fine"].to_list()
    tiered_medium = tiered_df["consensus_medium"].to_list()

    # ── Evaluate BROAD ──
    print(f"\n  ── BROAD (3 classes) ──")
    broad_labels = ["Epithelial", "Immune", "Stromal"]

    flat_broad_m = compute_metrics(gt_broad, flat_broad, broad_labels)
    tiered_broad_m = compute_metrics(gt_broad, tiered_broad, broad_labels)

    print(f"  {'Method':<12} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Flat':<12} {flat_broad_m['accuracy']:>9.1%} {flat_broad_m['macro_f1']:>9.3f}")
    print(f"  {'Tiered':<12} {tiered_broad_m['accuracy']:>9.1%} {tiered_broad_m['macro_f1']:>9.3f}")

    print(f"\n  Per-class F1 (broad):")
    print(f"  {'Class':<15} {'Flat':>8} {'Tiered':>8} {'Support':>8}")
    for label in broad_labels:
        f_f1 = flat_broad_m["per_class"].get(label, {}).get("f1", 0)
        t_f1 = tiered_broad_m["per_class"].get(label, {}).get("f1", 0)
        sup = tiered_broad_m["per_class"].get(label, {}).get("support", 0)
        delta = "  ▲" if t_f1 > f_f1 + 0.01 else ("  ▼" if t_f1 < f_f1 - 0.01 else "")
        print(f"  {label:<15} {f_f1:>7.3f} {t_f1:>7.3f} {sup:>8,}{delta}")

    # ── Evaluate MEDIUM ──
    print(f"\n  ── MEDIUM (~10 classes) ──")
    medium_labels = sorted(set(gt_medium) - {"Unknown"})

    flat_medium_m = compute_metrics(gt_medium, flat_medium, medium_labels)
    tiered_medium_m = compute_metrics(gt_medium, tiered_medium, medium_labels)

    print(f"  {'Method':<12} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Flat':<12} {flat_medium_m['accuracy']:>9.1%} {flat_medium_m['macro_f1']:>9.3f}")
    print(f"  {'Tiered':<12} {tiered_medium_m['accuracy']:>9.1%} {tiered_medium_m['macro_f1']:>9.3f}")

    print(f"\n  Per-class F1 (medium):")
    print(f"  {'Class':<20} {'Flat':>8} {'Tiered':>8} {'Support':>8}")
    for label in medium_labels:
        f_f1 = flat_medium_m["per_class"].get(label, {}).get("f1", 0)
        t_f1 = tiered_medium_m["per_class"].get(label, {}).get("f1", 0)
        sup = tiered_medium_m["per_class"].get(label, {}).get("support", 0)
        delta = "  ▲" if t_f1 > f_f1 + 0.01 else ("  ▼" if t_f1 < f_f1 - 0.01 else "")
        print(f"  {label:<20} {f_f1:>7.3f} {t_f1:>7.3f} {sup:>8,}{delta}")

    # ── Evaluate FINE-GRAINED ──
    print(f"\n  ── FINE-GRAINED ──")
    # Map GT fine labels through broad mapping for comparison
    # (GT uses different label names than CellTypist)
    # We compare at broad level for fine-grained predictions
    # since label vocabularies differ completely

    # Instead: show distribution of fine-grained predictions
    flat_fine_counts = {}
    for f in flat_fine:
        flat_fine_counts[f] = flat_fine_counts.get(f, 0) + 1
    tiered_fine_counts = {}
    for f in tiered_fine:
        tiered_fine_counts[f] = tiered_fine_counts.get(f, 0) + 1

    print(f"  Flat: {len(flat_fine_counts)} unique types")
    print(f"  Tiered: {len(tiered_fine_counts)} unique types")

    # Show top 15 fine-grained types
    print(f"\n  Top fine-grained predictions (tiered):")
    sorted_fine = sorted(tiered_fine_counts.items(), key=lambda x: -x[1])[:15]
    for label, count in sorted_fine:
        broad = map_to_broad(label)
        medium = _map_to_medium(label, broad)
        pct = count / len(tiered_fine) * 100
        print(f"    {label:<35} {count:>6} ({pct:>5.1f}%)  [{medium}]")

    # ── High-confidence stats ──
    print(f"\n  ── Confidence Stats ──")
    flat_hc = flat_df["is_high_confidence"].sum()
    tiered_hc = tiered_df["is_high_confidence"].sum()
    print(f"  Flat HC:   {flat_hc:>6,} / {len(flat_df):,} ({flat_hc/len(flat_df)*100:.1f}%)")
    print(f"  Tiered HC: {tiered_hc:>6,} / {len(tiered_df):,} ({tiered_hc/len(tiered_df)*100:.1f}%)")

    flat_agree = flat_df["n_models_agree"].mean()
    tiered_agree = tiered_df["n_models_agree"].mean()
    print(f"  Flat mean agreement:   {flat_agree:.2f}")
    print(f"  Tiered mean agreement: {tiered_agree:.2f}")

    flat_conf = flat_df["consensus_confidence"].mean()
    tiered_conf = tiered_df["consensus_confidence"].mean()
    print(f"  Flat mean confidence:   {flat_conf:.3f}")
    print(f"  Tiered mean confidence: {tiered_conf:.3f}")

    return {
        "name": name,
        "n_cells": len(gt_labels),
        "flat_broad": flat_broad_m,
        "tiered_broad": tiered_broad_m,
        "flat_medium": flat_medium_m,
        "tiered_medium": tiered_medium_m,
        "flat_hc_pct": flat_hc / len(flat_df) * 100,
        "tiered_hc_pct": tiered_hc / len(tiered_df) * 100,
    }


def main():
    datasets = [
        ("xenium-breast-tumor-rep1", "rep1",
         Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1")),
        ("xenium-breast-tumor-rep2", "rep2",
         Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2")),
    ]

    results = []
    for name, rep, data_path in datasets:
        if not data_path.exists():
            print(f"SKIP: {name} not found at {data_path}")
            continue

        gt_dict = load_ground_truth(data_path, rep=rep)
        if not gt_dict:
            print(f"SKIP: {name} has no ground truth")
            continue

        result = run_dataset(name, data_path, gt_dict, sample_size=20000)
        if result:
            results.append(result)

    # ── Final Summary ──
    if results:
        print(f"\n\n{'='*80}")
        print(f"  FINAL SUMMARY: Flat vs Tiered Consensus")
        print(f"{'='*80}")
        print(f"\n  {'Dataset':<30} {'Level':<10} {'Flat Acc':>10} {'Tier Acc':>10} {'Flat F1':>10} {'Tier F1':>10} {'Delta':>8}")
        print(f"  {'-'*90}")

        for r in results:
            for level in ["broad", "medium"]:
                flat = r[f"flat_{level}"]
                tiered = r[f"tiered_{level}"]
                delta_f1 = tiered["macro_f1"] - flat["macro_f1"]
                arrow = "▲" if delta_f1 > 0.005 else ("▼" if delta_f1 < -0.005 else "─")
                print(
                    f"  {r['name']:<30} {level:<10} "
                    f"{flat['accuracy']:>9.1%} {tiered['accuracy']:>9.1%} "
                    f"{flat['macro_f1']:>9.3f} {tiered['macro_f1']:>9.3f} "
                    f"{arrow}{abs(delta_f1):>+6.3f}"
                )

        print(f"\n  {'Dataset':<30} {'Flat HC%':>10} {'Tier HC%':>10}")
        print(f"  {'-'*52}")
        for r in results:
            print(f"  {r['name']:<30} {r['flat_hc_pct']:>9.1f}% {r['tiered_hc_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
