"""Test auto CellTypist model selection across multiple datasets.

Verifies that:
1. Tissue detection from filesystem paths works correctly
2. Model selection returns the right models per tissue
3. Consensus annotation runs and produces quality metrics

Usage:
    uv run python scripts/test_auto_model_selection.py
    uv run python scripts/test_auto_model_selection.py --annotate  # also run CT annotation
    uv run python scripts/test_auto_model_selection.py --annotate --datasets breast lung liver
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dapidl.pipeline.components.annotators.auto_selector import (
    DATASET_MODELS,
    TISSUE_MODELS,
    _extract_dataset_name,
    get_models_for_dataset,
    get_tissue_for_dataset,
)


# ── Test 1: Path → Dataset Name Extraction ──────────────────────────────────

def test_name_extraction():
    """Test that dataset names are correctly extracted from various path formats."""
    print("\n" + "=" * 70)
    print("TEST 1: Dataset Name Extraction from Paths")
    print("=" * 70)

    test_cases = [
        # (input, expected_name)
        ("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/outs/", "xenium-breast-tumor-rep1"),
        ("/mnt/work/datasets/raw/xenium/xenium-lung-2fov", "xenium-lung-2fov"),
        ("/mnt/work/datasets/raw/merscope/merscope-breast", "merscope-breast"),
        ("xenium-liver-cancer", "xenium-liver-cancer"),
        ("/mnt/work/datasets/raw/xenium/xenium-heart-normal/outs", "xenium-heart-normal"),
        ("/some/path/xenium-colon-cancer/outs/", "xenium-colon-cancer"),
        ("breast_tumor_rep1", "breast_tumor_rep1"),
    ]

    passed = 0
    for input_path, expected in test_cases:
        result = _extract_dataset_name(input_path)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} '{input_path}' → '{result}' (expected: '{expected}')")

    print(f"\n  {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


# ── Test 2: Tissue Detection ────────────────────────────────────────────────

def test_tissue_detection():
    """Test tissue detection from paths and names."""
    print("\n" + "=" * 70)
    print("TEST 2: Tissue Detection from Paths")
    print("=" * 70)

    test_cases = [
        # (input, expected_tissue)
        ("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1/outs/", "breast"),
        ("/mnt/work/datasets/raw/xenium/xenium-lung-2fov", "lung"),
        ("/mnt/work/datasets/raw/xenium/xenium-liver-normal", "liver"),
        ("/mnt/work/datasets/raw/xenium/xenium-heart-normal/outs", "heart"),
        ("/mnt/work/datasets/raw/xenium/xenium-colon-cancer", "colon"),
        ("/mnt/work/datasets/raw/xenium/xenium-kidney-cancer", "kidney"),
        ("/mnt/work/datasets/raw/xenium/xenium-skin-normal-sample1", "skin"),
        ("/mnt/work/datasets/raw/xenium/xenium-tonsil-lymphoid", "tonsil"),
        ("/mnt/work/datasets/raw/xenium/xenium-pancreas-cancer", "pancreas"),
        ("/mnt/work/datasets/raw/xenium/xenium-brain-gbm", "brain"),
        ("/mnt/work/datasets/raw/xenium/xenium-ovarian-cancer", "ovary"),
        ("/mnt/work/datasets/raw/xenium/xenium-cervical-cancer-prime", "cervix"),
        ("/mnt/work/datasets/raw/xenium/xenium-lymph-node-normal", "lymph_node"),
        ("/mnt/work/datasets/raw/xenium/xenium-mouse-brain", "mouse_brain"),
        ("/mnt/work/datasets/raw/xenium/xenium-colorectal-cancer", "colorectal"),
        ("/mnt/work/datasets/raw/merscope/merscope-breast", "breast"),
        # Edge cases
        ("/some/random/path/unknown-tissue-sample1", "generic"),
        ("some-random-dataset", "generic"),
    ]

    passed = 0
    for input_path, expected in test_cases:
        result = get_tissue_for_dataset(input_path)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            print(f"  {status} '{input_path}' → '{result}' (expected: '{expected}')")

    # Only print failures (success is quiet)
    if passed == len(test_cases):
        print(f"  ✓ All {passed}/{len(test_cases)} passed")
    else:
        print(f"\n  {passed}/{len(test_cases)} passed")

    return passed == len(test_cases)


# ── Test 3: Model Selection ─────────────────────────────────────────────────

def test_model_selection():
    """Test that the right CellTypist models are selected per dataset."""
    print("\n" + "=" * 70)
    print("TEST 3: Model Selection per Dataset Path")
    print("=" * 70)

    # Scan all local datasets and check what models get selected
    local_roots = [
        Path("/mnt/work/datasets/raw/xenium"),
        Path("/mnt/work/datasets/raw/merscope"),
    ]

    all_ok = True
    for root in local_roots:
        if not root.exists():
            continue
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue

            # Simulate the full path the pipeline would see
            data_path = dataset_dir / "outs" if (dataset_dir / "outs").is_dir() else dataset_dir
            path_str = str(data_path)

            tissue = get_tissue_for_dataset(path_str)
            models = get_models_for_dataset(path_str)

            # Check if this dataset has a known mapping
            name = dataset_dir.name
            expected = DATASET_MODELS.get(name)

            if expected:
                match = models == expected["models"]
                status = "✓" if match else "✗"
                if not match:
                    all_ok = False
                    print(f"  {status} {name}: tissue={tissue}")
                    print(f"      got:      {models}")
                    print(f"      expected: {expected['models']}")
                else:
                    print(f"  {status} {name}: tissue={tissue} → {models}")
            else:
                print(f"  ? {name}: tissue={tissue} → {models} (no DATASET_MODELS entry)")

    return all_ok


# ── Test 4: Consensus Annotation with Auto-Selected Models ──────────────────

def test_consensus_annotation(datasets_filter=None):
    """Run consensus annotation on selected datasets and report metrics.

    Args:
        datasets_filter: List of tissue keywords to filter datasets (e.g., ["breast", "lung"])
    """
    import numpy as np

    print("\n" + "=" * 70)
    print("TEST 4: Consensus Annotation with Auto-Selected Models")
    print("=" * 70)

    # Select datasets to test
    test_datasets = []
    local_roots = [
        Path("/mnt/work/datasets/raw/xenium"),
        Path("/mnt/work/datasets/raw/merscope"),
    ]

    for root in local_roots:
        if not root.exists():
            continue
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            name = dataset_dir.name
            if datasets_filter:
                if not any(kw in name.lower() for kw in datasets_filter):
                    continue
            data_path = dataset_dir / "outs" if (dataset_dir / "outs").is_dir() else dataset_dir
            test_datasets.append((name, data_path))

    if not test_datasets:
        print("  No datasets matched filter. Available:")
        for root in local_roots:
            if root.exists():
                for d in sorted(root.iterdir()):
                    if d.is_dir():
                        print(f"    - {d.name}")
        return False

    print(f"\n  Testing {len(test_datasets)} datasets:\n")

    results = []
    for name, data_path in test_datasets:
        tissue = get_tissue_for_dataset(str(data_path))
        models = get_models_for_dataset(str(data_path))

        print(f"  ── {name} ──")
        print(f"  Tissue: {tissue}")
        print(f"  Models: {models}")

        try:
            result = _run_consensus_on_dataset(name, data_path, tissue, models)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "error": str(e)})
        print()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<35} {'Tissue':<12} {'Cells':>8} {'HC%':>6} {'Agree':>6} {'Epi':>6} {'Imm':>6} {'Str':>6}")
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<35} {'ERROR':<12} {r['error']}")
            continue

        print(
            f"{r['name']:<35} {r['tissue']:<12} "
            f"{r['n_cells']:>8,} "
            f"{r['pct_high_conf']:>5.1f}% "
            f"{r['mean_agreement']:>5.2f} "
            f"{r.get('pct_epithelial', 0):>5.1f}% "
            f"{r.get('pct_immune', 0):>5.1f}% "
            f"{r.get('pct_stromal', 0):>5.1f}%"
        )

    return True


def _run_consensus_on_dataset(name, data_path, tissue, models):
    """Run consensus annotation on a single dataset."""
    from dapidl.pipeline.components.annotators.auto_selector import AutoModelSelector

    # Load expression data
    adata = _load_expression(data_path)
    n_cells = len(adata)
    print(f"  Loaded: {n_cells:,} cells, {adata.n_vars:,} genes")

    # Sample if large
    sample_size = min(20_000, n_cells)
    if n_cells > sample_size:
        import numpy as np
        idx = np.random.choice(n_cells, sample_size, replace=False)
        adata_sample = adata[idx].copy()
        print(f"  Sampled: {sample_size:,} cells for speed")
    else:
        adata_sample = adata

    # Run consensus
    t0 = time.time()
    selector = AutoModelSelector(tissue_type=tissue, candidate_models=models)
    consensus = selector.build_consensus(
        adata_sample,
        models=models,
        min_agreement=0.5,
        confidence_weight=False,  # unweighted voting (popV-style)
    )
    elapsed = time.time() - t0

    df = consensus.annotations_df
    stats = consensus.stats

    # Compute metrics
    n_cells_annotated = len(df)
    n_high_conf = df.filter(
        (df["consensus_score"] >= 0.5) & (df["consensus_confidence"] >= 0.3)
    ).height
    pct_high_conf = n_high_conf / n_cells_annotated * 100

    # Agreement stats
    import numpy as np
    mean_agreement = df["n_models_agree"].mean()
    mean_confidence = df["consensus_confidence"].mean()

    # Broad category distribution
    broad_counts = dict(
        df.group_by("consensus_broad")
        .agg(pl.len().alias("count"))
        .iter_rows()
    )
    total = sum(broad_counts.values())

    pct_epi = broad_counts.get("Epithelial", 0) / total * 100
    pct_imm = broad_counts.get("Immune", 0) / total * 100
    pct_str = broad_counts.get("Stromal", 0) / total * 100
    pct_other = broad_counts.get("Other", 0) / total * 100

    print(f"  Time: {elapsed:.1f}s")
    print(f"  High-confidence: {n_high_conf:,}/{n_cells_annotated:,} ({pct_high_conf:.1f}%)")
    print(f"  Mean agreement: {mean_agreement:.2f} / {len(models)}")
    print(f"  Mean confidence: {mean_confidence:.3f}")
    print(f"  Distribution: Epi={pct_epi:.1f}% Imm={pct_imm:.1f}% Str={pct_str:.1f}% Other={pct_other:.1f}%")

    # Per-model breakdown
    for model in models:
        model_short = model.replace(".pkl", "")
        broad_col = f"{model_short}_broad"
        if broad_col in df.columns:
            model_counts = dict(
                df.group_by(broad_col)
                .agg(pl.len().alias("count"))
                .iter_rows()
            )
            print(f"  {model_short}: {model_counts}")

    return {
        "name": name,
        "tissue": tissue,
        "models": models,
        "n_cells": n_cells_annotated,
        "n_high_conf": n_high_conf,
        "pct_high_conf": pct_high_conf,
        "mean_agreement": mean_agreement,
        "mean_confidence": mean_confidence,
        "pct_epithelial": pct_epi,
        "pct_immune": pct_imm,
        "pct_stromal": pct_str,
        "pct_other": pct_other,
        "broad_counts": broad_counts,
        "elapsed": elapsed,
    }


def _load_expression(data_path):
    """Load expression data from a dataset path."""
    import anndata as ad
    import scanpy as sc

    # Xenium: cell_feature_matrix.h5 or cell_feature_matrix/
    h5_path = data_path / "cell_feature_matrix.h5"
    if h5_path.exists():
        return sc.read_10x_h5(h5_path)

    mtx_dir = data_path / "cell_feature_matrix"
    if mtx_dir.exists():
        return sc.read_10x_mtx(mtx_dir)

    # MERSCOPE: cell_by_gene.csv
    cbg = data_path / "cell_by_gene.csv"
    if cbg.exists():
        import pandas as pd
        import scipy.sparse as sp

        expr_df = pd.read_csv(cbg, index_col=0)
        X = sp.csr_matrix(expr_df.values)
        adata = ad.AnnData(X=X)
        adata.obs_names = expr_df.index.astype(str)
        adata.var_names = expr_df.columns
        adata.obs["cell_id"] = adata.obs_names
        return adata

    raise FileNotFoundError(f"No expression data found in {data_path}")


import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Test auto CellTypist model selection")
    parser.add_argument("--annotate", action="store_true", help="Run consensus annotation (slow)")
    parser.add_argument("--datasets", nargs="*", help="Filter datasets by tissue keyword")
    args = parser.parse_args()

    print("=" * 70)
    print("DAPIDL Auto CellTypist Model Selection Test")
    print("=" * 70)

    # Fast tests (no annotation)
    ok1 = test_name_extraction()
    ok2 = test_tissue_detection()
    ok3 = test_model_selection()

    if args.annotate:
        ok4 = test_consensus_annotation(datasets_filter=args.datasets)
    else:
        print("\n  (Skipping annotation test — use --annotate to run)")
        ok4 = True

    # Final status
    all_ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 70)
    if all_ok:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
