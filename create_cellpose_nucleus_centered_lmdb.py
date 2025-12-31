#!/usr/bin/env python3
"""
Create nucleus-centered LMDB datasets using Cellpose nucleus detection.

This script creates DALI-compatible LMDB datasets with patches centered on
Cellpose-detected nucleus centroids instead of Xenium cell centroids.

Key difference from cell-centered datasets:
- Cell-centered: Patches centered on Xenium transcript-based cell centroids
  (often off-center from nucleus, ~30% problematic)
- Nucleus-centered: Patches centered on Cellpose-detected nucleus centroids
  (always centered on nucleus)

Output: experiment_cellpose_p{size}/dataset/patches.lmdb/
"""

import argparse
import json
import lmdb
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


def load_dapi_image(xenium_path: Path) -> np.ndarray:
    """Load DAPI morphology image."""
    # Try multiple possible paths
    possible_paths = [
        xenium_path / "outs" / "morphology_focus.ome.tif",
        xenium_path / "outs" / "morphology_focus" / "morphology_focus_0000.ome.tif",
        xenium_path / "morphology_focus.ome.tif",
    ]

    morph_path = None
    for p in possible_paths:
        if p.exists():
            morph_path = p
            break

    if morph_path is None:
        raise FileNotFoundError(f"DAPI image not found. Tried: {possible_paths}")

    print(f"Loading DAPI image from {morph_path}")
    return tifffile.imread(str(morph_path))


def get_1to1_matches(matches_path: Path) -> pd.DataFrame:
    """Get nuclei that match exactly 1 Xenium cell (1:1 matches only)."""
    matches = pd.read_csv(matches_path)
    matched = matches[matches['matched'] == True]

    # Count cells per nucleus
    cells_per_nucleus = matched.groupby('cellpose_id').size()
    unique_nuclei = cells_per_nucleus[cells_per_nucleus == 1].index

    one_to_one = matched[matched['cellpose_id'].isin(unique_nuclei)]
    print(f"Total matched pairs: {len(matched)}")
    print(f"1:1 nucleus-cell matches: {len(one_to_one)}")

    return one_to_one


def load_ground_truth(dataset_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load ground truth metadata and class mapping."""
    metadata = pd.read_parquet(dataset_path / "metadata.parquet")
    with open(dataset_path / "class_mapping.json") as f:
        class_mapping = json.load(f)

    # Convert predicted_type to label index
    metadata['label'] = metadata['predicted_type'].map(class_mapping)

    return metadata, class_mapping


def load_nucleus_centroids(nuclei_path: Path) -> pd.DataFrame:
    """Load Cellpose nucleus centroid coordinates."""
    nuclei = pd.read_csv(nuclei_path)
    # Columns: cellpose_id, centroid_x_um, centroid_y_um, centroid_x_px, centroid_y_px
    return nuclei


def extract_patch(image: np.ndarray, cx: float, cy: float, patch_size: int) -> np.ndarray:
    """Extract a patch centered on (cx, cy)."""
    h, w = image.shape
    half = patch_size // 2

    # Round to nearest pixel
    cx_int = int(round(cx))
    cy_int = int(round(cy))

    # Calculate bounds
    x1 = cx_int - half
    x2 = cx_int + half
    y1 = cy_int - half
    y2 = cy_int + half

    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None  # Out of bounds

    return image[y1:y2, x1:x2]


def compute_normalization_stats(
    image: np.ndarray,
    merged: pd.DataFrame,
    patch_size: int,
    n_samples: int = 10000
) -> Dict:
    """Compute normalization statistics from sampled patches."""
    print(f"Computing normalization stats from {n_samples} patches...")

    # Sample random indices
    indices = np.random.choice(len(merged), min(n_samples, len(merged)), replace=False)

    all_patches = []
    for idx in tqdm(indices, desc="Sampling patches"):
        row = merged.iloc[idx]
        patch = extract_patch(image, row['centroid_x_px'], row['centroid_y_px'], patch_size)
        if patch is not None:
            all_patches.append(patch)

    all_patches = np.stack(all_patches)

    # Compute percentile-based normalization
    p_low = float(np.percentile(all_patches, 1.0))
    p_high = float(np.percentile(all_patches, 99.5))

    # Normalize and compute mean/std
    normalized = np.clip((all_patches - p_low) / (p_high - p_low + 1e-8), 0, 1)
    mean = float(np.mean(normalized))
    std = float(np.std(normalized))

    return {
        "p_low": p_low,
        "p_high": p_high,
        "mean": mean,
        "std": std
    }


def create_lmdb_dataset(
    image: np.ndarray,
    merged: pd.DataFrame,
    output_dir: Path,
    patch_size: int,
    class_mapping: Dict[str, int]
):
    """Create LMDB dataset with nucleus-centered patches."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = output_dir / "patches.lmdb"

    # Estimate LMDB size (2 bytes per pixel + 8 bytes label, with generous overhead)
    n_samples = len(merged)
    bytes_per_sample = 8 + (patch_size * patch_size * 2)  # label + patch
    # LMDB needs significant overhead for B-tree structure, especially for small values
    map_size = max(int(n_samples * bytes_per_sample * 3), 500 * 1024 * 1024)  # 3x overhead, min 500MB

    print(f"Creating LMDB at {lmdb_path} (estimated size: {map_size / 1e9:.2f} GB)")

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True
    )

    valid_count = 0
    skipped_count = 0
    labels_list = []

    with env.begin(write=True) as txn:
        for idx, (_, row) in enumerate(tqdm(merged.iterrows(), total=len(merged), desc="Writing LMDB")):
            patch = extract_patch(image, row['centroid_x_px'], row['centroid_y_px'], patch_size)

            if patch is None:
                skipped_count += 1
                continue

            label = int(row['label'])
            labels_list.append(label)

            # Key: 8-byte big-endian integer
            key = struct.pack(">q", valid_count)

            # Value: 8-byte label + patch bytes
            label_bytes = struct.pack(">q", label)
            patch_bytes = patch.astype(np.uint16).tobytes()
            value = label_bytes + patch_bytes

            txn.put(key, value)
            valid_count += 1

    env.close()

    print(f"Written {valid_count} patches, skipped {skipped_count} (out of bounds)")

    # Save labels for stratified splitting
    labels_array = np.array(labels_list, dtype=np.int64)
    np.save(output_dir / "labels.npy", labels_array)

    # Save metadata (inside patches.lmdb for DALI loader compatibility)
    metadata = {
        "n_samples": valid_count,
        "patch_shape": [patch_size, patch_size],
        "dtype": "uint16",
        "format": "lmdb",
        "centered_on": "nucleus",
        "source": "cellpose"
    }
    with open(lmdb_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save class mapping (in both locations for compatibility)
    with open(lmdb_path / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    # Compute and save normalization stats (in both locations for compatibility)
    merged_valid = merged[merged.index.isin(merged.index[:valid_count + skipped_count])]
    norm_stats = compute_normalization_stats(image, merged, patch_size)
    with open(lmdb_path / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    with open(output_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    # Print class distribution
    print(f"\nClass distribution:")
    unique, counts = np.unique(labels_array, return_counts=True)
    inv_mapping = {v: k for k, v in class_mapping.items()}
    for label, count in zip(unique, counts):
        class_name = inv_mapping.get(label, f"Unknown_{label}")
        print(f"  {label}: {class_name}: {count:,}")

    return valid_count


def main():
    parser = argparse.ArgumentParser(description="Create nucleus-centered LMDB datasets")
    parser.add_argument(
        "--xenium-path",
        type=Path,
        default=Path("/home/chrism/datasets/xenium_breast_tumor"),
        help="Path to Xenium output directory"
    )
    parser.add_argument(
        "--matches-path",
        type=Path,
        default=Path("xenium_cellpose_matches_centroid.csv"),
        help="Path to Cellpose matches CSV"
    )
    parser.add_argument(
        "--nuclei-path",
        type=Path,
        default=Path("xenium_cellpose_nuclei.csv"),
        help="Path to Cellpose nuclei CSV"
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=Path("experiment_groundtruth_finegrained_p256/dataset"),
        help="Path to ground truth dataset"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("."),
        help="Base output directory"
    )
    parser.add_argument(
        "--patch-sizes",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256],
        help="Patch sizes to generate"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NUCLEUS-CENTERED LMDB DATASET CREATION")
    print("=" * 60)

    # Load DAPI image
    dapi = load_dapi_image(args.xenium_path)
    print(f"DAPI image shape: {dapi.shape}, dtype: {dapi.dtype}")

    # Load 1:1 matches
    one_to_one = get_1to1_matches(args.matches_path)

    # Load ground truth
    gt_metadata, class_mapping = load_ground_truth(args.ground_truth_path)
    print(f"Ground truth cells: {len(gt_metadata)}")
    print(f"Classes: {len(class_mapping)}")

    # Load nucleus centroids
    nuclei = load_nucleus_centroids(args.nuclei_path)
    print(f"Cellpose nuclei: {len(nuclei)}")

    # Merge all data
    # 1. Start with 1:1 matches (cellpose_id, cell_id)
    # 2. Join with ground truth (cell_id -> label)
    # 3. Join with nuclei (cellpose_id -> centroid_x_px, centroid_y_px)

    merged = one_to_one.merge(
        gt_metadata[['cell_id', 'label', 'predicted_type']],
        on='cell_id',
        how='inner'
    )
    merged = merged.merge(
        nuclei[['cellpose_id', 'centroid_x_px', 'centroid_y_px']],
        on='cellpose_id',
        how='inner',
        suffixes=('_cell', '_nucleus')
    )

    # Use nucleus centroids (not cell centroids!)
    merged = merged.rename(columns={
        'centroid_x_px_nucleus': 'centroid_x_px',
        'centroid_y_px_nucleus': 'centroid_y_px'
    })

    print(f"\nFinal merged dataset: {len(merged)} samples")

    # Remove rows with missing labels
    merged = merged.dropna(subset=['label'])
    merged['label'] = merged['label'].astype(int)
    print(f"After removing missing labels: {len(merged)} samples")

    # Create datasets for each patch size
    for patch_size in args.patch_sizes:
        print(f"\n{'=' * 60}")
        print(f"Creating {patch_size}x{patch_size} dataset")
        print("=" * 60)

        output_dir = args.output_base / f"experiment_cellpose_p{patch_size}" / "dataset"
        n_samples = create_lmdb_dataset(
            dapi, merged, output_dir, patch_size, class_mapping
        )

        print(f"Created {output_dir} with {n_samples} samples")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
