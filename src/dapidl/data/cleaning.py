"""Data cleaning utilities for DAPIDL datasets.

Implements spatial consistency filtering to remove likely annotation errors
based on the principle that cells of the same type tend to cluster spatially.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import KDTree


# Cell types that are legitimately sparse/isolated and should be exempt from filtering
# These are often immune cells infiltrating epithelial or stromal regions
DEFAULT_SOLITARY_EXEMPT_TYPES = frozenset({
    # Dendritic cells (antigen presenting, highly motile)
    "Dendritic_Cell",
    "Dendritic",
    "DC",
    "DCs",
    "pDC",
    "LAMP3",
    "IRF7",
    # Mast cells (tissue-resident, scattered in connective tissue)
    "Mast_Cell",
    "Mast",
    # NK cells (tissue patrolling)
    "NK_Cell",
    "NK",
    # T cells (tumor-infiltrating lymphocytes are scattered throughout)
    "T_Cell",
    "T_Cells",
    "CD4",
    "CD8",
    "Treg",
    # Macrophages (tissue-resident and tumor-associated, often scattered)
    "Macrophage",
    "Macrophages",
    "Monocyte",
    # B cells can also be scattered in some tissues
    "B_Cell",
    "B_Cells",
    "Plasma",
    # Endothelial cells (line blood vessels, scattered throughout tissue)
    "Endothelial",
    # Rare stromal subtypes
    "Pericyte",
    "Adipocyte",
    # DCIS subtypes (often scattered in breast)
    "DCIS",
})


def compute_spatial_coherence(
    cell_coordinates: np.ndarray,
    cell_labels: np.ndarray,
    k: int = 20,
    exclude_self: bool = True,
) -> np.ndarray:
    """Compute spatial coherence score for each cell.

    Spatial coherence measures what fraction of a cell's k nearest neighbors
    share the same label. High coherence (>0.5) suggests reliable annotation.
    Low coherence (<0.2) suggests potential segmentation or annotation error.

    Args:
        cell_coordinates: (N, 2) array of cell centroid (x, y) positions
        cell_labels: (N,) array of integer cell type labels
        k: Number of nearest neighbors to consider (default 20)
        exclude_self: Whether to exclude the cell itself from neighbors (default True)

    Returns:
        (N,) array of coherence scores in [0, 1]
    """
    n_cells = len(cell_coordinates)

    if n_cells < k + 1:
        logger.warning(
            f"Only {n_cells} cells available, less than k={k}+1. "
            "Returning all 1.0 coherence (no filtering)."
        )
        return np.ones(n_cells, dtype=np.float32)

    # Build KD-tree for efficient neighbor queries
    tree = KDTree(cell_coordinates)

    # Query k+1 neighbors (includes self if exclude_self=True)
    query_k = k + 1 if exclude_self else k
    _, neighbor_indices = tree.query(cell_coordinates, k=query_k)

    # Compute coherence for each cell
    coherence = np.zeros(n_cells, dtype=np.float32)

    for i in range(n_cells):
        neighbors = neighbor_indices[i]

        if exclude_self:
            # Remove self (first neighbor is always self with distance 0)
            neighbors = neighbors[1:]

        # Count neighbors with matching label
        neighbor_labels = cell_labels[neighbors]
        matching = np.sum(neighbor_labels == cell_labels[i])
        coherence[i] = matching / k

    return coherence


def filter_spatially_inconsistent(
    metadata: pl.DataFrame,
    cell_coordinates: np.ndarray,
    cell_labels: np.ndarray,
    coherence_threshold: float = 0.20,
    k: int = 20,
    solitary_exempt_types: Optional[frozenset[str]] = None,
    label_to_name: Optional[dict[int, str]] = None,
) -> tuple[np.ndarray, dict]:
    """Filter cells with low spatial coherence.

    Removes cells whose labels disagree with their neighbors, likely indicating
    segmentation artifacts or annotation noise. Exempts certain cell types that
    are legitimately isolated (e.g., dendritic cells, mast cells).

    Args:
        metadata: Polars DataFrame with cell metadata (must have 'cell_id' column)
        cell_coordinates: (N, 2) array of cell centroid (x, y) positions
        cell_labels: (N,) array of integer cell type labels
        coherence_threshold: Minimum coherence to keep (default 0.20 = 20% neighbor agreement)
        k: Number of neighbors for coherence computation (default 20)
        solitary_exempt_types: Set of cell type names to exempt from filtering
                               (default: common immune patrol cells)
        label_to_name: Mapping from label index to cell type name (for exemption)

    Returns:
        Tuple of:
        - Boolean mask array (True = keep, False = filter out)
        - Statistics dictionary with filtering info
    """
    if solitary_exempt_types is None:
        solitary_exempt_types = DEFAULT_SOLITARY_EXEMPT_TYPES

    n_cells = len(cell_labels)
    logger.info(f"Computing spatial coherence for {n_cells} cells with k={k}...")

    # Compute coherence scores
    coherence = compute_spatial_coherence(cell_coordinates, cell_labels, k=k)

    # Create initial mask based on threshold
    keep_mask = coherence >= coherence_threshold

    # Identify exempt cells (solitary cell types that shouldn't be filtered)
    n_exempt = 0
    if label_to_name is not None:
        exempt_labels = set()
        for label_idx, name in label_to_name.items():
            # Normalize name for matching (handle "T cells" vs "T_Cells" etc.)
            name_normalized = name.upper().replace("_", " ").replace("-", " ")
            for exempt_type in solitary_exempt_types:
                exempt_normalized = exempt_type.upper().replace("_", " ").replace("-", " ")
                # Check if exempt pattern is in the normalized name
                if exempt_normalized in name_normalized:
                    exempt_labels.add(label_idx)
                    break
                # Also check individual words for flexibility
                name_words = set(name_normalized.split())
                exempt_words = set(exempt_normalized.split())
                # If all exempt words appear in name, it's a match
                if exempt_words and exempt_words.issubset(name_words):
                    exempt_labels.add(label_idx)
                    break

        if exempt_labels:
            logger.info(
                f"Exempting {len(exempt_labels)} cell types from filtering: "
                f"{[label_to_name[l] for l in exempt_labels]}"
            )
            # Restore cells with exempt labels even if low coherence
            for label_idx in exempt_labels:
                exempt_mask = (cell_labels == label_idx) & (~keep_mask)
                n_exempt += np.sum(exempt_mask)
                keep_mask[exempt_mask] = True

    # Compute statistics
    n_filtered = n_cells - np.sum(keep_mask)
    stats = {
        "n_cells_original": n_cells,
        "n_cells_kept": int(np.sum(keep_mask)),
        "n_cells_filtered": n_filtered,
        "n_cells_exempt": n_exempt,
        "filter_rate": n_filtered / n_cells if n_cells > 0 else 0.0,
        "coherence_threshold": coherence_threshold,
        "k_neighbors": k,
        "coherence_mean": float(np.mean(coherence)),
        "coherence_std": float(np.std(coherence)),
        "coherence_percentiles": {
            "p5": float(np.percentile(coherence, 5)),
            "p25": float(np.percentile(coherence, 25)),
            "p50": float(np.percentile(coherence, 50)),
            "p75": float(np.percentile(coherence, 75)),
            "p95": float(np.percentile(coherence, 95)),
        },
    }

    # Per-class filtering breakdown
    class_stats = {}
    unique_labels = np.unique(cell_labels)
    for label in unique_labels:
        class_mask = cell_labels == label
        class_keep = keep_mask[class_mask]
        class_name = label_to_name.get(label, f"class_{label}") if label_to_name else f"class_{label}"
        n_class = np.sum(class_mask)
        n_class_kept = np.sum(class_keep)
        class_stats[class_name] = {
            "original": int(n_class),
            "kept": int(n_class_kept),
            "filtered": int(n_class - n_class_kept),
            "keep_rate": n_class_kept / n_class if n_class > 0 else 0.0,
            "mean_coherence": float(np.mean(coherence[class_mask])),
        }

    stats["per_class"] = class_stats

    logger.info(
        f"Spatial consistency filtering: "
        f"kept {stats['n_cells_kept']}/{n_cells} cells "
        f"({stats['filter_rate']*100:.1f}% filtered, {n_exempt} exempt)"
    )

    return keep_mask, stats


def clean_dataset_spatial(
    data_path: Path,
    output_path: Optional[Path] = None,
    coherence_threshold: float = 0.20,
    k: int = 20,
    solitary_exempt_types: Optional[frozenset[str]] = None,
) -> dict:
    """Apply spatial consistency filtering to a dataset.

    Loads metadata and coordinates, filters inconsistent cells, and saves
    cleaned dataset to output path.

    Args:
        data_path: Path to dataset directory (must have metadata.parquet)
        output_path: Output path for cleaned dataset (default: {data_path}_cleaned)
        coherence_threshold: Minimum coherence to keep
        k: Number of neighbors for coherence
        solitary_exempt_types: Cell types to exempt from filtering

    Returns:
        Statistics dictionary from filtering
    """
    import json
    import shutil
    import zarr

    data_path = Path(data_path)
    if output_path is None:
        output_path = data_path.parent / f"{data_path.name}_cleaned"

    logger.info(f"Cleaning dataset: {data_path} -> {output_path}")

    # Load metadata
    metadata = pl.read_parquet(data_path / "metadata.parquet")

    # Load labels and class mapping
    labels = np.load(data_path / "labels.npy")
    with open(data_path / "class_mapping.json") as f:
        class_mapping = json.load(f)
    label_to_name = {v: k for k, v in class_mapping.items()}

    # Extract coordinates from metadata
    # Expect 'x_centroid' and 'y_centroid' or similar columns
    x_col = None
    y_col = None
    for col in metadata.columns:
        if "x" in col.lower() and "centroid" in col.lower():
            x_col = col
        if "y" in col.lower() and "centroid" in col.lower():
            y_col = col

    if x_col is None or y_col is None:
        # Try alternate column names
        for col in metadata.columns:
            if col.lower() in ("x", "x_location"):
                x_col = col
            if col.lower() in ("y", "y_location"):
                y_col = col

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not find coordinate columns in metadata. "
            f"Available columns: {metadata.columns}"
        )

    coordinates = np.column_stack([
        metadata[x_col].to_numpy(),
        metadata[y_col].to_numpy(),
    ])

    # Filter
    keep_mask, stats = filter_spatially_inconsistent(
        metadata=metadata,
        cell_coordinates=coordinates,
        cell_labels=labels,
        coherence_threshold=coherence_threshold,
        k=k,
        solitary_exempt_types=solitary_exempt_types,
        label_to_name=label_to_name,
    )

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter and save metadata
    keep_indices = np.where(keep_mask)[0]
    filtered_metadata = metadata[keep_indices.tolist()]
    filtered_metadata.write_parquet(output_path / "metadata.parquet")

    # Filter and save labels
    filtered_labels = labels[keep_mask]
    np.save(output_path / "labels.npy", filtered_labels)

    # Filter and save patches (Zarr)
    src_patches = zarr.open(data_path / "patches.zarr", mode="r")
    patch_shape = src_patches.shape[1:]  # (H, W) or (H, W, C)

    dst_patches = zarr.open(
        output_path / "patches.zarr",
        mode="w",
        shape=(len(keep_indices), *patch_shape),
        dtype=src_patches.dtype,
        chunks=(1000, *patch_shape),
    )

    # Copy filtered patches in batches
    batch_size = 1000
    for i in range(0, len(keep_indices), batch_size):
        batch_indices = keep_indices[i : i + batch_size]
        dst_patches[i : i + len(batch_indices)] = np.array(
            [src_patches[int(idx)] for idx in batch_indices]
        )

    logger.info(f"Saved {len(keep_indices)} filtered patches to {output_path / 'patches.zarr'}")

    # Copy class mapping (unchanged)
    shutil.copy(data_path / "class_mapping.json", output_path / "class_mapping.json")

    # Save filtering stats
    with open(output_path / "spatial_filtering_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Copy normalization stats if they exist
    norm_stats_file = data_path / "normalization_stats.json"
    if norm_stats_file.exists():
        shutil.copy(norm_stats_file, output_path / "normalization_stats.json")

    logger.info(f"Dataset cleaned and saved to {output_path}")

    return stats
