#!/usr/bin/env python3
"""Convert NephNuc (BBBC051) dataset to DAPIDL LMDB format.

The NephNuc dataset provides:
- 242,594 kidney cell nuclei
- 8 cell types from different nephron segments
- 3D DAPI z-stacks (7 slices × 32×32 pixels)

This script converts to DAPIDL format with:
- 2D patches (max projection or central slice)
- Cell Ontology standardized labels
- LMDB storage compatible with MultiTissueDataset
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
import tifffile
from loguru import logger
from tqdm import tqdm

# Cell type mappings: NephNuc label -> (standardized name, CL ID)
NEPHNUC_TO_CL = {
    # Proximal tubule segments
    "S1S2": ("proximal_tubule_epithelial_cell", "CL:1000838"),
    "S2S3": ("proximal_tubule_epithelial_cell", "CL:1000838"),
    # Other tubule segments
    "TAL": ("thick_ascending_limb_epithelial_cell", "CL:1001107"),
    "DCT": ("distal_convoluted_tubule_epithelial_cell", "CL:1000849"),
    "CD": ("collecting_duct_epithelial_cell", "CL:1001225"),
    "CNT": ("connecting_tubule_epithelial_cell", "CL:1001106"),
    "CD_CNT": ("collecting_duct_epithelial_cell", "CL:1001225"),  # Combined
    # Glomerular cells
    "Nestin": ("podocyte", "CL:0000653"),
    # Endothelial cells
    "CD31_glom": ("glomerular_endothelial_cell", "CL:1001005"),
    "CD31glom": ("glomerular_endothelial_cell", "CL:1001005"),
    "CD31_inter": ("kidney_interstitial_endothelial_cell", "CL:0000115"),
    "CD31int": ("kidney_interstitial_endothelial_cell", "CL:0000115"),
    # Immune cells
    "CD45.0": ("leukocyte", "CL:0000738"),
    "CD45.1": ("leukocyte", "CL:0000738"),
    "CD45.2": ("leukocyte", "CL:0000738"),
}

# Coarse category mapping
COARSE_CATEGORIES = {
    "proximal_tubule_epithelial_cell": "Epithelial",
    "thick_ascending_limb_epithelial_cell": "Epithelial",
    "distal_convoluted_tubule_epithelial_cell": "Epithelial",
    "collecting_duct_epithelial_cell": "Epithelial",
    "connecting_tubule_epithelial_cell": "Epithelial",
    "podocyte": "Epithelial",  # Specialized epithelial
    "glomerular_endothelial_cell": "Vascular",
    "kidney_interstitial_endothelial_cell": "Vascular",
    "leukocyte": "Immune",
}


def extract_all_zips(images_dir: Path) -> None:
    """Extract all cell type zips if not already extracted."""
    for specimen_dir in images_dir.iterdir():
        if not specimen_dir.is_dir() or specimen_dir.name.startswith("_"):
            continue

        for zip_file in specimen_dir.glob("*.zip"):
            celltype = zip_file.stem
            extract_dir = specimen_dir / celltype
            if not extract_dir.exists():
                logger.info(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(specimen_dir)


def process_3d_to_2d(img_3d: np.ndarray, method: str = "max") -> np.ndarray:
    """Convert 3D z-stack to 2D image.

    Args:
        img_3d: 3D array of shape (Z, H, W)
        method: 'max' for maximum projection, 'central' for central slice

    Returns:
        2D array of shape (H, W)
    """
    if method == "max":
        return np.max(img_3d, axis=0)
    elif method == "central":
        central_idx = img_3d.shape[0] // 2
        return img_3d[central_idx]
    else:
        raise ValueError(f"Unknown method: {method}")


def resize_patch(patch: np.ndarray, target_size: int = 128) -> np.ndarray:
    """Resize patch to target size using bilinear interpolation."""
    from skimage.transform import resize

    if patch.shape[0] == target_size and patch.shape[1] == target_size:
        return patch

    # Resize maintaining uint8 range
    resized = resize(
        patch,
        (target_size, target_size),
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.asarray(resized, dtype=np.uint8)


def load_nuclei_from_specimen(
    specimen_dir: Path,
    projection_method: str = "max",
    target_size: int = 128,
) -> list[dict]:
    """Load all nuclei from a specimen directory.

    Returns list of dicts with:
        - patch: 2D numpy array
        - cell_type: standardized cell type name
        - cl_id: Cell Ontology ID
        - coarse: coarse category
        - specimen: specimen ID
        - original_file: original filename
    """
    nuclei = []

    for celltype_dir in specimen_dir.iterdir():
        if not celltype_dir.is_dir() or celltype_dir.name.startswith("_"):
            continue

        # The actual images are in <celltype>/<celltype>/All_3D/
        all_3d_dir = celltype_dir / celltype_dir.name / "All_3D"
        if not all_3d_dir.exists():
            # Try alternative structure
            all_3d_dir = celltype_dir / "All_3D"
            if not all_3d_dir.exists():
                logger.warning(f"No All_3D dir found in {celltype_dir}")
                continue

        celltype_name = celltype_dir.name
        if celltype_name not in NEPHNUC_TO_CL:
            logger.warning(f"Unknown cell type: {celltype_name}")
            continue

        cl_name, cl_id = NEPHNUC_TO_CL[celltype_name]
        coarse = COARSE_CATEGORIES.get(cl_name, "Other")

        tif_files = list(all_3d_dir.glob("*.tif"))
        logger.info(f"  {celltype_name}: {len(tif_files)} nuclei")

        for tif_file in tif_files:
            if tif_file.name.startswith("."):
                continue

            try:
                img_3d = tifffile.imread(tif_file)
                img_2d = process_3d_to_2d(img_3d, projection_method)
                patch = resize_patch(img_2d, target_size)

                nuclei.append({
                    "patch": patch,
                    "cell_type": cl_name,
                    "cl_id": cl_id,
                    "coarse": coarse,
                    "specimen": specimen_dir.name,
                    "original_file": tif_file.name,
                })
            except Exception as e:
                logger.warning(f"Failed to process {tif_file}: {e}")

    return nuclei


def create_lmdb_dataset(
    nuclei: list[dict],
    output_path: Path,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> None:
    """Create LMDB dataset from nuclei list.

    Creates structure compatible with DAPIDL MultiTissueDataset:
    - patches.lmdb: image patches
    - metadata.parquet: labels and metadata
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(nuclei))

    n_train = int(len(indices) * split_ratios[0])
    n_val = int(len(indices) * split_ratios[1])

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    logger.info(f"Split sizes: train={n_train}, val={n_val}, test={len(indices) - n_train - n_val}")

    # Create metadata
    metadata_rows = []
    for i, nucleus in enumerate(nuclei):
        split = "train" if i in splits["train"] else ("val" if i in splits["val"] else "test")
        metadata_rows.append({
            "cell_id": f"nephnuc_{i:06d}",
            "cell_type": nucleus["cell_type"],
            "cl_id": nucleus["cl_id"],
            "coarse_type": nucleus["coarse"],
            "specimen": nucleus["specimen"],
            "split": split,
            "confidence": 1.0,  # Ground truth
            "confidence_tier": 1,  # Highest tier
        })

    metadata_df = pl.DataFrame(metadata_rows)
    metadata_df.write_parquet(output_path / "metadata.parquet")

    # Create LMDB
    map_size = len(nuclei) * 128 * 128 * 10  # Generous estimate
    env = lmdb.open(str(output_path / "patches.lmdb"), map_size=map_size)

    with env.begin(write=True) as txn:
        for i, nucleus in enumerate(tqdm(nuclei, desc="Writing LMDB")):
            key = f"nephnuc_{i:06d}".encode()
            value = nucleus["patch"].tobytes()
            txn.put(key, value)

    env.close()

    # Write class mapping
    class_counts = metadata_df.group_by("cell_type").agg(pl.len().alias("count"))
    class_counts.write_csv(output_path / "class_counts.csv")

    coarse_counts = metadata_df.group_by("coarse_type").agg(pl.len().alias("count"))
    coarse_counts.write_csv(output_path / "coarse_counts.csv")

    logger.info(f"Created LMDB dataset at {output_path}")
    logger.info(f"Total nuclei: {len(nuclei)}")
    logger.info(f"Classes: {metadata_df['cell_type'].n_unique()}")
    logger.info(f"Coarse categories: {dict(coarse_counts.iter_rows())}")


def main():
    parser = argparse.ArgumentParser(description="Convert NephNuc to DAPIDL LMDB")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to NephNuc images directory (containing F33, F44, F59)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("~/datasets/derived/nephnuc-kidney-p128").expanduser(),
        help="Output LMDB directory",
    )
    parser.add_argument(
        "--projection",
        choices=["max", "central"],
        default="max",
        help="Method for 3D to 2D conversion",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Target patch size (default: 128)",
    )
    parser.add_argument(
        "--extract-zips",
        action="store_true",
        help="Extract zip files before processing",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.extract_zips:
        logger.info("Extracting zip files...")
        extract_all_zips(input_dir)

    # Process all specimens
    all_nuclei = []
    for specimen_dir in sorted(input_dir.iterdir()):
        if not specimen_dir.is_dir() or specimen_dir.name.startswith("_"):
            continue

        logger.info(f"Processing specimen: {specimen_dir.name}")
        nuclei = load_nuclei_from_specimen(
            specimen_dir,
            projection_method=args.projection,
            target_size=args.patch_size,
        )
        all_nuclei.extend(nuclei)
        logger.info(f"  Total: {len(nuclei)} nuclei")

    logger.info(f"Total nuclei across all specimens: {len(all_nuclei)}")

    # Create LMDB
    create_lmdb_dataset(all_nuclei, args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
