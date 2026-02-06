#!/usr/bin/env python3
"""Register ALL raw Xenium and MERSCOPE datasets to ClearML with S3 backing.

This script uploads raw spatial transcriptomics datasets to S3 and registers
them as ClearML datasets for the universal pipeline.
"""

from pathlib import Path
from clearml import Dataset
from loguru import logger

# S3 configuration (AWS S3)
S3_ENDPOINT = "s3://dapidl"

# Base paths
RAW_XENIUM = Path("~/datasets/raw/xenium").expanduser()
RAW_MERSCOPE = Path("~/datasets/raw/merscope").expanduser()

# Tissue to CellTypist model mapping (for documentation)
TISSUE_MODELS = {
    "breast": ["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
    "colon": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
    "colorectal": ["Cells_Intestinal_Tract.pkl", "Immune_All_High.pkl"],
    "heart": ["Healthy_Adult_Heart.pkl", "Immune_All_High.pkl"],
    "kidney": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "liver": ["Healthy_Human_Liver.pkl", "Immune_All_High.pkl"],
    "lung": ["Human_Lung_Atlas.pkl", "Cells_Lung_Airway.pkl", "Immune_All_High.pkl"],
    "lymph_node": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "ovary": ["Immune_All_High.pkl", "Immune_All_Low.pkl"],
    "pancreas": ["Adult_Human_PancreaticIslet.pkl", "Immune_All_High.pkl"],
    "skin": ["Adult_Human_Skin.pkl", "Immune_All_High.pkl"],
    "tonsil": ["Cells_Human_Tonsil.pkl", "Immune_All_High.pkl"],
}

# Dataset definitions
# Format: (directory_name, tissue_type, description_extra, tags_extra)
XENIUM_DATASETS = [
    # Breast (ground truth available for rep1)
    (
        "breast_tumor_rep1",
        "breast",
        "Ground truth available (Janesick et al. 2023)",
        ["ground-truth", "rep1"],
    ),
    ("breast_tumor_rep2", "breast", "Replicate 2 for cross-validation", ["rep2"]),
    # Colon
    (
        "colon_cancer_colon-panel",
        "colon",
        "Colon cancer with colon-specific gene panel",
        ["cancer", "colon-panel"],
    ),
    (
        "colon_normal_colon-panel",
        "colon",
        "Normal colon tissue with colon-specific gene panel",
        ["normal", "colon-panel"],
    ),
    (
        "colorectal_cancer_io-panel",
        "colorectal",
        "Colorectal cancer with immuno-oncology panel",
        ["cancer", "io-panel"],
    ),
    # Heart
    (
        "heart_normal_multi-tissue-panel",
        "heart",
        "Normal heart tissue with multi-tissue panel",
        ["normal", "multi-tissue-panel"],
    ),
    # Kidney
    (
        "kidney_cancer_multi-tissue-panel",
        "kidney",
        "Kidney cancer with multi-tissue panel",
        ["cancer", "multi-tissue-panel"],
    ),
    (
        "kidney_normal_multi-tissue-panel",
        "kidney",
        "Normal kidney tissue with multi-tissue panel",
        ["normal", "multi-tissue-panel"],
    ),
    # Liver
    (
        "liver_cancer_multi-tissue-panel",
        "liver",
        "Liver cancer with multi-tissue panel",
        ["cancer", "multi-tissue-panel"],
    ),
    (
        "liver_normal_multi-tissue-panel",
        "liver",
        "Normal liver tissue with multi-tissue panel",
        ["normal", "multi-tissue-panel"],
    ),
    # Lung
    ("lung_2fov", "lung", "Small 2-FOV test dataset", ["2fov", "test-dataset"]),
    (
        "lung_cancer_lung-panel",
        "lung",
        "Lung cancer with lung-specific gene panel",
        ["cancer", "lung-panel"],
    ),
    # Lymph node
    ("lymph_node_normal", "lymph_node", "Normal lymph node tissue", ["normal"]),
    # Ovary
    (
        "ovarian_cancer",
        "ovary",
        "Xenium Prime ovarian cancer with 5K gene panel",
        ["cancer", "xenium-prime", "5k-panel"],
    ),
    ("ovary_cancer_ff", "ovary", "Ovarian cancer fresh-frozen sample", ["cancer", "fresh-frozen"]),
    # Pancreas
    (
        "pancreas_cancer_multi-tissue-panel",
        "pancreas",
        "Pancreatic cancer with multi-tissue panel",
        ["cancer", "multi-tissue-panel"],
    ),
    # Skin
    ("skin_normal_sample1", "skin", "Normal skin sample 1", ["normal", "sample1"]),
    ("skin_normal_sample2", "skin", "Normal skin sample 2", ["normal", "sample2"]),
    # Tonsil
    (
        "tonsil_lymphoid-hyperplasia",
        "tonsil",
        "Tonsil with lymphoid hyperplasia",
        ["hyperplasia", "lymphoid"],
    ),
    (
        "tonsil_reactive-hyperplasia",
        "tonsil",
        "Tonsil with reactive hyperplasia",
        ["hyperplasia", "reactive"],
    ),
]

MERSCOPE_DATASETS = [
    ("breast", "breast", "MERSCOPE breast cancer with 500-gene panel", ["cancer"]),
]


def get_dataset_size(path: Path) -> str:
    """Calculate total dataset size."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size

    if total > 1e9:
        return f"{total / 1e9:.1f} GB"
    elif total > 1e6:
        return f"{total / 1e6:.1f} MB"
    else:
        return f"{total / 1e3:.1f} KB"


def count_cells(path: Path) -> int | None:
    """Count cells from cells.parquet if available."""
    try:
        import polars as pl

        cells_path = path / "cells.parquet"
        if not cells_path.exists():
            # Check outs subdirectory
            cells_path = path / "outs" / "cells.parquet"
        if cells_path.exists():
            df = pl.read_parquet(cells_path)
            return len(df)
    except Exception:
        pass
    return None


def check_if_registered(name: str, project: str) -> bool:
    """Check if dataset is already registered in ClearML."""
    try:
        datasets = Dataset.list_datasets(
            dataset_project=project,
            partial_name=name,
            only_completed=False,
        )
        for ds in datasets:
            if ds.get("name") == name:
                logger.info(f"Dataset already registered: {name} (ID: {ds.get('id')})")
                return True
    except Exception as e:
        logger.warning(f"Error checking for existing dataset: {e}")
    return False


def upload_dataset(
    name: str,
    path: Path,
    platform: str,
    tissue: str,
    description: str,
    tags: list[str],
    project: str = "DAPIDL/raw-data",
    skip_existing: bool = True,
) -> str | None:
    """Upload a single dataset to ClearML."""

    # Check if already registered
    if skip_existing and check_if_registered(name, project):
        return None

    size = get_dataset_size(path)
    cells = count_cells(path)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Uploading: {name}")
    logger.info(f"Path: {path}")
    logger.info(f"Size: {size}")
    if cells:
        logger.info(f"Cells: {cells:,}")
    logger.info(f"{'=' * 60}")

    # Build full description
    models = TISSUE_MODELS.get(tissue, ["Immune_All_High.pkl"])
    full_description = f"""Raw {platform.upper()} {tissue} dataset

Platform: {platform.upper()}
Tissue: {tissue}
Size: {size}
{f"Cells: {cells:,}" if cells else ""}

{description}

Recommended CellTypist models: {", ".join(models)}

Contents:
- morphology.ome.tif: DAPI nuclear staining images
- cells.parquet: Cell metadata and coordinates
- cell_feature_matrix.h5: Gene expression matrix
- transcripts.parquet: Transcript locations
"""

    # Create dataset
    dataset = Dataset.create(
        dataset_name=name,
        dataset_project=project,
        dataset_tags=["raw", platform, tissue] + tags,
        description=full_description,
        output_uri=S3_ENDPOINT,
    )

    logger.info(f"Created dataset ID: {dataset.id}")

    # Add files
    logger.info("Adding files...")
    dataset.add_files(str(path), verbose=True)

    # Upload to S3
    logger.info("Uploading to S3...")
    dataset.upload(
        output_url=S3_ENDPOINT,
        verbose=True,
    )

    # Finalize
    logger.info("Finalizing dataset...")
    dataset.finalize()

    logger.info(f"Dataset uploaded successfully: {name}")
    logger.info(f"ClearML URL: https://app.clear.ml/datasets/simple/{dataset.id}")

    return dataset.id


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Register raw datasets to ClearML")
    parser.add_argument(
        "--platform",
        choices=["xenium", "merscope", "all"],
        default="all",
        help="Platform to upload (default: all)",
    )
    parser.add_argument("--tissue", type=str, default=None, help="Only upload specific tissue type")
    parser.add_argument("--force", action="store_true", help="Re-upload even if dataset exists")
    parser.add_argument("--dry-run", action="store_true", help="List datasets without uploading")
    args = parser.parse_args()

    logger.info("Raw Dataset Registration to ClearML")
    logger.info(f"S3 Endpoint: {S3_ENDPOINT}")

    datasets_to_upload = []

    # Collect Xenium datasets
    if args.platform in ["xenium", "all"]:
        for dir_name, tissue, desc, tags in XENIUM_DATASETS:
            if args.tissue and tissue != args.tissue:
                continue

            path = RAW_XENIUM / dir_name
            if not path.exists():
                logger.warning(f"Xenium path not found: {path}")
                continue

            name = f"xenium-{dir_name.replace('_', '-')}-raw"
            datasets_to_upload.append((name, path, "xenium", tissue, desc, tags))

    # Collect MERSCOPE datasets
    if args.platform in ["merscope", "all"]:
        for dir_name, tissue, desc, tags in MERSCOPE_DATASETS:
            if args.tissue and tissue != args.tissue:
                continue

            path = RAW_MERSCOPE / dir_name
            if not path.exists():
                logger.warning(f"MERSCOPE path not found: {path}")
                continue

            name = f"merscope-{dir_name.replace('_', '-')}-raw"
            datasets_to_upload.append((name, path, "merscope", tissue, desc, tags))

    logger.info(f"\nFound {len(datasets_to_upload)} datasets to process")

    if args.dry_run:
        logger.info("\nDRY RUN - Datasets that would be uploaded:")
        for name, path, platform, tissue, desc, tags in datasets_to_upload:
            size = get_dataset_size(path)
            logger.info(f"  {name}: {path} ({size})")
        return

    # Upload datasets
    uploaded = 0
    skipped = 0
    failed = 0

    for name, path, platform, tissue, desc, tags in datasets_to_upload:
        try:
            result = upload_dataset(
                name=name,
                path=path,
                platform=platform,
                tissue=tissue,
                description=desc,
                tags=tags,
                skip_existing=not args.force,
            )
            if result:
                uploaded += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"Failed to upload {name}: {e}")
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info("Registration Complete!")
    logger.info(f"  Uploaded: {uploaded}")
    logger.info(f"  Skipped (existing): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
