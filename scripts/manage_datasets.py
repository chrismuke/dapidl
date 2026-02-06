#!/usr/bin/env python
"""Comprehensive dataset management for DAPIDL.

This script handles:
1. Discovery of all local datasets (raw and derived)
2. Annotation status audit
3. S3 upload with progress tracking
4. ClearML registration with metadata
5. Ground truth validation

Usage:
    # Discover and audit all datasets
    uv run python scripts/manage_datasets.py audit

    # Upload all derived datasets to S3
    uv run python scripts/manage_datasets.py upload --to-s3

    # Register all datasets with ClearML
    uv run python scripts/manage_datasets.py register --clearml

    # Full pipeline: audit + upload + register
    uv run python scripts/manage_datasets.py full

    # Download SPATCH datasets (requires manual GSA-Human registration)
    uv run python scripts/manage_datasets.py download-spatch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

RAW_XENIUM_DIR = Path.home() / "datasets" / "raw" / "xenium"
RAW_MERSCOPE_DIR = Path.home() / "datasets" / "raw" / "merscope"
DERIVED_DIR = Path.home() / "datasets" / "derived"

# S3 Configuration
S3_BUCKET = "dapidl"
S3_ENDPOINT = ""
S3_REGION = "eu-central-1"

# Datasets with manually supervised ground truth (expert annotations)
GOLD_STANDARD_DATASETS = {
    "breast_tumor_rep1": {
        "gt_file": "celltypes_ground_truth_rep1_supervised.xlsx",
        "source": "Janesick et al. 2023",
        "cells": 167781,
    },
    "breast_tumor_rep2": {
        "gt_file": "celltypes_ground_truth_rep2_supervised.xlsx",
        "source": "Janesick et al. 2023",
        "cells": 118752,
    },
}

# SPATCH datasets (require GSA-Human registration)
SPATCH_DATASETS = {
    "COAD": {
        "tissue": "colon_adenocarcinoma",
        "bioimage_url": "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/900/S-BIAD1900/Files/web/xenium/COAD/",
        "files": ["xenium_COAD_DAPI.tif", "xenium_COAD_CODEX.tif", "xenium_COAD_HE.tif"],
        "gsa_accession": "HRA011129",
    },
    "HCC": {
        "tissue": "hepatocellular_carcinoma",
        "bioimage_url": "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/900/S-BIAD1900/Files/web/xenium/HCC/",
        "files": ["xenium_HCC_DAPI.tif", "xenium_HCC_CODEX.tif", "xenium_HCC_HE.tif"],
        "gsa_accession": "HRA011129",
    },
    "OV": {
        "tissue": "ovarian_cancer",
        "bioimage_url": "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/900/S-BIAD1900/Files/web/xenium/OV/",
        "files": ["xenium_OV_DAPI.tif", "xenium_OV_CODEX.tif", "xenium_OV_HE.tif"],
        "gsa_accession": "HRA011129",
    },
}


# =============================================================================
# Dataset Discovery
# =============================================================================


def discover_raw_datasets() -> list[dict[str, Any]]:
    """Discover all raw Xenium and MERSCOPE datasets."""
    datasets = []

    # Xenium datasets
    if RAW_XENIUM_DIR.exists():
        for dataset_dir in RAW_XENIUM_DIR.iterdir():
            if not dataset_dir.is_dir():
                continue

            info = {
                "name": dataset_dir.name,
                "platform": "xenium",
                "path": str(dataset_dir),
                "has_manual_gt": dataset_dir.name in GOLD_STANDARD_DATASETS,
            }

            # Check for pipeline outputs
            pipeline_dir = dataset_dir / "pipeline_outputs"
            outs_pipeline_dir = dataset_dir / "outs" / "pipeline_outputs"

            for pd in [pipeline_dir, outs_pipeline_dir]:
                if pd.exists():
                    info["pipeline_annotation"] = (
                        pd / "annotation" / "annotations.parquet"
                    ).exists()
                    info["ensemble_annotation"] = (
                        pd / "ensemble_annotation" / "annotations.parquet"
                    ).exists()
                    info["cl_standardization"] = (
                        pd / "cl_standardization" / "cl_annotations.parquet"
                    ).exists()
                    break
            else:
                info["pipeline_annotation"] = False
                info["ensemble_annotation"] = False
                info["cl_standardization"] = False

            # Check for ground truth files
            gt_files = list(dataset_dir.glob("*supervised*.xlsx")) + list(
                dataset_dir.glob("*ground_truth*.csv")
            )
            info["gt_files"] = [str(f.name) for f in gt_files]

            datasets.append(info)

    # MERSCOPE datasets
    if RAW_MERSCOPE_DIR.exists():
        for dataset_dir in RAW_MERSCOPE_DIR.iterdir():
            if not dataset_dir.is_dir():
                continue

            info = {
                "name": dataset_dir.name,
                "platform": "merscope",
                "path": str(dataset_dir),
                "has_manual_gt": False,
            }

            pipeline_dir = dataset_dir / "pipeline_outputs"
            if pipeline_dir.exists():
                info["pipeline_annotation"] = (
                    pipeline_dir / "annotation" / "annotations.parquet"
                ).exists()
            else:
                info["pipeline_annotation"] = False

            info["ensemble_annotation"] = False
            info["cl_standardization"] = False
            info["gt_files"] = []

            datasets.append(info)

    return datasets


def discover_derived_datasets() -> list[dict[str, Any]]:
    """Discover all derived LMDB datasets."""
    datasets = []

    if not DERIVED_DIR.exists():
        return datasets

    for dataset_dir in DERIVED_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue

        lmdb_dir = dataset_dir / "patches.lmdb"
        if not lmdb_dir.exists():
            continue

        info = {
            "name": dataset_dir.name,
            "path": str(dataset_dir),
            "has_lmdb": True,
        }

        # Parse name components
        parts = dataset_dir.name.split("-")
        if len(parts) >= 4:
            info["platform"] = parts[0]
            info["tissue"] = parts[1]
            info["granularity"] = "finegrained" if "finegrained" in dataset_dir.name else "coarse"
            info["patch_size"] = (
                int(parts[-1].replace("p", "")) if parts[-1].startswith("p") else None
            )

        # Check for metadata
        metadata_path = dataset_dir / "metadata.parquet"
        labels_path = dataset_dir / "labels.npy"
        class_mapping_path = dataset_dir / "class_mapping.json"

        info["has_metadata"] = metadata_path.exists()
        info["has_labels"] = labels_path.exists()
        info["has_class_mapping"] = class_mapping_path.exists()

        # Get size
        size_bytes = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
        info["size_gb"] = round(size_bytes / (1024**3), 2)

        # Get sample count from metadata
        if metadata_path.exists():
            try:
                meta_df = pl.read_parquet(metadata_path)
                info["n_samples"] = len(meta_df)
            except Exception:
                info["n_samples"] = None
        else:
            info["n_samples"] = None

        # Get class mapping
        if class_mapping_path.exists():
            try:
                with open(class_mapping_path) as f:
                    info["n_classes"] = len(json.load(f))
            except Exception:
                info["n_classes"] = None
        else:
            info["n_classes"] = None

        datasets.append(info)

    return sorted(datasets, key=lambda x: x["name"])


# =============================================================================
# S3 Operations
# =============================================================================


def upload_to_s3(local_path: Path, s3_path: str, dry_run: bool = False) -> bool:
    """Upload a directory to S3."""
    if dry_run:
        logger.info(f"[DRY RUN] Would upload {local_path} to s3://{S3_BUCKET}/{s3_path}")
        return True

    try:
        from dapidl.utils.s3 import upload_to_s3 as _upload

        _upload(local_path, s3_path, delete_local=False)
        logger.info(f"Uploaded {local_path.name} to S3")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False


def upload_derived_datasets(dry_run: bool = False) -> dict[str, bool]:
    """Upload all derived LMDB datasets to S3."""
    datasets = discover_derived_datasets()
    results = {}

    for ds in datasets:
        local_path = Path(ds["path"])
        s3_path = f"datasets/derived/{ds['name']}"

        results[ds["name"]] = upload_to_s3(local_path, s3_path, dry_run=dry_run)

    return results


# =============================================================================
# ClearML Registration
# =============================================================================


def register_with_clearml(
    dataset_name: str,
    dataset_path: Path,
    s3_uri: str | None = None,
    metadata: dict | None = None,
    dry_run: bool = False,
) -> str | None:
    """Register a dataset with ClearML."""
    if dry_run:
        logger.info(f"[DRY RUN] Would register {dataset_name} with ClearML")
        return "dry-run-id"

    try:
        from clearml import Dataset

        # Determine tags
        tags = ["auto-registered", "jan-2026"]
        if metadata:
            if metadata.get("granularity"):
                tags.append(metadata["granularity"])
            if metadata.get("platform"):
                tags.append(metadata["platform"])
            if metadata.get("tissue"):
                tags.append(metadata["tissue"])

        # Create dataset
        ds = Dataset.create(
            dataset_name=dataset_name,
            dataset_project="DAPIDL/Derived",
            dataset_tags=tags,
        )

        # Add metadata to description
        description = f"LMDB Dataset: {dataset_name}\n"
        if s3_uri:
            description += f"S3 URI: {s3_uri}\n"
        if metadata:
            description += f"\nMetadata:\n{json.dumps(metadata, indent=2)}"
        ds.set_description(description)

        # Finalize
        ds.finalize()

        logger.info(f"Registered {dataset_name} with ClearML: {ds.id}")
        return ds.id

    except Exception as e:
        logger.error(f"Failed to register {dataset_name} with ClearML: {e}")
        return None


def register_all_derived_datasets(dry_run: bool = False) -> dict[str, str | None]:
    """Register all derived datasets with ClearML."""
    datasets = discover_derived_datasets()
    results = {}

    for ds in datasets:
        metadata = {
            "platform": ds.get("platform"),
            "tissue": ds.get("tissue"),
            "granularity": ds.get("granularity"),
            "patch_size": ds.get("patch_size"),
            "n_samples": ds.get("n_samples"),
            "n_classes": ds.get("n_classes"),
            "size_gb": ds.get("size_gb"),
        }

        s3_uri = f"s3://{S3_BUCKET}/datasets/derived/{ds['name']}"

        results[ds["name"]] = register_with_clearml(
            dataset_name=ds["name"],
            dataset_path=Path(ds["path"]),
            s3_uri=s3_uri,
            metadata=metadata,
            dry_run=dry_run,
        )

    return results


# =============================================================================
# SPATCH Dataset Download
# =============================================================================


def download_spatch_images(output_dir: Path, dry_run: bool = False) -> None:
    """Download SPATCH benchmark images from BioImage Archive.

    Note: This only downloads the imaging data (DAPI, CODEX, H&E).
    The transcript/cell data requires manual registration at GSA-Human.
    """
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id, info in SPATCH_DATASETS.items():
        tissue_dir = output_dir / f"spatch_{dataset_id.lower()}"
        tissue_dir.mkdir(exist_ok=True)

        logger.info(f"Downloading SPATCH {dataset_id} ({info['tissue']})...")

        for filename in info["files"]:
            url = info["bioimage_url"] + filename
            output_file = tissue_dir / filename

            if output_file.exists():
                logger.info(f"  Skipping {filename} (already exists)")
                continue

            if dry_run:
                logger.info(f"  [DRY RUN] Would download {filename}")
                continue

            cmd = [
                "curl",
                "-sL",
                "-A",
                "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
                "-o",
                str(output_file),
                url,
            ]

            logger.info(f"  Downloading {filename}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"  Failed: {result.stderr}")

    logger.info(
        "\nNOTE: SPATCH transcript data requires manual registration at GSA-Human.\n"
        "1. Go to: https://ngdc.cncb.ac.cn/gsa-human/browse/HRA011129\n"
        "2. Register for access\n"
        "3. Download the Xenium raw data (transcripts.parquet, cells.parquet)\n"
        "4. Process using SPATCH scripts: https://github.com/zenglab-pku/SPATCH"
    )


# =============================================================================
# Audit Report
# =============================================================================


def generate_audit_report() -> str:
    """Generate a comprehensive audit report."""
    raw_datasets = discover_raw_datasets()
    derived_datasets = discover_derived_datasets()

    report = []
    report.append("=" * 80)
    report.append("DAPIDL DATASET AUDIT REPORT")
    report.append("=" * 80)
    report.append("")

    # Raw datasets summary
    report.append("## RAW DATASETS")
    report.append("")

    xenium_datasets = [d for d in raw_datasets if d["platform"] == "xenium"]
    merscope_datasets = [d for d in raw_datasets if d["platform"] == "merscope"]

    report.append(f"Total Xenium datasets: {len(xenium_datasets)}")
    report.append(f"Total MERSCOPE datasets: {len(merscope_datasets)}")
    report.append(f"Datasets with manual GT: {sum(1 for d in raw_datasets if d['has_manual_gt'])}")
    report.append(
        f"Datasets with pipeline annotation: {sum(1 for d in raw_datasets if d.get('pipeline_annotation'))}"
    )
    report.append(
        f"Datasets with CL standardization: {sum(1 for d in raw_datasets if d.get('cl_standardization'))}"
    )
    report.append("")

    # Detailed raw dataset table
    report.append("| Dataset | Platform | Manual GT | Pipeline | Ensemble | CL Std |")
    report.append("|---------|----------|-----------|----------|----------|--------|")
    for ds in sorted(raw_datasets, key=lambda x: x["name"]):
        report.append(
            f"| {ds['name'][:30]:<30} | {ds['platform']:<8} | "
            f"{'Yes' if ds['has_manual_gt'] else 'No':<9} | "
            f"{'Yes' if ds.get('pipeline_annotation') else 'No':<8} | "
            f"{'Yes' if ds.get('ensemble_annotation') else 'No':<8} | "
            f"{'Yes' if ds.get('cl_standardization') else 'No':<6} |"
        )
    report.append("")

    # Derived datasets summary
    report.append("## DERIVED LMDB DATASETS")
    report.append("")
    total_size = sum(d.get("size_gb", 0) for d in derived_datasets)
    report.append(f"Total datasets: {len(derived_datasets)}")
    report.append(f"Total storage: {total_size:.2f} GB")
    report.append("")

    # By patch size
    by_patch = {}
    for ds in derived_datasets:
        ps = ds.get("patch_size", "unknown")
        by_patch[ps] = by_patch.get(ps, 0) + 1
    report.append("By patch size:")
    for ps, count in sorted(by_patch.items()):
        report.append(f"  p{ps}: {count} datasets")
    report.append("")

    # SPATCH datasets
    report.append("## SPATCH BENCHMARK (External)")
    report.append("")
    report.append("Available via BioImage Archive S-BIAD1900:")
    for ds_id, info in SPATCH_DATASETS.items():
        report.append(f"  - {ds_id}: {info['tissue']}")
    report.append("")
    report.append("NOTE: Requires manual GSA-Human registration for transcript data")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DAPIDL Dataset Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit all datasets")
    audit_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload datasets to S3")
    upload_parser.add_argument("--to-s3", action="store_true", help="Upload to S3")
    upload_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register with ClearML")
    register_parser.add_argument("--clearml", action="store_true", help="Register with ClearML")
    register_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # Full command
    full_parser = subparsers.add_parser("full", help="Full pipeline: audit + upload + register")
    full_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # Download SPATCH
    spatch_parser = subparsers.add_parser("download-spatch", help="Download SPATCH images")
    spatch_parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "datasets" / "raw" / "spatch",
        help="Output directory",
    )
    spatch_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    args = parser.parse_args()

    if args.command == "audit":
        if args.json:
            raw = discover_raw_datasets()
            derived = discover_derived_datasets()
            print(json.dumps({"raw": raw, "derived": derived}, indent=2))
        else:
            print(generate_audit_report())

    elif args.command == "upload":
        if args.to_s3:
            results = upload_derived_datasets(dry_run=args.dry_run)
            success = sum(1 for v in results.values() if v)
            print(f"\nUploaded {success}/{len(results)} datasets to S3")

    elif args.command == "register":
        if args.clearml:
            results = register_all_derived_datasets(dry_run=args.dry_run)
            success = sum(1 for v in results.values() if v)
            print(f"\nRegistered {success}/{len(results)} datasets with ClearML")

    elif args.command == "full":
        print("=== AUDIT ===")
        print(generate_audit_report())

        print("\n=== UPLOAD TO S3 ===")
        upload_results = upload_derived_datasets(dry_run=args.dry_run)
        success = sum(1 for v in upload_results.values() if v)
        print(f"Uploaded {success}/{len(upload_results)} datasets")

        print("\n=== REGISTER WITH CLEARML ===")
        register_results = register_all_derived_datasets(dry_run=args.dry_run)
        success = sum(1 for v in register_results.values() if v)
        print(f"Registered {success}/{len(register_results)} datasets")

    elif args.command == "download-spatch":
        download_spatch_images(args.output, dry_run=args.dry_run)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
