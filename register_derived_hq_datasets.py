#!/usr/bin/env python3
"""Register derived_hq datasets to ClearML using S3 storage.

This script:
1. Uploads derived_hq datasets to S3 (if not already uploaded)
2. Registers them in ClearML using external file references

S3 Configuration:
- Bucket: dapidl (AWS S3, eu-central-1)
- Path: datasets/derived_hq/
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from clearml import Dataset

# S3 configuration (AWS S3)
S3_BUCKET = "dapidl"
S3_REGION = "eu-central-1"
S3_BASE_PATH = "datasets/derived_hq"

# ClearML S3 URI format (for external files)
S3_CLEARML_BASE = f"s3://{S3_BUCKET}/{S3_BASE_PATH}"

# Local paths
DERIVED_HQ_PATH = Path("/mnt/work/datasets/derived_hq")

# AWS credentials - use the 'dapidl' profile from ~/.aws/credentials
AWS_PROFILE = "dapidl"

# Parent dataset mapping for lineage
PARENT_DATASETS = {
    "xenium-breast-rep1": "xenium-breast-cancer-rep1-raw",
    "xenium-breast-rep2": "xenium-breast-tumor-rep2-raw",
    "xenium-lung": "xenium-lung-2fov-raw",
    "xenium-ovarian": "xenium-ovarian-cancer-ffpe-raw",
    "xenium-spinal": None,  # No raw data uploaded yet
    "merscope-breast": "merscope-breast-raw",
    "combined": None,  # Combined from multiple sources
}


def get_parent_dataset_id(dataset_name: str) -> str | None:
    """Get the parent dataset ID for lineage tracking.

    Only returns finalized datasets to avoid ClearML parent errors.
    """
    # Extract platform prefix from name
    for prefix, parent_name in PARENT_DATASETS.items():
        if dataset_name.startswith(prefix):
            if parent_name:
                datasets = Dataset.list_datasets(dataset_project="DAPIDL", partial_name=parent_name)
                # Find first finalized dataset
                for d in datasets:
                    try:
                        ds = Dataset.get(dataset_id=d["id"])
                        if ds.is_final():
                            return d["id"]
                    except Exception:
                        continue
            return None
    return None


def parse_metadata(dataset_path: Path) -> dict:
    """Parse dataset metadata from metadata.json."""
    metadata_file = dataset_path / "metadata.json"
    if not metadata_file.exists():
        return {}

    with open(metadata_file) as f:
        metadata = json.load(f)

    return metadata


def extract_dataset_info(name: str, metadata: dict) -> dict:
    """Extract structured info from dataset name and metadata."""
    info = {
        "name": name,
        "platform": "unknown",
        "tissue": "unknown",
        "granularity": "unknown",
        "patch_size": 0,
        "n_samples": metadata.get("n_samples", 0),
        "annotation_method": "consensus",
    }

    # Parse name format: platform-tissue-consensus-granularity-pXXX
    parts = name.split("-")

    if parts[0] == "xenium":
        info["platform"] = "xenium"
        if "breast" in name:
            info["tissue"] = "breast"
        elif "lung" in name:
            info["tissue"] = "lung"
        elif "ovarian" in name:
            info["tissue"] = "ovarian"
        elif "spinal" in name:
            info["tissue"] = "spinal"
    elif parts[0] == "merscope":
        info["platform"] = "merscope"
        info["tissue"] = "breast"
    elif parts[0] == "combined":
        info["platform"] = "multi-platform"
        info["tissue"] = "multi-tissue"

    # Parse granularity
    if "coarse" in name:
        info["granularity"] = "coarse"
    elif "finegrained" in name:
        info["granularity"] = "finegrained"

    # Parse patch size
    for part in parts:
        if part.startswith("p") and part[1:].isdigit():
            info["patch_size"] = int(part[1:])

    return info


def generate_description(info: dict, metadata: dict) -> str:
    """Generate a detailed description for the dataset."""
    class_dist = metadata.get("class_distribution", {})
    class_mapping = metadata.get("class_mapping", metadata.get("index_to_class", {}))

    desc = f"""DAPIDL Derived Dataset: {info["name"]}

Platform: {info["platform"].upper()}
Tissue: {info["tissue"].title()}
Annotation Method: Multi-annotator consensus (CellTypist + scType + SingleR)
Granularity: {info["granularity"].title()} ({len(class_mapping)} classes)
Patch Size: {info["patch_size"]}x{info["patch_size"]} pixels
Total Samples: {info["n_samples"]:,}

Class Distribution:
"""
    for class_name, count in sorted(class_dist.items()):
        pct = count / info["n_samples"] * 100 if info["n_samples"] > 0 else 0
        desc += f"  - {class_name}: {count:,} ({pct:.1f}%)\n"

    desc += f"""
Format: LMDB (patches) + numpy (labels) + JSON (metadata)
Storage: S3 (AWS)
Generated: High-quality consensus annotations
"""
    return desc


def generate_tags(info: dict) -> list[str]:
    """Generate tags for the dataset."""
    tags = [
        info["platform"],
        info["tissue"],
        info["granularity"],
        f"p{info['patch_size']}",
        "consensus",
        "derived",
        "lmdb",
        "s3",
    ]

    # Add specific tags
    if info["platform"] == "multi-platform":
        tags.append("combined")
    if info["n_samples"] > 100000:
        tags.append("large-scale")

    return tags


def check_s3_exists(dataset_name: str) -> bool:
    """Check if dataset already exists on S3."""
    s3_path = f"s3://{S3_BUCKET}/{S3_BASE_PATH}/{dataset_name}/"

    cmd = [
        "aws",
        "s3",
        "ls",
        s3_path,
        "--region",
        S3_REGION,
        "--profile",
        AWS_PROFILE,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def upload_to_s3(local_path: Path, dataset_name: str) -> bool:
    """Upload dataset to S3."""
    s3_path = f"s3://{S3_BUCKET}/{S3_BASE_PATH}/{dataset_name}/"

    print(f"  Uploading to S3: {s3_path}")

    cmd = [
        "aws",
        "s3",
        "sync",
        str(local_path),
        s3_path,
        "--region",
        S3_REGION,
        "--profile",
        AWS_PROFILE,
        "--no-progress",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False

    return True


def register_dataset(
    name: str, local_path: Path, s3_uri: str, parent_id: str | None = None, dry_run: bool = False
) -> str | None:
    """Register dataset in ClearML with S3 external files."""
    metadata = parse_metadata(local_path)
    info = extract_dataset_info(name, metadata)
    description = generate_description(info, metadata)
    tags = generate_tags(info)

    print(f"\n{'=' * 60}")
    print(f"Registering: {name}")
    print(f"  S3 URI: {s3_uri}")
    print(f"  Samples: {info['n_samples']:,}")
    print(f"  Platform: {info['platform']}")
    print(f"  Tissue: {info['tissue']}")
    print(f"  Granularity: {info['granularity']}")
    print(f"  Patch Size: {info['patch_size']}")
    print(f"  Parent ID: {parent_id or 'None'}")
    print(f"  Tags: {tags}")
    print(f"{'=' * 60}")

    if dry_run:
        print("  [DRY RUN] Would register dataset")
        return "dry-run-id"

    try:
        # Create dataset
        dataset = Dataset.create(
            dataset_name=name,
            dataset_project="DAPIDL/datasets",
            dataset_tags=tags,
            description=description,
            parent_datasets=[parent_id] if parent_id else None,
            output_uri=s3_uri,  # Use S3 as storage backend
        )

        print(f"  Created dataset ID: {dataset.id}")

        # Add files from local path (they'll be synced to S3)
        # Actually, use add_external_files since data is already on S3
        dataset.add_external_files(
            source_url=s3_uri,
            verbose=True,
        )

        # Set metadata
        dataset.set_metadata(
            {
                "platform": info["platform"],
                "tissue": info["tissue"],
                "granularity": info["granularity"],
                "patch_size": info["patch_size"],
                "n_samples": info["n_samples"],
                "annotation_method": info["annotation_method"],
                **metadata,  # Include original metadata
            }
        )

        # Upload (registers S3 files)
        print("  Uploading (registering S3 files)...")
        dataset.upload()

        # Finalize
        print("  Finalizing...")
        dataset.finalize()

        print(f"  Successfully registered: {name}")
        print(f"  ClearML URL: https://app.clear.ml/datasets/simple/{dataset.id}")

        return dataset.id

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Register derived_hq datasets to ClearML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually register, just show what would happen",
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip S3 upload (assume data already uploaded)"
    )
    parser.add_argument("--dataset", type=str, help="Register specific dataset only")
    args = parser.parse_args()

    print("=" * 60)
    print("DAPIDL Dataset Registration - derived_hq")
    print("=" * 60)
    print(f"Local path: {DERIVED_HQ_PATH}")
    print(f"S3 base: {S3_CLEARML_BASE}")
    print(f"Dry run: {args.dry_run}")
    print(f"Skip upload: {args.skip_upload}")
    print("=" * 60)

    # Find all datasets
    datasets = sorted(
        [d for d in DERIVED_HQ_PATH.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    if args.dataset:
        datasets = [d for d in datasets if d.name == args.dataset]
        if not datasets:
            print(f"ERROR: Dataset '{args.dataset}' not found")
            sys.exit(1)

    print(f"\nFound {len(datasets)} datasets to process")

    # Check what's already registered
    registered = set()
    try:
        existing = Dataset.list_datasets(dataset_project="DAPIDL")
        registered = {d["name"] for d in existing}
    except Exception:
        pass

    results = {"uploaded": [], "registered": [], "failed": [], "skipped": []}

    for dataset_path in datasets:
        name = dataset_path.name

        # Skip if already registered
        if name in registered:
            print(f"\nSkipping {name} - already registered in ClearML")
            results["skipped"].append(name)
            continue

        # Build S3 URI
        s3_uri = f"{S3_CLEARML_BASE}/{name}"

        # Upload to S3 if needed
        if not args.skip_upload:
            if check_s3_exists(name):
                print(f"\n{name} already exists on S3")
            else:
                print(f"\nUploading {name} to S3...")
                if not args.dry_run:
                    if upload_to_s3(dataset_path, name):
                        results["uploaded"].append(name)
                    else:
                        print(f"  Failed to upload {name}")
                        results["failed"].append(name)
                        continue

        # Get parent dataset for lineage
        parent_id = get_parent_dataset_id(name)

        # Register in ClearML
        dataset_id = register_dataset(
            name=name,
            local_path=dataset_path,
            s3_uri=s3_uri,
            parent_id=parent_id,
            dry_run=args.dry_run,
        )

        if dataset_id:
            results["registered"].append((name, dataset_id))
        else:
            results["failed"].append(name)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Skipped (already registered): {len(results['skipped'])}")
    print(f"Uploaded to S3: {len(results['uploaded'])}")
    print(f"Registered in ClearML: {len(results['registered'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["registered"]:
        print("\nRegistered datasets:")
        for name, dataset_id in results["registered"]:
            print(f"  {name}: {dataset_id}")

    if results["failed"]:
        print("\nFailed datasets:")
        for name in results["failed"]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
