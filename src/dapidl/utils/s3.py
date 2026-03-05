"""S3 utilities for DAPIDL dataset handling.

This module provides the correct pattern for data storage:
1. Upload data to S3 directly (not via ClearML)
2. Register with ClearML using add_external_files() (metadata only)

NEVER upload large data directly to ClearML file server.

AWS credentials are resolved via standard boto3 chain:
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- ~/.aws/credentials profiles (AWS_PROFILE)
- IAM instance profile (on EC2)
"""

import os
from pathlib import Path
from typing import Optional

import boto3
from loguru import logger

# S3 Configuration — AWS S3 defaults (eu-central-1).
# Override with DAPIDL_S3_* env vars if needed.
# For S3-compatible storage: set DAPIDL_S3_ENDPOINT to the endpoint URL.
S3_ENDPOINT = os.environ.get("DAPIDL_S3_ENDPOINT", "")
S3_REGION = os.environ.get("DAPIDL_S3_REGION", "eu-central-1")
S3_BUCKET = os.environ.get("DAPIDL_S3_BUCKET", "dapidl")


def _get_s3_client():
    """Get a boto3 S3 client with optional endpoint override."""
    kwargs = {"region_name": S3_REGION}
    if S3_ENDPOINT:
        kwargs["endpoint_url"] = S3_ENDPOINT
    return boto3.client("s3", **kwargs)


def get_s3_uri(path: str) -> str:
    """Convert a relative path to full S3 URI."""
    if path.startswith("s3://"):
        return path
    return f"s3://{S3_BUCKET}/{path.lstrip('/')}"


def upload_to_s3(
    local_path: Path,
    s3_path: str,
    delete_local: bool = False,
) -> str:
    """Upload a file or directory to S3 using boto3.

    Args:
        local_path: Local file or directory to upload
        s3_path: Destination path in S3 (e.g., "datasets/lmdb/my-dataset")
        delete_local: Delete local copy after successful upload

    Returns:
        Full S3 URI of uploaded data
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local path not found: {local_path}")

    s3_uri = get_s3_uri(s3_path)
    # Parse bucket and key prefix from s3_uri
    s3_parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = s3_parts[0]
    key_prefix = s3_parts[1] if len(s3_parts) > 1 else ""

    s3 = _get_s3_client()
    logger.info(f"Uploading to S3: {local_path} -> {s3_uri}")

    count = 0
    if local_path.is_dir():
        for file in local_path.rglob("*"):
            if file.is_file():
                rel = file.relative_to(local_path)
                key = f"{key_prefix}/{rel}" if key_prefix else str(rel)
                s3.upload_file(str(file), bucket, key)
                count += 1
    else:
        key = f"{key_prefix}/{local_path.name}" if key_prefix else local_path.name
        s3.upload_file(str(local_path), bucket, key)
        count = 1

    logger.info(f"Upload complete: {count} files -> {s3_uri}")

    if delete_local:
        import shutil

        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()
        logger.info(f"Deleted local copy: {local_path}")

    return s3_uri


def download_from_s3(
    s3_uri: str,
    local_path: Optional[Path] = None,
) -> Path:
    """Download a file or directory from S3 using boto3.

    Args:
        s3_uri: S3 URI to download
        local_path: Optional local destination. If None, uses cache directory.

    Returns:
        Path to downloaded data
    """
    if local_path is None:
        # Use cache directory
        parts = s3_uri.replace("s3://", "").split("/")
        dataset_name = parts[-1] if parts[-1] else parts[-2]
        local_path = Path.home() / ".cache" / "dapidl" / "s3_downloads" / dataset_name

    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    # Parse bucket and prefix from s3_uri
    s3_parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = s3_parts[0]
    prefix = s3_parts[1] if len(s3_parts) > 1 else ""

    s3 = _get_s3_client()
    logger.info(f"Downloading from S3: {s3_uri} -> {local_path}")

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel_key = obj["Key"][len(prefix):].lstrip("/")
            if not rel_key:
                continue
            dest = local_path / rel_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(dest))
            count += 1

    logger.info(f"Download complete: {count} files -> {local_path}")
    return local_path


def register_dataset_from_s3(
    s3_uri: str,
    dataset_name: str,
    dataset_project: str,
    metadata: Optional[dict] = None,
    parent_datasets: Optional[list[str]] = None,
) -> str:
    """Register an S3 dataset with ClearML WITHOUT uploading.

    This is the correct pattern:
    1. Data is already on S3
    2. ClearML only stores metadata and reference to S3 location

    Args:
        s3_uri: S3 URI where data is stored
        dataset_name: Name for ClearML dataset
        dataset_project: ClearML project name
        metadata: Optional metadata dict
        parent_datasets: Optional list of parent dataset IDs for lineage

    Returns:
        ClearML dataset ID
    """
    from clearml import Dataset

    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        parent_datasets=parent_datasets,
        # NOTE: Do NOT set output_uri - we don't upload to ClearML
    )

    # Add external reference to S3 - this does NOT upload!
    dataset.add_external_files(
        source_url=s3_uri,
        dataset_path="/",
    )

    # Set metadata
    base_metadata = {
        "s3_uri": s3_uri,
        "registration_type": "external_reference",
        "uploaded_to_clearml": False,
    }
    if metadata:
        base_metadata.update(metadata)
    dataset.set_metadata(base_metadata)

    dataset.finalize()
    # NOTE: Do NOT call dataset.upload() - files are already on S3

    logger.info(f"Registered dataset with ClearML: {dataset.id} -> {s3_uri}")
    return dataset.id


def upload_and_register_dataset(
    local_path: Path,
    s3_path: str,
    dataset_name: str,
    dataset_project: str,
    metadata: Optional[dict] = None,
    parent_datasets: Optional[list[str]] = None,
    delete_local: bool = False,
) -> tuple[str, str]:
    """Upload to S3 and register with ClearML.

    This is the recommended workflow:
    1. Upload data to S3
    2. Register with ClearML (metadata only)

    Args:
        local_path: Local file/directory to upload
        s3_path: Destination path in S3
        dataset_name: Name for ClearML dataset
        dataset_project: ClearML project name
        metadata: Optional metadata dict
        parent_datasets: Optional parent dataset IDs
        delete_local: Delete local copy after upload

    Returns:
        Tuple of (s3_uri, clearml_dataset_id)
    """
    # Step 1: Upload to S3
    s3_uri = upload_to_s3(local_path, s3_path, delete_local=delete_local)

    # Step 2: Register with ClearML
    dataset_id = register_dataset_from_s3(
        s3_uri=s3_uri,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        metadata=metadata,
        parent_datasets=parent_datasets,
    )

    return s3_uri, dataset_id
