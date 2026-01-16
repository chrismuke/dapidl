"""S3 utilities for DAPIDL dataset handling.

This module provides the correct pattern for data storage:
1. Upload data to S3 directly (not via ClearML)
2. Register with ClearML using add_external_files() (metadata only)

NEVER upload large data directly to ClearML file server.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

# S3 Configuration (iDrive e2)
S3_ENDPOINT = "https://s3.eu-central-2.idrivee2.com"
S3_REGION = "eu-central-2"
S3_BUCKET = "dapidl"

# Set from environment or use defaults
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "evkizOGyflbhx5uSi4oV")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "zHoIBfkh2qgKub9c2R5rgmD0ISfSJDDQQ55cZkk9")


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
    """Upload a file or directory to S3.

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

    # Build aws s3 sync/cp command
    cmd = [
        "aws", "s3",
        "sync" if local_path.is_dir() else "cp",
        str(local_path),
        s3_uri,
        "--endpoint-url", S3_ENDPOINT,
        "--region", S3_REGION,
    ]

    # Set credentials in environment
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = S3_ACCESS_KEY
    env["AWS_SECRET_ACCESS_KEY"] = S3_SECRET_KEY

    logger.info(f"Uploading to S3: {local_path} -> {s3_uri}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Upload complete: {s3_uri}")

        if delete_local:
            import shutil
            if local_path.is_dir():
                shutil.rmtree(local_path)
            else:
                local_path.unlink()
            logger.info(f"Deleted local copy: {local_path}")

        return s3_uri

    except subprocess.CalledProcessError as e:
        logger.error(f"S3 upload failed: {e.stderr}")
        raise RuntimeError(f"Failed to upload to S3: {e.stderr}")


def download_from_s3(
    s3_uri: str,
    local_path: Optional[Path] = None,
) -> Path:
    """Download a file or directory from S3.

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
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aws", "s3",
        "sync",
        s3_uri,
        str(local_path),
        "--endpoint-url", S3_ENDPOINT,
        "--region", S3_REGION,
    ]

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = S3_ACCESS_KEY
    env["AWS_SECRET_ACCESS_KEY"] = S3_SECRET_KEY

    logger.info(f"Downloading from S3: {s3_uri} -> {local_path}")

    try:
        subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        logger.info(f"Download complete: {local_path}")
        return local_path

    except subprocess.CalledProcessError as e:
        logger.error(f"S3 download failed: {e.stderr}")
        raise RuntimeError(f"Failed to download from S3: {e.stderr}")


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
