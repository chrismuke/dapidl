"""Content-addressable S3 cache for pipeline step outputs.

Uploads step outputs to S3 with content-based keys so identical outputs
are never re-uploaded, and outputs survive EC2 spot instance termination.

S3 structure:
    s3://dapidl/pipeline-cache/{step_name}/{content_hash_16char}/
        manifest.json   (file list with sizes and individual hashes)
        {output files...}

The manifest is uploaded LAST — its presence signals a complete upload.
"""

import json
from pathlib import Path

from loguru import logger

from dapidl.utils.s3 import S3_BUCKET, _get_s3_client

CACHE_PREFIX = "pipeline-cache"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB streaming chunks
HASH_LENGTH = 16  # hex chars from xxh128 digest
EXCLUDE_FILES = {"lock.mdb"}  # LMDB lock files change every write


def _xxhash_available() -> bool:
    try:
        import xxhash  # noqa: F401
        return True
    except ImportError:
        return False


def compute_file_hash(path: Path) -> str:
    """Stream-hash a single file using xxh128 (fallback: md5).

    Returns hex digest truncated to HASH_LENGTH chars.
    """
    if _xxhash_available():
        import xxhash
        h = xxhash.xxh128()
    else:
        import hashlib
        h = hashlib.md5()

    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)

    return h.hexdigest()[:HASH_LENGTH]


def compute_directory_hash(directory: Path) -> tuple[str, list[dict]]:
    """Hash all files in a directory using streaming xxhash.

    Excludes files in EXCLUDE_FILES (e.g. lock.mdb).
    Files are sorted by relative path for deterministic hashing.

    Returns:
        (16-char hex hash, manifest list of {path, size, hash} dicts)
    """
    if _xxhash_available():
        import xxhash
        combined = xxhash.xxh128()
    else:
        import hashlib
        combined = hashlib.md5()

    manifest = []
    files = sorted(
        (f for f in directory.rglob("*") if f.is_file() and f.name not in EXCLUDE_FILES),
        key=lambda f: str(f.relative_to(directory)),
    )

    for file_path in files:
        rel = str(file_path.relative_to(directory))
        file_hash = compute_file_hash(file_path)
        size = file_path.stat().st_size
        manifest.append({"path": rel, "size": size, "hash": file_hash})
        # Feed relative path + hash into combined hash for determinism
        combined.update(f"{rel}:{file_hash}".encode())

    return combined.hexdigest()[:HASH_LENGTH], manifest


def upload_step_output(step_name: str, output_path: Path, force: bool = False) -> str:
    """Upload step output to S3 with content-addressable key.

    1. Compute content hash of output_path (file or directory)
    2. Check if manifest.json already exists on S3 (dedup) — skip if force=True
    3. If not, upload all files + manifest.json (manifest last)
    4. Return S3 URI

    Keeps local copy intact for same-instance reuse.

    Args:
        step_name: Pipeline step name (e.g. "annotation", "lmdb_creation")
        output_path: Local path to the step output (file or directory)
        force: If True, overwrite existing cache (used with --no-cache)

    Returns:
        S3 URI like s3://dapidl/pipeline-cache/{step}/{hash}/
    """
    output_path = Path(output_path)

    if output_path.is_dir():
        content_hash, manifest = compute_directory_hash(output_path)
    else:
        content_hash = compute_file_hash(output_path)
        size = output_path.stat().st_size
        manifest = [{"path": output_path.name, "size": size, "hash": content_hash}]

    s3_key_prefix = f"{CACHE_PREFIX}/{step_name}/{content_hash}"
    s3_uri = f"s3://{S3_BUCKET}/{s3_key_prefix}/"

    s3 = _get_s3_client()

    # Check if already uploaded (manifest exists = complete upload)
    manifest_key = f"{s3_key_prefix}/manifest.json"
    if not force:
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=manifest_key)
            logger.info(f"Content cache hit — already on S3: {s3_uri}")
            return s3_uri
        except s3.exceptions.ClientError:
            pass  # Not found, proceed with upload
    else:
        logger.info(f"Force upload enabled — overwriting cache if exists")

    logger.info(f"Uploading step output to S3: {output_path} → {s3_uri}")
    total_size = sum(entry["size"] for entry in manifest)
    logger.info(f"  {len(manifest)} files, {total_size / 1024 / 1024:.1f} MB total")

    # Upload data files first
    count = 0
    if output_path.is_dir():
        for entry in manifest:
            file_path = output_path / entry["path"]
            s3_key = f"{s3_key_prefix}/{entry['path']}"
            s3.upload_file(str(file_path), S3_BUCKET, s3_key)
            count += 1
            if count % 100 == 0:
                logger.info(f"  Uploaded {count}/{len(manifest)} files...")
    else:
        s3_key = f"{s3_key_prefix}/{output_path.name}"
        s3.upload_file(str(output_path), S3_BUCKET, s3_key)
        count = 1

    # Upload manifest LAST (signals complete upload)
    manifest_data = {
        "step_name": step_name,
        "content_hash": content_hash,
        "files": manifest,
        "total_size": total_size,
        "is_directory": output_path.is_dir(),
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=manifest_key,
        Body=json.dumps(manifest_data, indent=2),
        ContentType="application/json",
    )

    logger.info(f"Upload complete: {count} files → {s3_uri}")
    return s3_uri


def download_cached_output(s3_uri: str, local_dir: Path | None = None) -> Path:
    """Download from S3 content cache to local directory.

    Default local path: ~/.cache/dapidl/pipeline-cache/{step}/{hash}/
    Skips download if local dir exists and manifest matches.

    Args:
        s3_uri: S3 URI like s3://dapidl/pipeline-cache/{step}/{hash}/
        local_dir: Override local destination directory

    Returns:
        Path to the local directory containing the downloaded files
    """
    # Parse step and hash from URI
    # s3://dapidl/pipeline-cache/{step}/{hash}/
    uri_path = s3_uri.replace(f"s3://{S3_BUCKET}/", "").strip("/")
    parts = uri_path.split("/")
    # parts = ["pipeline-cache", step_name, content_hash]
    if len(parts) < 3 or parts[0] != CACHE_PREFIX:
        raise ValueError(f"Invalid content cache URI: {s3_uri}")

    step_name = parts[1]
    content_hash = parts[2]

    if local_dir is None:
        local_dir = Path.home() / ".cache" / "dapidl" / CACHE_PREFIX / step_name / content_hash

    local_dir = Path(local_dir)
    manifest_path = local_dir / "manifest.json"

    # Check if already downloaded and valid
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
            if existing.get("content_hash") == content_hash:
                # Verify file count matches
                expected_files = len(existing.get("files", []))
                actual_files = sum(1 for f in local_dir.rglob("*") if f.is_file() and f.name != "manifest.json")
                if actual_files >= expected_files:
                    logger.info(f"Content cache local hit: {local_dir}")
                    return local_dir
        except (json.JSONDecodeError, OSError):
            pass  # Re-download

    local_dir.mkdir(parents=True, exist_ok=True)

    s3 = _get_s3_client()
    s3_prefix = f"{CACHE_PREFIX}/{step_name}/{content_hash}/"

    logger.info(f"Downloading from content cache: {s3_uri} → {local_dir}")

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            rel_key = obj["Key"][len(s3_prefix):]
            if not rel_key:
                continue
            dest = local_dir / rel_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(S3_BUCKET, obj["Key"], str(dest))
            count += 1

    logger.info(f"Download complete: {count} files → {local_dir}")
    return local_dir
