"""Reproducibility information capture for ML experiments.

This module captures all information needed to exactly reproduce a training run:
- Git commit and any uncommitted changes
- Python environment (version, packages)
- Dataset fingerprint
- CLI command used
- System info

Design Notes for Future Migration:
----------------------------------
This module is backend-agnostic. It captures information as dataclasses/dicts
that can be logged to any experiment tracking system (W&B, ClearML, ZenML, MLflow).

To migrate to a different backend:
1. Create a new tracker in tracking/backends/<name>.py implementing ExperimentTracker protocol
2. Update the tracker factory in tracking/factory.py
3. The reproducibility info capture here remains unchanged

See tracking/base.py for the ExperimentTracker protocol (to be implemented).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import platform
import socket
import subprocess
import sys
from typing import Any


@dataclass
class GitInfo:
    """Git repository state."""

    commit: str
    branch: str
    remote_url: str | None
    is_dirty: bool
    dirty_files: list[str] = field(default_factory=list)
    diff: str | None = None  # Patch of uncommitted changes (if dirty)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentInfo:
    """Python environment information."""

    python_version: str
    platform: str
    hostname: str
    packages: dict[str, str]  # name -> version
    cuda_version: str | None = None
    gpu_info: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetInfo:
    """Dataset fingerprint for reproducibility."""

    path: str
    hash: str  # SHA256 of key files
    n_samples: int | None = None
    n_classes: int | None = None
    class_names: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityInfo:
    """Complete reproducibility information for a training run.

    This dataclass captures everything needed to reproduce an experiment:
    - Code state (git)
    - Environment (python, packages, hardware)
    - Data (dataset fingerprint)
    - Execution context (command, timestamp)

    Usage:
        info = get_reproducibility_info(
            dataset_path="./dataset",
            cli_command="dapidl train -d ./dataset --epochs 50"
        )
        # Log info.to_dict() to your experiment tracker
    """

    git: GitInfo
    environment: EnvironmentInfo
    dataset: DatasetInfo
    cli_command: str
    timestamp: str
    dapidl_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary for logging."""
        return {
            "git": self.git.to_dict(),
            "environment": self.environment.to_dict(),
            "dataset": self.dataset.to_dict(),
            "cli_command": self.cli_command,
            "timestamp": self.timestamp,
            "dapidl_version": self.dapidl_version,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary with prefixed keys (for W&B config)."""
        flat = {}
        for key, value in self.git.to_dict().items():
            if key == "dirty_files":
                flat[f"git/{key}"] = ",".join(value) if value else ""
            elif key == "diff":
                continue  # Skip diff in flat dict (too large)
            else:
                flat[f"git/{key}"] = value

        for key, value in self.environment.to_dict().items():
            if key == "packages":
                continue  # Skip packages in flat dict (too large)
            else:
                flat[f"env/{key}"] = value

        flat["dataset/path"] = self.dataset.path
        flat["dataset/hash"] = self.dataset.hash
        flat["dataset/n_samples"] = self.dataset.n_samples
        flat["dataset/n_classes"] = self.dataset.n_classes

        flat["cli_command"] = self.cli_command
        flat["timestamp"] = self.timestamp
        flat["dapidl_version"] = self.dapidl_version

        return flat

    def get_reproduce_command(self) -> str:
        """Generate command to reproduce this run."""
        lines = [
            "# Reproduce this run:",
            f"# 1. Checkout the exact code version:",
            f"git checkout {self.git.commit}",
            "",
            "# 2. Verify/restore environment:",
            "uv sync",
            "",
            "# 3. Run the same command:",
            self.cli_command,
        ]
        if self.git.is_dirty:
            lines.insert(3, "# WARNING: Code had uncommitted changes! Apply the logged diff.")
        return "\n".join(lines)


def get_git_info(repo_path: str | Path | None = None, include_diff: bool = True) -> GitInfo:
    """Capture git repository state.

    Args:
        repo_path: Path to git repo (default: current directory)
        include_diff: Include diff of uncommitted changes (can be large)

    Returns:
        GitInfo with commit, branch, dirty state, and optionally diff
    """
    cwd = str(repo_path) if repo_path else None

    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    # Get commit hash
    commit = run_git(["rev-parse", "HEAD"]) or "unknown"

    # Get branch name
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"

    # Get remote URL
    remote_url = run_git(["remote", "get-url", "origin"])

    # Check if dirty
    status = run_git(["status", "--porcelain"])
    is_dirty = bool(status)

    # Get list of dirty files
    dirty_files = []
    if status:
        for line in status.split("\n"):
            if line.strip():
                # Format: "XY filename" where X=index, Y=worktree
                dirty_files.append(line[3:])

    # Get diff if dirty and requested
    diff = None
    if is_dirty and include_diff:
        # Get both staged and unstaged changes
        diff_staged = run_git(["diff", "--cached"]) or ""
        diff_unstaged = run_git(["diff"]) or ""
        diff = diff_staged + "\n" + diff_unstaged if diff_staged or diff_unstaged else None

    return GitInfo(
        commit=commit,
        branch=branch,
        remote_url=remote_url,
        is_dirty=is_dirty,
        dirty_files=dirty_files,
        diff=diff,
    )


def get_environment_info() -> EnvironmentInfo:
    """Capture Python environment information."""
    # Get installed packages
    packages = {}
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for pkg in json.loads(result.stdout):
                packages[pkg["name"]] = pkg["version"]
    except Exception:
        # Fallback to importlib.metadata
        try:
            from importlib.metadata import distributions

            for dist in distributions():
                packages[dist.metadata["Name"]] = dist.version
        except Exception:
            pass

    # Get CUDA version
    cuda_version = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            cuda_version = result.stdout.strip()
    except Exception:
        pass

    # Get GPU info
    gpu_info = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except Exception:
        pass

    return EnvironmentInfo(
        python_version=sys.version,
        platform=platform.platform(),
        hostname=socket.gethostname(),
        packages=packages,
        cuda_version=cuda_version,
        gpu_info=gpu_info,
    )


def compute_dataset_hash(dataset_path: str | Path) -> str:
    """Compute a fingerprint hash for a dataset.

    Hashes key files that define the dataset content:
    - dataset_info.json (metadata)
    - labels.npy (ground truth)
    - patches.zarr/.zarray (shape/dtype, not full data for speed)

    Args:
        dataset_path: Path to dataset directory

    Returns:
        SHA256 hash string (first 16 chars for brevity)
    """
    dataset_path = Path(dataset_path)
    hasher = hashlib.sha256()

    # Files to hash (in order for consistency)
    files_to_hash = [
        "dataset_info.json",
        "labels.npy",
        "patches.zarr/.zarray",
        "metadata.parquet",
    ]

    for filename in files_to_hash:
        filepath = dataset_path / filename
        if filepath.exists():
            # For small files, hash content
            if filepath.stat().st_size < 10_000_000:  # 10MB limit
                hasher.update(filepath.read_bytes())
            else:
                # For large files, hash size + first/last 1MB
                size = filepath.stat().st_size
                hasher.update(str(size).encode())
                with open(filepath, "rb") as f:
                    hasher.update(f.read(1_000_000))
                    f.seek(-1_000_000, 2)
                    hasher.update(f.read())

    return hasher.hexdigest()[:16]


def get_dataset_info(dataset_path: str | Path) -> DatasetInfo:
    """Get dataset information including hash.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        DatasetInfo with path, hash, and metadata
    """
    dataset_path = Path(dataset_path)

    # Compute hash
    dataset_hash = compute_dataset_hash(dataset_path)

    # Load metadata if available
    n_samples = None
    n_classes = None
    class_names = None

    info_file = dataset_path / "dataset_info.json"
    if info_file.exists():
        try:
            with open(info_file) as f:
                info = json.load(f)
                n_samples = info.get("n_samples")
                n_classes = info.get("n_classes")
                class_names = info.get("class_names")
        except Exception:
            pass

    return DatasetInfo(
        path=str(dataset_path.resolve()),
        hash=dataset_hash,
        n_samples=n_samples,
        n_classes=n_classes,
        class_names=class_names,
    )


def get_reproducibility_info(
    dataset_path: str | Path,
    cli_command: str | None = None,
    repo_path: str | Path | None = None,
) -> ReproducibilityInfo:
    """Capture complete reproducibility information for a run.

    This is the main entry point for capturing all info needed to reproduce
    a training run. Call this at the start of training and log the result
    to your experiment tracker.

    Args:
        dataset_path: Path to the dataset
        cli_command: The CLI command used to start training (optional)
        repo_path: Path to git repo (default: current directory)

    Returns:
        ReproducibilityInfo that can be logged to any tracker

    Example:
        info = get_reproducibility_info(
            dataset_path="./dataset",
            cli_command="dapidl train -d ./dataset --epochs 50 --wandb"
        )
        wandb.config.update(info.to_flat_dict())
    """
    # Get dapidl version
    try:
        from importlib.metadata import version

        dapidl_version = version("dapidl")
    except Exception:
        dapidl_version = "unknown"

    return ReproducibilityInfo(
        git=get_git_info(repo_path),
        environment=get_environment_info(),
        dataset=get_dataset_info(dataset_path),
        cli_command=cli_command or "unknown",
        timestamp=datetime.now().isoformat(),
        dapidl_version=dapidl_version,
    )


# CLI command tracking - to be called from cli.py
_current_cli_command: str | None = None


def set_cli_command(command: str) -> None:
    """Store the CLI command for later retrieval.

    Call this from the CLI entry point to capture the exact command used.
    """
    global _current_cli_command
    _current_cli_command = command


def get_cli_command() -> str | None:
    """Get the stored CLI command."""
    return _current_cli_command
