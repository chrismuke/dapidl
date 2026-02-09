#!/usr/bin/env python3
"""Migrate ClearML from Cloud (app.clear.ml) to Self-Hosted (clearml.chrism.io).

Reads from ClearML Cloud via REST API.
Writes to self-hosted via ClearML SDK.

All data (~1 TB) is already on AWS S3 (s3://dapidl/, eu-central-1).
ClearML only stores metadata, metrics, and task references.

Usage:
    # Dry run - see what would be migrated
    uv run python scripts/migrate_clearml_to_selfhosted.py --datasets
    uv run python scripts/migrate_clearml_to_selfhosted.py --experiments

    # Actually migrate
    uv run python scripts/migrate_clearml_to_selfhosted.py --datasets --apply
    uv run python scripts/migrate_clearml_to_selfhosted.py --experiments --apply

    # Everything
    uv run python scripts/migrate_clearml_to_selfhosted.py --all --apply

    # List cloud projects
    uv run python scripts/migrate_clearml_to_selfhosted.py --list-projects
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

# ==================== Cloud API (source) ====================
CLOUD_API_BASE = "https://api.clear.ml"
CLOUD_ACCESS_KEY = "0G0GWAC1W2XMMU3RFQ3VC5Z4829S4B"
CLOUD_SECRET_KEY = "Vlx_bRAztq_0fqCK4Yv32QIlR5RsZChqLTP96C_5Pf7pydRBpdiRbodKv_SvqzWoj3I"

# ==================== Self-Hosted (target) ====================
SELFHOSTED_CONFIG = str(Path.home() / ".clearml" / "clearml-chrism.conf")
SELFHOSTED_API_BASE = "https://api.clearml.chrism.io"
SELFHOSTED_ACCESS_KEY = "62K19E8UZHM0WNPLA3WAELE1MWA4KX"
SELFHOSTED_SECRET_KEY = "BW4ddZkkz3Hb7iPIj6J3w6HmMHkR-qpbP79pU2mzumGA6E9tODnvhjrQ631cU18p42o"

# ==================== ID Mapping ====================
MAPPING_FILE = Path(__file__).parent.parent / "configs" / "clearml_id_mapping.json"

# ==================== S3 Path Inference ====================
# S3 bucket structure:
#   s3://dapidl/raw-data/{name}/          — raw spatial data
#   s3://dapidl/datasets/{name}/          — derived LMDB datasets (old style, flat)
#   s3://dapidl/datasets/derived/{name}/  — derived LMDB datasets (new style)
#   s3://dapidl/datasets/lmdb/{name}/     — LMDB datasets
#   s3://dapidl/datasets/annotated/{name}/ — annotated datasets
#   s3://dapidl/clearml-outputs/...       — ClearML artifact storage

# Map ClearML project to S3 path prefix
PROJECT_S3_PREFIX = {
    "DAPIDL/raw-data": "raw-data",
    "DAPIDL/Derived": "datasets",
    "DAPIDL/lmdb": "datasets/lmdb",
    "DAPIDL/annotated": "datasets/annotated",
    "DAPIDL/datasets": "datasets",
}

# Dataset projects to migrate (registered datasets, NOT pipeline intermediates)
DATASET_PROJECTS = [
    "DAPIDL/Derived",
    "DAPIDL/raw-data",
    "DAPIDL/lmdb",
    "DAPIDL/annotated",
    "DAPIDL/datasets",
]

# Experiment projects to migrate
EXPERIMENT_PROJECTS = [
    "DAPIDL/training",
    "DAPIDL/HPO",
    "DAPIDL/Annotation-Benchmark",
    "DAPIDL/Finegrained-Benchmark",
    "DAPIDL/CL-Standardized-Benchmark",
]


class CloudAPI:
    """REST client for reading from ClearML Cloud."""

    def __init__(self):
        self.token = self._login()
        self._project_cache: dict[str, str] = {}

    def _login(self) -> str:
        resp = requests.post(
            f"{CLOUD_API_BASE}/auth.login",
            auth=(CLOUD_ACCESS_KEY, CLOUD_SECRET_KEY),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()["data"]["token"]

    def call(self, endpoint: str, data: dict) -> dict:
        resp = requests.post(
            f"{CLOUD_API_BASE}/{endpoint}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=data,
        )
        resp.raise_for_status()
        return resp.json()

    def get_all_tasks(
        self,
        task_type: list[str] | None = None,
        project_name: str | None = None,
        status: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> list[dict]:
        """Paginated fetch of all matching tasks."""
        all_tasks = []
        page = 0
        page_size = 500

        query: dict = {"page": page, "page_size": page_size}
        if task_type:
            query["type"] = task_type
        if status:
            query["status"] = status
        if fields:
            query["only_fields"] = fields
        if project_name:
            proj_ids = self._resolve_project_ids(project_name)
            if proj_ids:
                query["project"] = proj_ids

        while True:
            query["page"] = page
            result = self.call("tasks.get_all", query)
            tasks = result.get("data", {}).get("tasks", [])
            all_tasks.extend(tasks)
            if len(tasks) < page_size:
                break
            page += 1

        return all_tasks

    def _resolve_project_ids(self, name: str) -> list[str]:
        """Resolve project name to list of matching project IDs."""
        result = self.call(
            "projects.get_all",
            {
                "name": name,
                "only_fields": ["id", "name"],
            },
        )
        return [p["id"] for p in result.get("data", {}).get("projects", [])]

    def get_task(self, task_id: str) -> dict:
        result = self.call("tasks.get_by_id", {"task": task_id})
        return result.get("data", {}).get("task", {})

    def get_scalar_metrics(self, task_id: str) -> dict:
        """Get scalar metrics histogram for a task."""
        result = self.call(
            "events.scalar_metrics_iter_histogram",
            {
                "task": task_id,
            },
        )
        return result.get("data", {})

    def get_all_projects(self) -> list[dict]:
        """Get all projects."""
        result = self.call(
            "projects.get_all",
            {
                "page": 0,
                "page_size": 500,
                "only_fields": ["id", "name"],
            },
        )
        return result.get("data", {}).get("projects", [])

    def get_project_name(self, project_id: str) -> str:
        """Resolve project ID to name with caching."""
        if project_id not in self._project_cache:
            try:
                result = self.call("projects.get_by_id", {"project": project_id})
                self._project_cache[project_id] = (
                    result.get("data", {}).get("project", {}).get("name", "DAPIDL")
                )
            except Exception:
                self._project_cache[project_id] = "DAPIDL"
        return self._project_cache[project_id]


class SelfHostedAPI:
    """REST client for writing to self-hosted ClearML."""

    def __init__(self):
        self.token = self._login()
        self._project_cache: dict[str, str] = {}  # name -> id

    def _login(self) -> str:
        resp = requests.post(
            f"{SELFHOSTED_API_BASE}/auth.login",
            auth=(SELFHOSTED_ACCESS_KEY, SELFHOSTED_SECRET_KEY),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()["data"]["token"]

    def call(self, endpoint: str, data: dict) -> dict:
        resp = requests.post(
            f"{SELFHOSTED_API_BASE}/{endpoint}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=data,
        )
        resp.raise_for_status()
        return resp.json()

    def get_or_create_project(self, name: str) -> str:
        """Get or create a project by name, return project ID."""
        if name in self._project_cache:
            return self._project_cache[name]

        # Try to find existing
        result = self.call(
            "projects.get_all",
            {"name": name, "only_fields": ["id", "name"]},
        )
        projects = result.get("data", {}).get("projects", [])
        # Exact match
        for p in projects:
            if p["name"] == name:
                self._project_cache[name] = p["id"]
                return p["id"]

        # Create new
        result = self.call(
            "projects.create",
            {"name": name, "description": f"Migrated from ClearML Cloud"},
        )
        project_id = result.get("data", {}).get("id", "")
        self._project_cache[name] = project_id
        return project_id

    def create_dataset_task(
        self,
        *,
        name: str,
        project_name: str,
        tags: list[str],
        comment: str,
        configuration: dict,
        s3_uri: str | None,
        cloud_id: str,
    ) -> str:
        """Create a dataset task directly via REST API.

        Returns the new task ID.
        """
        project_id = self.get_or_create_project(project_name)

        # Create the task
        create_data: dict = {
            "name": name,
            "project": project_id,
            "type": "data_processing",
            "tags": tags,
        }
        if comment:
            create_data["comment"] = comment

        result = self.call("tasks.create", create_data)
        task_id = result.get("data", {}).get("id", "")

        # Copy configuration from cloud task (contains dataset structure)
        if configuration:
            self.call(
                "tasks.edit",
                {
                    "task": task_id,
                    "configuration": configuration,
                },
            )

        # Add metadata about the migration as a system tag and in configuration
        migration_config = {
            "migration_info": {
                "type": "legacy.conf",
                "value": json.dumps(
                    {
                        "s3_uri": s3_uri or "unknown",
                        "cloud_id": cloud_id,
                        "migrated": datetime.now().isoformat(),
                        "registration_type": "external_reference",
                    }
                ),
            }
        }
        # Merge with existing configuration
        merged_config = dict(configuration) if configuration else {}
        merged_config.update(migration_config)
        self.call(
            "tasks.edit",
            {"task": task_id, "configuration": merged_config},
        )

        # Mark as completed (stopped → completed)
        self.call("tasks.started", {"task": task_id})
        self.call("tasks.completed", {"task": task_id})

        return task_id


def setup_selfhosted_env():
    """Configure environment for self-hosted ClearML SDK calls.

    Clears env var overrides that would conflict with the config file.
    Also sets AWS_PROFILE to use the correct S3 credentials (dapidl profile)
    instead of any stale env vars (e.g. old iDrive credentials).
    """
    os.environ["CLEARML_CONFIG_FILE"] = SELFHOSTED_CONFIG
    for key in [
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_FILES_HOST",
    ]:
        os.environ.pop(key, None)

    # Clear stale AWS env vars (may point to old iDrive e2 credentials)
    # and use the dapidl profile from ~/.aws/credentials instead
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_SESSION_TOKEN", None)
    os.environ["AWS_PROFILE"] = "dapidl"


def load_mapping() -> dict:
    """Load existing ID mapping or create empty one."""
    if MAPPING_FILE.exists():
        return json.loads(MAPPING_FILE.read_text())
    return {
        "migration_date": None,
        "source": "app.clear.ml",
        "target": "clearml.chrism.io",
        "datasets": {},
        "tasks": {},
    }


def save_mapping(mapping: dict):
    """Save ID mapping to disk."""
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping["migration_date"] = datetime.now().isoformat()
    MAPPING_FILE.write_text(json.dumps(mapping, indent=2))


def infer_s3_uri(name: str, project_name: str) -> str | None:
    """Infer S3 URI for a dataset based on name and project.

    The S3 bucket has this structure:
      s3://dapidl/raw-data/{name}/
      s3://dapidl/datasets/{name}/
      s3://dapidl/datasets/derived/{name}/
      s3://dapidl/datasets/lmdb/{name}/
      s3://dapidl/datasets/annotated/{name}/
    """
    prefix = PROJECT_S3_PREFIX.get(project_name)
    if prefix:
        return f"s3://dapidl/{prefix}/{name}"
    # Default: try datasets/
    return f"s3://dapidl/datasets/{name}"


def extract_s3_uri_from_task(task: dict) -> str | None:
    """Try to extract an S3 URI from cloud task artifacts or configuration."""
    # Check artifacts for S3 URIs (most reliable)
    artifacts = task.get("execution", {}).get("artifacts", [])
    for artifact in artifacts:
        uri = artifact.get("uri", "")
        if uri.startswith("s3://") and "dapidl" in uri:
            # Return the base path (without artifact-specific path)
            return uri

    # Check output destination
    output = task.get("output", {})
    dest = output.get("destination", "")
    if dest.startswith("s3://") and "dapidl" in dest:
        return dest

    return None


# ==================== Dataset Migration ====================


def migrate_datasets(
    cloud: CloudAPI, selfhosted: SelfHostedAPI, mapping: dict, apply: bool
):
    """Migrate dataset references from Cloud to Self-Hosted via REST API.

    Copies task configuration (including dataset manifest and S3 references)
    directly through the REST APIs — no ClearML SDK or S3 access needed.
    """
    print("\n" + "=" * 60)
    print("DATASET MIGRATION")
    print("=" * 60)

    # ClearML stores datasets in hidden sub-projects like
    # "DAPIDL/Derived/.datasets/dataset-name", so we can't filter
    # by project directly. Fetch all and filter by project prefix.
    print("Fetching all dataset tasks from ClearML Cloud...")
    tasks = cloud.get_all_tasks(
        task_type=["data_processing"],
        fields=[
            "id",
            "name",
            "status",
            "project",
            "tags",
            "configuration",
            "runtime",
            "created",
            "comment",
            "output",
            "execution",
        ],
    )
    print(f"Found {len(tasks)} dataset tasks total")

    # Filter to tasks in our target projects (by project name prefix)
    filtered = []
    for t in tasks:
        proj_name = cloud.get_project_name(t.get("project", ""))
        if any(proj_name.startswith(dp) for dp in DATASET_PROJECTS):
            filtered.append(t)
    tasks = filtered
    print(f"In target projects: {len(tasks)}")

    # Only migrate completed tasks
    completed = [t for t in tasks if t.get("status") == "completed"]
    print(f"Completed: {len(completed)}")

    # Group by name to find latest version
    by_name: dict[str, list[dict]] = defaultdict(list)
    for t in completed:
        by_name[t["name"]].append(t)

    # Keep latest version per name
    latest_tasks = []
    for _name, versions in by_name.items():
        latest = max(versions, key=lambda t: t.get("created", ""))
        latest_tasks.append(latest)

    print(f"Unique datasets (latest version): {len(latest_tasks)}")

    mode = "APPLYING" if apply else "DRY RUN"
    print(f"\n--- {mode} ---\n")

    success = 0
    skipped = 0
    failed = 0

    for task in sorted(latest_tasks, key=lambda t: t["name"]):
        cloud_id = task["id"]
        name = task["name"]
        project_name = cloud.get_project_name(task.get("project", ""))
        tags = task.get("tags", [])

        # Skip if already migrated
        if cloud_id in mapping.get("datasets", {}):
            print(f"  SKIP {name:55s} (already migrated)")
            skipped += 1
            continue

        # Get parent project name (strip hidden .datasets subproject)
        # e.g. "DAPIDL/Derived/.datasets/xenium-foo" -> "DAPIDL/Derived"
        parent_project = project_name
        if "/.datasets" in project_name:
            parent_project = project_name.split("/.datasets")[0]

        # Determine S3 URI
        s3_uri = infer_s3_uri(name, parent_project)

        if apply:
            try:
                # Get full task configuration from cloud
                configuration = task.get("configuration", {})
                comment = task.get("comment", "")

                new_id = selfhosted.create_dataset_task(
                    name=name,
                    project_name=parent_project,
                    tags=(tags or []) + ["migrated-from-cloud"],
                    comment=comment,
                    configuration=configuration,
                    s3_uri=s3_uri,
                    cloud_id=cloud_id,
                )

                mapping.setdefault("datasets", {})[cloud_id] = new_id
                save_mapping(mapping)

                success += 1
                print(f"  OK   {name:55s} {cloud_id[:8]}..→{new_id[:8]}..")

            except Exception as e:
                failed += 1
                print(f"  FAIL {name:55s} {e}")
        else:
            success += 1
            s3_flag = "s3" if s3_uri else "??"
            print(f"  [{s3_flag}] {name:55s} {cloud_id[:12]} tags={tags[:3]}")

    print(
        f"\nDatasets: {success} {'migrated' if apply else 'would migrate'}, "
        f"{skipped} skipped, {failed} failed"
    )


# ==================== Experiment Migration ====================


def migrate_experiments(cloud: CloudAPI, mapping: dict, apply: bool):
    """Migrate training experiments from Cloud to Self-Hosted."""
    print("\n" + "=" * 60)
    print("EXPERIMENT MIGRATION")
    print("=" * 60)

    all_tasks = []
    for proj in EXPERIMENT_PROJECTS:
        print(f"Fetching from {proj}...")
        tasks = cloud.get_all_tasks(
            task_type=["training"],
            project_name=proj,
            status=["completed"],
            fields=[
                "id",
                "name",
                "status",
                "project",
                "tags",
                "hyperparams",
                "configuration",
                "created",
                "comment",
                "started",
                "completed",
            ],
        )
        all_tasks.extend(tasks)
        print(f"  Found {len(tasks)} completed training tasks")

    print(f"\nTotal training tasks to migrate: {len(all_tasks)}")

    mode = "APPLYING" if apply else "DRY RUN"
    print(f"\n--- {mode} ---\n")

    success = 0
    skipped = 0
    failed = 0

    for task in sorted(all_tasks, key=lambda t: t.get("created", "")):
        cloud_id = task["id"]
        name = task["name"]
        project_name = cloud.get_project_name(task.get("project", ""))

        # Skip if already migrated
        if cloud_id in mapping.get("tasks", {}):
            print(f"  SKIP {name:55s} (already migrated)")
            skipped += 1
            continue

        if apply:
            try:
                setup_selfhosted_env()
                from clearml import Task as CMLTask

                # Fetch full task details from cloud
                full_task = cloud.get_task(cloud_id)

                # Create task on self-hosted
                new_task = CMLTask.create(
                    project_name=project_name,
                    task_name=name,
                    task_type=CMLTask.TaskTypes.training,
                )

                # Copy hyperparameters
                hyperparams = full_task.get("hyperparams", {})
                if hyperparams:
                    flat_params = {}
                    for section, params in hyperparams.items():
                        for param_name, param_data in params.items():
                            key = f"{section}/{param_name}"
                            flat_params[key] = param_data.get("value", "")
                    if flat_params:
                        new_task.set_parameters(flat_params)

                # Copy tags
                tags = full_task.get("tags", [])
                new_task.set_tags((tags or []) + ["migrated-from-cloud"])

                # Copy comment
                comment = full_task.get("comment", "")
                if comment:
                    new_task.set_comment(comment)

                # Fetch and copy scalar metrics
                metrics_copied = 0
                try:
                    metrics_data = cloud.get_scalar_metrics(cloud_id)
                    logger = new_task.get_logger()

                    for metric_key, metric_variants in metrics_data.items():
                        if not isinstance(metric_variants, dict):
                            continue
                        for variant_key, variant_data in metric_variants.items():
                            if not isinstance(variant_data, dict):
                                continue
                            x_values = variant_data.get("x", [])
                            y_values = variant_data.get("y", [])
                            for x, y in zip(x_values, y_values, strict=False):
                                logger.report_scalar(
                                    title=metric_key,
                                    series=variant_key,
                                    value=y,
                                    iteration=int(x),
                                )
                                metrics_copied += 1
                except Exception as e:
                    print(f"    WARN: Could not copy metrics: {e}")

                new_task.close()

                mapping.setdefault("tasks", {})[cloud_id] = new_task.id
                save_mapping(mapping)

                success += 1
                print(
                    f"  OK   {name:55s} {cloud_id[:8]}..→{new_task.id[:8]}.. ({metrics_copied} pts)"
                )

            except Exception as e:
                failed += 1
                print(f"  FAIL {name:55s} {e}")
        else:
            success += 1
            print(f"  {name:55s} {cloud_id[:12]} [{project_name}]")

    print(
        f"\nExperiments: {success} {'migrated' if apply else 'would migrate'}, "
        f"{skipped} skipped, {failed} failed"
    )


# ==================== Main ====================


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ClearML from Cloud to Self-Hosted",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run datasets
  uv run python scripts/migrate_clearml_to_selfhosted.py --datasets

  # Actually migrate datasets
  uv run python scripts/migrate_clearml_to_selfhosted.py --datasets --apply

  # Migrate experiments
  uv run python scripts/migrate_clearml_to_selfhosted.py --experiments --apply

  # Everything
  uv run python scripts/migrate_clearml_to_selfhosted.py --all --apply

  # List cloud projects
  uv run python scripts/migrate_clearml_to_selfhosted.py --list-projects
""",
    )
    parser.add_argument("--datasets", action="store_true", help="Migrate datasets")
    parser.add_argument("--experiments", action="store_true", help="Migrate training experiments")
    parser.add_argument("--all", action="store_true", help="Migrate everything")
    parser.add_argument("--apply", action="store_true", help="Actually apply (default: dry run)")
    parser.add_argument("--list-projects", action="store_true", help="List all cloud projects")
    args = parser.parse_args()

    if not (args.datasets or args.experiments or args.all or args.list_projects):
        parser.print_help()
        sys.exit(1)

    # Load mapping
    mapping = load_mapping()

    print("Connecting to ClearML Cloud (source)...")
    cloud = CloudAPI()
    print("OK")

    if args.list_projects:
        projects = cloud.get_all_projects()
        print(f"\nCloud projects ({len(projects)}):")
        for p in sorted(projects, key=lambda x: x["name"]):
            print(f"  {p['name']:50s} {p['id'][:12]}")
        return

    print("Connecting to self-hosted ClearML (target)...")
    selfhosted = SelfHostedAPI()
    print("OK")

    if not args.apply:
        print("\n*** DRY RUN - use --apply to actually migrate ***\n")

    if args.datasets or args.all:
        migrate_datasets(cloud, selfhosted, mapping, apply=args.apply)

    if args.experiments or args.all:
        migrate_experiments(cloud, mapping, apply=args.apply)

    if args.apply:
        save_mapping(mapping)
        print(f"\nID mapping saved to: {MAPPING_FILE}")

    print("\nDone!")


if __name__ == "__main__":
    main()
