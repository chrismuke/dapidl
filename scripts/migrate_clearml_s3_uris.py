#!/usr/bin/env python3
"""Migrate ClearML dataset S3 URIs from iDrive e2 to AWS S3.

Replaces all occurrences of:
  s3://s3.eu-central-2.idrivee2.com/dapidl/...
with:
  s3://dapidl/...

in ClearML dataset task configurations (Dataset Content, Dataset Struct).

Usage:
    # Dry run (default) - shows what would change
    uv run python scripts/migrate_clearml_s3_uris.py

    # Actually apply changes
    uv run python scripts/migrate_clearml_s3_uris.py --apply
"""

import argparse
import json
import sys

import requests

# ClearML Cloud API
API_BASE = "https://api.clear.ml"
ACCESS_KEY = "0G0GWAC1W2XMMU3RFQ3VC5Z4829S4B"
SECRET_KEY = "Vlx_bRAztq_0fqCK4Yv32QIlR5RsZChqLTP96C_5Pf7pydRBpdiRbodKv_SvqzWoj3I"

# URI replacement
OLD_PREFIX = "s3://s3.eu-central-2.idrivee2.com/dapidl/"
NEW_PREFIX = "s3://dapidl/"

# Also catch the output_uri pattern
OLD_OUTPUT_URI = "s3://s3.eu-central-2.idrivee2.com/dapidl"
NEW_OUTPUT_URI = "s3://dapidl"


def get_token() -> str:
    resp = requests.post(
        f"{API_BASE}/auth.login",
        auth=(ACCESS_KEY, SECRET_KEY),
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["data"]["token"]


def api_call(token: str, endpoint: str, data: dict) -> dict:
    resp = requests.post(
        f"{API_BASE}/{endpoint}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=data,
    )
    resp.raise_for_status()
    return resp.json()


def get_all_dataset_tasks(token: str) -> list[dict]:
    """Get all dataset tasks (type=data_processing) from DAPIDL project."""
    all_tasks = []
    page = 0
    page_size = 500

    while True:
        result = api_call(
            token,
            "tasks.get_all",
            {
                "type": ["data_processing"],
                "page": page,
                "page_size": page_size,
                "only_fields": ["id", "name", "status", "configuration", "output", "execution"],
            },
        )
        tasks = result.get("data", {}).get("tasks", [])
        all_tasks.extend(tasks)

        if len(tasks) < page_size:
            break
        page += 1

    return all_tasks


def find_idrive_refs(task: dict) -> list[str]:
    """Find which configuration keys contain iDrive references."""
    refs = []
    config = task.get("configuration", {})
    for key, val_obj in config.items():
        value = val_obj.get("value", "")
        if "idrivee2" in value:
            refs.append(f"configuration.{key}")

    # Check execution parameters
    params = task.get("execution", {}).get("parameters", {}) or {}
    for key, val in params.items():
        if isinstance(val, str) and "idrivee2" in val:
            refs.append(f"execution.parameters.{key}")

    # Check output
    output = task.get("output", {})
    result = output.get("result", "") or ""
    if "idrivee2" in result:
        refs.append("output.result")

    return refs


def migrate_task(token: str, task: dict, apply: bool = False) -> bool:
    """Migrate a single task's S3 URIs. Returns True if changes were made."""
    task_id = task["id"]
    name = task["name"]
    config = task.get("configuration", {})
    changed = False

    # Update configuration values
    new_config = {}
    for key, val_obj in config.items():
        value = val_obj.get("value", "")
        if "idrivee2" in value:
            new_value = value.replace(OLD_PREFIX, NEW_PREFIX)
            new_value = new_value.replace(OLD_OUTPUT_URI, NEW_OUTPUT_URI)
            new_config[key] = {**val_obj, "value": new_value}
            changed = True

    if changed and apply:
        # Use tasks.edit to update configuration
        update_data = {"task": task_id, "configuration": new_config, "force": True}
        try:
            api_call(token, "tasks.edit", update_data)
        except Exception as e:
            print(f"  ERROR updating {name} ({task_id}): {e}")
            return False

    return changed


def main():
    parser = argparse.ArgumentParser(description="Migrate ClearML S3 URIs from iDrive to AWS")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry run)")
    args = parser.parse_args()

    print("Authenticating with ClearML Cloud...")
    token = get_token()
    print("OK")

    print("Fetching all dataset tasks...")
    tasks = get_all_dataset_tasks(token)
    print(f"Found {len(tasks)} dataset tasks")

    # Find tasks with iDrive references
    idrive_tasks = []
    for t in tasks:
        refs = find_idrive_refs(t)
        if refs:
            idrive_tasks.append((t, refs))

    print(f"Tasks with iDrive S3 refs: {len(idrive_tasks)}")
    print()

    if not idrive_tasks:
        print("Nothing to migrate!")
        return

    mode = "APPLYING" if args.apply else "DRY RUN"
    print(f"=== {mode} ===")
    print()

    success = 0
    failed = 0
    for task, refs in sorted(idrive_tasks, key=lambda x: x[0]["name"]):
        name = task["name"]
        tid = task["id"]
        status = task["status"]
        ref_str = ", ".join(refs)

        if args.apply:
            ok = migrate_task(token, task, apply=True)
            if ok:
                success += 1
                print(f"  OK  {name:55s} [{status}] ({ref_str})")
            else:
                failed += 1
                print(f"  FAIL {name:55s} [{status}] ({ref_str})")
        else:
            success += 1
            print(f"  {name:55s} [{status}] ({ref_str})")

    print()
    print(f"Summary: {success} {'migrated' if args.apply else 'would migrate'}, {failed} failed")

    if not args.apply:
        print()
        print("Run with --apply to actually update the URIs.")


if __name__ == "__main__":
    main()
