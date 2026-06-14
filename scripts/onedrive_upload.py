"""OneDrive upload via Microsoft device-code OAuth flow.

Designed for headless servers: the OAuth grant happens in a browser on ANY
device (phone, laptop) via https://microsoft.com/devicelogin. We then store
the refresh_token locally and use Microsoft Graph to upload files.

Why not rclone? rclone's OneDrive backend doesn't expose device-code auth,
and its standard headless flow requires rclone on the user's laptop. msal's
device-code flow lets us avoid that — the user only needs a browser.

First run: prints a short user code + URL, waits for the user to complete
auth, then uploads. Subsequent runs use the cached refresh_token transparently.

Token cache (refresh_token) lives at ~/.config/onedrive/msal_cache.json.

Usage:
    uv run python scripts/onedrive_upload.py \
        --src /home/chrism/git/dapidl/pipeline_output/pilot_qc_collages_v2 \
        --dst-folder dapidl-qc/pilot_qc_collages_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import msal
import requests

# Azure CLI's well-known public client_id. rclone's client isn't registered
# for the device-code grant ("AADSTS70002"), but the Azure CLI client is —
# Microsoft maintains it as a stable mobile/public client and grants Graph
# API access broadly. Works for personal MS accounts via /common.
CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"
AUTHORITY = "https://login.microsoftonline.com/common"
SCOPES = ["Files.ReadWrite.All"]  # msal injects openid/profile/offline_access

CACHE_PATH = Path.home() / ".config/onedrive/msal_cache.json"
GRAPH = "https://graph.microsoft.com/v1.0"


def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if CACHE_PATH.exists():
        cache.deserialize(CACHE_PATH.read_text())
    return cache


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(cache.serialize())
        os.chmod(CACHE_PATH, 0o600)


def get_token() -> str:
    cache = _load_cache()
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    # Try silent first (refresh_token already cached).
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(cache)
            return result["access_token"]

    # Device-code flow: ask Microsoft for a code, surface it to the user.
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"device-code init failed: {flow}")
    print("\n" + "=" * 72)
    print(" 1. Open in any browser:  " + flow["verification_uri"])
    print(" 2. Sign in with christian.mess@gmx.de")
    print(" 3. Enter this code:      " + flow["user_code"])
    print(f"    (expires in {flow['expires_in']} s)")
    print("=" * 72 + "\n", flush=True)

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(f"device-code auth failed: {result}")
    _save_cache(cache)
    return result["access_token"]


def _http(method: str, url: str, token: str, **kw) -> requests.Response:
    headers = kw.pop("headers", {})
    headers["Authorization"] = "Bearer " + token
    return requests.request(method, url, headers=headers, timeout=600, **kw)


def ensure_folder(token: str, path_segments: list[str]) -> str:
    """Create the folder chain under /me/drive/root. Returns the leaf item id."""
    parent = "root"
    for seg in path_segments:
        # Check for existing child
        r = _http("GET", f"{GRAPH}/me/drive/items/{parent}/children?$top=200", token)
        r.raise_for_status()
        existing = next((c for c in r.json().get("value", []) if c["name"] == seg), None)
        if existing:
            parent = existing["id"]
            continue
        # Create
        r = _http("POST", f"{GRAPH}/me/drive/items/{parent}/children", token,
                  json={"name": seg, "folder": {}, "@microsoft.graph.conflictBehavior": "rename"})
        r.raise_for_status()
        parent = r.json()["id"]
    return parent


def upload_file(token: str, local: Path, remote_parent_id: str) -> None:
    size = local.stat().st_size
    # Small (<4 MiB): one PUT request to /content
    if size <= 4 * 1024 * 1024:
        url = (f"{GRAPH}/me/drive/items/{remote_parent_id}:/"
               f"{requests.utils.quote(local.name)}:/content")
        with open(local, "rb") as f:
            r = _http("PUT", url, token, data=f.read())
        r.raise_for_status()
        return
    # Large (>4 MiB): upload session
    r = _http("POST",
              f"{GRAPH}/me/drive/items/{remote_parent_id}:/"
              f"{requests.utils.quote(local.name)}:/createUploadSession", token,
              json={"item": {"@microsoft.graph.conflictBehavior": "replace"}})
    r.raise_for_status()
    upload_url = r.json()["uploadUrl"]
    chunk = 10 * 1024 * 1024  # 10 MiB
    with open(local, "rb") as f:
        pos = 0
        while pos < size:
            data = f.read(chunk)
            end = pos + len(data) - 1
            headers = {
                "Content-Length": str(len(data)),
                "Content-Range": f"bytes {pos}-{end}/{size}",
            }
            for attempt in range(3):
                try:
                    r = requests.put(upload_url, headers=headers, data=data, timeout=600)
                    if r.status_code in (200, 201, 202):
                        break
                except requests.exceptions.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            r.raise_for_status()
            pos += len(data)


def upload_tree(token: str, src: Path, remote_root_id: str) -> tuple[int, int]:
    n_files = 0
    n_bytes = 0
    folder_cache: dict[Path, str] = {src: remote_root_id}
    files = sorted(p for p in src.rglob("*") if p.is_file())
    total = len(files)
    print(f"Uploading {total} files ({sum(p.stat().st_size for p in files) / 1e6:.1f} MB)", flush=True)
    for i, p in enumerate(files, start=1):
        rel = p.relative_to(src)
        parent = src
        parent_id = remote_root_id
        for part in rel.parts[:-1]:
            parent = parent / part
            if parent in folder_cache:
                parent_id = folder_cache[parent]
                continue
            r = _http("GET", f"{GRAPH}/me/drive/items/{parent_id}/children?$top=400", token)
            r.raise_for_status()
            ex = next((c for c in r.json().get("value", []) if c["name"] == part), None)
            if ex:
                parent_id = ex["id"]
            else:
                r = _http("POST", f"{GRAPH}/me/drive/items/{parent_id}/children", token,
                          json={"name": part, "folder": {},
                                "@microsoft.graph.conflictBehavior": "rename"})
                r.raise_for_status()
                parent_id = r.json()["id"]
            folder_cache[parent] = parent_id
        upload_file(token, p, parent_id)
        n_files += 1
        n_bytes += p.stat().st_size
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {rel}", flush=True)
    return n_files, n_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst-folder", type=str, required=True,
                    help="Slash-separated path under /me/drive/root, e.g. dapidl-qc/pilot")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"source not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    print("Authenticating…", flush=True)
    token = get_token()
    print("✓ auth ok", flush=True)

    path_segments = [s for s in args.dst_folder.strip("/").split("/") if s]
    print(f"Ensuring remote folder: {'/'.join(path_segments)}", flush=True)
    leaf_id = ensure_folder(token, path_segments)

    n_files, n_bytes = upload_tree(token, args.src, leaf_id)
    print(f"\nDONE — {n_files} files / {n_bytes/1e6:.1f} MB uploaded to "
          f"OneDrive:/{'/'.join(path_segments)}/")


if __name__ == "__main__":
    main()
