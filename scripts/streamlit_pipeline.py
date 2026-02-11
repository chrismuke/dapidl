"""DAPIDL Pipeline Launcher — Streamlit GUI for ClearML pipeline configuration and monitoring.

Run with:
    uv run --with streamlit streamlit run scripts/streamlit_pipeline.py

Requires ClearML self-hosted credentials in ~/.clearml/clearml-chrism.conf
"""

from __future__ import annotations

import base64
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# ClearML REST API client (lightweight, no SDK dependency)
# ---------------------------------------------------------------------------

CLEARML_CONFIG_PATH = Path.home() / ".clearml" / "clearml-chrism.conf"
CLEARML_ID_MAPPING = Path(__file__).resolve().parent.parent / "configs" / "clearml_id_mapping.json"


@dataclass
class ClearMLClient:
    """Minimal ClearML REST API client for reading server state."""

    api_server: str = ""
    access_key: str = ""
    secret_key: str = ""
    web_server: str = ""
    _token: str = ""
    _token_expiry: float = 0.0

    @classmethod
    def from_config(cls, config_path: Path = CLEARML_CONFIG_PATH) -> "ClearMLClient":
        """Parse HOCON-style ClearML config file."""
        text = config_path.read_text()
        client = cls()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("api_server:"):
                client.api_server = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("web_server:"):
                client.web_server = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("access_key:"):
                client.access_key = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("secret_key:"):
                client.secret_key = line.split(":", 1)[1].strip().strip('"')
        return client

    @classmethod
    def from_env_or_config(cls) -> "ClearMLClient":
        """Create client from environment variables, falling back to config file."""
        if os.environ.get("CLEARML_API_HOST"):
            return cls(
                api_server=os.environ["CLEARML_API_HOST"],
                access_key=os.environ["CLEARML_API_ACCESS_KEY"],
                secret_key=os.environ["CLEARML_API_SECRET_KEY"],
                web_server=os.environ.get("CLEARML_WEB_HOST", ""),
            )
        return cls.from_config()

    @property
    def _headers(self) -> dict[str, str]:
        if time.time() > self._token_expiry - 60:
            self._authenticate()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def _authenticate(self) -> None:
        import requests

        creds = base64.b64encode(f"{self.access_key}:{self.secret_key}".encode()).decode()
        resp = requests.post(
            f"{self.api_server}/auth.login",
            json={},
            headers={"Content-Type": "application/json", "Authorization": f"Basic {creds}"},
            timeout=10,
        )
        resp.raise_for_status()
        self._token = resp.json()["data"]["token"]
        self._token_expiry = time.time() + 3600  # tokens valid ~24h, refresh hourly

    def _post(self, endpoint: str, payload: dict) -> dict:
        import requests

        resp = requests.post(f"{self.api_server}/{endpoint}", json=payload, headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"]

    # -- High-level queries --

    def get_queues(self) -> list[dict]:
        return self._post("queues.get_all", {})["queues"]

    def get_projects(self) -> list[dict]:
        return self._post("projects.get_all", {"page": 0, "page_size": 200, "only_fields": ["name", "id"]})["projects"]

    def _resolve_project_names(self, tasks: list[dict]) -> list[dict]:
        """Replace bare project IDs with {id, name} dicts by batch-fetching project names."""
        project_ids = {t["project"] for t in tasks if isinstance(t.get("project"), str)}
        if not project_ids:
            return tasks
        proj_data = self._post("projects.get_all", {
            "id": list(project_ids),
            "only_fields": ["id", "name"],
        })
        id_to_name = {p["id"]: p["name"] for p in proj_data["projects"]}
        for t in tasks:
            pid = t.get("project")
            if isinstance(pid, str):
                t["project"] = {"id": pid, "name": id_to_name.get(pid, pid[:8])}
        return tasks

    def get_datasets(self) -> list[dict]:
        """Return dataset tasks (type=data_processing)."""
        data = self._post("tasks.get_all", {
            "type": ["data_processing"],
            "status": ["completed"],
            "page": 0,
            "page_size": 500,
            "only_fields": ["id", "name", "project", "status", "tags", "last_update"],
            "order_by": ["-last_update"],
        })
        return self._resolve_project_names(data["tasks"])

    def get_pipelines(self, limit: int = 20) -> list[dict]:
        """Return recent pipeline controller tasks."""
        data = self._post("tasks.get_all", {
            "system_tags": ["pipeline"],
            "page": 0,
            "page_size": limit,
            "only_fields": ["id", "name", "status", "project", "started", "completed", "last_update", "status_reason"],
            "order_by": ["-last_update"],
        })
        return self._resolve_project_names(data["tasks"])

    def get_task(self, task_id: str) -> dict:
        return self._post("tasks.get_by_id", {"task": task_id})["task"]

    def get_pipeline_steps(self, pipeline_id: str) -> list[dict]:
        """Get child tasks (steps) of a pipeline controller."""
        data = self._post("tasks.get_all", {
            "parent": pipeline_id,
            "page": 0,
            "page_size": 50,
            "only_fields": ["id", "name", "status", "type", "started", "completed", "last_update"],
            "order_by": ["started"],
        })
        return data["tasks"]


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_state() -> None:
    """Initialize session state defaults."""
    defaults = {
        "datasets": [],  # list of (tissue, source_id, platform, tier)
        "running_process": None,
        "launch_log": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_client() -> ClearMLClient:
    """Get or create a cached ClearML client."""
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


# ---------------------------------------------------------------------------
# Dataset picker
# ---------------------------------------------------------------------------

def dataset_picker(client: ClearMLClient) -> None:
    """Widget for adding datasets to the pipeline configuration."""
    st.subheader("Datasets")

    # Load available datasets from ClearML
    datasets = client.get_datasets()
    dataset_names = {d["id"]: d["name"] for d in datasets}

    # Show currently selected datasets
    if st.session_state.datasets:
        st.markdown("**Selected datasets:**")
        to_remove = None
        for i, (tissue, source, platform, tier) in enumerate(st.session_state.datasets):
            name = dataset_names.get(source, source[:12] + "...")
            cols = st.columns([3, 1, 1, 1, 1])
            cols[0].text(f"{tissue} — {name}")
            cols[1].text(platform)
            cols[2].text(f"Tier {tier}")
            if cols[4].button("Remove", key=f"rm_{i}"):
                to_remove = i
        if to_remove is not None:
            st.session_state.datasets.pop(to_remove)
            st.rerun()
    else:
        st.info("No datasets added yet. Add one below.")

    # Add new dataset form
    with st.expander("Add dataset", expanded=not st.session_state.datasets):
        c1, c2 = st.columns(2)
        tissue = c1.text_input("Tissue name", placeholder="e.g. breast, lung", key="new_tissue")
        platform = c2.selectbox("Platform", ["xenium", "merscope"], key="new_platform")

        # Dataset source: dropdown of available + manual entry
        source_options = ["-- Manual ID/path --"] + [f"{d['name']}  ({d['id'][:8]}...)" for d in datasets]
        source_choice = st.selectbox("Dataset source", source_options, key="new_source_choice")
        if source_choice == "-- Manual ID/path --":
            source_id = st.text_input("ClearML dataset ID or local path", key="new_source_id")
        else:
            idx = source_options.index(source_choice) - 1
            source_id = datasets[idx]["id"]
            st.caption(f"ID: `{source_id}`")

        tier = st.selectbox("Confidence tier", [1, 2, 3], index=1, key="new_tier",
                            help="1=Ground truth, 2=Consensus, 3=Single-method")

        if st.button("Add dataset", type="primary", disabled=not (tissue and source_id)):
            st.session_state.datasets.append((tissue, source_id, platform, tier))
            st.rerun()


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

RECIPES = ["default", "gt", "no_cl", "annotate_only"]
QUEUES = ["gpu-local", "gpu-cloud", "gpu-training", "cpu-local", "default"]
BACKBONES = ["efficientnetv2_rw_s", "convnext_tiny", "resnet50", "densenet121"]
PATCH_SIZES = ["32", "64", "128", "256"]
SAMPLING_STRATEGIES = ["sqrt", "equal", "proportional"]
ANNOTATORS = ["celltypist", "ground_truth", "popv"]
SEGMENTERS = ["native", "cellpose"]


def pipeline_config() -> dict:
    """Render pipeline configuration widgets and return config dict."""
    st.subheader("Pipeline Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        recipe = st.selectbox("Recipe", RECIPES, help="Processing recipe defining step sequence")
        annotator = st.selectbox("Annotator", ANNOTATORS)
        segmenter = st.selectbox("Segmenter", SEGMENTERS)
        sampling = st.selectbox("Sampling", SAMPLING_STRATEGIES, help="Tissue sampling strategy for multi-dataset training")

    with col2:
        patch_size = st.selectbox("Patch size", PATCH_SIZES, index=2)
        backbone = st.selectbox("Backbone", BACKBONES)
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64, step=8)

    with col3:
        gpu_queue = st.selectbox("GPU queue", QUEUES, index=0, help="ClearML queue for GPU-heavy steps")
        default_queue = st.selectbox("CPU queue", QUEUES, index=3, help="ClearML queue for CPU steps")
        local = st.checkbox("Run locally", value=False, help="Run on this machine instead of ClearML agents")
        skip_training = st.checkbox("Skip training", value=False, help="Prepare-only mode (create LMDB datasets)")
        fine_grained = st.checkbox("Fine-grained classes", value=False, help="~20 classes instead of 3 broad categories")
        validate = st.checkbox("Cross-modal validation", value=False)

    return {
        "recipe": recipe,
        "annotator": annotator,
        "segmenter": segmenter,
        "sampling": sampling,
        "patch_size": patch_size,
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "gpu_queue": gpu_queue,
        "default_queue": default_queue,
        "local": local,
        "skip_training": skip_training,
        "fine_grained": fine_grained,
        "validate": validate,
    }


def build_cli_command(config: dict) -> list[str]:
    """Build the CLI command from config and selected datasets."""
    cmd = ["uv", "run", "dapidl", "clearml-pipeline", "run"]

    for tissue, source, platform, tier in st.session_state.datasets:
        cmd.extend(["-t", tissue, source, platform, str(tier)])

    cmd.extend(["--recipe", config["recipe"]])
    cmd.extend(["--annotator", config["annotator"]])
    cmd.extend(["--segmenter", config["segmenter"]])
    cmd.extend(["--sampling", config["sampling"]])
    cmd.extend(["--patch-size", config["patch_size"]])
    cmd.extend(["--backbone", config["backbone"]])
    cmd.extend(["--epochs", str(config["epochs"])])
    cmd.extend(["--batch-size", str(config["batch_size"])])
    cmd.extend(["--gpu-queue", config["gpu_queue"]])
    cmd.extend(["--default-queue", config["default_queue"]])

    if config["local"]:
        cmd.append("--local")
    if config["skip_training"]:
        cmd.append("--skip-training")
    if config["fine_grained"]:
        cmd.append("--fine-grained")
    if config["validate"]:
        cmd.append("--validate")

    return cmd


# ---------------------------------------------------------------------------
# Pipeline launcher
# ---------------------------------------------------------------------------

def launch_section(config: dict) -> None:
    """Render the launch button and command preview."""
    st.subheader("Launch")

    if not st.session_state.datasets:
        st.warning("Add at least one dataset to launch a pipeline.")
        return

    cmd = build_cli_command(config)
    preview = " ".join(cmd)
    st.code(preview, language="bash")

    col1, _col2 = st.columns([1, 3])

    if col1.button("Launch Pipeline", type="primary", use_container_width=True):
        env = os.environ.copy()
        env["CLEARML_CONFIG_FILE"] = str(CLEARML_CONFIG_PATH)
        # Clear any conflicting env vars
        env.pop("CLEARML_API_ACCESS_KEY", None)
        env.pop("CLEARML_API_SECRET_KEY", None)

        try:
            result = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            st.session_state.running_process = result
            st.session_state.launch_log = ""
            st.success("Pipeline launched! Check the Monitor tab for progress.")
        except Exception as e:
            st.error(f"Failed to launch: {e}")

    # Show live output from running process
    if st.session_state.running_process is not None:
        proc = st.session_state.running_process
        stdout = proc.stdout
        if proc.poll() is None:
            st.info("Pipeline process is running...")
            # Non-blocking read on Unix
            if stdout is not None:
                try:
                    import fcntl
                    fd = stdout.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    try:
                        chunk = stdout.read(4096)
                        if chunk:
                            st.session_state.launch_log += chunk
                    except (BlockingIOError, TypeError):
                        pass
                except Exception:
                    pass
        else:
            if stdout is not None:
                remaining = stdout.read()
                if remaining:
                    st.session_state.launch_log += remaining
            rc = proc.returncode
            if rc == 0:
                st.success(f"Pipeline process exited successfully (code {rc})")
            else:
                st.error(f"Pipeline process failed (code {rc})")
            st.session_state.running_process = None

        if st.session_state.launch_log:
            st.code(st.session_state.launch_log[-3000:], language="text")


# ---------------------------------------------------------------------------
# Pipeline monitor
# ---------------------------------------------------------------------------

STATUS_COLORS = {
    "completed": ":green[completed]",
    "in_progress": ":orange[in_progress]",
    "failed": ":red[failed]",
    "stopped": ":red[stopped]",
    "created": ":blue[created]",
    "queued": ":blue[queued]",
    "published": ":violet[published]",
}


def pipeline_monitor(client: ClearMLClient) -> None:
    """Show recent pipelines and their step status."""
    st.subheader("Recent Pipelines")

    if st.button("Refresh", key="refresh_pipelines"):
        st.rerun()

    pipelines = client.get_pipelines(limit=15)

    if not pipelines:
        st.info("No pipelines found on the ClearML server.")
        return

    for pipe in pipelines:
        status_display = STATUS_COLORS.get(pipe["status"], pipe["status"])
        last_update = pipe.get("last_update", "")
        if last_update:
            try:
                dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
                last_update = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, AttributeError):
                pass

        with st.expander(f"{pipe['name']}  —  {status_display}  ({last_update})", expanded=pipe["status"] == "in_progress"):
            st.caption(f"ID: `{pipe['id']}`  |  Project: `{pipe.get('project', {}).get('name', 'N/A') if isinstance(pipe.get('project'), dict) else pipe.get('project', 'N/A')}`")

            # Link to ClearML web UI
            web_url = client.web_server or "https://clearml.chrism.io"
            st.markdown(f"[Open in ClearML]({web_url}/projects/*/experiments/{pipe['id']})")

            # Fetch steps
            try:
                steps = client.get_pipeline_steps(pipe["id"])
                if steps:
                    for step in steps:
                        step_status = STATUS_COLORS.get(step["status"], step["status"])
                        st.markdown(f"  - **{step['name']}** — {step_status}")
                else:
                    st.caption("No child steps found.")
            except Exception as e:
                st.caption(f"Could not load steps: {e}")


# ---------------------------------------------------------------------------
# Dataset browser
# ---------------------------------------------------------------------------

def dataset_browser(client: ClearMLClient) -> None:
    """Browse datasets registered on ClearML."""
    st.subheader("Registered Datasets")

    if st.button("Refresh", key="refresh_datasets"):
        st.rerun()

    datasets = client.get_datasets()

    if not datasets:
        st.info("No datasets found.")
        return

    # Group by project
    by_project: dict[str, list[dict]] = {}
    for d in datasets:
        proj = d.get("project", {})
        proj_name = proj.get("name", "Unknown") if isinstance(proj, dict) else str(proj)
        by_project.setdefault(proj_name, []).append(d)

    for proj_name in sorted(by_project):
        with st.expander(f"{proj_name} ({len(by_project[proj_name])} datasets)"):
            for d in by_project[proj_name]:
                tags = ", ".join(d.get("tags", [])[:5]) if d.get("tags") else ""
                tag_str = f"  `{tags}`" if tags else ""
                st.markdown(f"- **{d['name']}**{tag_str}  \n  `{d['id']}`")


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DAPIDL Pipeline Launcher",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("DAPIDL Pipeline Launcher")
    st.caption("Configure, launch, and monitor cell-type prediction pipelines on ClearML")

    init_state()

    try:
        client = get_client()
    except Exception as e:
        st.error(f"Could not connect to ClearML: {e}")
        st.info(f"Ensure {CLEARML_CONFIG_PATH} exists with valid credentials.")
        return

    # Tabs for different views
    tab_launch, tab_monitor, tab_datasets = st.tabs(["Launch Pipeline", "Monitor", "Datasets"])

    with tab_launch:
        dataset_picker(client)
        st.divider()
        config = pipeline_config()
        st.divider()
        launch_section(config)

    with tab_monitor:
        pipeline_monitor(client)

    with tab_datasets:
        dataset_browser(client)

    # Sidebar: server info
    with st.sidebar:
        st.markdown("### Server")
        st.caption(f"API: `{client.api_server}`")
        st.caption(f"Web: `{client.web_server}`")
        st.caption(f"Config: `{CLEARML_CONFIG_PATH}`")

        st.markdown("### Quick Reference")
        st.markdown("""
**Recipes:**
- `default` — annotation + standardization + LMDB
- `gt` — ground truth + LMDB
- `no_cl` — annotation + LMDB (skip CL standardization)
- `annotate_only` — annotation + standardization only

**Confidence Tiers:**
- Tier 1 = Ground truth labels
- Tier 2 = Consensus annotations
- Tier 3 = Single-method predictions

**Queues:**
- `gpu-local` — local RTX 3090
- `gpu-cloud` — AWS g6.xlarge spot
- `gpu-training` — shared GPU pool
- `cpu-local` — local CPU agent
        """)


if __name__ == "__main__":
    main()
