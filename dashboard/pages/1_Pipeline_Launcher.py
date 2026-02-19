"""Pipeline Launcher — dataset picker, config, and CLI command builder."""

from __future__ import annotations

import re

import streamlit as st

from components.auth import get_current_user, logout_button, require_auth
from components.clearml_client import ClearMLClient
from components.constants import (
    ANNOTATORS,
    BACKBONES,
    BUILTIN_RECIPES,
    GPU_TARGETS,
    PATCH_SIZES,
    PIPELINE_TEMPLATE_TASK_ID,
    RECIPE_DESCRIPTIONS,
    SEGMENTERS,
)
from components.ui_helpers import recipe_flow

st.set_page_config(page_title="Pipeline Launcher", page_icon=":rocket:", layout="wide")
if not require_auth():
    st.stop()
logout_button()
st.title("Pipeline Launcher")


def get_client() -> ClearMLClient:
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


def init_state() -> None:
    if "datasets" not in st.session_state:
        st.session_state.datasets = []


# ---------------------------------------------------------------------------
# Dataset name parsing — derive tissue & platform from naming convention
# ---------------------------------------------------------------------------

_RAW_PATTERN = re.compile(
    r"^(?P<platform>xenium|merscope)-(?P<tissue>[a-z_-]+?)-.*-raw$"
)


def parse_dataset_name(name: str) -> tuple[str, str]:
    """Extract (platform, tissue) from dataset name like 'xenium-breast-cancer-rep1-raw'.

    Returns ("unknown", "unknown") if the name doesn't match the convention.
    """
    m = _RAW_PATTERN.match(name)
    if m:
        return m.group("platform"), m.group("tissue")
    return "unknown", "unknown"


# ---------------------------------------------------------------------------
# Dataset picker — only raw datasets, auto-derive tissue/platform
# ---------------------------------------------------------------------------

def dataset_picker(client: ClearMLClient) -> None:
    st.subheader("Datasets")

    clearml_datasets = client.get_datasets()
    # Filter to raw datasets only
    raw_datasets = [d for d in clearml_datasets if d["name"].endswith("-raw")]
    dataset_names = {d["id"]: d["name"] for d in raw_datasets}

    # Currently selected
    if st.session_state.datasets:
        st.markdown("**Selected datasets:**")
        to_remove = None
        for i, (source_id, tier) in enumerate(st.session_state.datasets):
            name = dataset_names.get(source_id, source_id[:12] + "...")
            platform, tissue = parse_dataset_name(name)
            cols = st.columns([4, 1, 1, 1])
            cols[0].text(f"{name}")
            cols[1].text(f"{platform}/{tissue}")
            cols[2].text(f"Tier {tier}")
            if cols[3].button("Remove", key=f"rm_{i}"):
                to_remove = i
        if to_remove is not None:
            st.session_state.datasets.pop(to_remove)
            st.rerun()
    else:
        st.info("No datasets added yet. Add one below.")

    # Add dataset form
    with st.expander("Add dataset", expanded=not st.session_state.datasets):
        # Single searchable selectbox — only raw datasets
        source_options = ["-- Manual ID/path --"] + [
            f"{d['name']}  ({d['id'][:8]}...)" for d in raw_datasets
        ]
        source_choice = st.selectbox(
            "Dataset source (type to search)",
            source_options,
            key="new_source_choice",
            help="Only raw datasets are shown. Derived datasets (LMDB, annotations) are auto-cached by the pipeline.",
        )
        if source_choice == "-- Manual ID/path --":
            source_id = st.text_input("ClearML dataset ID or local path", key="new_source_id")
        else:
            idx = source_options.index(source_choice) - 1
            source_id = raw_datasets[idx]["id"]
            name = raw_datasets[idx]["name"]
            platform, tissue = parse_dataset_name(name)
            st.caption(f"ID: `{source_id}` | platform: **{platform}** | tissue: **{tissue}**")

        tier = st.selectbox(
            "Confidence tier", [1, 2, 3], index=1, key="new_tier",
            help="1=Ground truth, 2=Consensus, 3=Single-method",
        )

        if st.button("Add dataset", type="primary", disabled=not source_id):
            st.session_state.datasets.append((source_id, tier))
            st.rerun()


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

def pipeline_config() -> dict:
    st.subheader("Pipeline Configuration")

    # Recipe selector with flow visualization
    recipe_names = list(BUILTIN_RECIPES.keys())
    recipe = st.selectbox(
        "Recipe",
        recipe_names,
        format_func=lambda r: f"{r} — {RECIPE_DESCRIPTIONS.get(r, '')}",
    )
    st.caption(recipe_flow(recipe, BUILTIN_RECIPES[recipe]))

    col1, col2, col3 = st.columns(3)

    with col1:
        annotator = st.selectbox("Annotator", ANNOTATORS)
        segmenter = st.selectbox("Segmenter", SEGMENTERS)

    with col2:
        patch_size = st.selectbox("Patch size", PATCH_SIZES, index=2)
        backbone = st.selectbox("Backbone", BACKBONES)
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64, step=8)

    with col3:
        gpu_target = st.radio(
            "Run on",
            list(GPU_TARGETS.keys()),
            help="Local: split CPU/GPU steps. Cloud: all steps on one AWS instance.",
        )
        gpu_queue, default_queue = GPU_TARGETS[gpu_target]
        skip_training = st.checkbox("Skip training", value=False, help="Prepare-only mode")
        fine_grained = st.checkbox("Fine-grained classes", value=False, help="~20 classes vs 3 broad")
        validate = st.checkbox("Cross-modal validation", value=False)
        no_cache = st.checkbox("Disable step caching", value=False, help="Force all steps to re-run")

    return {
        "recipe": recipe, "annotator": annotator, "segmenter": segmenter,
        "patch_size": patch_size, "backbone": backbone,
        "epochs": epochs, "batch_size": batch_size, "gpu_queue": gpu_queue,
        "default_queue": default_queue, "skip_training": skip_training,
        "fine_grained": fine_grained, "validate": validate, "no_cache": no_cache,
    }


# ---------------------------------------------------------------------------
# ClearML parameter builder
# ---------------------------------------------------------------------------

def build_clearml_params(config: dict) -> dict[str, str]:
    """Convert UI form state to flat ClearML hyperparameter dict.

    Keys match DAPIDLPipelineConfig.to_clearml_parameters() format so the
    controller script can reconstruct a full config via from_clearml_parameters().
    """
    params: dict[str, str] = {}

    # Datasets spec — one line per "tissue source_id platform tier"
    # Derive tissue and platform from the dataset name via ClearML lookup
    lines = []
    for source_id, tier in st.session_state.datasets:
        lines.append(source_id)
    params["datasets/spec"] = "\n".join(lines)

    # Training
    params["training/epochs"] = str(config["epochs"])
    params["training/batch_size"] = str(config["batch_size"])
    params["training/backbone"] = config["backbone"]
    params["training/sampling_strategy"] = "sqrt"

    # Execution
    params["execution/gpu_queue"] = config["gpu_queue"]
    params["execution/default_queue"] = config["default_queue"]
    params["execution/execute_remotely"] = "True"
    params["execution/skip_training"] = str(config["skip_training"])
    params["execution/cache_data_steps"] = str(not config["no_cache"])

    # Annotation
    params["annotation/fine_grained"] = str(config["fine_grained"])

    # Segmentation
    params["segmentation/segmenter"] = config["segmenter"]

    # LMDB
    params["lmdb/patch_sizes"] = config["patch_size"]

    # Validation
    params["validation/enabled"] = str(config["validate"])

    # Pipeline metadata
    params["pipeline/project"] = "DAPIDL/pipelines"

    return params


def build_cli_command(config: dict, dataset_names: dict[str, str]) -> str:
    """Build the CLI command string from config and selected datasets."""
    parts = ["uv run dapidl clearml-pipeline run"]

    for source_id, tier in st.session_state.datasets:
        name = dataset_names.get(source_id, "")
        platform, tissue = parse_dataset_name(name) if name else ("unknown", "unknown")
        parts.append(f"  -t {tissue} {source_id} {platform} {tier}")

    parts.append(f"  --recipe {config['recipe']}")
    parts.append(f"  --annotator {config['annotator']}")
    parts.append(f"  --segmenter {config['segmenter']}")
    parts.append(f"  --patch-size {config['patch_size']}")
    parts.append(f"  --backbone {config['backbone']}")
    parts.append(f"  --epochs {config['epochs']}")
    parts.append(f"  --batch-size {config['batch_size']}")
    parts.append(f"  --gpu-queue {config['gpu_queue']}")
    parts.append(f"  --default-queue {config['default_queue']}")

    if config["skip_training"]:
        parts.append("  --skip-training")
    if config["fine_grained"]:
        parts.append("  --fine-grained")
    if config["validate"]:
        parts.append("  --validate")
    if config["no_cache"]:
        parts.append("  --no-cache")

    return " \\\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()

    try:
        client = get_client()
    except Exception as exc:
        st.error(f"Could not connect to ClearML: {exc}")
        return

    dataset_picker(client)
    st.divider()
    config = pipeline_config()
    st.divider()

    # GPU status indicator
    with st.expander("Worker status", expanded=False):
        try:
            workers = client.get_worker_status()
            if workers:
                for w in workers:
                    task_info = f"running **{w['task']}**" if w["task"] else "idle"
                    gpu_info = f" ({w['gpu_usage']})" if w["gpu_usage"] else ""
                    st.markdown(f"- `{w['id']}` — {task_info}{gpu_info}")
            else:
                st.caption("No workers registered.")
        except Exception:
            st.caption("Could not fetch worker status.")

    # Launch section
    st.subheader("Launch Pipeline")

    if not st.session_state.datasets:
        st.warning("Add at least one dataset to build a pipeline command.")
    else:
        user = get_current_user()
        # Build name lookup for CLI preview
        all_datasets = client.get_datasets()
        dataset_names = {d["id"]: d["name"] for d in all_datasets}
        cmd = build_cli_command(config, dataset_names)
        if user:
            cmd += f"  # launched by {user}"

        # Command preview
        with st.expander("CLI command preview", expanded=False):
            st.code(cmd, language="bash")
            st.caption("Or copy this command to run manually on a machine with the dapidl environment.")

        # Remote launch via ClearML REST API
        if st.button("Launch Pipeline", type="primary"):
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            task_name = f"pipeline-{timestamp}"
            if user:
                task_name = f"pipeline-{user}-{timestamp}"

            with st.spinner("Cloning template and enqueuing..."):
                new_id = client.clone_task(PIPELINE_TEMPLATE_TASK_ID, task_name)
                if not new_id:
                    st.error("Failed to clone template task. Check ClearML connection.")
                else:
                    params = build_clearml_params(config)
                    client.edit_task_hyperparams(new_id, params)
                    queue = "services"
                    ok = client.enqueue_task(new_id, queue)
                    if ok:
                        web_host = client.web_server or "https://clearml.chrism.io"
                        st.success(
                            f"Pipeline launched! "
                            f"[View in ClearML]({web_host}/projects/*/experiments/{new_id})"
                        )
                    else:
                        st.error(f"Failed to enqueue task {new_id} to queue '{queue}'.")


main()
