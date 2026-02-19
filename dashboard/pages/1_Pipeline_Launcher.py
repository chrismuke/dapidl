"""Pipeline Launcher — dataset picker, config, and CLI command builder."""

from __future__ import annotations

import streamlit as st

from components.auth import get_current_user, logout_button, require_auth
from components.clearml_client import ClearMLClient
from components.constants import (
    ANNOTATORS,
    BACKBONES,
    BUILTIN_RECIPES,
    PATCH_SIZES,
    PIPELINE_TEMPLATE_TASK_ID,
    QUEUES,
    RECIPE_DESCRIPTIONS,
    SAMPLING_STRATEGIES,
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
# Dataset picker
# ---------------------------------------------------------------------------

def dataset_picker(client: ClearMLClient) -> None:
    st.subheader("Datasets")

    clearml_datasets = client.get_datasets()
    dataset_names = {d["id"]: d["name"] for d in clearml_datasets}

    # Currently selected
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

    # Add dataset form
    with st.expander("Add dataset", expanded=not st.session_state.datasets):
        c1, c2 = st.columns(2)
        tissue = c1.text_input("Tissue name", placeholder="e.g. breast, lung", key="new_tissue")
        platform = c2.selectbox("Platform", ["xenium", "merscope"], key="new_platform")

        # Single searchable selectbox — click and type to filter
        source_options = ["-- Manual ID/path --"] + [
            f"{d['name']}  ({d['id'][:8]}...)" for d in clearml_datasets
        ]
        source_choice = st.selectbox(
            "Dataset source (type to search)",
            source_options,
            key="new_source_choice",
            help="Click and start typing to filter datasets by name",
        )
        if source_choice == "-- Manual ID/path --":
            source_id = st.text_input("ClearML dataset ID or local path", key="new_source_id")
        else:
            idx = source_options.index(source_choice) - 1
            source_id = clearml_datasets[idx]["id"]
            st.caption(f"ID: `{source_id}`")

        tier = st.selectbox(
            "Confidence tier", [1, 2, 3], index=1, key="new_tier",
            help="1=Ground truth, 2=Consensus, 3=Single-method",
        )

        if st.button("Add dataset", type="primary", disabled=not (tissue and source_id)):
            st.session_state.datasets.append((tissue, source_id, platform, tier))
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
        sampling = st.selectbox("Sampling", SAMPLING_STRATEGIES, help="Tissue sampling strategy")

    with col2:
        patch_size = st.selectbox("Patch size", PATCH_SIZES, index=2)
        backbone = st.selectbox("Backbone", BACKBONES)
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64, step=8)

    with col3:
        gpu_queue = st.selectbox("GPU queue", QUEUES, index=0, help="Queue for GPU-heavy steps")
        default_queue = st.selectbox("CPU queue", QUEUES, index=3, help="Queue for CPU steps")
        local = st.checkbox("Run locally", value=False)
        skip_training = st.checkbox("Skip training", value=False, help="Prepare-only mode")
        fine_grained = st.checkbox("Fine-grained classes", value=False, help="~20 classes vs 3 broad")
        validate = st.checkbox("Cross-modal validation", value=False)
        no_cache = st.checkbox("Disable step caching", value=False, help="Force all steps to re-run")

    return {
        "recipe": recipe, "annotator": annotator, "segmenter": segmenter,
        "sampling": sampling, "patch_size": patch_size, "backbone": backbone,
        "epochs": epochs, "batch_size": batch_size, "gpu_queue": gpu_queue,
        "default_queue": default_queue, "local": local, "skip_training": skip_training,
        "fine_grained": fine_grained, "validate": validate, "no_cache": no_cache,
    }


# ---------------------------------------------------------------------------
# CLI command builder
# ---------------------------------------------------------------------------

def build_clearml_params(config: dict) -> dict[str, str]:
    """Convert UI form state to flat ClearML hyperparameter dict.

    Keys match DAPIDLPipelineConfig.to_clearml_parameters() format so the
    controller script can reconstruct a full config via from_clearml_parameters().
    """
    params: dict[str, str] = {}

    # Datasets spec — one line per dataset: "dataset_id" (simplified)
    lines = []
    for tissue, source, platform, tier in st.session_state.datasets:
        lines.append(source)
    params["datasets/spec"] = "\n".join(lines)

    # Training
    params["training/epochs"] = str(config["epochs"])
    params["training/batch_size"] = str(config["batch_size"])
    params["training/backbone"] = config["backbone"]
    params["training/sampling_strategy"] = config["sampling"]

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


def build_cli_command(config: dict) -> str:
    """Build the CLI command string from config and selected datasets."""
    parts = ["uv run dapidl clearml-pipeline run"]

    for tissue, source, platform, tier in st.session_state.datasets:
        parts.append(f"  -t {tissue} {source} {platform} {tier}")

    parts.append(f"  --recipe {config['recipe']}")
    parts.append(f"  --annotator {config['annotator']}")
    parts.append(f"  --segmenter {config['segmenter']}")
    parts.append(f"  --sampling {config['sampling']}")
    parts.append(f"  --patch-size {config['patch_size']}")
    parts.append(f"  --backbone {config['backbone']}")
    parts.append(f"  --epochs {config['epochs']}")
    parts.append(f"  --batch-size {config['batch_size']}")
    parts.append(f"  --gpu-queue {config['gpu_queue']}")
    parts.append(f"  --default-queue {config['default_queue']}")

    if config["local"]:
        parts.append("  --local")
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

    # Launch section
    st.subheader("Launch Pipeline")

    if not st.session_state.datasets:
        st.warning("Add at least one dataset to build a pipeline command.")
    else:
        user = get_current_user()
        cmd = build_cli_command(config)
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
                    # Controllers must run on 'services' queue (long-running
                    # orchestrator handled by clearml-agent-services).  Using
                    # cpu-local would deadlock since the controller occupies
                    # the only worker while its child tasks wait in the queue.
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
