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
        # Search filter
        search = st.text_input("Search datasets", placeholder="Filter by name...", key="ds_search")
        filtered = clearml_datasets
        if search:
            search_lower = search.lower()
            filtered = [d for d in clearml_datasets if search_lower in d["name"].lower()]

        c1, c2 = st.columns(2)
        tissue = c1.text_input("Tissue name", placeholder="e.g. breast, lung", key="new_tissue")
        platform = c2.selectbox("Platform", ["xenium", "merscope"], key="new_platform")

        source_options = ["-- Manual ID/path --"] + [
            f"{d['name']}  ({d['id'][:8]}...)" for d in filtered
        ]
        source_choice = st.selectbox("Dataset source", source_options, key="new_source_choice")
        if source_choice == "-- Manual ID/path --":
            source_id = st.text_input("ClearML dataset ID or local path", key="new_source_id")
        else:
            idx = source_options.index(source_choice) - 1
            source_id = filtered[idx]["id"]
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

    return {
        "recipe": recipe, "annotator": annotator, "segmenter": segmenter,
        "sampling": sampling, "patch_size": patch_size, "backbone": backbone,
        "epochs": epochs, "batch_size": batch_size, "gpu_queue": gpu_queue,
        "default_queue": default_queue, "local": local, "skip_training": skip_training,
        "fine_grained": fine_grained, "validate": validate,
    }


# ---------------------------------------------------------------------------
# CLI command builder
# ---------------------------------------------------------------------------

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

    # Launch section (command preview only — Docker can't run uv)
    st.subheader("Command Preview")

    if not st.session_state.datasets:
        st.warning("Add at least one dataset to build a pipeline command.")
    else:
        user = get_current_user()
        cmd = build_cli_command(config)
        if user:
            cmd += f"  # launched by {user}"
        st.code(cmd, language="bash")
        st.caption("Copy this command and run it on a machine with the dapidl environment.")


main()
