"""Pipeline Launcher — full-featured dataset picker, tabbed config, and launch."""

from __future__ import annotations

import datetime
import re

import streamlit as st
from components.auth import get_current_user, logout_button, require_auth
from components.clearml_client import ClearMLClient
from components.constants import (
    ANNOTATION_STRATEGIES,
    ANNOTATION_STRATEGY_DESCRIPTIONS,
    AUGMENTATION_LEVEL_DESCRIPTIONS,
    AUGMENTATION_LEVELS,
    BACKBONE_DESCRIPTIONS,
    BACKBONES,
    BUILTIN_RECIPES,
    CL_TARGET_LEVEL_DESCRIPTIONS,
    CL_TARGET_LEVELS,
    GPU_TARGETS,
    NORMALIZATION_METHOD_DESCRIPTIONS,
    NORMALIZATION_METHODS,
    PATCH_SIZES,
    PIPELINE_TEMPLATE_TASK_ID,
    RECIPE_DESCRIPTIONS,
    SAMPLING_STRATEGIES,
    SAMPLING_STRATEGY_DESCRIPTIONS,
    TRAINING_MODE_DESCRIPTIONS,
    TRAINING_MODES,
)

st.set_page_config(page_title="Pipeline Launcher", page_icon=":rocket:", layout="wide")
if not require_auth():
    st.stop()
logout_button()
st.title("Pipeline Launcher")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RAW_PATTERN = re.compile(
    r"^(?P<platform>xenium|merscope)-(?P<tissue>[a-z_-]+?)-.*-raw$"
)


def _parse_dataset_name(name: str) -> tuple[str, str]:
    """Extract (platform, tissue) from a raw dataset name."""
    m = _RAW_PATTERN.match(name)
    if m:
        return m.group("platform"), m.group("tissue")
    return "unknown", "unknown"


def _get_client() -> ClearMLClient:
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


def _init_state() -> None:
    if "datasets" not in st.session_state:
        st.session_state.datasets = []


# ---------------------------------------------------------------------------
# Dataset picker — raw datasets only, per-dataset tier/recipe/weight
# ---------------------------------------------------------------------------


def _dataset_picker(client: ClearMLClient) -> None:
    st.subheader("Datasets")

    raw_datasets = client.get_raw_datasets()

    # --- Show selected datasets with inline controls ---
    if st.session_state.datasets:
        to_remove: int | None = None
        for i, ds in enumerate(st.session_state.datasets):
            platform, tissue = _parse_dataset_name(ds["name"])
            cols = st.columns([3, 1, 1, 1, 1, 0.5])
            cols[0].markdown(f"**{ds['name']}**  \n`{platform}` / `{tissue}`")
            new_tier = cols[1].selectbox(
                "Tier", [1, 2, 3], index=ds["tier"] - 1,
                key=f"tier_{i}", format_func=lambda t: f"T{t}",
            )
            new_recipe = cols[2].selectbox(
                "Recipe", list(BUILTIN_RECIPES.keys()),
                index=list(BUILTIN_RECIPES.keys()).index(ds.get("recipe", "default")),
                key=f"recipe_{i}",
            )
            new_weight = cols[3].number_input(
                "Weight", min_value=0.1, max_value=10.0, value=ds.get("weight", 1.0),
                step=0.1, key=f"weight_{i}",
            )
            if cols[4].button("Remove", key=f"rm_{i}"):
                to_remove = i
            # Update in-place
            ds["tier"] = new_tier
            ds["recipe"] = new_recipe
            ds["weight"] = new_weight

        if to_remove is not None:
            st.session_state.datasets.pop(to_remove)
            st.rerun()
    else:
        st.info("No datasets added yet. Add one below.")

    # --- Add dataset form ---
    with st.expander("Add dataset", expanded=not st.session_state.datasets):
        source_options = ["-- Manual ID/path --"] + [
            f"{d['name']}  ({d['id'][:8]}...)" for d in raw_datasets
        ]
        source_choice = st.selectbox(
            "Dataset source (type to search)", source_options, key="new_source_choice",
            help="Only raw datasets are shown.",
        )

        if source_choice == "-- Manual ID/path --":
            source_id = st.text_input("ClearML dataset ID or local path", key="new_source_id")
            source_name = source_id
        else:
            idx = source_options.index(source_choice) - 1
            source_id = raw_datasets[idx]["id"]
            source_name = raw_datasets[idx]["name"]
            platform, tissue = _parse_dataset_name(source_name)
            st.caption(f"ID: `{source_id}` | platform: **{platform}** | tissue: **{tissue}**")

        add_cols = st.columns(3)
        add_tier = add_cols[0].selectbox(
            "Confidence tier", [1, 2, 3], index=1, key="new_tier",
            help="1=Ground truth, 2=Consensus, 3=Single-method",
        )
        add_recipe = add_cols[1].selectbox(
            "Recipe", list(BUILTIN_RECIPES.keys()), key="new_recipe",
            format_func=lambda r: f"{r} — {RECIPE_DESCRIPTIONS.get(r, '')}",
        )
        add_weight = add_cols[2].number_input(
            "Weight", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="new_weight",
        )

        if st.button("Add dataset", type="primary", disabled=not source_id):
            st.session_state.datasets.append({
                "id": source_id,
                "name": source_name,
                "tier": add_tier,
                "recipe": add_recipe,
                "weight": add_weight,
            })
            st.rerun()


# ---------------------------------------------------------------------------
# Pipeline configuration — top bar + 7 tabs
# ---------------------------------------------------------------------------


def _pipeline_config() -> dict:
    """Render all config tabs and return a flat dict of all settings."""
    cfg: dict = {}

    # --- Top bar: pipeline name + GPU target ---
    top = st.columns([3, 2])
    cfg["pipeline_name"] = top[0].text_input("Pipeline name (optional)", value="", key="pipe_name")
    gpu_target = top[1].radio(
        "GPU target", list(GPU_TARGETS.keys()), horizontal=True,
        help="Local: split CPU/GPU steps. Cloud: all steps on one AWS instance.",
    )
    cfg["gpu_queue"], cfg["default_queue"] = GPU_TARGETS[gpu_target]

    # --- Tabs ---
    tabs = st.tabs([
        "Basic", "Annotation", "Ontology", "LMDB",
        "Training", "Validation", "Execution",
    ])

    # == Basic ==
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        cfg["backbone"] = c1.selectbox(
            "Backbone", BACKBONES,
            format_func=lambda b: BACKBONE_DESCRIPTIONS.get(b, b),
        )
        cfg["epochs"] = c1.number_input("Epochs", 1, 500, 100)
        cfg["batch_size"] = c1.number_input("Batch size", 8, 512, 64, step=8)

        cfg["patch_sizes"] = c2.multiselect(
            "Patch sizes", PATCH_SIZES, default=[128],
            help="Generate LMDB datasets at these sizes",
        )
        cfg["sampling_strategy"] = c2.selectbox(
            "Sampling strategy", SAMPLING_STRATEGIES,
            index=SAMPLING_STRATEGIES.index("sqrt"),
            format_func=lambda s: SAMPLING_STRATEGY_DESCRIPTIONS.get(s, s),
        )

        cfg["skip_training"] = c3.checkbox("Skip training (prepare-only)")
        cfg["fine_grained"] = c3.checkbox("Fine-grained classes", value=True, help="~20 vs 3 broad")

    # == Annotation ==
    with tabs[1]:
        c1, c2 = st.columns(2)
        cfg["annotation_strategy"] = c1.selectbox(
            "Strategy", ANNOTATION_STRATEGIES,
            format_func=lambda s: ANNOTATION_STRATEGY_DESCRIPTIONS.get(s, s),
        )

        import json as _json
        from pathlib import Path as _Path

        # Load method presets
        presets_path = _Path(__file__).parent.parent.parent / "configs" / "default_methods.json"
        presets: dict = {}
        if presets_path.exists():
            with open(presets_path) as f:
                presets = _json.load(f)

        preset_choice = c1.selectbox(
            "Method Preset",
            options=["custom"] + list(presets.keys()),
            index=list(presets.keys()).index("breast_standard") + 1
            if "breast_standard" in presets
            else 0,
        )

        if preset_choice != "custom":
            cfg["methods"] = presets[preset_choice]
            c2.json(cfg["methods"])
        else:
            methods_json = c1.text_area(
                "Methods (JSON)",
                value=_json.dumps(presets.get("breast_standard", []), indent=2),
                height=200,
            )
            try:
                cfg["methods"] = _json.loads(methods_json)
            except _json.JSONDecodeError:
                c1.error("Invalid JSON")
                cfg["methods"] = []
        cfg["confidence_threshold"] = c2.slider(
            "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
        )
        cfg["min_agreement"] = c2.number_input(
            "Min agreement", 1, 10, 2, help="Minimum annotators that must agree",
        )

        # Ground truth fields (conditional)
        if cfg["annotation_strategy"] == "ground_truth":
            st.divider()
            st.markdown("**Ground truth settings**")
            gc1, gc2 = st.columns(2)
            cfg["gt_file"] = gc1.text_input("Ground truth file path")
            cfg["gt_sheet"] = gc1.text_input("Excel sheet name (optional)")
            cfg["gt_cell_id_col"] = gc2.text_input("Cell ID column", value="Barcode")
            cfg["gt_label_col"] = gc2.text_input("Label column", value="Cluster")

    # == Ontology ==
    with tabs[2]:
        c1, c2 = st.columns(2)
        cfg["ontology_enabled"] = c1.checkbox("Enable CL standardization", value=True)
        cfg["cl_target_level"] = c1.selectbox(
            "Target level", CL_TARGET_LEVELS, index=1,
            format_func=lambda lv: CL_TARGET_LEVEL_DESCRIPTIONS.get(lv, lv),
            disabled=not cfg["ontology_enabled"],
        )
        cfg["cl_min_confidence"] = c2.slider(
            "Min confidence", 0.0, 1.0, 0.5, 0.05,
            disabled=not cfg["ontology_enabled"],
        )
        cfg["cl_fuzzy_threshold"] = c2.slider(
            "Fuzzy threshold", 0.0, 1.0, 0.85, 0.01,
            disabled=not cfg["ontology_enabled"],
        )
        cfg["cl_include_unmapped"] = c2.checkbox(
            "Include unmapped (as 'Unknown')", value=False,
            disabled=not cfg["ontology_enabled"],
        )

    # == LMDB ==
    with tabs[3]:
        c1, c2 = st.columns(2)
        cfg["normalization"] = c1.selectbox(
            "Normalization", NORMALIZATION_METHODS,
            format_func=lambda n: NORMALIZATION_METHOD_DESCRIPTIONS.get(n, n),
        )
        cfg["normalize_physical_size"] = c1.checkbox(
            "Normalize physical size (cross-platform)", value=True,
        )
        cfg["lmdb_skip_if_exists"] = c1.checkbox(
            "Skip if LMDB exists", value=True,
        )

        cfg["exclude_edge_cells"] = c2.checkbox("Exclude edge cells", value=True)
        cfg["edge_margin_px"] = c2.number_input(
            "Edge margin (px)", 0, 256, 64,
            disabled=not cfg["exclude_edge_cells"],
        )
        cfg["lmdb_min_confidence"] = c2.slider(
            "Min annotation confidence", 0.0, 1.0, 0.0, 0.05,
        )

    # == Training ==
    with tabs[4]:
        c1, c2, c3 = st.columns(3)

        cfg["training_mode"] = c1.selectbox(
            "Mode", TRAINING_MODES,
            index=TRAINING_MODES.index("hierarchical"),
            format_func=lambda m: TRAINING_MODE_DESCRIPTIONS.get(m, m),
        )
        cfg["learning_rate"] = c1.select_slider(
            "Learning rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
            value=1e-4, format_func=lambda x: f"{x:.0e}",
        )
        cfg["weight_decay"] = c1.select_slider(
            "Weight decay",
            options=[0.0, 1e-6, 1e-5, 1e-4, 1e-3],
            value=1e-5, format_func=lambda x: f"{x:.0e}",
        )
        cfg["dropout"] = c1.slider("Dropout", 0.0, 0.8, 0.3, 0.05)

        cfg["augmentation"] = c2.selectbox(
            "Augmentation", AUGMENTATION_LEVELS,
            index=AUGMENTATION_LEVELS.index("standard"),
            format_func=lambda a: AUGMENTATION_LEVEL_DESCRIPTIONS.get(a, a),
        )
        cfg["use_weighted_loss"] = c2.checkbox("Weighted loss", value=True)
        cfg["use_weighted_sampler"] = c2.checkbox("Weighted sampler", value=True)
        cfg["max_weight_ratio"] = c2.number_input(
            "Max weight ratio", 1.0, 100.0, 10.0, 1.0,
            help="Cap rare class weights",
        )

        cfg["val_split"] = c3.slider("Val split", 0.05, 0.3, 0.15, 0.05)
        cfg["test_split"] = c3.slider("Test split", 0.05, 0.3, 0.15, 0.05)
        cfg["patience"] = c3.number_input("Patience", 1, 50, 15)
        cfg["min_delta"] = c3.number_input(
            "Min delta", 0.0, 0.1, 0.001, 0.001, format="%.3f",
        )
        cfg["num_workers"] = c3.number_input("Num workers", 0, 16, 0)

        # Tier weights
        st.markdown("**Tier weights**")
        tw = st.columns(3)
        cfg["tier1_weight"] = tw[0].number_input("Tier 1 (GT)", 0.0, 5.0, 1.0, 0.1)
        cfg["tier2_weight"] = tw[1].number_input("Tier 2 (Consensus)", 0.0, 5.0, 0.8, 0.1)
        cfg["tier3_weight"] = tw[2].number_input("Tier 3 (Predicted)", 0.0, 5.0, 0.5, 0.1)

        # Curriculum params (conditional on hierarchical)
        if cfg["training_mode"] == "hierarchical":
            st.divider()
            st.markdown("**Curriculum learning**")
            cur = st.columns(3)
            cfg["coarse_only_epochs"] = cur[0].number_input("Phase 1 (coarse only)", 0, 100, 20)
            cfg["coarse_medium_epochs"] = cur[1].number_input("Phase 2 (coarse + medium)", 0, 200, 50)
            cfg["warmup_epochs"] = cur[2].number_input("Warmup epochs", 0, 20, 5)

    # == Validation ==
    with tabs[5]:
        cfg["validate"] = st.checkbox("Enable cross-modal validation", value=False)
        if cfg["validate"]:
            c1, c2 = st.columns(2)
            cfg["run_leiden_check"] = c1.checkbox("Leiden clustering check", value=True)
            cfg["run_dapi_check"] = c1.checkbox("DAPI model check", value=True)
            cfg["run_consensus_check"] = c1.checkbox("Consensus check", value=True)
            cfg["min_ari_threshold"] = c2.slider("Min ARI threshold", 0.0, 1.0, 0.5, 0.05)
            cfg["min_agreement_threshold"] = c2.slider("Min agreement threshold", 0.0, 1.0, 0.5, 0.05)

    # == Execution ==
    with tabs[6]:
        c1, c2 = st.columns(2)
        cfg["cache_data_steps"] = c1.checkbox("Cache data steps", value=True)
        cfg["cache_training"] = c1.checkbox("Cache training", value=False)
        cfg["pipeline_project"] = c2.text_input("Pipeline project", value="DAPIDL/pipelines")

    return cfg


# ---------------------------------------------------------------------------
# ClearML parameter builder
# ---------------------------------------------------------------------------


def _build_clearml_params(cfg: dict) -> dict[str, str]:
    """Convert UI form state to flat ClearML hyperparameter dict.

    Keys match DAPIDLPipelineConfig.to_clearml_parameters() format.
    """
    params: dict[str, str] = {}

    # -- Datasets spec: "name [recipe=X] [weight]" per line --
    lines: list[str] = []
    for ds in st.session_state.datasets:
        parts = [ds["name"] if ds["name"] != ds["id"] else ds["id"]]
        if ds.get("recipe", "default") != "default":
            parts.append(f"recipe={ds['recipe']}")
        if ds.get("weight", 1.0) != 1.0:
            parts.append(str(ds["weight"]))
        lines.append(" ".join(parts))
    params["datasets/spec"] = "\n".join(lines)

    # -- Annotation --
    params["annotation/strategy"] = cfg["annotation_strategy"]
    import json as _json

    params["annotation/methods"] = _json.dumps(cfg.get("methods", []))
    params["annotation/confidence_threshold"] = str(cfg["confidence_threshold"])
    params["annotation/min_agreement"] = str(cfg["min_agreement"])
    params["annotation/fine_grained"] = str(cfg["fine_grained"])
    params["annotation/extended_consensus"] = str(cfg.get("extended_consensus", False))
    if cfg["annotation_strategy"] == "ground_truth":
        params["annotation/ground_truth_file"] = cfg.get("gt_file", "")
        params["annotation/ground_truth_sheet"] = cfg.get("gt_sheet", "")
        params["annotation/ground_truth_cell_id_col"] = cfg.get("gt_cell_id_col", "Barcode")
        params["annotation/ground_truth_label_col"] = cfg.get("gt_label_col", "Cluster")

    # -- Ontology --
    params["ontology/enabled"] = str(cfg["ontology_enabled"])
    params["ontology/target_level"] = cfg["cl_target_level"]
    params["ontology/min_confidence"] = str(cfg["cl_min_confidence"])
    params["ontology/fuzzy_threshold"] = str(cfg["cl_fuzzy_threshold"])
    params["ontology/include_unmapped"] = str(cfg["cl_include_unmapped"])

    # -- LMDB --
    params["lmdb/patch_sizes"] = ",".join(str(p) for p in cfg["patch_sizes"])
    params["lmdb/normalization"] = cfg["normalization"]
    params["lmdb/normalize_physical_size"] = str(cfg["normalize_physical_size"])
    params["lmdb/exclude_edge_cells"] = str(cfg["exclude_edge_cells"])
    params["lmdb/edge_margin_px"] = str(cfg["edge_margin_px"])
    params["lmdb/min_confidence"] = str(cfg["lmdb_min_confidence"])
    params["lmdb/skip_if_exists"] = str(cfg["lmdb_skip_if_exists"])

    # -- Training --
    params["training/backbone"] = cfg["backbone"]
    params["training/mode"] = cfg["training_mode"]
    params["training/epochs"] = str(cfg["epochs"])
    params["training/batch_size"] = str(cfg["batch_size"])
    params["training/learning_rate"] = str(cfg["learning_rate"])
    params["training/weight_decay"] = str(cfg["weight_decay"])
    params["training/dropout"] = str(cfg["dropout"])
    params["training/augmentation"] = cfg["augmentation"]
    params["training/use_weighted_loss"] = str(cfg["use_weighted_loss"])
    params["training/use_weighted_sampler"] = str(cfg["use_weighted_sampler"])
    params["training/max_weight_ratio"] = str(cfg["max_weight_ratio"])
    params["training/val_split"] = str(cfg["val_split"])
    params["training/test_split"] = str(cfg["test_split"])
    params["training/patience"] = str(cfg["patience"])
    params["training/min_delta"] = str(cfg["min_delta"])
    params["training/num_workers"] = str(cfg["num_workers"])
    params["training/sampling_strategy"] = cfg["sampling_strategy"]
    params["training/tier1_weight"] = str(cfg["tier1_weight"])
    params["training/tier2_weight"] = str(cfg["tier2_weight"])
    params["training/tier3_weight"] = str(cfg["tier3_weight"])
    if cfg["training_mode"] == "hierarchical":
        params["training/coarse_only_epochs"] = str(cfg.get("coarse_only_epochs", 20))
        params["training/coarse_medium_epochs"] = str(cfg.get("coarse_medium_epochs", 50))
        params["training/warmup_epochs"] = str(cfg.get("warmup_epochs", 5))

    # -- Validation --
    params["validation/enabled"] = str(cfg.get("validate", False))
    if cfg.get("validate"):
        params["validation/run_leiden_check"] = str(cfg.get("run_leiden_check", True))
        params["validation/run_dapi_check"] = str(cfg.get("run_dapi_check", True))
        params["validation/run_consensus_check"] = str(cfg.get("run_consensus_check", True))
        params["validation/min_ari_threshold"] = str(cfg.get("min_ari_threshold", 0.5))
        params["validation/min_agreement_threshold"] = str(cfg.get("min_agreement_threshold", 0.5))

    # -- Execution --
    params["execution/gpu_queue"] = cfg["gpu_queue"]
    params["execution/default_queue"] = cfg["default_queue"]
    params["execution/execute_remotely"] = "True"
    params["execution/skip_training"] = str(cfg["skip_training"])
    params["execution/cache_data_steps"] = str(cfg["cache_data_steps"])
    params["execution/cache_training"] = str(cfg["cache_training"])

    # -- Pipeline metadata --
    params["pipeline/project"] = cfg.get("pipeline_project", "DAPIDL/pipelines")

    return params


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------


def _build_cli_preview(cfg: dict) -> str:
    """Build an approximate CLI command string for reference."""
    parts = ["uv run dapidl clearml-pipeline run"]

    for ds in st.session_state.datasets:
        name = ds["name"]
        platform, tissue = _parse_dataset_name(name)
        parts.append(f"  -t {tissue} {ds['id']} {platform} {ds['tier']}")

    parts.append(f"  --backbone {cfg['backbone']}")
    parts.append(f"  --epochs {cfg['epochs']}")
    parts.append(f"  --batch-size {cfg['batch_size']}")
    if cfg["patch_sizes"]:
        parts.append(f"  --patch-size {','.join(str(p) for p in cfg['patch_sizes'])}")
    parts.append(f"  --sampling {cfg['sampling_strategy']}")
    parts.append(f"  --gpu-queue {cfg['gpu_queue']}")
    parts.append(f"  --default-queue {cfg['default_queue']}")

    if cfg["skip_training"]:
        parts.append("  --skip-training")
    if not cfg["fine_grained"]:
        parts.append("  --no-fine-grained")
    if cfg.get("validate"):
        parts.append("  --validate")
    if not cfg["cache_data_steps"]:
        parts.append("  --no-cache")

    return " \\\n".join(parts)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def _launch_section(client: ClearMLClient, cfg: dict) -> None:
    st.subheader("Launch Pipeline")

    if not st.session_state.datasets:
        st.warning("Add at least one dataset to build a pipeline command.")
        return

    # Worker status
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

    # CLI preview
    cmd = _build_cli_preview(cfg)
    user = get_current_user()
    if user:
        cmd += f"  # launched by {user}"
    with st.expander("CLI command preview", expanded=False):
        st.code(cmd, language="bash")
        st.caption("Copy this command to run manually if preferred.")

    # Parameter preview
    with st.expander("ClearML parameters preview", expanded=False):
        params = _build_clearml_params(cfg)
        for key in sorted(params):
            val = params[key]
            if "\n" in val:
                st.text(f"{key}:")
                for line in val.splitlines():
                    st.text(f"  {line}")
            else:
                st.text(f"{key} = {val}")

    # Launch button
    if st.button("Launch Pipeline", type="primary"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        task_name = cfg.get("pipeline_name") or ""
        if not task_name:
            task_name = f"pipeline-{timestamp}"
        if user:
            task_name = f"{task_name}-{user}" if cfg.get("pipeline_name") else f"pipeline-{user}-{timestamp}"

        with st.spinner("Cloning template and enqueuing..."):
            new_id = client.clone_task(PIPELINE_TEMPLATE_TASK_ID, task_name)
            if not new_id:
                st.error("Failed to clone template task. Check ClearML connection.")
                return

            params = _build_clearml_params(cfg)
            client.edit_task_hyperparams(new_id, params)
            ok = client.enqueue_task(new_id, "services")
            if ok:
                web_host = client.web_server or "https://clearml.chrism.io"
                st.success(
                    f"Pipeline launched! "
                    f"[View in ClearML]({web_host}/projects/*/experiments/{new_id})"
                )
            else:
                st.error(f"Failed to enqueue task {new_id} to 'services' queue.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()

    try:
        client = _get_client()
    except Exception as exc:
        st.error(f"Could not connect to ClearML: {exc}")
        return

    _dataset_picker(client)
    st.divider()
    cfg = _pipeline_config()
    st.divider()
    _launch_section(client, cfg)


main()
