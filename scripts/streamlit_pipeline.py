"""DAPIDL Biologist Pipeline Studio — Streamlit GUI.

Configure, sweep, and analyze the DAPIDL pipeline on local / AWS / DigitalOcean
compute. Runs are launched by cloning a ClearML controller *template task*,
overriding its parameters, and enqueuing it — so runs execute on ClearML agents,
not this dashboard host.

Run with:
    uv run --with streamlit streamlit run scripts/streamlit_pipeline.py

Requires ClearML self-hosted credentials in ~/.clearml/clearml-chrism.conf and a
controller template task (see configs/studio.json / create-controller-task).
"""
from __future__ import annotations

import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Dependency-light studio logic (pure python + requests; no torch).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio import param_schema as ps  # noqa: E402
from studio.clearml_launch import CLEARML_CONFIG_PATH, ClearMLClient  # noqa: E402

STUDIO_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "studio.json"

STATUS_COLORS = {
    "completed": ":green[completed]", "in_progress": ":orange[in_progress]",
    "failed": ":red[failed]", "stopped": ":red[stopped]", "created": ":blue[created]",
    "queued": ":blue[queued]", "published": ":violet[published]",
}


# ---------------------------------------------------------------------------
# Config + session
# ---------------------------------------------------------------------------
def load_studio_config() -> dict:
    """Load configs/studio.json (compute targets, template id, services queue)."""
    cfg = json.loads(STUDIO_CONFIG_PATH.read_text()) if STUDIO_CONFIG_PATH.exists() else {}
    cfg.setdefault("compute_targets", {"Local 3090": "gpu-local", "AWS (scales)": "gpu-cloud"})
    cfg.setdefault("services_queue", "services")
    cfg.setdefault("template_task_id", "")
    return cfg


def init_state() -> None:
    if "datasets" not in st.session_state:
        st.session_state.datasets = []  # list of (tissue, source_id, platform, tier)


def get_client() -> ClearMLClient:
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


def resolve_template_id(client: ClearMLClient, studio_cfg: dict) -> tuple[str, str | None]:
    """Return (template_task_id, error_message). Uses studio.json if set, else
    auto-discovers the most recent draft 'dapidl-pipeline-controller' task."""
    tid = studio_cfg.get("template_task_id") or ""
    if tid:
        return tid, None
    try:
        data = client._post("tasks.get_all", {
            "type": ["controller"], "name": "dapidl-pipeline-controller",
            "status": ["created"], "page": 0, "page_size": 1,
            "only_fields": ["id", "name"], "order_by": ["-last_update"],
        })
        tasks = data.get("tasks", [])
        if tasks:
            return tasks[0]["id"], None
    except Exception as e:  # noqa: BLE001
        return "", f"Could not query ClearML for a template task: {e}"
    return "", (
        "No controller template task found. Create one, then set its ID in "
        "configs/studio.json:\n\n"
        "  uv run dapidl clearml-pipeline create-controller-task -t breast <dataset_id> xenium 2"
    )


def launch_one(client, template_id, name, params, queue, tags=None) -> str:
    """Clone the template, set params, enqueue. Returns the new task id."""
    new_id = client.clone_task(template_id, name, tags=tags)
    client.set_task_params(new_id, params)
    client.enqueue_task(new_id, queue)
    return new_id


# ---------------------------------------------------------------------------
# Dataset picker (unchanged behavior)
# ---------------------------------------------------------------------------
def dataset_picker(client: ClearMLClient) -> None:
    st.markdown("**Datasets**")
    try:
        datasets = client.get_datasets()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load datasets: {e}")
        datasets = []
    dataset_names = {d["id"]: d["name"] for d in datasets}

    if st.session_state.datasets:
        for i, (tissue, source, platform, tier) in enumerate(st.session_state.datasets):
            name = dataset_names.get(source, source[:12] + "…")
            cols = st.columns([3, 1, 1, 1])
            cols[0].text(f"{tissue} — {name}")
            cols[1].text(platform)
            cols[2].text(f"Tier {tier}")
            if cols[3].button("Remove", key=f"rm_{i}"):
                st.session_state.datasets.pop(i)
                st.rerun()
    else:
        st.info("No datasets added yet.")

    with st.expander("Add dataset", expanded=not st.session_state.datasets):
        c1, c2 = st.columns(2)
        tissue = c1.text_input("Tissue name", placeholder="e.g. breast", key="new_tissue")
        platform = c2.selectbox("Platform", ["xenium", "merscope"], key="new_platform")
        options = ["-- Manual ID/path --"] + [f"{d['name']}  ({d['id'][:8]}…)" for d in datasets]
        choice = st.selectbox("Dataset source", options, key="new_source_choice")
        if choice == "-- Manual ID/path --":
            source_id = st.text_input("ClearML dataset ID or local path", key="new_source_id")
        else:
            source_id = datasets[options.index(choice) - 1]["id"]
            st.caption(f"ID: `{source_id}`")
        tier = st.selectbox("Confidence tier", [1, 2, 3], index=1, key="new_tier",
                            help="1=Ground truth, 2=Consensus, 3=Single-method")
        if st.button("Add dataset", type="primary", disabled=not (tissue and source_id)):
            st.session_state.datasets.append((tissue, source_id, platform, tier))
            st.rerun()


# ---------------------------------------------------------------------------
# Schema-driven configuration form
# ---------------------------------------------------------------------------
def render_config_form(key_prefix: str, compute_labels: list[str]) -> dict:
    """Render the biologist parameter widgets; return a selections dict."""
    c1, c2, c3 = st.columns(3)
    with c1:
        detect = st.selectbox("What to detect", list(ps.DETECT_LEVELS), key=f"{key_prefix}_detect",
                              help="Broad = Epithelial/Immune/Stromal/Endothelial; Fine = ~12 subtypes")
        label = st.selectbox("Label method", list(ps.LABEL_METHODS), key=f"{key_prefix}_label",
                             help="How training labels are generated from transcriptomics")
        seg = st.selectbox("Segmentation", list(ps.SEGMENTATIONS), key=f"{key_prefix}_seg",
                          help="Nucleus segmentation source")
    with c2:
        model = st.selectbox("Image model", list(ps.IMAGE_MODELS), key=f"{key_prefix}_model",
                            help="CNN backbone (EfficientNet-V2-S is the production default)")
        patch = st.selectbox("Patch size (px)", ps.PATCH_SIZES, index=2, key=f"{key_prefix}_patch")
        epochs = st.slider("Epochs", 1, 200, 50, key=f"{key_prefix}_epochs")
    with c3:
        compute = st.selectbox("Compute target", compute_labels, key=f"{key_prefix}_compute",
                             help="Local 3090 (free, serial) · AWS (scales for sweeps) · DigitalOcean")
    with st.expander("Advanced"):
        a1, a2 = st.columns(2)
        batch = a1.number_input("Batch size", 8, 512, 64, step=8, key=f"{key_prefix}_batch")
        lr = a2.number_input("Learning rate", 1e-6, 1e-2, 1e-4, format="%.6f", key=f"{key_prefix}_lr")
    return {
        "detect_level": detect, "label_method": label, "segmentation": seg,
        "image_model": model, "patch_size": int(patch), "epochs": int(epochs),
        "compute_target": compute, "batch_size": int(batch), "learning_rate": float(lr),
    }


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def configure_run_tab(client: ClearMLClient, studio_cfg: dict) -> None:
    st.subheader("Configure & Run")
    dataset_picker(client)
    st.divider()
    targets = studio_cfg["compute_targets"]
    sel = render_config_form("cfg", list(targets))
    st.divider()

    template_id, terr = resolve_template_id(client, studio_cfg)
    if terr:
        st.warning(terr)
    errs = ps.validate(sel, st.session_state.datasets, None)
    for e in errs:
        st.error(e)

    run_name = st.text_input("Run name", value="studio-run", key="cfg_runname")
    if st.button("▶ Launch run", type="primary", disabled=bool(errs or not template_id)):
        params = ps.build_param_overrides(
            sel, st.session_state.datasets, targets, studio_cfg["services_queue"]
        )
        queue = targets[sel["compute_target"]]
        try:
            new_id = launch_one(client, template_id, run_name, params, queue)
            web = client.web_server or "https://clearml.chrism.io"
            st.success(f"Launched task `{new_id[:8]}…` on queue **{queue}**.")
            st.markdown(f"[Open in ClearML]({web}/projects/*/experiments/{new_id})")
        except Exception as e:  # noqa: BLE001
            st.error(f"Launch failed: {e}")


def sweep_tab(client: ClearMLClient, studio_cfg: dict) -> None:
    st.subheader("Parameter Sweep")
    if not st.session_state.datasets:
        st.warning("Add at least one dataset in the **Configure & Run** tab first.")
        return
    targets = studio_cfg["compute_targets"]
    base = render_config_form("swp", list(targets))

    st.markdown("**Vary these parameters** (cartesian product):")
    axes: dict[str, list] = {}
    col_a, col_b = st.columns(2)
    with col_a:
        if st.checkbox("Patch size", key="swp_ax_patch"):
            axes["patch_size"] = st.multiselect("Patch sizes", ps.PATCH_SIZES, default=[64, 128], key="swp_patch_vals")
        if st.checkbox("Image model", key="swp_ax_model"):
            axes["image_model"] = st.multiselect("Image models", list(ps.IMAGE_MODELS),
                                                 default=list(ps.IMAGE_MODELS)[:2], key="swp_model_vals")
    with col_b:
        if st.checkbox("Label method", key="swp_ax_label"):
            axes["label_method"] = st.multiselect("Label methods", list(ps.LABEL_METHODS), key="swp_label_vals")
        if st.checkbox("Segmentation", key="swp_ax_seg"):
            axes["segmentation"] = st.multiselect("Segmentation methods", list(ps.SEGMENTATIONS), key="swp_seg_vals")

    sweep_id = st.text_input("Sweep name", value="exp1", key="swp_id")
    template_id, terr = resolve_template_id(client, studio_cfg)
    if terr:
        st.warning(terr)
    errs = ps.validate(base, st.session_state.datasets, axes)
    for e in errs:
        st.error(e)

    runs = []
    if axes and not errs:
        runs = ps.expand_sweep(base, axes, st.session_state.datasets, targets, sweep_id, studio_cfg["services_queue"])
        st.info(f"Will launch **{len(runs)}** runs, tagged `sweep-{sweep_id}`.")

    if st.button("▶ Launch sweep", type="primary", disabled=bool(errs or not runs or not template_id)):
        queue = targets[base["compute_target"]]
        rows = []
        for r in runs:
            try:
                nid = launch_one(client, template_id, r["name"], r["params"], queue, tags=[r["tag"]])
                rows.append({"run": r["name"], "task": nid[:8], "status": "queued"})
            except Exception as e:  # noqa: BLE001
                rows.append({"run": r["name"], "task": "-", "status": f"FAILED: {e}"})
        ok = sum(1 for x in rows if x["status"] == "queued")
        st.success(f"Submitted {ok}/{len(rows)} runs (tag `sweep-{sweep_id}`). See the **Results** tab.")
        st.dataframe(rows, use_container_width=True)


def results_tab(client: ClearMLClient) -> None:
    st.subheader("Results")
    tag = st.text_input("Sweep tag", value="sweep-exp1", key="res_tag",
                        help="The tag shown when you launched a sweep, e.g. sweep-exp1")
    if st.button("Load results", key="res_load") and tag:
        try:
            tasks = client.get_tasks_by_tag(tag)
        except Exception as e:  # noqa: BLE001
            st.error(f"Query failed: {e}")
            return
        if not tasks:
            st.info("No runs found for that tag yet.")
            return
        rows = []
        for t in tasks:
            row = {"run": (t.get("name") or "")[-40:], "status": t.get("status", "")}
            with contextlib.suppress(Exception):
                for k, v in client.get_task_scalars(t["id"]).items():
                    if isinstance(v, (int, float)) and ("f1" in k.lower() or "acc" in k.lower()):
                        row[k] = round(float(v), 3)
            rows.append(row)
        st.dataframe(rows, use_container_width=True)
        web = client.web_server or "https://clearml.chrism.io"
        ids = ",".join(t["id"] for t in tasks)
        st.markdown(f"[Compare these runs in ClearML]({web}/projects/*/compare-experiments;ids={ids})")


def pipeline_monitor(client: ClearMLClient) -> None:
    st.subheader("Recent Pipelines")
    if st.button("Refresh", key="refresh_pipelines"):
        st.rerun()
    try:
        pipelines = client.get_pipelines(limit=15)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load pipelines: {e}")
        return
    if not pipelines:
        st.info("No pipelines found.")
        return
    for pipe in pipelines:
        status = STATUS_COLORS.get(pipe["status"], pipe["status"])
        last = pipe.get("last_update", "")
        if last:
            with contextlib.suppress(ValueError, AttributeError):
                last = datetime.fromisoformat(last.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{pipe['name']} — {status} ({last})", expanded=pipe["status"] == "in_progress"):
            web = client.web_server or "https://clearml.chrism.io"
            st.markdown(f"[Open in ClearML]({web}/projects/*/experiments/{pipe['id']})")
            try:
                for step in client.get_pipeline_steps(pipe["id"]):
                    st.markdown(f"  - **{step['name']}** — {STATUS_COLORS.get(step['status'], step['status'])}")
            except Exception as e:  # noqa: BLE001
                st.caption(f"Could not load steps: {e}")


def dataset_browser(client: ClearMLClient) -> None:
    st.subheader("Registered Datasets")
    if st.button("Refresh", key="refresh_datasets"):
        st.rerun()
    try:
        datasets = client.get_datasets()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load datasets: {e}")
        return
    by_project: dict[str, list[dict]] = {}
    for d in datasets:
        proj = d.get("project", {})
        name = proj.get("name", "Unknown") if isinstance(proj, dict) else str(proj)
        by_project.setdefault(name, []).append(d)
    for proj_name in sorted(by_project):
        with st.expander(f"{proj_name} ({len(by_project[proj_name])})"):
            for d in by_project[proj_name]:
                tags = ", ".join(d.get("tags", [])[:5])
                st.markdown(f"- **{d['name']}** {f'`{tags}`' if tags else ''}  \n  `{d['id']}`")


# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="DAPIDL Pipeline Studio", page_icon="🔬",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("🔬 DAPIDL Pipeline Studio")
    st.caption("Configure, sweep, and compare DAPI → cell-type pipelines — no MLOps required.")

    init_state()
    studio_cfg = load_studio_config()
    try:
        client = get_client()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not connect to ClearML: {e}")
        st.info(f"Ensure {CLEARML_CONFIG_PATH} exists with valid credentials.")
        return

    tabs = st.tabs(["Configure & Run", "Sweep", "Results", "Monitor", "Datasets"])
    with tabs[0]:
        configure_run_tab(client, studio_cfg)
    with tabs[1]:
        sweep_tab(client, studio_cfg)
    with tabs[2]:
        results_tab(client)
    with tabs[3]:
        pipeline_monitor(client)
    with tabs[4]:
        dataset_browser(client)

    with st.sidebar:
        st.markdown("### Server")
        st.caption(f"API: `{client.api_server}`")
        st.caption(f"Web: `{client.web_server}`")
        tid = studio_cfg.get("template_task_id") or "(auto-discover)"
        st.caption(f"Template task: `{tid}`")
        st.markdown("### Compute targets")
        for label, queue in studio_cfg["compute_targets"].items():
            st.caption(f"- {label} → `{queue}`")


if __name__ == "__main__":
    main()
