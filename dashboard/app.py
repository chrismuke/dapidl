"""DAPIDL Dashboard — Home page and application entry point.

Run locally:
    cd dashboard && streamlit run app.py

Deploy via Coolify:
    Docker build context: dashboard/
    Dockerfile: dashboard/Dockerfile
"""

from __future__ import annotations

import streamlit as st

from components.auth import logout_button, require_auth
from components.clearml_client import ClearMLClient
from components.ui_helpers import status_badge

st.set_page_config(
    page_title="DAPIDL Dashboard",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_client() -> ClearMLClient:
    """Get the ClearML client from session state (set during login)."""
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


def main() -> None:
    if not require_auth():
        st.stop()

    logout_button()
    st.title("DAPIDL Dashboard")
    st.caption("Cell-type prediction pipeline management on ClearML")

    try:
        client = get_client()
    except Exception as exc:
        st.error(f"Could not connect to ClearML: {exc}")
        st.info("Set CLEARML_API_HOST / CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY environment variables.")
        return

    # Sidebar — server info
    with st.sidebar:
        st.markdown("### ClearML Server")
        web_url = client.web_server or "https://clearml.chrism.io"
        st.caption(f"API: `{client.api_server}`")
        st.markdown(f"[Open ClearML Web UI]({web_url})")

    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)

    pipelines = client.get_pipelines(limit=50)
    active = [p for p in pipelines if p.get("status") == "in_progress"]
    failed = [p for p in pipelines if p.get("status") == "failed"]
    completed = [p for p in pipelines if p.get("status") == "completed"]

    col1.metric("Active Pipelines", len(active))
    col2.metric("Completed", len(completed))
    col3.metric("Failed", len(failed))

    workers = client.get_workers()
    col4.metric("Workers Online", len(workers))

    # Active pipelines summary
    if active:
        st.markdown("### Active Pipelines")
        for p in active:
            name = p.get("name", "Unknown")
            proj = p.get("project", {})
            proj_name = proj.get("name", "?") if isinstance(proj, dict) else str(proj)
            link = f"{web_url}/projects/*/experiments/{p['id']}"
            st.markdown(f"- {status_badge('in_progress')} **{name}** ({proj_name}) — [view]({link})")
    else:
        st.info("No pipelines currently running.")

    # Queue overview
    queues = client.get_queue_stats()
    queued_total = sum(len(q.get("entries", [])) for q in queues)
    if queued_total > 0:
        st.markdown("### Queue Summary")
        q_cols = st.columns(min(len(queues), 5))
        for i, q in enumerate(queues):
            entries = len(q.get("entries", []))
            if entries > 0:
                q_cols[i % len(q_cols)].metric(q.get("name", "?"), f"{entries} queued")


if __name__ == "__main__":
    main()
