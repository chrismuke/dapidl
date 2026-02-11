"""Pipeline Monitor — live pipeline tracking with auto-refresh and step timelines."""

from __future__ import annotations

import time

import streamlit as st

from components.auth import logout_button, require_auth
from components.clearml_client import ClearMLClient
from components.ui_helpers import (
    clearml_task_url,
    format_datetime,
    format_duration,
    status_badge,
    step_progress_columns,
)

st.set_page_config(page_title="Pipeline Monitor", page_icon=":bar_chart:", layout="wide")
if not require_auth():
    st.stop()
logout_button()
st.title("Pipeline Monitor")


def get_client() -> ClearMLClient:
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


def main() -> None:
    try:
        client = get_client()
    except Exception as exc:
        st.error(f"Could not connect to ClearML: {exc}")
        return

    web_url = client.web_server or "https://clearml.chrism.io"

    # Controls bar
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 2])

    with ctrl1:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with ctrl2:
        refresh_interval = st.selectbox("Interval", [15, 30, 60], format_func=lambda x: f"{x}s")
    with ctrl3:
        limit = st.selectbox("Show last", [15, 30, 50], index=0)
    with ctrl4:
        status_filter = st.multiselect(
            "Filter by status",
            ["in_progress", "completed", "failed", "stopped", "created", "queued"],
            default=[],
            help="Leave empty to show all",
        )

    if st.button("Refresh", key="refresh_pipelines"):
        st.rerun()

    # Fetch pipelines
    pipelines = client.get_pipelines(limit=limit)

    if status_filter:
        pipelines = [p for p in pipelines if p.get("status") in status_filter]

    if not pipelines:
        st.info("No pipelines found matching your filters.")
    else:
        st.caption(f"Showing {len(pipelines)} pipeline(s)")

        for pipe in pipelines:
            status = pipe.get("status", "unknown")
            name = pipe.get("name", "Unknown")
            last_update = format_datetime(pipe.get("last_update"))
            duration = format_duration(pipe.get("started"), pipe.get("completed"))
            proj = pipe.get("project", {})
            proj_name = proj.get("name", "?") if isinstance(proj, dict) else str(proj)

            header = f"{name}  —  {status_badge(status)}"
            if duration:
                header += f"  ({duration})"
            header += f"  —  {last_update}"

            with st.expander(header, expanded=(status == "in_progress")):
                # Metadata row
                mc1, mc2 = st.columns(2)
                mc1.caption(f"ID: `{pipe['id']}`  |  Project: `{proj_name}`")
                mc2.markdown(f"[Open in ClearML]({clearml_task_url(web_url, pipe['id'])})")

                if pipe.get("status_reason"):
                    st.caption(f"Reason: {pipe['status_reason']}")

                # Step timeline
                steps = client.get_pipeline_steps(pipe["id"])
                if steps:
                    step_data = step_progress_columns(steps)
                    cols = st.columns(len(step_data))
                    for col, sd in zip(cols, step_data):
                        # Colored block for each step
                        col.markdown(
                            f"<div style='background:{sd['color']};padding:8px;border-radius:4px;"
                            f"text-align:center;color:white;font-size:0.8em;'>"
                            f"<b>{sd['name']}</b><br>{sd['status']}"
                            f"{'<br>' + sd['duration'] if sd['duration'] else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Detailed step list
                    with st.container():
                        for step in steps:
                            s_status = status_badge(step.get("status", "unknown"))
                            s_dur = format_duration(step.get("started"), step.get("completed"))
                            s_link = clearml_task_url(web_url, step["id"])
                            dur_str = f" ({s_dur})" if s_dur else ""
                            st.markdown(
                                f"- {s_status} **{step.get('name', '?')}**{dur_str}"
                                f" — [view]({s_link})"
                            )
                else:
                    st.caption("No child steps found.")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


main()
