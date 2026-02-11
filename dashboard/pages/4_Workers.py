"""Workers — ClearML agent health monitoring and queue overview."""

from __future__ import annotations

import time

import streamlit as st

from components.clearml_client import ClearMLClient
from components.ui_helpers import worker_health_indicator

st.set_page_config(page_title="Workers", page_icon=":gear:", layout="wide")
st.title("Workers & Queues")


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

    # Controls
    ctrl1, ctrl2 = st.columns([1, 3])
    with ctrl1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="workers_auto")
    with ctrl2:
        refresh_interval = st.selectbox("Interval", [15, 30, 60], key="workers_interval", format_func=lambda x: f"{x}s")

    if st.button("Refresh", key="refresh_workers"):
        st.rerun()

    # --- Workers section ---
    st.subheader("Agents")

    workers = client.get_workers()

    if not workers:
        st.info("No workers registered with ClearML.")
    else:
        # Summary
        health_counts = {"Healthy": 0, "Idle": 0, "Offline": 0}
        for w in workers:
            last = w.get("last_activity_time") or w.get("register_time")
            _, label = worker_health_indicator(last)
            if label in health_counts:
                health_counts[label] += 1
            else:
                health_counts["Offline"] += 1

        parts = []
        if health_counts["Healthy"]:
            parts.append(f":large_green_circle: {health_counts['Healthy']} active")
        if health_counts["Idle"]:
            parts.append(f":large_yellow_circle: {health_counts['Idle']} idle")
        if health_counts["Offline"]:
            parts.append(f":red_circle: {health_counts['Offline']} offline")
        st.markdown("  ".join(parts) if parts else "No workers")

        # Worker details
        for w in workers:
            worker_id = w.get("id", "unknown")
            worker_name = worker_id.split(":")[0] if ":" in worker_id else worker_id
            last = w.get("last_activity_time") or w.get("register_time")
            emoji, health_label = worker_health_indicator(last)

            # Current task
            task = w.get("task")
            task_info = ""
            if task and isinstance(task, dict):
                task_name = task.get("name", task.get("id", "?")[:8])
                task_id = task.get("id", "")
                if task_id:
                    task_info = f"running [{task_name}]({web_url}/projects/*/experiments/{task_id})"
                else:
                    task_info = f"running {task_name}"

            # Queues the worker listens on
            queues_raw = w.get("queues", [])
            queue_names = []
            for q in queues_raw:
                if isinstance(q, dict):
                    queue_names.append(q.get("name", q.get("id", "?")[:8]))
                else:
                    queue_names.append(str(q))

            with st.container():
                c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
                c1.markdown(f":{emoji}: **{worker_name}**")
                c2.caption(health_label)
                c3.caption(task_info or "idle")
                c4.caption(", ".join(queue_names) if queue_names else "no queues")

    # --- Queue section ---
    st.subheader("Queues")

    queues = client.get_queue_stats()

    if not queues:
        st.info("No queues found.")
    else:
        q_cols = st.columns(min(len(queues), 5))
        for i, q in enumerate(queues):
            name = q.get("name", "?")
            entries = len(q.get("entries", []))
            workers_list = q.get("workers", [])
            n_workers = len(workers_list) if isinstance(workers_list, list) else 0

            col = q_cols[i % len(q_cols)]
            col.metric(name, f"{entries} queued")
            col.caption(f"{n_workers} worker(s)")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


main()
