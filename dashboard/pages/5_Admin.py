"""Admin — ClearML worker queue control and system settings."""

from __future__ import annotations

import streamlit as st

from components.auth import logout_button, require_admin
from components.clearml_client import ClearMLClient
from components.constants import WORKER_DEFAULT_QUEUES
from components.ui_helpers import worker_health_indicator

st.set_page_config(page_title="Admin", page_icon=":shield:", layout="wide")
if not require_admin():
    st.stop()
logout_button()
st.title("Admin Settings")


def get_client() -> ClearMLClient:
    if "clearml_client" not in st.session_state:
        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
    return st.session_state.clearml_client


def _get_queue_id_by_name(queues: list[dict], name: str) -> str | None:
    """Look up a queue ID by name from the full queue list."""
    for q in queues:
        if q.get("name") == name:
            return q["id"]
    return None


def _worker_queue_ids(worker: dict) -> list[tuple[str, str]]:
    """Return list of (queue_id, queue_name) tuples for a worker."""
    result = []
    for q in worker.get("queues", []):
        if isinstance(q, dict):
            result.append((q.get("id", ""), q.get("name", q.get("id", "?")[:8])))
    return result


def main() -> None:
    try:
        client = get_client()
    except Exception as exc:
        st.error(f"Could not connect to ClearML: {exc}")
        return

    st.subheader("Worker Queue Control")
    st.caption(
        "Disable a worker by removing it from all queues (agent stays alive but idle). "
        "Enable restores default queue assignments."
    )

    workers = client.get_workers()
    all_queues = client.get_queue_stats()

    if not workers:
        st.info("No workers registered with ClearML.")
        return

    for w in workers:
        worker_id = w.get("id", "unknown")
        worker_name = worker_id.split(":")[0] if ":" in worker_id else worker_id
        last = w.get("last_activity_time") or w.get("register_time")
        emoji, health_label = worker_health_indicator(last)

        current_queues = _worker_queue_ids(w)
        queue_names = [name for _, name in current_queues]
        is_disabled = len(current_queues) == 0

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 1, 2, 2])
            c1.markdown(f":{emoji}: **{worker_name}**")
            c2.caption(health_label)
            c3.caption(
                ":red[disabled — no queues]" if is_disabled
                else ", ".join(queue_names)
            )

            with c4:
                if is_disabled:
                    # Enable: restore default queues
                    default_queues = WORKER_DEFAULT_QUEUES.get(worker_name, [])
                    if not default_queues:
                        st.caption("No default queues configured")
                    elif st.button(
                        "Enable",
                        key=f"enable_{worker_id}",
                        type="primary",
                    ):
                        ok = True
                        for qname in default_queues:
                            qid = _get_queue_id_by_name(all_queues, qname)
                            if qid:
                                if not client.add_worker_to_queue(worker_id, qid):
                                    ok = False
                            else:
                                st.toast(f"Queue '{qname}' not found", icon="⚠️")
                                ok = False
                        if ok:
                            st.toast(
                                f"Enabled {worker_name} → {', '.join(default_queues)}",
                                icon="✅",
                            )
                        st.rerun()
                else:
                    # Disable: remove from all queues
                    if st.button(
                        "Disable",
                        key=f"disable_{worker_id}",
                    ):
                        ok = True
                        for qid, qname in current_queues:
                            if not client.remove_worker_from_queue(worker_id, qid):
                                st.toast(
                                    f"Failed to remove from {qname}", icon="⚠️",
                                )
                                ok = False
                        if ok:
                            st.toast(f"Disabled {worker_name}", icon="🔴")
                        st.rerun()


main()
