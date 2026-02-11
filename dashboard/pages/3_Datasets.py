"""Dataset Browser — searchable view of ClearML datasets grouped by project."""

from __future__ import annotations

import streamlit as st

from components.auth import logout_button, require_auth
from components.clearml_client import ClearMLClient
from components.ui_helpers import format_datetime

st.set_page_config(page_title="Datasets", page_icon=":open_file_folder:", layout="wide")
if not require_auth():
    st.stop()
logout_button()
st.title("Datasets")


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

    if st.button("Refresh", key="refresh_datasets"):
        st.rerun()

    datasets = client.get_datasets()

    if not datasets:
        st.info("No datasets found on the ClearML server.")
        return

    # Search filter
    search = st.text_input("Search by name", placeholder="Filter datasets...", key="ds_filter")
    if search:
        search_lower = search.lower()
        datasets = [d for d in datasets if search_lower in d.get("name", "").lower()]

    # Group by project
    by_project: dict[str, list[dict]] = {}
    for d in datasets:
        proj = d.get("project", {})
        proj_name = proj.get("name", "Unknown") if isinstance(proj, dict) else str(proj)
        by_project.setdefault(proj_name, []).append(d)

    st.caption(f"{len(datasets)} dataset(s) in {len(by_project)} project(s)")

    for proj_name in sorted(by_project):
        proj_datasets = by_project[proj_name]
        with st.expander(f"{proj_name} ({len(proj_datasets)} datasets)"):
            for d in proj_datasets:
                tags = d.get("tags", [])
                tag_str = "  " + " ".join(f"`{t}`" for t in tags[:5]) if tags else ""
                last_update = format_datetime(d.get("last_update"))
                date_str = f"  ({last_update})" if last_update else ""

                st.markdown(f"**{d['name']}**{tag_str}{date_str}")
                st.code(d["id"], language=None)


main()
