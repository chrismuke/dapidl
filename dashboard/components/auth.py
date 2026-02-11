"""ClearML-backed authentication for the dashboard.

Users log in with their ClearML credentials (username + access key + secret key).
Authentication is validated by calling ClearML's auth.login endpoint.
On success, a per-user ClearMLClient is stored in session state so that all
pipeline operations are attributed to the logged-in user.

Fallback: if DASHBOARD_USERS env var is set (JSON dict of user:password pairs),
those credentials are accepted instead (useful for non-ClearML users).
"""

from __future__ import annotations

import json
import os

import streamlit as st

from .clearml_client import ClearMLClient


def _try_clearml_login(
    username: str, access_key: str, secret_key: str,
) -> ClearMLClient | None:
    """Attempt to authenticate against ClearML. Returns client on success."""
    api_host = os.environ.get("CLEARML_API_HOST", "https://api.clearml.chrism.io")
    web_host = os.environ.get("CLEARML_WEB_HOST", "https://clearml.chrism.io")
    client = ClearMLClient(
        api_server=api_host,
        access_key=access_key,
        secret_key=secret_key,
        web_server=web_host,
    )
    try:
        client._authenticate()
        return client
    except Exception:
        return None


def _try_local_login(username: str, password: str) -> bool:
    """Check credentials against DASHBOARD_USERS env var (JSON dict)."""
    users_json = os.environ.get("DASHBOARD_USERS", "")
    if not users_json:
        # Backward compat: single user from DASHBOARD_USERNAME/PASSWORD
        expected_user = os.environ.get("DASHBOARD_USERNAME", "")
        expected_pass = os.environ.get("DASHBOARD_PASSWORD", "")
        if expected_user and expected_pass:
            return username == expected_user and password == expected_pass
        return False
    try:
        users = json.loads(users_json)
        return users.get(username) == password
    except (json.JSONDecodeError, AttributeError):
        return False


def login_page() -> None:
    """Render the login form with two authentication modes."""
    st.markdown(
        "<h1 style='text-align:center; margin-top:2em;'>DAPIDL Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Sign in with your ClearML credentials</p>",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        tab_clearml, tab_local = st.tabs(["ClearML Login", "Local Login"])

        with tab_clearml:
            with st.form("clearml_login"):
                username = st.text_input("Display name", placeholder="e.g. chrism")
                access_key = st.text_input("ClearML Access Key")
                secret_key = st.text_input("ClearML Secret Key", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

                if submitted:
                    if not (username and access_key and secret_key):
                        st.error("All fields are required")
                    else:
                        client = _try_clearml_login(username, access_key, secret_key)
                        if client:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.auth_mode = "clearml"
                            st.session_state.clearml_client = client
                            st.rerun()
                        else:
                            st.error("ClearML authentication failed — check your keys")

        with tab_local:
            with st.form("local_login"):
                local_user = st.text_input("Username")
                local_pass = st.text_input("Password", type="password")
                local_submitted = st.form_submit_button("Sign in", use_container_width=True)

                if local_submitted:
                    if _try_local_login(local_user, local_pass):
                        st.session_state.authenticated = True
                        st.session_state.username = local_user
                        st.session_state.auth_mode = "local"
                        # Use default ClearML client from env vars
                        st.session_state.clearml_client = ClearMLClient.from_env_or_config()
                        st.rerun()
                    else:
                        st.error("Invalid username or password")


def require_auth() -> bool:
    """Gate a page behind authentication.

    Returns True if authenticated, False (and renders login) if not.
    """
    if not st.session_state.get("authenticated", False):
        login_page()
        return False
    return True


def logout_button() -> None:
    """Render a logout button and user info in the sidebar."""
    with st.sidebar:
        user = st.session_state.get("username", "")
        mode = st.session_state.get("auth_mode", "")
        if user:
            badge = "ClearML" if mode == "clearml" else "local"
            st.caption(f"Signed in as **{user}** ({badge})")
        if st.button("Sign out", use_container_width=True):
            for key in ("authenticated", "username", "auth_mode", "clearml_client"):
                st.session_state.pop(key, None)
            st.rerun()


def get_current_user() -> str:
    """Return the logged-in username, or empty string."""
    return st.session_state.get("username", "")
