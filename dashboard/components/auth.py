"""ClearML-backed authentication for the dashboard.

Users log in with their ClearML username + password (fixed users mode).
Authentication is validated by calling ClearML's auth.login endpoint.
On success, a per-user ClearMLClient is stored in session state so that all
pipeline operations are attributed to the logged-in user.
"""

from __future__ import annotations

import os

import streamlit as st

from .clearml_client import ClearMLClient


def _try_clearml_login(
    username: str, password: str,
) -> ClearMLClient | None:
    """Authenticate against ClearML using username + password (fixed users mode).

    In fixed users mode, ClearML accepts username:password as HTTP Basic Auth
    credentials on the auth.login endpoint — same as access_key:secret_key.
    """
    api_host = os.environ.get("CLEARML_API_HOST", "https://api.clearml.chrism.io")
    web_host = os.environ.get("CLEARML_WEB_HOST", "https://clearml.chrism.io")
    client = ClearMLClient(
        api_server=api_host,
        access_key=username,
        secret_key=password,
        web_server=web_host,
    )
    try:
        client._authenticate()
        return client
    except Exception:
        return None


def login_page() -> None:
    """Render the ClearML login form."""
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
        with st.form("clearml_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                if not (username and password):
                    st.error("Username and password are required")
                else:
                    client = _try_clearml_login(username, password)
                    if client:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.clearml_client = client
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
        if user:
            st.caption(f"Signed in as **{user}**")
        if st.button("Sign out", use_container_width=True):
            for key in ("authenticated", "username", "clearml_client"):
                st.session_state.pop(key, None)
            st.rerun()


def get_current_user() -> str:
    """Return the logged-in username, or empty string."""
    return st.session_state.get("username", "")
