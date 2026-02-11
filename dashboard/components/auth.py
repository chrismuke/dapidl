"""Simple session-based authentication for the dashboard.

Credentials are read from environment variables:
    DASHBOARD_USERNAME (default: admin)
    DASHBOARD_PASSWORD (required)
"""

from __future__ import annotations

import hashlib
import hmac
import os

import streamlit as st


def _check_credentials(username: str, password: str) -> bool:
    """Verify username and password against environment variables."""
    expected_user = os.environ.get("DASHBOARD_USERNAME", "admin")
    expected_pass = os.environ.get("DASHBOARD_PASSWORD", "")
    if not expected_pass:
        return False
    user_ok = hmac.compare_digest(username, expected_user)
    pass_ok = hmac.compare_digest(
        hashlib.sha256(password.encode()).hexdigest(),
        hashlib.sha256(expected_pass.encode()).hexdigest(),
    )
    return user_ok and pass_ok


def login_page() -> None:
    """Render a centered login form."""
    st.markdown(
        "<h1 style='text-align:center; margin-top:2em;'>DAPIDL Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Please sign in to continue</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                if _check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")


def require_auth() -> bool:
    """Gate a page behind authentication.

    Call at the top of every page. Returns True if authenticated,
    False (and renders login) if not.

    Usage::

        from components.auth import require_auth
        if not require_auth():
            st.stop()
        # ... rest of page
    """
    if not st.session_state.get("authenticated", False):
        login_page()
        return False
    return True


def logout_button() -> None:
    """Render a logout button in the sidebar."""
    with st.sidebar:
        user = st.session_state.get("username", "")
        if user:
            st.caption(f"Signed in as **{user}**")
        if st.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.pop("username", None)
            st.rerun()
