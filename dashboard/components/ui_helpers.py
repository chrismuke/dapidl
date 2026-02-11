"""Reusable UI formatting helpers for the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone

from .constants import STATUS_COLORS


def status_badge(status: str) -> str:
    """Return a Streamlit-colored markdown badge for a task status."""
    return STATUS_COLORS.get(status, f":gray[{status}]")


def format_datetime(iso_str: str | None) -> str:
    """Format an ISO datetime string to 'YYYY-MM-DD HH:MM'."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        return str(iso_str)[:16]


def format_duration(started: str | None, completed: str | None) -> str:
    """Return a human-readable duration between two ISO timestamps.

    If *completed* is None, shows time elapsed since *started*.
    """
    if not started:
        return ""
    try:
        dt_start = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""

    if completed:
        try:
            dt_end = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            dt_end = datetime.now(timezone.utc)
    else:
        dt_end = datetime.now(timezone.utc)

    delta = dt_end - dt_start
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return ""
    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes = total_seconds // 60
    hours = minutes // 60
    mins = minutes % 60

    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def clearml_task_url(web_server: str, task_id: str) -> str:
    """Build a clickable ClearML web UI link for a task."""
    base = web_server.rstrip("/") if web_server else "https://clearml.chrism.io"
    return f"{base}/projects/*/experiments/{task_id}"


def step_progress_columns(steps: list[dict]) -> list[dict]:
    """Prepare step data for horizontal column rendering.

    Returns a list of dicts with 'name', 'status', 'color', 'duration'.
    """
    result = []
    color_map = {
        "completed": "#28a745",
        "in_progress": "#fd7e14",
        "failed": "#dc3545",
        "stopped": "#dc3545",
        "created": "#6c757d",
        "queued": "#17a2b8",
    }
    for step in steps:
        result.append({
            "name": step.get("name", "?"),
            "status": step.get("status", "unknown"),
            "color": color_map.get(step.get("status", ""), "#6c757d"),
            "duration": format_duration(step.get("started"), step.get("completed")),
        })
    return result


def recipe_flow(recipe_name: str, steps: list[str]) -> str:
    """Format recipe steps as a flow arrow string."""
    return f"**{recipe_name}**: " + " → ".join(steps)


def worker_health_indicator(last_activity: str | None) -> tuple[str, str]:
    """Return (emoji, label) for a worker's health based on last activity.

    Returns ('green_circle'/'yellow_circle'/'red_circle', description).
    """
    from .constants import WORKER_HEALTH_OK, WORKER_HEALTH_WARN

    if not last_activity:
        return "red_circle", "Unknown"
    try:
        dt = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "red_circle", "Unknown"

    seconds_ago = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds_ago < WORKER_HEALTH_OK:
        return "large_green_circle", "Healthy"
    if seconds_ago < WORKER_HEALTH_WARN:
        return "large_yellow_circle", "Idle"
    return "red_circle", "Offline"
