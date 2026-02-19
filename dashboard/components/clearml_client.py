"""Lightweight ClearML REST API client — no SDK dependency.

Extracted from scripts/streamlit_pipeline.py and enhanced with worker/queue
monitoring endpoints and graceful error handling for the dashboard.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CLEARML_CONFIG_PATH = Path.home() / ".clearml" / "clearml-chrism.conf"


@dataclass
class ClearMLClient:
    """Minimal ClearML REST API client for reading server state."""

    api_server: str = ""
    access_key: str = ""
    secret_key: str = ""
    web_server: str = ""
    _token: str = field(default="", repr=False)
    _token_expiry: float = 0.0

    @classmethod
    def from_config(cls, config_path: Path = CLEARML_CONFIG_PATH) -> ClearMLClient:
        """Parse HOCON-style ClearML config file."""
        text = config_path.read_text()
        client = cls()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("api_server:"):
                client.api_server = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("web_server:"):
                client.web_server = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("access_key:"):
                client.access_key = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("secret_key:"):
                client.secret_key = line.split(":", 1)[1].strip().strip('"')
        return client

    @classmethod
    def from_env_or_config(cls) -> ClearMLClient:
        """Create client from environment variables, falling back to config file."""
        if os.environ.get("CLEARML_API_HOST"):
            return cls(
                api_server=os.environ["CLEARML_API_HOST"],
                access_key=os.environ["CLEARML_API_ACCESS_KEY"],
                secret_key=os.environ["CLEARML_API_SECRET_KEY"],
                web_server=os.environ.get("CLEARML_WEB_HOST", ""),
            )
        return cls.from_config()

    @property
    def _headers(self) -> dict[str, str]:
        if time.time() > self._token_expiry - 60:
            self._authenticate()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def _authenticate(self) -> None:
        creds = base64.b64encode(f"{self.access_key}:{self.secret_key}".encode()).decode()
        resp = requests.post(
            f"{self.api_server}/auth.login",
            json={},
            headers={"Content-Type": "application/json", "Authorization": f"Basic {creds}"},
            timeout=10,
        )
        resp.raise_for_status()
        self._token = resp.json()["data"]["token"]
        self._token_expiry = time.time() + 3600

    def _post(self, endpoint: str, payload: dict) -> dict:
        """POST to ClearML API with automatic error handling."""
        try:
            resp = requests.post(
                f"{self.api_server}/{endpoint}",
                json=payload,
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["data"]
        except requests.ConnectionError:
            logger.error("Cannot reach ClearML API at %s", self.api_server)
            return {}
        except requests.HTTPError as exc:
            logger.error("ClearML API error on %s: %s", endpoint, exc)
            return {}

    # -- Projects & queues --

    def get_queues(self) -> list[dict]:
        data = self._post("queues.get_all", {})
        return data.get("queues", [])

    def get_projects(self) -> list[dict]:
        data = self._post(
            "projects.get_all",
            {"page": 0, "page_size": 200, "only_fields": ["name", "id"]},
        )
        return data.get("projects", [])

    def _resolve_project_names(self, tasks: list[dict]) -> list[dict]:
        """Replace bare project IDs with {id, name} dicts."""
        project_ids = {t["project"] for t in tasks if isinstance(t.get("project"), str)}
        if not project_ids:
            return tasks
        proj_data = self._post(
            "projects.get_all",
            {"id": list(project_ids), "only_fields": ["id", "name"]},
        )
        id_to_name = {p["id"]: p["name"] for p in proj_data.get("projects", [])}
        for t in tasks:
            pid = t.get("project")
            if isinstance(pid, str):
                t["project"] = {"id": pid, "name": id_to_name.get(pid, pid[:8])}
        return tasks

    # -- Datasets --

    def get_datasets(self) -> list[dict]:
        """Return dataset tasks (type=data_processing)."""
        data = self._post("tasks.get_all", {
            "type": ["data_processing"],
            "status": ["completed"],
            "page": 0,
            "page_size": 500,
            "only_fields": ["id", "name", "project", "status", "tags", "last_update"],
            "order_by": ["-last_update"],
        })
        return self._resolve_project_names(data.get("tasks", []))

    # -- Pipelines --

    def get_pipelines(self, limit: int = 20) -> list[dict]:
        """Return recent pipeline controller tasks."""
        data = self._post("tasks.get_all", {
            "system_tags": ["pipeline"],
            "page": 0,
            "page_size": limit,
            "only_fields": [
                "id", "name", "status", "project",
                "started", "completed", "last_update", "status_reason",
            ],
            "order_by": ["-last_update"],
        })
        return self._resolve_project_names(data.get("tasks", []))

    def get_task(self, task_id: str) -> dict:
        data = self._post("tasks.get_by_id", {"task": task_id})
        return data.get("task", {})

    def get_pipeline_steps(self, pipeline_id: str) -> list[dict]:
        """Get child tasks (steps) of a pipeline controller."""
        data = self._post("tasks.get_all", {
            "parent": pipeline_id,
            "page": 0,
            "page_size": 50,
            "only_fields": [
                "id", "name", "status", "type",
                "started", "completed", "last_update",
            ],
            "order_by": ["started"],
        })
        return data.get("tasks", [])

    # -- Workers (NEW) --

    def get_workers(self) -> list[dict]:
        """Return all registered workers/agents."""
        data = self._post("workers.get_all", {})
        return data if isinstance(data, list) else data.get("workers", [])

    # -- Task management (clone / edit / enqueue) --

    def clone_task(self, task_id: str, new_name: str) -> str:
        """Clone a task. Returns new task ID."""
        data = self._post("tasks.clone", {"task": task_id, "new_task_name": new_name})
        return data.get("id", "")

    def edit_task_hyperparams(self, task_id: str, params: dict[str, str]) -> bool:
        """Update individual hyperparameters on a task (partial update).

        Keys use slash-separated format matching
        ``DAPIDLPipelineConfig.to_clearml_parameters()`` output
        (e.g. ``datasets/spec``, ``training/epochs``).  All keys are stored
        under the ClearML ``Args`` section.  The task must be in ``created``
        status (i.e. call this before ``enqueue_task``).
        """
        return bool(self._post("tasks.edit_hyper_params", {
            "task": task_id,
            "hyperparams": [
                {"section": "Args", "name": key, "value": str(value)}
                for key, value in params.items()
            ],
        }))

    def enqueue_task(self, task_id: str, queue_name: str) -> bool:
        """Enqueue a task to a named queue."""
        queues = self.get_queues()
        queue_id = next((q["id"] for q in queues if q.get("name") == queue_name), None)
        if not queue_id:
            logger.error("Queue %r not found", queue_name)
            return False
        return bool(self._post("tasks.enqueue", {"task": task_id, "queue": queue_id}))

    # -- Worker status --

    def get_worker_status(self) -> list[dict]:
        """Return simplified worker status for dashboard display.

        Each entry: {id, name, queue, task, gpu_usage}.
        """
        workers = self.get_workers()
        result = []
        for w in workers:
            task = w.get("task", {}) or {}
            queues = w.get("queues", [])
            queue_name = queues[0] if queues else ""
            # GPU stats from last report
            gpu_usage = ""
            machine = w.get("machine", {}) or {}
            gpus = machine.get("gpus", [])
            if gpus:
                g = gpus[0]
                used = g.get("mem_used", 0)
                total = g.get("mem_total", 0)
                name = g.get("name", "GPU")
                if total > 0:
                    gpu_usage = f"{name}: {used // (1024**2)}MB / {total // (1024**2)}MB"
            result.append({
                "id": w.get("id", ""),
                "name": w.get("id", "").split(":")[0] if w.get("id") else "",
                "queue": queue_name,
                "task": task.get("name", ""),
                "task_id": task.get("id", ""),
                "gpu_usage": gpu_usage,
            })
        return result

    # -- Queue stats --

    def get_queue_stats(self) -> list[dict]:
        """Return queues with entry counts for overview metrics."""
        data = self._post("queues.get_all", {})
        return data.get("queues", [])
