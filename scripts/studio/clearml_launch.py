"""ClearML REST client + launch helpers for the biologist studio.

`requests`-only (no ClearML SDK), so the dashboard stays lightweight and these
methods are unit-testable by monkeypatching `_post`. Launch works by cloning a
controller *template task*, overriding its hyperparameters, and enqueuing it to a
queue — so the run executes on a ClearML agent, not the dashboard host.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path

CLEARML_CONFIG_PATH = Path.home() / ".clearml" / "clearml-chrism.conf"


@dataclass
class ClearMLClient:
    """Minimal ClearML REST client for reading server state and launching runs."""

    api_server: str = ""
    access_key: str = ""
    secret_key: str = ""
    web_server: str = ""
    _token: str = ""
    _token_expiry: float = 0.0

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: Path = CLEARML_CONFIG_PATH) -> ClearMLClient:
        """Parse a HOCON-style ClearML config file."""
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
        """Create from environment variables, falling back to the config file."""
        if os.environ.get("CLEARML_API_HOST"):
            return cls(
                api_server=os.environ["CLEARML_API_HOST"],
                access_key=os.environ["CLEARML_API_ACCESS_KEY"],
                secret_key=os.environ["CLEARML_API_SECRET_KEY"],
                web_server=os.environ.get("CLEARML_WEB_HOST", ""),
            )
        return cls.from_config()

    # -- transport --------------------------------------------------------------
    @property
    def _headers(self) -> dict[str, str]:
        if time.time() > self._token_expiry - 60:
            self._authenticate()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def _authenticate(self) -> None:
        import requests

        creds = base64.b64encode(f"{self.access_key}:{self.secret_key}".encode()).decode()
        resp = requests.post(
            f"{self.api_server}/auth.login",
            json={},
            headers={"Content-Type": "application/json", "Authorization": f"Basic {creds}"},
            timeout=10,
        )
        resp.raise_for_status()
        self._token = resp.json()["data"]["token"]
        self._token_expiry = time.time() + 3600  # tokens valid ~24h, refresh hourly

    def _post(self, endpoint: str, payload: dict) -> dict:
        import requests

        resp = requests.post(
            f"{self.api_server}/{endpoint}", json=payload, headers=self._headers, timeout=30
        )
        resp.raise_for_status()
        return resp.json()["data"]

    # -- high-level reads -------------------------------------------------------
    def get_queues(self) -> list[dict]:
        return self._post("queues.get_all", {})["queues"]

    def get_projects(self) -> list[dict]:
        return self._post(
            "projects.get_all", {"page": 0, "page_size": 200, "only_fields": ["name", "id"]}
        )["projects"]

    def _resolve_project_names(self, tasks: list[dict]) -> list[dict]:
        project_ids = {t["project"] for t in tasks if isinstance(t.get("project"), str)}
        if not project_ids:
            return tasks
        proj_data = self._post(
            "projects.get_all", {"id": list(project_ids), "only_fields": ["id", "name"]}
        )
        id_to_name = {p["id"]: p["name"] for p in proj_data["projects"]}
        for t in tasks:
            pid = t.get("project")
            if isinstance(pid, str):
                t["project"] = {"id": pid, "name": id_to_name.get(pid, pid[:8])}
        return tasks

    def get_datasets(self) -> list[dict]:
        data = self._post("tasks.get_all", {
            "type": ["data_processing"], "status": ["completed"],
            "page": 0, "page_size": 500,
            "only_fields": ["id", "name", "project", "status", "tags", "last_update"],
            "order_by": ["-last_update"],
        })
        return self._resolve_project_names(data["tasks"])

    def get_pipelines(self, limit: int = 20) -> list[dict]:
        data = self._post("tasks.get_all", {
            "system_tags": ["pipeline"], "page": 0, "page_size": limit,
            "only_fields": ["id", "name", "status", "project", "started", "completed",
                            "last_update", "status_reason"],
            "order_by": ["-last_update"],
        })
        return self._resolve_project_names(data["tasks"])

    def get_task(self, task_id: str) -> dict:
        return self._post("tasks.get_by_id", {"task": task_id})["task"]

    def get_pipeline_steps(self, pipeline_id: str) -> list[dict]:
        data = self._post("tasks.get_all", {
            "parent": pipeline_id, "page": 0, "page_size": 50,
            "only_fields": ["id", "name", "status", "type", "started", "completed", "last_update"],
            "order_by": ["started"],
        })
        return data["tasks"]

    # -- launch: clone template -> override params -> enqueue -------------------
    def clone_task(self, template_id: str, name: str, tags: list[str] | None = None) -> str:
        """Clone a (template) task; return the new task id. Optional user tags let a
        sweep's runs be grouped/queried together later."""
        payload: dict = {"task": template_id, "new_task_name": name}
        if tags:
            payload["new_task_tags"] = list(tags)
        return self._post("tasks.clone", payload)["id"]

    def set_task_params(self, task_id: str, params: dict[str, str]) -> dict:
        """Set hyperparameters on a task. Slash keys ("training/backbone") become
        sectioned hyperparams (section="training", name="backbone")."""
        hyperparams: dict[str, dict] = {}
        for key, value in params.items():
            section, sep, name = key.partition("/")
            if not sep:
                section, name = "Args", key
            hyperparams.setdefault(section, {})[name] = {
                "section": section, "name": name, "value": str(value),
            }
        return self._post("tasks.edit", {"task": task_id, "hyperparams": hyperparams, "force": True})

    def enqueue_task(self, task_id: str, queue_name: str) -> dict:
        """Enqueue a task to a named queue (a ClearML agent picks it up)."""
        return self._post("tasks.enqueue", {"task": task_id, "queue_name": queue_name})

    def get_tasks_by_tag(self, tag: str, limit: int = 100) -> list[dict]:
        """Return tasks carrying a user tag (used to group a sweep's runs)."""
        data = self._post("tasks.get_all", {
            "tags": [tag], "page": 0, "page_size": limit,
            "only_fields": ["id", "name", "status", "tags", "started", "completed", "last_metrics"],
            "order_by": ["-last_update"],
        })
        return data["tasks"]

    def get_task_scalars(self, task_id: str) -> dict[str, float]:
        """Flatten a task's last reported scalars to {"metric/variant": value}."""
        task = self.get_task(task_id)
        out: dict[str, float] = {}
        for metric in (task.get("last_metrics") or {}).values():
            for variant in metric.values():
                name = f"{variant.get('metric', '')}/{variant.get('variant', '')}".strip("/")
                if name:
                    out[name] = variant.get("value")
        return out
