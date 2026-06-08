"""Tests for the studio ClearML REST launch client (mocked _post — no network)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from studio.clearml_launch import ClearMLClient  # noqa: E402


def _client_capturing(calls, ret):
    c = ClearMLClient(api_server="http://x")

    def fake_post(endpoint, payload):
        calls.append((endpoint, payload))
        return ret(endpoint, payload) if callable(ret) else ret

    c._post = fake_post  # type: ignore[method-assign]
    return c


def test_clone_task_payload():
    calls = []
    c = _client_capturing(calls, {"id": "new1"})
    new_id = c.clone_task("tmpl1", "run-A")
    assert new_id == "new1"
    ep, pl = calls[-1]
    assert ep == "tasks.clone"
    assert pl["task"] == "tmpl1"
    assert pl["new_task_name"] == "run-A"


def test_set_task_params_builds_sectioned_hyperparams():
    calls = []
    c = _client_capturing(calls, {})
    c.set_task_params("new1", {"training/backbone": "resnet50", "training/epochs": "50"})
    ep, pl = calls[-1]
    assert ep == "tasks.edit"
    assert pl["task"] == "new1"
    assert pl["hyperparams"]["training"]["backbone"]["value"] == "resnet50"
    assert pl["hyperparams"]["training"]["backbone"]["section"] == "training"
    assert pl["hyperparams"]["training"]["epochs"]["value"] == "50"


def test_enqueue_task_payload():
    calls = []
    c = _client_capturing(calls, {"queued": 1})
    c.enqueue_task("new1", "gpu-cloud")
    ep, pl = calls[-1]
    assert ep == "tasks.enqueue"
    assert pl["task"] == "new1"
    assert pl["queue_name"] == "gpu-cloud"


def test_get_tasks_by_tag_filters_on_tag():
    calls = []
    c = _client_capturing(calls, {"tasks": [{"id": "a"}]})
    out = c.get_tasks_by_tag("sweep-42")
    assert out == [{"id": "a"}]
    assert "sweep-42" in calls[-1][1]["tags"]


def test_get_task_scalars_flattens_last_metrics():
    c = _client_capturing(
        [],
        {"task": {"last_metrics": {"h1": {"v1": {"metric": "val", "variant": "macro_f1", "value": 0.71}}}}},
    )
    scalars = c.get_task_scalars("t1")
    assert scalars["val/macro_f1"] == 0.71


def test_clone_task_with_tags_attaches_new_task_tags():
    calls = []
    c = _client_capturing(calls, {"id": "n2"})
    c.clone_task("tmpl", "run-B", tags=["sweep-9"])
    assert calls[-1][1]["new_task_tags"] == ["sweep-9"]
