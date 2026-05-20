"""Tests for the dapidl QC pass (no ClearML)."""

import json
import struct

import numpy as np
import polars as pl

from dapidl.pipeline.steps.quality_control import run_quality_control


def _build_dataset(path):
    """Tiny single-slide dataset: 6 patches (3 sharp blobs, 3 flat) + metadata."""
    import lmdb

    yy, xx = np.mgrid[0:32, 0:32]
    blob = (200 + 4000 * np.exp(-(((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 5.0**2)))).astype(np.uint16)
    flat = np.full((32, 32), 200, dtype=np.uint16)
    patches = [blob, blob, blob, flat, flat, flat]

    env = lmdb.open(str(path / "patches.lmdb"), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, p in enumerate(patches):
            buf = struct.pack("I", 32) + struct.pack("I", 32) + p.tobytes()
            txn.put(str(i).encode(), buf)
    env.close()

    pl.DataFrame({
        "cell_id": [f"c{i}" for i in range(6)],
        "broad_category": ["Epithelial"] * 6,
    }).write_parquet(path / "metadata.parquet")
    (path / "slide_stats.json").write_text(json.dumps([{"source": "slideA", "n_written": 6}]))
    (path / "class_mapping.json").write_text(json.dumps({"Epithelial": 0}))


def test_run_writes_sidecar_with_scores(tmp_path):
    _build_dataset(tmp_path)
    run_quality_control(tmp_path, use_clearml=False, montage_top_n=4)

    sidecar = tmp_path / "qc" / "qc_scores.parquet"
    assert sidecar.exists()
    df = pl.read_parquet(sidecar)
    assert set(["cell_id", "focus_score", "detection_score", "qc_score"]).issubset(df.columns)
    assert df.height == 6
    by_id = dict(zip(df["cell_id"], df["detection_score"]))
    assert by_id["c0"] > by_id["c3"]


def test_provenance_written(tmp_path):
    _build_dataset(tmp_path)
    run_quality_control(tmp_path, use_clearml=False, montage_top_n=4)
    meta = json.loads((tmp_path / "qc" / "qc_scores.meta.json").read_text())
    assert meta["scorer"] == "classical"
    assert "date" in meta


def test_montage_reads_only_top_n(tmp_path, monkeypatch):
    _build_dataset(tmp_path)  # 6 patches, all class "Epithelial"
    captured = {}

    import dapidl.pipeline.steps.quality_control as step

    real = step.build_class_montage

    def spy(patches, scores, cell_type, top_n=64, cols=8):
        captured["n_patches"] = patches.shape[0]
        captured["top_n"] = top_n
        return real(patches, scores, cell_type, top_n=top_n, cols=cols)

    monkeypatch.setattr(step, "build_class_montage", spy)
    run_quality_control(tmp_path, use_clearml=False, montage_top_n=4)
    # 6 patches in the class but only top_n=4 should be read into the montage
    assert captured["n_patches"] <= 4


from unittest.mock import MagicMock, patch as mock_patch


def test_clearml_logging_called(tmp_path):
    _build_dataset(tmp_path)
    fake_task = MagicMock()
    fake_logger = MagicMock()
    fake_task.get_logger.return_value = fake_logger
    with mock_patch("clearml.Task") as TaskCls:
        TaskCls.current_task.return_value = None
        TaskCls.init.return_value = fake_task
        TaskCls.TaskTypes.qc = "qc"
        run_quality_control(tmp_path, use_clearml=True, montage_top_n=4)
    assert fake_logger.report_image.called
    assert fake_logger.report_histogram.called
    assert fake_task.upload_artifact.called
