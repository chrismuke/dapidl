"""Tests for the `dapidl qc` CLI command."""

from unittest.mock import patch

from click.testing import CliRunner

from dapidl.cli import main


def test_qc_command_help():
    result = CliRunner().invoke(main, ["qc", "--help"])
    assert result.exit_code == 0
    assert "dataset" in result.output.lower()


def test_qc_command_invokes_step(tmp_path):
    with patch("dapidl.pipeline.steps.quality_control.run_quality_control") as run:
        result = CliRunner().invoke(
            main, ["qc", "--dataset", str(tmp_path), "--no-clearml"]
        )
    assert result.exit_code == 0
    assert run.called
    _, kwargs = run.call_args
    assert kwargs["use_clearml"] is False
