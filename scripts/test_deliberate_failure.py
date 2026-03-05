"""Deliberate failure test for ClearML autoscaler verification.

Verifies that task failures are properly reported and don't crash the autoscaler.
"""
from clearml import Task


def main() -> None:
    task = Task.init(project_name="DAPIDL/tests", task_name="deliberate-failure-test")
    logger = task.get_logger()
    logger.report_text("About to raise deliberate error...")
    raise RuntimeError("DELIBERATE_FAILURE: This task is supposed to fail")


if __name__ == "__main__":
    main()
