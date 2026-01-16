#!/usr/bin/env python
"""ClearML step runner for cross_validation step.

This is a thin wrapper that imports the main step runner and runs the cross_validation step.
Having a separate script file ensures ClearML creates unique tasks per step.

IMPORTANT: We use continue_last_task=True to connect to the cloned task when
running under ClearML agent, since CLEARML_TASK_ID isn't set in non-Docker mode.
"""
import os
import sys
from pathlib import Path

# Add the scripts directory to path so we can import the main runner
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Check if we're running under ClearML agent (has CLEARML_WORKER_ID but no CLEARML_TASK_ID)
worker_id = os.environ.get("CLEARML_WORKER_ID")
task_id = os.environ.get("CLEARML_TASK_ID")

if worker_id and not task_id:
    # Running under agent but CLEARML_TASK_ID not set
    # Use continue_last_task to connect to the most recent task
    from clearml import Task
    task = Task.init(continue_last_task=True)
    print(f"[step_runner] Connected to task: {task.id}")

# Import and run the main step runner with this step name
from clearml_step_runner import run_step

if __name__ == "__main__":
    run_step("cross_validation")
