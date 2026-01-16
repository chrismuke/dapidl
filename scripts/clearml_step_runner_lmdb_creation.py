#!/usr/bin/env python
"""ClearML step runner for lmdb_creation step.

This is a thin wrapper that imports the main step runner and runs the lmdb_creation step.
Having a separate script file ensures ClearML creates unique tasks per step.
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
    # Query the workers API to find what task this worker is executing
    from clearml.backend_api.session.client import APIClient
    from clearml import Task

    try:
        client = APIClient()
        # Get all workers using the proper client API
        response = client.workers.get_all(last_seen=60)
        # Response is a list of worker objects
        workers = response.workers if hasattr(response, 'workers') else []
        print(f"[step_runner] Found {len(workers)} workers, looking for {worker_id}")
        for w in workers:
            w_id = w.id if hasattr(w, 'id') else w.get('id')
            if w_id == worker_id:
                task_info = w.task if hasattr(w, 'task') else w.get('task')
                if task_info:
                    current_task_id = task_info.id if hasattr(task_info, 'id') else task_info.get('id')
                    print(f"[step_runner] Found task for worker {worker_id}: {current_task_id}")
                    # Set the environment variable so Task.init() will use it
                    os.environ['CLEARML_TASK_ID'] = current_task_id
                    break
    except Exception as e:
        print(f"[step_runner] Warning: Could not query worker API: {e}")
        import traceback
        traceback.print_exc()

    # Now init the task - should connect to the correct one if we found it
    task = Task.init()
    print(f"[step_runner] Connected to task: {task.id}")

# Import and run the main step runner with this step name
from clearml_step_runner import run_step

if __name__ == "__main__":
    run_step("lmdb_creation")
