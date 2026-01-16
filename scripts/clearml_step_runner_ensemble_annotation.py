#!/usr/bin/env python
"""ClearML step runner for ensemble_annotation step.

This is a thin wrapper that imports the main step runner and runs the ensemble_annotation step.
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
    # Running under agent but CLEARML_TASK_ID not set (non-Docker mode)
    # Query the workers API to find what task this worker is executing
    # IMPORTANT: Set CLEARML_TASK_ID BEFORE importing Task to ensure it's respected!

    # Only import Session for the query - avoid importing Task yet
    from clearml.backend_api import Session

    found_task_id = None
    try:
        session = Session()
        response = session.send_request(
            service='workers',
            action='get_all',
            method='GET',
            data={'last_seen': 300}
        )
        resp_json = response.json() if hasattr(response, 'json') else {}
        workers = resp_json.get('data', {}).get('workers', [])
        print(f"[step_runner] Found {len(workers)} workers, looking for {worker_id}")
        for w in workers:
            w_id = w.get('id', '')
            if w_id == worker_id:
                task_info = w.get('task')
                if task_info:
                    found_task_id = task_info.get('id')
                    print(f"[step_runner] Found task for worker {worker_id}: {found_task_id}")
                    # Set env var BEFORE importing Task
                    os.environ['CLEARML_TASK_ID'] = found_task_id
                break
    except Exception as e:
        print(f"[step_runner] Warning: Could not query worker API: {e}")
        import traceback
        traceback.print_exc()

    if found_task_id:
        # NOW import Task - after CLEARML_TASK_ID is set
        from clearml import Task
        task = Task.init()
        if task.id != found_task_id:
            print(f"[step_runner] ERROR: Expected task {found_task_id} but connected to {task.id}")
            print("[step_runner] CLEARML_TASK_ID environment variable was not respected!")
            sys.exit(1)
        print(f"[step_runner] Connected to task: {task.id}")
    else:
        # CRITICAL: Do NOT call Task.init() without task ID - it will connect to wrong task!
        print(f"[step_runner] ERROR: Could not find task ID for worker {worker_id}")
        print("[step_runner] Worker may not be executing a task, or timing issue with API")
        sys.exit(1)

# Import and run the main step runner with this step name
from clearml_step_runner import run_step

if __name__ == "__main__":
    run_step("ensemble_annotation")
