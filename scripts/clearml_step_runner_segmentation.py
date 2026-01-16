#!/usr/bin/env python
"""ClearML step runner for segmentation step.

This is a thin wrapper that imports the main step runner and runs the segmentation step.
Having a separate script file (not a symlink) ensures ClearML creates unique tasks per step.
"""
import sys
from pathlib import Path

# Add the scripts directory to path so we can import the main runner
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import and run the main step runner with this step name
from clearml_step_runner import run_step

if __name__ == "__main__":
    run_step("segmentation")
