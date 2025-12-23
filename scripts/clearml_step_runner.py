#!/usr/bin/env python3
"""ClearML Step Runner - Standalone script for remote pipeline execution.

This script is designed to be executed by ClearML agents. It avoids the
entry point issue where uv-managed CLI tools aren't available inside Docker.

Usage (by ClearML agent):
    python scripts/clearml_step_runner.py --step data_loader

The script reads step configuration from ClearML Task parameters.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Early logging to stderr for debugging
def early_log(msg):
    """Log to stderr before logging framework is available."""
    print(f"[clearml_step_runner] {msg}", file=sys.stderr, flush=True)

early_log(f"Python: {sys.version}")
early_log(f"Working dir: {Path.cwd()}")
early_log(f"Script path: {__file__}")

# Add src to path for development mode
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    early_log(f"Adding to path: {src_path}")
    sys.path.insert(0, str(src_path))
else:
    early_log(f"src path not found: {src_path}")


def run_step(step_name: str):
    """Run a pipeline step with configuration from ClearML Task parameters."""
    from clearml import Task
    from loguru import logger

    # Initialize/connect to existing task
    task = Task.current_task()
    if task is None:
        logger.error("No ClearML task found. This script must be run by ClearML agent.")
        sys.exit(1)

    logger.info(f"Running step: {step_name}")
    logger.info(f"Task ID: {task.id}")

    # Get step configuration from task parameters
    step_config = task.get_parameters_as_dict().get("step_config", {})
    logger.info(f"Step config: {step_config}")

    # Import and run the appropriate step
    if step_name == "data_loader":
        from dapidl.pipeline.steps.data_loader import DataLoaderConfig, DataLoaderStep

        config = DataLoaderConfig(
            dataset_id=step_config.get("dataset_id"),
            dataset_name=step_config.get("dataset_name"),
            dataset_project=step_config.get("dataset_project", "DAPIDL/datasets"),
            platform=step_config.get("platform", "auto"),
            local_path=step_config.get("local_path"),
        )
        step = DataLoaderStep(config)

    elif step_name == "segmentation":
        from dapidl.pipeline.steps.segmentation import (
            SegmentationStep,
            SegmentationStepConfig,
        )

        config = SegmentationStepConfig(
            segmenter=step_config.get("segmenter", "cellpose"),
            diameter=int(step_config.get("diameter", 40)),
            flow_threshold=float(step_config.get("flow_threshold", 0.4)),
            match_threshold_um=float(step_config.get("match_threshold_um", 5.0)),
            platform=step_config.get("platform", "xenium"),
        )
        step = SegmentationStep(config)

    elif step_name == "annotation":
        from dapidl.pipeline.steps.annotation import (
            AnnotationStep,
            AnnotationStepConfig,
        )

        model_names_str = step_config.get("model_names", "Cells_Adult_Breast.pkl")
        model_names = (
            model_names_str.split(",")
            if isinstance(model_names_str, str)
            else model_names_str
        )

        config = AnnotationStepConfig(
            annotator=step_config.get("annotator", "celltypist"),
            strategy=step_config.get("strategy", "consensus"),
            model_names=model_names,
            confidence_threshold=float(step_config.get("confidence_threshold", 0.5)),
            ground_truth_file=step_config.get("ground_truth_file"),
            fine_grained=step_config.get("fine_grained", "false").lower() == "true"
            if isinstance(step_config.get("fine_grained"), str)
            else bool(step_config.get("fine_grained", False)),
        )
        step = AnnotationStep(config)

    elif step_name == "patch_extraction":
        from dapidl.pipeline.steps.patch_extraction import (
            PatchExtractionConfig,
            PatchExtractionStep,
        )

        config = PatchExtractionConfig(
            patch_size=int(step_config.get("patch_size", 128)),
            output_format=step_config.get("output_format", "lmdb"),
            normalization_method=step_config.get("normalization_method", "adaptive"),
        )
        step = PatchExtractionStep(config)

    elif step_name == "training":
        from dapidl.pipeline.steps.training import TrainingStep, TrainingStepConfig

        config = TrainingStepConfig(
            backbone=step_config.get("backbone", "efficientnetv2_rw_s"),
            epochs=int(step_config.get("epochs", 50)),
            batch_size=int(step_config.get("batch_size", 128)),
            learning_rate=float(step_config.get("learning_rate", 3e-4)),
        )
        step = TrainingStep(config)

    else:
        logger.error(f"Unknown step: {step_name}")
        sys.exit(1)

    # Get input artifacts from parent tasks (via ClearML pipeline)
    from dapidl.pipeline.base import StepArtifacts

    inputs = {}

    # Check for artifacts from parent tasks
    pipeline_artifacts = task.artifacts
    for name, artifact in pipeline_artifacts.items():
        logger.info(f"Found artifact: {name}")
        # Load artifact based on type
        if artifact.type == "json":
            inputs[name] = json.loads(artifact.get())
        else:
            inputs[name] = artifact.get()

    artifacts = StepArtifacts(inputs=inputs, outputs={})

    # Execute step
    logger.info(f"Executing step: {step_name}")
    result = step.execute(artifacts)

    # Upload output artifacts
    for key, value in result.outputs.items():
        if isinstance(value, (dict, list)):
            task.upload_artifact(key, json.dumps(value))
        elif isinstance(value, (str, Path)) and Path(value).exists():
            task.upload_artifact(key, str(value))
        else:
            task.upload_artifact(key, str(value))

    logger.info(f"Step {step_name} completed successfully")
    logger.info(f"Outputs: {list(result.outputs.keys())}")


def main():
    import os

    early_log("Entered main()")

    valid_steps = ["data_loader", "segmentation", "annotation", "patch_extraction", "training"]

    # Method 1: Try to connect to ClearML task (works when running under agent)
    # Task.init() will reconnect to existing task when running remotely
    try:
        early_log("Attempting to connect to ClearML task...")
        from clearml import Task

        # Check if we're running under ClearML agent by looking for task ID env var
        task_id = os.environ.get("CLEARML_TASK_ID", "")
        early_log(f"CLEARML_TASK_ID: {task_id!r}")

        if task_id:
            # Running under ClearML agent - reconnect to the task
            early_log("Running under ClearML agent, reconnecting to task...")
            task = Task.init()
            early_log(f"Connected to task: {task.name} (id={task.id})")

            # Get step name from task name (pipeline creates tasks like "data_loader")
            if task.name in valid_steps:
                early_log(f"Running step '{task.name}' from ClearML task name")
                run_step(task.name)
                return

            # Also check step_config parameters
            params = task.get_parameters_as_dict()
            early_log(f"Task params: {list(params.keys())}")
            step_name = params.get("step_config", {}).get("step_name")
            early_log(f"step_name from params: {step_name}")
            if step_name and step_name in valid_steps:
                early_log(f"Running step '{step_name}' from task parameters")
                run_step(step_name)
                return

            # If task name is not a valid step, might be a prefixed name like "step-data_loader"
            for step in valid_steps:
                if step in task.name:
                    early_log(f"Found step '{step}' in task name '{task.name}'")
                    run_step(step)
                    return

            early_log(f"Could not determine step from task name: {task.name}")
    except Exception as e:
        early_log(f"Could not connect to ClearML task: {e}")
        early_log(traceback.format_exc())

    # Method 2: Get step name from CLEARML_TASK_NAME environment variable
    task_name = os.environ.get("CLEARML_TASK_NAME", "")
    early_log(f"CLEARML_TASK_NAME: {task_name!r}")

    if task_name in valid_steps:
        early_log(f"Running step '{task_name}' from CLEARML_TASK_NAME env var")
        run_step(task_name)
        return

    # Fallback to argparse for local execution/testing
    early_log("Falling back to argparse (local execution mode)")
    parser = argparse.ArgumentParser(description="ClearML Pipeline Step Runner")
    parser.add_argument(
        "--step",
        required=True,
        choices=valid_steps,
        help="Step to execute",
    )
    args = parser.parse_args()

    run_step(args.step)


if __name__ == "__main__":
    try:
        early_log("Starting main()")
        main()
    except Exception as e:
        early_log(f"FATAL ERROR: {type(e).__name__}: {e}")
        early_log(traceback.format_exc())
        sys.exit(1)
