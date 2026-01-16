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


def _parse_bool(value) -> bool:
    """Parse a boolean from various input types (str, bool, int)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)

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
            weight_decay=float(step_config.get("weight_decay", 1e-5)),
            use_weighted_loss=_parse_bool(step_config.get("use_weighted_loss", True)),
            use_weighted_sampler=_parse_bool(step_config.get("use_weighted_sampler", True)),
            max_weight_ratio=float(step_config.get("max_weight_ratio", 10.0)),
            patience=int(step_config.get("patience", 15)),
        )
        step = TrainingStep(config)

    elif step_name == "ensemble_annotation":
        from dapidl.pipeline.steps.ensemble_annotation import (
            EnsembleAnnotationConfig,
            EnsembleAnnotationStep,
        )

        # Parse celltypist_models list from comma-separated string
        models_str = step_config.get("celltypist_models", "Cells_Adult_Breast.pkl,Immune_All_High.pkl")
        models = models_str.split(",") if isinstance(models_str, str) else models_str

        config = EnsembleAnnotationConfig(
            celltypist_models=models,
            include_singler=_parse_bool(step_config.get("include_singler", True)),
            singler_reference=step_config.get("singler_reference", "blueprint"),
            include_sctype=_parse_bool(step_config.get("include_sctype", False)),
            min_agreement=int(step_config.get("min_agreement", 2)),
            confidence_threshold=float(step_config.get("confidence_threshold", 0.5)),
            use_confidence_weighting=_parse_bool(step_config.get("use_confidence_weighting", False)),
            fine_grained=_parse_bool(step_config.get("fine_grained", False)),
            skip_if_exists=_parse_bool(step_config.get("skip_if_exists", True)),
            create_derived_dataset=_parse_bool(step_config.get("create_derived_dataset", True)),
        )
        step = EnsembleAnnotationStep(config)

    elif step_name == "lmdb_creation":
        from dapidl.pipeline.steps.lmdb_creation import (
            LMDBCreationConfig,
            LMDBCreationStep,
        )

        config = LMDBCreationConfig(
            patch_size=int(step_config.get("patch_size", 128)),
            normalization_method=step_config.get("normalization_method", "adaptive"),
            normalize_physical_size=_parse_bool(step_config.get("normalize_physical_size", True)),
            target_pixel_size_um=float(step_config.get("target_pixel_size_um", 0.2125)),
            exclude_edge_cells=_parse_bool(step_config.get("exclude_edge_cells", True)),
            edge_margin_px=int(step_config.get("edge_margin_px", 64)),
            skip_if_exists=_parse_bool(step_config.get("skip_if_exists", True)),
            create_clearml_dataset=_parse_bool(step_config.get("create_clearml_dataset", True)),
        )
        step = LMDBCreationStep(config)

    elif step_name == "cross_validation":
        from dapidl.pipeline.steps.cross_validation import (
            CrossValidationConfig,
            CrossValidationStep,
        )

        config = CrossValidationConfig(
            run_leiden_check=_parse_bool(step_config.get("run_leiden_check", True)),
            run_dapi_check=_parse_bool(step_config.get("run_dapi_check", True)),
            run_consensus_check=_parse_bool(step_config.get("run_consensus_check", True)),
            min_ari_threshold=float(step_config.get("min_ari_threshold", 0.5)),
            min_agreement_threshold=float(step_config.get("min_agreement_threshold", 0.5)),
        )
        step = CrossValidationStep(config)

    else:
        logger.error(f"Unknown step: {step_name}")
        sys.exit(1)

    # Get input artifacts from parent tasks (via step_config parameters)
    # The pipeline controller passes parent outputs as step_config parameters
    from dapidl.pipeline.base import StepArtifacts

    inputs = {}

    # Extract parent outputs from step_config (set by pipeline parameter_override)
    parent_output_keys = [
        "data_path", "platform", "cells_parquet", "expression_path",
        "segmentation_result", "annotations", "dataset_path",
        "num_classes", "class_names", "metadata"
    ]
    for key in parent_output_keys:
        if key in step_config and step_config[key]:
            value = step_config[key]
            # Handle JSON-encoded values
            if isinstance(value, str) and value.startswith("{"):
                try:
                    inputs[key] = json.loads(value)
                except json.JSONDecodeError:
                    inputs[key] = value
            else:
                inputs[key] = value
            logger.info(f"Got parent output: {key} = {value[:100] if isinstance(value, str) and len(value) > 100 else value}")

    # Parent outputs go in 'outputs' - steps access via artifacts.outputs
    artifacts = StepArtifacts(inputs={}, outputs=inputs)

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

    valid_steps = [
        "data_loader", "segmentation", "annotation", "patch_extraction", "training",
        # SOTA pipeline steps
        "ensemble_annotation", "lmdb_creation", "cross_validation",
    ]

    # Log all CLEARML environment variables for debugging
    clearml_env_vars = {k: v for k, v in os.environ.items() if 'CLEARML' in k.upper()}
    early_log(f"CLEARML env vars: {clearml_env_vars}")

    # Log command line args
    early_log(f"sys.argv: {sys.argv}")

    # Method 0: Check argparse first (most reliable - ClearML sets this via argparse_args)
    # Parse args early to get step name
    parser = argparse.ArgumentParser(description="ClearML Pipeline Step Runner")
    parser.add_argument(
        "--step",
        required=False,  # Not required, we have fallbacks
        choices=valid_steps,
        help="Step to execute",
    )
    args, unknown = parser.parse_known_args()
    early_log(f"Parsed args: step={args.step}, unknown={unknown}")

    if args.step:
        early_log(f"Running step '{args.step}' from command line argument")
        run_step(args.step)
        return

    # Method 0.5: Extract step name from script filename
    # Script names like clearml_step_runner_data_loader.py contain the step name
    script_name = Path(sys.argv[0]).stem  # e.g., "clearml_step_runner_data_loader"
    early_log(f"Script name: {script_name}")
    if "_" in script_name:
        # Try to extract step from script name (after "clearml_step_runner_")
        script_parts = script_name.split("_", 3)  # ['clearml', 'step', 'runner', 'step_name']
        if len(script_parts) > 3:
            step_from_script = script_parts[3]
            early_log(f"Extracted step from script name: {step_from_script}")
            if step_from_script in valid_steps:
                early_log(f"Running step '{step_from_script}' from script name")
                run_step(step_from_script)
                return

    # Method 1: Try to connect to ClearML task (works when running under agent)
    try:
        early_log("Attempting to connect to ClearML task...")
        from clearml import Task

        # CLEARML_PROC_MASTER_ID contains the task ID when running under agent
        # Format: "PID:TASK_ID"
        proc_master_id = os.environ.get("CLEARML_PROC_MASTER_ID", "")
        early_log(f"CLEARML_PROC_MASTER_ID: {proc_master_id!r}")

        task = None
        if proc_master_id and ":" in proc_master_id:
            agent_task_id = proc_master_id.split(":", 1)[1]
            early_log(f"Extracted agent task_id: {agent_task_id}")
            # Get the task that the agent is running
            task = Task.get_task(task_id=agent_task_id)
            early_log(f"Task.get_task() returned: {task}")

        if task is None:
            # Fallback: try current_task (in case ClearML injected Task.init())
            task = Task.current_task()
            early_log(f"Task.current_task() returned: {task}")

        if task is None:
            # Call Task.init() with reuse_last_task_id=False to prevent reusing old tasks
            # Under ClearML agent, this should connect to the current running task
            early_log("No current task, calling Task.init(reuse_last_task_id=False)...")
            task = Task.init(reuse_last_task_id=False)
            early_log(f"Task.init() returned: {task}")

        if task:
            early_log(f"Connected to task: {task.name} (id={task.id})")

            # Get step name from task name (pipeline creates tasks like "data_loader")
            if task.name in valid_steps:
                early_log(f"Running step '{task.name}' from ClearML task name")
                run_step(task.name)
                return

            # Check task parameters for step name
            params = task.get_parameters_as_dict()
            early_log(f"Task params keys: {list(params.keys())}")
            early_log(f"Full params: {params}")

            # ClearML stores argparse_args under Args/ namespace
            step_name = params.get("Args", {}).get("step")
            early_log(f"step from Args/step: {step_name}")
            if step_name and step_name in valid_steps:
                early_log(f"Running step '{step_name}' from Args/step parameter")
                run_step(step_name)
                return

            # Also check step_config (legacy/fallback)
            step_name = params.get("step_config", {}).get("step_name")
            early_log(f"step_name from step_config: {step_name}")
            if step_name and step_name in valid_steps:
                early_log(f"Running step '{step_name}' from step_config parameters")
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

    # Final fallback - require --step argument
    early_log("ERROR: Could not determine step to run")
    early_log("Please provide --step argument or run under ClearML agent")
    sys.exit(1)


if __name__ == "__main__":
    try:
        early_log("Starting main()")
        main()
    except Exception as e:
        early_log(f"FATAL ERROR: {type(e).__name__}: {e}")
        early_log(traceback.format_exc())
        sys.exit(1)
