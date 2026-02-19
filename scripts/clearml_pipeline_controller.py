#!/usr/bin/env python3
"""ClearML Pipeline Controller — launchable from the ClearML Web UI.

This script runs as a ClearML task on a services (or default) queue agent.
It reads pipeline configuration from task parameters, builds a
UnifiedPipelineController, and orchestrates the pipeline while the agent
keeps the task alive.

Workflow:
    1. CLI: `dapidl clearml-pipeline create-controller-task -t lung bf8f913f xenium 2`
       → creates a ClearML task with editable parameters
    2. Web UI: clone the task → edit params → enqueue to `services`
    3. Agent picks up this script, reads params, runs the pipeline

The controller blocks until all pipeline steps complete (steps dispatch to
default/gpu queues via ClearML agents).
"""

import sys
import traceback
from pathlib import Path

# Early logging to stderr for debugging (before any imports that might fail)
def early_log(msg: str) -> None:
    print(f"[pipeline_controller] {msg}", file=sys.stderr, flush=True)

early_log(f"Python: {sys.version}")
early_log(f"Working dir: {Path.cwd()}")

# Add src to path for development mode
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
    early_log(f"Added to path: {src_path}")


def main() -> None:
    from clearml import Task
    from loguru import logger

    # Connect to the ClearML task the agent is running
    task = Task.init(
        project_name="DAPIDL/pipelines",
        task_name="dapidl-pipeline-controller",
        task_type=Task.TaskTypes.controller,
        reuse_last_task_id=False,
    )
    logger.info(f"Controller task: {task.id}")

    # Read all parameters (set when the task was created or edited in UI)
    params = task.get_parameters_as_dict()
    logger.info(f"Parameter sections: {list(params.keys())}")

    # ClearML nests parameters under section names — flatten to slash-separated keys.
    # The "Args" section is the default bucket where ClearML puts slash-separated
    # keys like "datasets/spec", "training/epochs", etc.  We strip the "Args/"
    # prefix so from_clearml_parameters() sees the expected key format.
    flat_params: dict[str, str] = {}
    for section, values in params.items():
        if isinstance(values, dict):
            for key, val in values.items():
                if section == "Args":
                    flat_params[key] = str(val) if val is not None else ""
                else:
                    flat_params[f"{section}/{key}"] = str(val) if val is not None else ""
        else:
            flat_params[section] = str(values)

    logger.info(f"Flat params ({len(flat_params)} keys): {list(flat_params.keys())}")

    # Build pipeline config from parameters
    from dapidl.pipeline.unified_config import DAPIDLPipelineConfig

    config = DAPIDLPipelineConfig.from_clearml_parameters(flat_params)

    # Resolve dataset names to IDs where needed
    from clearml import Dataset

    for tc in config.input.tissues:
        if tc.dataset_id and "-" in tc.dataset_id:
            # Looks like a dataset name — resolve to ID
            name = tc.dataset_id
            logger.info(f"Resolving dataset name '{name}'...")
            matches = Dataset.list_datasets(
                dataset_project="DAPIDL/datasets",
                partial_name=name,
                only_completed=False,
            )
            exact = [d for d in matches if d.get("name") == name]
            if exact:
                tc.dataset_id = exact[0]["id"]
                logger.info(f"  Resolved '{name}' → {tc.dataset_id[:8]}")
            elif matches:
                tc.dataset_id = matches[0]["id"]
                logger.warning(
                    f"  No exact match for '{name}', using closest: "
                    f"{matches[0].get('name')} ({tc.dataset_id[:8]})"
                )
            else:
                logger.error(f"  Dataset '{name}' not found in DAPIDL/datasets")
                task.mark_failed(status_reason=f"Dataset not found: {name}")
                sys.exit(1)

    # Log resolved config
    n_tissues = len(config.input.tissues)
    logger.info(f"Datasets: {n_tissues}")
    for tc in config.input.tissues:
        source = tc.local_path or tc.dataset_id
        logger.info(f"  {tc.tissue}/{tc.platform.value}: {source} (tier {tc.confidence_tier})")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Execute remotely: {config.execution.execute_remotely}")

    if n_tissues == 0:
        logger.error("No datasets configured. Edit 'datasets/spec' parameter.")
        logger.error("One ClearML dataset name per line, e.g.:")
        logger.error("  xenium-breast-tumor-rep1-raw")
        logger.error("  merscope-breast-raw")
        task.mark_failed(status_reason="No datasets configured — edit datasets/spec")
        sys.exit(1)

    # Choose orchestration mode
    use_orchestrator = flat_params.get("execution/use_orchestrator", "").lower() in ("true", "1", "yes")

    if use_orchestrator:
        # Task-based orchestrator: supports per-dataset recipes
        from dapidl.pipeline.orchestrator import PipelineOrchestrator

        logger.info("Using task-based orchestrator")
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run()

        if result.success:
            logger.info(f"Pipeline completed. LMDBs: {len(result.lmdb_paths)}")
            if result.model_path:
                logger.info(f"Model: {result.model_path}")
            task.get_logger().report_text(f"Pipeline completed. LMDBs: {len(result.lmdb_paths)}")
        else:
            logger.error(f"Pipeline failed: {result.error}")
            task.mark_failed(status_reason=result.error or "Pipeline failed")
            sys.exit(1)
    else:
        # Legacy mode: ClearML PipelineController DAG
        from dapidl.pipeline.unified_controller import UnifiedPipelineController

        controller = UnifiedPipelineController(config)
        controller.create_pipeline()

        logger.info("Starting pipeline (controller on agent, steps on queues)...")
        pipeline_id = controller.run()

        logger.info(f"Pipeline completed: {pipeline_id}")
        task.get_logger().report_text(f"Pipeline completed: {pipeline_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        early_log(f"FATAL: {type(e).__name__}: {e}")
        early_log(traceback.format_exc())
        sys.exit(1)
