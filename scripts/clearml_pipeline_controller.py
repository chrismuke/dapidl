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

import importlib
import importlib.util
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


def _import_module_directly(module_name: str, file_path: Path):
    """Import a module directly from file, bypassing __init__.py chains.

    The dapidl.pipeline.__init__.py imports heavy dependencies (torch, cellpose,
    scanpy) that aren't needed for the controller. This function loads specific
    modules without triggering the full import chain.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    # Note: get_parameters_as_dict() returns nested dicts for slash-separated keys,
    # e.g. "datasets/spec" becomes {"datasets": {"spec": value}}.
    def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
        result: dict[str, str] = {}
        for key, val in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                result.update(_flatten(val, full_key))
            else:
                result[full_key] = str(val) if val is not None else ""
        return result

    flat_params: dict[str, str] = {}
    for section, values in params.items():
        if isinstance(values, dict):
            for key, val in _flatten(values).items():
                if section == "Args":
                    flat_params[key] = val
                else:
                    flat_params[f"{section}/{key}"] = val
        else:
            flat_params[section] = str(values)

    logger.info(f"Flat params ({len(flat_params)} keys): {list(flat_params.keys())}")

    # Build pipeline config from parameters
    # Import directly to avoid dapidl.pipeline.__init__.py which pulls torch, cellpose, etc.
    # We need stub module entries so sub-imports like "from dapidl.pipeline.unified_config"
    # don't trigger the heavy __init__.py chain.
    import types
    _pipeline_src = src_path / "dapidl" / "pipeline"
    if "dapidl" not in sys.modules:
        sys.modules["dapidl"] = types.ModuleType("dapidl")
    if "dapidl.pipeline" not in sys.modules:
        sys.modules["dapidl.pipeline"] = types.ModuleType("dapidl.pipeline")

    unified_config = _import_module_directly(
        "dapidl.pipeline.unified_config", _pipeline_src / "unified_config.py"
    )
    DAPIDLPipelineConfig = unified_config.DAPIDLPipelineConfig

    # Convert dashboard-style params to unified_config format
    # Dashboard sends: tissues (JSON), gpu_queue, default_queue, skip_training, epochs, etc.
    # from_clearml_parameters expects: datasets/spec, execution/*, training/*, etc.
    import json as _json

    if "tissues" in flat_params and "datasets/spec" not in flat_params:
        tissues_json = flat_params.pop("tissues", "[]")
        try:
            tissues_list = _json.loads(tissues_json)
        except (ValueError, TypeError):
            tissues_list = []

        # Convert tissues list to datasets/spec format (one dataset_id per line)
        spec_lines = []
        TissueDatasetConfig = unified_config.TissueDatasetConfig
        Platform = unified_config.Platform
        parsed_tissues = []
        for t in tissues_list:
            tc = TissueDatasetConfig(
                tissue=t.get("tissue", "unknown"),
                dataset_id=t.get("dataset_id"),
                platform=Platform(t.get("platform", "auto")),
                confidence_tier=int(t.get("segmentation_method", 2)),
            )
            parsed_tissues.append(tc)

        # Map other dashboard params
        if "epochs" in flat_params and "training/epochs" not in flat_params:
            flat_params["training/epochs"] = flat_params.pop("epochs")
        if "gpu_queue" in flat_params and "execution/gpu_queue" not in flat_params:
            flat_params["execution/gpu_queue"] = flat_params.pop("gpu_queue")
        if "default_queue" in flat_params and "execution/default_queue" not in flat_params:
            flat_params["execution/default_queue"] = flat_params.pop("default_queue")
        if "skip_training" in flat_params and "execution/skip_training" not in flat_params:
            flat_params["execution/skip_training"] = flat_params.pop("skip_training")
        if "sampling_strategy" in flat_params and "training/sampling_strategy" not in flat_params:
            flat_params["training/sampling_strategy"] = flat_params.pop("sampling_strategy")

        logger.info(f"Parsed {len(parsed_tissues)} tissues from JSON")

    config = DAPIDLPipelineConfig.from_clearml_parameters(flat_params)

    # If we parsed tissues from JSON, inject them directly
    if "parsed_tissues" in dir():
        config.input.tissues = parsed_tissues

    # Resolve dataset names to IDs where needed
    from clearml import Dataset

    for tc in config.input.tissues:
        if tc.dataset_id and "-" in tc.dataset_id:
            # Looks like a dataset name — resolve to ID
            name = tc.dataset_id
            logger.info(f"Resolving dataset name '{name}'...")
            # Search across all DAPIDL sub-projects (raw-data, datasets, etc.)
            matches = Dataset.list_datasets(
                partial_name=name,
                only_completed=False,
            )
            # Filter to DAPIDL project tree
            matches = [d for d in matches if d.get("project", "").startswith("DAPIDL")]
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

    # Log resolved config — use report_text for guaranteed flush (loguru gets batched)
    n_tissues = len(config.input.tissues)
    tl = task.get_logger()
    tl.report_text(f"Datasets: {n_tissues}")
    for tc in config.input.tissues:
        source = tc.local_path or tc.dataset_id
        tl.report_text(f"  {tc.tissue}/{tc.platform.value}: {source} (tier {tc.confidence_tier})")
    tl.report_text(f"Epochs: {config.training.epochs}")
    tl.report_text(f"Execute remotely: {config.execution.execute_remotely}")
    tl.report_text(f"GPU queue: {config.execution.gpu_queue}")
    tl.report_text(f"Default queue: {config.execution.default_queue}")
    tl.report_text(f"Cache data steps: {config.execution.cache_data_steps}")

    # Disable caching when targeting cloud queues — cached tasks from local runs
    # would be reused instead of dispatching to the cloud queue.
    cloud_queues = {"gpu-cloud"}
    if config.execution.gpu_queue in cloud_queues or config.execution.default_queue in cloud_queues:
        if config.execution.cache_data_steps:
            config.execution.cache_data_steps = False
            tl.report_text("Auto-disabled cache_data_steps for cloud queue target")

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
        orchestrator_mod = _import_module_directly(
            "dapidl.pipeline.orchestrator", _pipeline_src / "orchestrator.py"
        )
        PipelineOrchestrator = orchestrator_mod.PipelineOrchestrator

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
        unified_ctrl = _import_module_directly(
            "dapidl.pipeline.unified_controller", _pipeline_src / "unified_controller.py"
        )
        UnifiedPipelineController = unified_ctrl.UnifiedPipelineController

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
