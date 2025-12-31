"""ClearML HyperParameterOptimizer wrapper for DAPIDL.

Provides Optuna-based Bayesian optimization with early stopping and pruning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

from .search_space import get_clearml_hyper_parameters

if TYPE_CHECKING:
    from clearml.automation import Objective

logger = logging.getLogger(__name__)


def create_hpo_optimizer(
    base_task_id: str,
    project_name: str = "DAPIDL/HPO",
    max_concurrent_tasks: int = 4,
    total_max_jobs: int = 100,
    execution_queue: str = "gpu",
    time_limit_per_job_minutes: int = 120,
    min_iteration_per_job: int = 10,
    max_iteration_per_job: int = 50,
    objective_metric_title: str = "val",
    objective_metric_series: str = "macro_f1",
    granularity_filter: str | None = None,
    centering_filter: str | None = None,
) -> HyperParameterOptimizer:
    """Create HPO optimizer for DAPIDL training.

    Args:
        base_task_id: ClearML task ID to use as template
        project_name: ClearML project name for HPO trials
        max_concurrent_tasks: Max parallel tasks
        total_max_jobs: Total number of trials to run
        execution_queue: ClearML queue name for workers
        time_limit_per_job_minutes: Max time per trial in minutes
        min_iteration_per_job: Min epochs before pruning can occur
        max_iteration_per_job: Max epochs per trial
        objective_metric_title: Metric title for optimization (e.g., 'val')
        objective_metric_series: Metric series for optimization (e.g., 'macro_f1')
        granularity_filter: Optional filter to restrict to 'coarse' or 'finegrained'
        centering_filter: Optional filter to restrict to 'xenium' or 'cellpose'

    Returns:
        Configured HyperParameterOptimizer
    """
    # Get hyperparameter definitions
    hyper_parameters = get_clearml_hyper_parameters()

    # Apply filters if specified
    if granularity_filter:
        hyper_parameters = [
            p if p.name != "General/granularity"
            else type(p)(p.name, values=[granularity_filter])
            for p in hyper_parameters
        ]

    if centering_filter:
        hyper_parameters = [
            p if p.name != "General/centering"
            else type(p)(p.name, values=[centering_filter])
            for p in hyper_parameters
        ]

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=hyper_parameters,

        # Optimization objective
        objective_metric_title=objective_metric_title,
        objective_metric_series=objective_metric_series,
        objective_metric_sign="max",  # Maximize F1

        # Execution settings
        max_number_of_concurrent_tasks=max_concurrent_tasks,
        optimizer_class=OptimizerOptuna,
        execution_queue=execution_queue,

        # Budget
        total_max_jobs=total_max_jobs,
        time_limit_per_job=time_limit_per_job_minutes * 60,  # Convert to seconds
        pool_period_min=0.5,  # Check every 30 seconds

        # Early stopping / Pruning
        min_iteration_per_job=min_iteration_per_job,
        max_iteration_per_job=max_iteration_per_job,

        # Optuna-specific settings
        optimizer_class_kwargs={
            "optuna_sampler": "TPESampler",  # Bayesian optimization
            "optuna_pruner": "MedianPruner",
            "optuna_pruner_kwargs": {
                "n_startup_trials": 5,
                "n_warmup_steps": 10,
                "interval_steps": 1,
            },
        },

        # Save top experiments
        save_top_k_tasks_only=10,
    )

    return optimizer


def run_hpo(
    base_task_id: str,
    project_name: str = "DAPIDL/HPO",
    task_name: str = "DAPIDL-HPO",
    max_concurrent_tasks: int = 4,
    total_max_jobs: int = 100,
    execution_queue: str = "gpu",
    granularity: str | None = None,
    centering: str | None = None,
    local: bool = False,
) -> tuple[Task, HyperParameterOptimizer]:
    """Run HPO optimization.

    Args:
        base_task_id: Template task ID
        project_name: ClearML project name
        task_name: HPO controller task name
        max_concurrent_tasks: Max parallel trials
        total_max_jobs: Total trials to run
        execution_queue: ClearML queue for workers
        granularity: Filter to 'coarse' or 'finegrained'
        centering: Filter to 'xenium' or 'cellpose'
        local: Run locally instead of on ClearML agents

    Returns:
        Tuple of (controller_task, optimizer)
    """
    # Create HPO controller task
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    # Build task name with filters
    if granularity:
        task_name = f"{task_name}-{granularity}"
    if centering:
        task_name = f"{task_name}-{centering}"
    task.set_name(task_name)

    # Create optimizer
    optimizer = create_hpo_optimizer(
        base_task_id=base_task_id,
        project_name=project_name,
        max_concurrent_tasks=max_concurrent_tasks,
        total_max_jobs=total_max_jobs,
        execution_queue=execution_queue if not local else None,
        granularity_filter=granularity,
        centering_filter=centering,
    )

    # Log configuration
    config = {
        "base_task_id": base_task_id,
        "total_max_jobs": total_max_jobs,
        "max_concurrent_tasks": max_concurrent_tasks,
        "granularity_filter": granularity,
        "centering_filter": centering,
        "local": local,
    }
    task.connect(config, name="HPO Config")

    logger.info(f"Starting HPO with {total_max_jobs} trials, {max_concurrent_tasks} concurrent")

    # Start optimization
    optimizer.set_report_period(0.5)  # Report every 30 seconds

    if local:
        # Local execution (sequential)
        optimizer.start_locally(job_complete_callback=_job_complete_callback)
    else:
        # Remote execution on ClearML agents
        optimizer.start()

    return task, optimizer


def _job_complete_callback(
    job_id: str,
    objective_value: float,
    objective_iteration: int,
    job_parameters: dict,
    top_performance_job_id: str,
) -> bool:
    """Callback when a trial completes.

    Returns:
        True to continue optimization, False to stop
    """
    logger.info(
        f"Trial {job_id} complete: F1={objective_value:.4f} at epoch {objective_iteration}"
    )
    logger.info(f"Parameters: {job_parameters}")
    logger.info(f"Current best: {top_performance_job_id}")
    return True  # Continue optimization


def get_top_experiments(
    optimizer: HyperParameterOptimizer,
    top_k: int = 10,
) -> list[dict]:
    """Get top performing experiments from HPO.

    Args:
        optimizer: The HPO optimizer
        top_k: Number of top experiments to return

    Returns:
        List of experiment info dicts
    """
    top_tasks = optimizer.get_top_experiments(top_k=top_k)

    results = []
    for task_info in top_tasks:
        task = Task.get_task(task_id=task_info.id)
        params = task.get_parameters_as_dict()

        results.append({
            "task_id": task_info.id,
            "objective_value": task_info.objective_value,
            "params": params.get("General", {}),
        })

    return results
