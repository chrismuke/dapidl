#!/usr/bin/env python3
"""Run DAPIDL Hyperparameter Optimization.

This script launches HPO experiments on ClearML using Optuna-based
Bayesian optimization.

Usage:
    # First, create a template task
    uv run python scripts/hpo_train_template.py --create-template

    # Then run HPO with the template task ID
    uv run python scripts/run_hpo.py --template-task-id <TASK_ID>

    # Run HPO for coarse classification only
    uv run python scripts/run_hpo.py --template-task-id <ID> --granularity coarse

    # Run HPO for finegrained with specific centering
    uv run python scripts/run_hpo.py --template-task-id <ID> --granularity finegrained --centering cellpose

    # Run locally for testing
    uv run python scripts/run_hpo.py --template-task-id <ID> --local --max-jobs 5
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DAPIDL Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--template-task-id",
        type=str,
        required=True,
        help="ClearML task ID to use as template (create with hpo_train_template.py --create-template)",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="DAPIDL/HPO",
        help="ClearML project name for HPO experiments",
    )

    parser.add_argument(
        "--task-name",
        type=str,
        default="DAPIDL-HPO",
        help="HPO controller task name",
    )

    parser.add_argument(
        "--max-jobs",
        type=int,
        default=100,
        help="Total number of trials to run",
    )

    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Maximum concurrent trials",
    )

    parser.add_argument(
        "--queue",
        type=str,
        default="gpu",
        help="ClearML queue for worker execution",
    )

    parser.add_argument(
        "--granularity",
        type=str,
        choices=["coarse", "finegrained"],
        default=None,
        help="Filter HPO to specific granularity (optional)",
    )

    parser.add_argument(
        "--centering",
        type=str,
        choices=["xenium", "cellpose"],
        default=None,
        help="Filter HPO to specific centering method (optional)",
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on ClearML agents",
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="Time limit per trial in minutes",
    )

    parser.add_argument(
        "--min-epochs",
        type=int,
        default=10,
        help="Minimum epochs before pruning can occur",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum epochs per trial",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DAPIDL Hyperparameter Optimization")
    logger.info("=" * 60)

    logger.info(f"Template task ID: {args.template_task_id}")
    logger.info(f"Project: {args.project_name}")
    logger.info(f"Max jobs: {args.max_jobs}")
    logger.info(f"Concurrent: {args.concurrent}")
    logger.info(f"Granularity filter: {args.granularity or 'all'}")
    logger.info(f"Centering filter: {args.centering or 'all'}")
    logger.info(f"Local execution: {args.local}")

    # Import HPO module
    try:
        from dapidl.hpo.optimizer import run_hpo, get_top_experiments
    except ImportError as e:
        logger.error(f"Failed to import HPO module: {e}")
        logger.error("Make sure dapidl is installed: uv sync")
        sys.exit(1)

    # Build task name with filters
    task_name = args.task_name
    if args.granularity:
        task_name = f"{task_name}-{args.granularity}"
    if args.centering:
        task_name = f"{task_name}-{args.centering}"

    logger.info(f"\nStarting HPO: {task_name}")

    # Run HPO
    task, optimizer = run_hpo(
        base_task_id=args.template_task_id,
        project_name=args.project_name,
        task_name=task_name,
        max_concurrent_tasks=args.concurrent,
        total_max_jobs=args.max_jobs,
        execution_queue=args.queue,
        granularity=args.granularity,
        centering=args.centering,
        local=args.local,
    )

    logger.info(f"\nHPO Controller Task ID: {task.id}")
    logger.info("HPO is now running...")

    if args.local:
        # Wait for local execution to complete
        logger.info("Running locally - waiting for completion...")
        optimizer.wait()

        # Get top experiments
        logger.info("\n" + "=" * 60)
        logger.info("TOP EXPERIMENTS")
        logger.info("=" * 60)

        top_experiments = get_top_experiments(optimizer, top_k=10)
        for i, exp in enumerate(top_experiments, 1):
            logger.info(f"\n#{i}: Task {exp['task_id']}")
            logger.info(f"   Objective: {exp['objective_value']:.4f}")
            logger.info(f"   Params: {exp['params']}")
    else:
        logger.info("\nRunning on ClearML agents.")
        logger.info("Monitor progress at: https://app.clear.ml")
        logger.info(f"Project: {args.project_name}")
        logger.info(f"Task: {task_name}")

        # Optionally wait for completion
        print("\nPress Ctrl+C to exit (HPO will continue running)")
        try:
            optimizer.wait()
        except KeyboardInterrupt:
            logger.info("\nExiting. HPO will continue running on ClearML.")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
