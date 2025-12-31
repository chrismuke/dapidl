#!/usr/bin/env python3
"""Analyze DAPIDL HPO Results.

This script aggregates and analyzes results from HPO experiments,
generating comparison reports and best configuration recommendations.

Usage:
    # Analyze all HPO experiments
    uv run python scripts/analyze_hpo_results.py --project DAPIDL/HPO

    # Analyze specific controller task
    uv run python scripts/analyze_hpo_results.py --controller-task-id <ID>

    # Export results to CSV
    uv run python scripts/analyze_hpo_results.py --project DAPIDL/HPO --output results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_experiments_from_project(project_name: str, status: str = "completed") -> list[dict]:
    """Get all experiments from a ClearML project."""
    from clearml import Task

    tasks = Task.get_tasks(
        project_name=project_name,
        task_filter={"status": [status]} if status else None,
    )

    experiments = []
    for task in tasks:
        if task.task_type != Task.TaskTypes.training:
            continue

        try:
            params = task.get_parameters_as_dict()
            metrics = task.get_last_scalar_metrics()

            exp = {
                "task_id": task.id,
                "task_name": task.name,
                "status": task.status,
                "created": str(task.data.created),
            }

            # Extract General parameters
            general = params.get("General", {})
            for key, value in general.items():
                exp[f"param_{key}"] = value

            # Extract metrics
            if metrics:
                for title, series_dict in metrics.items():
                    for series, value in series_dict.items():
                        metric_name = f"{title}_{series}".replace(" ", "_")
                        if isinstance(value, dict):
                            exp[f"metric_{metric_name}"] = value.get("last", value.get("value"))
                        else:
                            exp[f"metric_{metric_name}"] = value

            experiments.append(exp)

        except Exception as e:
            logger.warning(f"Failed to process task {task.id}: {e}")

    return experiments


def get_experiments_from_controller(controller_task_id: str) -> list[dict]:
    """Get all experiments spawned by an HPO controller task."""
    from clearml import Task
    from clearml.automation import HyperParameterOptimizer

    controller = Task.get_task(task_id=controller_task_id)

    # Get top experiments from optimizer
    try:
        optimizer = HyperParameterOptimizer.get(controller_task_id)
        top_experiments = optimizer.get_top_experiments(top_k=100)

        experiments = []
        for exp_info in top_experiments:
            task = Task.get_task(task_id=exp_info.id)
            params = task.get_parameters_as_dict()

            exp = {
                "task_id": exp_info.id,
                "objective_value": exp_info.objective_value,
                "objective_iteration": exp_info.objective_iteration,
            }

            # Add parameters
            general = params.get("General", {})
            for key, value in general.items():
                exp[f"param_{key}"] = value

            experiments.append(exp)

        return experiments

    except Exception as e:
        logger.warning(f"Could not get experiments from optimizer: {e}")
        # Fallback: get from project
        project_name = controller.get_project_name()
        return get_experiments_from_project(project_name)


def analyze_results(experiments: list[dict]) -> dict[str, Any]:
    """Analyze HPO results and generate insights."""
    if not experiments:
        return {"error": "No experiments to analyze"}

    df = pd.DataFrame(experiments)

    # Find objective column
    obj_col = None
    for col in df.columns:
        if "objective" in col.lower() or "macro_f1" in col.lower():
            obj_col = col
            break

    if obj_col is None:
        # Try to find best metric column
        metric_cols = [c for c in df.columns if c.startswith("metric_")]
        if metric_cols:
            obj_col = metric_cols[0]

    analysis = {
        "total_experiments": len(df),
        "columns": list(df.columns),
    }

    if obj_col and obj_col in df.columns:
        # Best experiment
        valid_df = df[df[obj_col].notna()]
        if len(valid_df) > 0:
            best_idx = valid_df[obj_col].idxmax()
            best_exp = valid_df.loc[best_idx].to_dict()
            analysis["best_experiment"] = best_exp
            analysis["best_objective"] = best_exp.get(obj_col)

            # Parameter importance (simple correlation)
            param_cols = [c for c in df.columns if c.startswith("param_")]
            param_importance = {}

            for col in param_cols:
                try:
                    # For numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        corr = df[col].corr(df[obj_col])
                        if not pd.isna(corr):
                            param_importance[col] = abs(corr)
                except Exception:
                    pass

            analysis["param_importance"] = dict(
                sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            )

            # Best value per parameter
            best_per_param = {}
            for col in param_cols:
                grouped = valid_df.groupby(col)[obj_col].mean()
                if len(grouped) > 0:
                    best_val = grouped.idxmax()
                    best_per_param[col] = {
                        "best_value": best_val,
                        "mean_objective": float(grouped[best_val]),
                    }
            analysis["best_per_param"] = best_per_param

    return analysis


def print_analysis(analysis: dict[str, Any]) -> None:
    """Print analysis results in a readable format."""
    print("\n" + "=" * 70)
    print("HPO ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\nTotal experiments analyzed: {analysis.get('total_experiments', 0)}")

    if "best_experiment" in analysis:
        print("\n" + "-" * 70)
        print("BEST EXPERIMENT")
        print("-" * 70)

        best = analysis["best_experiment"]
        print(f"Task ID: {best.get('task_id', 'N/A')}")
        print(f"Objective: {analysis.get('best_objective', 'N/A'):.4f}")

        print("\nParameters:")
        for key, value in best.items():
            if key.startswith("param_"):
                param_name = key.replace("param_", "")
                print(f"  {param_name}: {value}")

    if "param_importance" in analysis and analysis["param_importance"]:
        print("\n" + "-" * 70)
        print("PARAMETER IMPORTANCE (by correlation with objective)")
        print("-" * 70)

        for param, importance in list(analysis["param_importance"].items())[:10]:
            param_name = param.replace("param_", "")
            print(f"  {param_name}: {importance:.4f}")

    if "best_per_param" in analysis:
        print("\n" + "-" * 70)
        print("BEST VALUE PER PARAMETER")
        print("-" * 70)

        for param, info in analysis["best_per_param"].items():
            param_name = param.replace("param_", "")
            print(f"  {param_name}:")
            print(f"    Best value: {info['best_value']}")
            print(f"    Mean objective: {info['mean_objective']:.4f}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze DAPIDL HPO Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--project",
        type=str,
        default="DAPIDL/HPO",
        help="ClearML project name to analyze",
    )

    parser.add_argument(
        "--controller-task-id",
        type=str,
        default=None,
        help="Specific HPO controller task ID to analyze",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file path for analysis",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top experiments to show",
    )

    args = parser.parse_args()

    logger.info("Fetching HPO experiments...")

    if args.controller_task_id:
        experiments = get_experiments_from_controller(args.controller_task_id)
    else:
        experiments = get_experiments_from_project(args.project)

    logger.info(f"Found {len(experiments)} experiments")

    if not experiments:
        logger.error("No experiments found!")
        return

    # Analyze results
    analysis = analyze_results(experiments)
    print_analysis(analysis)

    # Export to CSV if requested
    if args.output:
        df = pd.DataFrame(experiments)
        df.to_csv(args.output, index=False)
        logger.info(f"Results exported to {args.output}")

    # Export analysis to JSON if requested
    if args.output_json:
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            if hasattr(obj, "item"):  # numpy types
                return obj.item()
            return obj

        analysis_json = json.loads(
            json.dumps(analysis, default=convert)
        )

        with open(args.output_json, "w") as f:
            json.dump(analysis_json, f, indent=2)
        logger.info(f"Analysis exported to {args.output_json}")

    # Show top experiments
    print(f"\n{'=' * 70}")
    print(f"TOP {args.top_k} EXPERIMENTS")
    print("=" * 70)

    df = pd.DataFrame(experiments)

    # Find objective column
    obj_col = None
    for col in df.columns:
        if "objective" in col.lower():
            obj_col = col
            break
        if "macro_f1" in col.lower():
            obj_col = col
            break

    if obj_col and obj_col in df.columns:
        top_df = df.nlargest(args.top_k, obj_col)

        for i, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"\n#{i}: {row.get('task_id', 'N/A')}")
            print(f"   Objective: {row.get(obj_col, 'N/A'):.4f}")

            # Show key parameters
            param_cols = [c for c in row.index if c.startswith("param_")]
            for col in param_cols[:5]:  # Show first 5 params
                param_name = col.replace("param_", "")
                print(f"   {param_name}: {row[col]}")


if __name__ == "__main__":
    main()
