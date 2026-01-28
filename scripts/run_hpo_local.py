#!/usr/bin/env python3
"""Pure Optuna HPO for DAPIDL - runs training in-process without ClearML task cloning.

This is the local-only version that directly executes training within the same process,
giving it full access to local filesystem paths for datasets.
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import optuna
import torch
from clearml import Task
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dapidl.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Dataset paths - local filesystem
DATASET_BASE = Path("/home/chrism/datasets/derived")

# Search space definition
SEARCH_SPACE = {
    "backbone": ["efficientnetv2_rw_s", "resnet18", "resnet50", "convnext_tiny"],
    "patch_size": [32, 64, 128, 256],
    "batch_size": [32, 64, 128, 256],
    "learning_rate_log": (-5, -3),  # Log scale: 1e-5 to 1e-3
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "max_weight_ratio": [5.0, 10.0, 15.0, 20.0],
}

# Memory constraints for RTX 3090 (24GB)
BACKBONE_MAX_BATCH = {
    "efficientnetv2_rw_s": 128,
    "convnext_tiny": 128,
    "resnet50": 256,
    "resnet18": 512,
}


def get_dataset_path(centering: str, granularity: str, patch_size: int) -> Path:
    """Construct dataset path from parameters."""
    dataset_name = f"xenium-breast-{centering}-{granularity}-p{patch_size}"
    return DATASET_BASE / dataset_name


def check_dataset_exists(centering: str, granularity: str, patch_size: int) -> bool:
    """Check if dataset exists."""
    path = get_dataset_path(centering, granularity, patch_size)
    lmdb_path = path / "patches.lmdb"
    return lmdb_path.exists()


def get_available_datasets(granularity: str) -> list[tuple[str, int]]:
    """Get list of available (centering, patch_size) combinations for a granularity."""
    available = []
    for centering in ["xenium", "cellpose"]:
        for patch_size in [32, 64, 128, 256]:
            if check_dataset_exists(centering, granularity, patch_size):
                available.append((centering, patch_size))
    return available


def run_training(
    backbone: str,
    patch_size: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    max_weight_ratio: float,
    centering: str,
    granularity: str,
    max_epochs: int = 30,
    trial_number: int = 0,
) -> dict[str, Any]:
    """Run a single training run with specified hyperparameters.

    Returns dict with best_val_f1, etc.
    """
    dataset_path = get_dataset_path(centering, granularity, patch_size)

    if not (dataset_path / "patches.lmdb").exists():
        raise ValueError(f"Dataset not found: {dataset_path}")

    # Create temporary output directory
    output_dir = Path(tempfile.mkdtemp(prefix=f"hpo_trial_{trial_number}_"))

    try:
        # Create trainer with W&B disabled to avoid clutter
        trainer = Trainer(
            data_path=dataset_path,
            output_path=output_dir,
            backbone_name=backbone,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout=dropout,
            max_weight_ratio=max_weight_ratio,
            epochs=max_epochs,
            use_wandb=False,  # Don't log each trial to W&B
            early_stopping_patience=10,
            backend="dali-lmdb",  # Use DALI with LMDB for speed
        )

        # Run training
        metrics = trainer.train()

        # Get best F1 achieved
        best_val_f1 = trainer.best_val_f1

        return {
            "best_val_f1": best_val_f1,
            "final_metrics": metrics,
        }

    finally:
        # Cleanup: remove temp directory and free GPU memory
        shutil.rmtree(output_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        gc.collect()


def create_objective(
    centering: str | None,
    granularity: str,
    max_epochs: int = 30,
    clearml_logger=None,
):
    """Create Optuna objective function for given dataset configuration.

    If centering is None, the optimizer will also search over centering methods.
    """
    available_datasets = get_available_datasets(granularity)
    if not available_datasets:
        raise ValueError(f"No datasets found for granularity={granularity}")

    logger.info(f"Available datasets for {granularity}: {available_datasets}")

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        backbone = trial.suggest_categorical("backbone", SEARCH_SPACE["backbone"])

        # Sample centering if not fixed
        if centering is None:
            available_centerings = list(set(c for c, _ in available_datasets))
            selected_centering = trial.suggest_categorical("centering", available_centerings)
        else:
            selected_centering = centering

        # Get patch sizes available for this centering
        available_patch_sizes = [p for c, p in available_datasets if c == selected_centering]
        patch_size = trial.suggest_categorical("patch_size", available_patch_sizes)

        # Sample other params
        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
        learning_rate = trial.suggest_float(
            "learning_rate",
            10 ** SEARCH_SPACE["learning_rate_log"][0],
            10 ** SEARCH_SPACE["learning_rate_log"][1],
            log=True,
        )
        dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
        max_weight_ratio = trial.suggest_categorical(
            "max_weight_ratio", SEARCH_SPACE["max_weight_ratio"]
        )

        # Adjust batch size for backbone memory requirements
        max_batch = BACKBONE_MAX_BATCH.get(backbone, 128)
        if batch_size > max_batch:
            batch_size = max_batch
            trial.set_user_attr("adjusted_batch_size", batch_size)

        logger.info(
            f"Trial {trial.number}: {backbone}, {selected_centering}/p{patch_size}, "
            f"bs{batch_size}, lr={learning_rate:.2e}, dropout={dropout}, mwr={max_weight_ratio}"
        )

        try:
            results = run_training(
                backbone=backbone,
                patch_size=patch_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                dropout=dropout,
                max_weight_ratio=max_weight_ratio,
                centering=selected_centering,
                granularity=granularity,
                max_epochs=max_epochs,
                trial_number=trial.number,
            )

            val_f1 = results["best_val_f1"]
            logger.info(f"Trial {trial.number} complete: val_f1={val_f1:.4f}")

            # Log to ClearML if available
            if clearml_logger:
                clearml_logger.report_scalar(
                    title="HPO Trials",
                    series="val_f1",
                    value=val_f1,
                    iteration=trial.number,
                )

            return val_f1

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Trial {trial.number} OOM: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.TrialPruned()

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.TrialPruned()

    return objective


def run_hpo(
    centering: str | None = None,
    granularity: str = "coarse",
    n_trials: int = 50,
    n_startup_trials: int = 5,
    max_epochs: int = 30,
    storage: str | None = None,
    study_name: str | None = None,
    clearml_logger=None,
) -> optuna.Study:
    """Run hyperparameter optimization.

    Args:
        centering: 'xenium' or 'cellpose', or None to search both
        granularity: 'coarse' or 'finegrained'
        n_trials: Number of trials to run
        n_startup_trials: Number of random trials before TPE kicks in
        max_epochs: Max epochs per trial
        storage: Optional SQLite path for persistent storage
        study_name: Study name for persistent storage
        clearml_logger: ClearML logger for tracking

    Returns:
        Completed Optuna study
    """
    centering_str = centering or "all"
    study_name = study_name or f"dapidl-hpo-{centering_str}-{granularity}"

    # Create sampler and pruner
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,  # Model parameter correlations
    )
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=10,  # Don't prune before epoch 10
        interval_steps=1,
    )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize F1
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # Create objective
    objective = create_objective(
        centering=centering,
        granularity=granularity,
        max_epochs=max_epochs,
        clearml_logger=clearml_logger,
    )

    logger.info(f"Starting HPO: {study_name}")
    logger.info(f"  Centering: {centering_str}")
    logger.info(f"  Granularity: {granularity}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Max epochs per trial: {max_epochs}")

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(Exception,),
        show_progress_bar=True,
    )

    return study


def print_results(study: optuna.Study):
    """Print HPO results summary."""
    print("\n" + "=" * 70)
    print("HPO RESULTS")
    print("=" * 70)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\nNo successful trials!")
        return

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best value (val_f1): {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        if key == "learning_rate":
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")

    print(f"\nTotal trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"  Completed: {len(completed)}")
    print(f"  Pruned: {len(pruned)}")
    print(f"  Failed: {len(failed)}")

    if len(completed) >= 5:
        print("\nTop 5 trials:")
        top_trials = sorted(completed, key=lambda t: t.value or 0, reverse=True)[:5]
        for i, t in enumerate(top_trials, 1):
            print(f"  {i}. Trial #{t.number}: val_f1={t.value:.4f}")
            print(f"     backbone={t.params.get('backbone', 'N/A')}, "
                  f"patch_size={t.params.get('patch_size', 'N/A')}, "
                  f"lr={t.params.get('learning_rate', 0):.2e}")

    print("=" * 70)


def save_results(study: optuna.Study, output_path: Path):
    """Save study results to JSON."""
    results = {
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else {},
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "params": t.params,
            }
            for t in study.trials
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run pure Optuna HPO for DAPIDL (local in-process execution)"
    )
    parser.add_argument(
        "--centering",
        choices=["xenium", "cellpose", "all"],
        default="all",
        help="Nucleus centering method (or 'all' to search both)",
    )
    parser.add_argument(
        "--granularity",
        choices=["coarse", "finegrained"],
        default="coarse",
        help="Classification granularity",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max epochs per trial",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="SQLite storage path for persistence (e.g., sqlite:///hpo.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--clearml-project",
        type=str,
        default="DAPIDL/HPO",
        help="ClearML project for logging results",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Disable ClearML logging",
    )
    args = parser.parse_args()

    # Handle centering
    centering = None if args.centering == "all" else args.centering

    # Check GPU memory before starting
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    # Initialize ClearML task for logging overall results
    clearml_logger = None
    task = None
    if not args.no_clearml:
        try:
            task = Task.init(
                project_name=args.clearml_project,
                task_name=f"HPO-Local-{args.granularity}-{args.centering}",
                task_type=Task.TaskTypes.optimizer,
            )
            task.connect({
                "centering": args.centering,
                "granularity": args.granularity,
                "n_trials": args.n_trials,
                "max_epochs": args.max_epochs,
            })
            clearml_logger = task.get_logger()
        except Exception as e:
            logger.warning(f"ClearML init failed: {e}")

    # Run HPO
    try:
        study = run_hpo(
            centering=centering,
            granularity=args.granularity,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            storage=args.storage,
            clearml_logger=clearml_logger,
        )

        # Print results
        print_results(study)

        # Save results
        if args.output:
            save_results(study, Path(args.output))
        else:
            # Default output path
            output_path = Path(f"hpo_results_{args.granularity}_{args.centering}.json")
            save_results(study, output_path)

        # Log best params to ClearML
        if task and study.best_trial:
            task.get_logger().report_single_value("best_val_f1", study.best_value)
            for key, value in study.best_params.items():
                if isinstance(value, float):
                    task.get_logger().report_single_value(f"best_{key}", value)
                else:
                    task.set_parameter(f"best/{key}", value)

    finally:
        if task:
            task.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
