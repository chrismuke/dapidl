#!/usr/bin/env python
"""Evaluate multi-scale ensemble using native-resolution patches.

This script evaluates each model on its OWN native-resolution dataset,
then combines predictions via soft voting. This is the correct approach
for multi-scale ensembles (as opposed to resizing from one resolution).

Approach 1: Evaluate each model on its native dataset independently
Approach 2: Combine predictions via soft voting for the same cells
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from dapidl.models import CellTypeClassifier
from dapidl.data.dataset import DAPIDLDataset, create_data_splits


def evaluate_model_on_native_dataset(
    model: CellTypeClassifier,
    dataset_path: str,
    device: str = "cuda",
    batch_size: int = 64,
    desc: str = "Evaluating",
) -> dict:
    """Evaluate a model on its native-resolution dataset."""
    model.eval()

    # Create data splits for this dataset
    _, _, test_dataset = create_data_splits(dataset_path, seed=42)

    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    all_probs = []
    all_preds = []
    all_labels = []
    all_cell_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
            patches, labels = batch
            patches = patches.to(device)

            logits = model(patches)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            # Get cell IDs from the dataset indices
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(labels)
            all_cell_ids.extend(range(start_idx, end_idx))

    all_probs = np.concatenate(all_probs, axis=0)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": np.array(all_preds),
        "probabilities": all_probs,
        "labels": np.array(all_labels),
        "cell_ids": np.array(all_cell_ids),
        "class_names": test_dataset.class_names,
    }


def soft_vote_ensemble(results_list: list[dict]) -> dict:
    """Combine predictions from multiple models via soft voting."""
    # Stack all probabilities: (num_models, num_samples, num_classes)
    all_probs = np.stack([r["probabilities"] for r in results_list], axis=0)

    # Average probabilities
    avg_probs = all_probs.mean(axis=0)

    # Get predictions from averaged probabilities
    ensemble_preds = avg_probs.argmax(axis=1)

    # Use labels from first result (should be same across all)
    labels = results_list[0]["labels"]

    accuracy = accuracy_score(labels, ensemble_preds)
    macro_f1 = f1_score(labels, ensemble_preds, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": ensemble_preds,
        "probabilities": avg_probs,
        "labels": labels,
        "class_names": results_list[0]["class_names"],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate multi-scale ensemble with native-resolution patches"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Model configs in format 'patch_size:model_path:dataset_path'",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Parse configs
    configs = []
    for spec in args.configs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid config format: {spec}. Expected 'patch_size:model_path:dataset_path'")
        patch_size, model_path, dataset_path = parts
        configs.append({
            "patch_size": int(patch_size),
            "model_path": Path(model_path),
            "dataset_path": Path(dataset_path),
        })

    configs = sorted(configs, key=lambda x: x["patch_size"])

    print("=" * 60)
    print("Multi-Scale Ensemble Evaluation (Native Resolution)")
    print("=" * 60)
    print(f"\nConfigurations:")
    for cfg in configs:
        print(f"  {cfg['patch_size']}x{cfg['patch_size']}: {cfg['model_path']}")
        print(f"       dataset: {cfg['dataset_path']}")
    print()

    # Load models and evaluate on native datasets
    results = {}
    for cfg in configs:
        patch_size = cfg["patch_size"]
        print(f"\n{'='*60}")
        print(f"Evaluating {patch_size}x{patch_size} model on native dataset")
        print("=" * 60)

        # Load model
        model = CellTypeClassifier.from_checkpoint(str(cfg["model_path"]))
        model = model.to(args.device)
        model.eval()

        # Evaluate on native dataset
        result = evaluate_model_on_native_dataset(
            model=model,
            dataset_path=str(cfg["dataset_path"]),
            device=args.device,
            batch_size=args.batch_size,
            desc=f"  {patch_size}x{patch_size}",
        )

        results[patch_size] = result
        print(f"\n  {patch_size}x{patch_size} Results:")
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    Macro F1: {result['macro_f1']:.4f}")

    # Print individual results summary
    print("\n" + "=" * 60)
    print("Individual Model Results Summary")
    print("=" * 60)
    print(f"\n{'Patch Size':<12} {'Accuracy':<12} {'Macro F1':<12}")
    print("-" * 36)
    for patch_size in sorted(results.keys()):
        r = results[patch_size]
        print(f"{patch_size:>6}x{patch_size:<5} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f}")

    # Combine via soft voting
    print("\n" + "=" * 60)
    print("Ensemble Results (Soft Voting)")
    print("=" * 60)

    results_list = [results[ps] for ps in sorted(results.keys())]
    ensemble_result = soft_vote_ensemble(results_list)

    print(f"\n  Ensemble Accuracy: {ensemble_result['accuracy']:.4f}")
    print(f"  Ensemble Macro F1: {ensemble_result['macro_f1']:.4f}")

    # Compare with best individual
    best_acc = max(r["accuracy"] for r in results.values())
    best_f1 = max(r["macro_f1"] for r in results.values())
    best_acc_size = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_f1_size = max(results.keys(), key=lambda k: results[k]["macro_f1"])

    print(f"\n  Best single (accuracy): {best_acc_size}x{best_acc_size} ({best_acc:.4f})")
    print(f"  Best single (F1): {best_f1_size}x{best_f1_size} ({best_f1:.4f})")
    print(f"\n  Improvement over best:")
    print(f"    Accuracy: {ensemble_result['accuracy'] - best_acc:+.4f}")
    print(f"    Macro F1: {ensemble_result['macro_f1'] - best_f1:+.4f}")

    # Print classification report for ensemble
    print("\n" + "=" * 60)
    print("Ensemble Classification Report")
    print("=" * 60)
    print(
        classification_report(
            ensemble_result["labels"],
            ensemble_result["predictions"],
            target_names=ensemble_result["class_names"],
        )
    )


if __name__ == "__main__":
    main()
