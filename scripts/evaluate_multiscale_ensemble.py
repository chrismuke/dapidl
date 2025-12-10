#!/usr/bin/env python
"""Evaluate multi-scale ensemble on test set.

This script loads models trained at different patch sizes and evaluates
them individually and as an ensemble on the test set.

Uses forward_single_scale method to resize the largest patches to each model's
expected input size. This means only one dataset (the largest patch size)
is needed.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dapidl.models import CellTypeClassifier, MultiScaleEnsemble
from dapidl.data.dataset import DAPIDLDataset, create_train_val_test_splits


def evaluate_single_model(
    model: CellTypeClassifier,
    dataloader: DataLoader,
    device: str = "cuda",
    desc: str = "Evaluating",
) -> dict:
    """Evaluate a single model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            patches, labels = batch
            patches = patches.to(device)
            labels = labels.numpy()

            logits = model(patches)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def evaluate_ensemble_single_scale(
    ensemble: MultiScaleEnsemble,
    dataloader: DataLoader,
    input_size: int,
    device: str = "cuda",
) -> dict:
    """Evaluate the ensemble using forward_single_scale (resize from one input size)."""
    ensemble.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating ensemble"):
            patches, labels = batch
            patches = patches.to(device)
            labels = labels.numpy()

            # Use forward_single_scale which resizes to each model's input size
            logits = ensemble.forward_single_scale(patches, target_size=input_size)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate multi-scale ensemble")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model checkpoints in format 'patch_size:path' (e.g., '64:model_64.pt')",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the largest patch size dataset (e.g., 256x256)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Input patch size of the dataset (default: 256)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fusion", type=str, default="soft", choices=["soft", "hard", "weighted"])
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Weights for weighted fusion",
    )
    args = parser.parse_args()

    # Parse models
    model_paths = {}
    for spec in args.models:
        patch_size, path = spec.split(":")
        model_paths[int(patch_size)] = Path(path)

    patch_sizes = sorted(model_paths.keys())
    print(f"Patch sizes: {patch_sizes}")
    print(f"Fusion method: {args.fusion}")
    print(f"Input size: {args.input_size}")

    # Load models
    print("\nLoading models...")
    models = []
    for patch_size in patch_sizes:
        print(f"  Loading {patch_size}x{patch_size} model from {model_paths[patch_size]}")
        model = CellTypeClassifier.from_checkpoint(str(model_paths[patch_size]))
        model = model.to(args.device)
        model.eval()
        models.append(model)

    # Create dataset and splits using the standard function (ensures consistent splits)
    print(f"\nLoading dataset from {args.dataset}...")
    train_dataset, val_dataset, test_dataset = create_train_val_test_splits(
        args.dataset, seed=42
    )
    print(f"Test samples: {len(test_dataset)}")
    class_names = test_dataset.class_names

    # Create dataloader for test set
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Evaluate individual models
    # Note: We need to resize patches to each model's expected input size
    print("\n" + "=" * 60)
    print("Individual Model Results")
    print("=" * 60)

    individual_results = {}
    for model, patch_size in zip(models, patch_sizes):
        print(f"\nEvaluating {patch_size}x{patch_size} model...")
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc=f"  {patch_size}x{patch_size}"):
                patches = patches.to(args.device)
                labels = labels.numpy()

                # Resize patches to model's expected input size
                if patch_size != args.input_size:
                    patches = torch.nn.functional.interpolate(
                        patches.float(),
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                logits = model(patches)
                preds = logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        individual_results[patch_size] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "predictions": all_preds,
            "labels": all_labels,
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")

    # Create and evaluate ensemble
    print("\n" + "=" * 60)
    print(f"Ensemble Results ({args.fusion} fusion)")
    print("=" * 60)

    ensemble = MultiScaleEnsemble(
        models=models,
        patch_sizes=patch_sizes,
        fusion_method=args.fusion,
        weights=args.weights,
    )
    ensemble = ensemble.to(args.device)

    ensemble_result = evaluate_ensemble_single_scale(
        ensemble, dataloader, args.input_size, args.device
    )
    print(f"\nEnsemble:")
    print(f"  Accuracy: {ensemble_result['accuracy']:.4f}")
    print(f"  Macro F1: {ensemble_result['macro_f1']:.4f}")

    # Compare with best individual model
    best_individual_acc = max(r["accuracy"] for r in individual_results.values())
    best_individual_f1 = max(r["macro_f1"] for r in individual_results.values())
    best_acc_size = max(individual_results.keys(), key=lambda k: individual_results[k]["accuracy"])
    best_f1_size = max(individual_results.keys(), key=lambda k: individual_results[k]["macro_f1"])

    print(f"\nBest single model (accuracy): {best_acc_size}x{best_acc_size} ({best_individual_acc:.4f})")
    print(f"Best single model (F1): {best_f1_size}x{best_f1_size} ({best_individual_f1:.4f})")
    print(f"\nImprovement over best single model:")
    print(f"  Accuracy: {ensemble_result['accuracy'] - best_individual_acc:+.4f}")
    print(f"  Macro F1: {ensemble_result['macro_f1'] - best_individual_f1:+.4f}")

    # Print detailed classification report
    print("\n" + "=" * 60)
    print("Ensemble Classification Report")
    print("=" * 60)
    print(
        classification_report(
            ensemble_result["labels"],
            ensemble_result["predictions"],
            target_names=class_names,
        )
    )


if __name__ == "__main__":
    main()
