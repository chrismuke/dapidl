"""Cross-Platform Transfer Testing Step.

Tests a model trained on one platform (e.g., Xenium) against data
from another platform (e.g., MERSCOPE) to measure transfer learning success.

Key features:
- Physical-space normalization for fair comparison
- Automatic platform detection
- Per-class and aggregate metrics
- Comparison to same-platform baseline (if available)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class CrossPlatformTransferConfig:
    """Configuration for cross-platform transfer testing."""

    # Model to test
    model_path: str = ""

    # Target platform data
    target_dataset_id: str = ""  # ClearML dataset ID
    target_local_path: str = ""  # Or local path
    target_platform: str = "auto"  # "xenium", "merscope", or "auto"

    # Source platform (for logging)
    source_platform: str = "unknown"

    # Physical normalization settings
    patch_size: int = 128
    normalize_physical_size: bool = True
    target_pixel_size_um: float = 0.2125  # Xenium default (reference scale)

    # Testing settings
    batch_size: int = 64
    max_samples: int = 0  # 0 = all samples
    confidence_threshold: float = 0.0  # Minimum prediction confidence

    # Class mapping
    class_mapping: dict[str, int] = field(default_factory=dict)

    # Comparison baseline
    baseline_metrics_path: str = ""  # Path to same-platform metrics JSON

    # Output
    output_dir: str = ""


class CrossPlatformTransferStep(PipelineStep):
    """Test model transfer between spatial transcriptomics platforms.

    This step measures how well a model trained on one platform generalizes
    to another platform. Key considerations:
    - Pixel size differences (Xenium: 0.2125 µm, MERSCOPE: 0.108 µm)
    - Intensity normalization differences
    - Cell type distribution shifts

    Usage:
        config = CrossPlatformTransferConfig(
            model_path="/path/to/xenium_model.pt",
            target_dataset_id="merscope-dataset-id",
            source_platform="xenium",
        )
        step = CrossPlatformTransferStep(config)
        results = step.execute(artifacts)
    """

    name = "cross_platform_transfer"
    description = "Test model transfer between platforms"

    # Platform pixel sizes in microns
    PIXEL_SIZES = {
        "xenium": 0.2125,
        "merscope": 0.108,
        "cosmx": 0.18,
    }

    def __init__(self, config: CrossPlatformTransferConfig | None = None):
        self.config = config or CrossPlatformTransferConfig()

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to trained model (.pt file)",
                },
                "target_dataset_id": {
                    "type": "string",
                    "description": "ClearML Dataset ID for target platform data",
                },
                "target_local_path": {
                    "type": "string",
                    "description": "Local path to target platform data (alternative to dataset_id)",
                },
                "target_platform": {
                    "type": "string",
                    "enum": ["auto", "xenium", "merscope", "cosmx"],
                    "default": "auto",
                    "description": "Target platform type",
                },
                "source_platform": {
                    "type": "string",
                    "default": "unknown",
                    "description": "Platform the model was trained on",
                },
                "patch_size": {
                    "type": "integer",
                    "default": 128,
                    "description": "Patch size in pixels",
                },
                "normalize_physical_size": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply physical-space normalization",
                },
                "batch_size": {
                    "type": "integer",
                    "default": 64,
                    "description": "Batch size for inference",
                },
                "max_samples": {
                    "type": "integer",
                    "default": 0,
                    "description": "Max samples to test (0 = all)",
                },
            },
            "required": ["model_path"],
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate required inputs."""
        cfg = self.config

        # Need either model_path in config or from artifacts
        model_path = cfg.model_path or artifacts.get_input("model_path")
        if not model_path:
            raise ValueError("model_path required (config or artifacts)")

        # Need target data from config or artifacts
        has_target = (
            cfg.target_dataset_id
            or cfg.target_local_path
            or artifacts.get_input("target_data_path")
        )
        if not has_target:
            raise ValueError("Target data required (dataset_id, local_path, or artifact)")

        return True

    def execute(self, artifacts: StepArtifacts, **params: Any) -> StepArtifacts:
        """Execute cross-platform transfer testing."""
        cfg = self.config

        # Resolve model path
        model_path = resolve_artifact_path(
            cfg.model_path or artifacts.get_input("model_path"),
            "model_path"
        )
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")

        # Resolve target data path
        target_path = self._resolve_target_data(artifacts)
        logger.info(f"Target data path: {target_path}")

        # Detect target platform
        target_platform = self._detect_platform(target_path, cfg.target_platform)
        logger.info(f"Target platform: {target_platform}")
        logger.info(f"Source platform: {cfg.source_platform}")

        # Get pixel size for target platform
        target_pixel_size = self.PIXEL_SIZES.get(target_platform, 0.2125)
        logger.info(f"Target pixel size: {target_pixel_size} µm")

        # Load model
        model, class_names = self._load_model(model_path)
        num_classes = len(class_names)
        logger.info(f"Model classes: {class_names}")

        # Load and prepare target data
        patches, labels, cell_ids = self._load_target_data(
            target_path,
            target_platform,
            target_pixel_size,
            class_names,
        )

        if len(patches) == 0:
            logger.warning("No valid patches found in target data")
            artifacts.set_output("transfer_metrics", {"error": "No valid patches"})
            return artifacts

        logger.info(f"Loaded {len(patches)} patches for testing")

        # Run inference
        predictions, confidences = self._run_inference(model, patches)

        # Calculate metrics
        metrics = self._calculate_metrics(
            labels,
            predictions,
            confidences,
            class_names,
        )

        # Add transfer-specific metrics
        metrics["source_platform"] = cfg.source_platform
        metrics["target_platform"] = target_platform
        metrics["physical_normalized"] = cfg.normalize_physical_size
        metrics["num_samples"] = len(patches)

        # Compare to baseline if available
        if cfg.baseline_metrics_path:
            baseline = self._load_baseline(cfg.baseline_metrics_path)
            metrics["baseline_comparison"] = self._compare_to_baseline(metrics, baseline)

        # Log results
        self._log_results(metrics, class_names)

        # Save results
        output_dir = Path(cfg.output_dir) if cfg.output_dir else target_path.parent / "transfer_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "transfer_metrics.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to: {results_path}")

        # Set outputs
        artifacts.set_output("transfer_metrics", metrics)
        artifacts.set_output("transfer_results_path", str(results_path))
        artifacts.set_output("source_platform", cfg.source_platform)
        artifacts.set_output("target_platform", target_platform)

        return artifacts

    def _resolve_target_data(self, artifacts: StepArtifacts) -> Path:
        """Resolve target data path from config or artifacts."""
        cfg = self.config

        if cfg.target_local_path:
            return Path(cfg.target_local_path)

        if cfg.target_dataset_id:
            from clearml import Dataset
            dataset = Dataset.get(dataset_id=cfg.target_dataset_id)
            return Path(dataset.get_local_copy())

        # From artifacts
        target_path = resolve_artifact_path(
            artifacts.get_input("target_data_path"),
            "target_data_path"
        )
        if target_path:
            return target_path

        raise ValueError("No target data path available")

    def _detect_platform(self, data_path: Path, hint: str) -> str:
        """Detect platform from data structure."""
        if hint != "auto":
            return hint

        # Check for platform-specific files
        if (data_path / "morphology_focus.ome.tif").exists():
            return "xenium"
        if (data_path / "images" / "mosaic_DAPI_z0.tif").exists():
            return "merscope"
        if (data_path / "cell_metadata.csv").exists():
            return "merscope"
        if (data_path / "cells.parquet").exists():
            return "xenium"

        logger.warning("Could not detect platform, assuming xenium")
        return "xenium"

    def _load_model(self, model_path: Path) -> tuple[torch.nn.Module, list[str]]:
        """Load trained model and class names."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Get class names from checkpoint (try multiple keys)
        class_names = checkpoint.get("class_names", [])
        if not class_names and "class_mapping" in checkpoint:
            # class_mapping is {name: idx}
            mapping = checkpoint["class_mapping"]
            class_names = sorted(mapping.keys(), key=lambda x: mapping[x])

        # Try config.class_mapping
        if not class_names and self.config.class_mapping:
            mapping = self.config.class_mapping
            class_names = sorted(mapping.keys(), key=lambda x: mapping[x])

        # Fall back to generic names
        if not class_names:
            num_classes = checkpoint.get("num_classes", 4)
            # Try to infer from model state dict
            for key, val in checkpoint.get("model_state_dict", {}).items():
                if "classifier" in key and "weight" in key:
                    num_classes = val.shape[0]
                    break
            class_names = [f"class_{i}" for i in range(num_classes)]

        # Reconstruct model
        from dapidl.models.classifier import CellTypeClassifier

        backbone = checkpoint.get("backbone", "efficientnetv2_rw_s")
        num_classes = len(class_names)

        model = CellTypeClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Set to inference mode
        model.train(False)

        if torch.cuda.is_available():
            model = model.cuda()

        return model, class_names

    def _load_target_data(
        self,
        data_path: Path,
        platform: str,
        pixel_size: float,
        class_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load patches from target platform data."""
        cfg = self.config

        # Check for existing LMDB dataset
        lmdb_path = data_path / "patches.lmdb"
        if lmdb_path.exists():
            try:
                return self._load_from_lmdb(lmdb_path, class_names)
            except Exception as e:
                logger.warning(f"Failed to load LMDB: {e}, trying Zarr...")

        # Check for Zarr dataset
        zarr_path = data_path / "patches.zarr"
        if zarr_path.exists():
            return self._load_from_zarr(zarr_path, data_path, class_names)

        # Extract patches directly from raw data
        return self._extract_patches(data_path, platform, pixel_size, class_names)

    def _load_from_lmdb(
        self,
        lmdb_path: Path,
        class_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load patches from existing LMDB dataset."""
        cfg = self.config

        env = lmdb.open(str(lmdb_path), readonly=True)
        with env.begin() as txn:
            metadata = json.loads(txn.get(b"__metadata__").decode())
            n_samples = metadata["length"]
            patch_size = metadata.get("patch_size", cfg.patch_size)

            if cfg.max_samples > 0:
                n_samples = min(n_samples, cfg.max_samples)

            patches = []
            labels = []
            cell_ids = []

            for i in range(n_samples):
                key = f"{i:08d}".encode()
                patch_bytes = txn.get(key)
                if patch_bytes is None:
                    continue

                patch = np.frombuffer(patch_bytes, dtype=np.float32)
                patch = patch.reshape(patch_size, patch_size)
                patches.append(patch)

                label_key = f"label_{i:08d}".encode()
                label = int(txn.get(label_key).decode())
                labels.append(label)
                cell_ids.append(f"cell_{i}")

        env.close()

        return np.array(patches), np.array(labels), cell_ids

    def _load_from_zarr(
        self,
        zarr_path: Path,
        data_path: Path,
        class_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load patches from existing Zarr dataset."""
        import zarr

        cfg = self.config

        # Load zarr array
        patches_zarr = zarr.open(str(zarr_path), mode="r")
        n_samples = patches_zarr.shape[0]

        if cfg.max_samples > 0:
            n_samples = min(n_samples, cfg.max_samples)

        # Load labels from .npy file
        labels_path = data_path / "labels.npy"
        if labels_path.exists():
            all_labels = np.load(labels_path)
        else:
            # No labels available, use zeros
            all_labels = np.zeros(patches_zarr.shape[0], dtype=np.int64)

        # Load metadata for cell IDs if available
        metadata_path = data_path / "metadata.parquet"
        if metadata_path.exists():
            metadata = pl.read_parquet(metadata_path)
            if "cell_id" in metadata.columns:
                all_cell_ids = metadata["cell_id"].to_list()
            else:
                all_cell_ids = [f"cell_{i}" for i in range(patches_zarr.shape[0])]
        else:
            all_cell_ids = [f"cell_{i}" for i in range(patches_zarr.shape[0])]

        patches = []
        labels = []
        cell_ids = []

        for i in range(n_samples):
            patch = patches_zarr[i]

            # Normalize if needed
            if patch.dtype == np.uint16:
                patch = patch.astype(np.float32) / 65535.0
            elif patch.dtype == np.uint8:
                patch = patch.astype(np.float32) / 255.0
            elif patch.dtype != np.float32:
                patch = patch.astype(np.float32)

            patches.append(patch)
            labels.append(all_labels[i])
            cell_ids.append(all_cell_ids[i] if i < len(all_cell_ids) else f"cell_{i}")

        logger.info(f"Loaded {len(patches)} patches from Zarr")
        return np.array(patches), np.array(labels), cell_ids

    def _extract_patches(
        self,
        data_path: Path,
        platform: str,
        pixel_size: float,
        class_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract patches directly from raw platform data."""
        cfg = self.config

        if platform == "xenium":
            from dapidl.data.xenium import XeniumDataReader
            reader = XeniumDataReader(data_path)
        else:
            from dapidl.data.merscope import MerscopeDataReader
            reader = MerscopeDataReader(data_path)

        image = reader.image
        centroids = reader.get_centroids_pixels()
        cell_ids = reader.get_cell_ids()

        # Load annotations if available
        annot_path = data_path / "annotations.parquet"
        if not annot_path.exists():
            annot_path = data_path / "metadata.parquet"

        if annot_path.exists():
            annot_df = pl.read_parquet(annot_path)
            label_col = self._find_label_column(annot_df)
            if label_col:
                id_to_label = dict(zip(
                    annot_df["cell_id"].to_list() if "cell_id" in annot_df.columns
                    else annot_df["EntityID"].to_list(),
                    annot_df[label_col].to_list()
                ))
            else:
                id_to_label = {}
        else:
            id_to_label = {}

        # Calculate extraction size for physical normalization
        if cfg.normalize_physical_size:
            target_physical_um = cfg.patch_size * cfg.target_pixel_size_um
            extract_size = int(target_physical_um / pixel_size)
            if extract_size % 2 != 0:
                extract_size += 1
        else:
            extract_size = cfg.patch_size

        half = extract_size // 2
        h, w = image.shape[:2]

        patches = []
        labels = []
        valid_cell_ids = []

        class_to_idx = {name: i for i, name in enumerate(class_names)}

        max_samples = cfg.max_samples if cfg.max_samples > 0 else len(centroids)

        for i, (cx, cy) in enumerate(centroids[:max_samples]):
            cx, cy = int(cx), int(cy)

            # Skip edge cells
            if cx < half or cx >= w - half or cy < half or cy >= h - half:
                continue

            # Extract patch
            patch = image[cy-half:cy+half, cx-half:cx+half]
            if patch.shape != (extract_size, extract_size):
                continue

            # Resize if physical normalization
            if cfg.normalize_physical_size and extract_size != cfg.patch_size:
                patch = cv2.resize(
                    patch,
                    (cfg.patch_size, cfg.patch_size),
                    interpolation=cv2.INTER_LINEAR
                )

            # Normalize to float32
            if patch.dtype == np.uint16:
                patch = patch.astype(np.float32) / 65535.0
            elif patch.dtype == np.uint8:
                patch = patch.astype(np.float32) / 255.0

            patches.append(patch)

            # Get label
            cid = cell_ids[i]
            label_name = id_to_label.get(cid, class_names[0])
            label_idx = class_to_idx.get(label_name, 0)
            labels.append(label_idx)
            valid_cell_ids.append(cid)

        return np.array(patches), np.array(labels), valid_cell_ids

    def _find_label_column(self, df: pl.DataFrame) -> str | None:
        """Find the label column in annotations dataframe."""
        for col in ["broad_category", "cell_type", "predicted_labels", "label"]:
            if col in df.columns:
                return col
        return None

    def _run_inference(
        self,
        model: torch.nn.Module,
        patches: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run model inference on patches."""
        cfg = self.config
        device = next(model.parameters()).device

        all_predictions = []
        all_confidences = []

        # Process in batches
        for i in range(0, len(patches), cfg.batch_size):
            batch = patches[i:i + cfg.batch_size]

            # Convert to tensor (N, 1, H, W) -> (N, 3, H, W) via model adapter
            batch_tensor = torch.from_numpy(batch).float().unsqueeze(1)
            batch_tensor = batch_tensor.to(device)

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                confs = probs.max(dim=1).values

            all_predictions.extend(preds.cpu().numpy())
            all_confidences.extend(confs.cpu().numpy())

        return np.array(all_predictions), np.array(all_confidences)

    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        class_names: list[str],
    ) -> dict[str, Any]:
        """Calculate testing metrics."""
        cfg = self.config

        # Filter by confidence threshold
        if cfg.confidence_threshold > 0:
            mask = confidences >= cfg.confidence_threshold
            labels = labels[mask]
            predictions = predictions[mask]
            confidences = confidences[mask]

        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
            "mean_confidence": float(np.mean(confidences)),
        }

        # Add classification report with proper label handling
        try:
            # Get unique labels present in data
            unique_labels = np.unique(np.concatenate([labels, predictions]))
            # Filter class names to only those present
            present_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]
            metrics["classification_report"] = classification_report(
                labels, predictions,
                labels=unique_labels,
                target_names=present_class_names,
                output_dict=True,
                zero_division=0,
            )
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            metrics["classification_report"] = {}

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                per_class[name] = {
                    "support": int(mask.sum()),
                    "accuracy": float((predictions[mask] == i).mean()),
                    "mean_confidence": float(confidences[mask].mean()),
                }
        metrics["per_class"] = per_class

        return metrics

    def _load_baseline(self, baseline_path: str) -> dict[str, Any]:
        """Load baseline metrics for comparison."""
        path = Path(baseline_path)
        if not path.exists():
            logger.warning(f"Baseline not found: {baseline_path}")
            return {}

        with open(path) as f:
            return json.load(f)

    def _compare_to_baseline(
        self,
        metrics: dict[str, Any],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare transfer metrics to same-platform baseline."""
        if not baseline:
            return {}

        comparison = {}

        for key in ["accuracy", "macro_f1", "weighted_f1"]:
            if key in metrics and key in baseline:
                transfer_val = metrics[key]
                baseline_val = baseline[key]
                comparison[key] = {
                    "transfer": transfer_val,
                    "baseline": baseline_val,
                    "delta": transfer_val - baseline_val,
                    "retention": transfer_val / baseline_val if baseline_val > 0 else 0,
                }

        return comparison

    def _log_results(self, metrics: dict[str, Any], class_names: list[str]) -> None:
        """Log testing results."""
        logger.info("=" * 60)
        logger.info("CROSS-PLATFORM TRANSFER RESULTS")
        logger.info("=" * 60)
        logger.info(f"Source: {metrics['source_platform']} -> Target: {metrics['target_platform']}")
        logger.info(f"Samples: {metrics['num_samples']}")
        logger.info(f"Physical normalized: {metrics['physical_normalized']}")
        logger.info("-" * 60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
        logger.info("-" * 60)

        # Per-class
        logger.info("Per-class performance:")
        for name in class_names:
            if name in metrics.get("per_class", {}):
                pc = metrics["per_class"][name]
                logger.info(f"  {name}: acc={pc['accuracy']:.3f}, n={pc['support']}")

        # Baseline comparison
        if "baseline_comparison" in metrics and metrics["baseline_comparison"]:
            logger.info("-" * 60)
            logger.info("Comparison to same-platform baseline:")
            for key, vals in metrics["baseline_comparison"].items():
                logger.info(
                    f"  {key}: {vals['transfer']:.4f} vs {vals['baseline']:.4f} "
                    f"(delta={vals['delta']:+.4f}, retention={vals['retention']:.1%})"
                )

        logger.info("=" * 60)

    def get_queue(self) -> str:
        """GPU queue for inference."""
        return "gpu"

    def create_clearml_task(self, project: str, task_name: str):
        """Create a ClearML task for this step."""
        from clearml import Task

        task = Task.init(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.testing,
        )

        # Connect parameters
        task.connect(self.config, name="step_config")

        return task
