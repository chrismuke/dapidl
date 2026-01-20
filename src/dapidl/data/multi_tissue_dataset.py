"""Multi-Tissue Dataset for cross-tissue universal training.

Combines multiple LMDB datasets from different tissues into a unified training set
with tissue-balanced sampling and CL-standardized labels.

Features:
- Combines datasets from Xenium/MERSCOPE across tissues (breast, lung, ovarian, etc.)
- Multiple sampling strategies: equal, proportional, sqrt-balanced
- Confidence-weighted training (Tier 1/2/3 label confidence)
- Cell Ontology standardized labels for cross-tissue consistency
- Hierarchical label support (coarse/medium/fine)

Usage:
    datasets = [
        {"path": "xenium-breast/lmdb", "tissue": "breast", "platform": "xenium"},
        {"path": "merscope-liver/lmdb", "tissue": "liver", "platform": "merscope"},
    ]
    dataset = MultiTissueDataset(datasets, sampling_strategy="sqrt")
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import lmdb
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from loguru import logger

from dapidl.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    compute_dataset_stats,
)
from dapidl.data.hierarchical_dataset import HierarchicalLabels
from dapidl.models.hierarchical import HierarchyConfig


SamplingStrategy = Literal["equal", "proportional", "sqrt"]
ConfidenceTier = Literal[1, 2, 3]


@dataclass
class TissueDatasetConfig:
    """Configuration for a single tissue dataset."""

    path: str | Path
    tissue: str
    platform: str = "xenium"
    confidence_tier: ConfidenceTier = 2
    weight_multiplier: float = 1.0

    def __post_init__(self):
        self.path = Path(self.path)


@dataclass
class MultiTissueConfig:
    """Configuration for multi-tissue training."""

    datasets: list[TissueDatasetConfig] = field(default_factory=list)
    sampling_strategy: SamplingStrategy = "sqrt"
    use_hierarchical: bool = True
    confidence_weights: dict[int, float] = field(default_factory=lambda: {1: 1.0, 2: 0.8, 3: 0.5})
    min_samples_per_class: int = 20
    standardize_labels: bool = True  # Use Cell Ontology standardization

    def add_dataset(
        self,
        path: str | Path,
        tissue: str,
        platform: str = "xenium",
        confidence_tier: ConfidenceTier = 2,
        weight_multiplier: float = 1.0,
    ) -> "MultiTissueConfig":
        """Add a dataset to the configuration.

        Args:
            path: Path to LMDB dataset
            tissue: Tissue type (breast, lung, liver, etc.)
            platform: Platform (xenium, merscope)
            confidence_tier: Label confidence (1=ground truth, 2=consensus, 3=predicted)
            weight_multiplier: Additional weight multiplier for this dataset

        Returns:
            Self for chaining
        """
        self.datasets.append(TissueDatasetConfig(
            path=path,
            tissue=tissue,
            platform=platform,
            confidence_tier=confidence_tier,
            weight_multiplier=weight_multiplier,
        ))
        return self


class MultiTissueDataset(Dataset):
    """Multi-tissue dataset combining samples from multiple LMDB datasets.

    Supports tissue-balanced sampling, confidence weighting, and Cell Ontology
    standardized labels for cross-tissue generalization training.
    """

    def __init__(
        self,
        config: MultiTissueConfig,
        split: str = "train",
        transform: Callable | None = None,
        indices: np.ndarray | None = None,
        hierarchical_labels: HierarchicalLabels | None = None,
    ) -> None:
        """Initialize multi-tissue dataset.

        Args:
            config: Multi-tissue configuration
            split: One of 'train', 'val', 'test'
            transform: Optional custom transform
            indices: Optional subset indices (global indices across all datasets)
            hierarchical_labels: Pre-computed hierarchical label mapper
        """
        self.config = config
        self.split = split

        # Load all datasets
        self._load_datasets()

        # Build unified class mapping
        self._build_unified_labels()

        # Build hierarchical labels if needed
        if config.use_hierarchical:
            if hierarchical_labels is None:
                hierarchical_labels = HierarchicalLabels(self.unified_class_mapping)
            self.hierarchical_labels = hierarchical_labels
            self.hierarchy_config = hierarchical_labels.hierarchy_config
        else:
            self.hierarchical_labels = None
            self.hierarchy_config = None

        # Handle indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.total_samples)

        # Load per-dataset normalization stats
        # CRITICAL: Each platform has different intensity distributions
        # (e.g., MERSCOPE is ~70x brighter than Xenium)
        self.per_dataset_stats: list[dict[str, float]] = []
        for ds_config in config.datasets:
            stats = compute_dataset_stats(ds_config.path)
            self.per_dataset_stats.append(stats)
            logger.debug(
                f"Loaded normalization stats for {ds_config.tissue}/{ds_config.platform}: "
                f"p_low={stats['p_low']:.0f}, p_high={stats['p_high']:.0f}"
            )

        # For backward compatibility, expose first dataset's stats
        self.stats = self.per_dataset_stats[0]

        # Build per-dataset transforms (each uses its own normalization stats)
        self.per_dataset_transforms: list[Callable] = []
        if transform is not None:
            # If explicit transform provided, use it for all datasets
            for _ in config.datasets:
                self.per_dataset_transforms.append(transform)
            self.transform = transform  # backward compatibility
        elif split == "train":
            for stats in self.per_dataset_stats:
                self.per_dataset_transforms.append(get_train_transforms(stats=stats))
            self.transform = self.per_dataset_transforms[0]  # backward compatibility
        else:
            for stats in self.per_dataset_stats:
                self.per_dataset_transforms.append(get_val_transforms(stats=stats))
            self.transform = self.per_dataset_transforms[0]  # backward compatibility

        self._log_dataset_info()

    def _load_datasets(self) -> None:
        """Load all LMDB datasets and build global index mapping."""
        # Store LMDB paths for lazy opening (fork-safe)
        self.lmdb_paths: list[Path] = []
        self._lmdb_envs: list[lmdb.Environment | None] = []  # Lazily opened
        self.dataset_offsets = [0]  # Cumulative offsets for global indexing
        self.dataset_sizes = []
        self.tissue_indices = []  # Which tissue each sample belongs to
        self.platform_indices = []  # Which platform each sample belongs to
        self.confidence_tiers = []  # Confidence tier per sample

        all_labels = []
        all_class_mappings = []

        for i, ds_config in enumerate(self.config.datasets):
            lmdb_path = ds_config.path / "patches.lmdb"

            if not lmdb_path.exists():
                raise FileNotFoundError(
                    f"LMDB not found: {lmdb_path}. "
                    "Run export-lmdb to create it."
                )

            # Store path for lazy opening (don't open yet - not fork-safe)
            self.lmdb_paths.append(lmdb_path)
            self._lmdb_envs.append(None)  # Will be opened lazily

            # Load labels
            labels = np.load(ds_config.path / "labels.npy")
            n_samples = len(labels)
            self.dataset_sizes.append(n_samples)

            # Update offsets
            self.dataset_offsets.append(self.dataset_offsets[-1] + n_samples)

            # Track tissue/platform per sample
            self.tissue_indices.extend([i] * n_samples)
            self.platform_indices.extend([ds_config.platform] * n_samples)
            self.confidence_tiers.extend([ds_config.confidence_tier] * n_samples)

            # Load class mapping
            with open(ds_config.path / "class_mapping.json") as f:
                class_mapping = json.load(f)
            all_class_mappings.append(class_mapping)
            all_labels.append(labels)

            # Load metadata for patch size
            metadata_path = ds_config.path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                ds_patch_size = metadata.get("patch_shape", [128, 128])[0]
            else:
                ds_patch_size = 128  # Default fallback

            # Validate/set patch size
            if i == 0:
                self.patch_size = ds_patch_size
            elif ds_patch_size != self.patch_size:
                raise ValueError(
                    f"Dataset {ds_config.path} has patch_size={ds_patch_size}, "
                    f"expected {self.patch_size}. All datasets must have same patch size."
                )

            logger.info(
                f"Loaded {ds_config.tissue}/{ds_config.platform}: "
                f"{n_samples} samples, {len(class_mapping)} classes"
            )

        self.total_samples = self.dataset_offsets[-1]
        self.tissue_indices = np.array(self.tissue_indices)
        self.confidence_tiers = np.array(self.confidence_tiers)

        # Store for label remapping
        self._all_labels = all_labels
        self._all_class_mappings = all_class_mappings

    def _build_unified_labels(self) -> None:
        """Build unified class mapping across all datasets.

        When standardize_labels is True, uses Cell Ontology to map
        equivalent cell types to the same label.
        """
        if self.config.standardize_labels:
            self._build_standardized_labels()
        else:
            self._build_simple_union_labels()

    def _build_simple_union_labels(self) -> None:
        """Build unified labels as simple union of all class names."""
        # Collect all unique class names
        all_classes = set()
        for mapping in self._all_class_mappings:
            all_classes.update(mapping.keys())

        # Sort for deterministic ordering
        unified_classes = sorted(all_classes)
        self.unified_class_mapping = {name: i for i, name in enumerate(unified_classes)}
        self.num_classes = len(unified_classes)
        self.class_names = unified_classes

        # Remap all labels to unified indices
        self.labels = np.zeros(self.total_samples, dtype=np.int64)
        offset = 0

        for i, (labels, mapping) in enumerate(zip(self._all_labels, self._all_class_mappings)):
            # Build local-to-unified mapping
            local_to_unified = {}
            reverse_mapping = {v: k for k, v in mapping.items()}

            for local_idx in range(len(mapping)):
                class_name = reverse_mapping[local_idx]
                local_to_unified[local_idx] = self.unified_class_mapping[class_name]

            # Remap labels
            for j, local_label in enumerate(labels):
                self.labels[offset + j] = local_to_unified[local_label]

            offset += len(labels)

    def _build_standardized_labels(self) -> None:
        """Build unified labels using Cell Ontology standardization.

        Maps equivalent cell types from different datasets to the same
        CL-standardized label.
        """
        try:
            from dapidl.ontology import map_label, get_term
            has_ontology = True
        except ImportError:
            logger.warning("Ontology module not available, using simple union")
            self._build_simple_union_labels()
            return

        # Map all class names to CL IDs
        cl_to_unified = {}  # CL ID -> unified class name
        all_cl_ids = set()

        for mapping in self._all_class_mappings:
            for class_name in mapping.keys():
                cl_id = map_label(class_name)
                if cl_id != "UNMAPPED":
                    all_cl_ids.add(cl_id)
                    # Use canonical name from ontology
                    term = get_term(cl_id)
                    if term and term.name:
                        cl_to_unified[cl_id] = term.name
                    else:
                        cl_to_unified[cl_id] = class_name
                else:
                    # Keep original name for unmapped types
                    all_cl_ids.add(f"UNMAPPED:{class_name}")
                    cl_to_unified[f"UNMAPPED:{class_name}"] = class_name

        # Build unified class mapping
        unified_classes = sorted(set(cl_to_unified.values()))
        self.unified_class_mapping = {name: i for i, name in enumerate(unified_classes)}
        self.num_classes = len(unified_classes)
        self.class_names = unified_classes

        # Build name-to-unified mapping for each original class
        name_to_unified = {}
        for mapping in self._all_class_mappings:
            for class_name in mapping.keys():
                cl_id = map_label(class_name)
                if cl_id != "UNMAPPED":
                    unified_name = cl_to_unified[cl_id]
                else:
                    unified_name = class_name
                name_to_unified[class_name] = self.unified_class_mapping.get(
                    unified_name, 0
                )

        # Remap all labels to unified indices
        self.labels = np.zeros(self.total_samples, dtype=np.int64)
        offset = 0

        for i, (labels, mapping) in enumerate(zip(self._all_labels, self._all_class_mappings)):
            reverse_mapping = {v: k for k, v in mapping.items()}

            for j, local_label in enumerate(labels):
                class_name = reverse_mapping[local_label]
                self.labels[offset + j] = name_to_unified[class_name]

            offset += len(labels)

        logger.info(
            f"Standardized to {self.num_classes} unified classes "
            f"(CL-mapped: {len([c for c in all_cl_ids if not c.startswith('UNMAPPED')])})"
        )

    def _log_dataset_info(self) -> None:
        """Log dataset statistics."""
        logger.info(
            f"MultiTissueDataset: {len(self)} samples from "
            f"{len(self.config.datasets)} datasets, {self.num_classes} classes"
        )

        # Per-tissue breakdown
        for i, ds_config in enumerate(self.config.datasets):
            mask = self.tissue_indices[self.indices] == i
            n_samples = mask.sum()
            logger.info(
                f"  {ds_config.tissue}/{ds_config.platform}: {n_samples} samples "
                f"(tier {ds_config.confidence_tier})"
            )

    def _get_dataset_and_local_idx(self, global_idx: int) -> tuple[int, int]:
        """Convert global index to dataset index and local index.

        Args:
            global_idx: Global sample index

        Returns:
            Tuple of (dataset_index, local_index)
        """
        for i, (start, end) in enumerate(zip(
            self.dataset_offsets[:-1],
            self.dataset_offsets[1:]
        )):
            if start <= global_idx < end:
                return i, global_idx - start
        raise IndexError(f"Global index {global_idx} out of range")

    def __len__(self) -> int:
        return len(self.indices)

    def _get_lmdb_env(self, dataset_idx: int) -> lmdb.Environment:
        """Lazily open LMDB environment (fork-safe).

        LMDB environments must be opened per-worker after fork to avoid
        segmentation faults in DataLoader workers.
        """
        if self._lmdb_envs[dataset_idx] is None:
            self._lmdb_envs[dataset_idx] = lmdb.open(
                str(self.lmdb_paths[dataset_idx]),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._lmdb_envs[dataset_idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Get a single sample.

        Args:
            idx: Sample index within current split

        Returns:
            Tuple of (image_tensor, labels_dict)
            labels_dict contains 'label' (unified) and optionally
            'coarse', 'medium', 'fine' for hierarchical mode
        """
        global_idx = self.indices[idx]
        dataset_idx, local_idx = self._get_dataset_and_local_idx(global_idx)

        # Load patch from LMDB (lazily opened for fork-safety)
        # Keys are stored as big-endian uint64, values as label (8 bytes) + patch data
        env = self._get_lmdb_env(dataset_idx)
        with env.begin(write=False) as txn:
            key = struct.pack(">Q", local_idx)
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Sample {local_idx} not found in dataset {dataset_idx}")
            # Skip first 8 bytes (label prefix), rest is patch data
            patch_bytes = value[8:]
            patch = np.frombuffer(patch_bytes, dtype=np.uint16).reshape(
                self.patch_size, self.patch_size
            ).copy()

        # Get unified label
        label = self.labels[global_idx]

        # Apply per-dataset transform (uses correct normalization stats)
        # This is critical for multi-platform data (Xenium vs MERSCOPE have 70x intensity difference)
        transform = self.per_dataset_transforms[dataset_idx]
        if transform is not None:
            transformed = transform(image=patch)
            patch = transformed["image"]

        # Build labels dict
        labels_out = {"label": int(label)}

        # Add hierarchical labels if enabled
        if self.hierarchical_labels is not None:
            coarse_labels, medium_labels, _ = self.hierarchical_labels.convert_labels(
                np.array([label])
            )
            labels_out["coarse"] = int(coarse_labels[0])
            labels_out["medium"] = int(medium_labels[0])
            labels_out["fine"] = int(label)

        # Add metadata
        labels_out["tissue_idx"] = int(self.tissue_indices[global_idx])
        labels_out["confidence_tier"] = int(self.confidence_tiers[global_idx])

        return patch, labels_out

    def get_class_weights(
        self,
        max_weight_ratio: float = 10.0,
        include_confidence: bool = True,
    ) -> torch.Tensor:
        """Compute class weights incorporating confidence tiers.

        Args:
            max_weight_ratio: Maximum ratio between weights
            include_confidence: Whether to include confidence tier weighting

        Returns:
            Tensor of class weights
        """
        subset_labels = self.labels[self.indices]
        class_counts = np.bincount(subset_labels, minlength=self.num_classes).astype(float)
        class_counts = np.maximum(class_counts, 1)

        weights = 1.0 / class_counts

        if max_weight_ratio > 0:
            min_weight = weights.min()
            max_allowed = min_weight * max_weight_ratio
            weights = np.minimum(weights, max_allowed)

        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler.

        Combines class weights with:
        - Tissue sampling strategy (equal/proportional/sqrt)
        - Confidence tier weights

        Returns:
            Array of sample weights
        """
        class_weights = self.get_class_weights().numpy()
        subset_labels = self.labels[self.indices]
        sample_weights = class_weights[subset_labels]

        # Apply tissue sampling strategy
        if self.config.sampling_strategy != "proportional":
            tissue_weights = self._compute_tissue_weights()
            subset_tissue_indices = self.tissue_indices[self.indices]
            tissue_multipliers = tissue_weights[subset_tissue_indices]
            sample_weights = sample_weights * tissue_multipliers

        # Apply confidence tier weights
        subset_tiers = self.confidence_tiers[self.indices]
        conf_weights = self.config.confidence_weights
        tier_multipliers = np.array([
            conf_weights.get(int(t), 1.0) for t in subset_tiers
        ])
        sample_weights = sample_weights * tier_multipliers

        return sample_weights

    def _compute_tissue_weights(self) -> np.ndarray:
        """Compute tissue weights based on sampling strategy.

        Returns:
            Array of weights per tissue
        """
        subset_tissue = self.tissue_indices[self.indices]
        n_tissues = len(self.config.datasets)
        tissue_counts = np.bincount(subset_tissue, minlength=n_tissues).astype(float)
        tissue_counts = np.maximum(tissue_counts, 1)

        if self.config.sampling_strategy == "equal":
            # Equal samples from each tissue
            weights = 1.0 / tissue_counts
        elif self.config.sampling_strategy == "sqrt":
            # Square root balancing (compromise between equal and proportional)
            weights = 1.0 / np.sqrt(tissue_counts)
        else:  # proportional
            weights = np.ones(n_tissues)

        # Apply per-dataset weight multipliers
        for i, ds_config in enumerate(self.config.datasets):
            weights[i] *= ds_config.weight_multiplier

        weights = weights / weights.sum() * n_tissues
        return weights

    def get_hierarchical_class_weights(
        self,
        max_weight_ratio: float = 10.0,
    ) -> dict[str, torch.Tensor]:
        """Get class weights at each hierarchy level.

        Args:
            max_weight_ratio: Maximum ratio between weights

        Returns:
            Dict with coarse_weights, medium_weights, fine_weights
        """
        if self.hierarchical_labels is None:
            raise ValueError("Hierarchical labels not enabled")

        subset_labels = self.labels[self.indices]
        return self.hierarchical_labels.get_class_weights_per_level(
            subset_labels, max_weight_ratio
        )

    def close(self) -> None:
        """Close all LMDB environments."""
        for env in self._lmdb_envs:
            if env is not None:
                env.close()
        self._lmdb_envs = [None] * len(self._lmdb_envs)


def create_multi_tissue_splits(
    config: MultiTissueConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str = "label",  # "label", "tissue", or "both"
) -> tuple[MultiTissueDataset, MultiTissueDataset, MultiTissueDataset]:
    """Create train/val/test splits for multi-tissue dataset.

    Args:
        config: Multi-tissue configuration
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        stratify_by: What to stratify by ("label", "tissue", or "both")

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create full dataset to get total samples and labels
    full_dataset = MultiTissueDataset(config, split="train")
    n_samples = full_dataset.total_samples
    indices = np.arange(n_samples)

    # Build stratification labels
    if stratify_by == "label":
        stratify_labels = full_dataset.labels
    elif stratify_by == "tissue":
        stratify_labels = full_dataset.tissue_indices
    elif stratify_by == "both":
        # Combine tissue and label into compound stratification
        stratify_labels = (
            full_dataset.tissue_indices * full_dataset.num_classes +
            full_dataset.labels
        )
    else:
        stratify_labels = None

    # Filter rare strata if needed
    if stratify_labels is not None and config.min_samples_per_class > 0:
        unique, counts = np.unique(stratify_labels, return_counts=True)
        valid_strata = unique[counts >= config.min_samples_per_class]
        if len(valid_strata) < len(unique):
            mask = np.isin(stratify_labels, valid_strata)
            indices = indices[mask]
            stratify_labels = stratify_labels[mask]
            logger.info(f"Filtered to {len(indices)} samples for stratification")

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    if stratify_labels is not None:
        temp_strat = stratify_labels[np.isin(indices, temp_indices)]
        temp_positions = np.array([np.where(indices == i)[0][0] for i in temp_indices])
        temp_strat = stratify_labels[temp_positions]
    else:
        temp_strat = None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=seed,
        stratify=temp_strat,
    )

    logger.info(
        f"Multi-tissue splits: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    # Clean up full dataset
    full_dataset.close()

    # Create split datasets (share hierarchical labels if computed)
    hierarchical_labels = None
    if config.use_hierarchical:
        # Create shared hierarchical labels
        temp_ds = MultiTissueDataset(config, split="train", indices=train_indices)
        hierarchical_labels = temp_ds.hierarchical_labels
        temp_ds.close()

    train_dataset = MultiTissueDataset(
        config, split="train", indices=train_indices,
        hierarchical_labels=hierarchical_labels,
    )
    val_dataset = MultiTissueDataset(
        config, split="val", indices=val_indices,
        hierarchical_labels=hierarchical_labels,
    )
    test_dataset = MultiTissueDataset(
        config, split="test", indices=test_indices,
        hierarchical_labels=hierarchical_labels,
    )

    return train_dataset, val_dataset, test_dataset


def create_multi_tissue_dataloaders(
    train_dataset: MultiTissueDataset,
    val_dataset: MultiTissueDataset,
    test_dataset: MultiTissueDataset | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    use_weighted_sampler: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders for multi-tissue datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        use_weighted_sampler: Use WeightedRandomSampler
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    def multi_tissue_collate(batch):
        """Custom collate function for multi-tissue batch."""
        images = torch.stack([item[0] for item in batch])

        # Extract all label fields
        labels_keys = batch[0][1].keys()
        labels = {}
        for key in labels_keys:
            if key in ["tissue_idx", "confidence_tier"]:
                labels[key] = torch.tensor([item[1][key] for item in batch])
            else:
                labels[key] = torch.tensor([item[1][key] for item in batch])

        return images, labels

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers and num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "collate_fn": multi_tissue_collate,
    }

    # Training loader with weighted sampling
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Test loader
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader
