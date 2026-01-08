"""Tests for universal cross-tissue training pipeline.

Tests cover:
- MultiTissueConfig and TissueDatasetConfig dataclasses
- MultiTissueDataset initialization and sampling strategies
- UniversalTrainingConfig and TissueDatasetSpec
- UniversalDAPITrainingStep validation and parameter schema
- UniversalPipelineConfig and TissueConfig
- UniversalDAPIPipelineController configuration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import lmdb
import numpy as np
import pytest
import torch

from dapidl.data.multi_tissue_dataset import (
    MultiTissueConfig,
    MultiTissueDataset,
    TissueDatasetConfig,
    create_multi_tissue_splits,
)
from dapidl.pipeline.steps.universal_training import (
    TissueDatasetSpec,
    UniversalTrainingConfig,
    UniversalDAPITrainingStep,
)
from dapidl.pipeline.universal_controller import (
    TissueConfig,
    UniversalPipelineConfig,
    UniversalDAPIPipelineController,
    create_universal_pipeline,
)
from dapidl.pipeline.base import StepArtifacts


class TestTissueDatasetConfig:
    """Tests for TissueDatasetConfig dataclass."""

    def test_basic_config(self):
        """Test basic tissue dataset configuration."""
        config = TissueDatasetConfig(
            path="/path/to/dataset",
            tissue="breast",
            platform="xenium",
            confidence_tier=1,
            weight_multiplier=1.5,
        )

        assert config.path == Path("/path/to/dataset")
        assert config.tissue == "breast"
        assert config.platform == "xenium"
        assert config.confidence_tier == 1
        assert config.weight_multiplier == 1.5

    def test_default_values(self):
        """Test default configuration values."""
        config = TissueDatasetConfig(
            path="/path/to/dataset",
            tissue="lung",
        )

        assert config.platform == "xenium"
        assert config.confidence_tier == 2
        assert config.weight_multiplier == 1.0


class TestMultiTissueConfig:
    """Tests for MultiTissueConfig dataclass."""

    def test_empty_config(self):
        """Test empty multi-tissue configuration."""
        config = MultiTissueConfig()

        assert len(config.datasets) == 0
        assert config.sampling_strategy == "sqrt"
        assert config.use_hierarchical is True
        assert config.standardize_labels is True
        assert config.min_samples_per_class == 20

    def test_add_dataset_fluent(self):
        """Test fluent API for adding datasets."""
        config = (
            MultiTissueConfig()
            .add_dataset(
                path="/data/breast",
                tissue="breast",
                platform="xenium",
                confidence_tier=1,
            )
            .add_dataset(
                path="/data/lung",
                tissue="lung",
                platform="merscope",
                confidence_tier=2,
            )
        )

        assert len(config.datasets) == 2
        assert config.datasets[0].tissue == "breast"
        assert config.datasets[0].confidence_tier == 1
        assert config.datasets[1].tissue == "lung"
        assert config.datasets[1].platform == "merscope"

    def test_confidence_weights(self):
        """Test confidence tier weights configuration."""
        config = MultiTissueConfig(
            confidence_weights={1: 1.0, 2: 0.8, 3: 0.5}
        )

        assert config.confidence_weights[1] == 1.0
        assert config.confidence_weights[2] == 0.8
        assert config.confidence_weights[3] == 0.5

    def test_sampling_strategies(self):
        """Test valid sampling strategies."""
        for strategy in ["equal", "proportional", "sqrt"]:
            config = MultiTissueConfig(sampling_strategy=strategy)
            assert config.sampling_strategy == strategy


class TestTissueDatasetSpec:
    """Tests for TissueDatasetSpec in universal training."""

    def test_basic_spec(self):
        """Test basic tissue dataset spec creation."""
        spec = TissueDatasetSpec(
            path="/path/to/lmdb",
            tissue="breast",
            platform="xenium",
            confidence_tier=1,
            weight_multiplier=1.0,
        )

        assert spec.path == "/path/to/lmdb"
        assert spec.tissue == "breast"
        assert spec.platform == "xenium"
        assert spec.confidence_tier == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        spec = TissueDatasetSpec(
            path="/path/to/lmdb",
            tissue="lung",
            platform="merscope",
            confidence_tier=2,
            weight_multiplier=0.8,
        )

        d = spec.to_dict()

        assert d["path"] == "/path/to/lmdb"
        assert d["tissue"] == "lung"
        assert d["platform"] == "merscope"
        assert d["confidence_tier"] == 2
        assert d["weight_multiplier"] == 0.8


class TestUniversalTrainingConfig:
    """Tests for UniversalTrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = UniversalTrainingConfig()

        assert config.backbone == "efficientnetv2_rw_s"
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.sampling_strategy == "sqrt"
        assert config.standardize_labels is True
        assert config.coarse_only_epochs == 20
        assert config.coarse_medium_epochs == 50

    def test_add_dataset(self):
        """Test fluent API for adding datasets."""
        config = (
            UniversalTrainingConfig()
            .add_dataset("/path/breast", "breast", "xenium", 1)
            .add_dataset("/path/lung", "lung", "merscope", 2)
        )

        assert len(config.datasets) == 2
        assert config.datasets[0].tissue == "breast"
        assert config.datasets[1].platform == "merscope"

    def test_tier_weights(self):
        """Test confidence tier weight configuration."""
        config = UniversalTrainingConfig(
            tier1_weight=1.0,
            tier2_weight=0.7,
            tier3_weight=0.4,
        )

        assert config.tier1_weight == 1.0
        assert config.tier2_weight == 0.7
        assert config.tier3_weight == 0.4


class TestUniversalDAPITrainingStep:
    """Tests for UniversalDAPITrainingStep."""

    def test_step_creation(self):
        """Test step creation with default config."""
        step = UniversalDAPITrainingStep()

        assert step.name == "universal_training"
        assert step.queue == "gpu"
        assert step.config is not None

    def test_step_with_config(self):
        """Test step creation with custom config."""
        config = UniversalTrainingConfig(
            epochs=50,
            backbone="resnet18",
            batch_size=32,
        )
        step = UniversalDAPITrainingStep(config)

        assert step.config.epochs == 50
        assert step.config.backbone == "resnet18"
        assert step.config.batch_size == 32

    def test_parameter_schema(self):
        """Test parameter schema generation."""
        step = UniversalDAPITrainingStep()
        schema = step.get_parameter_schema()

        assert schema["type"] == "object"
        assert "properties" in schema

        props = schema["properties"]
        assert "sampling_strategy" in props
        assert "backbone" in props
        assert "epochs" in props
        assert "batch_size" in props
        assert "coarse_only_epochs" in props
        assert "patience" in props

        # Check enum values
        assert props["sampling_strategy"]["enum"] == ["equal", "proportional", "sqrt"]
        assert "efficientnetv2_rw_s" in props["backbone"]["enum"]
        assert "resnet18" in props["backbone"]["enum"]

    def test_validate_inputs_with_dataset_configs(self):
        """Test input validation with dataset_configs."""
        step = UniversalDAPITrainingStep()

        # Valid inputs
        artifacts = StepArtifacts(
            inputs={},
            outputs={
                "dataset_configs": [
                    {"path": "/path/to/dataset1", "tissue": "breast"},
                    {"path": "/path/to/dataset2", "tissue": "lung"},
                ]
            },
        )
        assert step.validate_inputs(artifacts) is True

        # Invalid - missing required fields
        artifacts = StepArtifacts(
            inputs={},
            outputs={
                "dataset_configs": [
                    {"path": "/path/to/dataset1"},  # Missing tissue
                ]
            },
        )
        assert step.validate_inputs(artifacts) is False

        # Invalid - empty list
        artifacts = StepArtifacts(
            inputs={},
            outputs={"dataset_configs": []},
        )
        assert step.validate_inputs(artifacts) is False

    def test_validate_inputs_with_patches_path(self):
        """Test input validation with patches_path_N pattern."""
        step = UniversalDAPITrainingStep()

        artifacts = StepArtifacts(
            inputs={},
            outputs={
                "patches_path_0": "/path/to/patches1",
                "patches_path_1": "/path/to/patches2",
            },
        )
        assert step.validate_inputs(artifacts) is True

    def test_validate_inputs_no_datasets(self):
        """Test input validation fails with no datasets."""
        step = UniversalDAPITrainingStep()

        artifacts = StepArtifacts(
            inputs={},
            outputs={"some_other_key": "value"},
        )
        assert step.validate_inputs(artifacts) is False


class TestTissueConfig:
    """Tests for TissueConfig in universal controller."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TissueConfig()

        assert config.dataset_id is None
        assert config.local_path is None
        assert config.tissue == "unknown"
        assert config.platform == "xenium"
        assert config.confidence_tier == 2
        assert config.weight_multiplier == 1.0
        assert config.annotator == "celltypist"

    def test_custom_config(self):
        """Test custom tissue configuration."""
        config = TissueConfig(
            dataset_id="abc123",
            tissue="breast",
            platform="merscope",
            confidence_tier=1,
            annotator="ground_truth",
            ground_truth_file="/path/to/labels.parquet",
        )

        assert config.dataset_id == "abc123"
        assert config.tissue == "breast"
        assert config.platform == "merscope"
        assert config.confidence_tier == 1
        assert config.annotator == "ground_truth"
        assert config.ground_truth_file == "/path/to/labels.parquet"


class TestUniversalPipelineConfig:
    """Tests for UniversalPipelineConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = UniversalPipelineConfig()

        assert config.name == "dapidl-universal"
        assert config.project == "DAPIDL/universal"
        assert len(config.tissues) == 0
        assert config.sampling_strategy == "sqrt"
        assert config.segmenter == "cellpose"
        assert config.epochs == 100
        assert config.standardize_labels is True

    def test_add_tissue_fluent(self):
        """Test fluent API for adding tissues."""
        config = (
            UniversalPipelineConfig(name="test-pipeline")
            .add_tissue(
                tissue="breast",
                dataset_id="abc123",
                platform="xenium",
                confidence_tier=1,
            )
            .add_tissue(
                tissue="lung",
                local_path="/data/lung",
                platform="merscope",
                confidence_tier=2,
            )
        )

        assert len(config.tissues) == 2
        assert config.tissues[0].tissue == "breast"
        assert config.tissues[0].dataset_id == "abc123"
        assert config.tissues[1].tissue == "lung"
        assert config.tissues[1].local_path == "/data/lung"

    def test_execution_config(self):
        """Test execution configuration."""
        config = UniversalPipelineConfig(
            execute_remotely=False,
            default_queue="custom-queue",
            gpu_queue="custom-gpu",
        )

        assert config.execute_remotely is False
        assert config.default_queue == "custom-queue"
        assert config.gpu_queue == "custom-gpu"


class TestUniversalDAPIPipelineController:
    """Tests for UniversalDAPIPipelineController."""

    def test_controller_creation(self):
        """Test controller creation with default config."""
        controller = UniversalDAPIPipelineController()

        assert controller.config is not None
        assert controller._pipeline is None

    def test_controller_with_config(self):
        """Test controller creation with custom config."""
        config = UniversalPipelineConfig(name="custom-pipeline")
        config.add_tissue("breast", dataset_id="abc123")

        controller = UniversalDAPIPipelineController(config)

        assert controller.config.name == "custom-pipeline"
        assert len(controller.config.tissues) == 1

    def test_create_universal_pipeline_factory(self):
        """Test factory function."""
        config = UniversalPipelineConfig(name="factory-test")
        config.add_tissue("breast", local_path="/data/breast")

        controller = create_universal_pipeline(config)

        assert isinstance(controller, UniversalDAPIPipelineController)
        assert controller.config.name == "factory-test"

    def test_create_pipeline_requires_tissues(self):
        """Test pipeline creation fails without tissues."""
        controller = UniversalDAPIPipelineController()

        with pytest.raises(ValueError, match="No tissue datasets configured"):
            controller.create_pipeline()

    def test_create_pipeline_structure(self):
        """Test pipeline structure is created correctly."""
        # Mock the PipelineController at import time
        with patch("clearml.PipelineController") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline

            config = (
                UniversalPipelineConfig()
                .add_tissue("breast", local_path="/data/breast")
                .add_tissue("lung", local_path="/data/lung")
            )
            controller = UniversalDAPIPipelineController(config)
            controller.create_pipeline()

            # Should have called add_step for each tissue's processing chain + final training
            # Per tissue: data_loader, segmentation, annotation, patch_extraction = 4 steps
            # Plus 1 final universal_training step
            # Total: 4 * 2 + 1 = 9 steps
            expected_steps = 2 * 4 + 1
            assert mock_pipeline.add_step.call_count == expected_steps

            # Check pipeline parameters were added
            assert mock_pipeline.add_parameter.call_count >= 3  # sampling, epochs, backbone

    def test_get_status_not_created(self):
        """Test status when pipeline not created."""
        controller = UniversalDAPIPipelineController()
        status = controller.get_status()

        assert status["status"] == "not_created"


class TestMultiTissueDatasetWithMockLMDB:
    """Tests for MultiTissueDataset with mock LMDB data."""

    @pytest.fixture
    def mock_lmdb_datasets(self, tmp_path):
        """Create mock LMDB datasets for testing."""
        datasets = []

        for i, (tissue, n_samples, n_classes) in enumerate([
            ("breast", 100, 5),
            ("lung", 80, 4),
        ]):
            ds_path = tmp_path / f"dataset_{tissue}"
            ds_path.mkdir()

            # Create LMDB with random patches
            lmdb_path = ds_path / "patches.lmdb"
            env = lmdb.open(str(lmdb_path), map_size=100 * 1024 * 1024)

            with env.begin(write=True) as txn:
                for j in range(n_samples):
                    key = f"{j:08d}".encode()
                    # Random 128x128 uint16 patch
                    patch = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
                    txn.put(key, patch.tobytes())

            env.close()

            # Create labels
            labels = np.random.randint(0, n_classes, n_samples)
            np.save(ds_path / "labels.npy", labels)

            # Create class mapping
            class_names = [f"CellType_{tissue}_{k}" for k in range(n_classes)]
            class_mapping = {name: idx for idx, name in enumerate(class_names)}
            with open(ds_path / "class_mapping.json", "w") as f:
                json.dump(class_mapping, f)

            datasets.append({
                "path": str(ds_path),
                "tissue": tissue,
                "n_samples": n_samples,
                "n_classes": n_classes,
            })

        return datasets

    def test_load_single_dataset(self, mock_lmdb_datasets):
        """Test loading a single mock dataset."""
        ds_info = mock_lmdb_datasets[0]
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
        )
        config.add_dataset(
            path=ds_info["path"],
            tissue=ds_info["tissue"],
        )

        # Mock the compute_dataset_stats to avoid actual computation
        with patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}

            dataset = MultiTissueDataset(config, split="train")

            assert len(dataset) == ds_info["n_samples"]
            assert dataset.num_classes == ds_info["n_classes"]

            dataset.close()

    def test_load_multiple_datasets(self, mock_lmdb_datasets):
        """Test loading multiple mock datasets."""
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
        )
        for ds_info in mock_lmdb_datasets:
            config.add_dataset(
                path=ds_info["path"],
                tissue=ds_info["tissue"],
            )

        with patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}

            dataset = MultiTissueDataset(config, split="train")

            # Total samples from both datasets
            expected_samples = sum(ds["n_samples"] for ds in mock_lmdb_datasets)
            assert len(dataset) == expected_samples

            # Classes are union of all classes
            assert dataset.num_classes == sum(ds["n_classes"] for ds in mock_lmdb_datasets)

            dataset.close()

    def test_getitem(self, mock_lmdb_datasets):
        """Test getting individual samples."""
        from unittest.mock import patch as mock_patch

        ds_info = mock_lmdb_datasets[0]
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
        )
        config.add_dataset(
            path=ds_info["path"],
            tissue=ds_info["tissue"],
        )

        with mock_patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}
            with mock_patch("dapidl.data.multi_tissue_dataset.get_train_transforms") as mock_transforms:
                # Simple transform that just returns a tensor
                def simple_transform(image):
                    return {"image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0)}
                mock_transforms.return_value = simple_transform

                dataset = MultiTissueDataset(config, split="train")

                # Get a sample
                image_patch, labels = dataset[0]

                assert isinstance(image_patch, torch.Tensor)
                assert image_patch.shape == (1, 128, 128)
                assert "label" in labels
                assert "tissue_idx" in labels
                assert "confidence_tier" in labels

                dataset.close()

    def test_class_weights(self, mock_lmdb_datasets):
        """Test class weight computation."""
        ds_info = mock_lmdb_datasets[0]
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
        )
        config.add_dataset(
            path=ds_info["path"],
            tissue=ds_info["tissue"],
        )

        with patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}

            dataset = MultiTissueDataset(config, split="train")

            weights = dataset.get_class_weights(max_weight_ratio=10.0)

            assert isinstance(weights, torch.Tensor)
            assert weights.shape == (dataset.num_classes,)
            assert (weights > 0).all()

            # Check max weight ratio constraint
            ratio = weights.max() / weights.min()
            assert ratio <= 10.0 + 0.01  # Small tolerance

            dataset.close()

    def test_sample_weights(self, mock_lmdb_datasets):
        """Test per-sample weight computation."""
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
            sampling_strategy="sqrt",
        )
        for ds_info in mock_lmdb_datasets:
            config.add_dataset(
                path=ds_info["path"],
                tissue=ds_info["tissue"],
                confidence_tier=2,
            )

        with patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}

            dataset = MultiTissueDataset(config, split="train")

            sample_weights = dataset.get_sample_weights()

            assert len(sample_weights) == len(dataset)
            assert (sample_weights > 0).all()

            dataset.close()

    def test_tissue_weights_equal(self, mock_lmdb_datasets):
        """Test equal sampling weights across tissues."""
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
            sampling_strategy="equal",
        )
        for ds_info in mock_lmdb_datasets:
            config.add_dataset(
                path=ds_info["path"],
                tissue=ds_info["tissue"],
            )

        with patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}

            dataset = MultiTissueDataset(config, split="train")
            tissue_weights = dataset._compute_tissue_weights()

            # For equal sampling, smaller datasets should have higher weights
            # to compensate for fewer samples
            assert len(tissue_weights) == 2

            dataset.close()

    def test_dataset_not_found_error(self, tmp_path):
        """Test error when LMDB not found."""
        config = MultiTissueConfig(use_hierarchical=False)
        config.add_dataset(
            path=str(tmp_path / "nonexistent"),
            tissue="test",
        )

        with pytest.raises(FileNotFoundError, match="LMDB not found"):
            MultiTissueDataset(config)


class TestMultiTissueSplits:
    """Tests for multi-tissue dataset splitting."""

    @pytest.fixture
    def simple_mock_dataset(self, tmp_path):
        """Create a simple mock dataset."""
        ds_path = tmp_path / "dataset"
        ds_path.mkdir()

        # Create minimal LMDB
        lmdb_path = ds_path / "patches.lmdb"
        env = lmdb.open(str(lmdb_path), map_size=50 * 1024 * 1024)
        n_samples = 100

        with env.begin(write=True) as txn:
            for j in range(n_samples):
                key = f"{j:08d}".encode()
                patch = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
                txn.put(key, patch.tobytes())
        env.close()

        # Labels with 5 classes, each with >= 20 samples
        labels = np.array([i % 5 for i in range(n_samples)])
        np.save(ds_path / "labels.npy", labels)

        class_mapping = {f"Type_{i}": i for i in range(5)}
        with open(ds_path / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f)

        return str(ds_path)

    def test_create_splits(self, simple_mock_dataset):
        """Test creating train/val/test splits."""
        config = MultiTissueConfig(
            use_hierarchical=False,
            standardize_labels=False,
            min_samples_per_class=5,  # Lower for test dataset
        )
        config.add_dataset(
            path=simple_mock_dataset,
            tissue="test",
        )

        with patch("dapidl.data.multi_tissue_dataset.compute_dataset_stats") as mock_stats:
            mock_stats.return_value = {"mean": 0.5, "std": 0.2, "p_low": 100, "p_high": 60000}

            train_ds, val_ds, test_ds = create_multi_tissue_splits(
                config,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            # Check split sizes roughly match ratios
            total = len(train_ds) + len(val_ds) + len(test_ds)
            assert 95 <= total <= 100  # Some samples may be filtered

            assert len(train_ds) > len(val_ds)
            assert len(train_ds) > len(test_ds)

            train_ds.close()
            val_ds.close()
            test_ds.close()


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_config_to_step_flow(self):
        """Test creating training step from pipeline config."""
        # Create pipeline config
        pipeline_config = (
            UniversalPipelineConfig(name="integration-test")
            .add_tissue("breast", local_path="/data/breast", confidence_tier=1)
            .add_tissue("lung", local_path="/data/lung", confidence_tier=2)
        )

        # Create training config from pipeline config
        train_config = UniversalTrainingConfig(
            epochs=pipeline_config.epochs,
            batch_size=pipeline_config.batch_size,
            backbone=pipeline_config.backbone,
            sampling_strategy=pipeline_config.sampling_strategy,
            tier1_weight=pipeline_config.tier1_weight,
            tier2_weight=pipeline_config.tier2_weight,
            tier3_weight=pipeline_config.tier3_weight,
            standardize_labels=pipeline_config.standardize_labels,
        )

        # Add datasets from pipeline tissues
        for tissue in pipeline_config.tissues:
            train_config.add_dataset(
                path=tissue.local_path or "",
                tissue=tissue.tissue,
                platform=tissue.platform,
                confidence_tier=tissue.confidence_tier,
                weight_multiplier=tissue.weight_multiplier,
            )

        # Create step
        step = UniversalDAPITrainingStep(train_config)

        assert len(step.config.datasets) == 2
        assert step.config.datasets[0].tissue == "breast"
        assert step.config.datasets[0].confidence_tier == 1
        assert step.config.datasets[1].tissue == "lung"
        assert step.config.datasets[1].confidence_tier == 2

    def test_multi_tissue_config_serialization(self):
        """Test that configs can be serialized and deserialized."""
        config = (
            UniversalTrainingConfig(
                epochs=50,
                backbone="resnet18",
                sampling_strategy="equal",
            )
            .add_dataset("/path/breast", "breast", "xenium", 1)
            .add_dataset("/path/lung", "lung", "merscope", 2)
        )

        # Serialize dataset specs
        specs_dict = [ds.to_dict() for ds in config.datasets]

        # Verify serialization
        assert len(specs_dict) == 2
        assert specs_dict[0]["tissue"] == "breast"
        assert specs_dict[1]["platform"] == "merscope"

        # Reconstruct from dict
        new_config = UniversalTrainingConfig()
        for d in specs_dict:
            new_config.add_dataset(
                path=d["path"],
                tissue=d["tissue"],
                platform=d["platform"],
                confidence_tier=d["confidence_tier"],
                weight_multiplier=d["weight_multiplier"],
            )

        assert len(new_config.datasets) == 2
        assert new_config.datasets[0].tissue == "breast"
        assert new_config.datasets[1].confidence_tier == 2
