"""Tests for hierarchical multi-head classification components.

Tests cover:
- HierarchyConfig and HierarchicalOutput dataclasses
- HierarchicalClassifier model architecture
- HierarchicalLoss with consistency penalty
- CurriculumScheduler for progressive training
- HierarchicalLabels label conversion
"""

import pytest
import torch
import numpy as np

from dapidl.models.hierarchical import (
    HierarchyConfig,
    HierarchicalOutput,
    HierarchicalClassifier,
)
from dapidl.training.hierarchical_loss import (
    HierarchicalLoss,
    CurriculumScheduler,
    get_hierarchical_class_weights,
)


class TestHierarchyConfig:
    """Tests for HierarchyConfig dataclass."""

    def test_basic_config(self):
        """Test basic hierarchy configuration."""
        config = HierarchyConfig(
            num_coarse=3,
            num_medium=10,
            num_fine=25,
            coarse_names=["Epithelial", "Immune", "Stromal"],
            medium_names=[f"Type_{i}" for i in range(10)],
            fine_to_coarse={i: i % 3 for i in range(25)},
            medium_to_coarse={i: i % 3 for i in range(10)},
        )

        assert config.num_coarse == 3
        assert config.num_medium == 10
        assert config.num_fine == 25
        assert len(config.coarse_names) == 3
        assert len(config.medium_names) == 10
        assert len(config.fine_to_coarse) == 25
        assert len(config.medium_to_coarse) == 10

    def test_consistency_mapping(self):
        """Test that fine-to-coarse mappings are consistent."""
        config = HierarchyConfig(
            num_coarse=3,
            num_medium=5,
            num_fine=15,
            coarse_names=["A", "B", "C"],
            medium_names=[f"M{i}" for i in range(5)],
            fine_to_coarse={i: i // 5 for i in range(15)},  # 0-4→0, 5-9→1, 10-14→2
            medium_to_coarse={i: i // 2 for i in range(5)},  # 0,1→0, 2,3→1, 4→2
        )

        # All fine indices should map to valid coarse indices
        for fine_idx, coarse_idx in config.fine_to_coarse.items():
            assert 0 <= fine_idx < config.num_fine
            assert 0 <= coarse_idx < config.num_coarse


class TestHierarchicalOutput:
    """Tests for HierarchicalOutput dataclass."""

    @pytest.fixture
    def hierarchy_config(self):
        """Create test hierarchy config."""
        return HierarchyConfig(
            num_coarse=3,
            num_medium=8,
            num_fine=20,
            coarse_names=["Epithelial", "Immune", "Stromal"],
            medium_names=[f"Type_{i}" for i in range(8)],
            fine_to_coarse={i: i % 3 for i in range(20)},
            medium_to_coarse={i: i % 3 for i in range(8)},
        )

    def test_output_creation(self, hierarchy_config):
        """Test output dataclass creation."""
        batch_size = 4
        coarse_logits = torch.randn(batch_size, 3)
        medium_logits = torch.randn(batch_size, 8)
        fine_logits = torch.randn(batch_size, 20)

        output = HierarchicalOutput(
            coarse_logits=coarse_logits,
            medium_logits=medium_logits,
            fine_logits=fine_logits,
        )

        assert output.coarse_logits.shape == (batch_size, 3)
        assert output.medium_logits.shape == (batch_size, 8)
        assert output.fine_logits.shape == (batch_size, 20)

    def test_probability_properties(self, hierarchy_config):
        """Test that probability properties sum to 1."""
        batch_size = 4
        output = HierarchicalOutput(
            coarse_logits=torch.randn(batch_size, 3),
            medium_logits=torch.randn(batch_size, 8),
            fine_logits=torch.randn(batch_size, 20),
        )

        # Check that probabilities sum to ~1
        assert torch.allclose(output.coarse_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(output.medium_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(output.fine_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_get_predictions_basic(self, hierarchy_config):
        """Test basic prediction extraction."""
        batch_size = 2
        # Make coarse logits have clear winner at index 1
        coarse_logits = torch.tensor([[-5.0, 10.0, -5.0], [-5.0, -5.0, 10.0]])

        output = HierarchicalOutput(
            coarse_logits=coarse_logits,
            medium_logits=None,
            fine_logits=None,
        )

        # Pass hierarchy_config to get_predictions
        preds, confs, level = output.get_predictions(
            hierarchy_config=hierarchy_config,
            fine_threshold=0.5,
        )

        assert preds.shape == (batch_size,)
        assert preds[0].item() == 1  # Immune
        assert preds[1].item() == 2  # Stromal


class TestHierarchicalClassifier:
    """Tests for HierarchicalClassifier model."""

    @pytest.fixture
    def hierarchy_config(self):
        """Create test hierarchy config."""
        return HierarchyConfig(
            num_coarse=3,
            num_medium=8,
            num_fine=20,
            coarse_names=["Epithelial", "Immune", "Stromal"],
            medium_names=[f"Type_{i}" for i in range(8)],
            fine_to_coarse={i: i % 3 for i in range(20)},
            medium_to_coarse={i: i % 3 for i in range(8)},
        )

    def test_model_creation(self, hierarchy_config):
        """Test model can be created."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",  # Use smaller backbone for tests
            pretrained=False,
            dropout=0.3,
        )

        assert model is not None
        assert model.hierarchy_config.num_coarse == 3
        assert model.hierarchy_config.num_medium == 8
        assert model.hierarchy_config.num_fine == 20

    def test_forward_pass(self, hierarchy_config):
        """Test forward pass produces correct output shapes."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",
            pretrained=False,
        )

        batch_size = 4
        x = torch.randn(batch_size, 1, 128, 128)  # Single-channel DAPI

        output = model(x)

        assert isinstance(output, HierarchicalOutput)
        assert output.coarse_logits.shape == (batch_size, 3)
        assert output.medium_logits.shape == (batch_size, 8)
        assert output.fine_logits.shape == (batch_size, 20)

    def test_active_heads(self, hierarchy_config):
        """Test setting active heads."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",
            pretrained=False,
        )

        # Set only coarse active
        model.set_active_heads({"coarse"})

        x = torch.randn(2, 1, 128, 128)
        output = model(x)

        assert output.coarse_logits is not None
        assert output.medium_logits is None
        assert output.fine_logits is None

        # Set coarse + medium active
        model.set_active_heads({"coarse", "medium"})
        output = model(x)

        assert output.coarse_logits is not None
        assert output.medium_logits is not None
        assert output.fine_logits is None

    def test_shared_projection(self, hierarchy_config):
        """Test with shared projection layer."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",
            pretrained=False,
            use_shared_projection=True,
            projection_dim=256,
        )

        x = torch.randn(2, 1, 128, 128)
        output = model(x)

        assert output.coarse_logits.shape == (2, 3)
        assert output.medium_logits.shape == (2, 8)
        assert output.fine_logits.shape == (2, 20)

    def test_save_load_checkpoint(self, hierarchy_config, tmp_path):
        """Test saving and loading model checkpoint."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",
            pretrained=False,
        )

        checkpoint_path = tmp_path / "model.pt"
        model.save_checkpoint(str(checkpoint_path), epoch=10)

        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_model = HierarchicalClassifier.from_checkpoint(str(checkpoint_path))

        assert loaded_model.hierarchy_config.num_coarse == model.hierarchy_config.num_coarse
        assert loaded_model.hierarchy_config.num_medium == model.hierarchy_config.num_medium
        assert loaded_model.hierarchy_config.num_fine == model.hierarchy_config.num_fine


class TestHierarchicalLoss:
    """Tests for HierarchicalLoss function."""

    @pytest.fixture
    def hierarchy_config(self):
        """Create test hierarchy config."""
        return HierarchyConfig(
            num_coarse=3,
            num_medium=8,
            num_fine=20,
            coarse_names=["Epithelial", "Immune", "Stromal"],
            medium_names=[f"Type_{i}" for i in range(8)],
            fine_to_coarse={i: i % 3 for i in range(20)},
            medium_to_coarse={i: i % 3 for i in range(8)},
        )

    def test_loss_creation(self, hierarchy_config):
        """Test loss function creation."""
        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            coarse_weight=1.0,
            medium_weight=0.5,
            fine_weight=0.3,
            consistency_weight=0.1,
        )

        assert loss_fn.coarse_weight == 1.0
        assert loss_fn.medium_weight == 0.5
        assert loss_fn.fine_weight == 0.3
        assert loss_fn.consistency_weight == 0.1

    def test_forward_coarse_only(self, hierarchy_config):
        """Test loss with coarse labels only."""
        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            consistency_weight=0.0,  # Disable for simpler test
        )

        batch_size = 4
        output = HierarchicalOutput(
            coarse_logits=torch.randn(batch_size, 3),
            medium_logits=None,
            fine_logits=None,
        )
        coarse_targets = torch.randint(0, 3, (batch_size,))

        loss, loss_dict = loss_fn(output, coarse_targets)

        assert loss.item() > 0
        assert "loss_coarse" in loss_dict
        assert "loss_total" in loss_dict

    def test_forward_all_levels(self, hierarchy_config):
        """Test loss with all hierarchy levels."""
        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            consistency_weight=0.1,
        )

        batch_size = 4
        output = HierarchicalOutput(
            coarse_logits=torch.randn(batch_size, 3),
            medium_logits=torch.randn(batch_size, 8),
            fine_logits=torch.randn(batch_size, 20),
        )
        coarse_targets = torch.randint(0, 3, (batch_size,))
        medium_targets = torch.randint(0, 8, (batch_size,))
        fine_targets = torch.randint(0, 20, (batch_size,))

        loss, loss_dict = loss_fn(output, coarse_targets, medium_targets, fine_targets)

        assert loss.item() > 0
        assert "loss_coarse" in loss_dict
        assert "loss_medium" in loss_dict
        assert "loss_fine" in loss_dict
        assert "loss_consistency" in loss_dict
        assert "loss_total" in loss_dict

    def test_focal_loss(self, hierarchy_config):
        """Test with focal loss enabled."""
        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            use_focal=True,
            focal_gamma=2.0,
        )

        batch_size = 4
        output = HierarchicalOutput(
            coarse_logits=torch.randn(batch_size, 3),
            medium_logits=None,
            fine_logits=None,
        )
        coarse_targets = torch.randint(0, 3, (batch_size,))

        loss, loss_dict = loss_fn(output, coarse_targets)

        assert loss.item() > 0

    def test_class_weights(self, hierarchy_config):
        """Test with class weights."""
        coarse_weights = torch.tensor([1.0, 2.0, 3.0])

        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            coarse_class_weights=coarse_weights,
        )

        batch_size = 4
        output = HierarchicalOutput(
            coarse_logits=torch.randn(batch_size, 3),
            medium_logits=None,
            fine_logits=None,
        )
        coarse_targets = torch.randint(0, 3, (batch_size,))

        loss, _ = loss_fn(output, coarse_targets)

        assert loss.item() > 0


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_default_schedule(self):
        """Test default curriculum schedule."""
        scheduler = CurriculumScheduler(
            coarse_only_epochs=20,
            coarse_medium_epochs=50,
            warmup_epochs=5,
        )

        # Phase 1: Coarse only
        assert scheduler.get_active_heads(1) == {"coarse"}
        assert scheduler.get_active_heads(10) == {"coarse"}
        assert scheduler.get_active_heads(20) == {"coarse"}

        # Phase 2: Coarse + Medium
        assert scheduler.get_active_heads(21) == {"coarse", "medium"}
        assert scheduler.get_active_heads(35) == {"coarse", "medium"}
        assert scheduler.get_active_heads(50) == {"coarse", "medium"}

        # Phase 3: All heads
        assert scheduler.get_active_heads(51) == {"coarse", "medium", "fine"}
        assert scheduler.get_active_heads(100) == {"coarse", "medium", "fine"}

    def test_loss_weights_warmup(self):
        """Test loss weight warmup during phase transitions."""
        scheduler = CurriculumScheduler(
            coarse_only_epochs=20,
            coarse_medium_epochs=50,
            warmup_epochs=5,
        )

        base_weights = (1.0, 0.5, 0.3, 0.1)

        # Phase 1: Only coarse weight should be non-zero
        weights = scheduler.get_loss_weights(10, base_weights)
        assert weights["coarse_weight"] == 1.0
        assert weights["medium_weight"] == 0.0
        assert weights["fine_weight"] == 0.0

        # Phase 2 start: Medium should warm up
        weights = scheduler.get_loss_weights(21, base_weights)
        assert weights["coarse_weight"] == 1.0
        assert weights["medium_weight"] > 0  # Starting warmup
        assert weights["medium_weight"] < 0.5  # Not fully warmed up

        # Phase 2 after warmup
        weights = scheduler.get_loss_weights(30, base_weights)
        assert weights["medium_weight"] == 0.5  # Fully warmed up

        # Phase 3 start
        weights = scheduler.get_loss_weights(51, base_weights)
        assert weights["fine_weight"] > 0
        assert weights["fine_weight"] < 0.3  # Starting warmup

    def test_phase_names(self):
        """Test phase name generation."""
        scheduler = CurriculumScheduler(
            coarse_only_epochs=20,
            coarse_medium_epochs=50,
        )

        assert "Coarse Only" in scheduler.get_phase_name(10)
        assert "Medium" in scheduler.get_phase_name(30)
        assert "All" in scheduler.get_phase_name(60)


class TestGetHierarchicalClassWeights:
    """Tests for get_hierarchical_class_weights function."""

    def test_inverse_weights(self):
        """Test inverse frequency weights."""
        labels = [0, 0, 0, 1, 1, 2]  # Imbalanced: 3, 2, 1

        weights = get_hierarchical_class_weights(labels, num_classes=3, method="inverse")

        # Most frequent class should have lowest weight
        assert weights[0] < weights[1]
        assert weights[1] < weights[2]

    def test_inverse_sqrt_weights(self):
        """Test inverse sqrt frequency weights."""
        labels = [0] * 100 + [1] * 10 + [2]  # Very imbalanced

        weights = get_hierarchical_class_weights(
            labels, num_classes=3, method="inverse_sqrt"
        )

        # Weights should be moderated compared to pure inverse
        assert weights[0] < weights[1] < weights[2]

    def test_max_weight_ratio(self):
        """Test maximum weight ratio capping."""
        labels = [0] * 100 + [1]  # 100:1 ratio

        weights = get_hierarchical_class_weights(
            labels, num_classes=2, method="inverse", max_weight_ratio=5.0
        )

        # Weight ratio should be capped at 5
        ratio = weights[1] / weights[0]
        assert ratio <= 5.0 + 0.01  # Small tolerance

    def test_tensor_input(self):
        """Test with tensor input."""
        labels = torch.tensor([0, 0, 1, 1, 2])

        weights = get_hierarchical_class_weights(labels, num_classes=3)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.fixture
    def hierarchy_config(self):
        """Create test hierarchy config."""
        return HierarchyConfig(
            num_coarse=3,
            num_medium=6,
            num_fine=12,
            coarse_names=["Epithelial", "Immune", "Stromal"],
            medium_names=[f"M{i}" for i in range(6)],
            fine_to_coarse={i: i % 3 for i in range(12)},
            medium_to_coarse={i: i % 3 for i in range(6)},
        )

    def test_forward_backward(self, hierarchy_config):
        """Test complete forward-backward pass."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",
            pretrained=False,
        )

        loss_fn = HierarchicalLoss(
            hierarchy_config=hierarchy_config,
            coarse_weight=1.0,
            medium_weight=0.5,
            fine_weight=0.3,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Forward pass
        x = torch.randn(2, 1, 64, 64)
        output = model(x)

        coarse_targets = torch.randint(0, 3, (2,))
        medium_targets = torch.randint(0, 6, (2,))
        fine_targets = torch.randint(0, 12, (2,))

        loss, loss_dict = loss_fn(output, coarse_targets, medium_targets, fine_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients were computed
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_curriculum_training_simulation(self, hierarchy_config):
        """Simulate curriculum training across phases."""
        model = HierarchicalClassifier(
            hierarchy_config=hierarchy_config,
            backbone_name="resnet18",
            pretrained=False,
        )

        loss_fn = HierarchicalLoss(hierarchy_config=hierarchy_config)
        scheduler = CurriculumScheduler(
            coarse_only_epochs=5,
            coarse_medium_epochs=10,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        x = torch.randn(2, 1, 64, 64)
        coarse_targets = torch.randint(0, 3, (2,))
        medium_targets = torch.randint(0, 6, (2,))
        fine_targets = torch.randint(0, 12, (2,))

        # Train for 3 epochs in each phase
        for epoch in [1, 6, 11]:
            active_heads = scheduler.get_active_heads(epoch)
            model.set_active_heads(active_heads)

            # Update loss weights
            weights = scheduler.get_loss_weights(epoch)
            loss_fn.coarse_weight = weights["coarse_weight"]
            loss_fn.medium_weight = weights["medium_weight"]
            loss_fn.fine_weight = weights["fine_weight"]

            # Forward-backward
            output = model(x)

            # Only pass targets for active heads
            medium = medium_targets if "medium" in active_heads else None
            fine = fine_targets if "fine" in active_heads else None

            loss, _ = loss_fn(output, coarse_targets, medium, fine)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
