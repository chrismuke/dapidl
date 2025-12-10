"""Base protocol for experiment trackers.

This module defines the interface that all experiment tracking backends must implement.
This enables easy migration between W&B, ClearML, ZenML, MLflow, etc.

Design Philosophy:
------------------
1. **Protocol-based**: Use Python Protocol for duck typing, no inheritance required
2. **Minimal interface**: Only essential methods, backends can extend
3. **Backend-agnostic data**: Use standard Python types (dict, Path, etc.)
4. **Lifecycle methods**: init -> log -> finish pattern

Migration Guide:
----------------
To add a new tracking backend (e.g., ClearML):

1. Create tracking/backends/clearml.py
2. Implement the ExperimentTracker protocol
3. Register in tracking/factory.py
4. Users select backend via config or CLI flag

Example implementation:
    class ClearMLTracker:
        def init(self, config, reproducibility):
            from clearml import Task
            self.task = Task.init(...)
            self.task.connect(config)

        def log_metrics(self, metrics, step=None):
            for k, v in metrics.items():
                self.task.get_logger().report_scalar(k, k, v, step)
        ...
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dapidl.tracking.reproducibility import ReproducibilityInfo


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol for experiment tracking backends.

    All tracking backends (W&B, ClearML, ZenML, MLflow) should implement
    this interface. This enables swapping backends without changing training code.

    Lifecycle:
        1. init() - Start experiment, log config and reproducibility
        2. log_*() - Log metrics, artifacts, visualizations during training
        3. finish() - Finalize experiment, cleanup

    Example usage:
        tracker = WandbTracker()  # or ClearMLTracker(), ZenMLTracker()
        tracker.init(config, reproducibility_info)

        for epoch in range(epochs):
            metrics = train_epoch()
            tracker.log_metrics(metrics, step=epoch)

        tracker.log_artifact(model_path, "model", "model")
        tracker.finish()
    """

    def init(
        self,
        project: str,
        config: dict[str, Any],
        reproducibility: ReproducibilityInfo | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize experiment tracking.

        Args:
            project: Project/workspace name
            config: Hyperparameters and configuration
            reproducibility: Code/env/data reproducibility info
            tags: Optional tags for filtering runs
            name: Optional run name (auto-generated if not provided)
        """
        ...

    def log_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """Log scalar metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/epoch number
        """
        ...

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Log a file or directory as an artifact.

        Args:
            path: Path to file or directory
            name: Artifact name
            artifact_type: Type (e.g., "model", "dataset", "code")
            metadata: Optional metadata to attach

        Returns:
            Artifact ID/URI if available
        """
        ...

    def log_image(
        self,
        image: Any,  # PIL Image, matplotlib Figure, numpy array, or path
        name: str,
        step: int | None = None,
    ) -> None:
        """Log an image.

        Args:
            image: Image data (PIL, matplotlib, numpy, or path)
            name: Image name/key
            step: Optional step number
        """
        ...

    def log_table(
        self,
        data: list[list[Any]],
        columns: list[str],
        name: str,
    ) -> None:
        """Log tabular data.

        Args:
            data: List of rows
            columns: Column names
            name: Table name/key
        """
        ...

    def log_text(
        self,
        text: str,
        name: str,
    ) -> None:
        """Log text content (e.g., git diff, config dump).

        Args:
            text: Text content
            name: Name/key for the text
        """
        ...

    def set_tags(self, tags: list[str]) -> None:
        """Add tags to the current run.

        Args:
            tags: List of tag strings
        """
        ...

    def finish(self, status: str = "success") -> None:
        """Finalize the experiment.

        Args:
            status: Final status ("success", "failed", "aborted")
        """
        ...

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        ...

    @property
    def run_url(self) -> str | None:
        """Get URL to view the run (if available)."""
        ...


class NullTracker:
    """No-op tracker for when tracking is disabled.

    This implements the ExperimentTracker protocol but does nothing.
    Useful for local debugging or when --no-wandb is set.
    """

    def init(self, project, config, reproducibility=None, tags=None, name=None):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path, name, artifact_type, metadata=None):
        return None

    def log_image(self, image, name, step=None):
        pass

    def log_table(self, data, columns, name):
        pass

    def log_text(self, text, name):
        pass

    def set_tags(self, tags):
        pass

    def finish(self, status="success"):
        pass

    @property
    def run_id(self):
        return None

    @property
    def run_url(self):
        return None
