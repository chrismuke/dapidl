"""Documentation Generation Step.

Generates markdown documentation from pipeline results and optionally
exports to an Obsidian vault for knowledge management.

Key features:
- Auto-generates experiment summaries
- Creates per-class performance breakdowns
- Links related experiments
- Supports Obsidian-specific formatting (tags, links, callouts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""

    # Output settings
    output_dir: str = ""  # Local output directory
    obsidian_vault_path: str = ""  # Path to Obsidian vault (e.g., ~/obsidian/llmbrain)
    obsidian_folder: str = "DAPIDL"  # Subfolder within vault

    # Document settings
    experiment_name: str = ""
    include_metrics: bool = True
    include_config: bool = True
    include_class_breakdown: bool = True
    include_training_curves: bool = False  # Requires matplotlib

    # Obsidian-specific
    use_obsidian_callouts: bool = True
    add_tags: bool = True
    default_tags: list[str] = field(default_factory=lambda: ["dapidl", "experiment"])

    # Template customization
    template: str = "default"  # "default", "minimal", "detailed"


class DocumentationStep(PipelineStep):
    """Generate documentation from pipeline results.

    Creates markdown documentation summarizing:
    - Training configuration
    - Model performance metrics
    - Per-class breakdowns
    - Cross-platform transfer results (if available)
    - Validation results

    Optionally exports to an Obsidian vault for knowledge management.

    Usage:
        config = DocumentationConfig(
            obsidian_vault_path="/home/user/obsidian/brain",
            experiment_name="xenium-breast-v1",
        )
        step = DocumentationStep(config)
        results = step.execute(artifacts)
    """

    name = "documentation"
    description = "Generate experiment documentation"

    def __init__(self, config: DocumentationConfig | None = None):
        self.config = config or DocumentationConfig()

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Local output directory for documentation",
                },
                "obsidian_vault_path": {
                    "type": "string",
                    "description": "Path to Obsidian vault (leave empty to skip)",
                },
                "obsidian_folder": {
                    "type": "string",
                    "default": "DAPIDL",
                    "description": "Subfolder within Obsidian vault",
                },
                "experiment_name": {
                    "type": "string",
                    "description": "Name for this experiment",
                },
                "include_metrics": {
                    "type": "boolean",
                    "default": True,
                },
                "include_config": {
                    "type": "boolean",
                    "default": True,
                },
                "include_class_breakdown": {
                    "type": "boolean",
                    "default": True,
                },
                "template": {
                    "type": "string",
                    "enum": ["default", "minimal", "detailed"],
                    "default": "default",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate inputs - documentation can work with minimal info."""
        return True

    def execute(self, artifacts: StepArtifacts, **params: Any) -> StepArtifacts:
        """Generate documentation from pipeline artifacts."""
        cfg = self.config

        # Collect all available information
        doc_data = self._collect_data(artifacts)

        # Generate markdown content
        if cfg.template == "minimal":
            content = self._generate_minimal(doc_data)
        elif cfg.template == "detailed":
            content = self._generate_detailed(doc_data)
        else:
            content = self._generate_default(doc_data)

        # Determine experiment name
        exp_name = cfg.experiment_name or doc_data.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exp_name}_{timestamp}.md"

        # Save locally
        if cfg.output_dir:
            output_path = Path(cfg.output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            logger.info(f"Saved documentation to: {output_path}")
            artifacts.set_output("doc_local_path", str(output_path))

        # Export to Obsidian
        if cfg.obsidian_vault_path:
            obsidian_path = self._export_to_obsidian(content, exp_name, doc_data)
            if obsidian_path:
                logger.info(f"Exported to Obsidian: {obsidian_path}")
                artifacts.set_output("doc_obsidian_path", str(obsidian_path))

        artifacts.set_output("documentation_content", content)
        artifacts.set_output("experiment_name", exp_name)

        return artifacts

    def _collect_data(self, artifacts: StepArtifacts) -> dict[str, Any]:
        """Collect all relevant data from artifacts."""
        data = {}

        # Training metrics
        if artifacts.get_input("test_metrics"):
            data["test_metrics"] = artifacts.get_input("test_metrics")
        if artifacts.get_input("training_history"):
            data["training_history"] = artifacts.get_input("training_history")

        # Model info
        data["model_path"] = artifacts.get_input("model_path", "")
        data["backbone"] = artifacts.get_input("backbone", "efficientnetv2_rw_s")
        data["num_classes"] = artifacts.get_input("num_classes", 0)
        data["class_names"] = artifacts.get_input("class_names", [])

        # Dataset info
        data["dataset_path"] = artifacts.get_input("dataset_path", "")
        data["platform"] = artifacts.get_input("platform", "unknown")
        data["patch_size"] = artifacts.get_input("patch_size", 128)

        # Transfer results
        if artifacts.get_input("transfer_metrics"):
            data["transfer_metrics"] = artifacts.get_input("transfer_metrics")

        # Validation results
        if artifacts.get_input("validation_results"):
            data["validation_results"] = artifacts.get_input("validation_results")

        # Cross-modal validation
        if artifacts.get_input("cross_modal_validation"):
            data["cross_modal_validation"] = artifacts.get_input("cross_modal_validation")

        # Experiment name from path
        if data["model_path"]:
            model_path = Path(data["model_path"])
            if "experiment_" in str(model_path):
                parts = str(model_path).split("experiment_")
                if len(parts) > 1:
                    exp_part = parts[1].split("/")[0]
                    data["experiment_name"] = f"experiment_{exp_part}"

        return data

    def _generate_default(self, data: dict[str, Any]) -> str:
        """Generate default markdown documentation."""
        cfg = self.config
        lines = []

        # Title and metadata
        exp_name = data.get("experiment_name", "DAPIDL Experiment")
        lines.append(f"# {exp_name}")
        lines.append("")

        # Tags for Obsidian
        if cfg.add_tags:
            tags = cfg.default_tags.copy()
            if data.get("platform"):
                tags.append(data["platform"])
            lines.append(f"tags: {', '.join(tags)}")
            lines.append("")

        # Date
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Summary callout
        if cfg.use_obsidian_callouts and data.get("test_metrics"):
            tm = data["test_metrics"]
            lines.append("> [!summary] Results Summary")
            lines.append(f"> - **Test F1:** {tm.get('f1', 'N/A'):.4f}" if isinstance(tm.get('f1'), float) else f"> - **Test F1:** {tm.get('f1', 'N/A')}")
            lines.append(f"> - **Accuracy:** {tm.get('accuracy', 'N/A'):.4f}" if isinstance(tm.get('accuracy'), float) else f"> - **Accuracy:** {tm.get('accuracy', 'N/A')}")
            lines.append(f"> - **Platform:** {data.get('platform', 'unknown')}")
            lines.append(f"> - **Backbone:** {data.get('backbone', 'unknown')}")
            lines.append("")

        # Configuration
        if cfg.include_config:
            lines.append("## Configuration")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Platform | {data.get('platform', 'unknown')} |")
            lines.append(f"| Backbone | {data.get('backbone', 'unknown')} |")
            lines.append(f"| Patch Size | {data.get('patch_size', 128)} |")
            lines.append(f"| Num Classes | {data.get('num_classes', 'unknown')} |")
            if data.get("class_names"):
                lines.append(f"| Classes | {', '.join(data['class_names'])} |")
            lines.append("")

        # Metrics
        if cfg.include_metrics and data.get("test_metrics"):
            lines.append("## Performance Metrics")
            lines.append("")
            tm = data["test_metrics"]

            lines.append("### Overall")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key in ["accuracy", "f1", "precision", "recall"]:
                if key in tm:
                    val = tm[key]
                    if isinstance(val, float):
                        lines.append(f"| {key.title()} | {val:.4f} |")
                    else:
                        lines.append(f"| {key.title()} | {val} |")
            lines.append("")

        # Per-class breakdown
        if cfg.include_class_breakdown and data.get("test_metrics"):
            tm = data["test_metrics"]
            if "classification_report" in tm:
                lines.append("### Per-Class Performance")
                lines.append("")
                lines.append("| Class | Precision | Recall | F1-Score | Support |")
                lines.append("|-------|-----------|--------|----------|---------|")

                report = tm["classification_report"]
                class_names = data.get("class_names", [])

                for name in class_names:
                    if name in report:
                        r = report[name]
                        lines.append(
                            f"| {name} | {r.get('precision', 0):.3f} | "
                            f"{r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | "
                            f"{int(r.get('support', 0))} |"
                        )
                lines.append("")

        # Training history
        if data.get("training_history"):
            history = data["training_history"]
            lines.append("## Training History")
            lines.append("")
            lines.append(f"- **Epochs:** {len(history.get('train_losses', []))}")
            lines.append(f"- **Best Epoch:** {history.get('best_epoch', 'N/A')}")
            if isinstance(history.get('best_val_f1'), float):
                lines.append(f"- **Best Val F1:** {history['best_val_f1']:.4f}")
            lines.append("")

        # Transfer results
        if data.get("transfer_metrics"):
            tm = data["transfer_metrics"]
            lines.append("## Cross-Platform Transfer")
            lines.append("")

            if cfg.use_obsidian_callouts:
                lines.append("> [!info] Transfer Testing")
                lines.append(f"> - **Source:** {tm.get('source_platform', 'unknown')}")
                lines.append(f"> - **Target:** {tm.get('target_platform', 'unknown')}")
                lines.append(f"> - **Macro F1:** {tm.get('macro_f1', 'N/A'):.4f}" if isinstance(tm.get('macro_f1'), float) else f"> - **Macro F1:** {tm.get('macro_f1', 'N/A')}")
                lines.append(f"> - **Accuracy:** {tm.get('accuracy', 'N/A'):.4f}" if isinstance(tm.get('accuracy'), float) else f"> - **Accuracy:** {tm.get('accuracy', 'N/A')}")
            else:
                lines.append(f"- **Source:** {tm.get('source_platform', 'unknown')}")
                lines.append(f"- **Target:** {tm.get('target_platform', 'unknown')}")
                lines.append(f"- **Macro F1:** {tm.get('macro_f1', 'N/A')}")
                lines.append(f"- **Accuracy:** {tm.get('accuracy', 'N/A')}")
            lines.append("")

            # Baseline comparison
            if "baseline_comparison" in tm:
                lines.append("### Baseline Comparison")
                lines.append("")
                lines.append("| Metric | Transfer | Baseline | Delta | Retention |")
                lines.append("|--------|----------|----------|-------|-----------|")
                for key, vals in tm["baseline_comparison"].items():
                    lines.append(
                        f"| {key} | {vals['transfer']:.4f} | {vals['baseline']:.4f} | "
                        f"{vals['delta']:+.4f} | {vals['retention']:.1%} |"
                    )
                lines.append("")

        # Cross-modal validation
        if data.get("cross_modal_validation"):
            cmv = data["cross_modal_validation"]
            lines.append("## Cross-Modal Validation")
            lines.append("")

            if cmv.get("leiden_ari"):
                lines.append(f"- **Leiden ARI:** {cmv['leiden_ari']:.4f}")
            if cmv.get("dapi_agreement"):
                lines.append(f"- **DAPI Agreement:** {cmv['dapi_agreement']:.4f}")
            if cmv.get("consensus_agreement"):
                lines.append(f"- **Consensus Agreement:** {cmv['consensus_agreement']:.4f}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by DAPIDL Pipeline on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def _generate_minimal(self, data: dict[str, Any]) -> str:
        """Generate minimal documentation."""
        lines = []
        exp_name = data.get("experiment_name", "Experiment")

        lines.append(f"# {exp_name}")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"**Platform:** {data.get('platform', 'unknown')}")
        lines.append("")

        if data.get("test_metrics"):
            tm = data["test_metrics"]
            lines.append("## Results")
            lines.append(f"- F1: {tm.get('f1', 'N/A')}")
            lines.append(f"- Accuracy: {tm.get('accuracy', 'N/A')}")

        return "\n".join(lines)

    def _generate_detailed(self, data: dict[str, Any]) -> str:
        """Generate detailed documentation with all available info."""
        # Start with default
        content = self._generate_default(data)

        lines = content.split("\n")

        # Add raw JSON section
        lines.append("")
        lines.append("## Raw Data")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data, indent=2, default=str))
        lines.append("```")

        return "\n".join(lines)

    def _export_to_obsidian(
        self,
        content: str,
        exp_name: str,
        data: dict[str, Any],
    ) -> Path | None:
        """Export documentation to Obsidian vault."""
        cfg = self.config

        vault_path = Path(cfg.obsidian_vault_path).expanduser()
        if not vault_path.exists():
            logger.warning(f"Obsidian vault not found: {vault_path}")
            return None

        # Create DAPIDL folder if needed
        dapidl_folder = vault_path / cfg.obsidian_folder
        dapidl_folder.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d")
        safe_name = exp_name.replace(" ", "-").replace("/", "-")
        filename = f"{safe_name}-{timestamp}.md"

        # Add Obsidian-specific frontmatter
        frontmatter = [
            "---",
            f"title: {exp_name}",
            f"date: {datetime.now().strftime('%Y-%m-%d')}",
            f"platform: {data.get('platform', 'unknown')}",
        ]

        if data.get("test_metrics"):
            tm = data["test_metrics"]
            if isinstance(tm.get("f1"), float):
                frontmatter.append(f"f1_score: {tm['f1']:.4f}")
            if isinstance(tm.get("accuracy"), float):
                frontmatter.append(f"accuracy: {tm['accuracy']:.4f}")

        if cfg.add_tags:
            tags = cfg.default_tags.copy()
            if data.get("platform"):
                tags.append(data["platform"])
            frontmatter.append(f"tags: [{', '.join(tags)}]")

        frontmatter.append("---")
        frontmatter.append("")

        # Combine frontmatter with content
        full_content = "\n".join(frontmatter) + content

        # Write file
        output_path = dapidl_folder / filename
        output_path.write_text(full_content)

        return output_path

    def get_queue(self) -> str:
        """CPU queue for documentation."""
        return "default"

    def create_clearml_task(self, project: str, task_name: str):
        """Create a ClearML task for this step."""
        from clearml import Task

        task = Task.init(
            project_name=project,
            task_name=task_name,
            task_type=Task.TaskTypes.data_processing,
        )

        # Connect parameters
        task.connect(self.config, name="step_config")

        return task
