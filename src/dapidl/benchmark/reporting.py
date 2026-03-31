"""Reporting module for the DAPIDL segmentation benchmark.

Generates a markdown report with summary tables and comparison plots
from the collected benchmark metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serializer with numpy support
# ---------------------------------------------------------------------------


def _numpy_safe(obj: Any) -> Any:
    """JSON default handler that converts numpy scalars/arrays to Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Summary stats helpers
# ---------------------------------------------------------------------------


def _avg_metric(
    fov_data: dict[str, dict],
    category: str,
    key: str,
) -> float:
    """Return mean of fov_data[fov][category][key] across all FOVs."""
    values = []
    for fov_metrics in fov_data.values():
        cat = fov_metrics.get(category, {})
        if isinstance(cat, dict) and key in cat:
            values.append(float(cat[key]))
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _build_summary_table(all_metrics: dict[str, dict]) -> str:
    """Build the method-level summary table in markdown format."""
    header = (
        "| Method | Avg Cells | Avg Solidity | Avg Recovery | Avg Runtime (s) |\n"
        "|--------|----------:|-------------:|-------------:|----------------:|"
    )
    rows = [header]
    for method, fov_data in all_metrics.items():
        avg_cells = _avg_metric(fov_data, "morphometric", "n_detected")
        avg_solidity = _avg_metric(fov_data, "morphometric", "mean_solidity")
        avg_recovery = _avg_metric(fov_data, "biological", "native_recovery_rate")
        avg_runtime = _avg_metric(fov_data, "practical", "runtime_seconds")
        rows.append(
            f"| {method} | {avg_cells:.0f} | {avg_solidity:.3f} |"
            f" {avg_recovery:.3f} | {avg_runtime:.2f} |"
        )
    return "\n".join(rows)


def _build_per_method_tables(all_metrics: dict[str, dict]) -> str:
    """Build per-FOV breakdown tables for each method."""
    sections: list[str] = []
    for method, fov_data in all_metrics.items():
        lines = [f"### {method}", ""]
        header = (
            "| FOV | Cells | Solidity | Recovery | Runtime (s) |\n"
            "|-----|------:|---------:|---------:|------------:|"
        )
        lines.append(header)
        for fov_label, fov_metrics in sorted(fov_data.items()):
            morph = fov_metrics.get("morphometric", {})
            bio = fov_metrics.get("biological", {})
            practical = fov_metrics.get("practical", {})
            cells = morph.get("n_detected", 0)
            solidity = morph.get("mean_solidity", 0.0)
            recovery = bio.get("native_recovery_rate", 0.0)
            runtime = practical.get("runtime_seconds", 0.0)
            lines.append(
                f"| {fov_label} | {cells} | {solidity:.3f} |"
                f" {recovery:.3f} | {runtime:.2f} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def _generate_comparison_chart(
    all_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Create a 4-panel horizontal bar chart and save as PNG.

    Panels: avg cells detected, avg solidity, avg recovery rate, avg runtime.
    The best-performing method per panel is highlighted in green; others in
    steelblue.

    Args:
        all_metrics: Nested metrics dict as passed to generate_report.
        output_path: Destination path for the PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(all_metrics.keys())
    if not methods:
        return

    avg_cells = [_avg_metric(all_metrics[m], "morphometric", "n_detected") for m in methods]
    avg_solidity = [_avg_metric(all_metrics[m], "morphometric", "mean_solidity") for m in methods]
    avg_recovery = [_avg_metric(all_metrics[m], "biological", "native_recovery_rate") for m in methods]
    avg_runtime = [_avg_metric(all_metrics[m], "practical", "runtime_seconds") for m in methods]

    metrics_data = [
        ("Avg Cells Detected", avg_cells, True),   # True = higher is better
        ("Avg Solidity", avg_solidity, True),
        ("Avg Recovery Rate", avg_recovery, True),
        ("Avg Runtime (s)", avg_runtime, False),    # False = lower is better
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, max(4, len(methods) * 0.5 + 2)))
    fig.suptitle("Segmentation Method Comparison", fontsize=14, y=1.02)

    for ax, (title, values, higher_is_better) in zip(axes, metrics_data):
        if higher_is_better:
            best_idx = int(np.argmax(values)) if values else 0
        else:
            best_idx = int(np.argmin(values)) if values else 0

        colors = ["green" if i == best_idx else "steelblue" for i in range(len(methods))]
        y_pos = list(range(len(methods)))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()  # top method at top

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    all_metrics: dict[str, dict],
    output_dir: Path,
) -> Path:
    """Generate a markdown report and comparison plots from benchmark metrics.

    The ``all_metrics`` structure is::

        {
            method_name: {
                fov_label: {
                    "morphometric": {...},
                    "biological":   {...},
                    "practical":    {...},
                }
            }
        }

    Creates the following files inside *output_dir*:

    - ``report.md``  — summary table + per-FOV breakdown tables per method.
    - ``all_metrics.json``  — raw metrics (numpy-safe JSON).
    - ``comparison_chart.png``  — 4-panel horizontal bar chart.

    Args:
        all_metrics: Nested dict described above.
        output_dir: Directory where output files are written (created if absent).

    Returns:
        Path to the generated ``report.md``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Write raw JSON
    # ------------------------------------------------------------------
    json_path = output_dir / "all_metrics.json"
    with open(json_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=_numpy_safe)
    logger.info("Wrote metrics JSON to %s", json_path)

    # ------------------------------------------------------------------
    # 2. Generate comparison chart
    # ------------------------------------------------------------------
    chart_path = output_dir / "comparison_chart.png"
    try:
        _generate_comparison_chart(all_metrics, chart_path)
        logger.info("Wrote comparison chart to %s", chart_path)
        chart_line = f"![Comparison Chart](comparison_chart.png)\n"
    except ImportError:
        logger.warning("matplotlib not available — skipping comparison chart")
        chart_line = "_Chart not generated (matplotlib unavailable)._\n"

    # ------------------------------------------------------------------
    # 3. Build and write markdown report
    # ------------------------------------------------------------------
    report_path = output_dir / "report.md"

    n_methods = len(all_metrics)
    n_fovs = max((len(v) for v in all_metrics.values()), default=0)

    md_lines: list[str] = [
        "# Segmentation Benchmark Report",
        "",
        f"**Methods evaluated:** {n_methods}  ",
        f"**FOVs per method:** {n_fovs}",
        "",
        "## Summary",
        "",
        _build_summary_table(all_metrics),
        "",
        "## Comparison Chart",
        "",
        chart_line,
        "## Per-Method FOV Breakdown",
        "",
        _build_per_method_tables(all_metrics),
        "",
    ]

    report_path.write_text("\n".join(md_lines))
    logger.info("Wrote report to %s", report_path)

    return report_path
