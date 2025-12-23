"""Swappable pipeline components.

Components are registered with the pipeline registry and can be
selected at runtime via configuration:

Segmenters (nucleus detection):
- cellpose: Deep learning-based segmentation using Cellpose
- native: Pass-through using existing platform boundaries

Annotators (cell type labeling):
- celltypist: Gene expression-based classification
- ground_truth: Load from curated Excel/CSV files
- popv: Voting-based annotation ensemble
"""

# Import subpackages to trigger registration
from dapidl.pipeline.components import annotators, segmenters

__all__ = ["segmenters", "annotators"]
