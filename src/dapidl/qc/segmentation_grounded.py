"""DEPRECATED shim — moved to starpose.qc.segmentation_grounded (starpose 0.3.0).

Kept so existing dapidl imports keep working; import from starpose.qc directly
in new code. Removed in a later cleanup.
"""
from starpose.qc.segmentation_grounded import *  # noqa: F401,F403
from starpose.qc.segmentation_grounded import __all__  # noqa: F401
