#!/usr/bin/env python3
"""DAPIDL segmentation benchmark runner.

Evaluates one or more nucleus/cell segmentation methods on a set of
representative FOV tiles drawn from a MERSCOPE (or Xenium) dataset.

Usage examples::

    # Run all six methods with defaults
    uv run python scripts/run_segmentation_benchmark.py

    # Run a subset of methods
    uv run python scripts/run_segmentation_benchmark.py --methods cellpose_sam,stardist

    # Fewer FOVs, no consensus
    uv run python scripts/run_segmentation_benchmark.py --n-fovs 3 --no-consensus

    # Custom paths
    uv run python scripts/run_segmentation_benchmark.py \\
        --dapi-path /path/to/mosaic_DAPI_z3.tif \\
        --cell-metadata /path/to/cell_metadata.csv \\
        --output-dir ./my_benchmark_results
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

ALL_METHOD_NAMES = [
    "cellpose_sam",
    "cellpose3_cyto3",
    "cellpose3_nuclei",
    "stardist",
    "mesmer",
    "instanseg",
]


def get_adapter(name: str):
    """Return an instantiated SegmenterAdapter for the given method name.

    Returns None and logs a warning if the required library is not installed.

    Args:
        name: One of the supported method names (see ALL_METHOD_NAMES).

    Returns:
        A SegmenterAdapter instance, or None if the dependency is missing.
    """
    if name == "cellpose_sam":
        try:
            from dapidl.benchmark.segmenters.cellpose_adapter import CellposeSAMAdapter
            return CellposeSAMAdapter()
        except ImportError:
            logger.warning("cellpose not installed — skipping cellpose_sam")
            return None

    elif name == "cellpose_cyto3":
        try:
            from dapidl.benchmark.segmenters.cellpose_adapter import CellposeCyto3Adapter
            return CellposeCyto3Adapter()
        except ImportError:
            logger.warning("cellpose not installed — skipping cellpose_cyto3")
            return None

    elif name == "cellpose_nuclei":
        try:
            from dapidl.benchmark.segmenters.cellpose_adapter import CellposeNucleiAdapter
            return CellposeNucleiAdapter()
        except ImportError:
            logger.warning("cellpose not installed — skipping cellpose_nuclei")
            return None

    elif name == "stardist":
        try:
            from dapidl.benchmark.segmenters.stardist_adapter import StarDistAdapter
            return StarDistAdapter()
        except ImportError:
            logger.warning("stardist not installed — skipping stardist")
            return None

    elif name == "mesmer":
        try:
            from dapidl.benchmark.segmenters.mesmer_adapter import MesmerAdapter
            return MesmerAdapter()
        except ImportError:
            logger.warning("deepcell / mesmer not installed — skipping mesmer")
            return None

    elif name == "instanseg":
        try:
            from dapidl.benchmark.segmenters.instanseg_adapter import InstanSegAdapter
            return InstanSegAdapter()
        except ImportError:
            logger.warning("instanseg not installed — skipping instanseg")
            return None

    elif name == "cellpose3_cyto3":
        try:
            from dapidl.benchmark.segmenters.cellpose_adapter import CellposeCyto3CP3Adapter
            return CellposeCyto3CP3Adapter()
        except ImportError:
            logger.warning("cellpose not installed — skipping cellpose3_cyto3")
            return None

    elif name == "cellpose3_nuclei":
        try:
            from dapidl.benchmark.segmenters.cellpose_adapter import CellposeNucleiCP3Adapter
            return CellposeNucleiCP3Adapter()
        except ImportError:
            logger.warning("cellpose not installed — skipping cellpose3_nuclei")
            return None

    else:
        logger.warning("Unknown method name '%s' — skipping", name)
        return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DAPIDL segmentation benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHOD_NAMES),
        help="Comma-separated list of segmentation methods to evaluate.",
    )
    parser.add_argument(
        "--n-fovs",
        type=int,
        default=5,
        help="Number of representative FOVs to benchmark on.",
    )
    parser.add_argument(
        "--no-consensus",
        action="store_true",
        default=False,
        help="Disable consensus method evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_output/segmentation_benchmark",
        help="Directory where results and the report are written.",
    )
    parser.add_argument(
        "--dapi-path",
        default="/mnt/work/datasets/raw/merscope/merscope-breast/images/mosaic_DAPI_z3.tif",
        help="Path to the DAPI TIFF mosaic.",
    )
    parser.add_argument(
        "--cell-metadata",
        default="/mnt/work/datasets/raw/merscope/merscope-breast/cell_metadata.csv",
        help="Path to the cell metadata CSV.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)

    dapi_path = Path(args.dapi_path)
    cell_metadata_path = Path(args.cell_metadata)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not dapi_path.exists():
        logger.error("DAPI file not found: %s", dapi_path)
        return 1
    if not cell_metadata_path.exists():
        logger.error("Cell metadata not found: %s", cell_metadata_path)
        return 1

    # Try to load transform from adjacent CSV
    scale, offset_x, offset_y = 9.259259, 357.2, 2007.97
    transform_path = dapi_path.parent / "micron_to_mosaic_pixel_transform.csv"
    if transform_path.exists():
        from dapidl.benchmark.fov_selector import load_transform
        try:
            scale, offset_x, offset_y = load_transform(transform_path)
            logger.info(
                "Loaded transform from %s: scale=%.4f, offset_x=%.2f, offset_y=%.2f",
                transform_path, scale, offset_x, offset_y,
            )
        except Exception as exc:
            logger.warning("Could not load transform from %s: %s — using defaults", transform_path, exc)
    else:
        logger.info("No transform file found at %s — using defaults", transform_path)

    # Build adapter list
    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    segmenters = []
    for name in requested_methods:
        adapter = get_adapter(name)
        if adapter is not None:
            segmenters.append(adapter)

    if not segmenters:
        logger.error("No segmenters could be loaded. Install at least one segmentation library.")
        return 1

    logger.info("Running benchmark with methods: %s", [s.name for s in segmenters])
    logger.info("  n_fovs=%d  consensus=%s", args.n_fovs, not args.no_consensus)
    logger.info("  dapi_path=%s", dapi_path)
    logger.info("  cell_metadata=%s", cell_metadata_path)
    logger.info("  output_dir=%s", output_dir)

    from dapidl.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(
        dapi_path=dapi_path,
        cell_metadata_path=cell_metadata_path,
        output_dir=output_dir,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    report_path = runner.run(
        segmenters=segmenters,
        n_fovs=args.n_fovs,
        run_consensus=not args.no_consensus,
    )

    logger.info("Benchmark finished. Report written to: %s", report_path)
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
