# scripts/spatial_gnn_probe.py
"""Spatial-GNN probe driver: registry -> stage1 -> gate -> stage2 -> readout.
Run phases individually; GPU phases print nvidia-smi guidance first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")
OUT = Path("pipeline_output/spatial_gnn_probe_2026_06")


def phase_registry() -> None:
    from dapidl.graph.registry import build_spatial_registry
    OUT.mkdir(parents=True, exist_ok=True)
    reg = build_spatial_registry(LMDB_DIR)
    reg.write_parquet(OUT / "spatial_registry.parquet")
    logger.info(f"registry: {len(reg)} rows verified-aligned -> {OUT/'spatial_registry.parquet'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["registry", "embed", "stage1", "stage2"])
    args = ap.parse_args()
    {"registry": phase_registry}.get(args.phase, lambda: logger.error("phase not yet implemented"))()


if __name__ == "__main__":
    main()
