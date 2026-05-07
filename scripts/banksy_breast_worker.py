"""Single-slide BANKSY+scType worker — subprocess-isolated.

Runs BANKSY spatial clustering on ONE breast slide, then labels each cluster
via scType custom_default markers. Writes predictions to a JSON file that
the parent orchestrator picks up.

Why subprocess isolation? In the 2026-05-04 annotation run, BANKSY died
silently in `initialize_banksy` when run alongside training (no traceback,
RSS jump but no segfault). Hypothesis: TensorFlow / scikit-learn /
banksy_py's umap_learn pull in C-libraries (libgomp, libstdcxx) that
collide with torch's at fork time. Running each slide in a fresh python
process avoids any cross-slide state pollution.

Memory guard: BANKSY allocates ~3 dense neighborhood-aggregated copies of
the HVG matrix on top of the raw adata, so RAM scales as
n_cells × (n_genes + 4 × n_hvg) × 4 bytes. breast_s6 (692k × 8232) needs
~42 GB without subsampling, which OOM-killed the parent tmux scope on
2026-05-05 01:08. The worker now estimates RAM up front and either
auto-subsamples (when --max-cells / BANKSY_MAX_CELLS is set) or aborts
with exit 3 before the expensive `initialize_banksy` call.

Usage:
    uv run python scripts/banksy_breast_worker.py <slide> <out_json> \\
        [--max-cells N] [--max-mem-gb GB]
        slide  ∈ {rep1, rep2, breast_s0, breast_s1, breast_s3, breast_s6}
        out_json  e.g. pipeline_output/annotation_run_2026_05/per_slide/rep1_banksy.json
        --max-cells N    subsample to N cells if estimate exceeds budget (default: env BANKSY_MAX_CELLS)
        --max-mem-gb GB  RAM budget for BANKSY itself (default: env BANKSY_MAX_GB or 25)

Exit codes:
    0 = success
    1 = BANKSY ran but produced no predictions (logic / library error)
    2 = bad CLI args
    3 = pre-flight: estimated RAM exceeds budget and no --max-cells fallback set
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

# Limit thread fan-out so BANKSY's leiden + PCA don't fight scType / scanpy on the same cores
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import numpy as np
from loguru import logger

from annotation_benchmark_2026_03 import (
    load_xenium_adata, load_sthelar_adata, preprocess_adata,
    get_default_markers,
)
from annotation_run_2026_05 import run_banksy_sctype


# BANKSY uses n_top_genes = min(2000, n_vars) — see annotation_run_2026_05.py:203
BANKSY_N_HVG = 2000
# Empirical multiplier: raw adata copy + HVG slice + ~3 neighborhood-aggregated
# dense decay copies + PCA scratch, all float32. Calibrated against the
# 2026-05-05 OOM event: breast_s6 (692k × 8232) hit ~42 GB before getting killed.
BANKSY_RAM_OVERHEAD = 1.2


def estimate_banksy_ram_gb(n_cells: int, n_genes: int,
                            n_hvg: int = BANKSY_N_HVG) -> float:
    """Estimate peak RAM (in GB) for run_banksy_sctype on an adata of this shape.

    Memory profile (float32 = 4 bytes/cell):
      - raw adata kept loaded throughout: n_cells × n_genes
      - HVG slice (.copy()):              n_cells × n_hvg
      - 3 decay-type AGG matrices, dense: 3 × n_cells × n_hvg
      - PCA + leiden + scratch:           ~20% overhead

    Calibration: breast_s6 (692_184 × 8232) → 42 GB observed before OOM,
    estimate 41.4 GB at default constants. Within ±5% — good enough.
    """
    cells_x_genes_bytes = n_cells * (n_genes + 4 * n_hvg) * 4
    return cells_x_genes_bytes * BANKSY_RAM_OVERHEAD / 1e9


def maybe_subsample(adata, target: int, seed: int = 42):
    """Return adata subsampled to target cells (deterministic, no replacement)."""
    if len(adata) <= target:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(adata), size=target, replace=False))
    logger.warning(f"  subsample: {len(adata):,} → {target:,} cells (seed={seed})")
    sub = adata[idx].copy()
    del adata
    gc.collect()
    return sub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BANKSY+scType single-slide worker")
    p.add_argument("slide", help="rep1 | rep2 | breast_s{0,1,3,6} | skin_s{1..4}")
    p.add_argument("out_json", help="output JSON path")
    p.add_argument("--max-cells", type=int,
                   default=int(os.environ.get("BANKSY_MAX_CELLS", "0")) or None,
                   help="subsample to this many cells if estimate exceeds budget "
                        "(default from BANKSY_MAX_CELLS env, or none)")
    p.add_argument("--max-mem-gb", type=float,
                   default=float(os.environ.get("BANKSY_MAX_GB", "25")),
                   help="RAM budget for BANKSY (default 25, or BANKSY_MAX_GB env)")
    return p.parse_args()


def main():
    args = parse_args()
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"BANKSY worker: slide={args.slide}, out={out_json}, "
                f"budget={args.max_mem_gb} GB, max_cells={args.max_cells}")

    if args.slide.startswith("rep"):
        adata = load_xenium_adata(args.slide)
    else:
        adata = load_sthelar_adata(args.slide)

    n_cells, n_genes = len(adata), adata.n_vars
    est_gb = estimate_banksy_ram_gb(n_cells, n_genes)
    logger.info(f"  loaded: {n_cells:,} cells × {n_genes} genes  "
                f"→ est. peak RAM = {est_gb:.1f} GB (budget {args.max_mem_gb} GB)")

    if est_gb > args.max_mem_gb:
        if args.max_cells and args.max_cells < n_cells:
            adata = maybe_subsample(adata, args.max_cells)
            new_est = estimate_banksy_ram_gb(len(adata), adata.n_vars)
            logger.info(f"  post-subsample est. RAM = {new_est:.1f} GB")
            if new_est > args.max_mem_gb:
                logger.warning(
                    f"  even after subsample, est. {new_est:.1f} GB exceeds "
                    f"budget {args.max_mem_gb} GB — proceeding anyway")
        else:
            logger.error(
                f"BANKSY pre-flight ABORT: estimated {est_gb:.1f} GB > "
                f"budget {args.max_mem_gb} GB and no --max-cells fallback set. "
                f"Re-run with --max-cells N or BANKSY_MAX_CELLS=N to subsample.")
            out_json.write_text(json.dumps({
                "slide": args.slide,
                "method": "banksy_sctype",
                "error": f"pre-flight RAM {est_gb:.1f} GB > budget {args.max_mem_gb} GB",
                "n_cells": n_cells,
                "n_genes": n_genes,
            }))
            sys.exit(3)

    # STHELAR breast slides keep centroids in obsm['spatial'] not obs.
    # BANKSY's run_banksy_sctype expects x_centroid/y_centroid in obs.
    if ("x_centroid" not in adata.obs.columns and
            "spatial" in adata.obsm.keys()):
        spatial = np.asarray(adata.obsm["spatial"])
        if spatial.shape[1] >= 2:
            adata.obs["x_centroid"] = spatial[:, 0]
            adata.obs["y_centroid"] = spatial[:, 1]
            logger.info(f"  populated x/y_centroid from obsm['spatial'] "
                        f"(shape={spatial.shape})")

    adata_pp = preprocess_adata(adata)
    # preprocess_adata may have stripped obsm — re-attach centroids on the
    # *preprocessed* adata if they got lost.
    if "x_centroid" not in adata_pp.obs.columns and "x_centroid" in adata.obs.columns:
        adata_pp.obs["x_centroid"] = adata.obs["x_centroid"].values
        adata_pp.obs["y_centroid"] = adata.obs["y_centroid"].values
    markers = get_default_markers()

    result = run_banksy_sctype(adata_pp, markers)

    if result.get("predictions") is None:
        logger.error(f"BANKSY failed: {result.get('error')}")
        out_json.write_text(json.dumps({
            "slide": args.slide,
            "method": "banksy_sctype",
            "error": str(result.get("error")),
        }))
        sys.exit(1)

    # Save raw predictions (string array) + confidence (1.0 for BANKSY +
    # cluster-label scoring) for downstream consensus integration.
    payload = {
        "slide": args.slide,
        "method": result["method"],
        "raw_preds": [str(p) for p in result["predictions"]],
        "conf": [float(c) for c in result["confidence"]],
        "n_cells": int(len(result["predictions"])),
        "subsampled_from": n_cells if len(result["predictions"]) < n_cells else None,
    }
    out_json.write_text(json.dumps(payload))
    logger.info(f"BANKSY worker DONE: wrote {out_json} ({payload['n_cells']:,} cells"
                + (f", subsampled from {n_cells:,}" if payload["subsampled_from"] else "")
                + ")")


if __name__ == "__main__":
    main()
