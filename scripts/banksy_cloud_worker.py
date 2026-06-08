"""Cloud BANKSY+scType worker — loads a pre-extracted h5ad (no zarr, no RAM gate).

Mirrors banksy_breast_worker.py's post-load path EXACTLY (centroid population,
preprocess_adata, run_banksy_sctype) so the 757,374-cell post-preprocess order
reproduces the stored breast_s1_gt.json order — letting the result splice into
banksy_integrate_results.py by row index. Meant to run on a big-RAM box where
the ~47 GB BANKSY peak fits; the local pre-flight RAM gate is intentionally
absent here.

    uv run python scripts/banksy_cloud_worker.py <h5ad> <out_json> [slide_name]
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from annotation_benchmark_2026_03 import preprocess_adata  # noqa: E402
from banksy_cloud_markers import DEFAULT_MARKERS  # noqa: E402  (avoids dapidl.pipeline/pydantic)
from banksy_cloud_run import run_banksy_sctype  # noqa: E402  (standalone copy)


def main() -> None:
    h5ad_path = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    slide = sys.argv[3] if len(sys.argv) > 3 else "breast_s1"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path)
    raw_n = len(adata)
    logger.info(f"loaded {h5ad_path}: {raw_n:,} cells x {adata.n_vars} genes")

    # Centroids: BANKSY's run_banksy_sctype expects x_centroid/y_centroid in obs.
    if "x_centroid" not in adata.obs.columns and "spatial" in adata.obsm:
        spatial = np.asarray(adata.obsm["spatial"])
        adata.obs["x_centroid"] = spatial[:, 0]
        adata.obs["y_centroid"] = spatial[:, 1]
        logger.info(f"  populated x/y_centroid from obsm['spatial'] {spatial.shape}")

    adata_pp = preprocess_adata(adata)
    if "x_centroid" not in adata_pp.obs.columns and "x_centroid" in adata.obs.columns:
        adata_pp.obs["x_centroid"] = adata.obs["x_centroid"].values
        adata_pp.obs["y_centroid"] = adata.obs["y_centroid"].values
    logger.info(f"  preprocessed: {len(adata_pp):,} cells x {adata_pp.n_vars} genes")

    markers = DEFAULT_MARKERS
    result = run_banksy_sctype(adata_pp, markers)

    if result.get("predictions") is None:
        logger.error(f"BANKSY failed: {result.get('error')}")
        out_json.write_text(json.dumps({
            "slide": slide, "method": "banksy_sctype",
            "error": str(result.get("error")),
        }))
        sys.exit(1)

    payload = {
        "slide": slide,
        "method": result["method"],
        "raw_preds": [str(p) for p in result["predictions"]],
        "conf": [float(c) for c in result["confidence"]],
        "n_cells": int(len(result["predictions"])),
        "raw_n_cells": int(raw_n),
        "aligned_with_gt": True,
    }
    out_json.write_text(json.dumps(payload))
    logger.info(f"DONE: wrote {out_json} ({payload['n_cells']:,} preds, raw_n={raw_n:,})")


if __name__ == "__main__":
    main()
