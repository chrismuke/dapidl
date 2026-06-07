"""Extract breast_s1's minimal adata for the cloud BANKSY run.

Saves the EXACT output of load_sthelar_adata("breast_s1") to an h5ad so the
cloud worker can reproduce preprocess_adata + run_banksy_sctype and land on the
same 757,374-cell post-preprocess order as the stored breast_s1_gt.json
(index alignment is what lets banksy_integrate_results.py splice it in).

Keeps X (raw counts, sparse), obsm['spatial'] (centroids), var; drops nothing
that preprocessing relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from scipy.sparse import issparse  # noqa: E402

from annotation_benchmark_2026_03 import load_sthelar_adata  # noqa: E402

OUT = Path("/tmp/breast_s1_minimal.h5ad")


def main() -> None:
    adata = load_sthelar_adata("breast_s1")
    # Trim heavy state preprocess_adata would discard anyway, to shrink transfer.
    for k in list(adata.layers.keys()):
        del adata.layers[k]
    for k in list(adata.obsp.keys()):
        del adata.obsp[k]
    keep_obsm = {"spatial"}
    for k in list(adata.obsm.keys()):
        if k not in keep_obsm:
            del adata.obsm[k]
    print(f"adata: {adata.n_obs:,} cells x {adata.n_vars} genes | "
          f"X sparse={issparse(adata.X)} | obsm={list(adata.obsm.keys())}")
    assert "spatial" in adata.obsm, "spatial centroids missing — BANKSY needs them"
    adata.write_h5ad(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
