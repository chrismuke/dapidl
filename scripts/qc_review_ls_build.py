"""Build a stratified p64 QC review sample for Label Studio.

Outputs (to $DSET/qc/review_ls/):
  - <row_idx>.png   plain DAPI, 1-99% stretch, upscaled to DISP px (NO baked overlays)
  - manifest.parquet  per-patch: row_idx, slide, cell_class, assigned_group,
                      nuc_points / cell_points  (LS polygon percent-coords, JSON strings)

The nucleus/cell polygons are pushed to Label Studio as *prediction* layers
(toggleable overlays), so the image itself stays a clean DAPI crop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lmdb
import numpy as np
import polars as pl
from PIL import Image
from skimage import measure

from dapidl.qc.review_select import select_review_rows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_qc_collages_v3 import (  # noqa: E402 -- reuse tested helpers (after sys.path.insert)
    assign_group,
    build_mask_index,
    load_mask_from_index,
    load_patch,
)

DSET = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1")
OUT = DSET / "qc" / "review_ls"
N_PER, PS, DISP = 5, 64, 256  # patches/bucket, native size, display size


def mask_to_pct(mask: np.ndarray, max_pts: int = 40):
    """Largest contour of a bool mask -> list of [x%, y%] points (LS polygon coords)."""
    if not mask.any():
        return None
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)  # (N, 2) as (row, col)
    if len(c) > max_pts:
        c = c[:: max(1, len(c) // max_pts)]
    h, w = mask.shape
    return [[float(col) / w * 100.0, float(row) / h * 100.0] for row, col in c]


def render_dapi(patch: np.ndarray, size: int = DISP) -> Image.Image:
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, [1.0, 99.0])
    hi = hi if hi > lo else lo + 1.0
    g = (np.clip((p - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(g, mode="L").resize((size, size), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", choices=["stratified", "anomaly_disagree"], default="stratified")
    ap.add_argument("--n", type=int, default=300, help="count for anomaly_disagree selection")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    anom = DSET / "qc" / "seg_scores_anom.parquet"
    scores = pl.read_parquet(anom if anom.exists() else DSET / "qc" / "seg_scores.parquet")
    has_anom = "anomaly_pct" in scores.columns
    reg = pl.read_parquet(DSET / "patch_registry.parquet").select(["row_idx", "slide", "raw_label"])
    idx2coarse = {v: k for k, v in json.loads((DSET / "class_mapping.json").read_text()).items()}

    j = scores.join(reg, on=["row_idx", "slide"], how="left")
    qc = [assign_group(b, r or "", g or "Weak-passing")
          for b, r, g in zip(j["broken"], j["broken_reason"], j["grade"], strict=False)]
    cc = [idx2coarse.get(int(i), "Unknown") for i in j["coarse_idx"]]
    df = j.with_columns([pl.Series("qc_group", qc), pl.Series("cell_class", cc)])

    if args.select == "anomaly_disagree":
        if not has_anom:
            raise SystemExit("--select anomaly_disagree needs seg_scores_anom.parquet "
                             "(run scripts/qc_anomaly_score.py first)")
        df = select_review_rows(df, mode="anomaly_disagree", n=args.n)
    else:
        df = df.sample(fraction=1.0, shuffle=True, seed=args.seed).group_by(
            ["slide", "cell_class", "qc_group"]).head(N_PER)
    print(f"[build] select={args.select} -> {df.height} patches", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    nidx = build_mask_index(DSET / "qc" / "masks", "nuc")
    cidx = build_mask_index(DSET / "qc" / "masks", "cell")
    env = lmdb.open(str(DSET / "patches.lmdb"), readonly=True, lock=False, readahead=False)

    rows = []
    for r in df.iter_rows(named=True):
        ri = int(r["row_idx"])
        render_dapi(load_patch(env, ri, PS)).save(out_dir / f"{ri}.png")
        nuc = mask_to_pct(load_mask_from_index(nidx, ri, PS))
        cell = mask_to_pct(load_mask_from_index(cidx, ri, PS))
        ap_val = r.get("anomaly_pct") if has_anom else None
        rows.append({
            "row_idx": ri, "slide": r["slide"], "cell_class": r["cell_class"],
            "assigned_group": r["qc_group"],
            "anomaly_pct": float(ap_val) if ap_val is not None else None,
            "nuc_points": json.dumps(nuc) if nuc else None,
            "cell_points": json.dumps(cell) if cell else None,
        })
    env.close()
    out = pl.DataFrame(rows)
    out.write_parquet(out_dir / "manifest.parquet")
    print(f"[build] wrote {out.height} PNGs + manifest -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
