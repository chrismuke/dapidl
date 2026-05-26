"""Pairing audit for the breast-pool sources (spec §4.2).

For every cell present in each slide we check whether its nucleus polygon is
geometrically nested in its cell polygon under the same `cell_id`. Catches:
- nucleus polygons with no matching cell (orphans)
- nucleus centroid sitting outside the cell polygon
- nucleus area >= cell area (impossible for a proper containment)
- nucleus polygon extending outside the cell polygon

Two source families share the same checks, only the I/O differs:
- Xenium: per-vertex parquet rows -> shapely Polygons (cells.parquet has the
  pre-computed centroid; cell_boundaries.parquet + nucleus_boundaries.parquet
  carry the polygon vertices, in microns).
- STHELAR: precomputed polygons in shapes/{cell,nucleus}_boundaries/shapes.parquet
  via geopandas (in microns when read directly; we keep micron units throughout
  so the two pipelines stay symmetric).

Per-source sample size is bounded with --sample-n (default 50000) so STHELAR's
~700k-cell slides finish in minutes instead of an hour.

Outputs:
    pipeline_output/pairing_audit/
        audit_summary.parquet   one row per source x checked metric
        audit_summary.md        markdown rendering for the readout
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
from loguru import logger
from shapely.geometry import Polygon

XENIUM_BASE = Path("/mnt/work/datasets/raw/xenium")
STHELAR_BASE = Path("/mnt/work/datasets/STHELAR/sdata_slides")
DEFAULT_OUT = Path("/mnt/work/git/dapidl/pipeline_output/pairing_audit")


def _polygons_from_vertices(df: pl.DataFrame) -> gpd.GeoDataFrame:
    """Group vertex rows by cell_id -> shapely Polygons. Skips cells with <3 vertices."""
    g = df.group_by("cell_id", maintain_order=True).agg(
        pl.col("vertex_x").alias("xs"), pl.col("vertex_y").alias("ys")
    )
    geoms, cids = [], []
    for r in g.iter_rows(named=True):
        xs, ys = r["xs"], r["ys"]
        if xs is None or len(xs) < 3:
            continue
        poly = Polygon(zip(xs, ys))
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
        geoms.append(poly)
        cids.append(str(r["cell_id"]))
    return gpd.GeoDataFrame({"cell_id": cids}, geometry=geoms)


def audit_xenium(rep_name: str, sample_n: int, rng: np.random.Generator) -> dict:
    outs = XENIUM_BASE / f"xenium-breast-tumor-{rep_name}" / "outs"
    cells = pl.read_parquet(outs / "cells.parquet")
    nuc_v = pl.read_parquet(outs / "nucleus_boundaries.parquet")
    cell_v = pl.read_parquet(outs / "cell_boundaries.parquet")
    n_cells_total = cells.height
    logger.info(f"{rep_name}: {n_cells_total} cells, {nuc_v.height} nuc vertex rows")

    if sample_n and n_cells_total > sample_n:
        idx = rng.choice(n_cells_total, size=sample_n, replace=False)
        sampled = cells[idx]["cell_id"].cast(pl.Utf8).to_list()
    else:
        sampled = cells["cell_id"].cast(pl.Utf8).to_list()
    sampled_set = set(sampled)

    nuc_v = nuc_v.with_columns(pl.col("cell_id").cast(pl.Utf8))
    cell_v = cell_v.with_columns(pl.col("cell_id").cast(pl.Utf8))
    nuc_v = nuc_v.filter(pl.col("cell_id").is_in(sampled_set))
    cell_v = cell_v.filter(pl.col("cell_id").is_in(sampled_set))

    nuc_gdf = _polygons_from_vertices(nuc_v).rename(columns={"geometry": "nuc_geom"})
    cell_gdf = _polygons_from_vertices(cell_v).rename(columns={"geometry": "cell_geom"})
    return _summarise(rep_name, sampled, nuc_gdf, cell_gdf)


def audit_sthelar(slide_zarr: Path, sample_n: int, rng: np.random.Generator) -> dict:
    inner = slide_zarr / slide_zarr.name if (slide_zarr / slide_zarr.name).exists() else slide_zarr
    nuc_path = inner / "shapes" / "nucleus_boundaries" / "shapes.parquet"
    cell_path = inner / "shapes" / "cell_boundaries" / "shapes.parquet"
    if not (nuc_path.exists() and cell_path.exists()):
        raise FileNotFoundError(f"missing shapes parquet under {inner}")

    nuc_gdf = gpd.read_parquet(nuc_path)
    cell_gdf = gpd.read_parquet(cell_path)
    nuc_gdf.index = nuc_gdf.index.astype(str)
    cell_gdf.index = cell_gdf.index.astype(str)
    n_cells_total = len(cell_gdf)
    logger.info(f"{slide_zarr.name}: {n_cells_total} cell polygons, "
                f"{len(nuc_gdf)} nucleus polygons")

    if sample_n and n_cells_total > sample_n:
        idx = rng.choice(n_cells_total, size=sample_n, replace=False)
        sampled = cell_gdf.index[idx].tolist()
    else:
        sampled = cell_gdf.index.tolist()

    nuc_sub = nuc_gdf.loc[nuc_gdf.index.isin(sampled)].reset_index(names="cell_id")
    cell_sub = cell_gdf.loc[cell_gdf.index.isin(sampled)].reset_index(names="cell_id")
    nuc_sub = nuc_sub.rename(columns={"geometry": "nuc_geom"}).set_geometry("nuc_geom")
    cell_sub = cell_sub.rename(columns={"geometry": "cell_geom"}).set_geometry("cell_geom")

    slide_name = "sthelar_" + slide_zarr.name.replace("sdata_", "").replace(".zarr", "")
    return _summarise(slide_name, sampled, nuc_sub, cell_sub)


def _summarise(source: str, sampled: list[str],
               nuc_gdf: gpd.GeoDataFrame, cell_gdf: gpd.GeoDataFrame) -> dict:
    n_sampled = len(sampled)
    nuc_gdf = nuc_gdf.set_index("cell_id") if "cell_id" in nuc_gdf.columns else nuc_gdf
    cell_gdf = cell_gdf.set_index("cell_id") if "cell_id" in cell_gdf.columns else cell_gdf
    joined = cell_gdf.join(nuc_gdf, how="left", lsuffix="_c", rsuffix="_n")
    paired = joined.dropna(subset=["nuc_geom"])
    n_paired = len(paired)

    # Geometric checks on paired rows.
    centroid_in = paired.apply(
        lambda r: r["cell_geom"].contains(r["nuc_geom"].centroid)
                  if r["nuc_geom"] is not None else False, axis=1
    )
    area_ok = paired.apply(
        lambda r: r["nuc_geom"].area < r["cell_geom"].area
                  if r["nuc_geom"] is not None else False, axis=1
    )
    # Within = nucleus polygon fully inside cell polygon (= no crossings).
    within = paired.apply(
        lambda r: r["nuc_geom"].within(r["cell_geom"])
                  if r["nuc_geom"] is not None else False, axis=1
    )

    return {
        "source": source,
        "n_sampled": int(n_sampled),
        "n_paired": int(n_paired),
        "frac_paired": float(n_paired / max(n_sampled, 1)),
        "frac_centroid_in_cell": float(centroid_in.mean()) if n_paired else 0.0,
        "frac_area_nuc_lt_cell": float(area_ok.mean()) if n_paired else 0.0,
        "frac_within": float(within.mean()) if n_paired else 0.0,
        "n_crossings": int((~within).sum()) if n_paired else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=50000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for rep in ("rep1", "rep2"):
        rows.append(audit_xenium(rep, args.sample_n, rng))
    for slide_zarr in sorted(STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        if not slide_zarr.is_dir():
            continue
        rows.append(audit_sthelar(slide_zarr, args.sample_n, rng))

    df = pl.DataFrame(rows)
    df.write_parquet(args.out / "audit_summary.parquet")

    md = ["# Pairing audit — breast pool", "",
          "Pass criteria: `frac_paired ≥ 0.90` AND `frac_centroid_in_cell ≥ 0.95`.",
          ""]
    md.append("| source | n_sampled | n_paired | frac_paired | "
              "frac_centroid_in_cell | frac_area_nuc<cell | frac_within | n_crossings |")
    md.append("|" + "---|" * 8)
    pass_all = True
    for r in df.iter_rows(named=True):
        gate = (r["frac_paired"] >= 0.90 and r["frac_centroid_in_cell"] >= 0.95)
        pass_all &= gate
        marker = "✅" if gate else "❌"
        md.append(f"| {r['source']} {marker} | {r['n_sampled']:,} | "
                  f"{r['n_paired']:,} | {r['frac_paired']:.3f} | "
                  f"{r['frac_centroid_in_cell']:.3f} | {r['frac_area_nuc_lt_cell']:.3f} | "
                  f"{r['frac_within']:.3f} | {r['n_crossings']:,} |")
    md.append("")
    md.append(f"**Overall**: {'PASS' if pass_all else 'FAIL'}")
    (args.out / "audit_summary.md").write_text("\n".join(md) + "\n")
    logger.info(f"wrote audit_summary.{{parquet,md}} to {args.out}")
    for r in df.iter_rows(named=True):
        logger.info(f"  {r['source']}: paired={r['frac_paired']:.3f}  "
                    f"centroid_in_cell={r['frac_centroid_in_cell']:.3f}  "
                    f"within={r['frac_within']:.3f}")


if __name__ == "__main__":
    main()
