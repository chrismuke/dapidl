#!/usr/bin/env python3
"""Generate patch cascades (32, 64, 128, 256, 512, 1024 px) for STHELAR + MERSCOPE.

For each selected cell, produces a 2-row × 6-column matplotlib figure:
- Row 1: raw staining patches at each size
- Row 2: same patches with nuclei + cell boundaries overlaid, colored by cell type
- Legend: cell types present in the surrounding context

STHELAR slides have BOTH DAPI ("morpho") and H&E ("he") modalities — two figures
per cell. MERSCOPE has DAPI only — one figure per cell.

Cells are picked per slide for: diverse cell types, good DAPI signal, valid
boundaries on all 6 patch sizes (within image bounds), and high circularity.

Outputs:
    /home/chrism/obsidian/llmbrain/DAPIDL/Patch-Cascades-20260429/
        sthelar/<slide>/<idx>_dapi.png
        sthelar/<slide>/<idx>_he.png
        merscope/<slide>/<idx>_dapi.png

Usage:
    uv run python scripts/sthelar_merscope_cascades.py [--n-cells 10]
                                                       [--platforms sthelar,merscope]
                                                       [--slides ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import polars as pl
import tifffile
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cascade_render import make_annotation_overlay, render_cascade_figure  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | <cyan>{time:HH:mm:ss}</cyan> | {message}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PATCH_SIZES = [32, 64, 128, 256, 512, 1024]
N_PER_SLIDE = 10

OBSIDIAN_OUT = Path("/home/chrism/obsidian/llmbrain/DAPIDL/Patch-Cascades-20260429")
STHELAR_DIR = Path("/mnt/work/datasets/STHELAR/sdata_slides")
MERSCOPE_DIR = Path("/home/chrism/datasets/raw/merscope")

XENIUM_PIXEL_SIZE = 0.2125  # µm/pixel — STHELAR DAPI ("morpho") native pixel size

# STHELAR label1 → CL name (same mapping used for the modality experiments)
STHELAR_TO_CL = {
    "T": "T cell",
    "B": "B cell",
    "Perivascular": "pericyte",
    "Monocyte/Macrophage": "macrophage",
    "Mast": "mast cell",
    "Epithelial": "epithelial cell",
    "Fibroblast": "fibroblast",
    "Endothelial": "endothelial cell",
    "Adipocyte": "adipocyte",
}

# Consistent palette across both platforms (RGB uint8)
CELLTYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "T cell":            (220,  60,  60),  # red
    "B cell":            (255, 140,   0),  # orange
    "macrophage":        (200,  50,  50),  # dark red
    "mast cell":         (255, 105, 180),  # pink
    "pericyte":          (210, 105,  30),  # rust
    "epithelial cell":   ( 30,  80, 200),  # blue
    "fibroblast":        ( 34, 139,  34),  # green
    "endothelial cell":  (251, 188,   4),  # yellow
    "adipocyte":         (147, 112, 219),  # purple
    # MERSCOPE broad classes
    "Epithelial":        ( 30,  80, 200),
    "Immune":            (220,  60,  60),
    "Stromal":           ( 34, 139,  34),
    "Endothelial":       (251, 188,   4),
    "Unknown":           (150, 150, 150),
}


# ---------------------------------------------------------------------------
# H&E affine helper (mirrors sthelar_multi_tissue_he_lmdb.get_he_to_global_affine)
# ---------------------------------------------------------------------------

def get_he_to_global_affine(reader) -> np.ndarray | None:
    """Read 2D affine HE → global from the zarr's multiscales attrs.

    Returns a 3x3 matrix mapping HE (y_he, x_he, 1) → global (x_g, y_g, 1).
    """
    multiscales = reader.store["images"]["he"].attrs.get("multiscales", [])
    for ms in multiscales:
        for ct in ms.get("coordinateTransformations", []):
            if ct.get("output", {}).get("name") == "global" and ct.get("type") == "affine":
                A = np.array(ct["affine"])  # 4x4 in (c, y, x) order
                M = np.eye(3)
                M[0, 0] = A[1, 1]; M[0, 1] = A[1, 2]; M[0, 2] = A[1, 3]
                M[1, 0] = A[2, 1]; M[1, 1] = A[2, 2]; M[1, 2] = A[2, 3]
                return M
    return None


def he_global_to_pixel(M_he2global: np.ndarray | None,
                       gx: float, gy: float) -> tuple[float, float]:
    """Map a global pixel (x, y) through the inverse HE→global affine.

    Returns (x_he_px, y_he_px). The HE global uses (x, y) order; HE local uses
    (y_he, x_he) so the affine is built with HE = (y, x) on input and global
    = (x, y) on output. The forward maps HE_local_y, HE_local_x → global_x, global_y.
    To go global → HE_local we invert.
    """
    if M_he2global is None:
        return float(gx), float(gy)
    M_inv = np.linalg.inv(M_he2global)
    out = M_inv @ np.array([gx, gy, 1.0])
    # M_inv maps global (x, y, 1) -> HE local (y, x, 1). Swap to (x_he, y_he):
    y_he, x_he = out[0], out[1]
    return float(x_he), float(y_he)


# ---------------------------------------------------------------------------
# DAPI normalisation
# ---------------------------------------------------------------------------

def adaptive_normalize_dapi(image: np.ndarray) -> np.ndarray:
    """Normalize a uint16 DAPI patch to float32 in [0, 1]."""
    img = image.astype(np.float32)
    p_low, p_high = np.percentile(img, [1, 99.5])
    if p_high <= p_low:
        p_high = p_low + 1.0
    return np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

def select_cells_sthelar(
    nuc_df: pl.DataFrame,
    image_h: int,
    image_w: int,
    n_cells: int,
    rng: np.random.Generator,
    margin: int = 600,
) -> pl.DataFrame:
    """Pick `n_cells` STHELAR cells with diverse types, away from image edges.

    Filters: only cells with mapped label1 + within margin from image bounds
    so the largest patch (1024 px) fits comfortably.
    """
    df = nuc_df.with_columns(
        pl.col("label1").replace_strict(STHELAR_TO_CL, default=None).alias("cl_name")
    ).filter(pl.col("cl_name").is_not_null())

    # boundary check: 1024//2 = 512 + buffer
    df = df.filter(
        (pl.col("x_centroid_px") > margin)
        & (pl.col("x_centroid_px") < image_w - margin)
        & (pl.col("y_centroid_px") > margin)
        & (pl.col("y_centroid_px") < image_h - margin)
    )
    if len(df) == 0:
        return df

    # PanNuke_proba > 0.6 if available — picks well-segmented nuclei
    if "PanNuke_proba" in df.columns:
        df = df.filter(pl.col("PanNuke_proba") > 0.6)

    # diversity: try to sample evenly across cell types
    types = df["cl_name"].unique().to_list()
    rng.shuffle(types)
    per_type = max(1, n_cells // max(1, len(types)))
    chunks = []
    for t in types:
        sub = df.filter(pl.col("cl_name") == t)
        if len(sub) == 0:
            continue
        sub_sample = sub.sample(n=min(per_type, len(sub)), seed=int(rng.integers(2**31)))
        chunks.append(sub_sample)
    sampled = pl.concat(chunks) if chunks else df.head(0)

    # top up if under
    if len(sampled) < n_cells:
        need = n_cells - len(sampled)
        existing_ids = set(sampled["cell_id"].to_list())
        rest = df.filter(~pl.col("cell_id").is_in(existing_ids))
        if len(rest) > 0:
            extra = rest.sample(n=min(need, len(rest)), seed=int(rng.integers(2**31)))
            sampled = pl.concat([sampled, extra])

    return sampled.head(n_cells)


# ---------------------------------------------------------------------------
# STHELAR cascade
# ---------------------------------------------------------------------------

def render_sthelar_slide(
    zarr_path: Path,
    output_dir: Path,
    n_cells: int = N_PER_SLIDE,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate cascade figures for one STHELAR slide. Returns a stats dict."""
    from dapidl.data.sthelar import SthelarDataReader

    rng = rng if rng is not None else np.random.default_rng(42)
    slide_name = zarr_path.name.replace("sdata_", "").replace(".zarr", "")
    tissue = slide_name.rsplit("_s", 1)[0]
    out = output_dir / slide_name
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{slide_name}] loading reader…")
    try:
        reader = SthelarDataReader(zarr_path)
    except Exception as e:
        logger.error(f"[{slide_name}] reader failed: {e}")
        return {"slide": slide_name, "ok": False, "err": str(e)}

    nuc_df = reader.nucleus_df
    if "label1" not in nuc_df.columns:
        logger.warning(f"[{slide_name}] no label1 column; skipping")
        return {"slide": slide_name, "ok": False, "err": "no label1"}

    h, w = reader.image_shape
    logger.info(f"[{slide_name}] image {h}x{w}, {len(nuc_df)} nuclei")

    sel = select_cells_sthelar(nuc_df, h, w, n_cells, rng)
    if len(sel) == 0:
        logger.warning(f"[{slide_name}] no cells passed selection")
        return {"slide": slide_name, "ok": False, "err": "no cells"}

    logger.info(f"[{slide_name}] selected {len(sel)} cells: "
                f"{sel['cl_name'].to_list()}")

    # Lazy zarr handles for region reads
    morpho_zarr = reader.store["images"]["morpho"]["0"]   # (1, H, W) uint16
    he_zarr = reader.store["images"]["he"]["0"]           # (3, H_he, W_he) uint8
    he_h, he_w = he_zarr.shape[1], he_zarr.shape[2]

    M_he2global = get_he_to_global_affine(reader)
    M_global2he = np.linalg.inv(M_he2global) if M_he2global is not None else None

    # Boundary parquets — micron-coords. Some slides have nested zarr layout
    # (zarr_path/zarr_path.name/shapes/...), others have flat (zarr_path/shapes/...).
    def _find_shapes_parquet(kind: str) -> Path | None:
        candidates = [
            zarr_path / "shapes" / kind / "shapes.parquet",
            zarr_path / zarr_path.name / "shapes" / kind / "shapes.parquet",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    cell_shapes_path = _find_shapes_parquet("cell_boundaries")
    nuc_shapes_path = _find_shapes_parquet("nucleus_boundaries")
    cell_gdf = gpd.read_parquet(cell_shapes_path) if cell_shapes_path else None
    nuc_gdf = gpd.read_parquet(nuc_shapes_path) if nuc_shapes_path else None

    if cell_gdf is None and nuc_gdf is None:
        logger.warning(f"[{slide_name}] no boundary parquets found")

    # Pre-compute lookup of nucleus_df to find neighbors in patch by pixel coords
    all_nuc_px = nuc_df.select([
        pl.col("cell_id"),
        pl.col("x_centroid_px").alias("x_px"),
        pl.col("y_centroid_px").alias("y_px"),
        pl.col("label1"),
    ]).with_columns(
        pl.col("label1").replace_strict(STHELAR_TO_CL, default="Unknown").alias("cl_name")
    )

    n_done = 0
    for idx, row in enumerate(sel.iter_rows(named=True)):
        cx_px = float(row["x_centroid_px"])
        cy_px = float(row["y_centroid_px"])
        focal_type = row["cl_name"]
        cell_id = row["cell_id"]

        # ---- DAPI cascade ----
        try:
            dapi_panels, dapi_legend = build_panels_sthelar(
                cx_px, cy_px, focal_type, cell_id,
                modality="dapi",
                morpho_zarr=morpho_zarr, he_zarr=he_zarr,
                he_h=he_h, he_w=he_w, M_global2he=M_global2he,
                cell_gdf=cell_gdf, nuc_gdf=nuc_gdf,
                all_nuc_px=all_nuc_px,
                image_h=h, image_w=w,
            )
            render_cascade_figure(
                dapi_panels,
                out / f"{idx:02d}_dapi.png",
                title=f"{slide_name} | {focal_type} | DAPI | cell {cell_id}",
                legend_items=dapi_legend,
            )
        except Exception as e:
            logger.error(f"[{slide_name}] cell {idx} DAPI failed: {e}")

        # ---- H&E cascade ----
        try:
            he_panels, he_legend = build_panels_sthelar(
                cx_px, cy_px, focal_type, cell_id,
                modality="he",
                morpho_zarr=morpho_zarr, he_zarr=he_zarr,
                he_h=he_h, he_w=he_w, M_global2he=M_global2he,
                cell_gdf=cell_gdf, nuc_gdf=nuc_gdf,
                all_nuc_px=all_nuc_px,
                image_h=h, image_w=w,
            )
            render_cascade_figure(
                he_panels,
                out / f"{idx:02d}_he.png",
                title=f"{slide_name} | {focal_type} | H&E | cell {cell_id}",
                legend_items=he_legend,
            )
        except Exception as e:
            logger.error(f"[{slide_name}] cell {idx} HE failed: {e}")

        n_done += 1

    return {"slide": slide_name, "ok": True, "n_cells": n_done}


def build_panels_sthelar(
    cx_px: float, cy_px: float, focal_type: str, focal_cell_id: str,
    modality: str,
    morpho_zarr, he_zarr,
    he_h: int, he_w: int,
    M_global2he: np.ndarray | None,
    cell_gdf, nuc_gdf,
    all_nuc_px: pl.DataFrame,
    image_h: int, image_w: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[tuple[str, tuple[int, int, int]]]]:
    """Build the 6 (raw, overlay) panels for one cell, one modality."""
    panels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    legend_types: set[str] = {focal_type}

    for sz in PATCH_SIZES:
        half = sz // 2

        if modality == "dapi":
            y0, y1 = int(round(cy_px - half)), int(round(cy_px + half))
            x0, x1 = int(round(cx_px - half)), int(round(cx_px + half))
            y0c = max(0, y0); y1c = min(image_h, y1)
            x0c = max(0, x0); x1c = min(image_w, x1)
            patch_uint16 = np.asarray(morpho_zarr[0, y0c:y1c, x0c:x1c])
            # pad if near border
            if patch_uint16.shape != (sz, sz):
                pad = np.zeros((sz, sz), dtype=patch_uint16.dtype)
                pad[(y0c - y0):(y0c - y0) + patch_uint16.shape[0],
                    (x0c - x0):(x0c - x0) + patch_uint16.shape[1]] = patch_uint16
                patch_uint16 = pad
            patch_norm = adaptive_normalize_dapi(patch_uint16)
            patch_for_overlay: np.ndarray = patch_norm
            # transform: patch-local pixel = global pixel - top-left
            tl_x, tl_y = x0, y0
            scale = 1.0
            origin_in_he = False
        else:  # he — extract HE patch and warp to DAPI orientation
            if M_global2he is not None:
                # Map the 4 corners of the desired DAPI-aligned patch into HE space
                # to find the HE bounding box we need to read (with buffer for the warp).
                corners_global = np.array([
                    [cx_px - half, cy_px - half],
                    [cx_px + half, cy_px - half],
                    [cx_px + half, cy_px + half],
                    [cx_px - half, cy_px + half],
                ])
                corners_he = np.empty_like(corners_global, dtype=np.float64)
                for i, (gx, gy) in enumerate(corners_global):
                    v = M_global2he @ np.array([gx, gy, 1.0])
                    # M_global2he maps (x_g, y_g, 1) -> (y_he, x_he, 1); store as (x_he, y_he)
                    corners_he[i] = (v[1], v[0])
                buf = 4
                x0_he = int(np.floor(corners_he[:, 0].min())) - buf
                x1_he = int(np.ceil(corners_he[:, 0].max())) + buf
                y0_he = int(np.floor(corners_he[:, 1].min())) - buf
                y1_he = int(np.ceil(corners_he[:, 1].max())) + buf
            else:
                # Identity affine — extract HE region directly at global coords.
                x0_he = int(round(cx_px - half)); x1_he = int(round(cx_px + half))
                y0_he = int(round(cy_px - half)); y1_he = int(round(cy_px + half))

            x0c = max(0, x0_he); x1c = min(he_w, x1_he)
            y0c = max(0, y0_he); y1c = min(he_h, y1_he)
            he_chunk = np.asarray(he_zarr[:, y0c:y1c, x0c:x1c])  # (3, h, w)
            target_h = y1_he - y0_he
            target_w = x1_he - x0_he
            if he_chunk.shape[1] != target_h or he_chunk.shape[2] != target_w:
                pad = np.zeros((3, target_h, target_w), dtype=he_chunk.dtype)
                pad[:, (y0c - y0_he):(y0c - y0_he) + he_chunk.shape[1],
                    (x0c - x0_he):(x0c - x0_he) + he_chunk.shape[2]] = he_chunk
                he_chunk = pad
            he_region = he_chunk.transpose(1, 2, 0).copy()  # (H_r, W_r, 3) uint8

            if M_global2he is not None:
                # Build composite: patch coord (out_x, out_y, 1) -> HE region (x_r, y_r, 1)
                # 1) patch -> global:    [out_x + (cx_px - half), out_y + (cy_px - half)]
                # 2) global -> HE (x, y): swap rows of M_global2he so output is (x_he, y_he, 1)
                # 3) HE -> region:       subtract (x0_he, y0_he)
                T_p2g = np.array([
                    [1.0, 0.0, cx_px - half],
                    [0.0, 1.0, cy_px - half],
                    [0.0, 0.0, 1.0],
                ])
                M_g2he_xy = np.array([
                    [M_global2he[1, 0], M_global2he[1, 1], M_global2he[1, 2]],
                    [M_global2he[0, 0], M_global2he[0, 1], M_global2he[0, 2]],
                    [0.0, 0.0, 1.0],
                ])
                T_he2region = np.array([
                    [1.0, 0.0, -x0_he],
                    [0.0, 1.0, -y0_he],
                    [0.0, 0.0, 1.0],
                ])
                T_combined = T_he2region @ M_g2he_xy @ T_p2g
                M_2x3 = T_combined[:2, :].astype(np.float32)
                patch_for_overlay = cv2.warpAffine(
                    he_region, M_2x3, (sz, sz),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
                )
            else:
                patch_for_overlay = he_region
                if patch_for_overlay.shape[0] != sz or patch_for_overlay.shape[1] != sz:
                    patch_for_overlay = cv2.resize(patch_for_overlay, (sz, sz),
                                                   interpolation=cv2.INTER_LINEAR)

            # After warping, polygons can use the same DAPI-aligned local coord
            # system as the DAPI branch — patch-local pixel = global pixel - top-left.
            tl_x, tl_y = cx_px - half, cy_px - half
            origin_in_he = False  # HE patch is now in DAPI-aligned coord space

        # ---- Find nuclei within the patch + their boundaries ----
        # Use pixel space search (DAPI/global coords), then transform overlay coords.
        in_patch = all_nuc_px.filter(
            (pl.col("x_px") >= cx_px - half - 60)
            & (pl.col("x_px") <= cx_px + half + 60)
            & (pl.col("y_px") >= cy_px - half - 60)
            & (pl.col("y_px") <= cy_px + half + 60)
        )

        # When the HE patch is resized from sz_he → sz for display, scale local
        # polygon coords by sz / sz_he (== 1 / scale).
        if origin_in_he:
            display_scale = sz / float(sz_he)
        else:
            display_scale = 1.0

        polys_local: list[np.ndarray] = []
        polys_color: list[tuple[int, int, int]] = []
        polys_kind: list[str] = []
        for nrow in in_patch.iter_rows(named=True):
            cid = nrow["cell_id"]
            cl = nrow["cl_name"]
            if cl != "Unknown":
                legend_types.add(cl)
            color = CELLTYPE_COLORS.get(cl, (150, 150, 150))
            # Cell boundary
            if cell_gdf is not None and cid in cell_gdf.index:
                geom = cell_gdf.loc[cid].geometry
                if geom is not None and not geom.is_empty:
                    coords = np.asarray(geom.exterior.coords)  # micron
                    pix = coords / XENIUM_PIXEL_SIZE  # global pixel
                    if origin_in_he and M_global2he is not None:
                        pts_local = np.zeros_like(pix)
                        for i, (gx, gy) in enumerate(pix):
                            v = M_global2he @ np.array([gx, gy, 1.0])
                            pts_local[i] = (v[1] - tl_x, v[0] - tl_y)
                        pts_local *= display_scale
                    else:
                        pts_local = pix - np.array([tl_x, tl_y])
                    polys_local.append(pts_local)
                    polys_color.append(color)
                    polys_kind.append("cell")
            # Nucleus boundary
            if nuc_gdf is not None and cid in nuc_gdf.index:
                geom = nuc_gdf.loc[cid].geometry
                if geom is not None and not geom.is_empty:
                    coords = np.asarray(geom.exterior.coords)
                    pix = coords / XENIUM_PIXEL_SIZE
                    if origin_in_he and M_global2he is not None:
                        pts_local = np.zeros_like(pix)
                        for i, (gx, gy) in enumerate(pix):
                            v = M_global2he @ np.array([gx, gy, 1.0])
                            pts_local[i] = (v[1] - tl_x, v[0] - tl_y)
                        pts_local *= display_scale
                    else:
                        pts_local = pix - np.array([tl_x, tl_y])
                    polys_local.append(pts_local)
                    polys_color.append(color)
                    polys_kind.append("nucleus")

        overlay = make_annotation_overlay(
            patch_for_overlay, polys_local, polys_color, polys_kind,
        )
        panels[sz] = (patch_for_overlay, overlay)

    legend_items = sorted(
        [(t, CELLTYPE_COLORS.get(t, (150, 150, 150))) for t in legend_types],
        key=lambda x: x[0],
    )
    return panels, legend_items


# ---------------------------------------------------------------------------
# MERSCOPE — DAPI cascades with marker-based broad annotation
# ---------------------------------------------------------------------------

# Markers used to compute broad classes (sum across panel; gene must be in this slide's panel to count)
MERSCOPE_MARKERS = {
    "Immune": ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD14", "CD68",
               "CD79A", "MS4A1", "CD20", "CD163", "ITGAX", "FCER1A", "TPSAB1"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT5", "KRT14", "CDH1",
                   "MUC1", "KRT7"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "CLDN5"],
    "Stromal": ["ACTA2", "COL1A1", "COL3A1", "FAP", "PDGFRA", "PDGFRB", "DCN"],
}


def annotate_merscope_broad(cells_df: pl.DataFrame,
                            cell_by_gene: pl.DataFrame) -> pl.DataFrame:
    """Add a 'broad_class' column to cells_df based on simple marker sums."""
    gene_cols = [c for c in cell_by_gene.columns if c not in {"cell", "cell_id", ""}]
    available = {cls: [g for g in markers if g in gene_cols]
                 for cls, markers in MERSCOPE_MARKERS.items()}
    sum_exprs = []
    for cls, gs in available.items():
        if not gs:
            sum_exprs.append(pl.lit(0.0).alias(f"_score_{cls}"))
        else:
            total = sum((pl.col(g) for g in gs), pl.lit(0.0))
            sum_exprs.append(total.alias(f"_score_{cls}"))
    scored = cell_by_gene.with_columns(sum_exprs)

    # Take id from first column
    id_col = cell_by_gene.columns[0]
    score_cols = [f"_score_{c}" for c in MERSCOPE_MARKERS.keys()]
    classes = list(MERSCOPE_MARKERS.keys())

    rows = scored.select([id_col] + score_cols).to_numpy()
    ids = rows[:, 0].astype(str)
    scores = rows[:, 1:].astype(float)
    max_idx = scores.argmax(axis=1)
    max_val = scores.max(axis=1)
    labels = np.array([classes[i] if max_val[i] > 0 else "Unknown" for i in max_idx])

    annot = pl.DataFrame({"cell_id_str": ids, "broad_class": labels})
    return cells_df.join(annot, on="cell_id_str", how="left").with_columns(
        pl.col("broad_class").fill_null("Unknown")
    )


def render_merscope_slide(
    merscope_path: Path,
    output_dir: Path,
    n_cells: int = N_PER_SLIDE,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate DAPI cascade figures for one MERSCOPE slide."""
    rng = rng if rng is not None else np.random.default_rng(42)
    slide_name = merscope_path.name
    out = output_dir / slide_name
    out.mkdir(parents=True, exist_ok=True)

    img_path = merscope_path / "images" / "mosaic_DAPI_z3.tif"
    if not img_path.exists():
        # try other z planes
        candidates = list((merscope_path / "images").glob("mosaic_DAPI_z*.tif"))
        if not candidates:
            logger.error(f"[{slide_name}] no DAPI image; skipping")
            return {"slide": slide_name, "ok": False, "err": "no DAPI"}
        img_path = candidates[0]

    image_mmap = tifffile.memmap(str(img_path), mode="r")
    if image_mmap.ndim == 3:
        image_mmap = image_mmap[0]
    img_h, img_w = image_mmap.shape
    logger.info(f"[{slide_name}] DAPI {img_h}x{img_w}")

    # Affine transform
    transform_path = merscope_path / "images" / "micron_to_mosaic_pixel_transform.csv"
    transform = np.loadtxt(str(transform_path))
    scale_x, tx = transform[0, 0], transform[0, 2]
    scale_y, ty = transform[1, 1], transform[1, 2]

    # Cell metadata
    cells_df = pl.read_csv(merscope_path / "cell_metadata.csv")
    id_col = cells_df.columns[0]
    cells_df = cells_df.with_columns(
        pl.col(id_col).cast(pl.Utf8).alias("cell_id_str"),
        (pl.col("center_x") * scale_x + tx).alias("x_px"),
        (pl.col("center_y") * scale_y + ty).alias("y_px"),
    )

    # Cell × gene → broad class
    cell_by_gene_path = merscope_path / "cell_by_gene.csv"
    if cell_by_gene_path.exists():
        cell_by_gene = pl.read_csv(cell_by_gene_path)
        # Make first column string
        first_col = cell_by_gene.columns[0]
        cell_by_gene = cell_by_gene.with_columns(pl.col(first_col).cast(pl.Utf8))
        cells_df = annotate_merscope_broad(cells_df, cell_by_gene)
    else:
        cells_df = cells_df.with_columns(pl.lit("Unknown").alias("broad_class"))

    margin = 600
    df = cells_df.filter(
        (pl.col("x_px") > margin) & (pl.col("x_px") < img_w - margin)
        & (pl.col("y_px") > margin) & (pl.col("y_px") < img_h - margin)
    )
    if len(df) == 0:
        logger.warning(f"[{slide_name}] no cells inside margin")
        return {"slide": slide_name, "ok": False, "err": "no cells"}

    # diversity sample
    classes = df["broad_class"].unique().to_list()
    rng.shuffle(classes)
    per = max(1, n_cells // max(1, len(classes)))
    chunks = []
    for c in classes:
        sub = df.filter(pl.col("broad_class") == c)
        if len(sub) == 0:
            continue
        chunks.append(sub.sample(n=min(per, len(sub)), seed=int(rng.integers(2**31))))
    sampled = pl.concat(chunks) if chunks else df.head(0)
    if len(sampled) < n_cells:
        rest = df.filter(~pl.col("cell_id_str").is_in(sampled["cell_id_str"].to_list()))
        if len(rest) > 0:
            sampled = pl.concat([sampled,
                                 rest.sample(n=min(n_cells - len(sampled), len(rest)),
                                             seed=int(rng.integers(2**31)))])
    sampled = sampled.head(n_cells)
    logger.info(f"[{slide_name}] selected {len(sampled)} cells: "
                f"{sampled['broad_class'].to_list()}")

    # All cells lookup for neighbor annotations
    nearby = cells_df.select(["cell_id_str", "x_px", "y_px", "broad_class"])

    n_done = 0
    for idx, row in enumerate(sampled.iter_rows(named=True)):
        cx_px = float(row["x_px"])
        cy_px = float(row["y_px"])
        focal_type = str(row["broad_class"])
        cell_id = row["cell_id_str"]

        try:
            panels, legend = build_panels_merscope(
                cx_px, cy_px, focal_type, cell_id,
                image_mmap=image_mmap, image_h=img_h, image_w=img_w,
                nearby=nearby,
            )
            render_cascade_figure(
                panels,
                out / f"{idx:02d}_dapi.png",
                title=f"{slide_name} | {focal_type} | DAPI | cell {cell_id}",
                legend_items=legend,
            )
            n_done += 1
        except Exception as e:
            logger.error(f"[{slide_name}] cell {idx} failed: {e}")

    return {"slide": slide_name, "ok": True, "n_cells": n_done}


def build_panels_merscope(
    cx_px: float, cy_px: float, focal_type: str, focal_cell_id: str,
    image_mmap: np.ndarray, image_h: int, image_w: int,
    nearby: pl.DataFrame,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[tuple[str, tuple[int, int, int]]]]:
    panels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    legend_types: set[str] = {focal_type}

    # MERSCOPE has no cell-boundary polygons in the local data — synthesise
    # circle markers from cell centroids as a visual cue. The marker is drawn
    # as a small filled disk centered on the centroid, scaled to the patch.
    for sz in PATCH_SIZES:
        half = sz // 2
        y0, y1 = int(round(cy_px - half)), int(round(cy_px + half))
        x0, x1 = int(round(cx_px - half)), int(round(cx_px + half))
        y0c = max(0, y0); y1c = min(image_h, y1)
        x0c = max(0, x0); x1c = min(image_w, x1)
        patch_uint16 = np.array(image_mmap[y0c:y1c, x0c:x1c])
        if patch_uint16.shape != (sz, sz):
            pad = np.zeros((sz, sz), dtype=patch_uint16.dtype)
            pad[(y0c - y0):(y0c - y0) + patch_uint16.shape[0],
                (x0c - x0):(x0c - x0) + patch_uint16.shape[1]] = patch_uint16
            patch_uint16 = pad
        patch_norm = adaptive_normalize_dapi(patch_uint16)

        # Find nearby cell centroids within the patch box
        in_patch = nearby.filter(
            (pl.col("x_px") >= cx_px - half - 30)
            & (pl.col("x_px") <= cx_px + half + 30)
            & (pl.col("y_px") >= cy_px - half - 30)
            & (pl.col("y_px") <= cy_px + half + 30)
        )

        # Build "fake polygons" (small circles) for centroid markers
        polys_local: list[np.ndarray] = []
        polys_color: list[tuple[int, int, int]] = []
        polys_kind: list[str] = []
        radius = max(2, int(round(sz * 0.012)))  # scale with patch
        theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        circle_template = np.stack([np.cos(theta), np.sin(theta)], axis=1) * radius
        for nrow in in_patch.iter_rows(named=True):
            cl = str(nrow["broad_class"])
            legend_types.add(cl)
            color = CELLTYPE_COLORS.get(cl, (150, 150, 150))
            cx_local = nrow["x_px"] - x0
            cy_local = nrow["y_px"] - y0
            poly = circle_template + np.array([cx_local, cy_local])
            polys_local.append(poly)
            polys_color.append(color)
            polys_kind.append("nucleus")  # outline only (no fill — keeps DAPI visible)

        overlay = make_annotation_overlay(patch_norm, polys_local, polys_color, polys_kind)
        panels[sz] = (patch_norm, overlay)

    legend_items = sorted(
        [(t, CELLTYPE_COLORS.get(t, (150, 150, 150))) for t in legend_types],
        key=lambda x: x[0],
    )
    return panels, legend_items


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=N_PER_SLIDE)
    ap.add_argument("--platforms", type=str, default="sthelar,merscope",
                    help="comma-separated subset")
    ap.add_argument("--slides", type=str, default=None,
                    help="Optional comma-separated list of slide names "
                         "(e.g. 'breast_s0,kidney_s0' or 'merscope-breast'). "
                         "If omitted, processes ALL slides.")
    ap.add_argument("--output", type=Path, default=OBSIDIAN_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    platforms = [p.strip().lower() for p in args.platforms.split(",")]
    slides_filter = set(s.strip() for s in args.slides.split(",")) if args.slides else None

    summary = {"sthelar": [], "merscope": []}

    if "sthelar" in platforms:
        sthelar_out = args.output / "sthelar"
        sthelar_out.mkdir(parents=True, exist_ok=True)
        zarrs = sorted(STHELAR_DIR.glob("sdata_*.zarr"))
        for zp in zarrs:
            sname = zp.name.replace("sdata_", "").replace(".zarr", "")
            if slides_filter is not None and sname not in slides_filter:
                continue
            t0 = time.time()
            stats = render_sthelar_slide(zp, sthelar_out, n_cells=args.n_cells, rng=rng)
            stats["wall_s"] = round(time.time() - t0, 1)
            summary["sthelar"].append(stats)
            logger.info(f"[sthelar/{sname}] {stats}")

    if "merscope" in platforms:
        merscope_out = args.output / "merscope"
        merscope_out.mkdir(parents=True, exist_ok=True)
        slides = sorted(MERSCOPE_DIR.glob("merscope-*"))
        for sp in slides:
            if slides_filter is not None and sp.name not in slides_filter:
                continue
            t0 = time.time()
            stats = render_merscope_slide(sp, merscope_out, n_cells=args.n_cells, rng=rng)
            stats["wall_s"] = round(time.time() - t0, 1)
            summary["merscope"].append(stats)
            logger.info(f"[merscope/{sp.name}] {stats}")

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"summary written to {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
