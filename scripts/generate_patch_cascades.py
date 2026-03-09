"""Generate individual DAPI and annotation images at 32, 64, 128, 256, 512 px.

For each selected cell, outputs:
  - dapi_{size}.png   — grayscale DAPI image
  - anno_{size}.png   — DAPI with segmented nuclei overlaid, color-coded by cell type

Selects cells with round nuclei (high circularity) and diverse cell types.

Usage:
    uv run python scripts/generate_patch_cascades.py
    uv run python scripts/generate_patch_cascades.py --n-cells 30
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from loguru import logger
from PIL import Image, ImageDraw

logger.remove()
logger.add(sys.stderr, level="INFO")

# ── Configuration ────────────────────────────────────────────────

PATCH_SIZES = [32, 64, 128, 256, 512]
N_PER_DATASET = 20
OUTPUT_DIR = Path("/mnt/work/git/dapidl/patch_cascades")

# Cell type colors (RGBA with alpha for overlay)
CELL_TYPE_COLORS = {
    "Epithelial": (66, 133, 244),    # Blue
    "Immune": (234, 67, 53),          # Red
    "Stromal": (52, 168, 83),         # Green
    "Endothelial": (251, 188, 4),     # Yellow
    "Unknown": (150, 150, 150),       # Gray
}

FINE_TYPE_COLORS = {
    "DCIS_1": (100, 149, 237), "DCIS_2": (65, 105, 225),
    "Invasive_Tumor": (30, 80, 200), "Prolif_Invasive_Tumor": (0, 50, 180),
    "Myoepi_KRT15+": (135, 180, 255), "Myoepi_ACTA2+": (120, 160, 240),
    "CD4+_T_Cells": (255, 99, 71), "CD8+_T_Cells": (220, 60, 60),
    "B_Cells": (255, 140, 0), "Macrophages_1": (200, 50, 50),
    "Macrophages_2": (180, 40, 40), "Mast_Cells": (255, 105, 180),
    "LAMP3+_DCs": (255, 69, 0), "IRF7+_DCs": (233, 80, 30),
    "Perivascular-Like": (210, 105, 30),
    "Stromal": (34, 139, 34), "Endothelial": (0, 200, 100),
    "Stromal_&_T_Cell_Hybrid": (60, 179, 113),
}

GT_BROAD_MAP = {
    "DCIS_1": "Epithelial", "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial", "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_KRT15+": "Epithelial", "Myoepi_ACTA2+": "Epithelial",
    "T_Cell_&_Tumor_Hybrid": "Epithelial",
    "CD4+_T_Cells": "Immune", "CD8+_T_Cells": "Immune",
    "B_Cells": "Immune", "Macrophages_1": "Immune", "Macrophages_2": "Immune",
    "Mast_Cells": "Immune", "LAMP3+_DCs": "Immune", "IRF7+_DCs": "Immune",
    "Perivascular-Like": "Immune",
    "Stromal": "Stromal", "Stromal_&_T_Cell_Hybrid": "Stromal", "Endothelial": "Stromal",
}

XENIUM_PIXEL_SIZE = 0.2125  # µm/pixel


# ── Circularity Computation ──────────────────────────────────────

def compute_cell_circularity(boundaries_df: pl.DataFrame) -> pl.DataFrame:
    """Compute circularity = 4π × area / perimeter² for each cell from boundary vertices.

    Returns DataFrame with columns: cell_id (str), circularity (float).
    """
    results = []
    for cell_id, group in boundaries_df.group_by("cell_id"):
        cid = cell_id[0]
        vx = group["vertex_x"].to_numpy()
        vy = group["vertex_y"].to_numpy()
        n = len(vx)
        if n < 3:
            continue

        # Shoelace formula for area
        area = 0.5 * abs(np.sum(vx[:-1] * vy[1:] - vx[1:] * vy[:-1])
                         + vx[-1] * vy[0] - vx[0] * vy[-1])

        # Perimeter
        dx = np.diff(vx, append=vx[0])
        dy = np.diff(vy, append=vy[0])
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter**2)
        else:
            circularity = 0.0

        results.append((str(cid), circularity))

    return pl.DataFrame({"cell_id_str": [r[0] for r in results],
                         "circularity": [r[1] for r in results]})


# ── Data Loading ─────────────────────────────────────────────────

def load_xenium_boundaries(data_path: Path, pixel_size: float) -> dict[str, list[tuple[float, float]]]:
    """Load cell boundaries as dict of cell_id -> list of (x_px, y_px) vertices."""
    boundary_path = data_path / "cell_boundaries.parquet"
    if not boundary_path.exists():
        # Try rglob
        candidates = list(data_path.rglob("cell_boundaries.parquet"))
        if not candidates:
            logger.warning("No cell_boundaries.parquet found")
            return {}
        boundary_path = candidates[0]

    logger.info(f"  Loading cell boundaries from {boundary_path.name}...")
    df = pl.read_parquet(boundary_path)

    boundaries: dict[str, list[tuple[float, float]]] = {}
    for cell_id, group in df.group_by("cell_id"):
        cid = str(cell_id[0])
        vx = (group["vertex_x"].to_numpy() / pixel_size).tolist()
        vy = (group["vertex_y"].to_numpy() / pixel_size).tolist()
        boundaries[cid] = list(zip(vx, vy))

    logger.info(f"  Loaded boundaries for {len(boundaries)} cells")
    return boundaries


def load_xenium_data(
    data_path: Path, gt_file: str
) -> tuple[np.ndarray, pl.DataFrame, dict[str, list[tuple[float, float]]]]:
    """Load Xenium DAPI image, GT-annotated cells, and cell boundaries."""
    import pandas as pd

    logger.info(f"Loading Xenium data from {data_path.name}...")

    # Load DAPI image
    for img_name in ["morphology_focus.ome.tif", "morphology.ome.tif"]:
        img_candidates = list(data_path.rglob(img_name))
        if img_candidates:
            img_path = img_candidates[0]
            break
    else:
        raise FileNotFoundError(f"No DAPI image found in {data_path}")

    logger.info(f"  Loading DAPI: {img_path.name}")
    image = tifffile.imread(str(img_path))
    if image.ndim == 3:
        image = image[0]
    logger.info(f"  Image shape: {image.shape}, dtype: {image.dtype}")

    # Load cells.parquet
    cells_path = list(data_path.rglob("cells.parquet"))[0]
    cells_df = pl.read_parquet(cells_path)
    cells_df = cells_df.with_columns(
        (pl.col("x_centroid") / XENIUM_PIXEL_SIZE).alias("x_px"),
        (pl.col("y_centroid") / XENIUM_PIXEL_SIZE).alias("y_px"),
    )

    # Load ground truth
    gt_path = data_path / gt_file
    gt_pd = pd.read_excel(gt_path)
    gt_df = pl.DataFrame({
        "cell_id": [str(b) for b in gt_pd.iloc[:, 0]],
        "gt_fine": gt_pd.iloc[:, 1].astype(str).tolist(),
    })
    gt_df = gt_df.with_columns(
        pl.col("gt_fine")
        .map_elements(lambda v: GT_BROAD_MAP.get(v, "Unknown"), return_dtype=pl.Utf8)
        .alias("gt_broad"),
    ).filter(pl.col("gt_broad") != "Unknown")

    cells_df = cells_df.with_columns(
        pl.col("cell_id").cast(pl.Utf8).alias("cell_id_str")
    )
    cells_df = cells_df.join(gt_df, left_on="cell_id_str", right_on="cell_id", how="inner")

    # Compute circularity from boundaries
    boundary_parquet = data_path / "cell_boundaries.parquet"
    if not boundary_parquet.exists():
        candidates = list(data_path.rglob("cell_boundaries.parquet"))
        boundary_parquet = candidates[0] if candidates else None

    if boundary_parquet and boundary_parquet.exists():
        boundaries_raw = pl.read_parquet(boundary_parquet)
        circ_df = compute_cell_circularity(boundaries_raw)
        cells_df = cells_df.join(circ_df, on="cell_id_str", how="left")
        cells_df = cells_df.with_columns(pl.col("circularity").fill_null(0.0))
        median_circ = cells_df["circularity"].median()
        logger.info(f"  Median circularity: {median_circ:.3f}")
    else:
        cells_df = cells_df.with_columns(pl.lit(0.5).alias("circularity"))

    # Load boundaries (pixel coords) for overlay
    boundaries = load_xenium_boundaries(data_path, XENIUM_PIXEL_SIZE)

    logger.info(f"  {len(cells_df)} cells with GT annotations")
    return image, cells_df, boundaries


def load_merscope_data(
    data_path: Path,
) -> tuple[np.ndarray, pl.DataFrame, dict[str, list[tuple[float, float]]]]:
    """Load MERSCOPE DAPI image, annotate with CellTypist, load boundaries."""
    import anndata as ad
    import scanpy as sc

    logger.info(f"Loading MERSCOPE data from {data_path.name}...")

    # Load DAPI image
    img_path = data_path / "images" / "mosaic_DAPI_z3.tif"
    logger.info(f"  Loading DAPI: {img_path.name}")
    image = tifffile.imread(str(img_path))
    if image.ndim == 3:
        image = image[0]
    logger.info(f"  Image shape: {image.shape}, dtype: {image.dtype}")

    # Load transform matrix
    transform_path = data_path / "images" / "micron_to_mosaic_pixel_transform.csv"
    transform = np.loadtxt(str(transform_path))
    scale_x, tx = transform[0, 0], transform[0, 2]
    scale_y, ty = transform[1, 1], transform[1, 2]

    # Load cell metadata
    cells_df = pl.read_csv(data_path / "cell_metadata.csv")
    id_col = cells_df.columns[0]
    cells_df = cells_df.with_columns(
        pl.col(id_col).cast(pl.Utf8).alias("cell_id_str"),
        (pl.col("center_x") * scale_x + tx).alias("x_px"),
        (pl.col("center_y") * scale_y + ty).alias("y_px"),
    )
    logger.info(f"  {len(cells_df)} total cells")

    # Load cell boundaries
    boundary_path = data_path / "cell_boundaries"
    boundaries: dict[str, list[tuple[float, float]]] = {}
    if boundary_path.exists():
        # MERSCOPE stores boundaries as HDF5 or parquet per FOV
        logger.info("  Loading MERSCOPE cell boundaries...")
        boundary_files = list(boundary_path.glob("*.parquet")) + list(boundary_path.glob("*.hdf5"))
        if boundary_files:
            for bf in boundary_files:
                try:
                    bdf = pl.read_parquet(bf)
                    for cell_id, group in bdf.group_by(bdf.columns[0]):
                        cid = str(cell_id[0])
                        vx = (group[bdf.columns[1]].to_numpy() * scale_x + tx).tolist()
                        vy = (group[bdf.columns[2]].to_numpy() * scale_y + ty).tolist()
                        boundaries[cid] = list(zip(vx, vy))
                except Exception:
                    pass
            logger.info(f"  Loaded MERSCOPE boundaries for {len(boundaries)} cells")

    # Annotate with CellTypist
    rng = np.random.default_rng(42)
    n_sample = min(50000, len(cells_df))
    sample_idx = rng.choice(len(cells_df), n_sample, replace=False)
    sample_ids = set(cells_df[sample_idx]["cell_id_str"].to_list())

    logger.info(f"  Loading expression matrix for {n_sample} cells...")
    expr_df = pl.read_csv(data_path / "cell_by_gene.csv")
    expr_id_col = expr_df.columns[0]
    expr_df = expr_df.with_columns(pl.col(expr_id_col).cast(pl.Utf8).alias("cell_id_str"))
    expr_sample = expr_df.filter(pl.col("cell_id_str").is_in(sample_ids))

    gene_cols = [c for c in expr_sample.columns if c not in [expr_id_col, "cell_id_str"]]
    expr_matrix = expr_sample.select(gene_cols).to_numpy().astype(np.float32)
    cell_ids_expr = expr_sample["cell_id_str"].to_list()

    adata = ad.AnnData(X=expr_matrix)
    adata.var_names = gene_cols
    adata.obs_names = cell_ids_expr

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    import celltypist
    from dapidl.data.annotation import map_to_broad_category

    logger.info("  Running CellTypist (Cells_Adult_Breast.pkl)...")
    model = celltypist.models.Model.load("Cells_Adult_Breast.pkl")
    result = celltypist.annotate(adata, model=model, majority_voting=False)
    preds = result.predicted_labels["predicted_labels"].values
    broad = [map_to_broad_category(p) for p in preds]

    anno_map = {}
    for cid, fine, brd in zip(cell_ids_expr, preds, broad):
        anno_map[cid] = (str(fine), brd)

    cells_df = cells_df.filter(pl.col("cell_id_str").is_in(set(cell_ids_expr)))
    cells_df = cells_df.with_columns(
        pl.col("cell_id_str")
        .map_elements(lambda c: anno_map.get(c, ("Unknown", "Unknown"))[0], return_dtype=pl.Utf8)
        .alias("gt_fine"),
        pl.col("cell_id_str")
        .map_elements(lambda c: anno_map.get(c, ("Unknown", "Unknown"))[1], return_dtype=pl.Utf8)
        .alias("gt_broad"),
    ).filter(pl.col("gt_broad") != "Unknown")

    # No circularity data for MERSCOPE — use default
    cells_df = cells_df.with_columns(pl.lit(0.5).alias("circularity"))

    logger.info(f"  {len(cells_df)} annotated cells")
    return image, cells_df, boundaries


# ── Patch Extraction & Image Generation ──────────────────────────

def normalize_to_uint8(patch: np.ndarray) -> np.ndarray:
    """Normalize uint16 DAPI patch to uint8 with adaptive percentile scaling."""
    p_low = np.percentile(patch, 1)
    p_high = np.percentile(patch, 99.5)
    if p_high <= p_low:
        p_high = p_low + 1
    clipped = np.clip(patch.astype(np.float32), p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)


def extract_patch(image: np.ndarray, x_px: float, y_px: float, size: int) -> np.ndarray | None:
    """Extract a single patch. Returns None if out of bounds."""
    h, w = image.shape[:2]
    cx, cy = int(round(x_px)), int(round(y_px))
    half = size // 2
    if cx - half < 0 or cx + half > w or cy - half < 0 or cy + half > h:
        return None
    return image[cy - half : cy + half, cx - half : cx + half].copy()


def create_annotation_overlay(
    patch: np.ndarray,
    center_x_px: float,
    center_y_px: float,
    size: int,
    cells_df: pl.DataFrame,
    boundaries: dict[str, list[tuple[float, float]]],
    center_cell_id: str,
    alpha: float = 0.35,
) -> Image.Image:
    """Create DAPI image with semi-transparent colored cell boundaries overlaid.

    Each cell boundary polygon is filled with its cell type color at `alpha` opacity.
    The center cell gets a brighter outline highlight.
    """
    # Normalize DAPI to RGB
    patch_u8 = normalize_to_uint8(patch)
    dapi_rgb = np.stack([patch_u8] * 3, axis=-1)
    base = Image.fromarray(dapi_rgb, mode="RGB")

    # Create transparent overlay for filled polygons
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Outline layer (for center cell highlight)
    outline_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_outline = ImageDraw.Draw(outline_layer)

    half = size // 2
    patch_x0 = center_x_px - half
    patch_y0 = center_y_px - half

    # Find all cells whose centroids are within the patch
    nearby = cells_df.filter(
        (pl.col("x_px") >= patch_x0)
        & (pl.col("x_px") < patch_x0 + size)
        & (pl.col("y_px") >= patch_y0)
        & (pl.col("y_px") < patch_y0 + size)
    )

    for row in nearby.iter_rows(named=True):
        cid = row["cell_id_str"]
        broad = row["gt_broad"]
        fine = row.get("gt_fine", broad)

        if cid not in boundaries:
            continue

        verts = boundaries[cid]
        # Transform to patch-local coordinates
        local_verts = [(vx - patch_x0, vy - patch_y0) for vx, vy in verts]

        # Skip if polygon is entirely outside
        xs = [v[0] for v in local_verts]
        ys = [v[1] for v in local_verts]
        if max(xs) < 0 or min(xs) > size or max(ys) < 0 or min(ys) > size:
            continue

        # Get color
        color = FINE_TYPE_COLORS.get(fine, CELL_TYPE_COLORS.get(broad, (150, 150, 150)))
        fill_alpha = int(255 * alpha)

        # Draw filled polygon on overlay
        if len(local_verts) >= 3:
            draw_overlay.polygon(local_verts, fill=(*color, fill_alpha))

            # Outline for all cells (thin)
            draw_overlay.polygon(local_verts, outline=(*color, 200), width=1)

            # Center cell gets bright thick outline
            if cid == center_cell_id:
                draw_outline.polygon(local_verts, outline=(255, 255, 255, 230), width=2)
                draw_outline.polygon(local_verts, outline=(*color, 255), width=1)

    # Composite: base + overlay + outline
    base_rgba = base.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay)
    composited = Image.alpha_composite(composited, outline_layer)
    return composited.convert("RGB")


# ── Cell Selection ───────────────────────────────────────────────

def select_round_cells(
    cells_df: pl.DataFrame,
    image_shape: tuple[int, int],
    n_cells: int,
    min_circularity: float = 0.65,
    seed: int = 42,
) -> pl.DataFrame:
    """Select cells with round nuclei, balanced across cell types."""
    h, w = image_shape
    max_half = max(PATCH_SIZES) // 2
    margin = max_half + 50

    # Filter to well-interior cells
    interior = cells_df.filter(
        (pl.col("x_px") > margin)
        & (pl.col("x_px") < w - margin)
        & (pl.col("y_px") > margin)
        & (pl.col("y_px") < h - margin)
    )

    # Filter by circularity (prefer round nuclei)
    round_cells = interior.filter(pl.col("circularity") >= min_circularity)
    if len(round_cells) < n_cells * 3:
        # Relax threshold if too few
        round_cells = interior.filter(pl.col("circularity") >= 0.4)
        logger.info(f"  Relaxed circularity to 0.4 ({len(round_cells)} candidates)")

    if len(round_cells) < n_cells:
        round_cells = interior
        logger.info(f"  Using all interior cells ({len(round_cells)} candidates)")

    # Sort by circularity descending — roundest cells first
    round_cells = round_cells.sort("circularity", descending=True)

    # Balance by cell type
    types = round_cells["gt_broad"].unique().to_list()
    per_type = max(1, n_cells // len(types))
    remainder = n_cells - per_type * len(types)

    rng = np.random.default_rng(seed)
    selected = []
    for ct in sorted(types):
        subset = round_cells.filter(pl.col("gt_broad") == ct)
        # Take from the top (roundest) with some randomness
        top_pool = subset.head(min(len(subset), per_type * 5))
        n_take = min(per_type, len(top_pool))
        if n_take > 0:
            idx = rng.choice(len(top_pool), n_take, replace=False)
            selected.append(top_pool[sorted(idx.tolist())])

    result = pl.concat(selected) if selected else round_cells.head(0)

    # Fill remaining quota
    if len(result) < n_cells:
        remaining = round_cells.filter(~pl.col("cell_id_str").is_in(result["cell_id_str"]))
        n_extra = min(n_cells - len(result), len(remaining))
        if n_extra > 0:
            idx = rng.choice(len(remaining), n_extra, replace=False)
            result = pl.concat([result, remaining[idx.tolist()]])

    return result.head(n_cells)


# ── Processing ───────────────────────────────────────────────────

def process_dataset(
    name: str,
    image: np.ndarray,
    cells_df: pl.DataFrame,
    boundaries: dict[str, list[tuple[float, float]]],
    n_cells: int,
    output_dir: Path,
    seed: int = 42,
) -> list[dict]:
    """Process one dataset: select cells, extract patches, save individual images."""
    selected = select_round_cells(cells_df, image.shape[:2], n_cells, seed=seed)
    logger.info(f"  Selected {len(selected)} cells from {name}")

    dist = dict(selected.group_by("gt_broad").agg(pl.len().alias("n")).iter_rows())
    logger.info(f"  Distribution: {dist}")

    if "circularity" in selected.columns:
        mean_circ = selected["circularity"].mean()
        logger.info(f"  Mean circularity of selected: {mean_circ:.3f}")

    metadata = []
    for i, row in enumerate(selected.iter_rows(named=True)):
        x_px = row["x_px"]
        y_px = row["y_px"]
        cell_type_fine = row["gt_fine"]
        cell_type_broad = row["gt_broad"]
        cell_id = row["cell_id_str"]

        # Create cell-specific directory
        cell_dir = output_dir / name / f"{i:02d}_{cell_type_broad}_{cell_id}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        files = {}
        for size in PATCH_SIZES:
            patch = extract_patch(image, x_px, y_px, size)
            if patch is None:
                continue

            # DAPI grayscale
            patch_u8 = normalize_to_uint8(patch)
            dapi_img = Image.fromarray(patch_u8, mode="L")
            dapi_path = cell_dir / f"dapi_{size}.png"
            dapi_img.save(str(dapi_path))

            # Annotation overlay
            if boundaries:
                anno_img = create_annotation_overlay(
                    patch, x_px, y_px, size,
                    cells_df, boundaries, center_cell_id=cell_id,
                )
            else:
                # Fallback: just draw a circle at center
                patch_rgb = np.stack([patch_u8] * 3, axis=-1)
                anno_img = Image.fromarray(patch_rgb, mode="RGB")
                draw = ImageDraw.Draw(anno_img)
                c = size // 2
                r = max(3, size // 16)
                color = CELL_TYPE_COLORS.get(cell_type_broad, (150, 150, 150))
                draw.ellipse([c - r, c - r, c + r, c + r], outline=color, width=2)

            anno_path = cell_dir / f"anno_{size}.png"
            anno_img.save(str(anno_path))

            files[size] = {
                "dapi": f"{cell_dir.name}/dapi_{size}.png",
                "anno": f"{cell_dir.name}/anno_{size}.png",
            }

        if files:
            metadata.append({
                "index": i,
                "dataset": name,
                "cell_id": cell_id,
                "cell_type_fine": cell_type_fine,
                "cell_type_broad": cell_type_broad,
                "circularity": float(row.get("circularity", 0)),
                "x_px": float(x_px),
                "y_px": float(y_px),
                "color_rgb": list(CELL_TYPE_COLORS.get(cell_type_broad, (150, 150, 150))),
                "patch_sizes_px": list(files.keys()),
                "files": files,
            })

    logger.info(f"  Saved {len(metadata)} cells ({len(metadata) * len(PATCH_SIZES) * 2} images) to {output_dir / name}")
    return metadata


def create_legend(output_dir: Path):
    """Create a color legend image."""
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    row_h = 32
    w = 400
    h = row_h * (len(CELL_TYPE_COLORS) + 1) + 20
    img = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    draw.text((10, 8), "Cell Type Color Legend", fill=(255, 255, 255), font=font)
    for i, (ct, color) in enumerate(CELL_TYPE_COLORS.items()):
        y = row_h * (i + 1) + 10
        draw.rectangle([10, y, 40, y + 20], fill=color)
        draw.text((50, y + 2), ct, fill=color, font=font)

    img.save(str(output_dir / "legend.png"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate patch cascade images")
    parser.add_argument("--n-cells", type=int, default=N_PER_DATASET,
                        help=f"Cells per dataset (default: {N_PER_DATASET})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old files
    for subdir in ["rep1", "rep2", "merscope-breast"]:
        old_dir = OUTPUT_DIR / subdir
        if old_dir.exists():
            import shutil
            shutil.rmtree(old_dir)
            logger.info(f"Cleaned old {subdir}")

    all_metadata = {}
    n = args.n_cells

    # ── Rep1 ──────────────────────────────────────────────────
    rep1_path = Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1")
    image1, cells1, bounds1 = load_xenium_data(
        rep1_path, "celltypes_ground_truth_rep1_supervised.xlsx"
    )
    meta1 = process_dataset("rep1", image1, cells1, bounds1, n, OUTPUT_DIR, seed=42)
    all_metadata["rep1"] = meta1
    del image1

    # ── Rep2 ──────────────────────────────────────────────────
    rep2_path = Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2")
    image2, cells2, bounds2 = load_xenium_data(
        rep2_path, "celltypes_ground_truth_rep2_supervised.xlsx"
    )
    meta2 = process_dataset("rep2", image2, cells2, bounds2, n, OUTPUT_DIR, seed=43)
    all_metadata["rep2"] = meta2
    del image2

    # ── MERSCOPE ──────────────────────────────────────────────
    merscope_path = Path("/mnt/work/datasets/raw/merscope/merscope-breast")
    image_m, cells_m, bounds_m = load_merscope_data(merscope_path)
    meta_m = process_dataset("merscope-breast", image_m, cells_m, bounds_m, n, OUTPUT_DIR, seed=44)
    all_metadata["merscope-breast"] = meta_m
    del image_m

    # ── Save metadata ─────────────────────────────────────────
    meta_out = {
        "description": f"Individual DAPI + annotation images at {PATCH_SIZES} px",
        "datasets": list(all_metadata.keys()),
        "cell_type_colors": {k: list(v) for k, v in CELL_TYPE_COLORS.items()},
        "fine_type_colors": {k: list(v) for k, v in FINE_TYPE_COLORS.items()},
        "patch_sizes_px": PATCH_SIZES,
        "cells": all_metadata,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    create_legend(OUTPUT_DIR)

    total = sum(len(v) for v in all_metadata.values())
    logger.info(f"\nDone! {total} cells × {len(PATCH_SIZES)} sizes × 2 types = {total * len(PATCH_SIZES) * 2} images")
    logger.info(f"  Rep1: {len(meta1)} cells")
    logger.info(f"  Rep2: {len(meta2)} cells")
    logger.info(f"  MERSCOPE: {len(meta_m)} cells")
    logger.info(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
