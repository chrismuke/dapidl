"""Generate patch cascade images at 32, 64, 128, 256 px for cells from rep1, rep2, MERSCOPE.

For each cell: grayscale DAPI cascade + color-annotated cascade with cell type.
Selects well-centered cells with diverse cell types.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

logger.remove()
logger.add(sys.stderr, level="INFO")

# ── Configuration ────────────────────────────────────────────────

PATCH_SIZES = [32, 64, 128, 256]
DISPLAY_SIZE = 256  # All patches resized to this for visual comparison
N_TOTAL = 50
BORDER_WIDTH = 4
OUTPUT_DIR = Path("/mnt/work/git/dapidl/patch_cascades")

# Cell type colors (RGB)
CELL_TYPE_COLORS = {
    "Epithelial": (66, 133, 244),    # Blue
    "Immune": (234, 67, 53),          # Red
    "Stromal": (52, 168, 83),         # Green
    "Endothelial": (251, 188, 4),     # Yellow
    "Unknown": (150, 150, 150),       # Gray
}

# Fine-grained colors for GT types
FINE_TYPE_COLORS = {
    # Epithelial
    "DCIS_1": (100, 149, 237),        # Cornflower blue
    "DCIS_2": (65, 105, 225),         # Royal blue
    "Invasive_Tumor": (30, 80, 200),  # Dark blue
    "Prolif_Invasive_Tumor": (0, 50, 180),
    "Myoepi_KRT15+": (135, 180, 255),
    "Myoepi_ACTA2+": (120, 160, 240),
    # Immune
    "CD4+_T_Cells": (255, 99, 71),    # Tomato
    "CD8+_T_Cells": (220, 60, 60),    # Crimson
    "B_Cells": (255, 140, 0),         # Dark orange
    "Macrophages_1": (200, 50, 50),
    "Macrophages_2": (180, 40, 40),
    "Mast_Cells": (255, 105, 180),    # Hot pink
    "LAMP3+_DCs": (255, 69, 0),       # Orange red
    "IRF7+_DCs": (233, 80, 30),
    "Perivascular-Like": (210, 105, 30),
    # Stromal
    "Stromal": (34, 139, 34),         # Forest green
    "Endothelial": (0, 200, 100),     # Spring green
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


# ── Data Loading ─────────────────────────────────────────────────

def load_xenium_data(data_path: Path, gt_file: str) -> tuple[np.ndarray, pl.DataFrame]:
    """Load Xenium DAPI image and cells with GT annotations."""
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
        image = image[0]  # Take first channel/page
    logger.info(f"  Image shape: {image.shape}, dtype: {image.dtype}")

    # Load cells.parquet
    cells_path = list(data_path.rglob("cells.parquet"))[0]
    cells_df = pl.read_parquet(cells_path)

    # Convert micron → pixel
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

    # Join GT with cells
    cells_df = cells_df.with_columns(
        pl.col("cell_id").cast(pl.Utf8).alias("cell_id_str")
    )
    cells_df = cells_df.join(gt_df, left_on="cell_id_str", right_on="cell_id", how="inner")

    logger.info(f"  {len(cells_df)} cells with GT annotations")
    return image, cells_df


def load_merscope_data(data_path: Path) -> tuple[np.ndarray, pl.DataFrame]:
    """Load MERSCOPE DAPI image and annotate with CellTypist."""
    import anndata as ad
    import scanpy as sc

    logger.info(f"Loading MERSCOPE data from {data_path.name}...")

    # Load DAPI image (memory-mapped for large files)
    img_path = data_path / "images" / "mosaic_DAPI_z3.tif"
    logger.info(f"  Loading DAPI: {img_path.name} (memory-mapped)")
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
    # First column is unnamed index = cell_id
    id_col = cells_df.columns[0]
    cells_df = cells_df.with_columns(
        pl.col(id_col).cast(pl.Utf8).alias("cell_id_str"),
        (pl.col("center_x") * scale_x + tx).alias("x_px"),
        (pl.col("center_y") * scale_y + ty).alias("y_px"),
    )
    logger.info(f"  {len(cells_df)} total cells")

    # Annotate with CellTypist — subsample for speed
    rng = np.random.default_rng(42)
    n_sample = min(50000, len(cells_df))
    sample_idx = rng.choice(len(cells_df), n_sample, replace=False)
    sample_ids = set(cells_df[sample_idx]["cell_id_str"].to_list())

    # Load expression for sample
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

    # Normalize + CellTypist
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    import celltypist
    from dapidl.data.annotation import map_to_broad_category

    logger.info("  Running CellTypist (Cells_Adult_Breast.pkl)...")
    model = celltypist.models.Model.load("Cells_Adult_Breast.pkl")
    result = celltypist.annotate(adata, model=model, majority_voting=False)
    preds = result.predicted_labels["predicted_labels"].values
    broad = [map_to_broad_category(p) for p in preds]

    # Build annotation map
    anno_map = {}
    for cid, fine, brd in zip(cell_ids_expr, preds, broad):
        anno_map[cid] = (str(fine), brd)

    # Add annotations to cells_df
    cells_df = cells_df.filter(pl.col("cell_id_str").is_in(set(cell_ids_expr)))
    cells_df = cells_df.with_columns(
        pl.col("cell_id_str")
        .map_elements(lambda c: anno_map.get(c, ("Unknown", "Unknown"))[0], return_dtype=pl.Utf8)
        .alias("gt_fine"),
        pl.col("cell_id_str")
        .map_elements(lambda c: anno_map.get(c, ("Unknown", "Unknown"))[1], return_dtype=pl.Utf8)
        .alias("gt_broad"),
    ).filter(pl.col("gt_broad") != "Unknown")

    logger.info(f"  {len(cells_df)} annotated cells")
    return image, cells_df


# ── Patch Extraction ─────────────────────────────────────────────

def extract_multi_scale_patches(
    image: np.ndarray,
    x_px: float,
    y_px: float,
    sizes: list[int] = PATCH_SIZES,
) -> dict[int, np.ndarray] | None:
    """Extract patches at multiple scales centered on (x_px, y_px).

    Returns dict of {size: patch_array} or None if largest patch is out of bounds.
    """
    h, w = image.shape[:2]
    cx, cy = int(round(x_px)), int(round(y_px))
    max_half = max(sizes) // 2

    # Check bounds for largest patch
    if cx - max_half < 0 or cx + max_half > w or cy - max_half < 0 or cy + max_half > h:
        return None

    patches = {}
    for size in sizes:
        half = size // 2
        patch = image[cy - half : cy + half, cx - half : cx + half].copy()
        patches[size] = patch

    return patches


def normalize_to_uint8(patch: np.ndarray) -> np.ndarray:
    """Normalize uint16 DAPI patch to uint8 with adaptive percentile scaling."""
    p_low = np.percentile(patch, 1)
    p_high = np.percentile(patch, 99.5)
    if p_high <= p_low:
        p_high = p_low + 1
    clipped = np.clip(patch.astype(np.float32), p_low, p_high)
    normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    return normalized


# ── Image Generation ─────────────────────────────────────────────

def create_grayscale_cascade(
    patches: dict[int, np.ndarray],
    display_size: int = DISPLAY_SIZE,
) -> Image.Image:
    """Create a horizontal cascade of grayscale DAPI patches."""
    n = len(patches)
    gap = 4
    total_w = n * display_size + (n - 1) * gap
    canvas = Image.new("L", (total_w, display_size), color=0)

    for i, size in enumerate(sorted(patches.keys())):
        patch_u8 = normalize_to_uint8(patches[size])
        patch_img = Image.fromarray(patch_u8, mode="L")
        patch_resized = patch_img.resize((display_size, display_size), Image.LANCZOS)
        x_offset = i * (display_size + gap)
        canvas.paste(patch_resized, (x_offset, 0))

    return canvas


def create_colored_cascade(
    patches: dict[int, np.ndarray],
    cell_type_broad: str,
    cell_type_fine: str,
    dataset_name: str,
    display_size: int = DISPLAY_SIZE,
    border: int = BORDER_WIDTH,
) -> Image.Image:
    """Create a horizontal cascade with colored borders and labels."""
    n = len(patches)
    gap = 4
    label_h = 36
    size_label_h = 20
    total_w = n * display_size + (n - 1) * gap
    total_h = display_size + label_h + size_label_h
    color = CELL_TYPE_COLORS.get(cell_type_broad, (150, 150, 150))

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    for i, size in enumerate(sorted(patches.keys())):
        patch_u8 = normalize_to_uint8(patches[size])
        patch_rgb = np.stack([patch_u8] * 3, axis=-1)
        patch_img = Image.fromarray(patch_rgb, mode="RGB")
        patch_resized = patch_img.resize((display_size, display_size), Image.LANCZOS)

        # Draw colored border
        bordered = Image.new("RGB", (display_size, display_size), color=color)
        inner = display_size - 2 * border
        patch_inner = patch_resized.resize((inner, inner), Image.LANCZOS)
        bordered.paste(patch_inner, (border, border))

        x_offset = i * (display_size + gap)
        canvas.paste(bordered, (x_offset, 0))

        # Size label below patch
        size_text = f"{size}×{size} px"
        physical = size * XENIUM_PIXEL_SIZE
        if "merscope" in dataset_name.lower():
            physical = size * 0.108  # MERSCOPE pixel size ~0.108 µm
        size_text += f" ({physical:.1f} µm)"
        draw.text(
            (x_offset + display_size // 2, display_size + 2),
            size_text,
            fill=(180, 180, 180),
            font=font_small,
            anchor="mt",
        )

    # Cell type label at bottom
    label = f"{cell_type_fine}  ({cell_type_broad})  —  {dataset_name}"
    draw.text(
        (total_w // 2, total_h - 8),
        label,
        fill=color,
        font=font,
        anchor="mb",
    )

    return canvas


# ── Cell Selection ───────────────────────────────────────────────

def select_cells(
    cells_df: pl.DataFrame,
    image_shape: tuple[int, int],
    n_cells: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Select well-centered cells with balanced cell types."""
    h, w = image_shape
    max_half = max(PATCH_SIZES) // 2
    margin = max_half + 50  # Extra margin for nice central cells

    # Filter to well-interior cells
    interior = cells_df.filter(
        (pl.col("x_px") > margin)
        & (pl.col("x_px") < w - margin)
        & (pl.col("y_px") > margin)
        & (pl.col("y_px") < h - margin)
    )

    # Prefer cells near center of image
    cx, cy = w / 2, h / 2
    interior = interior.with_columns(
        ((pl.col("x_px") - cx) ** 2 + (pl.col("y_px") - cy) ** 2).alias("dist_center")
    )

    # Take from center 50% of image
    center_radius = min(w, h) * 0.35
    central = interior.filter(pl.col("dist_center") < center_radius**2)
    if len(central) < n_cells * 2:
        central = interior  # Fall back to all interior cells

    # Balance by cell type
    types = central["gt_broad"].unique().to_list()
    per_type = max(1, n_cells // len(types))

    rng = np.random.default_rng(seed)
    selected = []
    for ct in types:
        subset = central.filter(pl.col("gt_broad") == ct)
        n_take = min(per_type, len(subset))
        if n_take > 0:
            idx = rng.choice(len(subset), n_take, replace=False)
            selected.append(subset[idx.tolist()])

    result = pl.concat(selected)

    # If not enough, add more randomly
    if len(result) < n_cells:
        remaining = central.filter(~pl.col("cell_id_str").is_in(result["cell_id_str"]))
        n_extra = min(n_cells - len(result), len(remaining))
        if n_extra > 0:
            idx = rng.choice(len(remaining), n_extra, replace=False)
            result = pl.concat([result, remaining[idx.tolist()]])

    return result.head(n_cells)


# ── Main ─────────────────────────────────────────────────────────

def process_dataset(
    name: str,
    image: np.ndarray,
    cells_df: pl.DataFrame,
    n_cells: int,
    output_dir: Path,
    seed: int = 42,
) -> list[dict]:
    """Process one dataset: select cells, extract patches, save cascades."""
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    selected = select_cells(cells_df, image.shape[:2], n_cells, seed=seed)
    logger.info(f"  Selected {len(selected)} cells from {name}")

    # Log cell type distribution
    dist = dict(selected.group_by("gt_broad").agg(pl.len().alias("n")).iter_rows())
    logger.info(f"  Distribution: {dist}")

    metadata = []
    for i, row in enumerate(selected.iter_rows(named=True)):
        x_px = row["x_px"]
        y_px = row["y_px"]
        cell_type_fine = row["gt_fine"]
        cell_type_broad = row["gt_broad"]
        cell_id = row["cell_id_str"]

        patches = extract_multi_scale_patches(image, x_px, y_px)
        if patches is None:
            continue

        # Grayscale cascade
        gray_cascade = create_grayscale_cascade(patches)
        gray_path = ds_dir / f"{i:02d}_{cell_type_broad}_{cell_id}_grayscale.png"
        gray_cascade.save(str(gray_path))

        # Colored cascade
        color_cascade = create_colored_cascade(
            patches, cell_type_broad, cell_type_fine, name
        )
        color_path = ds_dir / f"{i:02d}_{cell_type_broad}_{cell_id}_colored.png"
        color_cascade.save(str(color_path))

        metadata.append({
            "index": i,
            "dataset": name,
            "cell_id": cell_id,
            "cell_type_fine": cell_type_fine,
            "cell_type_broad": cell_type_broad,
            "x_px": float(x_px),
            "y_px": float(y_px),
            "color_rgb": list(CELL_TYPE_COLORS.get(cell_type_broad, (150, 150, 150))),
            "fine_color_rgb": list(FINE_TYPE_COLORS.get(cell_type_fine, CELL_TYPE_COLORS.get(cell_type_broad, (150, 150, 150)))),
            "grayscale_file": gray_path.name,
            "colored_file": color_path.name,
            "patch_sizes_px": PATCH_SIZES,
        })

    logger.info(f"  Saved {len(metadata)} cascade pairs to {ds_dir}")
    return metadata


def create_legend(output_dir: Path):
    """Create a color legend image."""
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_per_dataset = {
        "rep1": 17,
        "rep2": 17,
        "merscope-breast": 16,
    }
    all_metadata = {}

    # ── Rep1 ──────────────────────────────────────────────────
    rep1_path = Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep1")
    image1, cells1 = load_xenium_data(rep1_path, "celltypes_ground_truth_rep1_supervised.xlsx")
    meta1 = process_dataset("rep1", image1, cells1, n_per_dataset["rep1"], OUTPUT_DIR, seed=42)
    all_metadata["rep1"] = meta1
    del image1  # Free memory

    # ── Rep2 ──────────────────────────────────────────────────
    rep2_path = Path("/mnt/work/datasets/raw/xenium/xenium-breast-tumor-rep2")
    image2, cells2 = load_xenium_data(rep2_path, "celltypes_ground_truth_rep2_supervised.xlsx")
    meta2 = process_dataset("rep2", image2, cells2, n_per_dataset["rep2"], OUTPUT_DIR, seed=43)
    all_metadata["rep2"] = meta2
    del image2

    # ── MERSCOPE ──────────────────────────────────────────────
    merscope_path = Path("/mnt/work/datasets/raw/merscope/merscope-breast")
    image_m, cells_m = load_merscope_data(merscope_path)
    meta_m = process_dataset("merscope-breast", image_m, cells_m, n_per_dataset["merscope-breast"], OUTPUT_DIR, seed=44)
    all_metadata["merscope-breast"] = meta_m
    del image_m

    # ── Save metadata ─────────────────────────────────────────
    meta_out = {
        "description": "Patch cascade images at 32, 64, 128, 256 px centered on annotated cells",
        "datasets": list(all_metadata.keys()),
        "cell_type_colors": {k: list(v) for k, v in CELL_TYPE_COLORS.items()},
        "fine_type_colors": {k: list(v) for k, v in FINE_TYPE_COLORS.items()},
        "display_size_px": DISPLAY_SIZE,
        "patch_sizes_px": PATCH_SIZES,
        "border_width_px": BORDER_WIDTH,
        "cells": all_metadata,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    create_legend(OUTPUT_DIR)

    total = sum(len(v) for v in all_metadata.values())
    logger.info(f"\nDone! {total} cascade pairs saved to {OUTPUT_DIR}")
    logger.info(f"  Rep1: {len(meta1)} cells")
    logger.info(f"  Rep2: {len(meta2)} cells")
    logger.info(f"  MERSCOPE: {len(meta_m)} cells")


if __name__ == "__main__":
    main()
