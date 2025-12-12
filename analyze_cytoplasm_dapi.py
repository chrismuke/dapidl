#!/usr/bin/env python3
"""
Analyze DAPI signal distribution: nucleus vs cytoplasm vs background.

This script investigates whether there is measurable DAPI signal in the cytoplasm
(outside nucleus but inside cell boundary), which could indicate:
- RNA staining (weak DAPI binding to RNA)
- Mitochondrial DNA
- Background/autofluorescence
- Nuclear signal bleed-through
"""

import numpy as np
import polars as pl
import tifffile
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from skimage.draw import polygon as draw_polygon
import warnings
warnings.filterwarnings('ignore')

# Configuration
XENIUM_PATH = Path("/home/chrism/datasets/xenium_breast_tumor/outs")
OUTPUT_DIR = Path("/home/chrism/git/dapidl/cytoplasm_analysis")
PIXEL_SIZE = 0.2125  # µm/pixel for Xenium
N_SAMPLE_CELLS = 500  # Number of cells for statistical analysis
N_EXAMPLE_IMAGES = 20  # Number of example images to save
PATCH_SIZE = 256  # Size of example patches

OUTPUT_DIR.mkdir(exist_ok=True)


def load_dapi_image():
    """Load DAPI morphology image."""
    image_path = XENIUM_PATH / "morphology_focus.ome.tif"
    print(f"Loading DAPI image from {image_path}...")
    with tifffile.TiffFile(image_path) as tif:
        image = tif.asarray()
    print(f"  Shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.1f}")
    return image


def load_boundaries(parquet_path: Path, pixel_size: float = PIXEL_SIZE) -> dict:
    """Load cell/nucleus boundaries and convert to pixel coordinates."""
    print(f"Loading boundaries from {parquet_path}...")
    df = pl.read_parquet(parquet_path)

    # Convert to pixel coordinates
    df = df.with_columns([
        (pl.col("vertex_x") / pixel_size).alias("vertex_x_px"),
        (pl.col("vertex_y") / pixel_size).alias("vertex_y_px"),
    ])

    # Group by cell_id to get polygon vertices
    boundaries = {}
    for cell_id in df["cell_id"].unique().to_list():
        cell_df = df.filter(pl.col("cell_id") == cell_id)
        vertices = list(zip(
            cell_df["vertex_x_px"].to_list(),
            cell_df["vertex_y_px"].to_list()
        ))
        if len(vertices) >= 3:
            boundaries[cell_id] = vertices

    print(f"  Loaded {len(boundaries)} polygons")
    return boundaries


def load_cell_metadata():
    """Load cell metadata including centroids and areas."""
    cells_path = XENIUM_PATH / "cells.parquet"
    print(f"Loading cell metadata from {cells_path}...")
    df = pl.read_parquet(cells_path)

    # Convert centroids to pixels
    df = df.with_columns([
        (pl.col("x_centroid") / PIXEL_SIZE).alias("x_px"),
        (pl.col("y_centroid") / PIXEL_SIZE).alias("y_px"),
    ])

    print(f"  Loaded {len(df)} cells")
    return df


def create_mask_from_polygon(vertices: list, shape: tuple) -> np.ndarray:
    """Create a binary mask from polygon vertices."""
    mask = np.zeros(shape, dtype=bool)
    if len(vertices) < 3:
        return mask

    xs, ys = zip(*vertices)
    rr, cc = draw_polygon(ys, xs, shape=shape)
    mask[rr, cc] = True
    return mask


def analyze_single_cell(
    dapi_image: np.ndarray,
    cell_id: int,
    cell_boundary: list,
    nucleus_boundary: list,
    patch_size: int = PATCH_SIZE
) -> dict | None:
    """Analyze DAPI intensity for a single cell."""

    # Get bounding box
    cell_xs, cell_ys = zip(*cell_boundary)
    cx, cy = np.mean(cell_xs), np.mean(cell_ys)

    # Check bounds
    h, w = dapi_image.shape
    half = patch_size // 2
    x_min, x_max = int(cx - half), int(cx + half)
    y_min, y_max = int(cy - half), int(cy + half)

    if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
        return None

    # Extract patch
    patch = dapi_image[y_min:y_max, x_min:x_max].astype(np.float32)

    # Shift boundaries to patch coordinates
    cell_verts_local = [(x - x_min, y - y_min) for x, y in cell_boundary]
    nuc_verts_local = [(x - x_min, y - y_min) for x, y in nucleus_boundary]

    # Create masks
    cell_mask = create_mask_from_polygon(cell_verts_local, (patch_size, patch_size))
    nuc_mask = create_mask_from_polygon(nuc_verts_local, (patch_size, patch_size))

    # Cytoplasm = cell minus nucleus
    cyto_mask = cell_mask & ~nuc_mask

    # Background = outside cell (but within patch)
    bg_mask = ~cell_mask

    # Calculate intensities
    nuc_pixels = patch[nuc_mask]
    cyto_pixels = patch[cyto_mask]
    bg_pixels = patch[bg_mask]

    if len(nuc_pixels) < 10 or len(cyto_pixels) < 10:
        return None

    result = {
        "cell_id": cell_id,
        "centroid_x": cx,
        "centroid_y": cy,
        # Nucleus stats
        "nuc_mean": np.mean(nuc_pixels),
        "nuc_std": np.std(nuc_pixels),
        "nuc_median": np.median(nuc_pixels),
        "nuc_max": np.max(nuc_pixels),
        "nuc_area_px": np.sum(nuc_mask),
        # Cytoplasm stats
        "cyto_mean": np.mean(cyto_pixels),
        "cyto_std": np.std(cyto_pixels),
        "cyto_median": np.median(cyto_pixels),
        "cyto_max": np.max(cyto_pixels),
        "cyto_area_px": np.sum(cyto_mask),
        # Background stats
        "bg_mean": np.mean(bg_pixels),
        "bg_std": np.std(bg_pixels),
        "bg_median": np.median(bg_pixels),
        # Ratios
        "nuc_to_cyto_ratio": np.mean(nuc_pixels) / max(np.mean(cyto_pixels), 1),
        "cyto_to_bg_ratio": np.mean(cyto_pixels) / max(np.mean(bg_pixels), 1),
        "nuc_to_bg_ratio": np.mean(nuc_pixels) / max(np.mean(bg_pixels), 1),
        # For visualization
        "_patch": patch,
        "_cell_mask": cell_mask,
        "_nuc_mask": nuc_mask,
        "_cyto_mask": cyto_mask,
        "_cell_verts": cell_verts_local,
        "_nuc_verts": nuc_verts_local,
    }

    return result


def save_example_image(result: dict, output_path: Path):
    """Save visualization of a single cell analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    patch = result["_patch"]
    cell_mask = result["_cell_mask"]
    nuc_mask = result["_nuc_mask"]
    cyto_mask = result["_cyto_mask"]

    # Normalize for display
    vmin, vmax = np.percentile(patch, [1, 99])

    # 1. Raw DAPI
    ax = axes[0, 0]
    im = ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"Raw DAPI (cell {result['cell_id']})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. With boundaries overlaid
    ax = axes[0, 1]
    ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
    # Draw cell boundary
    cell_verts = result["_cell_verts"]
    nuc_verts = result["_nuc_verts"]
    if len(cell_verts) > 2:
        cell_poly = MplPolygon(cell_verts, fill=False, edgecolor='cyan', linewidth=2, label='Cell')
        ax.add_patch(cell_poly)
    if len(nuc_verts) > 2:
        nuc_poly = MplPolygon(nuc_verts, fill=False, edgecolor='yellow', linewidth=2, label='Nucleus')
        ax.add_patch(nuc_poly)
    ax.legend(loc='upper right')
    ax.set_title("Boundaries: Cell (cyan), Nucleus (yellow)")

    # 3. Segmentation masks
    ax = axes[0, 2]
    rgb_mask = np.zeros((*patch.shape, 3), dtype=np.float32)
    rgb_mask[nuc_mask, 0] = 1  # Red = nucleus
    rgb_mask[cyto_mask, 1] = 1  # Green = cytoplasm
    rgb_mask[~cell_mask, 2] = 0.3  # Blue tint = background
    ax.imshow(rgb_mask)
    ax.set_title("Masks: Nucleus (R), Cytoplasm (G)")

    # 4. Nucleus region intensity
    ax = axes[1, 0]
    nuc_display = np.where(nuc_mask, patch, np.nan)
    im = ax.imshow(nuc_display, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f"Nucleus: mean={result['nuc_mean']:.0f}, max={result['nuc_max']:.0f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 5. Cytoplasm region intensity
    ax = axes[1, 1]
    cyto_display = np.where(cyto_mask, patch, np.nan)
    im = ax.imshow(cyto_display, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(f"Cytoplasm: mean={result['cyto_mean']:.0f}, max={result['cyto_max']:.0f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 6. Intensity histogram
    ax = axes[1, 2]
    nuc_pixels = patch[nuc_mask]
    cyto_pixels = patch[cyto_mask]
    bg_pixels = patch[~cell_mask]

    bins = np.linspace(0, max(patch.max(), 1), 50)
    ax.hist(bg_pixels, bins=bins, alpha=0.5, label=f'Background (mean={result["bg_mean"]:.0f})', density=True)
    ax.hist(cyto_pixels, bins=bins, alpha=0.5, label=f'Cytoplasm (mean={result["cyto_mean"]:.0f})', density=True)
    ax.hist(nuc_pixels, bins=bins, alpha=0.5, label=f'Nucleus (mean={result["nuc_mean"]:.0f})', density=True)
    ax.set_xlabel("DAPI Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"Nuc/Cyto ratio: {result['nuc_to_cyto_ratio']:.2f}")

    plt.suptitle(
        f"Cell {result['cell_id']}: Nuc/Cyto={result['nuc_to_cyto_ratio']:.2f}, "
        f"Cyto/BG={result['cyto_to_bg_ratio']:.2f}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("DAPI Signal Analysis: Nucleus vs Cytoplasm vs Background")
    print("=" * 60)

    # Load data
    dapi_image = load_dapi_image()
    cell_bounds = load_boundaries(XENIUM_PATH / "cell_boundaries.parquet")
    nuc_bounds = load_boundaries(XENIUM_PATH / "nucleus_boundaries.parquet")
    cell_meta = load_cell_metadata()

    # Find cells that have both boundaries
    common_cells = set(cell_bounds.keys()) & set(nuc_bounds.keys())
    print(f"\nCells with both boundaries: {len(common_cells)}")

    # Sample cells for analysis
    sample_cells = list(common_cells)[:N_SAMPLE_CELLS]
    print(f"Analyzing {len(sample_cells)} cells...")

    results = []
    example_results = []

    for i, cell_id in enumerate(sample_cells):
        if i % 100 == 0:
            print(f"  Processing cell {i}/{len(sample_cells)}...")

        result = analyze_single_cell(
            dapi_image,
            cell_id,
            cell_bounds[cell_id],
            nuc_bounds[cell_id]
        )

        if result is not None:
            # Store stats (without large arrays)
            stats = {k: v for k, v in result.items() if not k.startswith("_")}
            results.append(stats)

            # Store some examples for visualization
            if len(example_results) < N_EXAMPLE_IMAGES:
                example_results.append(result)

    print(f"\nSuccessfully analyzed {len(results)} cells")

    # Convert to DataFrame
    df = pl.DataFrame(results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print("\n--- Mean Intensities ---")
    print(f"  Nucleus:    {df['nuc_mean'].mean():.1f} ± {df['nuc_mean'].std():.1f}")
    print(f"  Cytoplasm:  {df['cyto_mean'].mean():.1f} ± {df['cyto_mean'].std():.1f}")
    print(f"  Background: {df['bg_mean'].mean():.1f} ± {df['bg_mean'].std():.1f}")

    print("\n--- Intensity Ratios ---")
    print(f"  Nucleus / Cytoplasm:  {df['nuc_to_cyto_ratio'].mean():.2f} ± {df['nuc_to_cyto_ratio'].std():.2f}")
    print(f"  Cytoplasm / Background: {df['cyto_to_bg_ratio'].mean():.2f} ± {df['cyto_to_bg_ratio'].std():.2f}")
    print(f"  Nucleus / Background:   {df['nuc_to_bg_ratio'].mean():.2f} ± {df['nuc_to_bg_ratio'].std():.2f}")

    print("\n--- Areas (pixels) ---")
    print(f"  Nucleus area:    {df['nuc_area_px'].mean():.0f} ± {df['nuc_area_px'].std():.0f}")
    print(f"  Cytoplasm area:  {df['cyto_area_px'].mean():.0f} ± {df['cyto_area_px'].std():.0f}")

    # Calculate signal above background
    cyto_above_bg = df["cyto_mean"].mean() - df["bg_mean"].mean()
    nuc_above_bg = df["nuc_mean"].mean() - df["bg_mean"].mean()
    cyto_signal_pct = (cyto_above_bg / nuc_above_bg) * 100 if nuc_above_bg > 0 else 0

    print("\n--- Signal Above Background ---")
    print(f"  Nucleus signal above BG:    {nuc_above_bg:.1f}")
    print(f"  Cytoplasm signal above BG:  {cyto_above_bg:.1f}")
    print(f"  Cytoplasm as % of nuclear signal: {cyto_signal_pct:.1f}%")

    # Save example images
    print(f"\nSaving {len(example_results)} example images to {OUTPUT_DIR}...")
    for i, result in enumerate(example_results):
        output_path = OUTPUT_DIR / f"cell_{result['cell_id']:06d}_example.png"
        save_example_image(result, output_path)

    # Create summary figure
    print("Creating summary plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution of mean intensities
    ax = axes[0, 0]
    ax.hist(df["bg_mean"].to_numpy(), bins=50, alpha=0.5, label='Background')
    ax.hist(df["cyto_mean"].to_numpy(), bins=50, alpha=0.5, label='Cytoplasm')
    ax.hist(df["nuc_mean"].to_numpy(), bins=50, alpha=0.5, label='Nucleus')
    ax.set_xlabel("Mean DAPI Intensity")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Distribution of Mean Intensities by Region")

    # 2. Scatter: Nucleus vs Cytoplasm intensity
    ax = axes[0, 1]
    ax.scatter(df["nuc_mean"].to_numpy(), df["cyto_mean"].to_numpy(), alpha=0.3, s=10)
    ax.plot([0, df["nuc_mean"].max()], [0, df["nuc_mean"].max()], 'r--', label='1:1 line')
    ax.set_xlabel("Nucleus Mean Intensity")
    ax.set_ylabel("Cytoplasm Mean Intensity")
    ax.legend()
    ax.set_title("Nucleus vs Cytoplasm Intensity")

    # 3. Distribution of Nuc/Cyto ratio
    ax = axes[1, 0]
    ratios = df["nuc_to_cyto_ratio"].to_numpy()
    ax.hist(ratios, bins=50, edgecolor='black')
    ax.axvline(np.mean(ratios), color='r', linestyle='--', label=f'Mean: {np.mean(ratios):.2f}')
    ax.axvline(np.median(ratios), color='g', linestyle='--', label=f'Median: {np.median(ratios):.2f}')
    ax.set_xlabel("Nucleus / Cytoplasm Ratio")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Distribution of Nuc/Cyto Intensity Ratio")

    # 4. Box plot comparison
    ax = axes[1, 1]
    data = [df["bg_mean"].to_numpy(), df["cyto_mean"].to_numpy(), df["nuc_mean"].to_numpy()]
    bp = ax.boxplot(data, labels=['Background', 'Cytoplasm', 'Nucleus'])
    ax.set_ylabel("Mean DAPI Intensity")
    ax.set_title("Intensity Comparison by Region")

    # Add text with key findings
    textstr = f"""Key Findings:
• Cytoplasm signal: {cyto_signal_pct:.1f}% of nuclear signal (above background)
• Mean Nuc/Cyto ratio: {df['nuc_to_cyto_ratio'].mean():.2f}
• Cytoplasm clearly above background: {df['cyto_to_bg_ratio'].mean():.2f}x
• N cells analyzed: {len(df)}"""

    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    summary_path = OUTPUT_DIR / "summary_statistics.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {summary_path}")

    # Save statistics to CSV
    csv_path = OUTPUT_DIR / "cell_statistics.csv"
    df.write_csv(csv_path)
    print(f"  Saved: {csv_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files created:")
    print(f"  - summary_statistics.png")
    print(f"  - cell_statistics.csv")
    print(f"  - {N_EXAMPLE_IMAGES} example cell images")


if __name__ == "__main__":
    main()
