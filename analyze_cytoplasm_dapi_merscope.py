#!/usr/bin/env python3
"""
Analyze DAPI signal distribution in MERSCOPE data.

MERSCOPE data only provides bounding boxes, not polygon boundaries.
We estimate nucleus as a circle at the cell center (~8µm diameter typical)
and cytoplasm as the remaining area within the bounding box.

The transform matrix converts micron coordinates to pixel coordinates.
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.draw import disk
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Configuration
MERSCOPE_PATH = Path("/home/chrism/datasets/vz-ffpe-showcase/breast")
OUTPUT_DIR = Path("/home/chrism/git/dapidl/cytoplasm_analysis_merscope")
OUTPUT_DIR.mkdir(exist_ok=True)

# MERSCOPE parameters
# From transform matrix: 9.26 microns per pixel (so ~0.108 µm/pixel)
# Transform: pixel = micron * scale + offset
TRANSFORM_SCALE = 9.259316  # microns to pixels factor (actually pixels per micron)
PIXEL_SIZE_UM = 1.0 / TRANSFORM_SCALE  # ~0.108 µm/pixel

# Estimated nucleus diameter based on typical cell biology
NUCLEUS_DIAMETER_UM = 8.0  # typical nucleus diameter in microns
NUCLEUS_RADIUS_PIXELS = (NUCLEUS_DIAMETER_UM / 2) / PIXEL_SIZE_UM

N_SAMPLE_CELLS = 500
N_EXAMPLE_IMAGES = 20
PATCH_SIZE = 256  # pixels for visualization


def load_transform_matrix(path: Path) -> np.ndarray:
    """Load the micron to pixel transform matrix."""
    return np.loadtxt(path)  # Space-delimited


def micron_to_pixel(x_um: float, y_um: float, transform: np.ndarray) -> tuple:
    """Convert micron coordinates to pixel coordinates."""
    # Affine transform: [x_px, y_px, 1] = transform @ [x_um, y_um, 1]
    point = np.array([x_um, y_um, 1])
    result = transform @ point
    return int(result[0]), int(result[1])


def analyze_cell(
    dapi: np.ndarray,
    center_x_um: float,
    center_y_um: float,
    min_x_um: float,
    max_x_um: float,
    min_y_um: float,
    max_y_um: float,
    transform: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> dict | None:
    """
    Analyze DAPI intensity in nucleus, cytoplasm, and background regions.

    Since MERSCOPE only provides bounding boxes, we:
    1. Estimate nucleus as a circle at center (~8µm diameter)
    2. Cytoplasm = bounding box region - nucleus circle
    3. Background = patch outside bounding box
    """
    # Convert coordinates to pixels
    cx_px, cy_px = micron_to_pixel(center_x_um, center_y_um, transform)
    min_x_px, min_y_px = micron_to_pixel(min_x_um, min_y_um, transform)
    max_x_px, max_y_px = micron_to_pixel(max_x_um, max_y_um, transform)

    # Ensure min < max (coordinates might be flipped)
    if min_x_px > max_x_px:
        min_x_px, max_x_px = max_x_px, min_x_px
    if min_y_px > max_y_px:
        min_y_px, max_y_px = max_y_px, min_y_px

    # Define patch boundaries centered on cell
    half_size = patch_size // 2
    y_start = max(0, cy_px - half_size)
    y_end = min(dapi.shape[0], cy_px + half_size)
    x_start = max(0, cx_px - half_size)
    x_end = min(dapi.shape[1], cx_px + half_size)

    if y_end - y_start < 100 or x_end - x_start < 100:
        return None

    # Extract patch
    patch = dapi[y_start:y_end, x_start:x_end].astype(np.float32)

    # Local coordinates within patch
    local_cy = cy_px - y_start
    local_cx = cx_px - x_start
    local_min_x = max(0, min_x_px - x_start)
    local_max_x = min(patch.shape[1], max_x_px - x_start)
    local_min_y = max(0, min_y_px - y_start)
    local_max_y = min(patch.shape[0], max_y_px - y_start)

    # Create masks
    h, w = patch.shape

    # Nucleus mask: circle at center
    nucleus_mask = np.zeros((h, w), dtype=bool)
    rr, cc = disk((local_cy, local_cx), NUCLEUS_RADIUS_PIXELS, shape=(h, w))
    nucleus_mask[rr, cc] = True

    # Cell bounding box mask
    cell_mask = np.zeros((h, w), dtype=bool)
    cell_mask[local_min_y:local_max_y, local_min_x:local_max_x] = True

    # Cytoplasm = cell - nucleus
    cytoplasm_mask = cell_mask & ~nucleus_mask

    # Background = outside cell bounding box
    background_mask = ~cell_mask

    # Calculate intensities
    if nucleus_mask.sum() < 10 or cytoplasm_mask.sum() < 10 or background_mask.sum() < 10:
        return None

    nucleus_intensity = patch[nucleus_mask].mean()
    cytoplasm_intensity = patch[cytoplasm_mask].mean()
    background_intensity = patch[background_mask].mean()

    return {
        "patch": patch,
        "nucleus_mask": nucleus_mask,
        "cytoplasm_mask": cytoplasm_mask,
        "cell_mask": cell_mask,
        "background_mask": background_mask,
        "nucleus_mean": nucleus_intensity,
        "cytoplasm_mean": cytoplasm_intensity,
        "background_mean": background_intensity,
        "nucleus_max": patch[nucleus_mask].max(),
        "cytoplasm_max": patch[cytoplasm_mask].max(),
        "nuc_cyto_ratio": nucleus_intensity / max(cytoplasm_intensity, 1),
        "cyto_bg_ratio": cytoplasm_intensity / max(background_intensity, 1),
        "center_px": (cx_px, cy_px),
        "bbox_px": (min_x_px, max_x_px, min_y_px, max_y_px),
    }


def save_example_image(result: dict, cell_idx: int, output_dir: Path):
    """Save visualization of a single cell analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    patch = result["patch"]
    nuc_mask = result["nucleus_mask"]
    cyto_mask = result["cytoplasm_mask"]
    cell_mask = result["cell_mask"]

    # 1. Raw DAPI
    ax = axes[0, 0]
    im = ax.imshow(patch, cmap="gray", vmin=np.percentile(patch, 1), vmax=np.percentile(patch, 99))
    ax.set_title(f"Raw DAPI (cell {cell_idx})")
    plt.colorbar(im, ax=ax)

    # 2. Cell bbox and estimated nucleus overlay
    ax = axes[0, 1]
    ax.imshow(patch, cmap="gray", vmin=np.percentile(patch, 1), vmax=np.percentile(patch, 99))
    # Draw cell bounding box
    from matplotlib.patches import Rectangle, Circle
    h, w = patch.shape
    local_cy, local_cx = h // 2, w // 2

    # Find bbox bounds from mask
    rows = np.any(cell_mask, axis=1)
    cols = np.any(cell_mask, axis=0)
    if rows.any() and cols.any():
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         linewidth=2, edgecolor='cyan', facecolor='none', label='Cell bbox')
        ax.add_patch(rect)

    # Draw nucleus circle
    circle = Circle((local_cx, local_cy), NUCLEUS_RADIUS_PIXELS,
                   linewidth=2, edgecolor='yellow', facecolor='none', label='Est. nucleus')
    ax.add_patch(circle)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Boundaries: Cell bbox (cyan), Est. Nucleus (yellow)")

    # 3. Masks visualization
    ax = axes[0, 2]
    mask_vis = np.zeros((*patch.shape, 3), dtype=np.uint8)
    mask_vis[nuc_mask] = [0, 255, 0]  # Nucleus = green
    mask_vis[cyto_mask] = [255, 0, 0]  # Cytoplasm = red
    ax.imshow(mask_vis)
    ax.set_title("Masks: Nucleus (G), Cytoplasm (R)")

    # 4. Nucleus intensity
    ax = axes[1, 0]
    nucleus_vals = patch.copy()
    nucleus_vals[~nuc_mask] = 0
    im = ax.imshow(nucleus_vals, cmap="hot", vmin=0, vmax=np.percentile(patch, 99))
    ax.set_title(f"Nucleus: mean={result['nucleus_mean']:.0f}, max={result['nucleus_max']:.0f}")
    plt.colorbar(im, ax=ax)

    # 5. Cytoplasm intensity
    ax = axes[1, 1]
    cyto_vals = patch.copy()
    cyto_vals[~cyto_mask] = 0
    im = ax.imshow(cyto_vals, cmap="hot", vmin=0, vmax=np.percentile(patch, 99))
    ax.set_title(f"Cytoplasm: mean={result['cytoplasm_mean']:.0f}, max={result['cytoplasm_max']:.0f}")
    plt.colorbar(im, ax=ax)

    # 6. Histogram
    ax = axes[1, 2]
    bins = np.linspace(0, np.percentile(patch, 99.5), 50)
    ax.hist(patch[result["background_mask"]].flatten(), bins=bins, alpha=0.5,
            label=f"Background (mean={result['background_mean']:.0f})", density=True)
    ax.hist(patch[cyto_mask].flatten(), bins=bins, alpha=0.5,
            label=f"Cytoplasm (mean={result['cytoplasm_mean']:.0f})", density=True)
    ax.hist(patch[nuc_mask].flatten(), bins=bins, alpha=0.5,
            label=f"Nucleus (mean={result['nucleus_mean']:.0f})", density=True)
    ax.set_xlabel("DAPI Intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"Nuc/Cyto ratio: {result['nuc_cyto_ratio']:.2f}")
    ax.legend(fontsize=8)

    plt.suptitle(f"Cell {cell_idx}: Nuc/Cyto={result['nuc_cyto_ratio']:.2f}, Cyto/BG={result['cyto_bg_ratio']:.2f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"cell_{cell_idx:06d}_example.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("MERSCOPE Cytoplasm DAPI Analysis")
    print("=" * 60)

    # Load transform matrix
    transform_path = MERSCOPE_PATH / "images" / "micron_to_mosaic_pixel_transform.csv"
    transform = load_transform_matrix(transform_path)
    print(f"\nTransform matrix:\n{transform}")
    print(f"Pixel size: ~{PIXEL_SIZE_UM:.3f} µm/pixel")
    print(f"Estimated nucleus radius: {NUCLEUS_RADIUS_PIXELS:.1f} pixels ({NUCLEUS_DIAMETER_UM/2} µm)")

    # Load cell metadata
    print("\nLoading cell metadata...")
    cells = pd.read_csv(MERSCOPE_PATH / "cell_metadata.csv")
    print(f"Total cells: {len(cells):,}")
    print(f"Columns: {list(cells.columns)}")

    # Load DAPI image (memory-mapped)
    print("\nLoading DAPI image (memory-mapped)...")
    dapi_path = MERSCOPE_PATH / "images" / "mosaic_DAPI_z3.tif"
    dapi = tifffile.imread(dapi_path, aszarr=True)
    dapi = tifffile.imread(dapi_path)  # Load into memory for faster random access
    print(f"DAPI shape: {dapi.shape}, dtype: {dapi.dtype}")

    # Sample random cells
    print(f"\nSampling {N_SAMPLE_CELLS} random cells...")
    sample_indices = np.random.choice(len(cells), size=min(N_SAMPLE_CELLS, len(cells)), replace=False)
    sample_cells = cells.iloc[sample_indices]

    # Analyze cells
    results = []
    example_count = 0

    for idx, (_, row) in enumerate(tqdm(sample_cells.iterrows(), total=len(sample_cells), desc="Analyzing cells")):
        result = analyze_cell(
            dapi,
            center_x_um=row["center_x"],
            center_y_um=row["center_y"],
            min_x_um=row["min_x"],
            max_x_um=row["max_x"],
            min_y_um=row["min_y"],
            max_y_um=row["max_y"],
            transform=transform,
        )

        if result is None:
            continue

        # Save example images
        if example_count < N_EXAMPLE_IMAGES:
            save_example_image(result, idx + 1, OUTPUT_DIR)
            example_count += 1

        # Store statistics (without large arrays)
        results.append({
            "cell_idx": idx,
            "nucleus_mean": result["nucleus_mean"],
            "cytoplasm_mean": result["cytoplasm_mean"],
            "background_mean": result["background_mean"],
            "nucleus_max": result["nucleus_max"],
            "cytoplasm_max": result["cytoplasm_max"],
            "nuc_cyto_ratio": result["nuc_cyto_ratio"],
            "cyto_bg_ratio": result["cyto_bg_ratio"],
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    print(f"\nAnalyzed {len(df)} cells successfully")

    # Save statistics
    df.to_csv(OUTPUT_DIR / "cell_statistics.csv", index=False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nNucleus mean intensity:   {df['nucleus_mean'].mean():.1f} ± {df['nucleus_mean'].std():.1f}")
    print(f"Cytoplasm mean intensity: {df['cytoplasm_mean'].mean():.1f} ± {df['cytoplasm_mean'].std():.1f}")
    print(f"Background mean intensity:{df['background_mean'].mean():.1f} ± {df['background_mean'].std():.1f}")
    print(f"\nNuc/Cyto ratio:           {df['nuc_cyto_ratio'].mean():.2f} (median: {df['nuc_cyto_ratio'].median():.2f})")
    print(f"Cyto/BG ratio:            {df['cyto_bg_ratio'].mean():.2f} (median: {df['cyto_bg_ratio'].median():.2f})")

    cyto_signal = df['cytoplasm_mean'].mean() / df['nucleus_mean'].mean() * 100
    print(f"\nCytoplasm signal: {cyto_signal:.1f}% of nuclear signal (above background)")

    if df['cyto_bg_ratio'].median() < 1.0:
        print("\n*** Cytoplasm is DARKER than background (same as Xenium) ***")
        print("*** No significant extra-nuclear DAPI signal detected ***")
    else:
        print("\n*** Cytoplasm is BRIGHTER than background ***")
        print("*** Possible extra-nuclear DAPI signal detected ***")

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution of mean intensities
    ax = axes[0, 0]
    ax.hist(df["background_mean"], bins=50, alpha=0.5, label="Background", density=True)
    ax.hist(df["cytoplasm_mean"], bins=50, alpha=0.5, label="Cytoplasm", density=True)
    ax.hist(df["nucleus_mean"], bins=50, alpha=0.5, label="Nucleus", density=True)
    ax.set_xlabel("Mean DAPI Intensity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mean Intensities by Region")
    ax.legend()

    # 2. Nucleus vs Cytoplasm scatter
    ax = axes[0, 1]
    ax.scatter(df["nucleus_mean"], df["cytoplasm_mean"], alpha=0.5, s=10)
    ax.plot([0, df["nucleus_mean"].max()], [0, df["nucleus_mean"].max()], 'r--', label="1:1 line")
    ax.set_xlabel("Nucleus Mean Intensity")
    ax.set_ylabel("Cytoplasm Mean Intensity")
    ax.set_title("Nucleus vs Cytoplasm Intensity")
    ax.legend()

    # 3. Nuc/Cyto ratio distribution
    ax = axes[1, 0]
    ax.hist(df["nuc_cyto_ratio"], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(df["nuc_cyto_ratio"].mean(), color='r', linestyle='--', label=f"Mean: {df['nuc_cyto_ratio'].mean():.2f}")
    ax.axvline(df["nuc_cyto_ratio"].median(), color='g', linestyle='--', label=f"Median: {df['nuc_cyto_ratio'].median():.2f}")
    ax.set_xlabel("Nucleus / Cytoplasm Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Nuc/Cyto Intensity Ratio")
    ax.legend()

    # 4. Box plot comparison
    ax = axes[1, 1]
    data = [df["background_mean"], df["cytoplasm_mean"], df["nucleus_mean"]]
    bp = ax.boxplot(data, labels=["Background", "Cytoplasm", "Nucleus"])
    ax.set_ylabel("Mean DAPI Intensity")
    ax.set_title("Intensity Comparison by Region")

    # Add key findings text
    key_findings = f"""Key Findings (MERSCOPE):
• Cytoplasm signal: {cyto_signal:.1f}% of nuclear signal (above background)
• Mean Nuc/Cyto ratio: {df['nuc_cyto_ratio'].mean():.2f}
• Cyto/BG ratio: {df['cyto_bg_ratio'].mean():.2f} ({'< 1.0 = darker' if df['cyto_bg_ratio'].mean() < 1.0 else '>= 1.0 = brighter'})
• N cells analyzed: {len(df)}
• Note: Using estimated ~8µm nucleus diameter (no polygon boundaries available)"""

    fig.text(0.02, 0.02, key_findings, fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
             verticalalignment="bottom")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(OUTPUT_DIR / "summary_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Example images: {example_count}")
    print("Summary: summary_statistics.png")


if __name__ == "__main__":
    main()
