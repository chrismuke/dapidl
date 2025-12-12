#!/usr/bin/env python3
"""
Visualize MERSCOPE patches with cell boundary overlays.
"""

import numpy as np
import pandas as pd
import tifffile
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random

# Configuration
MERSCOPE_PATH = Path("/home/chrism/datasets/vz-ffpe-showcase/breast")
BOUNDARY_FILE = Path("/tmp/merscope_boundary_sample.hdf5")
OUTPUT_DIR = Path("/home/chrism/git/dapidl/merscope_patch_examples")
OUTPUT_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 256  # pixels
Z_INDEX = 3  # Middle z-plane


def load_transform_matrix(path: Path) -> np.ndarray:
    """Load the micron to pixel transform matrix."""
    return np.loadtxt(path)


def micron_to_pixel(coords_um: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Convert micron coordinates to pixel coordinates."""
    # coords_um: (N, 2) array of [x, y] in microns
    # Add homogeneous coordinate
    ones = np.ones((coords_um.shape[0], 1))
    coords_h = np.hstack([coords_um, ones])  # (N, 3)
    # Apply transform
    coords_px = (transform @ coords_h.T).T  # (N, 3)
    return coords_px[:, :2]  # (N, 2) pixel coordinates


def get_cells_from_hdf5(h5_path: Path, z_index: int = 3) -> dict:
    """Extract cell boundaries from HDF5 file."""
    cells = {}
    with h5py.File(h5_path, 'r') as f:
        for cell_id in f['featuredata'].keys():
            z_key = f'zIndex_{z_index}'
            try:
                coords = f[f'featuredata/{cell_id}/{z_key}/p_0/coordinates'][:]
                # Shape is (1, N, 2) - squeeze to (N, 2)
                coords = coords.squeeze()
                cells[int(cell_id)] = coords
            except KeyError:
                continue
    return cells


def main():
    print("Loading transform matrix...")
    transform_path = MERSCOPE_PATH / "images" / "micron_to_mosaic_pixel_transform.csv"
    transform = load_transform_matrix(transform_path)

    print("Loading cell metadata...")
    cells_meta = pd.read_csv(MERSCOPE_PATH / "cell_metadata.csv")

    print("Loading cell boundaries from HDF5...")
    cell_boundaries = get_cells_from_hdf5(BOUNDARY_FILE, z_index=Z_INDEX)
    print(f"Found {len(cell_boundaries)} cells with boundaries")

    print("Loading DAPI image...")
    dapi_path = MERSCOPE_PATH / "images" / "mosaic_DAPI_z3.tif"
    dapi = tifffile.imread(dapi_path)
    print(f"DAPI shape: {dapi.shape}")

    # Get cells that have boundaries
    available_cells = list(cell_boundaries.keys())

    # Sample 10 random cells
    sample_cells = random.sample(available_cells, min(10, len(available_cells)))

    print(f"\nGenerating {len(sample_cells)} patch visualizations...")

    for i, cell_id in enumerate(sample_cells):
        # Get cell metadata
        cell_row = cells_meta[cells_meta['Unnamed: 0'] == cell_id]
        if len(cell_row) == 0:
            print(f"  Cell {cell_id}: metadata not found, skipping")
            continue

        cell_row = cell_row.iloc[0]
        center_x_um = cell_row['center_x']
        center_y_um = cell_row['center_y']

        # Convert center to pixels
        center_px = micron_to_pixel(np.array([[center_x_um, center_y_um]]), transform)[0]
        cx_px, cy_px = int(center_px[0]), int(center_px[1])

        # Get patch boundaries
        half_size = PATCH_SIZE // 2
        y_start = max(0, cy_px - half_size)
        y_end = min(dapi.shape[0], cy_px + half_size)
        x_start = max(0, cx_px - half_size)
        x_end = min(dapi.shape[1], cx_px + half_size)

        if y_end - y_start < 100 or x_end - x_start < 100:
            print(f"  Cell {cell_id}: patch too small, skipping")
            continue

        # Extract patch
        patch = dapi[y_start:y_end, x_start:x_end]

        # Get all cell boundaries that might be in this patch
        patch_polygons = []
        patch_colors = []

        for cid, boundary_um in cell_boundaries.items():
            # Convert boundary to pixels
            boundary_px = micron_to_pixel(boundary_um, transform)

            # Check if any vertex is within the patch
            in_patch = (
                (boundary_px[:, 0] >= x_start) & (boundary_px[:, 0] < x_end) &
                (boundary_px[:, 1] >= y_start) & (boundary_px[:, 1] < y_end)
            )

            if in_patch.any():
                # Convert to patch-local coordinates
                local_boundary = boundary_px - np.array([x_start, y_start])
                patch_polygons.append(local_boundary)
                # Highlight the target cell
                if cid == cell_id:
                    patch_colors.append('yellow')
                else:
                    patch_colors.append('cyan')

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Raw DAPI
        ax = axes[0]
        vmin, vmax = np.percentile(patch, [1, 99])
        ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"Raw DAPI (cell {cell_id})")
        ax.axis('off')

        # Right: DAPI with boundaries
        ax = axes[1]
        ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)

        # Draw polygons
        for poly, color in zip(patch_polygons, patch_colors):
            # Close the polygon
            poly_closed = np.vstack([poly, poly[0]])
            ax.plot(poly_closed[:, 0], poly_closed[:, 1],
                   color=color, linewidth=2 if color == 'yellow' else 1,
                   alpha=1.0 if color == 'yellow' else 0.7)

        ax.set_title(f"Cell boundaries (target=yellow, {len(patch_polygons)} cells)")
        ax.axis('off')

        plt.suptitle(f"MERSCOPE Patch - Cell {cell_id} (center: {cx_px}, {cy_px})")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"patch_{i+1:02d}_cell_{cell_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path.name} ({len(patch_polygons)} cells visible)")

    print(f"\nDone! Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
