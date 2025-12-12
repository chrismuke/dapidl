#!/usr/bin/env python3
"""
Segment nuclei in MERSCOPE DAPI images using Cellpose.

Cellpose 4.0 uses the CPSAM model by default which works well for nuclei.
We'll process patches around cells and save the segmentation masks.
"""

import numpy as np
import pandas as pd
import tifffile
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cellpose import models
import random
from tqdm import tqdm
import pickle

# Configuration
MERSCOPE_PATH = Path("/home/chrism/datasets/vz-ffpe-showcase/breast")
BOUNDARY_FILE = Path("/tmp/merscope_boundary_sample.hdf5")
OUTPUT_DIR = Path("/home/chrism/git/dapidl/merscope_nuclei_segmentation")
OUTPUT_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 512  # Larger patches for better context
N_PATCHES = 20
Z_INDEX = 3

# Cellpose parameters for nuclei
NUCLEUS_DIAMETER = 70  # Expected nucleus diameter in pixels (~7.5µm at 0.108µm/pixel)


def load_transform_matrix(path: Path) -> np.ndarray:
    """Load the micron to pixel transform matrix."""
    return np.loadtxt(path)


def micron_to_pixel(coords_um: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Convert micron coordinates to pixel coordinates."""
    ones = np.ones((coords_um.shape[0], 1))
    coords_h = np.hstack([coords_um, ones])
    coords_px = (transform @ coords_h.T).T
    return coords_px[:, :2]


def get_cells_from_hdf5(h5_path: Path, z_index: int = 3) -> dict:
    """Extract cell boundaries from HDF5 file."""
    cells = {}
    with h5py.File(h5_path, 'r') as f:
        for cell_id in f['featuredata'].keys():
            z_key = f'zIndex_{z_index}'
            try:
                coords = f[f'featuredata/{cell_id}/{z_key}/p_0/coordinates'][:]
                coords = coords.squeeze()
                cells[int(cell_id)] = coords
            except KeyError:
                continue
    return cells


def create_random_colormap(n_labels):
    """Create a random colormap for label visualization."""
    np.random.seed(42)
    colors = np.random.rand(n_labels + 1, 4)
    colors[0] = [0, 0, 0, 0]  # Background is transparent
    colors[:, 3] = 0.6  # Set alpha
    return ListedColormap(colors)


def main():
    print("=" * 60)
    print("MERSCOPE Nucleus Segmentation with Cellpose")
    print("=" * 60)

    # Load transform matrix
    print("\nLoading transform matrix...")
    transform_path = MERSCOPE_PATH / "images" / "micron_to_mosaic_pixel_transform.csv"
    transform = load_transform_matrix(transform_path)

    # Load cell metadata
    print("Loading cell metadata...")
    cells_meta = pd.read_csv(MERSCOPE_PATH / "cell_metadata.csv")

    # Load cell boundaries
    print("Loading cell boundaries from HDF5...")
    cell_boundaries = get_cells_from_hdf5(BOUNDARY_FILE, z_index=Z_INDEX)
    print(f"Found {len(cell_boundaries)} cells with boundaries")

    # Load DAPI image
    print("Loading DAPI image...")
    dapi_path = MERSCOPE_PATH / "images" / "mosaic_DAPI_z3.tif"
    dapi = tifffile.imread(dapi_path)
    print(f"DAPI shape: {dapi.shape}, dtype: {dapi.dtype}")

    # Initialize Cellpose model
    print("\nInitializing Cellpose model (CPSAM for nuclei)...")
    model = models.CellposeModel(gpu=True)
    print("Model loaded!")

    # Get cells that have boundaries
    available_cells = list(cell_boundaries.keys())

    # Sample random cells for patch centers
    sample_cells = random.sample(available_cells, min(N_PATCHES, len(available_cells)))

    print(f"\nProcessing {len(sample_cells)} patches...")

    all_results = []

    for i, cell_id in enumerate(tqdm(sample_cells, desc="Segmenting nuclei")):
        # Get cell metadata
        cell_row = cells_meta[cells_meta['Unnamed: 0'] == cell_id]
        if len(cell_row) == 0:
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

        if y_end - y_start < 200 or x_end - x_start < 200:
            continue

        # Extract patch
        patch = dapi[y_start:y_end, x_start:x_end].astype(np.float32)

        # Normalize patch for cellpose
        patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

        # Run Cellpose segmentation
        masks, flows, styles = model.eval(
            patch_norm,
            diameter=NUCLEUS_DIAMETER,
            channels=[0, 0],  # Grayscale
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        n_nuclei = masks.max()

        # Get cell boundaries in this patch
        patch_cell_boundaries = []
        for cid, boundary_um in cell_boundaries.items():
            boundary_px = micron_to_pixel(boundary_um, transform)
            in_patch = (
                (boundary_px[:, 0] >= x_start) & (boundary_px[:, 0] < x_end) &
                (boundary_px[:, 1] >= y_start) & (boundary_px[:, 1] < y_end)
            )
            if in_patch.any():
                local_boundary = boundary_px - np.array([x_start, y_start])
                is_target = (cid == cell_id)
                patch_cell_boundaries.append((local_boundary, is_target))

        # Store results
        result = {
            'cell_id': cell_id,
            'patch_coords': (x_start, y_start, x_end, y_end),
            'n_nuclei': n_nuclei,
            'masks': masks,
            'patch': patch,
            'cell_boundaries': patch_cell_boundaries,
        }
        all_results.append(result)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Raw DAPI
        ax = axes[0]
        vmin, vmax = np.percentile(patch, [1, 99])
        ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f"Raw DAPI")
        ax.axis('off')

        # 2. Cellpose nuclei masks
        ax = axes[1]
        ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
        if n_nuclei > 0:
            cmap = create_random_colormap(n_nuclei)
            ax.imshow(masks, cmap=cmap, interpolation='nearest')
        ax.set_title(f"Cellpose Nuclei ({n_nuclei} detected)")
        ax.axis('off')

        # 3. Nuclei + Cell boundaries overlay
        ax = axes[2]
        ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax)
        if n_nuclei > 0:
            # Draw nucleus contours
            from skimage import measure
            for label_id in range(1, n_nuclei + 1):
                contours = measure.find_contours(masks == label_id, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'lime', linewidth=1.5)

        # Draw cell boundaries
        for boundary, is_target in patch_cell_boundaries:
            color = 'yellow' if is_target else 'cyan'
            lw = 2 if is_target else 1
            boundary_closed = np.vstack([boundary, boundary[0]])
            ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
                   color=color, linewidth=lw, alpha=0.8)

        ax.set_title(f"Nuclei (green) + Cells (cyan/yellow)")
        ax.axis('off')

        plt.suptitle(f"Patch {i+1}: Cell {cell_id} - {n_nuclei} nuclei detected")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"patch_{i+1:02d}_cell_{cell_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Save masks to pickle for later use
    print(f"\nSaving segmentation results...")
    masks_data = {
        'results': [{
            'cell_id': r['cell_id'],
            'patch_coords': r['patch_coords'],
            'n_nuclei': r['n_nuclei'],
            'masks': r['masks'],
        } for r in all_results]
    }
    with open(OUTPUT_DIR / 'nucleus_masks.pkl', 'wb') as f:
        pickle.dump(masks_data, f)

    # Summary statistics
    total_nuclei = sum(r['n_nuclei'] for r in all_results)
    avg_nuclei = total_nuclei / len(all_results) if all_results else 0

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Patches processed: {len(all_results)}")
    print(f"Total nuclei detected: {total_nuclei}")
    print(f"Average nuclei per patch: {avg_nuclei:.1f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
