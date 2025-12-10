"""Patch extraction from DAPI images."""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import zarr
from loguru import logger

from dapidl.data.xenium import XeniumDataReader
from dapidl.data.annotation import CellTypeAnnotator


class PatchExtractor:
    """Extract nucleus patches from DAPI images.

    Extracts square patches centered on cell nuclei for classification training.
    """

    def __init__(
        self,
        reader: XeniumDataReader,
        patch_size: int = 128,
        confidence_threshold: float = 0.5,
        annotator: CellTypeAnnotator | None = None,
    ) -> None:
        """Initialize patch extractor.

        Args:
            reader: XeniumDataReader instance
            patch_size: Size of square patches to extract (default 128)
            confidence_threshold: Minimum annotation confidence (default 0.5)
            annotator: CellTypeAnnotator instance (created if not provided)
        """
        self.reader = reader
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        self.annotator = annotator or CellTypeAnnotator(
            confidence_threshold=confidence_threshold
        )
        self._half_size = patch_size // 2

    def extract_patch(self, image: np.ndarray, x: float, y: float) -> np.ndarray | None:
        """Extract a single patch centered on (x, y).

        Args:
            image: Full DAPI image (H, W)
            x: X coordinate of patch center (pixels)
            y: Y coordinate of patch center (pixels)

        Returns:
            Patch array of shape (patch_size, patch_size), or None if out of bounds
        """
        h, w = image.shape
        x_int, y_int = int(round(x)), int(round(y))

        # Calculate patch boundaries
        x_min = x_int - self._half_size
        x_max = x_int + self._half_size
        y_min = y_int - self._half_size
        y_max = y_int + self._half_size

        # Check bounds
        if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
            return None

        return image[y_min:y_max, x_min:x_max].copy()

    def extract_all_patches(
        self,
        image: np.ndarray,
        centroids: np.ndarray,
        cell_ids: np.ndarray,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Extract patches for all valid cell centroids.

        Args:
            image: Full DAPI image (H, W)
            centroids: Array of (x, y) coordinates in pixels
            cell_ids: Array of cell IDs

        Yields:
            Tuples of (cell_id, patch_array)
        """
        h, w = image.shape
        n_valid = 0
        n_skipped = 0

        for cell_id, (x, y) in zip(cell_ids, centroids):
            patch = self.extract_patch(image, x, y)
            if patch is not None:
                n_valid += 1
                yield cell_id, patch
            else:
                n_skipped += 1

        logger.info(
            f"Extracted {n_valid} patches, skipped {n_skipped} (out of bounds)"
        )

    def prepare_dataset(
        self,
        output_path: Path | str,
        use_all_cells: bool = False,
    ) -> dict:
        """Prepare complete dataset from Xenium data.

        Args:
            output_path: Path to save dataset
            use_all_cells: If True, include all cells regardless of confidence.
                          If False, only include cells above confidence threshold.

        Returns:
            Dictionary with dataset statistics
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Loading DAPI image...")
        image = self.reader.image
        logger.info(f"Image shape: {image.shape}")

        # Get cell annotations
        logger.info("Running cell type annotation...")
        annotations = self.annotator.annotate_from_reader(self.reader)

        # Filter by confidence if requested
        if not use_all_cells:
            annotations = self.annotator.filter_by_confidence(annotations)

        # Filter by category if requested (for fine-grained classification)
        if self.annotator.filter_category:
            annotations = self.annotator.filter_by_category(
                annotations, self.annotator.filter_category
            )

        # Get class mapping
        class_mapping = self.annotator.get_class_mapping(annotations)
        logger.info(f"Class mapping: {class_mapping}")

        # Get centroids for annotated cells
        cells_df = self.reader.cells_df
        annotated_cells = annotations.join(
            cells_df.select(["cell_id", "x_centroid_px", "y_centroid_px"]),
            on="cell_id",
        )

        n_cells = len(annotated_cells)
        logger.info(f"Processing {n_cells} cells...")

        # Pre-allocate arrays
        patches_list = []
        labels_list = []
        metadata_list = []

        # Extract patches
        cell_ids = annotated_cells["cell_id"].to_numpy()
        centroids_x = annotated_cells["x_centroid_px"].to_numpy()
        centroids_y = annotated_cells["y_centroid_px"].to_numpy()

        # Determine column names (handles single and multi-model cases)
        pred_col = "predicted_type" if "predicted_type" in annotated_cells.columns else "predicted_type_1"
        broad_col = "broad_category" if "broad_category" in annotated_cells.columns else "broad_category_1"
        conf_col = "confidence" if "confidence" in annotated_cells.columns else "confidence_1"

        predicted_types = annotated_cells[pred_col].to_list()
        broad_categories = annotated_cells[broad_col].to_list()
        confidences = annotated_cells[conf_col].to_numpy()

        # Determine which labels to use based on fine_grained mode
        use_fine_grained = getattr(self.annotator, 'fine_grained', False)
        label_source = predicted_types if use_fine_grained else broad_categories

        # First pass: collect only labels and metadata
        logger.info("First pass: extracting labels and metadata...")
        n_extracted = 0
        for i in range(n_cells):
            patch = self.extract_patch(image, centroids_x[i], centroids_y[i])
            if patch is not None:
                labels_list.append(class_mapping[label_source[i]])
                metadata_list.append(
                    {
                        "cell_id": int(cell_ids[i]),
                        "predicted_type": predicted_types[i],
                        "broad_category": broad_categories[i],
                        "confidence": float(confidences[i]),
                        "x_centroid_px": float(centroids_x[i]),
                        "y_centroid_px": float(centroids_y[i]),
                    }
                )
                n_extracted += 1

            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{n_cells} cells...")

        logger.info(f"Extracted metadata for {n_extracted} patches from {n_cells} cells")

        # Create labels array
        labels_array = np.array(labels_list, dtype=np.int64)

        # Save labels and metadata first (smaller files)
        logger.info(f"Saving labels and metadata to {output_path}...")
        np.save(output_path / "labels.npy", labels_array)
        logger.info(f"Saved labels: {labels_array.shape}")

        metadata_df = pl.DataFrame(metadata_list)
        metadata_df.write_parquet(output_path / "metadata.parquet")
        logger.info(f"Saved metadata: {len(metadata_df)} rows")

        # Save class mapping
        with open(output_path / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f, indent=2)
        logger.info("Saved class mapping")

        # Second pass: save patches in batches to Zarr
        logger.info("Second pass: saving patches to Zarr...")
        zarr_path = output_path / "patches.zarr"
        batch_size = 1000  # Match chunk size for efficient writes
        zarr_store = zarr.open(
            zarr_path,
            mode='w',
            shape=(n_extracted, self.patch_size, self.patch_size),
            dtype=np.uint16,
            chunks=(min(batch_size, n_extracted), self.patch_size, self.patch_size),
        )

        patch_idx = 0
        batch_patches = []
        batch_start_idx = 0

        for i in range(n_cells):
            patch = self.extract_patch(image, centroids_x[i], centroids_y[i])
            if patch is not None:
                batch_patches.append(patch)

                # Write batch when full
                if len(batch_patches) >= batch_size:
                    batch_array = np.stack(batch_patches, axis=0)
                    zarr_store[batch_start_idx:batch_start_idx + len(batch_patches)] = batch_array
                    patch_idx += len(batch_patches)
                    batch_start_idx = patch_idx
                    batch_patches = []
                    logger.info(f"Saved {patch_idx}/{n_extracted} patches...")

        # Write remaining patches
        if batch_patches:
            batch_array = np.stack(batch_patches, axis=0)
            zarr_store[batch_start_idx:batch_start_idx + len(batch_patches)] = batch_array
            patch_idx += len(batch_patches)
            logger.info(f"Saved {patch_idx}/{n_extracted} patches...")

        logger.info(f"Saved patches: shape={zarr_store.shape}, dtype={zarr_store.dtype}")

        # Save dataset info
        # Use correct column based on fine_grained mode
        label_column = "predicted_type" if use_fine_grained else "broad_category"
        dataset_info = {
            "n_samples": n_extracted,
            "n_classes": len(class_mapping),
            "patch_size": self.patch_size,
            "image_shape": list(image.shape),
            "confidence_threshold": self.confidence_threshold,
            "fine_grained": use_fine_grained,
            "class_distribution": {
                cat: int((metadata_df[label_column] == cat).sum())
                for cat in class_mapping.keys()
            },
        }
        with open(output_path / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Class distribution: {dataset_info['class_distribution']}")

        return dataset_info

    def extract_and_save(self, output_path: Path | str) -> dict:
        """Convenience method matching CLI interface.

        Args:
            output_path: Path to save dataset

        Returns:
            Dataset statistics
        """
        return self.prepare_dataset(output_path, use_all_cells=False)
