"""PyTorch Dataset over the STHELAR instance-segmentation tile cache.

Reads the layout produced by `dapidl.data.sthelar_tile_cache.build_cache`:

    cache_root/
        manifest.parquet           # all-slide aggregate
        s0/
            images.zarr            # (n_tiles, 1024, 1024) uint16 DAPI
            instances.zarr         # (n_tiles, 1024, 1024) uint16 instance IDs
            labels.parquet         # one row per (tile_idx, instance_id, cell_id)
        s1/ ... s3/ ... s6/ ...

Each `__getitem__` returns a dict with the image tile, instance map, and three
levels of class labels (fine / medium / broad) per instance, plus boxes and
border flags. Per-image normalization (percentile p1→0, p99→1, then
standardize) is applied at __getitem__ time, matching the Xenium-calibrated
pattern from `src/dapidl/data/transforms.py`.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
import zarr
from loguru import logger
from torch.utils.data import Dataset


class STHELARInstanceDataset(Dataset):
    """Tile-cache backed dataset for joint instance seg + classification.

    Args:
        cache_root: path with one subdir per slide.
        slides: which slides to include (e.g. ["breast_s0", "breast_s1"]).
        split: "train" | "val" | "test".
        fine_classes: ordered list of fine class names (label index = position).
        medium_classes: ordered list of medium class names.
        broad_classes: ordered list of broad class names.
        transform: optional callable mapping (image[H,W,1], instance_map[H,W])
            → (image, instance_map). See `instance_augment.apply_transform`.
        drop_border: if True, instances with `is_border=True` are excluded
            from the per-instance label tensors (still present in instance_map).
    """

    def __init__(
        self,
        cache_root: str | Path,
        slides: list[str],
        split: str,
        fine_classes: list[str],
        medium_classes: list[str],
        broad_classes: list[str],
        transform=None,
        drop_border: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.slides = slides
        self.split = split
        self.transform = transform
        self.drop_border = drop_border
        self.fine_classes = fine_classes
        self.medium_classes = medium_classes
        self.broad_classes = broad_classes
        self._fine_to_idx = {n: i for i, n in enumerate(fine_classes)}
        self._medium_to_idx = {n: i for i, n in enumerate(medium_classes)}
        self._broad_to_idx = {n: i for i, n in enumerate(broad_classes)}

        # Load per-slide manifests, filter to split, build flat index.
        records: list[dict] = []
        self._slide_arrays: dict[str, dict] = {}
        for slide in slides:
            sdir = self.cache_root / slide
            if not sdir.exists():
                raise FileNotFoundError(f"Slide cache missing: {sdir}")
            man = pl.read_parquet(sdir / "manifest.parquet").filter(
                pl.col("split") == split
            )
            labels = pl.read_parquet(sdir / "labels.parquet")
            self._slide_arrays[slide] = {
                "images": zarr.open(str(sdir / "images.zarr"), mode="r"),
                "instances": zarr.open(str(sdir / "instances.zarr"), mode="r"),
                "labels": labels,
            }
            for row in man.iter_rows(named=True):
                records.append(
                    {"slide": slide, "tile_idx": int(row["tile_idx"])}
                )
        self._records = records
        logger.info(
            f"STHELARInstanceDataset[{split}]: {len(records)} tiles across {len(slides)} slides"
        )

    def __len__(self) -> int:
        return len(self._records)

    def _percentile_normalize(self, image: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(image, [1.0, 99.0])
        if p99 <= p1:
            return np.zeros_like(image, dtype=np.float32)
        x = ((image.astype(np.float32) - p1) / (p99 - p1)).clip(0.0, 1.0)
        # Standardize to roughly zero-mean unit-variance
        return (x - x.mean()) / max(float(x.std()), 1e-6)

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]
        slide = rec["slide"]
        tile_idx = rec["tile_idx"]
        arrs = self._slide_arrays[slide]

        image = np.asarray(arrs["images"][tile_idx], dtype=np.uint16)
        instance_map = np.asarray(arrs["instances"][tile_idx], dtype=np.uint16)
        labels_df = arrs["labels"].filter(pl.col("tile_idx") == tile_idx)
        if self.drop_border:
            labels_df = labels_df.filter(~pl.col("is_border"))

        image_n = self._percentile_normalize(image)
        if self.transform is not None:
            image_n, instance_map = self.transform(
                image_n.astype(np.float32), instance_map
            )

        # Extract per-instance arrays
        n_inst = labels_df.height
        if n_inst > 0:
            instance_ids = labels_df["instance_id"].to_numpy().astype(np.int64)
            fine = labels_df["fine"].to_list()
            medium = labels_df["medium"].to_list()
            broad = labels_df["broad"].to_list()
            cy = labels_df["cy_px"].to_numpy().astype(np.float32)
            cx = labels_df["cx_px"].to_numpy().astype(np.float32)
            is_border = labels_df["is_border"].to_numpy().astype(bool)

            fine_labels = np.array(
                [self._fine_to_idx.get(f, -1) for f in fine], dtype=np.int64
            )
            medium_labels = np.array(
                [self._medium_to_idx.get(m, -1) for m in medium], dtype=np.int64
            )
            broad_labels = np.array(
                [self._broad_to_idx.get(b, -1) for b in broad], dtype=np.int64
            )

            boxes = self._compute_bboxes(instance_map, instance_ids)
        else:
            instance_ids = np.zeros(0, dtype=np.int64)
            fine_labels = np.zeros(0, dtype=np.int64)
            medium_labels = np.zeros(0, dtype=np.int64)
            broad_labels = np.zeros(0, dtype=np.int64)
            cy = np.zeros(0, dtype=np.float32)
            cx = np.zeros(0, dtype=np.float32)
            is_border = np.zeros(0, dtype=bool)
            boxes = np.zeros((0, 4), dtype=np.float32)

        out = {
            "image": torch.from_numpy(image_n[None, ...]).float(),
            "instance_map": torch.from_numpy(instance_map.astype(np.int64)),
            "instance_ids": torch.from_numpy(instance_ids),
            "fine_labels": torch.from_numpy(fine_labels),
            "medium_labels": torch.from_numpy(medium_labels),
            "broad_labels": torch.from_numpy(broad_labels),
            "cy_px": torch.from_numpy(cy),
            "cx_px": torch.from_numpy(cx),
            "boxes": torch.from_numpy(boxes),
            "is_border": torch.from_numpy(is_border),
            "slide": slide,
            "tile_idx": tile_idx,
        }
        return out

    @staticmethod
    def _compute_bboxes(
        instance_map: np.ndarray, instance_ids: np.ndarray
    ) -> np.ndarray:
        """Return (N, 4) xyxy bboxes for the given instance IDs."""
        boxes = np.zeros((len(instance_ids), 4), dtype=np.float32)
        for i, iid in enumerate(instance_ids):
            ys, xs = np.where(instance_map == iid)
            if len(ys) == 0:
                # Instance was lost (e.g. cropped by augmentation); zero box.
                continue
            boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return boxes
