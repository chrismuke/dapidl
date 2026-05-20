"""Read raw patches back from a built dataset (LMDB or Zarr)."""

import struct
from pathlib import Path

import numpy as np


def read_patches(dataset_path: Path | str, indices) -> np.ndarray:
    """Return an (N, H, W) uint16 array of patches for the given indices.

    Supports both storage formats produced by the pipeline:
    - patches.lmdb: key=str(idx), value=4-byte H + 4-byte W + uint16 bytes
    - patches.zarr: indexed array
    """
    dataset_path = Path(dataset_path)
    lmdb_path = dataset_path / "patches.lmdb"
    zarr_path = dataset_path / "patches.zarr"
    if lmdb_path.exists():
        return _read_lmdb(lmdb_path, indices)
    if zarr_path.exists():
        return _read_zarr(zarr_path, indices)
    raise FileNotFoundError(f"No patches.lmdb or patches.zarr in {dataset_path}")


def _read_lmdb(lmdb_path: Path, indices) -> np.ndarray:
    import lmdb

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    patches = []
    try:
        with env.begin() as txn:
            for idx in indices:
                data = txn.get(str(int(idx)).encode())
                if data is None:
                    raise KeyError(f"patch {idx} missing in {lmdb_path}")
                h = struct.unpack("I", data[:4])[0]
                w = struct.unpack("I", data[4:8])[0]
                patches.append(
                    np.frombuffer(data[8:], dtype=np.uint16).reshape(h, w)
                )
    finally:
        env.close()
    return np.stack(patches)


def _read_zarr(zarr_path: Path, indices) -> np.ndarray:
    import zarr

    arr = zarr.open(str(zarr_path), mode="r")
    return np.stack([np.asarray(arr[int(i)]) for i in indices])
