"""Read raw patches back from a built dataset (LMDB or Zarr)."""

import struct
from pathlib import Path

import numpy as np

# Self-describing LMDB serialization tag (review 2026-05-29 B10). Builders stamp
# one of these under FORMAT_KEY so the reader no longer has to *guess* the layout
# from the first key's bytes. Untagged (legacy) LMDBs fall back to the heuristic.
FORMAT_KEY = b"__patch_format__"
FORMAT_STRKEY_HW = b"strkey-hwheader"     # str(idx) keys; value = u32 H + u32 W + uint16 patch
FORMAT_U64KEY_SQUARE = b"u64key-square"   # >Q idx keys; value = int64 label + square uint16 patch


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
    from math import isqrt

    import lmdb

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    patches = []
    try:
        with env.begin() as txn:
            # Prefer the explicit, self-describing format tag (B10). The two
            # layouts are:
            #   Format A (legacy/fixtures): ASCII-digit keys str(idx);
            #     value = 4B H + 4B W + uint16 patch.
            #   Format B (real datasets): 8-byte big-endian >Q keys;
            #     value = 8B int64 label + uint16 SQUARE patch (no H/W header).
            tag = txn.get(FORMAT_KEY)
            if tag == FORMAT_STRKEY_HW:
                format_a = True
            elif tag == FORMAT_U64KEY_SQUARE:
                format_a = False
            elif tag is not None:
                raise ValueError(f"unknown patch format tag {tag!r} in {lmdb_path}")
            else:
                # Untagged legacy LMDB: guess the scheme from the smallest
                # non-underscore key (digit keys -> Format A).
                cur = txn.cursor()
                if not cur.first():
                    raise KeyError(f"empty LMDB {lmdb_path}")
                first_key = cur.key()
                while first_key[:1] == b"_":  # skip __metadata__/__patch_format__
                    if not cur.next():
                        break
                    first_key = cur.key()
                format_a = first_key.isdigit()
            for idx in indices:
                i = int(idx)
                if format_a:
                    data = txn.get(str(i).encode())
                    if data is None:
                        raise KeyError(f"patch {i} missing in {lmdb_path}")
                    h = struct.unpack("I", data[:4])[0]
                    w = struct.unpack("I", data[4:8])[0]
                    patches.append(
                        np.frombuffer(data[8:], dtype=np.uint16).reshape(h, w)
                    )
                else:
                    data = txn.get(struct.pack(">Q", i))
                    if data is None:
                        raise KeyError(f"patch {i} missing in {lmdb_path}")
                    pix = np.frombuffer(data[8:], dtype=np.uint16)
                    side = isqrt(len(pix))
                    if side * side != len(pix):
                        raise ValueError(
                            f"patch {i}: {len(pix)} px not square in {lmdb_path}"
                        )
                    patches.append(pix.reshape(side, side))
    finally:
        env.close()
    return np.stack(patches)


def _read_zarr(zarr_path: Path, indices) -> np.ndarray:
    import zarr

    arr = zarr.open(str(zarr_path), mode="r")
    return np.stack([np.asarray(arr[int(i)]) for i in indices])
