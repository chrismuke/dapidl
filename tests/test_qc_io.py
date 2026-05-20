"""Tests for dapidl.qc.io patch reader."""

import struct

import numpy as np

from dapidl.qc.io import read_patches


def _write_lmdb(path, patches):
    import lmdb
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, p in enumerate(patches):
            h, w = p.shape
            buf = struct.pack("I", h) + struct.pack("I", w) + p.astype(np.uint16).tobytes()
            txn.put(str(i).encode(), buf)
    env.close()


def test_read_patches_from_lmdb(tmp_path):
    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(5)]
    _write_lmdb(tmp_path / "patches.lmdb", patches)
    out = read_patches(tmp_path, [0, 2, 4])
    assert out.shape == (3, 8, 8)
    assert out[1, 0, 0] == 2  # index 2 -> filled with 2


def test_missing_store_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        read_patches(tmp_path, [0])
