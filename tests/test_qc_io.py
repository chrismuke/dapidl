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


def _write_lmdb_format_b(path, patches, labels):
    import json
    import lmdb
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, (p, lab) in enumerate(zip(patches, labels)):
            label_bytes = np.array([lab], dtype=np.int64).tobytes()
            txn.put(struct.pack(">Q", i), label_bytes + p.astype(np.uint16).tobytes())
        txn.put(b"__metadata__", json.dumps({"patch_size": int(patches[0].shape[0])}).encode())
    env.close()


def test_read_patches_format_b(tmp_path):
    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(5)]
    _write_lmdb_format_b(tmp_path / "patches.lmdb", patches, [0, 1, 2, 3, -1])
    out = read_patches(tmp_path, [0, 2, 4])
    assert out.shape == (3, 8, 8)
    assert out[1, 0, 0] == 2  # index 2 patch is filled with 2


def test_read_patches_format_b_missing_index(tmp_path):
    import pytest
    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(3)]
    _write_lmdb_format_b(tmp_path / "patches.lmdb", patches, [0, 1, 2])
    with pytest.raises(KeyError):
        read_patches(tmp_path, [99])


def _write_lmdb_format_b_no_meta(path, patches, labels):
    import lmdb
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, (p, lab) in enumerate(zip(patches, labels)):
            label_bytes = np.array([lab], dtype=np.int64).tobytes()
            txn.put(struct.pack(">Q", i), label_bytes + p.astype(np.uint16).tobytes())
        # NOTE: deliberately no __metadata__ key (mirrors breast-6source)
    env.close()


def test_read_patches_format_b_without_metadata(tmp_path):
    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(5)]
    _write_lmdb_format_b_no_meta(tmp_path / "patches.lmdb", patches, [0, 1, 2, 3, -1])
    out = read_patches(tmp_path, [0, 2, 4])
    assert out.shape == (3, 8, 8)
    assert out[1, 0, 0] == 2


# --- B10: explicit, self-describing format tag (no more guessing from key bytes) ---


def _write_lmdb_tagged(path, patches, fmt_value, *, key_scheme):
    """Write an LMDB whose serialization is stamped with an explicit format tag.

    key_scheme="str_hw"  -> Format A bytes (str(idx) keys, u32 H + u32 W header)
    key_scheme="u64_square" -> Format B bytes (>Q keys, int64 label + square patch)
    The tag written is `fmt_value`, which may deliberately disagree with the bytes
    (used by the authority test below).
    """
    import lmdb

    from dapidl.qc.io import FORMAT_KEY

    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, p in enumerate(patches):
            p16 = p.astype(np.uint16)
            if key_scheme == "str_hw":
                h, w = p16.shape
                txn.put(str(i).encode(), struct.pack("I", h) + struct.pack("I", w) + p16.tobytes())
            else:  # u64_square
                label = np.array([0], dtype=np.int64).tobytes()
                txn.put(struct.pack(">Q", i), label + p16.tobytes())
        txn.put(FORMAT_KEY, fmt_value)
    env.close()


def test_explicit_format_b_tag_reads_correctly(tmp_path):
    from dapidl.qc.io import FORMAT_U64KEY_SQUARE

    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(5)]
    _write_lmdb_tagged(tmp_path / "patches.lmdb", patches, FORMAT_U64KEY_SQUARE, key_scheme="u64_square")
    out = read_patches(tmp_path, [0, 2, 4])
    assert out.shape == (3, 8, 8)
    assert out[1, 0, 0] == 2


def test_explicit_format_a_tag_reads_non_square(tmp_path):
    # The H/W-header path must preserve non-square shapes. A square-from-length
    # (Format B) parser could not reproduce a 6x8 patch, so a correct read proves
    # the A parser was selected via the tag.
    from dapidl.qc.io import FORMAT_STRKEY_HW

    patches = [np.full((6, 8), i, dtype=np.uint16) for i in range(4)]
    _write_lmdb_tagged(tmp_path / "patches.lmdb", patches, FORMAT_STRKEY_HW, key_scheme="str_hw")
    out = read_patches(tmp_path, [0, 3])
    assert out.shape == (2, 6, 8)
    assert out[1, 0, 0] == 3


def test_format_tag_overrides_key_heuristic(tmp_path):
    # Physically Format A (str(idx) keys) but mislabeled with the Format B tag.
    # An authoritative tag makes the reader look up >Q-packed keys, which don't
    # exist here -> KeyError. The byte heuristic alone would pick A (digit keys)
    # and succeed, so the KeyError proves the explicit tag overrode the guess.
    import pytest

    from dapidl.qc.io import FORMAT_U64KEY_SQUARE

    patches = [np.full((8, 8), i, dtype=np.uint16) for i in range(3)]
    _write_lmdb_tagged(tmp_path / "patches.lmdb", patches, FORMAT_U64KEY_SQUARE, key_scheme="str_hw")
    with pytest.raises(KeyError):
        read_patches(tmp_path, [0])


def test_unknown_format_tag_raises(tmp_path):
    # An explicit but unrecognized tag is a real error, not a cue to fall back to
    # guessing — the contract is declared yet unreadable.
    import lmdb
    import pytest

    from dapidl.qc.io import FORMAT_KEY

    env = lmdb.open(str(tmp_path / "patches.lmdb"), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put(
            struct.pack(">Q", 0),
            np.array([0], np.int64).tobytes() + np.zeros(64, np.uint16).tobytes(),
        )
        txn.put(FORMAT_KEY, b"some-future-format-v9")
    env.close()
    with pytest.raises(ValueError):
        read_patches(tmp_path, [0])
