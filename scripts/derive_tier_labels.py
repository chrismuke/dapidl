"""Derive labels_<tier>.npy for medium/fine from raw_labels.npy.

Requires raw_labels.npy (produced by derive_raw_labels.py) which is aligned
to the LMDB cell ordering.

Usage:
    uv run python scripts/derive_tier_labels.py --tier medium
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from loguru import logger

from dapidl.ontology.cl_mapper import get_mapper
from dapidl.ontology.training_tiers import (
    MEDIUM_NAMES, FINE_NAMES, derive_tier_label,
)

LMDB_DIR = Path("/mnt/work/datasets/derived/breast-6source-dapi-p128")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["medium", "fine"], required=True)
    args = ap.parse_args()

    raw_path = LMDB_DIR / "raw_labels.npy"
    if not raw_path.exists():
        raise SystemExit(f"missing {raw_path} -- run derive_raw_labels.py first")

    raw = np.load(raw_path, allow_pickle=True)
    logger.info(f"Loaded {len(raw):,} raw labels from {raw_path}")

    mapper = get_mapper()
    derived = np.array([derive_tier_label(str(r), args.tier, mapper) for r in raw],
                       dtype=object)

    base_names = MEDIUM_NAMES if args.tier == "medium" else FINE_NAMES
    extras = sorted(set(derived.tolist()) - set(base_names))
    class_names = list(base_names) + extras
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    label_arr = np.array([name_to_idx[d] for d in derived], dtype=np.int64)

    out_labels = LMDB_DIR / f"labels_{args.tier}.npy"
    out_map = LMDB_DIR / f"class_mapping_{args.tier}.json"
    np.save(out_labels, label_arr)
    out_map.write_text(json.dumps(name_to_idx, indent=2))

    counts = {n: int((label_arr == i).sum()) for i, n in enumerate(class_names)}
    logger.info(f"wrote {out_labels} ({len(label_arr):,} labels, {len(class_names)} classes)")
    for n, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {n:25s} {c:>10,d} ({100*c/len(label_arr):.1f}%)")


if __name__ == "__main__":
    main()
