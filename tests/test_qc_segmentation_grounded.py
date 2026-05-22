import numpy as np
from dapidl.qc.segmentation_grounded import SegQCConfig, select_center_nucleus


def _disk(h, w, cy, cx, r, label):
    yy, xx = np.ogrid[:h, :w]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    out = np.zeros((h, w), np.int32)
    out[m] = label
    return out


def test_select_center_nucleus_prefers_object_covering_center():
    masks = _disk(128, 128, 64, 64, 12, 1) + _disk(128, 128, 20, 20, 8, 2)
    probs = np.array([0.9, 0.8])
    cn = select_center_nucleus(masks, probs, SegQCConfig())
    assert cn is not None and cn.label == 1
    assert cn.prob == 0.9
    assert cn.mask.sum() > 0 and cn.mask[64, 64]


def test_select_center_nucleus_none_when_center_empty_and_far():
    masks = _disk(128, 128, 15, 15, 6, 1)  # only a far corner object
    cn = select_center_nucleus(masks, np.array([0.9]), SegQCConfig())
    assert cn is None
