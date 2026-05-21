"""Score QC of nucleus patches cropped around centroids (own vs source)."""
import numpy as np
import polars as pl
from starpose.qc.classical import ClassicalQualityScorer


def score_centroid_patches(image: np.ndarray, centroids: np.ndarray,
                           patch: int = 128) -> pl.DataFrame:
    """Crop patch x patch windows around each [y,x] centroid (in-bounds only),
    score with ClassicalQualityScorer (reference fitted on this set), return a
    polars DF with focus_score, detection_score, qc_score per kept centroid."""
    half = patch // 2
    h, w = image.shape
    patches, kept = [], []
    for i, (y, x) in enumerate(centroids):
        yi, xi = int(round(y)), int(round(x))
        if yi - half < 0 or yi + half > h or xi - half < 0 or xi + half > w:
            continue
        patches.append(image[yi - half:yi + half, xi - half:xi + half])
        kept.append(i)
    if not patches:
        return pl.DataFrame({"centroid_idx": [], "focus_score": [],
                             "detection_score": [], "qc_score": []})
    batch = np.stack(patches)
    scorer = ClassicalQualityScorer()
    ref = scorer.fit_reference(batch)
    scores = scorer.score_batch(batch, ref=ref)
    return pl.DataFrame({
        "centroid_idx": kept,
        "focus_score": [s.focus_score for s in scores],
        "detection_score": [s.detection_score for s in scores],
        "qc_score": [s.qc_score for s in scores],
    })
