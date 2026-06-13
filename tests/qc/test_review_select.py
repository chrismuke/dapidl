"""Tests for review-set selection strategies."""
import polars as pl

from dapidl.qc.review_select import select_review_rows


def test_anomaly_disagree_prefers_high_anomaly_good_crops():
    """High-anomaly crops with Good grades are the strongest disagreements."""
    df = pl.DataFrame({
        "row_idx": [1, 2, 3, 4],
        "slide": ["A"] * 4,
        "cell_class": ["Immune"] * 4,
        "grade": ["Good", "Good", "broken", "Weak-passing"],
        "anomaly_pct": [99.0, 5.0, 99.0, 80.0],
    })
    out = select_review_rows(df, mode="anomaly_disagree", n=2)
    picked = out["row_idx"].to_list()
    assert 1 in picked            # high anomaly + Good = top disagreement
    assert 3 not in picked        # already broken (classical agrees) -> not a disagreement
