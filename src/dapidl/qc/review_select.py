"""Review-set selection strategies for the Label Studio QC loop (pure polars)."""
from __future__ import annotations

import polars as pl

_GOOD = {"Good", "Weak-passing", "Excellent"}


def select_review_rows(df: pl.DataFrame, *, mode: str, n: int, seed: int = 0) -> pl.DataFrame:
    """Select rows for review based on the specified strategy.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with columns: row_idx, slide, cell_class, grade, anomaly_pct.
    mode : str
        Selection strategy:
        - "anomaly_disagree": Prefer high-anomaly crops with "Good" grades (classical agrees
          they're clean but anomaly detector flags them).
        - "random" (default): Random sampling with fixed seed for reproducibility.
    n : int
        Number of rows to select.
    seed : int, optional
        Random seed for reproducible sampling (default: 0).

    Returns
    -------
    pl.DataFrame
        Selected rows sorted by anomaly_pct (descending) for anomaly_disagree,
        or randomly shuffled for other modes.
    """
    if mode == "anomaly_disagree":
        # Filter to crops with good/passing grades (classical QC agrees they're clean)
        cand = df.filter(pl.col("grade").is_in(list(_GOOD)))
        # Sort by anomaly_pct descending and take top n
        return cand.sort("anomaly_pct", descending=True, nulls_last=True).head(n)
    # Default: random sampling
    return df.sample(n=min(n, df.height), shuffle=True, seed=seed)
