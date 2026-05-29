import polars as pl
from dapidl.pipeline.steps.quality_control_seg import reason_audit, stratified_audit


def test_stratified_audit_surfaces_class_concentration():
    df = pl.DataFrame({
        "source": ["a"] * 100,
        "cell_type": ["Immune"] * 50 + ["Epithelial"] * 50,
        "area_um2": [30.0] * 100,
        "broken": [True] * 40 + [False] * 10 + [False] * 50,  # broken concentrated in Immune
        "broken_reason": (["off_center"] * 40 + ["ok"] * 10 + ["ok"] * 50),
    })
    audit = stratified_audit(df, n_size_bins=2)
    imm = audit.filter((pl.col("cell_type") == "Immune"))["broken_rate"].max()
    epi = audit.filter((pl.col("cell_type") == "Epithelial"))["broken_rate"].max()
    assert imm > epi
    assert {"source", "cell_type", "size_bin", "n", "broken_rate"}.issubset(set(audit.columns))


def test_reason_audit_surfaces_class_correlated_reason():
    """B9: per-reason x per-class breakdown must expose a class-correlated drop
    (e.g. false_detection concentrated in the rare Immune class)."""
    df = pl.DataFrame({
        "source": ["a"] * 100,
        "cell_type": ["Immune"] * 50 + ["Epithelial"] * 50,
        "broken": [True] * 40 + [False] * 10 + [False] * 50,
        "broken_reason": ["false_detection"] * 40 + ["ok"] * 10 + ["ok"] * 50,
    })
    ra = reason_audit(df)
    assert {"source", "cell_type", "broken_reason", "n", "n_class",
            "frac_of_class"}.issubset(set(ra.columns))
    imm_fd = ra.filter((pl.col("cell_type") == "Immune")
                       & (pl.col("broken_reason") == "false_detection"))
    assert imm_fd["frac_of_class"].item() == 0.8        # 40/50 of Immune dropped
    epi_fd = ra.filter((pl.col("cell_type") == "Epithelial")
                       & (pl.col("broken_reason") == "false_detection"))
    assert epi_fd.height == 0                            # Epithelial untouched
