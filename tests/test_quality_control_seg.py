import polars as pl
from dapidl.pipeline.steps.quality_control_seg import stratified_audit


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
