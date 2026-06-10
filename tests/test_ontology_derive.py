# tests/test_ontology_derive.py
import pytest
from dapidl.ontology.training_tiers import derive_labels

CASES = [
    ("xenium_rep1", "B_Cells",                ("Immune", "B_Cell")),
    ("sthelar_breast_s0", "B cell",           ("Immune", "B_Cell")),
    ("xenium_rep1", "CD8+_T_Cells",           ("Immune", "T_Cell")),
    ("sthelar_breast_s0", "GZMK CD8 T cell",  ("Immune", "T_Cell")),
    ("xenium_rep1", "Mast_Cells",             ("Immune", "Mast_Cell")),
    ("sthelar_breast_s0", "Mast cell",        ("Immune", "Mast_Cell")),
    ("xenium_rep1", "Macrophages_1",          ("Immune", "Macrophage")),
    ("xenium_rep1", "Myoepi_ACTA2+",          ("Epithelial", "Epithelial_Basal")),
    ("xenium_rep1", "Invasive_Tumor",         ("Epithelial", "Epithelial_Luminal")),
    ("xenium_rep1", "Stromal",                ("Stromal", "Fibroblast")),
    ("sthelar_breast_s0", "CAF",              ("Stromal", "Fibroblast")),
    ("sthelar_breast_s0", "Plasma",           ("Immune", "B_Cell")),
    ("sthelar_breast_s0", "Endothelial_Pericyte_Smooth_muscle", ("Stromal", "Pericyte")),
    ("xenium_rep1", "Perivascular-Like",      ("Stromal", "Pericyte")),
    ("xenium_rep1", "Endothelial",            ("Endothelial", "Endothelial")),
]

@pytest.mark.parametrize("source,raw,expected", CASES)
def test_derive_labels_cross_source(source, raw, expected):
    assert derive_labels(raw, source) == expected

def test_unmapped_is_unknown_not_misbinned():
    assert derive_labels("not_a_real_celltype_xyz", "sthelar_breast_s0") == ("Unknown", "Unknown")

def test_caf_and_plasma_no_longer_unknown():
    assert derive_labels("CAF", "sthelar_breast_s0")[0] == "Stromal"
    assert derive_labels("Plasma", "sthelar_breast_s0")[0] == "Immune"
