"""Tests for the Cell Ontology module.

Tests cover:
- CL database loading and queries
- CL mapper with fallback chain
- Annotator-specific mappings
- Pipeline step integration
"""

import pytest

from dapidl.ontology import (
    # Database
    CLTerm,
    HierarchyLevel,
    get_all_terms,
    get_term,
    get_term_by_name,
    get_terms_by_level,
    get_broad_category,
    get_coarse_category,
    DAPIDL_BROAD_CATEGORIES,
    DAPIDL_COARSE_CATEGORIES,
    # Mapper
    CLMapper,
    MapperConfig,
    MappingResult,
    MappingMethod,
    map_label,
    # Annotator mappings
    get_annotator_mappings,
    get_gt_mappings,
    get_all_annotator_mappings,
    get_all_gt_mappings,
    SINGLER_TO_CL,
    CELLTYPIST_TO_CL,
)


class TestCLDatabase:
    """Tests for cl_database module."""

    def test_get_all_terms(self):
        """Test that all terms load correctly."""
        terms = get_all_terms()
        assert len(terms) > 50  # Should have ~54+ terms
        assert all(isinstance(t, CLTerm) for t in terms.values())
        assert all(k.startswith("CL:") for k in terms.keys())

    def test_get_term(self):
        """Test single term lookup."""
        # CD4+ T cell
        term = get_term("CL:0000624")
        assert term is not None
        assert term.cl_id == "CL:0000624"
        assert "CD4" in term.name
        assert term.parent_id == "CL:0000084"  # T cell
        assert len(term.synonyms) > 0

    def test_get_term_missing(self):
        """Test lookup of non-existent term."""
        term = get_term("CL:9999999")
        assert term is None

    def test_get_term_by_name(self):
        """Test lookup by canonical name."""
        term = get_term_by_name("T cell")
        assert term is not None
        assert term.cl_id == "CL:0000084"

    def test_get_terms_by_level(self):
        """Test filtering by hierarchy level."""
        coarse_terms = get_terms_by_level(HierarchyLevel.COARSE)
        assert len(coarse_terms) > 10  # Should have 10-15 coarse terms
        assert all(t.level == HierarchyLevel.COARSE for t in coarse_terms.values())

    def test_broad_categories(self):
        """Test DAPIDL broad category mappings."""
        assert len(DAPIDL_BROAD_CATEGORIES) == 5
        assert "Epithelial" in DAPIDL_BROAD_CATEGORIES
        assert "Immune" in DAPIDL_BROAD_CATEGORIES
        assert "Stromal" in DAPIDL_BROAD_CATEGORIES

    def test_coarse_categories(self):
        """Test DAPIDL coarse category list."""
        assert len(DAPIDL_COARSE_CATEGORIES) > 10
        assert "T_Cell" in DAPIDL_COARSE_CATEGORIES
        assert "Macrophage" in DAPIDL_COARSE_CATEGORIES
        assert "Fibroblast" in DAPIDL_COARSE_CATEGORIES

    def test_get_broad_category(self):
        """Test broad category lookup."""
        assert get_broad_category("CL:0000624") == "Immune"  # CD4+ T cell
        assert get_broad_category("CL:0000066") == "Epithelial"
        assert get_broad_category("CL:0000057") == "Stromal"  # Fibroblast
        assert get_broad_category("CL:0000115") == "Endothelial"

    def test_get_coarse_category(self):
        """Test coarse category lookup."""
        assert get_coarse_category("CL:0000624") == "T_Cell"  # CD4+ T cell
        assert get_coarse_category("CL:0000625") == "T_Cell"  # CD8+ T cell
        assert get_coarse_category("CL:0000235") == "Macrophage"
        assert get_coarse_category("CL:0000057") == "Fibroblast"


class TestCLMapper:
    """Tests for cl_mapper module."""

    @pytest.fixture
    def mapper(self):
        """Create a mapper with all mappings."""
        return CLMapper(
            annotator_mappings=get_all_annotator_mappings(),
            ground_truth_mappings=get_all_gt_mappings(),
        )

    def test_mapper_initialization(self, mapper):
        """Test mapper initializes correctly."""
        stats = mapper.get_mapping_stats()
        assert stats["curated_terms"] > 50
        assert stats["annotator_mappings"] > 100
        assert stats["gt_mappings"] > 30

    def test_exact_match(self, mapper):
        """Test exact match on annotator outputs."""
        result = mapper.map_with_info("CD4+ T-cells")
        assert result.cl_id == "CL:0000624"
        assert result.method == MappingMethod.EXACT
        assert result.confidence == 1.0

    def test_synonym_match(self, mapper):
        """Test synonym matching."""
        result = mapper.map_with_info("helper T cell")
        assert result.cl_id == "CL:0000624"
        assert result.method == MappingMethod.SYNONYM

    def test_fuzzy_match(self, mapper):
        """Test fuzzy string matching."""
        # Slight variation
        result = mapper.map_with_info("CD4-positive T cells")
        assert result.cl_id == "CL:0000624"
        assert result.coarse_category == "T_Cell"

    def test_ground_truth_match(self, mapper):
        """Test ground truth label mapping."""
        result = mapper.map_with_info("DCIS_1")
        assert result.cl_id == "CL:0000066"
        assert result.method == MappingMethod.GROUND_TRUTH
        assert result.coarse_category == "Epithelial_Luminal"

    def test_unmapped(self, mapper):
        """Test unmapped labels."""
        result = mapper.map_with_info("Completely random gibberish xyz123")
        assert result.cl_id == "UNMAPPED"
        assert result.method == MappingMethod.UNMAPPED
        assert result.confidence == 0.0

    def test_map_convenience(self, mapper):
        """Test simple map() convenience function."""
        cl_id = mapper.map("Macrophages")
        assert cl_id == "CL:0000235"

    def test_map_batch(self, mapper):
        """Test batch mapping."""
        labels = ["CD4+ T-cells", "Macrophages", "Unknown"]
        results = mapper.map_batch(labels)
        assert len(results) == 3
        assert results[0].cl_id == "CL:0000624"
        assert results[1].cl_id == "CL:0000235"

    def test_hierarchy_level_broad(self, mapper):
        """Test hierarchy rollup to broad level."""
        category = mapper.get_hierarchy_level("CL:0000624", "broad")
        assert category == "Immune"

    def test_hierarchy_level_coarse(self, mapper):
        """Test hierarchy rollup to coarse level."""
        category = mapper.get_hierarchy_level("CL:0000624", "coarse")
        assert category == "T_Cell"

    def test_hierarchy_level_fine(self, mapper):
        """Test hierarchy at fine level."""
        category = mapper.get_hierarchy_level("CL:0000624", "fine")
        assert "CD4" in category

    def test_add_custom_mapping(self, mapper):
        """Test adding custom mappings."""
        mapper.add_annotator_mapping("MyCustomCell", "CL:0000084")
        result = mapper.map_with_info("MyCustomCell")
        assert result.cl_id == "CL:0000084"
        assert result.method == MappingMethod.EXACT


class TestAnnotatorMappings:
    """Tests for annotator_mappings module."""

    def test_singler_mappings(self):
        """Test SingleR mappings exist."""
        assert len(SINGLER_TO_CL) > 10
        assert "CD4+ T-cells" in SINGLER_TO_CL
        assert "B-cells" in SINGLER_TO_CL

    def test_celltypist_mappings(self):
        """Test CellTypist mappings exist."""
        assert len(CELLTYPIST_TO_CL) > 30
        assert "CD4-positive, alpha-beta T cell" in CELLTYPIST_TO_CL
        assert "Macrophage" in CELLTYPIST_TO_CL

    def test_get_annotator_mappings(self):
        """Test factory function."""
        singler = get_annotator_mappings("singler")
        assert len(singler) > 0
        assert all(v.startswith("CL:") for v in singler.values())

        celltypist = get_annotator_mappings("celltypist")
        assert len(celltypist) > 0

    def test_get_gt_mappings(self):
        """Test ground truth mappings."""
        xenium = get_gt_mappings("xenium_breast")
        assert len(xenium) > 10
        assert "Invasive_Tumor" in xenium
        assert "CD4+_T_Cells" in xenium

    def test_get_all_annotator_mappings(self):
        """Test combined annotator mappings."""
        all_maps = get_all_annotator_mappings()
        assert len(all_maps) > 100  # Should combine multiple sources

    def test_get_all_gt_mappings(self):
        """Test combined ground truth mappings."""
        all_maps = get_all_gt_mappings()
        assert len(all_maps) > 30


class TestMapLabel:
    """Tests for map_label convenience function."""

    def test_map_label_basic(self):
        """Test basic label mapping."""
        cl_id = map_label("macrophage")
        assert cl_id == "CL:0000235"

    def test_map_label_case_insensitive(self):
        """Test case insensitivity."""
        cl_id1 = map_label("T Cell")
        cl_id2 = map_label("t cell")
        assert cl_id1 == cl_id2


class TestMappingResult:
    """Tests for MappingResult dataclass."""

    def test_result_fields(self):
        """Test result has all expected fields."""
        result = MappingResult(
            original_label="test",
            cl_id="CL:0000084",
            cl_name="T cell",
            confidence=1.0,
            method=MappingMethod.EXACT,
            broad_category="Immune",
            coarse_category="T_Cell",
        )
        assert result.original_label == "test"
        assert result.cl_id == "CL:0000084"
        assert result.confidence == 1.0


# Integration tests with pipeline step
class TestCLStandardizationStep:
    """Tests for the ClearML pipeline step."""

    @pytest.fixture
    def sample_annotations(self, tmp_path):
        """Create sample annotation DataFrame."""
        import polars as pl

        df = pl.DataFrame({
            "cell_id": [1, 2, 3, 4, 5],
            "predicted_type": [
                "CD4+ T-cells",
                "Macrophages",
                "Epithelial cells",
                "B-cells",
                "Unknown",
            ],
            "confidence": [0.9, 0.85, 0.95, 0.7, 0.3],
        })

        # Save to temp file
        path = tmp_path / "annotations.parquet"
        df.write_parquet(path)
        return path

    def test_step_initialization(self):
        """Test step can be initialized."""
        from dapidl.pipeline.steps import CLStandardizationStep, CLStandardizationConfig

        config = CLStandardizationConfig(target_level="coarse")
        step = CLStandardizationStep(config)
        assert step.name == "cl_standardization"

    def test_parameter_schema(self):
        """Test step has valid parameter schema."""
        from dapidl.pipeline.steps import CLStandardizationStep

        step = CLStandardizationStep()
        schema = step.get_parameter_schema()
        assert "properties" in schema
        assert "target_level" in schema["properties"]
