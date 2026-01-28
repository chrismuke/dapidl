"""Universal Ensemble Annotator - uses ALL human CellTypist models + SingleR.

Instead of guessing which tissue-specific model to use, this approach runs
ALL human CellTypist models plus SingleR and extracts the consensus prediction.

Benefits:
- No need to know the optimal model for each dataset
- Robust to tissue heterogeneity (e.g., immune infiltration in tumors)
- Consensus voting reduces individual model biases
- Cell Ontology mapping unifies vocabulary across models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import anndata


# ALL human CellTypist models (curated list)
ALL_HUMAN_CELLTYPIST_MODELS = [
    # General/Pan-tissue
    "Immune_All_High.pkl",
    "Immune_All_Low.pkl",
    "Pan_Fetal_Human.pkl",
    "Developing_Human_Organs.pkl",
    # Breast
    "Cells_Adult_Breast.pkl",
    # Lung
    "Human_Lung_Atlas.pkl",
    "Cells_Lung_Airway.pkl",
    "Human_IPF_Lung.pkl",
    "Human_PF_Lung.pkl",
    "Cells_Fetal_Lung.pkl",
    # Intestinal/Colon
    "Cells_Intestinal_Tract.pkl",
    "Human_Colorectal_Cancer.pkl",
    # Liver
    "Healthy_Human_Liver.pkl",
    # Heart
    "Healthy_Adult_Heart.pkl",
    # Skin
    "Adult_Human_Skin.pkl",
    "Fetal_Human_Skin.pkl",
    # Pancreas
    "Adult_Human_PancreaticIslet.pkl",
    "Fetal_Human_Pancreas.pkl",
    # Vascular/Endothelial
    "Adult_Human_Vascular.pkl",
    # Lymphoid
    "Cells_Human_Tonsil.pkl",
    # Brain/Neural
    "Adult_Human_MTG.pkl",
    "Adult_Human_PrefrontalCortex.pkl",
    "Developing_Human_Brain.pkl",
    "Developing_Human_Hippocampus.pkl",
    "Human_AdultAged_Hippocampus.pkl",
    "Human_Longitudinal_Hippocampus.pkl",
    # Reproductive
    "Human_Endometrium_Atlas.pkl",
    "Human_Placenta_Decidua.pkl",
    "Developing_Human_Gonads.pkl",
    # Thymus
    "Developing_Human_Thymus.pkl",
    # Blood/PBMC
    "Adult_COVID19_PBMC.pkl",
    "COVID19_HumanChallenge_Blood.pkl",
    # Retina
    "Human_Developmental_Retina.pkl",
    "Fetal_Human_Retina.pkl",
    # Other
    "Fetal_Human_AdrenalGlands.pkl",
    "Fetal_Human_Pituitary.pkl",
    "Human_Embryonic_YolkSac.pkl",
    "Nuclei_Human_InnerEar.pkl",
]

# Recommended subset for efficiency (core models that cover most cell types)
CORE_HUMAN_MODELS = [
    "Immune_All_High.pkl",  # Best for immune cells
    "Immune_All_Low.pkl",  # Complementary immune
    "Cells_Adult_Breast.pkl",  # Breast epithelial/stromal
    "Human_Lung_Atlas.pkl",  # Lung + general epithelial
    "Cells_Intestinal_Tract.pkl",  # Intestinal + goblet cells
    "Healthy_Human_Liver.pkl",  # Liver + hepatocytes
    "Adult_Human_Skin.pkl",  # Skin + keratinocytes
    "Cells_Human_Tonsil.pkl",  # Lymphoid tissue
    "Adult_Human_Vascular.pkl",  # Endothelial cells
    "Developing_Human_Organs.pkl",  # Pan-tissue coverage
]


@dataclass
class UniversalEnsembleConfig:
    """Configuration for universal ensemble annotation."""

    # Model selection
    use_all_models: bool = False  # If True, use ALL_HUMAN_CELLTYPIST_MODELS
    celltypist_models: list[str] = field(default_factory=lambda: CORE_HUMAN_MODELS.copy())

    # SingleR references
    include_singler_hpca: bool = True
    include_singler_blueprint: bool = True

    # Voting configuration
    voting_strategy: str = "ontology_hierarchical"  # unweighted, confidence_weighted, ontology_hierarchical
    min_votes: int = 2  # Minimum votes for confident prediction
    confidence_threshold: float = 0.5  # Minimum agreement ratio

    # Output
    granularity: str = "coarse"  # coarse, medium, fine

    def __post_init__(self):
        if self.use_all_models:
            self.celltypist_models = ALL_HUMAN_CELLTYPIST_MODELS.copy()


def get_available_models() -> list[str]:
    """Get list of available CellTypist models."""
    try:
        import celltypist
        return list(celltypist.models.get_all_models())
    except Exception:
        return []


def filter_available_models(requested: list[str]) -> list[str]:
    """Filter model list to only include available ones."""
    available = set(get_available_models())
    filtered = [m for m in requested if m in available]
    missing = [m for m in requested if m not in available]
    if missing:
        logger.warning(f"Some CellTypist models not available: {missing[:5]}...")
    return filtered


def run_universal_ensemble(
    adata: "anndata.AnnData",
    use_all_models: bool = False,
    include_singler: bool = True,
    granularity: str = "coarse",
    n_jobs: int = -1,
) -> "anndata.AnnData":
    """Run universal ensemble annotation with all human models.

    Args:
        adata: AnnData object with gene expression
        use_all_models: Use all 40+ human models (slow but thorough)
        include_singler: Include SingleR (HPCA + Blueprint)
        granularity: Output granularity (coarse/medium/fine)
        n_jobs: Parallel jobs for CellTypist (-1 = all cores)

    Returns:
        AnnData with consensus predictions in obs columns:
        - consensus_prediction: Final consensus label
        - consensus_confidence: Agreement ratio
        - consensus_coarse: Coarse category
        - consensus_votes: Number of agreeing methods
    """
    from dapidl.pipeline.components.annotators.popv_ensemble import (
        PopVEnsembleConfig,
        PopVStyleEnsembleAnnotator,
        VotingStrategy,
        GranularityLevel,
    )

    # Configure ensemble
    config = UniversalEnsembleConfig(
        use_all_models=use_all_models,
        include_singler_hpca=include_singler,
        include_singler_blueprint=include_singler,
        granularity=granularity,
    )

    # Filter to available models
    available_models = filter_available_models(config.celltypist_models)
    if not available_models:
        raise ValueError("No CellTypist models available!")

    logger.info(f"Universal ensemble: {len(available_models)} CellTypist models + SingleR={include_singler}")

    # Use existing PopV ensemble with universal model list
    popv_config = PopVEnsembleConfig(
        celltypist_models=available_models,
        include_singler_hpca=include_singler,
        include_singler_blueprint=include_singler,
        voting_strategy=VotingStrategy.ONTOLOGY_HIERARCHICAL,
        granularity=GranularityLevel(granularity),
    )

    annotator = PopVStyleEnsembleAnnotator(popv_config)
    result = annotator.annotate(adata)

    # Add results to adata
    adata.obs["consensus_prediction"] = result["consensus_prediction"]
    adata.obs["consensus_confidence"] = result["consensus_confidence"]
    adata.obs["consensus_coarse"] = result["consensus_coarse"]
    if "consensus_votes" in result:
        adata.obs["consensus_votes"] = result["consensus_votes"]

    return adata


# Convenience function for pipeline integration
def annotate_with_all_models(
    adata: "anndata.AnnData",
    quick: bool = True,
) -> "anndata.AnnData":
    """Quick function to annotate with universal ensemble.

    Args:
        adata: AnnData with gene expression
        quick: If True, use CORE_HUMAN_MODELS (10 models)
               If False, use ALL_HUMAN_CELLTYPIST_MODELS (40+ models)

    Returns:
        Annotated AnnData with consensus predictions
    """
    return run_universal_ensemble(
        adata,
        use_all_models=not quick,
        include_singler=True,
        granularity="coarse",
    )
