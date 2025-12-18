"""Cell type annotation using CellTypist with multiple annotation strategies.

Supported strategies:
1. popV - Ensemble prediction using popV (if installed)
2. hierarchical - Tissue-specific model + specialized refinement
3. consensus - Consensus voting across multiple CellTypist models (default)
"""

from enum import Enum
from pathlib import Path
from typing import Any

import anndata as ad
import celltypist
import numpy as np
import pandas as pd
import polars as pl
from celltypist import models
from loguru import logger

from dapidl.data.xenium import XeniumDataReader


# Default model for breast tissue
DEFAULT_MODEL = "Cells_Adult_Breast.pkl"


class AnnotationStrategy(str, Enum):
    """Available annotation strategies."""

    SINGLE = "single"  # Single model annotation
    CONSENSUS = "consensus"  # Consensus voting across multiple models (default)
    HIERARCHICAL = "hierarchical"  # Tissue-specific + specialized refinement
    POPV = "popv"  # popV ensemble prediction
    GROUND_TRUTH = "ground_truth"  # Use ground truth annotations from file


# Default strategy
DEFAULT_STRATEGY = AnnotationStrategy.CONSENSUS


def list_available_models(force_update: bool = False) -> pd.DataFrame:
    """List all available CellTypist models.

    Args:
        force_update: Whether to fetch the latest model list from the server

    Returns:
        DataFrame with model names and descriptions
    """
    return models.models_description(on_the_fly=False)


def get_downloaded_models() -> list[str]:
    """Get list of locally downloaded CellTypist models.

    Returns:
        List of model filenames that are available locally
    """
    return models.get_all_models()


def download_model(model_name: str, force_update: bool = False) -> None:
    """Download a specific CellTypist model.

    Args:
        model_name: Name of the model to download (e.g., 'Cells_Adult_Breast.pkl')
        force_update: Whether to re-download even if exists
    """
    models.download_models(model=model_name, force_update=force_update)


def is_popv_available() -> bool:
    """Check if popV is installed and available."""
    try:
        import popv
        return True
    except ImportError:
        return False


# Mapping from CellTypist and popV cell types to broad categories
# Includes both CellTypist (Cells_Adult_Breast.pkl) and popV (Tabula Sapiens) cell types
# Note: Uses prefix/substring matching - order matters (more specific first)
CELL_TYPE_HIERARCHY = {
    "Epithelial": [
        # CellTypist breast model
        "LummHR",
        "LummHR-SCGB",
        "LummHR-active",
        "LummHR-major",
        "Lumsec",
        "Lumsec-HLA",
        "Lumsec-KIT",
        "Lumsec-basal",
        "Lumsec-lac",
        "Lumsec-major",
        "Lumsec-myo",
        "Lumsec-prol",
        "basal",
        # popV Mammary/Epithelium models (Cell Ontology names)
        "luminal epithelial cell of mammary gland",
        "progenitor cell of mammary luminal epithelium",
        "basal cell",  # Also matches "basal cell of prostate epithelium"
        "myoepithelial cell",
        "epithelial cell",  # Generic
        "epithelial fate stem cell",
        "glandular epithelial cell",
        # Organ-specific epithelial (from popV Epithelium model)
        "hepatocyte",
        "enterocyte",
        "enterochromaffin",
        "enteroendocrine",
        "intestinal crypt stem cell",
        "intestinal tuft cell",
        "cholangiocyte",
        "kidney epithelial cell",
        "goblet cell",
        "club cell",
        "ciliated",
        "alveolar",
        "pulmonary",
        "acinar cell",
        "pancreatic",
        "paneth cell",
        "mucus secreting cell",
        "ductal cell",
        "duct epithelial cell",
        "urothelial cell",
        "keratinocyte",
        "keratocyte",
        "corneal epithelial cell",
        "conjunctival epithelial cell",
        "retinal pigment epithelial cell",
        "thymic epithelial cell",
        "salivary gland cell",
        "serous cell",
        "ionocyte",
        "ovarian surface epithelial cell",
        "mesothelial cell",
        "luminal cell of prostate",
        "sebum secreting cell",
    ],
    "Immune": [
        # CellTypist breast model
        "CD4",
        "CD8",
        "T_prol",
        "GD",  # gamma-delta T cells
        "NKT",
        "Treg",
        "b_naive",
        "bmem_switched",
        "bmem_unswitched",
        "plasma",
        "plasma_IgA",
        "plasma_IgG",
        "Macro",
        "Mono",
        "cDC",
        "mDC",
        "pDC",
        "mye-prol",
        "NK",
        "NK-ILCs",
        "Mast",
        "Neutrophil",
        # CellTypist Immune_All_High model
        "B cell",
        "B cells",
        "T cell",
        "T cells",
        "Macrophage",
        "Macrophages",
        "Dendritic",
        "DC",
        "Monocyte",
        "Monocytes",
        # popV Immune model (Cell Ontology names)
        "CD4-positive",
        "CD8-positive",
        "alpha-beta T cell",
        "alpha-beta thymocyte",
        "gamma-delta T cell",
        "regulatory T cell",
        "T follicular helper cell",
        "thymocyte",
        "natural killer cell",
        "mature NK T cell",
        "innate lymphoid cell",
        "plasma cell",
        "macrophage",
        "tissue-resident macrophage",
        "colon macrophage",
        "microglial cell",
        "monocyte",
        "classical monocyte",
        "intermediate monocyte",
        "non-classical monocyte",
        "mononuclear phagocyte",
        "neutrophil",
        "basophil",
        "mast cell",
        "granulocyte",
        "dendritic cell",
        "myeloid dendritic cell",
        "plasmacytoid dendritic cell",
        "Langerhans cell",
        "erythrocyte",
        "erythroid",
        "platelet",
        "hematopoietic",
        "leukocyte",
        "myeloid cell",
        "myeloid leukocyte",
    ],
    "Stromal": [
        # CellTypist breast model
        "Fibro",
        "Fibro-SFRP4",
        "Fibro-major",
        "Fibro-matrix",
        "Fibro-prematrix",
        "Fibroblast",
        "Fibroblasts",
        "pericytes",
        "Pericyte",
        "Pericytes",
        "vsmc",
        "Smooth muscle",
        # popV Stromal model (Cell Ontology names)
        "fibroblast",
        "fibroblast of breast",
        "myofibroblast",
        "vascular associated smooth muscle cell",
        "blood vessel smooth muscle cell",
        "smooth muscle cell",
        "pericyte",
        "adventitial cell",
        "mesenchymal stem cell",
        "mesenchymal cell",
        "stromal cell",
        "interstitial cell",
        "peritubular myoid cell",
        "stellate cell",
        "adipocyte",
        "fat cell",
    ],
    "Endothelial": [
        # CellTypist breast model
        "Vas",
        "Vas-arterial",
        "Vas-capillary",
        "Vas-venous",
        "Endothelial",
        "Lymph",
        "Lymph-immune",
        "Lymph-major",
        "Lymph-valve1",
        "Lymph-valve2",
        # popV models (Cell Ontology names)
        "endothelial cell",
        "vascular endothelial cell",
        "lymphatic endothelial cell",
        "capillary endothelial cell",
        "arterial endothelial cell",
        "venous endothelial cell",
    ],
}


# Ground truth cell type to broad category mapping (from Cell_Barcode_Type_Matrices.xlsx)
GROUND_TRUTH_MAPPING = {
    # Epithelial/Tumor cells
    "DCIS_1": "Epithelial",
    "DCIS_2": "Epithelial",
    "Invasive_Tumor": "Epithelial",
    "Prolif_Invasive_Tumor": "Epithelial",
    "Myoepi_ACTA2+": "Epithelial",
    "Myoepi_KRT15+": "Epithelial",
    # Immune cells
    "B_Cells": "Immune",
    "CD4+_T_Cells": "Immune",
    "CD8+_T_Cells": "Immune",
    "Macrophages_1": "Immune",
    "Macrophages_2": "Immune",
    "IRF7+_DCs": "Immune",
    "LAMP3+_DCs": "Immune",
    "Mast_Cells": "Immune",
    # Stromal cells
    "Stromal": "Stromal",
    "Endothelial": "Stromal",
    "Perivascular-Like": "Stromal",
    # Hybrid/Unlabeled - excluded from training by default
    "Stromal_&_T_Cell_Hybrid": "Hybrid",
    "T_Cell_&_Tumor_Hybrid": "Hybrid",
    "Unlabeled": "Unlabeled",
}


def map_to_broad_category(cell_type: str) -> str:
    """Map a detailed cell type to a broad category.

    Uses prefix matching - checks if the cell type starts with any keyword.

    Args:
        cell_type: Detailed cell type string from CellTypist

    Returns:
        Broad category name or 'Unknown' if no match
    """
    # First try exact match, then prefix match
    cell_type_lower = cell_type.lower()
    for broad_cat, keywords in CELL_TYPE_HIERARCHY.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if cell_type_lower == keyword_lower or cell_type_lower.startswith(keyword_lower):
                return broad_cat
    return "Unknown"


class CellTypeAnnotator:
    """Annotate cells using CellTypist models with multiple annotation strategies.

    Supported strategies:
    - single: Use a single CellTypist model
    - consensus: Combine predictions from multiple models using voting (default)
    - hierarchical: Use tissue-specific model + specialized refinement
    - popv: Use popV ensemble prediction (requires popv package)
    """

    def __init__(
        self,
        model_names: str | list[str] = DEFAULT_MODEL,
        confidence_threshold: float = 0.5,
        majority_voting: bool = True,
        strategy: AnnotationStrategy | str = DEFAULT_STRATEGY,
        fine_grained: bool = False,
        filter_category: str | None = None,
        ground_truth_file: str | Path | None = None,
        ground_truth_sheet: str = "Xenium R1 Fig1-5 (supervised)",
    ) -> None:
        """Initialize annotator.

        Args:
            model_names: Name(s) of CellTypist model(s) to use. Can be a single
                string or a list of model names for multi-model annotation.
            confidence_threshold: Minimum confidence score to accept prediction
            majority_voting: Whether to use majority voting for predictions
            strategy: Annotation strategy to use (single, consensus, hierarchical,
                popv, ground_truth)
            fine_grained: If True, use fine-grained cell type labels instead of
                broad categories (e.g., "CD4+ T cells" instead of "Immune")
            filter_category: If set, only include cells from this broad category
                (e.g., "Immune" to only train on immune cell subtypes)
            ground_truth_file: Path to Excel file with ground truth annotations
                (required for ground_truth strategy)
            ground_truth_sheet: Sheet name in Excel file (default for Xenium breast)
        """
        # Normalize to list
        if isinstance(model_names, str):
            self.model_names = [model_names]
        else:
            self.model_names = list(model_names)

        self.confidence_threshold = confidence_threshold
        self.majority_voting = majority_voting
        self.fine_grained = fine_grained
        self.filter_category = filter_category
        self.ground_truth_file = Path(ground_truth_file) if ground_truth_file else None
        self.ground_truth_sheet = ground_truth_sheet

        # Parse strategy
        if isinstance(strategy, str):
            strategy = AnnotationStrategy(strategy.lower())
        self.strategy = strategy

        # Validate strategy
        if self.strategy == AnnotationStrategy.POPV and not is_popv_available():
            logger.warning("popV not installed. Falling back to consensus strategy.")
            logger.warning("Install popV with: pip install popv")
            self.strategy = AnnotationStrategy.CONSENSUS

        if self.strategy == AnnotationStrategy.GROUND_TRUTH and not self.ground_truth_file:
            raise ValueError("ground_truth strategy requires ground_truth_file to be set")

        if self.strategy == AnnotationStrategy.GROUND_TRUTH and not self.ground_truth_file.exists():
            raise ValueError(f"Ground truth file not found: {self.ground_truth_file}")

        if self.strategy == AnnotationStrategy.CONSENSUS and len(self.model_names) < 2:
            logger.warning("Consensus strategy requires at least 2 models. Adding Immune_All_High.pkl")
            if "Immune_All_High.pkl" not in self.model_names:
                self.model_names.append("Immune_All_High.pkl")

        self._models: dict[str, Any] = {}
        logger.info(f"Annotation strategy: {self.strategy.value}")
        if self.fine_grained:
            logger.info(f"Fine-grained mode enabled (filter: {filter_category or 'none'})")

    def _load_model(self, model_name: str) -> Any:
        """Download and load a CellTypist model.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded CellTypist model
        """
        if model_name not in self._models:
            logger.info(f"Loading CellTypist model: {model_name}")
            models.download_models(model=model_name, force_update=False)
            self._models[model_name] = models.Model.load(model=model_name)
            logger.info(f"Model loaded with {len(self._models[model_name].cell_types)} cell types")
        return self._models[model_name]

    def _load_all_models(self) -> dict[str, Any]:
        """Load all configured models.

        Returns:
            Dictionary mapping model names to loaded models
        """
        for model_name in self.model_names:
            self._load_model(model_name)
        return self._models

    def create_anndata(self, reader: XeniumDataReader) -> ad.AnnData:
        """Create AnnData object from Xenium data.

        Args:
            reader: XeniumDataReader with loaded expression data

        Returns:
            AnnData object suitable for CellTypist
        """
        expr, genes, cell_ids = reader.load_expression_matrix()

        # Create AnnData
        adata = ad.AnnData(X=expr)
        adata.var_names = genes
        adata.obs_names = [str(cid) for cid in cell_ids]
        adata.obs["cell_id"] = cell_ids

        # Add spatial coordinates
        centroids_um = reader.get_centroids_microns()
        adata.obsm["spatial"] = centroids_um

        logger.info(f"Created AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
        return adata

    def _run_single_model(
        self, adata_norm: ad.AnnData, model_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run annotation with a single model.

        Args:
            adata_norm: Normalized AnnData object
            model_name: Name of the model to use

        Returns:
            Tuple of (cell_types, confidence_scores, probability_matrix)
        """
        model = self._load_model(model_name)

        logger.info(f"Running CellTypist prediction with {model_name}...")
        predictions = celltypist.annotate(
            adata_norm,
            model=model,
            majority_voting=self.majority_voting,
        )

        # Extract results
        pred_labels = predictions.predicted_labels
        label_col = "majority_voting" if self.majority_voting and "majority_voting" in pred_labels else "predicted_labels"
        cell_types = pred_labels[label_col].astype(str).values
        prob_matrix = predictions.probability_matrix.values
        conf_scores = prob_matrix.max(axis=1)

        return cell_types, conf_scores, prob_matrix

    def _annotate_popv(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run annotation using popV ensemble prediction with multiple models.

        popV 0.6.0+ uses pretrained HubModels from HuggingFace. This method
        runs multiple specialized models and combines their predictions:
        - Mammary: tissue-specific model for breast
        - Immune: all immune cell types
        - Epithelium: epithelial cells
        - Stromal: fibroblasts and stromal cells

        The gene_symbols='feature_name' parameter tells popV to map gene symbols
        to Ensembl IDs using the CELLxGENE census internally.

        Args:
            adata: AnnData object with gene expression (raw counts preferred)

        Returns:
            DataFrame with annotations including per-model predictions
        """
        try:
            from popv.hub import HubModel
        except ImportError:
            raise ImportError("popV not installed. Install with: pip install popv")

        logger.info("Running popV multi-model ensemble prediction...")

        # Check if data looks normalized
        x_sample = adata.X[:100].toarray() if hasattr(adata.X, 'toarray') else adata.X[:100]
        if x_sample.max() < 20 and not np.allclose(x_sample, x_sample.astype(int)):
            logger.warning("Data appears normalized. popV works best with raw counts.")

        # Check if genes are already Ensembl IDs
        sample_genes = list(adata.var_names[:5])
        genes_are_ensembl = all(g.startswith('ENSG') for g in sample_genes)

        # Determine gene_symbols parameter for popV
        # If genes are symbols, tell popV to map them using 'feature_name' column in census
        gene_symbols_param = None if genes_are_ensembl else 'feature_name'
        if gene_symbols_param:
            logger.info("Gene symbols detected - popV will map to Ensembl IDs internally")

        # Available popV HubModels for human breast tissue annotation
        # Order matters: more specific models first, then general models
        popv_models = [
            ("popV/tabula_sapiens_Mammary", "mammary"),      # Breast-specific
            ("popV/tabula_sapiens_Immune", "immune"),        # All immune cells
            ("popV/tabula_sapiens_Epithelium", "epithelium"), # Epithelial cells
            ("popV/tabula_sapiens_Stromal", "stromal"),      # Stromal/fibroblasts
        ]

        # Collect predictions from all models
        all_predictions: dict[str, dict] = {}  # cell_id -> {model: prediction}
        all_scores: dict[str, dict] = {}       # cell_id -> {model: score}
        cell_ids = adata.obs["cell_id"].values

        # Initialize with cell IDs
        for cid in cell_ids:
            all_predictions[cid] = {}
            all_scores[cid] = {}

        import tempfile

        for repo_name, model_key in popv_models:
            try:
                logger.info(f"Loading popV model: {repo_name}")
                hub_model = HubModel.pull_from_huggingface_hub(repo_name)

                with tempfile.TemporaryDirectory() as tmpdir:
                    # CRITICAL FIX: Pass gene_symbols parameter for symbol->Ensembl mapping
                    adata_annotated = hub_model.annotate_data(
                        adata.copy(),
                        save_path=tmpdir,
                        prediction_mode='fast',
                        gene_symbols=gene_symbols_param  # This enables internal gene mapping!
                    )

                # Extract predictions
                pred_key = "popv_prediction"
                score_key = "popv_prediction_score"

                if pred_key not in adata_annotated.obs.columns:
                    # Fall back to majority vote
                    if "popv_majority_vote_prediction" in adata_annotated.obs.columns:
                        pred_key = "popv_majority_vote_prediction"
                        score_key = "popv_majority_vote_score"

                if pred_key in adata_annotated.obs.columns:
                    predictions = adata_annotated.obs[pred_key].values
                    scores = adata_annotated.obs.get(score_key, pd.Series([0.5] * len(adata_annotated))).values

                    # Store per-model predictions
                    for i, cid in enumerate(cell_ids):
                        pred = str(predictions[i]) if hasattr(predictions[i], '__str__') else predictions[i]
                        all_predictions[cid][model_key] = pred
                        all_scores[cid][model_key] = float(scores[i]) if not pd.isna(scores[i]) else 0.5

                    # Log distribution
                    unique_preds = pd.Series(predictions).value_counts()
                    logger.info(f"  {model_key}: {len(unique_preds)} unique types, top: {unique_preds.head(3).to_dict()}")
                else:
                    logger.warning(f"  {model_key}: No prediction column found")

            except Exception as e:
                logger.warning(f"  {model_key} failed: {e}")
                continue

        if not any(all_predictions[cell_ids[0]]):
            logger.warning("All popV models failed. Falling back to CellTypist consensus.")
            return self._annotate_consensus(adata)

        # Combine predictions: use consensus across models
        final_cell_types = []
        final_confidence = []
        final_broad = []

        for cid in cell_ids:
            preds = all_predictions[cid]
            scores = all_scores[cid]

            if not preds:
                final_cell_types.append("Unknown")
                final_confidence.append(0.0)
                final_broad.append("Unknown")
                continue

            # Strategy: weighted voting by confidence score
            # Group predictions by broad category first
            broad_votes: dict[str, float] = {}
            type_votes: dict[str, tuple[float, str]] = {}  # broad -> (score, fine_type)

            for model_key, pred in preds.items():
                score = scores.get(model_key, 0.5)
                broad = map_to_broad_category(pred)

                # Skip "unassigned" predictions from models
                if pred.lower() in ("unassigned", "unknown", "nan"):
                    continue

                broad_votes[broad] = broad_votes.get(broad, 0) + score

                # Track best fine-grained prediction per broad category
                if broad not in type_votes or score > type_votes[broad][0]:
                    type_votes[broad] = (score, pred)

            if not broad_votes:
                final_cell_types.append("Unknown")
                final_confidence.append(0.0)
                final_broad.append("Unknown")
                continue

            # Select broad category with highest weighted votes
            best_broad = max(broad_votes, key=broad_votes.get)
            best_score, best_type = type_votes[best_broad]

            # Confidence = normalized vote weight
            total_votes = sum(broad_votes.values())
            confidence = broad_votes[best_broad] / total_votes if total_votes > 0 else 0.5

            final_cell_types.append(best_type)
            final_confidence.append(confidence)
            final_broad.append(best_broad)

        # Build results DataFrame with per-model predictions
        results_data = {
            "cell_id": cell_ids,
            "predicted_type": final_cell_types,
            "broad_category": final_broad,
            "confidence": final_confidence,
        }

        # Add individual model predictions
        for _, model_key in popv_models:
            results_data[f"popv_{model_key}"] = [
                all_predictions[cid].get(model_key, "")
                for cid in cell_ids
            ]
            results_data[f"popv_{model_key}_score"] = [
                all_scores[cid].get(model_key, 0.0)
                for cid in cell_ids
            ]

        results = pl.DataFrame(results_data)

        # Log final distribution
        logger.info(f"popV multi-model annotation complete for {len(results)} cells")
        self._log_category_distribution(results, "broad_category")

        return results

    def _annotate_hierarchical(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run hierarchical annotation: tissue-specific + specialized refinement.

        Strategy:
        1. Run tissue-specific model (first model) for initial classification
        2. For cells classified as specific types (e.g., Immune), run specialized model
        3. Merge results with refined predictions

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        import scanpy as sc

        logger.info("Running hierarchical annotation...")

        # Normalize data
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Step 1: Run primary (tissue-specific) model
        primary_model = self.model_names[0]
        cell_types_primary, conf_primary, _ = self._run_single_model(adata_norm, primary_model)
        broad_primary = np.array([map_to_broad_category(ct) for ct in cell_types_primary])

        # Initialize final results with primary predictions
        final_cell_types = cell_types_primary.copy()
        final_confidence = conf_primary.copy()
        final_broad = broad_primary.copy()
        refined_by = np.array([""] * len(cell_types_primary))

        # Step 2: Run specialized models for refinement
        for i, specialized_model in enumerate(self.model_names[1:], start=2):
            logger.info(f"Running specialized model {specialized_model} for refinement...")

            cell_types_spec, conf_spec, _ = self._run_single_model(adata_norm, specialized_model)
            broad_spec = np.array([map_to_broad_category(ct) for ct in cell_types_spec])

            # Determine which cells to refine based on model type
            # If specialized model is immune-focused, refine immune cells
            model_lower = specialized_model.lower()

            if "immune" in model_lower:
                # Refine cells that are either Immune or Unknown from primary
                # but have high confidence immune prediction from specialized model
                refine_mask = (
                    ((broad_primary == "Immune") | (broad_primary == "Unknown")) &
                    (broad_spec == "Immune") &
                    (conf_spec > self.confidence_threshold)
                )
            else:
                # For other specialized models, refine Unknown cells or low-confidence predictions
                refine_mask = (
                    ((broad_primary == "Unknown") | (conf_primary < self.confidence_threshold)) &
                    (broad_spec != "Unknown") &
                    (conf_spec > conf_primary)
                )

            # Apply refinement
            final_cell_types[refine_mask] = cell_types_spec[refine_mask]
            final_confidence[refine_mask] = conf_spec[refine_mask]
            final_broad[refine_mask] = broad_spec[refine_mask]
            refined_by[refine_mask] = specialized_model

            n_refined = refine_mask.sum()
            logger.info(f"  Refined {n_refined} cells using {specialized_model}")

        results = pl.DataFrame({
            "cell_id": adata.obs["cell_id"].values,
            "predicted_type": final_cell_types,
            "broad_category": final_broad,
            "confidence": final_confidence,
            "refined_by": refined_by,
            "primary_model": primary_model,
        })

        logger.info(f"Hierarchical annotation complete for {len(results)} cells")
        self._log_category_distribution(results, "broad_category")

        return results

    def _annotate_consensus(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run consensus annotation: voting across multiple CellTypist models.

        Strategy:
        1. Run all models independently
        2. For each cell, aggregate predictions using voting
        3. Final prediction is the consensus with confidence based on agreement

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        import scanpy as sc
        from collections import Counter

        logger.info("Running consensus annotation with voting...")

        # Normalize data
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        n_cells = len(adata)
        n_models = len(self.model_names)

        # Collect predictions from all models
        all_cell_types = []
        all_broad_categories = []
        all_confidences = []

        for model_name in self.model_names:
            cell_types, conf_scores, _ = self._run_single_model(adata_norm, model_name)
            broad_categories = [map_to_broad_category(ct) for ct in cell_types]

            all_cell_types.append(cell_types)
            all_broad_categories.append(broad_categories)
            all_confidences.append(conf_scores)

        # Convert to arrays for easier manipulation
        all_cell_types = np.array(all_cell_types)  # (n_models, n_cells)
        all_broad_categories = np.array(all_broad_categories)
        all_confidences = np.array(all_confidences)

        # Compute consensus for each cell
        final_cell_types = []
        final_broad = []
        final_confidence = []
        consensus_scores = []

        for cell_idx in range(n_cells):
            # Get predictions for this cell from all models
            cell_broads = all_broad_categories[:, cell_idx]
            cell_types = all_cell_types[:, cell_idx]
            cell_confs = all_confidences[:, cell_idx]

            # Vote on broad category (excluding Unknown if possible)
            valid_broads = [b for b in cell_broads if b != "Unknown"]
            if valid_broads:
                broad_counter = Counter(valid_broads)
            else:
                broad_counter = Counter(cell_broads)

            consensus_broad, consensus_count = broad_counter.most_common(1)[0]
            consensus_score = consensus_count / n_models

            # Find the best cell type prediction among models that agree on broad category
            best_type = None
            best_conf = 0.0
            for model_idx in range(n_models):
                if all_broad_categories[model_idx, cell_idx] == consensus_broad:
                    if all_confidences[model_idx, cell_idx] > best_conf:
                        best_conf = all_confidences[model_idx, cell_idx]
                        best_type = all_cell_types[model_idx, cell_idx]

            # If no model matches consensus broad (shouldn't happen), take highest confidence
            if best_type is None:
                best_idx = np.argmax(cell_confs)
                best_type = cell_types[best_idx]
                best_conf = cell_confs[best_idx]

            # Adjust confidence based on consensus score
            # High agreement (all models agree) = full confidence
            # Low agreement = slightly reduced confidence (keep it usable)
            # Use sqrt to soften the penalty: 50% consensus → ~71% multiplier instead of 50%
            adjusted_confidence = best_conf * np.sqrt(consensus_score)

            final_cell_types.append(best_type)
            final_broad.append(consensus_broad)
            final_confidence.append(adjusted_confidence)
            consensus_scores.append(consensus_score)

        # Build results with individual model predictions for reference
        results_data = {
            "cell_id": adata.obs["cell_id"].values,
            "predicted_type": final_cell_types,
            "broad_category": final_broad,
            "confidence": final_confidence,
            "consensus_score": consensus_scores,
        }

        # Add individual model predictions
        for i, model_name in enumerate(self.model_names):
            suffix = f"_{i+1}"
            results_data[f"predicted_type{suffix}"] = all_cell_types[i]
            results_data[f"broad_category{suffix}"] = all_broad_categories[i]
            results_data[f"confidence{suffix}"] = all_confidences[i]
            results_data[f"model{suffix}"] = model_name

        results = pl.DataFrame(results_data)

        # Log statistics
        logger.info(f"Consensus annotation complete for {len(results)} cells")
        logger.info(f"Mean consensus score: {np.mean(consensus_scores):.3f}")
        high_consensus = sum(1 for s in consensus_scores if s >= 0.5)
        logger.info(f"High consensus (≥50% agreement): {high_consensus} cells ({100*high_consensus/n_cells:.1f}%)")

        self._log_category_distribution(results, "broad_category")

        return results

    def _log_category_distribution(self, results: pl.DataFrame, col: str) -> None:
        """Log the distribution of categories."""
        logger.info(f"Broad category distribution:")
        for cat in results[col].unique().sort():
            count = (results[col] == cat).sum()
            pct = count / len(results) * 100
            logger.info(f"  {cat}: {count} ({pct:.1f}%)")

    def annotate(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run cell type annotation using the configured strategy.

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with cell_id and prediction columns.
        """
        if self.strategy == AnnotationStrategy.GROUND_TRUTH:
            return self._annotate_ground_truth(adata)
        elif self.strategy == AnnotationStrategy.POPV:
            return self._annotate_popv(adata)
        elif self.strategy == AnnotationStrategy.HIERARCHICAL:
            return self._annotate_hierarchical(adata)
        elif self.strategy == AnnotationStrategy.CONSENSUS:
            return self._annotate_consensus(adata)
        else:  # SINGLE
            return self._annotate_single(adata)

    def _annotate_ground_truth(self, adata: ad.AnnData) -> pl.DataFrame:
        """Load ground truth annotations from Excel file.

        Args:
            adata: AnnData object (used only for cell_id matching)

        Returns:
            DataFrame with annotations from ground truth file
        """
        logger.info(f"Loading ground truth from: {self.ground_truth_file}")
        logger.info(f"Sheet: {self.ground_truth_sheet}")

        # Load Excel file
        gt_df = pd.read_excel(
            self.ground_truth_file,
            sheet_name=self.ground_truth_sheet,
        )

        logger.info(f"Loaded {len(gt_df)} cells from ground truth")
        logger.info(f"Columns: {list(gt_df.columns)}")

        # Expected format: Barcode (cell_id), Cluster (cell_type)
        if "Barcode" not in gt_df.columns or "Cluster" not in gt_df.columns:
            raise ValueError(
                f"Ground truth file must have 'Barcode' and 'Cluster' columns. "
                f"Found: {list(gt_df.columns)}"
            )

        # Get cell IDs from AnnData
        adata_cell_ids = set(adata.obs["cell_id"].values)
        gt_cell_ids = set(gt_df["Barcode"].values)

        # Check overlap
        overlap = adata_cell_ids & gt_cell_ids
        logger.info(f"Cell ID overlap: {len(overlap)} / {len(adata_cell_ids)} cells")

        if len(overlap) == 0:
            raise ValueError("No matching cell IDs between AnnData and ground truth file")

        # Map ground truth cell types to broad categories
        gt_df["broad_category"] = gt_df["Cluster"].map(GROUND_TRUTH_MAPPING)

        # Check for unmapped types
        unmapped = gt_df[gt_df["broad_category"].isna()]["Cluster"].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped cell types in ground truth: {unmapped}")
            gt_df["broad_category"] = gt_df["broad_category"].fillna("Unknown")

        # Create result DataFrame with same format as other strategies
        results = pl.DataFrame({
            "cell_id": gt_df["Barcode"].values,
            "predicted_type": gt_df["Cluster"].values,
            "broad_category": gt_df["broad_category"].values,
            "confidence": np.ones(len(gt_df)),  # Ground truth has confidence 1.0
        })

        # Filter to only cells in AnnData
        results = results.filter(pl.col("cell_id").is_in(list(adata_cell_ids)))

        # Filter out Hybrid and Unlabeled cells (not useful for training)
        excluded_categories = ["Hybrid", "Unlabeled"]
        before_filter = len(results)
        results = results.filter(~pl.col("broad_category").is_in(excluded_categories))
        excluded_count = before_filter - len(results)
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} Hybrid/Unlabeled cells from training")

        logger.info(f"Ground truth annotation complete for {len(results)} cells")

        # Log distribution
        self._log_category_distribution(results, "broad_category")

        return results

    def _annotate_single(self, adata: ad.AnnData) -> pl.DataFrame:
        """Run single-model annotation (original behavior).

        Args:
            adata: AnnData object with gene expression

        Returns:
            DataFrame with annotations
        """
        import scanpy as sc

        # Normalize data (CellTypist expects log1p normalized data)
        logger.info("Normalizing expression data...")
        adata_norm = adata.copy()
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
        sc.pp.log1p(adata_norm)

        # Run predictions for each model
        results_data: dict[str, Any] = {"cell_id": adata.obs["cell_id"].values}

        for i, model_name in enumerate(self.model_names, start=1):
            cell_types, conf_scores, _ = self._run_single_model(adata_norm, model_name)

            # Map to broad categories
            broad_categories = [map_to_broad_category(ct) for ct in cell_types]

            # Add columns with suffix if multiple models
            if len(self.model_names) == 1:
                results_data["predicted_type"] = cell_types
                results_data["broad_category"] = broad_categories
                results_data["confidence"] = conf_scores
            else:
                results_data[f"predicted_type_{i}"] = cell_types
                results_data[f"broad_category_{i}"] = broad_categories
                results_data[f"confidence_{i}"] = conf_scores
                results_data[f"model_{i}"] = model_name

        # Create results DataFrame
        results = pl.DataFrame(results_data)

        # Log statistics
        logger.info(f"Annotation complete for {len(results)} cells with {len(self.model_names)} model(s)")

        # Log per-model stats
        for i, model_name in enumerate(self.model_names, start=1):
            suffix = "" if len(self.model_names) == 1 else f"_{i}"
            conf_col = f"confidence{suffix}"
            broad_col = f"broad_category{suffix}"

            high_conf = (results[conf_col] > self.confidence_threshold).sum()
            logger.info(
                f"[{model_name}] Confidence > {self.confidence_threshold}: {high_conf} cells"
            )
            logger.info(f"[{model_name}] Broad category distribution:")
            for cat in results[broad_col].unique().sort():
                count = (results[broad_col] == cat).sum()
                pct = count / len(results) * 100
                logger.info(f"  {cat}: {count} ({pct:.1f}%)")

        return results

    def annotate_from_reader(self, reader: XeniumDataReader) -> pl.DataFrame:
        """Convenience method to annotate directly from Xenium reader.

        Args:
            reader: XeniumDataReader instance

        Returns:
            DataFrame with annotations
        """
        adata = self.create_anndata(reader)
        return self.annotate(adata)

    def filter_by_confidence(self, annotations: pl.DataFrame) -> pl.DataFrame:
        """Filter annotations by confidence threshold.

        Args:
            annotations: DataFrame from annotate()

        Returns:
            Filtered DataFrame with only high-confidence predictions
        """
        # Determine confidence column name (handles all strategies)
        if "confidence" in annotations.columns:
            conf_col = "confidence"
        elif "confidence_1" in annotations.columns:
            conf_col = "confidence_1"  # Use first model's confidence for filtering
        else:
            raise ValueError("No confidence column found in annotations")

        filtered = annotations.filter(pl.col(conf_col) >= self.confidence_threshold)
        logger.info(
            f"Filtered from {len(annotations)} to {len(filtered)} cells "
            f"(threshold={self.confidence_threshold}, column={conf_col})"
        )
        return filtered

    def get_class_mapping(self, annotations: pl.DataFrame) -> dict[str, int]:
        """Get mapping from category names to integer labels.

        In fine-grained mode, uses predicted_type (fine-grained cell types).
        In normal mode, uses broad_category (Epithelial/Immune/Stromal/etc).

        Args:
            annotations: DataFrame with broad_category and/or predicted_type column

        Returns:
            Dictionary mapping category name to integer label
        """
        if self.fine_grained:
            # Use fine-grained cell types
            if "predicted_type" in annotations.columns:
                pred_col = "predicted_type"
            elif "predicted_type_1" in annotations.columns:
                pred_col = "predicted_type_1"
            else:
                raise ValueError("No predicted_type column found in annotations")

            cell_types = sorted(annotations[pred_col].unique().to_list())
            # Filter out Unknown if present, add at end
            if "Unknown" in cell_types:
                cell_types.remove("Unknown")
                cell_types.append("Unknown")
            return {ct: i for i, ct in enumerate(cell_types)}
        else:
            # Use broad categories (original behavior)
            if "broad_category" in annotations.columns:
                broad_col = "broad_category"
            elif "broad_category_1" in annotations.columns:
                broad_col = "broad_category_1"  # Use first model for class mapping
            else:
                raise ValueError("No broad_category column found in annotations")

            categories = sorted(annotations[broad_col].unique().to_list())
            # Filter out Unknown if present, add at end
            if "Unknown" in categories:
                categories.remove("Unknown")
                categories.append("Unknown")
            return {cat: i for i, cat in enumerate(categories)}

    def filter_by_category(
        self, annotations: pl.DataFrame, category: str
    ) -> pl.DataFrame:
        """Filter annotations to only include cells from a specific broad category.

        Useful for fine-grained classification where you want to train only on
        immune cells, epithelial cells, etc.

        Args:
            annotations: DataFrame with annotations
            category: Broad category to filter for (e.g., "Immune", "Epithelial")

        Returns:
            Filtered DataFrame containing only cells from the specified category
        """
        # Determine broad_category column name
        if "broad_category" in annotations.columns:
            broad_col = "broad_category"
        elif "broad_category_1" in annotations.columns:
            broad_col = "broad_category_1"
        else:
            raise ValueError("No broad_category column found in annotations")

        original_count = len(annotations)
        filtered = annotations.filter(pl.col(broad_col) == category)
        logger.info(
            f"Filtered to {category} cells: {original_count} -> {len(filtered)}"
        )
        return filtered
