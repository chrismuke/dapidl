"""Unified Pipeline Configuration for DAPIDL.

This module provides a consolidated configuration hierarchy that replaces the
4 overlapping pipeline configurations:

1. controller.py - PipelineConfig (~90 parameters)
2. enhanced_controller.py - EnhancedDAPIDLPipelineController
3. gui_pipeline_config.py - GUIPipelineConfig (40+ parameters)
4. universal_controller.py - UniversalPipelineConfig

Design principles:
- Single source of truth for all pipeline parameters
- Pydantic for validation and serialization
- Hierarchical organization matching ClearML GUI groups
- Backward compatibility with migration helpers
- Type-safe with full IDE support

Migration Guide:
    See MIGRATION section at the bottom of this file for mapping from old configs.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Enums for type-safe choices
# ============================================================================


class Platform(str, Enum):
    """Spatial transcriptomics platform."""

    AUTO = "auto"
    XENIUM = "xenium"
    MERSCOPE = "merscope"


class AnnotationStrategy(str, Enum):
    """Annotation method strategy."""

    ENSEMBLE = "ensemble"
    SINGLE = "single"
    CONSENSUS = "consensus"
    GROUND_TRUTH = "ground_truth"


class AnnotatorType(str, Enum):
    """Type of cell type annotator."""

    CELLTYPIST = "celltypist"
    SINGLER = "singler"
    POPV = "popv"
    SCTYPE = "sctype"
    GROUND_TRUTH = "ground_truth"


class SegmenterType(str, Enum):
    """Nucleus segmentation method."""

    CELLPOSE = "cellpose"
    NATIVE = "native"


class NormalizationMethod(str, Enum):
    """Image normalization method."""

    ADAPTIVE = "adaptive"
    PERCENTILE = "percentile"
    MINMAX = "minmax"


class OutputFormat(str, Enum):
    """Dataset output format."""

    LMDB = "lmdb"
    ZARR = "zarr"


class TrainingMode(str, Enum):
    """Training approach."""

    FLAT = "flat"
    HIERARCHICAL = "hierarchical"


class AugmentationLevel(str, Enum):
    """Data augmentation intensity."""

    NONE = "none"
    STANDARD = "standard"
    HEAVY = "heavy"


class SamplingStrategy(str, Enum):
    """Multi-tissue sampling strategy."""

    EQUAL = "equal"
    PROPORTIONAL = "proportional"
    SQRT = "sqrt"


class BackboneType(str, Enum):
    """CNN backbone architecture."""

    EFFICIENTNETV2_S = "efficientnetv2_rw_s"
    EFFICIENTNET_B0 = "efficientnet_b0"
    CONVNEXT_TINY = "convnext_tiny"
    RESNET50 = "resnet50"
    RESNET18 = "resnet18"


class CLTargetLevel(str, Enum):
    """Cell Ontology hierarchy level."""

    BROAD = "broad"  # 3-4 classes (Epithelial, Immune, Stromal, Other)
    COARSE = "coarse"  # ~10-15 classes
    FINE = "fine"  # ~30+ classes


# ============================================================================
# Sub-configurations (GUI parameter groups)
# ============================================================================


class InputConfig(BaseModel):
    """Input data selection configuration.

    ClearML GUI Group: Input Selection
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Primary data source (one of these should be set)
    dataset_id: str | None = Field(
        default=None,
        description="ClearML Dataset ID for raw spatial data",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Search for dataset by name (alternative to ID)",
    )
    dataset_project: str = Field(
        default="DAPIDL/datasets",
        description="ClearML project containing dataset",
    )
    local_path: str | None = Field(
        default=None,
        description="Local path override (for debugging or direct access)",
    )
    s3_uri: str | None = Field(
        default=None,
        description="S3 URI for raw data (e.g., s3://dapidl/raw-data/xenium-breast/)",
    )

    # Platform detection
    platform: Platform = Field(
        default=Platform.AUTO,
        description="Spatial platform: auto, xenium, or merscope",
    )

    # Multi-tissue support (for universal training)
    tissues: list["TissueDatasetConfig"] = Field(
        default_factory=list,
        description="Multiple tissue datasets for universal training",
    )

    @model_validator(mode="after")
    def validate_data_source(self) -> "InputConfig":
        """Ensure at least one data source is specified (unless using tissues)."""
        sources = [
            self.dataset_id,
            self.dataset_name,
            self.local_path,
            self.s3_uri,
        ]
        if not any(sources) and not self.tissues:
            # Allow empty config for later population
            pass
        return self

    def add_tissue(
        self,
        tissue: str,
        *,
        dataset_id: str | None = None,
        local_path: str | None = None,
        platform: Platform = Platform.XENIUM,
        confidence_tier: int = 2,
        weight_multiplier: float = 1.0,
    ) -> "InputConfig":
        """Add a tissue dataset for multi-tissue training."""
        self.tissues.append(
            TissueDatasetConfig(
                tissue=tissue,
                dataset_id=dataset_id,
                local_path=local_path,
                platform=platform,
                confidence_tier=confidence_tier,
                weight_multiplier=weight_multiplier,
            )
        )
        return self


class TissueDatasetConfig(BaseModel):
    """Configuration for a single tissue in multi-tissue training."""

    model_config = ConfigDict(extra="forbid")

    tissue: str = Field(description="Tissue type (breast, lung, liver, etc.)")
    dataset_id: str | None = Field(default=None, description="ClearML Dataset ID")
    local_path: str | None = Field(default=None, description="Local path")
    platform: Platform = Field(default=Platform.XENIUM)
    confidence_tier: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Label confidence: 1=ground truth, 2=consensus, 3=predicted",
    )
    weight_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight multiplier for sampling",
    )


class SegmentationConfig(BaseModel):
    """Nucleus segmentation configuration.

    ClearML GUI Group: Segmentation (usually part of Input or hidden)
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    segmenter: SegmenterType = Field(
        default=SegmenterType.CELLPOSE,
        description="Segmentation method: cellpose or native",
    )
    diameter: int = Field(
        default=40,
        ge=10,
        le=200,
        description="Expected nucleus diameter in pixels",
    )
    flow_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Cellpose flow threshold (higher = stricter)",
    )
    match_threshold_um: float = Field(
        default=5.0,
        ge=0.0,
        le=20.0,
        description="Distance threshold for matching segmented nuclei to cells (microns)",
    )


class AnnotationConfig(BaseModel):
    """Cell type annotation configuration.

    ClearML GUI Group: Annotation Configuration
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Annotation strategy
    strategy: AnnotationStrategy = Field(
        default=AnnotationStrategy.ENSEMBLE,
        description="Strategy: ensemble (multiple methods), single, consensus, or ground_truth",
    )

    # CellTypist configuration
    celltypist_models: list[str] = Field(
        default=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
        description="CellTypist model names",
    )

    # SingleR configuration
    include_singler: bool = Field(
        default=True,
        description="Include SingleR annotation (requires R)",
    )
    singler_reference: Literal["blueprint", "hpca", "monaco"] = Field(
        default="blueprint",
        description="SingleR reference dataset",
    )

    # Additional annotators
    include_sctype: bool = Field(
        default=False,
        description="Include scType marker-based annotation",
    )
    include_popv: bool = Field(
        default=False,
        description="Include popV annotation",
    )

    # Ground truth
    ground_truth_file: str | None = Field(
        default=None,
        description="Path to ground truth annotations file",
    )
    ground_truth_sheet: str | None = Field(
        default=None,
        description="Excel sheet name for ground truth (if applicable)",
    )
    ground_truth_cell_id_col: str = Field(
        default="Barcode",
        description="Cell ID column in ground truth file",
    )
    ground_truth_label_col: str = Field(
        default="Cluster",
        description="Label column in ground truth file",
    )

    # Consensus settings
    min_agreement: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum annotators that must agree",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum consensus confidence (0-1)",
    )
    use_confidence_weighting: bool = Field(
        default=True,
        description="Weight votes by annotator confidence",
    )

    # Classification granularity
    fine_grained: bool = Field(
        default=True,
        description="Use detailed cell types (True) or 3 broad categories (False)",
    )
    extended_consensus: bool = Field(
        default=False,
        description="Use 6 CellTypist models instead of 2 for extended consensus",
    )


class OntologyConfig(BaseModel):
    """Cell Ontology standardization configuration.

    ClearML GUI Group: Cell Ontology Standardization
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)

    enabled: bool = Field(
        default=True,
        alias="use_cell_ontology",
        description="Enable Cell Ontology label standardization",
    )
    target_level: CLTargetLevel = Field(
        default=CLTargetLevel.COARSE,
        description="Hierarchy level: broad, coarse, or fine",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum mapping confidence (0-1)",
    )
    include_unmapped: bool = Field(
        default=False,
        description="Include cells with unmapped labels (as 'Unknown')",
    )
    fuzzy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Fuzzy matching threshold for label mapping",
    )


class LMDBConfig(BaseModel):
    """LMDB dataset creation configuration.

    ClearML GUI Group: LMDB/Dataset Configuration
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Patch extraction
    patch_sizes: list[int] = Field(
        default=[128],
        description="Patch sizes to generate (e.g., [32, 64, 128, 256])",
    )
    primary_patch_size: int | None = Field(
        default=None,
        description="Primary patch size for training (uses first in list if None)",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.LMDB,
        description="Output format: lmdb or zarr",
    )

    # Normalization
    normalization: NormalizationMethod = Field(
        default=NormalizationMethod.ADAPTIVE,
        description="Image normalization: adaptive, percentile, or minmax",
    )
    normalize_physical_size: bool = Field(
        default=True,
        description="Normalize to consistent physical size (cross-platform)",
    )
    target_pixel_size_um: float = Field(
        default=0.2125,
        ge=0.05,
        le=1.0,
        description="Target pixel size in microns (Xenium default: 0.2125)",
    )

    # Filtering
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum annotation confidence to include",
    )
    exclude_edge_cells: bool = Field(
        default=True,
        description="Exclude cells near image edges",
    )
    edge_margin_px: int = Field(
        default=64,
        ge=0,
        le=256,
        description="Edge margin in pixels",
    )

    # Skip logic
    skip_if_exists: bool = Field(
        default=True,
        description="Skip if matching LMDB dataset already exists",
    )

    @field_validator("patch_sizes")
    @classmethod
    def validate_patch_sizes(cls, v: list[int]) -> list[int]:
        """Ensure patch sizes are valid powers of 2."""
        valid_sizes = {32, 64, 128, 256, 512}
        for size in v:
            if size not in valid_sizes:
                raise ValueError(f"Invalid patch size {size}. Must be one of {valid_sizes}")
        return sorted(v)


class TrainingConfig(BaseModel):
    """Model training configuration.

    ClearML GUI Group: Training Configuration
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Model architecture
    backbone: BackboneType = Field(
        default=BackboneType.EFFICIENTNETV2_S,
        description="CNN backbone architecture",
    )
    pretrained: bool = Field(
        default=True,
        description="Use ImageNet pretrained weights",
    )
    dropout: float = Field(
        default=0.3,
        ge=0.0,
        le=0.8,
        description="Dropout rate",
    )

    # Training mode
    mode: TrainingMode = Field(
        default=TrainingMode.HIERARCHICAL,
        description="Training mode: flat or hierarchical (with curriculum)",
    )

    # Hyperparameters
    epochs: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of training epochs",
    )
    batch_size: int = Field(
        default=64,
        ge=8,
        le=512,
        description="Training batch size",
    )
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-1,
        description="Initial learning rate",
    )
    weight_decay: float = Field(
        default=1e-5,
        ge=0.0,
        le=1e-1,
        description="Weight decay (L2 regularization)",
    )

    # Class balancing
    use_weighted_loss: bool = Field(
        default=True,
        description="Use class-weighted loss for imbalanced data",
    )
    use_weighted_sampler: bool = Field(
        default=True,
        description="Use weighted random sampling",
    )
    max_weight_ratio: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Maximum class weight ratio (caps rare class weights)",
    )

    # Data splits
    val_split: float = Field(
        default=0.15,
        ge=0.05,
        le=0.3,
        description="Validation set fraction",
    )
    test_split: float = Field(
        default=0.15,
        ge=0.05,
        le=0.3,
        description="Test set fraction",
    )
    stratified: bool = Field(
        default=True,
        description="Use stratified splits",
    )

    # Data loading
    num_workers: int = Field(
        default=0,
        ge=0,
        le=16,
        description="DataLoader workers (0 = disable multiprocessing)",
    )
    use_dali: bool = Field(
        default=False,
        description="Use NVIDIA DALI for data loading",
    )

    # Augmentation
    augmentation: AugmentationLevel = Field(
        default=AugmentationLevel.STANDARD,
        description="Data augmentation level",
    )
    cross_platform: bool = Field(
        default=False,
        description="Enable aggressive scale augmentation for cross-platform transfer",
    )

    # Early stopping
    patience: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Early stopping patience (epochs)",
    )
    min_delta: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Minimum improvement for early stopping",
    )

    # Curriculum learning (for hierarchical mode)
    coarse_only_epochs: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Phase 1 epochs (coarse classification only)",
    )
    coarse_medium_epochs: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Phase 2 epochs (coarse + medium classification)",
    )
    warmup_epochs: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Learning rate warmup epochs",
    )

    # Multi-tissue training
    sampling_strategy: SamplingStrategy = Field(
        default=SamplingStrategy.SQRT,
        description="Multi-tissue sampling: equal, proportional, or sqrt",
    )
    tier1_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Weight for tier 1 (ground truth) data",
    )
    tier2_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=5.0,
        description="Weight for tier 2 (consensus) data",
    )
    tier3_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Weight for tier 3 (predicted) data",
    )
    combine_patch_sizes: bool = Field(
        default=False,
        description="Train on all patch sizes together",
    )

    # Logging
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name",
    )
    wandb_entity: str | None = Field(
        default=None,
        description="Weights & Biases entity/team",
    )


class OutputConfig(BaseModel):
    """Output and storage configuration.

    ClearML GUI Group: Output Configuration
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Output directory
    output_dir: str = Field(
        default="./pipeline_output",
        description="Local output directory",
    )
    save_best: bool = Field(
        default=True,
        description="Save best model checkpoint",
    )
    save_final: bool = Field(
        default=True,
        description="Save final model checkpoint",
    )

    # S3 storage
    upload_to_s3: bool = Field(
        default=True,
        description="Upload datasets and models to S3",
    )
    s3_bucket: str = Field(
        default="dapidl",
        description="S3 bucket name",
    )
    s3_endpoint: str = Field(
        default="https://s3.eu-central-2.idrivee2.com",
        description="S3 endpoint URL (for iDrive e2 or other S3-compatible storage)",
    )
    s3_region: str = Field(
        default="eu-central-2",
        description="S3 region",
    )
    s3_models_prefix: str = Field(
        default="models",
        description="S3 prefix for model uploads",
    )

    # ClearML registration
    register_datasets: bool = Field(
        default=True,
        description="Register datasets with ClearML",
    )
    register_models: bool = Field(
        default=True,
        description="Register trained models with ClearML",
    )
    link_to_parent: bool = Field(
        default=True,
        description="Use dataset lineage (saves storage space)",
    )


class ValidationConfig(BaseModel):
    """Cross-modal validation configuration.

    ClearML GUI Group: Validation (optional)
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)

    enabled: bool = Field(
        default=False,
        alias="run_validation",
        description="Enable cross-modal validation step",
    )
    run_leiden_check: bool = Field(
        default=True,
        description="Compare with unsupervised Leiden clustering",
    )
    run_dapi_check: bool = Field(
        default=True,
        description="Use trained DAPI model for validation",
    )
    run_consensus_check: bool = Field(
        default=True,
        description="Check multi-method consensus agreement",
    )
    min_ari_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum Adjusted Rand Index threshold",
    )
    min_agreement_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum agreement threshold",
    )

    # Ground truth comparison
    run_ground_truth_comparison: bool = Field(
        default=False,
        description="Compare predictions with ground truth labels",
    )


class TransferTestConfig(BaseModel):
    """Cross-platform transfer testing configuration.

    ClearML GUI Group: Transfer Testing (optional)
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)

    enabled: bool = Field(
        default=False,
        alias="run_transfer_test",
        description="Enable cross-platform transfer testing",
    )
    target_dataset_id: str | None = Field(
        default=None,
        description="Target platform ClearML dataset ID",
    )
    target_local_path: str | None = Field(
        default=None,
        description="Target platform local path",
    )
    target_platform: Platform = Field(
        default=Platform.AUTO,
        description="Target platform: auto, xenium, or merscope",
    )


class ExecutionConfig(BaseModel):
    """Pipeline execution configuration.

    ClearML GUI Group: Execution (usually hidden/advanced)
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # ClearML settings
    execute_remotely: bool = Field(
        default=True,
        description="Execute pipeline steps on ClearML agents",
    )
    default_queue: str = Field(
        default="default",
        description="Default ClearML queue for CPU tasks",
    )
    gpu_queue: str = Field(
        default="gpu",
        description="ClearML queue for GPU tasks",
    )

    # Caching
    cache_data_steps: bool = Field(
        default=True,
        description="Cache data loading and annotation steps",
    )
    cache_training: bool = Field(
        default=False,
        description="Cache training step (disable for different random seeds)",
    )


class DocumentationConfig(BaseModel):
    """Documentation generation configuration.

    ClearML GUI Group: Documentation (optional)
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, populate_by_name=True)

    enabled: bool = Field(
        default=False,
        alias="run_documentation",
        description="Enable documentation generation",
    )
    obsidian_vault_path: str | None = Field(
        default=None,
        description="Path to Obsidian vault for documentation",
    )
    obsidian_folder: str = Field(
        default="DAPIDL",
        description="Subfolder within Obsidian vault",
    )
    template: Literal["default", "minimal", "detailed"] = Field(
        default="default",
        description="Documentation template style",
    )


# ============================================================================
# Main Unified Configuration
# ============================================================================


class DAPIDLPipelineConfig(BaseModel):
    """Unified configuration for DAPIDL pipelines.

    This is the single source of truth for all pipeline parameters,
    replacing the 4 overlapping configurations.

    ClearML GUI Groups:
        1. Input Selection (input)
        2. Annotation Configuration (annotation)
        3. Cell Ontology Standardization (ontology)
        4. LMDB/Dataset Configuration (lmdb)
        5. Training Configuration (training)
        6. Output Configuration (output)
        7. Validation (validation) - optional
        8. Transfer Testing (transfer) - optional
        9. Execution (execution) - advanced

    Example:
        ```python
        config = DAPIDLPipelineConfig(
            input=InputConfig(dataset_id="abc123", platform=Platform.XENIUM),
            annotation=AnnotationConfig(strategy=AnnotationStrategy.ENSEMBLE),
            lmdb=LMDBConfig(patch_sizes=[128, 256]),
            training=TrainingConfig(epochs=100, backbone=BackboneType.EFFICIENTNETV2_S),
            output=OutputConfig(upload_to_s3=True),
        )

        # Multi-tissue training
        config.input.add_tissue("breast", dataset_id="abc123", confidence_tier=1)
        config.input.add_tissue("lung", dataset_id="def456", confidence_tier=2)
        ```
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Pipeline metadata
    name: str = Field(
        default="dapidl-pipeline",
        description="Pipeline name",
    )
    project: str = Field(
        default="DAPIDL/pipelines",
        description="ClearML project name",
    )
    version: str = Field(
        default="2.0.0",
        description="Pipeline version",
    )

    # Sub-configurations (GUI groups)
    input: InputConfig = Field(default_factory=InputConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    lmdb: LMDBConfig = Field(default_factory=LMDBConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    transfer: TransferTestConfig = Field(default_factory=TransferTestConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    documentation: DocumentationConfig = Field(default_factory=DocumentationConfig)

    # ========================================================================
    # Serialization for ClearML GUI
    # ========================================================================

    def to_clearml_parameters(self) -> dict[str, Any]:
        """Convert to flat parameter dictionary for ClearML GUI.

        ClearML uses slash-separated parameter names for grouping.
        Returns string values for GUI compatibility.
        """
        params = {}

        # Helper to flatten nested config
        def add_params(prefix: str, config: BaseModel, exclude: set[str] | None = None):
            for field_name, value in config.model_dump().items():
                if exclude and field_name in exclude:
                    continue
                key = f"{prefix}/{field_name}"

                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    params[key] = ",".join(str(v) for v in value)
                elif isinstance(value, bool):
                    params[key] = str(value)
                elif isinstance(value, Enum):
                    params[key] = value.value
                elif value is None:
                    params[key] = ""
                else:
                    params[key] = str(value)

        # Add metadata
        params["pipeline/name"] = self.name
        params["pipeline/project"] = self.project
        params["pipeline/version"] = self.version

        # Add sub-configs (tissues serialized separately as tissue_N/* params)
        add_params("input", self.input, exclude={"tissues"})
        add_params("segmentation", self.segmentation)
        add_params("annotation", self.annotation)
        add_params("ontology", self.ontology)
        add_params("lmdb", self.lmdb)
        add_params("training", self.training)
        add_params("output", self.output)
        add_params("validation", self.validation)
        add_params("transfer", self.transfer)
        add_params("execution", self.execution)
        add_params("documentation", self.documentation)

        # Serialize tissues as flat tissue_N/* parameters (editable in ClearML UI)
        for i, tc in enumerate(self.input.tissues):
            params[f"tissue_{i}/tissue"] = tc.tissue
            params[f"tissue_{i}/dataset_id"] = tc.dataset_id or ""
            params[f"tissue_{i}/local_path"] = tc.local_path or ""
            params[f"tissue_{i}/platform"] = tc.platform.value
            params[f"tissue_{i}/confidence_tier"] = str(tc.confidence_tier)
            params[f"tissue_{i}/weight_multiplier"] = str(tc.weight_multiplier)

        return params

    @classmethod
    def from_clearml_parameters(cls, params: dict[str, str]) -> "DAPIDLPipelineConfig":
        """Create config from ClearML parameter dictionary.

        Handles deserialization of lists, booleans, and enums.
        """

        def parse_bool(val: str) -> bool:
            return val.lower() in ("true", "1", "yes")

        def parse_list_str(val: str) -> list[str]:
            if not val:
                return []
            return [s.strip() for s in val.split(",") if s.strip()]

        def parse_list_int(val: str) -> list[int]:
            if not val:
                return []
            return [int(s.strip()) for s in val.split(",") if s.strip()]

        def get_param(prefix: str, key: str, default: Any = None) -> Any:
            full_key = f"{prefix}/{key}"
            return params.get(full_key, default)

        # Build input config
        input_config = InputConfig(
            dataset_id=get_param("input", "dataset_id") or None,
            dataset_name=get_param("input", "dataset_name") or None,
            dataset_project=get_param("input", "dataset_project", "DAPIDL/datasets"),
            local_path=get_param("input", "local_path") or None,
            s3_uri=get_param("input", "s3_uri") or None,
            platform=Platform(get_param("input", "platform", "auto")),
        )

        # Build annotation config
        annotation_config = AnnotationConfig(
            strategy=AnnotationStrategy(get_param("annotation", "strategy", "ensemble")),
            celltypist_models=parse_list_str(
                get_param("annotation", "celltypist_models", "Cells_Adult_Breast.pkl,Immune_All_High.pkl")
            ),
            include_singler=parse_bool(get_param("annotation", "include_singler", "True")),
            singler_reference=get_param("annotation", "singler_reference", "blueprint"),
            min_agreement=int(get_param("annotation", "min_agreement", "2")),
            confidence_threshold=float(get_param("annotation", "confidence_threshold", "0.5")),
            fine_grained=parse_bool(get_param("annotation", "fine_grained", "True")),
        )

        # Build ontology config
        ontology_config = OntologyConfig(
            enabled=parse_bool(get_param("ontology", "enabled", "True")),
            target_level=CLTargetLevel(get_param("ontology", "target_level", "coarse")),
            min_confidence=float(get_param("ontology", "min_confidence", "0.5")),
            include_unmapped=parse_bool(get_param("ontology", "include_unmapped", "False")),
            fuzzy_threshold=float(get_param("ontology", "fuzzy_threshold", "0.85")),
        )

        # Build LMDB config
        lmdb_config = LMDBConfig(
            patch_sizes=parse_list_int(get_param("lmdb", "patch_sizes", "128")),
            normalization=NormalizationMethod(get_param("lmdb", "normalization", "adaptive")),
            normalize_physical_size=parse_bool(get_param("lmdb", "normalize_physical_size", "True")),
            skip_if_exists=parse_bool(get_param("lmdb", "skip_if_exists", "True")),
        )

        # Build training config
        training_config = TrainingConfig(
            backbone=BackboneType(get_param("training", "backbone", "efficientnetv2_rw_s")),
            mode=TrainingMode(get_param("training", "mode", "hierarchical")),
            epochs=int(get_param("training", "epochs", "100")),
            batch_size=int(get_param("training", "batch_size", "64")),
            learning_rate=float(get_param("training", "learning_rate", "1e-4")),
            weight_decay=float(get_param("training", "weight_decay", "1e-5")),
            patience=int(get_param("training", "patience", "15")),
            coarse_only_epochs=int(get_param("training", "coarse_only_epochs", "20")),
            coarse_medium_epochs=int(get_param("training", "coarse_medium_epochs", "50")),
            warmup_epochs=int(get_param("training", "warmup_epochs", "5")),
        )

        # Build output config
        output_config = OutputConfig(
            output_dir=get_param("output", "output_dir", "./pipeline_output"),
            upload_to_s3=parse_bool(get_param("output", "upload_to_s3", "True")),
            s3_bucket=get_param("output", "s3_bucket", "dapidl"),
            s3_endpoint=get_param("output", "s3_endpoint", "https://s3.eu-central-2.idrivee2.com"),
            register_datasets=parse_bool(get_param("output", "register_datasets", "True")),
            register_models=parse_bool(get_param("output", "register_models", "True")),
            link_to_parent=parse_bool(get_param("output", "link_to_parent", "True")),
        )

        # Build validation config
        validation_config = ValidationConfig(
            enabled=parse_bool(get_param("validation", "enabled", "False")),
            run_leiden_check=parse_bool(get_param("validation", "run_leiden_check", "True")),
            run_dapi_check=parse_bool(get_param("validation", "run_dapi_check", "True")),
            min_ari_threshold=float(get_param("validation", "min_ari_threshold", "0.5")),
        )

        # Build execution config
        execution_config = ExecutionConfig(
            execute_remotely=parse_bool(get_param("execution", "execute_remotely", "True")),
            default_queue=get_param("execution", "default_queue", "default"),
            gpu_queue=get_param("execution", "gpu_queue", "gpu"),
        )

        # Parse tissue_N/* parameters
        tissues: list[TissueDatasetConfig] = []
        i = 0
        while f"tissue_{i}/tissue" in params:
            tissues.append(
                TissueDatasetConfig(
                    tissue=params[f"tissue_{i}/tissue"],
                    dataset_id=params.get(f"tissue_{i}/dataset_id") or None,
                    local_path=params.get(f"tissue_{i}/local_path") or None,
                    platform=Platform(params.get(f"tissue_{i}/platform", "xenium")),
                    confidence_tier=int(params.get(f"tissue_{i}/confidence_tier", "2")),
                    weight_multiplier=float(params.get(f"tissue_{i}/weight_multiplier", "1.0")),
                )
            )
            i += 1
        input_config.tissues = tissues

        return cls(
            name=params.get("pipeline/name", "dapidl-pipeline"),
            project=params.get("pipeline/project", "DAPIDL/pipelines"),
            version=params.get("pipeline/version", "2.0.0"),
            input=input_config,
            annotation=annotation_config,
            ontology=ontology_config,
            lmdb=lmdb_config,
            training=training_config,
            output=output_config,
            validation=validation_config,
            execution=execution_config,
        )

    # ========================================================================
    # Migration helpers from old configs
    # ========================================================================

    @classmethod
    def from_pipeline_config(cls, old_config: Any) -> "DAPIDLPipelineConfig":
        """Migrate from controller.py PipelineConfig.

        Args:
            old_config: Instance of PipelineConfig from controller.py

        Returns:
            New unified DAPIDLPipelineConfig
        """
        return cls(
            name=old_config.name,
            project=old_config.project,
            version=old_config.version,
            input=InputConfig(
                dataset_id=old_config.dataset_id,
                dataset_name=old_config.dataset_name,
                dataset_project=old_config.dataset_project,
                local_path=old_config.local_path,
                platform=Platform(old_config.platform),
            ),
            segmentation=SegmentationConfig(
                segmenter=SegmenterType(old_config.segmenter),
                diameter=old_config.diameter,
                flow_threshold=old_config.flow_threshold,
                match_threshold_um=old_config.match_threshold_um,
            ),
            annotation=AnnotationConfig(
                strategy=AnnotationStrategy(old_config.annotation_strategy),
                celltypist_models=old_config.model_names,
                confidence_threshold=old_config.confidence_threshold,
                ground_truth_file=old_config.ground_truth_file,
                fine_grained=old_config.fine_grained,
                extended_consensus=old_config.extended_consensus,
            ),
            lmdb=LMDBConfig(
                patch_sizes=[old_config.patch_size],
                output_format=OutputFormat(old_config.output_format),
                normalization=NormalizationMethod(old_config.normalization),
            ),
            training=TrainingConfig(
                backbone=BackboneType(old_config.backbone),
                epochs=old_config.epochs,
                batch_size=old_config.batch_size,
                learning_rate=old_config.learning_rate,
            ),
            output=OutputConfig(),
            validation=ValidationConfig(
                enabled=old_config.run_validation,
                run_leiden_check=old_config.validation_leiden,
                run_dapi_check=old_config.validation_dapi,
                run_consensus_check=old_config.validation_consensus,
                min_ari_threshold=old_config.min_ari_threshold,
                min_agreement_threshold=old_config.min_agreement_threshold,
                run_ground_truth_comparison=old_config.run_ground_truth_comparison,
            ),
            transfer=TransferTestConfig(
                enabled=old_config.run_transfer_test,
                target_dataset_id=old_config.transfer_target_dataset_id,
                target_local_path=old_config.transfer_target_local_path,
                target_platform=Platform(old_config.transfer_target_platform),
            ),
            execution=ExecutionConfig(
                execute_remotely=old_config.execute_remotely,
                default_queue=old_config.default_queue,
                gpu_queue=old_config.gpu_queue,
                cache_training=old_config.cache_training,
            ),
            documentation=DocumentationConfig(
                enabled=old_config.run_documentation,
                obsidian_vault_path=old_config.obsidian_vault_path,
                obsidian_folder=old_config.obsidian_folder,
                template=old_config.doc_template,
            ),
        )

    @classmethod
    def from_gui_pipeline_config(cls, old_config: Any) -> "DAPIDLPipelineConfig":
        """Migrate from gui_pipeline_config.py GUIPipelineConfig.

        Args:
            old_config: Instance of GUIPipelineConfig

        Returns:
            New unified DAPIDLPipelineConfig
        """
        return cls(
            input=InputConfig(
                dataset_id=old_config.raw_dataset_id,
                dataset_name=old_config.raw_dataset_name,
                dataset_project=old_config.raw_dataset_project,
                s3_uri=old_config.s3_data_uri,
                platform=Platform(old_config.platform),
            ),
            annotation=AnnotationConfig(
                strategy=AnnotationStrategy(old_config.annotation_strategy),
                celltypist_models=old_config.celltypist_models,
                include_singler=old_config.include_singler,
                singler_reference=old_config.singler_reference,
                include_sctype=old_config.include_sctype,
                min_agreement=old_config.min_agreement,
                confidence_threshold=old_config.confidence_threshold,
                use_confidence_weighting=old_config.use_confidence_weighting,
                fine_grained=old_config.fine_grained,
            ),
            ontology=OntologyConfig(
                enabled=old_config.use_cell_ontology,
                target_level=CLTargetLevel(old_config.cl_target_level),
                min_confidence=old_config.cl_min_confidence,
                include_unmapped=old_config.cl_include_unmapped,
                fuzzy_threshold=old_config.cl_fuzzy_threshold,
            ),
            lmdb=LMDBConfig(
                patch_sizes=old_config.patch_sizes,
                primary_patch_size=old_config.primary_patch_size,
                normalization=NormalizationMethod(old_config.normalization),
                normalize_physical_size=old_config.normalize_physical_size,
                skip_if_exists=old_config.skip_if_lmdb_exists,
            ),
            training=TrainingConfig(
                backbone=BackboneType(old_config.backbone),
                mode=TrainingMode(old_config.training_mode),
                epochs=old_config.epochs,
                batch_size=old_config.batch_size,
                learning_rate=old_config.learning_rate,
                weight_decay=old_config.weight_decay,
                patience=old_config.patience,
                coarse_only_epochs=old_config.coarse_only_epochs,
                coarse_medium_epochs=old_config.coarse_medium_epochs,
                warmup_epochs=old_config.warmup_epochs,
                combine_patch_sizes=old_config.combine_patch_sizes,
            ),
            output=OutputConfig(
                output_dir=old_config.output_dir,
                upload_to_s3=old_config.upload_to_s3,
                s3_bucket=old_config.s3_bucket,
                s3_endpoint=old_config.s3_endpoint,
                register_datasets=old_config.register_datasets,
                register_models=old_config.register_models,
                link_to_parent=old_config.link_to_parent,
            ),
        )

    @classmethod
    def from_universal_config(cls, old_config: Any) -> "DAPIDLPipelineConfig":
        """Migrate from universal_controller.py UniversalPipelineConfig.

        Args:
            old_config: Instance of UniversalPipelineConfig

        Returns:
            New unified DAPIDLPipelineConfig
        """
        config = cls(
            name=old_config.name,
            project=old_config.project,
            version=old_config.version,
            input=InputConfig(platform=Platform.AUTO),
            segmentation=SegmentationConfig(
                segmenter=SegmenterType(old_config.segmenter),
                diameter=old_config.diameter,
                flow_threshold=old_config.flow_threshold,
            ),
            lmdb=LMDBConfig(
                patch_sizes=[old_config.patch_size],
                output_format=OutputFormat(old_config.output_format),
            ),
            training=TrainingConfig(
                backbone=BackboneType(old_config.backbone),
                epochs=old_config.epochs,
                batch_size=old_config.batch_size,
                learning_rate=old_config.learning_rate,
                coarse_only_epochs=old_config.coarse_only_epochs,
                coarse_medium_epochs=old_config.coarse_medium_epochs,
                sampling_strategy=SamplingStrategy(old_config.sampling_strategy),
                tier1_weight=old_config.tier1_weight,
                tier2_weight=old_config.tier2_weight,
                tier3_weight=old_config.tier3_weight,
            ),
            output=OutputConfig(
                output_dir=old_config.output_dir or "./pipeline_output",
            ),
            execution=ExecutionConfig(
                execute_remotely=old_config.execute_remotely,
                default_queue=old_config.default_queue,
                gpu_queue=old_config.gpu_queue,
            ),
        )

        # Add tissues from old config
        for tissue_cfg in old_config.tissues:
            config.input.add_tissue(
                tissue=tissue_cfg.tissue,
                dataset_id=tissue_cfg.dataset_id,
                local_path=tissue_cfg.local_path,
                platform=Platform(tissue_cfg.platform),
                confidence_tier=tissue_cfg.confidence_tier,
                weight_multiplier=tissue_cfg.weight_multiplier,
            )

        return config


# ============================================================================
# MIGRATION GUIDE
# ============================================================================
#
# Old Config                      | New Config Path
# --------------------------------|-------------------------------------------
# PipelineConfig                  |
#   .name                         | DAPIDLPipelineConfig.name
#   .project                      | DAPIDLPipelineConfig.project
#   .version                      | DAPIDLPipelineConfig.version
#   .dataset_id                   | .input.dataset_id
#   .dataset_name                 | .input.dataset_name
#   .dataset_project              | .input.dataset_project
#   .local_path                   | .input.local_path
#   .platform                     | .input.platform
#   .segmenter                    | .segmentation.segmenter
#   .diameter                     | .segmentation.diameter
#   .flow_threshold               | .segmentation.flow_threshold
#   .match_threshold_um           | .segmentation.match_threshold_um
#   .annotator                    | .annotation.strategy (use ground_truth)
#   .annotation_strategy          | .annotation.strategy
#   .model_names                  | .annotation.celltypist_models
#   .confidence_threshold         | .annotation.confidence_threshold
#   .ground_truth_file            | .annotation.ground_truth_file
#   .extended_consensus           | .annotation.extended_consensus
#   .patch_size                   | .lmdb.patch_sizes[0]
#   .output_format                | .lmdb.output_format
#   .normalization                | .lmdb.normalization
#   .backbone                     | .training.backbone
#   .epochs                       | .training.epochs
#   .batch_size                   | .training.batch_size
#   .learning_rate                | .training.learning_rate
#   .fine_grained                 | .annotation.fine_grained
#   .run_validation               | .validation.enabled
#   .validation_leiden            | .validation.run_leiden_check
#   .validation_dapi              | .validation.run_dapi_check
#   .validation_consensus         | .validation.run_consensus_check
#   .min_ari_threshold            | .validation.min_ari_threshold
#   .min_agreement_threshold      | .validation.min_agreement_threshold
#   .run_ground_truth_comparison  | .validation.run_ground_truth_comparison
#   .run_transfer_test            | .transfer.enabled
#   .transfer_target_dataset_id   | .transfer.target_dataset_id
#   .transfer_target_local_path   | .transfer.target_local_path
#   .transfer_target_platform     | .transfer.target_platform
#   .run_documentation            | .documentation.enabled
#   .obsidian_vault_path          | .documentation.obsidian_vault_path
#   .obsidian_folder              | .documentation.obsidian_folder
#   .doc_template                 | .documentation.template
#   .execute_remotely             | .execution.execute_remotely
#   .default_queue                | .execution.default_queue
#   .gpu_queue                    | .execution.gpu_queue
#   .cache_training               | .execution.cache_training
#
# GUIPipelineConfig               |
#   .raw_dataset_id               | .input.dataset_id
#   .raw_dataset_name             | .input.dataset_name
#   .raw_dataset_project          | .input.dataset_project
#   .s3_data_uri                  | .input.s3_uri
#   .platform                     | .input.platform
#   .annotation_strategy          | .annotation.strategy
#   .celltypist_models            | .annotation.celltypist_models
#   .include_singler              | .annotation.include_singler
#   .singler_reference            | .annotation.singler_reference
#   .include_sctype               | .annotation.include_sctype
#   .min_agreement                | .annotation.min_agreement
#   .confidence_threshold         | .annotation.confidence_threshold
#   .use_confidence_weighting     | .annotation.use_confidence_weighting
#   .fine_grained                 | .annotation.fine_grained
#   .use_cell_ontology            | .ontology.enabled
#   .cl_target_level              | .ontology.target_level
#   .cl_min_confidence            | .ontology.min_confidence
#   .cl_include_unmapped          | .ontology.include_unmapped
#   .cl_fuzzy_threshold           | .ontology.fuzzy_threshold
#   .patch_sizes                  | .lmdb.patch_sizes
#   .primary_patch_size           | .lmdb.primary_patch_size
#   .normalization                | .lmdb.normalization
#   .normalize_physical_size      | .lmdb.normalize_physical_size
#   .skip_if_lmdb_exists          | .lmdb.skip_if_exists
#   .backbone                     | .training.backbone
#   .training_mode                | .training.mode
#   .epochs                       | .training.epochs
#   .batch_size                   | .training.batch_size
#   .learning_rate                | .training.learning_rate
#   .weight_decay                 | .training.weight_decay
#   .patience                     | .training.patience
#   .coarse_only_epochs           | .training.coarse_only_epochs
#   .coarse_medium_epochs         | .training.coarse_medium_epochs
#   .warmup_epochs                | .training.warmup_epochs
#   .combine_patch_sizes          | .training.combine_patch_sizes
#   .output_dir                   | .output.output_dir
#   .upload_to_s3                 | .output.upload_to_s3
#   .s3_bucket                    | .output.s3_bucket
#   .s3_endpoint                  | .output.s3_endpoint
#   .register_datasets            | .output.register_datasets
#   .register_models              | .output.register_models
#   .link_to_parent               | .output.link_to_parent
#
# UniversalPipelineConfig         |
#   .name                         | DAPIDLPipelineConfig.name
#   .project                      | DAPIDLPipelineConfig.project
#   .version                      | DAPIDLPipelineConfig.version
#   .tissues                      | .input.tissues
#   .sampling_strategy            | .training.sampling_strategy
#   .tier1_weight                 | .training.tier1_weight
#   .tier2_weight                 | .training.tier2_weight
#   .tier3_weight                 | .training.tier3_weight
#   .segmenter                    | .segmentation.segmenter
#   .diameter                     | .segmentation.diameter
#   .flow_threshold               | .segmentation.flow_threshold
#   .patch_size                   | .lmdb.patch_sizes[0]
#   .output_format                | .lmdb.output_format
#   .backbone                     | .training.backbone
#   .epochs                       | .training.epochs
#   .batch_size                   | .training.batch_size
#   .learning_rate                | .training.learning_rate
#   .coarse_only_epochs           | .training.coarse_only_epochs
#   .coarse_medium_epochs         | .training.coarse_medium_epochs
#   .standardize_labels           | .ontology.enabled
#   .execute_remotely             | .execution.execute_remotely
#   .default_queue                | .execution.default_queue
#   .gpu_queue                    | .execution.gpu_queue
#   .output_dir                   | .output.output_dir
#
# TissueConfig                    | TissueDatasetConfig
#   .dataset_id                   | .dataset_id
#   .dataset_name                 | (removed - use dataset_id)
#   .local_path                   | .local_path
#   .tissue                       | .tissue
#   .platform                     | .platform
#   .confidence_tier              | .confidence_tier
#   .weight_multiplier            | .weight_multiplier
#   .annotator                    | (moved to annotation config)
#   .model_names                  | (moved to annotation config)
#   .ground_truth_file            | (moved to annotation config)
# ============================================================================
