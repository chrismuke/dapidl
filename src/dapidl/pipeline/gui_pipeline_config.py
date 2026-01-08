"""GUI-friendly pipeline configuration for ClearML.

This module provides configuration dataclasses with parameter groups
designed for easy editing in the ClearML web interface.

Parameter groups:
- Input Selection: Raw dataset selection (ClearML Dataset or S3)
- Annotation Configuration: Ensemble methods and consensus settings
- LMDB Configuration: Patch sizes and normalization
- Training Configuration: Model and curriculum settings
- Output Configuration: S3 upload and ClearML registration
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GUIPipelineConfig:
    """Enhanced pipeline configuration with ClearML GUI parameter groups.

    This configuration is designed for the ClearML web interface, with
    parameters organized into logical groups for easy editing.

    All parameters have sensible defaults so the pipeline can run with
    minimal configuration.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP: Input Selection
    # ═══════════════════════════════════════════════════════════════════════

    # Raw dataset selection (ClearML Dataset)
    raw_dataset_id: str | None = None  # ClearML Dataset ID
    raw_dataset_name: str | None = None  # Or search by name pattern
    raw_dataset_project: str = "DAPIDL/raw"

    # S3 direct access (alternative to ClearML Dataset)
    s3_data_uri: str | None = None  # s3://dapidl/raw-data/xenium-breast-rep1/

    # Platform detection
    platform: str = "auto"  # "auto", "xenium", "merscope"

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP: Annotation Configuration
    # ═══════════════════════════════════════════════════════════════════════

    # Ensemble strategy
    annotation_strategy: str = "ensemble"  # "ensemble", "single", "consensus"

    # CellTypist models (multi-select in GUI)
    celltypist_models: list[str] = field(
        default_factory=lambda: [
            "Cells_Adult_Breast.pkl",
            "Immune_All_High.pkl",
        ]
    )

    # Additional annotators to include
    include_singler: bool = True
    singler_reference: str = "blueprint"  # "blueprint", "hpca", "monaco"
    include_sctype: bool = False

    # Consensus settings
    min_agreement: int = 2  # Minimum annotators agreeing
    confidence_threshold: float = 0.5
    use_confidence_weighting: bool = True

    # Fine-grained vs coarse classification
    fine_grained: bool = True  # Use detailed cell types

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP: Cell Ontology Standardization
    # ═══════════════════════════════════════════════════════════════════════

    # Enable CL standardization (maps all labels to Cell Ontology IDs)
    use_cell_ontology: bool = True

    # Target hierarchy level for classification
    cl_target_level: str = "coarse"  # "broad", "coarse", "fine"

    # Mapping confidence threshold (lower = more permissive)
    cl_min_confidence: float = 0.5

    # Include cells with unmapped labels (as "Unknown")
    cl_include_unmapped: bool = False

    # Fuzzy matching threshold for label mapping
    cl_fuzzy_threshold: float = 0.85

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP: LMDB Dataset Configuration
    # ═══════════════════════════════════════════════════════════════════════

    # Patch sizes to generate (multi-select)
    patch_sizes: list[int] = field(default_factory=lambda: [128])

    # Normalization
    normalization: str = "adaptive"  # "adaptive", "percentile", "minmax"
    normalize_physical_size: bool = True  # Cross-platform compatibility

    # Skip existing
    skip_if_lmdb_exists: bool = True  # Check for existing LMDB dataset

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP: Training Configuration
    # ═══════════════════════════════════════════════════════════════════════

    # Model architecture
    backbone: str = "efficientnetv2_rw_s"
    training_mode: str = "hierarchical"  # "hierarchical", "flat"

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15

    # Curriculum learning (for hierarchical mode)
    coarse_only_epochs: int = 20
    coarse_medium_epochs: int = 50
    warmup_epochs: int = 5

    # Multi-dataset training
    combine_patch_sizes: bool = False  # Train on all patch sizes together
    primary_patch_size: int | None = None  # Or use specific patch size

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP: Output Configuration
    # ═══════════════════════════════════════════════════════════════════════

    # S3 upload
    upload_to_s3: bool = True
    s3_bucket: str = "dapidl"
    s3_endpoint: str = "https://s3.eu-central-2.idrivee2.com"

    # ClearML registration
    register_datasets: bool = True
    register_models: bool = True

    # Lineage tracking
    link_to_parent: bool = True  # Create dataset lineage to save space

    # Output directories
    output_dir: str = "./pipeline_output"

    def to_parameter_dict(self) -> dict[str, str]:
        """Convert to flat parameter dictionary for ClearML.

        ClearML's add_parameter() requires string values, so we
        serialize lists and booleans appropriately.
        """
        return {
            # Input Selection
            "input/raw_dataset_id": self.raw_dataset_id or "",
            "input/raw_dataset_name": self.raw_dataset_name or "",
            "input/raw_dataset_project": self.raw_dataset_project,
            "input/s3_data_uri": self.s3_data_uri or "",
            "input/platform": self.platform,
            # Annotation Configuration
            "annotation/strategy": self.annotation_strategy,
            "annotation/celltypist_models": ",".join(self.celltypist_models),
            "annotation/include_singler": str(self.include_singler),
            "annotation/singler_reference": self.singler_reference,
            "annotation/include_sctype": str(self.include_sctype),
            "annotation/min_agreement": str(self.min_agreement),
            "annotation/confidence_threshold": str(self.confidence_threshold),
            "annotation/use_confidence_weighting": str(self.use_confidence_weighting),
            "annotation/fine_grained": str(self.fine_grained),
            # Cell Ontology Configuration
            "ontology/use_cell_ontology": str(self.use_cell_ontology),
            "ontology/target_level": self.cl_target_level,
            "ontology/min_confidence": str(self.cl_min_confidence),
            "ontology/include_unmapped": str(self.cl_include_unmapped),
            "ontology/fuzzy_threshold": str(self.cl_fuzzy_threshold),
            # LMDB Configuration
            "lmdb/patch_sizes": ",".join(str(p) for p in self.patch_sizes),
            "lmdb/normalization": self.normalization,
            "lmdb/normalize_physical_size": str(self.normalize_physical_size),
            "lmdb/skip_if_lmdb_exists": str(self.skip_if_lmdb_exists),
            # Training Configuration
            "training/backbone": self.backbone,
            "training/training_mode": self.training_mode,
            "training/epochs": str(self.epochs),
            "training/batch_size": str(self.batch_size),
            "training/learning_rate": str(self.learning_rate),
            "training/weight_decay": str(self.weight_decay),
            "training/patience": str(self.patience),
            "training/coarse_only_epochs": str(self.coarse_only_epochs),
            "training/coarse_medium_epochs": str(self.coarse_medium_epochs),
            "training/warmup_epochs": str(self.warmup_epochs),
            "training/combine_patch_sizes": str(self.combine_patch_sizes),
            "training/primary_patch_size": str(self.primary_patch_size or ""),
            # Output Configuration
            "output/upload_to_s3": str(self.upload_to_s3),
            "output/s3_bucket": self.s3_bucket,
            "output/s3_endpoint": self.s3_endpoint,
            "output/register_datasets": str(self.register_datasets),
            "output/register_models": str(self.register_models),
            "output/link_to_parent": str(self.link_to_parent),
            "output/output_dir": self.output_dir,
        }

    @classmethod
    def from_parameter_dict(cls, params: dict[str, str]) -> "GUIPipelineConfig":
        """Create config from ClearML parameter dictionary.

        Handles deserialization of lists and booleans.
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

        return cls(
            # Input Selection
            raw_dataset_id=params.get("input/raw_dataset_id") or None,
            raw_dataset_name=params.get("input/raw_dataset_name") or None,
            raw_dataset_project=params.get("input/raw_dataset_project", "DAPIDL/raw"),
            s3_data_uri=params.get("input/s3_data_uri") or None,
            platform=params.get("input/platform", "auto"),
            # Annotation Configuration
            annotation_strategy=params.get("annotation/strategy", "ensemble"),
            celltypist_models=parse_list_str(
                params.get("annotation/celltypist_models", "Cells_Adult_Breast.pkl,Immune_All_High.pkl")
            ),
            include_singler=parse_bool(params.get("annotation/include_singler", "True")),
            singler_reference=params.get("annotation/singler_reference", "blueprint"),
            include_sctype=parse_bool(params.get("annotation/include_sctype", "False")),
            min_agreement=int(params.get("annotation/min_agreement", "2")),
            confidence_threshold=float(params.get("annotation/confidence_threshold", "0.5")),
            use_confidence_weighting=parse_bool(
                params.get("annotation/use_confidence_weighting", "True")
            ),
            fine_grained=parse_bool(params.get("annotation/fine_grained", "True")),
            # Cell Ontology Configuration
            use_cell_ontology=parse_bool(params.get("ontology/use_cell_ontology", "True")),
            cl_target_level=params.get("ontology/target_level", "coarse"),
            cl_min_confidence=float(params.get("ontology/min_confidence", "0.5")),
            cl_include_unmapped=parse_bool(params.get("ontology/include_unmapped", "False")),
            cl_fuzzy_threshold=float(params.get("ontology/fuzzy_threshold", "0.85")),
            # LMDB Configuration
            patch_sizes=parse_list_int(params.get("lmdb/patch_sizes", "128")),
            normalization=params.get("lmdb/normalization", "adaptive"),
            normalize_physical_size=parse_bool(
                params.get("lmdb/normalize_physical_size", "True")
            ),
            skip_if_lmdb_exists=parse_bool(params.get("lmdb/skip_if_lmdb_exists", "True")),
            # Training Configuration
            backbone=params.get("training/backbone", "efficientnetv2_rw_s"),
            training_mode=params.get("training/training_mode", "hierarchical"),
            epochs=int(params.get("training/epochs", "100")),
            batch_size=int(params.get("training/batch_size", "64")),
            learning_rate=float(params.get("training/learning_rate", "1e-4")),
            weight_decay=float(params.get("training/weight_decay", "1e-5")),
            patience=int(params.get("training/patience", "15")),
            coarse_only_epochs=int(params.get("training/coarse_only_epochs", "20")),
            coarse_medium_epochs=int(params.get("training/coarse_medium_epochs", "50")),
            warmup_epochs=int(params.get("training/warmup_epochs", "5")),
            combine_patch_sizes=parse_bool(
                params.get("training/combine_patch_sizes", "False")
            ),
            primary_patch_size=int(params["training/primary_patch_size"])
            if params.get("training/primary_patch_size")
            else None,
            # Output Configuration
            upload_to_s3=parse_bool(params.get("output/upload_to_s3", "True")),
            s3_bucket=params.get("output/s3_bucket", "dapidl"),
            s3_endpoint=params.get(
                "output/s3_endpoint", "https://s3.eu-central-2.idrivee2.com"
            ),
            register_datasets=parse_bool(params.get("output/register_datasets", "True")),
            register_models=parse_bool(params.get("output/register_models", "True")),
            link_to_parent=parse_bool(params.get("output/link_to_parent", "True")),
            output_dir=params.get("output/output_dir", "./pipeline_output"),
        )
