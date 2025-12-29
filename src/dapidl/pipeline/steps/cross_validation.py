"""Cross-Modal Validation Pipeline Step.

Step 6 (Optional): Cross-modal validation of cell type predictions.

This step validates CellTypist annotations using orthogonal approaches:
1. Leiden clustering consistency (transcriptomic self-validation)
2. DAPI morphology model agreement (independent modality)
3. Multi-method consensus scoring

This enables validation WITHOUT ground truth annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger

from dapidl.pipeline.base import PipelineStep, StepArtifacts, resolve_artifact_path


@dataclass
class CrossValidationConfig:
    """Configuration for cross-modal validation step."""

    # Validation methods to run
    run_leiden_check: bool = True
    run_dapi_check: bool = True
    run_consensus_check: bool = True
    run_unsupervised_check: bool = True  # Truly independent validation

    # Ground truth comparison (optional - for datasets with known labels)
    run_ground_truth_comparison: bool = False
    ground_truth_file: str | None = None  # Excel/CSV/Parquet with ground truth
    ground_truth_sheet: str | None = None  # Sheet name for Excel files
    ground_truth_cell_id_col: str = "Barcode"  # Column with cell IDs
    ground_truth_label_col: str = "Cluster"  # Column with ground truth labels

    # Leiden clustering parameters
    leiden_resolutions: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.8, 1.0])
    min_ari_threshold: float = 0.5  # Warn if below this

    # DAPI model validation (circular - trained on CellTypist)
    dapi_batch_size: int = 256
    min_agreement_threshold: float = 0.5  # Warn if below this

    # Unsupervised validation (truly independent - uses pretrained backbone)
    unsupervised_backbone: str = "phikon"  # "phikon" (recommended) or "efficientnetv2_rw_s"
    unsupervised_n_clusters: int | None = None  # None = auto from labels
    unsupervised_clustering_method: str = "kmeans"  # "kmeans", "leiden", "hdbscan"
    unsupervised_batch_size: int = 64  # Smaller for ViT
    min_unsupervised_ari: float = 0.3  # Warn if below (0.3 = moderate support)

    # Confidence tier thresholds
    high_confidence_threshold: float = 0.7
    medium_confidence_threshold: float = 0.4

    # Morphology clustering (optional, uses trained model)
    run_morphology_clustering: bool = False
    morphology_n_clusters: int = 3
    morphology_method: str = "kmeans"  # "kmeans", "leiden", "hdbscan"

    # Output
    output_dir: str | None = None
    save_confidence_tiers: bool = True
    save_embeddings: bool = False  # Large file!


class CrossValidationStep(PipelineStep):
    """Cross-modal validation of cell type predictions.

    Validates CellTypist annotations using:
    - Leiden clustering (transcriptomic consistency)
    - DAPI morphology model (independent modality)
    - Multi-method consensus

    Queue: gpu (requires GPU for DAPI model inference)
    """

    name = "cross_validation"
    queue = "gpu"

    def __init__(self, config: CrossValidationConfig | None = None):
        """Initialize cross-validation step."""
        self.config = config or CrossValidationConfig()
        self._task = None

    def get_parameter_schema(self) -> dict[str, Any]:
        """Return JSON schema for ClearML UI parameters."""
        return {
            "type": "object",
            "properties": {
                "run_leiden_check": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run Leiden clustering validation",
                },
                "run_dapi_check": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run DAPI model agreement validation (circular)",
                },
                "run_consensus_check": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run multi-method consensus check",
                },
                "run_unsupervised_check": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run unsupervised morphology validation (truly independent)",
                },
                "unsupervised_backbone": {
                    "type": "string",
                    "default": "phikon",
                    "enum": ["phikon", "efficientnetv2_rw_s"],
                    "description": "Backbone for unsupervised validation (phikon recommended)",
                },
                "min_ari_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum Leiden ARI threshold (warn if below)",
                },
                "min_unsupervised_ari": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Minimum unsupervised ARI (0.3 = moderate support)",
                },
                "min_agreement_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Minimum DAPI agreement threshold",
                },
                "high_confidence_threshold": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Threshold for high confidence tier",
                },
            },
        }

    def validate_inputs(self, artifacts: StepArtifacts) -> bool:
        """Validate required inputs are present."""
        required = ["model_path", "patches_path", "annotations_parquet", "class_mapping"]
        for key in required:
            if key not in artifacts.inputs and key not in artifacts.outputs:
                logger.error(f"Missing required input: {key}")
                return False
        return True

    def execute(self, artifacts: StepArtifacts) -> StepArtifacts:
        """Run cross-modal validation."""
        cfg = self.config

        # Resolve artifact paths
        model_path = resolve_artifact_path(
            artifacts.outputs.get("model_path") or artifacts.inputs.get("model_path"),
            "model_path",
        )
        patches_path = resolve_artifact_path(
            artifacts.outputs.get("patches_path") or artifacts.inputs.get("patches_path"),
            "patches_path",
        )
        annotations_path = resolve_artifact_path(
            artifacts.outputs.get("annotations_parquet")
            or artifacts.inputs.get("annotations_parquet"),
            "annotations_parquet",
        )

        # Get class mapping
        class_mapping = artifacts.outputs.get("class_mapping") or artifacts.inputs.get(
            "class_mapping"
        )
        if isinstance(class_mapping, str):
            import json
            class_mapping = json.loads(class_mapping)

        # Setup output directory
        if cfg.output_dir:
            output_dir = Path(cfg.output_dir)
        else:
            output_dir = patches_path.parent / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("CROSS-MODAL VALIDATION")
        logger.info("=" * 60)

        # Initialize results
        results = {
            "leiden": {},
            "dapi": {},
            "consensus": {},
            "confidence_tiers": {},
            "warnings": [],
            "passed": True,
        }

        # Load annotations - prefer patches metadata.parquet (aligned with LMDB indices)
        # over annotations.parquet (which may have different row order)
        logger.info(f"Loading annotations from {annotations_path}")

        # Check if there's a metadata.parquet in the patches directory (aligned with LMDB)
        patches_metadata = patches_path.parent / "metadata.parquet"
        if patches_metadata.exists():
            logger.info(f"Using aligned metadata from {patches_metadata}")
            annotations_df = pl.read_parquet(patches_metadata)
        else:
            annotations_df = pl.read_parquet(annotations_path)

        # Support both column naming conventions
        if "broad_category" in annotations_df.columns:
            celltypist_labels = annotations_df["broad_category"].to_numpy()
        elif "cell_type" in annotations_df.columns:
            celltypist_labels = annotations_df["cell_type"].to_numpy()
        else:
            raise ValueError(f"No cell type column found. Available: {annotations_df.columns}")

        celltypist_confidence = annotations_df["confidence"].to_numpy()

        # Check for consensus score
        has_consensus = "consensus_score" in annotations_df.columns
        if has_consensus:
            consensus_scores = annotations_df["consensus_score"].to_numpy()
        else:
            consensus_scores = None

        # 1. LEIDEN CLUSTERING CHECK
        if cfg.run_leiden_check:
            logger.info("\n1. LEIDEN CLUSTERING CHECK")
            logger.info("-" * 40)

            try:
                leiden_results = self._run_leiden_check(
                    artifacts, annotations_df, celltypist_labels, cfg
                )
                results["leiden"] = leiden_results

                if leiden_results.get("best_ari", 0) < cfg.min_ari_threshold:
                    warning = f"Low Leiden ARI: {leiden_results['best_ari']:.3f}"
                    results["warnings"].append(warning)
                    logger.warning(warning)
            except Exception as e:
                logger.warning(f"Leiden check failed: {e}")
                results["leiden"]["error"] = str(e)

        # 2. DAPI MODEL CHECK
        if cfg.run_dapi_check:
            logger.info("\n2. DAPI MODEL CHECK")
            logger.info("-" * 40)

            try:
                dapi_results = self._run_dapi_check(
                    model_path,
                    patches_path,
                    celltypist_labels,
                    celltypist_confidence,
                    class_mapping,
                    cfg,
                )
                results["dapi"] = dapi_results

                if dapi_results.get("agreement_rate", 0) < cfg.min_agreement_threshold:
                    warning = f"Low DAPI agreement: {dapi_results['agreement_rate']:.1%}"
                    results["warnings"].append(warning)
                    logger.warning(warning)

            except Exception as e:
                logger.warning(f"DAPI check failed: {e}")
                results["dapi"]["error"] = str(e)

        # 3. CONSENSUS CHECK
        if cfg.run_consensus_check and has_consensus:
            logger.info("\n3. CONSENSUS CHECK")
            logger.info("-" * 40)

            consensus_results = self._run_consensus_analysis(consensus_scores, cfg)
            results["consensus"] = consensus_results

        # 4. UNSUPERVISED MORPHOLOGY CHECK (truly independent)
        if cfg.run_unsupervised_check:
            logger.info("\n4. UNSUPERVISED MORPHOLOGY CHECK")
            logger.info("-" * 40)
            logger.info(f"Using {cfg.unsupervised_backbone} backbone (pretrained, no CellTypist)")

            try:
                unsupervised_results = self._run_unsupervised_check(
                    patches_path, celltypist_labels, cfg
                )
                results["unsupervised"] = unsupervised_results

                if unsupervised_results.get("ari", 0) < cfg.min_unsupervised_ari:
                    warning = f"Low unsupervised ARI: {unsupervised_results['ari']:.3f} (weak morphological support)"
                    results["warnings"].append(warning)
                    logger.warning(warning)
                else:
                    logger.info(f"Unsupervised ARI: {unsupervised_results['ari']:.3f} - {unsupervised_results['interpretation']}")

            except Exception as e:
                logger.warning(f"Unsupervised check failed: {e}")
                results["unsupervised"] = {"error": str(e)}

        # 5. GROUND TRUTH COMPARISON (optional)
        if cfg.run_ground_truth_comparison and cfg.ground_truth_file:
            logger.info("\n5. GROUND TRUTH COMPARISON")
            logger.info("-" * 40)

            try:
                gt_results = self._run_ground_truth_comparison(
                    annotations_df, celltypist_labels, cfg
                )
                results["ground_truth"] = gt_results

                if gt_results.get("accuracy", 0) < 0.7:
                    warning = f"Low ground truth accuracy: {gt_results['accuracy']:.1%}"
                    results["warnings"].append(warning)
                    logger.warning(warning)
                else:
                    logger.info(f"Ground truth accuracy: {gt_results['accuracy']:.1%}")
                    logger.info(f"Ground truth macro F1: {gt_results['f1_macro']:.3f}")

            except Exception as e:
                logger.warning(f"Ground truth comparison failed: {e}")
                results["ground_truth"] = {"error": str(e)}

        # 6. COMPUTE CONFIDENCE TIERS
        logger.info("\n5. CONFIDENCE TIERS")
        logger.info("-" * 40)

        dapi_agreement = results.get("dapi", {}).get("agreement_mask")
        dapi_conf = results.get("dapi", {}).get("dapi_confidence")

        tier_results = self._compute_tiers(
            celltypist_confidence, dapi_conf, consensus_scores, dapi_agreement, cfg
        )
        results["confidence_tiers"] = tier_results

        # Save results
        self._save_results(results, output_dir)

        # Print summary
        self._print_summary(results)

        # Check overall pass/fail
        if results["warnings"]:
            results["passed"] = False

        return StepArtifacts(
            inputs=artifacts.inputs,
            outputs={
                **artifacts.outputs,
                "validation_results": results,
                "validation_report": str(output_dir / "validation_report.json"),
                "confidence_tiers_path": str(output_dir / "confidence_tiers.parquet"),
                "validation_passed": results["passed"],
            },
        )

    def _run_leiden_check(
        self,
        artifacts: StepArtifacts,
        annotations_df: pl.DataFrame,
        celltypist_labels: np.ndarray,
        cfg: CrossValidationConfig,
    ) -> dict:
        """Run Leiden clustering check."""
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        expression_path = resolve_artifact_path(
            artifacts.outputs.get("expression_path") or artifacts.inputs.get("expression_path"),
            "expression_path",
        )

        if expression_path is None or not expression_path.exists():
            logger.warning("No expression data - skipping Leiden check")
            return {"skipped": True, "reason": "No expression data"}

        try:
            import anndata as ad
            import scanpy as sc

            # Load expression data
            if expression_path.suffix in [".h5ad", ".h5"]:
                adata = ad.read_h5ad(expression_path)
            else:
                adata = sc.read_10x_h5(expression_path)

            # Add labels
            adata.obs["celltypist_labels"] = celltypist_labels[: len(adata)]

            # Preprocess if needed
            if "neighbors" not in adata.uns:
                logger.info("Computing neighbors...")
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
                sc.pp.pca(adata, n_comps=50)
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

            # Run Leiden at multiple resolutions
            best_ari = -1.0
            best_nmi = -1.0
            best_res = cfg.leiden_resolutions[0]
            all_results = []

            for res in cfg.leiden_resolutions:
                key = f"leiden_r{res}"
                sc.tl.leiden(adata, resolution=res, key_added=key)
                leiden_labels = adata.obs[key].values

                ari = adjusted_rand_score(adata.obs["celltypist_labels"], leiden_labels)
                nmi = normalized_mutual_info_score(adata.obs["celltypist_labels"], leiden_labels)

                logger.info(f"Leiden r={res}: ARI={ari:.3f}, NMI={nmi:.3f}")
                all_results.append({"resolution": res, "ari": ari, "nmi": nmi})

                if ari > best_ari:
                    best_ari = ari
                    best_nmi = nmi
                    best_res = res

            return {
                "best_ari": best_ari,
                "best_nmi": best_nmi,
                "best_resolution": best_res,
                "all_results": all_results,
            }

        except Exception as e:
            logger.warning(f"Leiden check failed: {e}")
            return {"error": str(e)}

    def _run_dapi_check(
        self,
        model_path: Path,
        patches_path: Path,
        celltypist_labels: np.ndarray,
        celltypist_confidence: np.ndarray,
        class_mapping: dict,
        cfg: CrossValidationConfig,
    ) -> dict:
        """Run DAPI model check."""
        import torch.nn.functional as F
        from dapidl.models.classifier import CellTypeClassifier

        # Load model
        logger.info(f"Loading model from {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = CellTypeClassifier.from_checkpoint(model_path)
        model.to(device)
        model.set_mode_inference() if hasattr(model, 'set_mode_inference') else None

        # Load patches - support both LMDB and Zarr formats
        logger.info(f"Loading patches from {patches_path}")

        is_lmdb = (patches_path / "data.mdb").exists()
        if is_lmdb:
            import lmdb
            import json as json_module

            # Load LMDB metadata (try JSON first, fall back to parquet)
            metadata_json = patches_path / "metadata.json"
            metadata_parquet = patches_path.parent / "metadata.parquet"

            if metadata_json.exists():
                with open(metadata_json) as f:
                    lmdb_meta = json_module.load(f)
                patch_shape = tuple(lmdb_meta["patch_shape"])
                dtype = np.dtype(lmdb_meta["dtype"])
                n_samples = lmdb_meta["n_samples"]
            elif metadata_parquet.exists():
                # Infer from parquet metadata and LMDB data
                meta_df = pl.read_parquet(metadata_parquet)
                n_samples = len(meta_df)

                # Probe first patch to detect shape and dtype
                with lmdb.open(str(patches_path), readonly=True, lock=False).begin() as txn:
                    probe_data = txn.get(b"00000000")
                    if probe_data is None:
                        raise ValueError("LMDB is empty")
                    byte_size = len(probe_data)

                # Determine dtype and shape based on byte size
                if byte_size == 128 * 128 * 4:  # float32
                    dtype = np.float32
                    patch_shape = (128, 128)
                elif byte_size == 256 * 256 * 1:  # uint8 256x256
                    dtype = np.uint8
                    patch_shape = (256, 256)
                elif byte_size == 128 * 128 * 2:  # uint16
                    dtype = np.uint16
                    patch_shape = (128, 128)
                else:
                    raise ValueError(f"Unknown LMDB patch format: {byte_size} bytes")

                logger.info(f"Detected from LMDB: patch_shape={patch_shape}, dtype={dtype}")
            else:
                raise FileNotFoundError(f"No metadata found for LMDB at {patches_path}")

            # Open LMDB
            env = lmdb.open(str(patches_path), readonly=True, lock=False)

            def get_lmdb_patch(idx: int) -> np.ndarray:
                with env.begin() as txn:
                    key = f"{idx:08d}".encode()
                    data = txn.get(key)
                    if data is None:
                        raise KeyError(f"Patch {idx} not found in LMDB")
                    patch = np.frombuffer(data, dtype=dtype).reshape(patch_shape)
                return patch

            dataset = type("LMDBDataset", (), {
                "__len__": lambda self: n_samples,
                "__getitem__": lambda self, idx: (get_lmdb_patch(idx), 0),
            })()
        else:
            from dapidl.data.dataset import DAPIDLDataset
            dataset = DAPIDLDataset(patches_path, transform=None)

        # Get class names
        class_names = [None] * len(class_mapping)
        for name, idx in class_mapping.items():
            class_names[idx] = name

        # Run inference
        all_probs = []
        all_preds = []
        n_samples = min(len(dataset), len(celltypist_labels))

        # Normalize patches to match training transforms
        # Training uses: [0,1] percentile norm -> A.Normalize(mean=0.5, std=0.25)
        def normalize_patch(patch: np.ndarray) -> torch.Tensor:
            """Normalize patch to match training transforms."""
            if patch.dtype != np.float32:
                patch = patch.astype(np.float32)

            # Step 1: Ensure [0, 1] range (percentile normalization)
            if patch.max() > 1.0:
                p_low, p_high = np.percentile(patch, [1, 99])
                if p_high > p_low:
                    patch = (patch - p_low) / (p_high - p_low)
                else:
                    max_val = 65535.0 if patch.max() > 255 else 255.0
                    patch = patch / max_val
            patch = np.clip(patch, 0, 1)

            # Step 2: Apply same standardization as training
            # A.Normalize(mean=0.5, std=0.25, max_pixel_value=1.0)
            patch = (patch - 0.5) / 0.25

            return torch.from_numpy(patch)

        logger.info(f"Running inference on {n_samples} samples...")

        with torch.no_grad():
            for i in range(0, n_samples, cfg.dapi_batch_size):
                end_idx = min(i + cfg.dapi_batch_size, n_samples)
                batch_patches = []

                for j in range(i, end_idx):
                    patch, _ = dataset[j]
                    if isinstance(patch, np.ndarray):
                        patch = normalize_patch(patch)
                    batch_patches.append(patch)

                batch = torch.stack(batch_patches).to(device)
                if batch.ndim == 3:
                    batch = batch.unsqueeze(1)

                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())

                # Progress logging every 1000 samples
                if (i + cfg.dapi_batch_size) % 5000 < cfg.dapi_batch_size:
                    logger.info(f"  Processed {min(end_idx, n_samples)}/{n_samples}")

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()

        # Confidence scores
        dapi_confidence = all_probs.max(axis=1)

        # Convert labels to indices
        label_to_idx = {name: i for i, name in enumerate(class_names)}
        ct_indices = np.array([label_to_idx.get(str(l), -1) for l in celltypist_labels[:n_samples]])

        # Agreement
        valid_mask = ct_indices >= 0
        agreement_mask = all_preds[valid_mask] == ct_indices[valid_mask]
        agreement_rate = agreement_mask.mean()

        logger.info(f"DAPI Agreement Rate: {agreement_rate:.1%}")

        # Cross-modal correlation
        valid_ct_conf = celltypist_confidence[:n_samples][valid_mask]
        valid_dapi_conf = dapi_confidence[valid_mask]

        if len(valid_ct_conf) > 0:
            correlation = np.corrcoef(valid_ct_conf, valid_dapi_conf)[0, 1]
        else:
            correlation = 0.0

        logger.info(f"Confidence correlation: {correlation:.3f}")

        # Per-class agreement
        per_class = {}
        for name, idx in label_to_idx.items():
            mask = ct_indices == idx
            if mask.sum() > 0:
                class_agr = (all_preds[mask] == idx).mean()
                per_class[name] = float(class_agr)
                logger.info(f"  {name}: {class_agr:.1%}")

        return {
            "agreement_rate": float(agreement_rate),
            "cross_modal_correlation": float(correlation),
            "per_class_agreement": per_class,
            "mean_dapi_confidence": float(dapi_confidence.mean()),
            "dapi_confidence": dapi_confidence,
            "agreement_mask": agreement_mask,
        }

    def _run_consensus_analysis(self, consensus_scores: np.ndarray, cfg: CrossValidationConfig) -> dict:
        """Analyze multi-method consensus statistics."""
        high_consensus = (consensus_scores >= 0.5).mean()
        low_consensus = (consensus_scores < 0.3).mean()

        logger.info(f"High consensus (>=0.5): {high_consensus:.1%}")
        logger.info(f"Low consensus (<0.3): {low_consensus:.1%}")

        return {
            "mean_consensus": float(consensus_scores.mean()),
            "high_consensus_fraction": float(high_consensus),
            "low_consensus_fraction": float(low_consensus),
        }

    def _run_unsupervised_check(
        self,
        patches_path: Path,
        celltypist_labels: np.ndarray,
        cfg: CrossValidationConfig,
    ) -> dict:
        """Run unsupervised morphology validation (truly independent).

        Uses a pretrained backbone (Phikon or ImageNet) to extract features,
        clusters them, and compares to CellTypist labels. This avoids the
        circularity of using a model trained on CellTypist.
        """
        import lmdb
        from dapidl.validation import unsupervised_morphology_validation

        # Load patches from LMDB
        is_lmdb = (patches_path / "data.mdb").exists()
        if not is_lmdb:
            logger.warning("Unsupervised check requires LMDB patches")
            return {"error": "LMDB patches not found"}

        # Detect patch format
        with lmdb.open(str(patches_path), readonly=True, lock=False).begin() as txn:
            probe_data = txn.get(b"00000000")
            if probe_data is None:
                return {"error": "LMDB is empty"}
            byte_size = len(probe_data)

        # Determine dtype and shape
        if byte_size == 128 * 128 * 4:  # float32
            dtype = np.float32
            patch_shape = (128, 128)
        elif byte_size == 256 * 256 * 1:  # uint8 256x256
            dtype = np.uint8
            patch_shape = (256, 256)
        elif byte_size == 128 * 128 * 2:  # uint16
            dtype = np.uint16
            patch_shape = (128, 128)
        else:
            return {"error": f"Unknown patch format: {byte_size} bytes"}

        # Load patches
        n_samples = len(celltypist_labels)
        logger.info(f"Loading {n_samples} patches from LMDB...")

        env = lmdb.open(str(patches_path), readonly=True, lock=False)
        patches = []
        with env.begin() as txn:
            for i in range(n_samples):
                key = f"{i:08d}".encode()
                data = txn.get(key)
                if data is None:
                    break
                patch = np.frombuffer(data, dtype=dtype).reshape(patch_shape)
                patches.append(patch)
        env.close()

        patches = np.array(patches, dtype=np.float32)
        if patches.max() > 1.0:
            patches = patches / patches.max()  # Normalize to [0, 1]

        logger.info(f"Loaded {len(patches)} patches, shape: {patches.shape}")

        # Run unsupervised validation
        results = unsupervised_morphology_validation(
            patches=patches,
            celltypist_labels=celltypist_labels[:len(patches)],
            n_clusters=cfg.unsupervised_n_clusters,
            backbone_name=cfg.unsupervised_backbone,
            clustering_method=cfg.unsupervised_clustering_method,
            batch_size=cfg.unsupervised_batch_size,
        )

        return results

    def _compute_tiers(
        self,
        celltypist_confidence: np.ndarray,
        dapi_confidence: np.ndarray | None,
        consensus_scores: np.ndarray | None,
        dapi_agreement: np.ndarray | None,
        cfg: CrossValidationConfig,
    ) -> dict:
        """Compute confidence tiers for cells."""
        scores = celltypist_confidence.copy()

        # Boost/penalize based on DAPI agreement
        if dapi_agreement is not None:
            full_agreement = np.ones(len(scores), dtype=bool)
            full_agreement[: len(dapi_agreement)] = dapi_agreement
            scores = np.where(full_agreement, scores * 1.1, scores * 0.8)

        # Factor in consensus
        if consensus_scores is not None:
            scores = scores * np.sqrt(consensus_scores)

        scores = np.clip(scores, 0, 1)

        # Assign tiers
        high_mask = scores >= cfg.high_confidence_threshold
        medium_mask = (scores >= cfg.medium_confidence_threshold) & ~high_mask
        low_mask = ~high_mask & ~medium_mask

        results = {
            "high_count": int(high_mask.sum()),
            "high_fraction": float(high_mask.mean()),
            "medium_count": int(medium_mask.sum()),
            "medium_fraction": float(medium_mask.mean()),
            "low_count": int(low_mask.sum()),
            "low_fraction": float(low_mask.mean()),
        }

        logger.info(f"High confidence: {results['high_count']} ({results['high_fraction']:.1%})")
        logger.info(f"Medium confidence: {results['medium_count']} ({results['medium_fraction']:.1%})")
        logger.info(f"Low confidence: {results['low_count']} ({results['low_fraction']:.1%})")

        return results

    def _save_results(self, results: dict, output_dir: Path) -> None:
        """Save results to files."""
        import json

        report_path = output_dir / "validation_report.json"

        # Remove numpy arrays before JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: v for k, v in value.items()
                    if not isinstance(v, np.ndarray)
                }
            elif not isinstance(value, np.ndarray):
                json_results[key] = value

        with open(report_path, "w") as f:
            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Saved validation report to {report_path}")

    def _print_summary(self, results: dict) -> None:
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if results["leiden"]:
            if "best_ari" in results["leiden"]:
                print(f"\nLeiden ARI: {results['leiden']['best_ari']:.3f}")
            elif "error" in results["leiden"]:
                print(f"\nLeiden: Error - {results['leiden']['error']}")

        if results["dapi"]:
            if "agreement_rate" in results["dapi"]:
                print(f"DAPI Agreement: {results['dapi']['agreement_rate']:.1%}")

        if results["consensus"]:
            print(f"Consensus Mean: {results['consensus'].get('mean_consensus', 'N/A'):.3f}")

        if results.get("unsupervised"):
            unsup = results["unsupervised"]
            if "ari" in unsup:
                print(f"\nUnsupervised Morphology Validation:")
                print(f"  ARI: {unsup['ari']:.3f} ({unsup.get('interpretation', 'N/A')})")
                print(f"  NMI: {unsup.get('nmi', 0):.3f}")
                print(f"  Purity: {unsup.get('overall_purity', 0):.1%}")
                print(f"  Backbone: {unsup.get('backbone', 'N/A')}")
            elif "error" in unsup:
                print(f"\nUnsupervised: Error - {unsup['error']}")

        tiers = results["confidence_tiers"]
        print(f"\nConfidence Tiers:")
        print(f"  High: {tiers.get('high_fraction', 0):.1%}")
        print(f"  Medium: {tiers.get('medium_fraction', 0):.1%}")
        print(f"  Low: {tiers.get('low_fraction', 0):.1%}")

        if results.get("ground_truth"):
            gt = results["ground_truth"]
            if "accuracy" in gt:
                print(f"\nGround Truth Comparison:")
                print(f"  Accuracy: {gt['accuracy']:.1%}")
                print(f"  Macro F1: {gt['f1_macro']:.3f}")
                print(f"  Matched cells: {gt.get('matched_cells', 0)}")

        if results["warnings"]:
            print(f"\nWARNINGS ({len(results['warnings'])}):")
            for w in results["warnings"]:
                print(f"  - {w}")

        print("\n" + "=" * 60)

    def _run_ground_truth_comparison(
        self,
        annotations_df: pl.DataFrame,
        celltypist_labels: np.ndarray,
        cfg: CrossValidationConfig,
    ) -> dict:
        """Compare CellTypist predictions to ground truth annotations.

        Args:
            annotations_df: DataFrame with CellTypist predictions
            celltypist_labels: Predicted labels array
            cfg: Configuration with ground truth file path

        Returns:
            Dictionary with accuracy, F1, per-class metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )

        # Load ground truth
        gt_path = Path(cfg.ground_truth_file)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        logger.info(f"Loading ground truth from {gt_path}")

        # Load based on file type
        if gt_path.suffix == ".xlsx":
            import pandas as pd
            # If no sheet specified, use first sheet (index 0)
            sheet = cfg.ground_truth_sheet if cfg.ground_truth_sheet else 0
            gt_df = pd.read_excel(gt_path, sheet_name=sheet)
            gt_df = pl.from_pandas(gt_df)
        elif gt_path.suffix == ".csv":
            gt_df = pl.read_csv(gt_path)
        elif gt_path.suffix == ".parquet":
            gt_df = pl.read_parquet(gt_path)
        else:
            raise ValueError(f"Unsupported ground truth format: {gt_path.suffix}")

        logger.info(f"Ground truth has {len(gt_df)} cells")

        # Get cell ID and label columns
        cell_id_col = cfg.ground_truth_cell_id_col
        label_col = cfg.ground_truth_label_col

        if cell_id_col not in gt_df.columns:
            # Try common alternatives
            for alt in ["Barcode", "cell_id", "CellID", "barcode"]:
                if alt in gt_df.columns:
                    cell_id_col = alt
                    break
            else:
                raise ValueError(f"Cell ID column not found. Available: {gt_df.columns}")

        if label_col not in gt_df.columns:
            # Try common alternatives
            for alt in ["Cluster", "Annotation", "cell_type", "CellType", "label", "broad_category"]:
                if alt in gt_df.columns:
                    label_col = alt
                    break
            else:
                raise ValueError(f"Label column not found. Available: {gt_df.columns}")

        logger.info(f"Using columns: {cell_id_col} (ID), {label_col} (label)")

        # Map ground truth labels to broad categories (if needed)
        from dapidl.data.annotation import map_to_broad_category

        gt_df = gt_df.with_columns([
            pl.col(label_col).map_elements(
                map_to_broad_category, return_dtype=pl.Utf8
            ).alias("gt_broad_category")
        ])

        # Get predicted broad categories
        if "cell_id" in annotations_df.columns:
            pred_cell_ids = annotations_df["cell_id"].to_list()
        elif "Barcode" in annotations_df.columns:
            pred_cell_ids = annotations_df["Barcode"].to_list()
        else:
            raise ValueError(f"No cell ID column in predictions. Available: {annotations_df.columns}")

        # Match predictions to ground truth
        gt_cell_ids = set(gt_df[cell_id_col].to_list())
        matched_indices = []
        matched_gt_labels = []
        matched_pred_labels = []

        for i, (cell_id, pred_label) in enumerate(zip(pred_cell_ids, celltypist_labels)):
            if cell_id in gt_cell_ids:
                gt_row = gt_df.filter(pl.col(cell_id_col) == cell_id)
                if len(gt_row) > 0:
                    gt_label = gt_row["gt_broad_category"][0]
                    matched_indices.append(i)
                    matched_gt_labels.append(gt_label)
                    matched_pred_labels.append(pred_label)

        logger.info(f"Matched {len(matched_indices)} cells to ground truth")

        if len(matched_indices) < 10:
            return {
                "error": f"Only {len(matched_indices)} cells matched to ground truth",
                "matched_cells": len(matched_indices),
            }

        # Calculate metrics
        accuracy = accuracy_score(matched_gt_labels, matched_pred_labels)
        f1_macro = f1_score(matched_gt_labels, matched_pred_labels, average="macro")
        f1_weighted = f1_score(matched_gt_labels, matched_pred_labels, average="weighted")

        # Get classification report
        report = classification_report(
            matched_gt_labels, matched_pred_labels, output_dict=True
        )

        # Confusion matrix
        labels = sorted(set(matched_gt_labels) | set(matched_pred_labels))
        cm = confusion_matrix(matched_gt_labels, matched_pred_labels, labels=labels)

        logger.info(f"Ground truth accuracy: {accuracy:.1%}")
        logger.info(f"Ground truth macro F1: {f1_macro:.3f}")
        logger.info(f"Per-class F1:")
        for label in labels:
            if label in report:
                logger.info(f"  {label}: F1={report[label].get('f1-score', 0):.3f}")

        return {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "matched_cells": len(matched_indices),
            "total_gt_cells": len(gt_df),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }
