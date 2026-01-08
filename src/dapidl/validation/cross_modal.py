"""Cross-Modal Validation for Cell Type Annotations.

Validates CellTypist predictions using orthogonal approaches:
1. Leiden clustering consistency (transcriptomic self-validation)
2. DAPI morphology model agreement (independent modality)
3. Multi-method consensus scoring

This enables validation WITHOUT ground truth annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

if TYPE_CHECKING:
    import anndata as ad

    from dapidl.models.classifier import CellTypeClassifier


@dataclass
class ValidationMetrics:
    """Cross-modal validation metrics for a dataset."""

    # Leiden clustering metrics
    leiden_ari: float  # Adjusted Rand Index vs CellTypist
    leiden_nmi: float  # Normalized Mutual Information
    leiden_resolution: float  # Best resolution used

    # DAPI morphology metrics
    dapi_agreement_rate: float  # Fraction of cells where DAPI agrees
    dapi_mean_confidence: float  # Mean DAPI model confidence
    cross_modal_correlation: float  # Correlation between confidences

    # Consensus metrics
    consensus_mean: float  # Mean multi-model consensus score
    high_confidence_fraction: float  # Fraction with confidence >= 0.7

    # Per-class breakdown
    per_class_agreement: dict[str, float]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CROSS-MODAL VALIDATION SUMMARY",
            "=" * 60,
            "",
            "1. LEIDEN CLUSTERING CONSISTENCY",
            f"   ARI (vs CellTypist): {self.leiden_ari:.3f}",
            f"   NMI (vs CellTypist): {self.leiden_nmi:.3f}",
            f"   Best resolution: {self.leiden_resolution}",
            self._interpret_ari(self.leiden_ari),
            "",
            "2. DAPI MORPHOLOGY VALIDATION",
            f"   Agreement rate: {self.dapi_agreement_rate:.1%}",
            f"   Mean DAPI confidence: {self.dapi_mean_confidence:.3f}",
            f"   Cross-modal correlation: {self.cross_modal_correlation:.3f}",
            "",
            "3. MULTI-METHOD CONSENSUS",
            f"   Mean consensus score: {self.consensus_mean:.3f}",
            f"   High-confidence cells: {self.high_confidence_fraction:.1%}",
            "",
            "4. PER-CLASS AGREEMENT (DAPI vs CellTypist)",
        ]
        for cls, rate in sorted(self.per_class_agreement.items()):
            lines.append(f"   {cls}: {rate:.1%}")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    @staticmethod
    def _interpret_ari(ari: float) -> str:
        if ari >= 0.8:
            return "   -> HIGH: Excellent agreement with transcriptomic structure"
        elif ari >= 0.5:
            return "   -> MODERATE: Good agreement, some discrepancies"
        else:
            return "   -> LOW: Significant discrepancy - investigate clusters"


def compute_leiden_metrics(
    adata: "ad.AnnData",
    celltypist_key: str = "celltypist_labels",
    resolutions: list[float] | None = None,
) -> tuple[float, float, float]:
    """Compare CellTypist predictions with Leiden clustering.

    Args:
        adata: AnnData with CellTypist predictions and expression data
        celltypist_key: Column name for CellTypist labels
        resolutions: Leiden resolutions to test (default: [0.3, 0.5, 0.8, 1.0])

    Returns:
        (best_ari, best_nmi, best_resolution)
    """
    import scanpy as sc

    if resolutions is None:
        resolutions = [0.3, 0.5, 0.8, 1.0]

    # Ensure we have neighbors computed
    if "neighbors" not in adata.uns:
        logger.info("Computing neighbors for Leiden clustering...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

    best_ari = -1.0
    best_nmi = -1.0
    best_resolution = resolutions[0]

    celltypist_labels = adata.obs[celltypist_key].values

    for resolution in resolutions:
        key = f"leiden_r{resolution}"
        sc.tl.leiden(adata, resolution=resolution, key_added=key)

        leiden_labels = adata.obs[key].values

        ari = adjusted_rand_score(celltypist_labels, leiden_labels)
        nmi = normalized_mutual_info_score(celltypist_labels, leiden_labels)

        logger.info(f"Leiden r={resolution}: ARI={ari:.3f}, NMI={nmi:.3f}")

        if ari > best_ari:
            best_ari = ari
            best_nmi = nmi
            best_resolution = resolution

    return best_ari, best_nmi, best_resolution


def compute_dapi_agreement(
    model: "CellTypeClassifier",
    patches: torch.Tensor,
    celltypist_labels: np.ndarray,
    celltypist_confidence: np.ndarray,
    class_names: list[str],
    device: str = "cuda",
) -> tuple[float, float, float, dict[str, float]]:
    """Compare DAPI model predictions with CellTypist.

    Args:
        model: Trained DAPI classifier
        patches: Tensor of DAPI patches (N, 1, H, W)
        celltypist_labels: CellTypist predicted labels (N,)
        celltypist_confidence: CellTypist confidence scores (N,)
        class_names: List of class names
        device: Device for inference

    Returns:
        (agreement_rate, mean_dapi_confidence, cross_modal_correlation, per_class_agreement)
    """
    model.set_mode_inference()
    model.to(device)

    # Batch inference
    batch_size = 256
    all_probs = []
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_predictions.append(predictions.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    # Confidence scores
    dapi_confidence = all_probs.max(axis=1)

    # Convert CellTypist labels to indices
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    celltypist_indices = np.array([label_to_idx.get(l, -1) for l in celltypist_labels])

    # Agreement (only for valid labels)
    valid_mask = celltypist_indices >= 0
    agreement = (all_predictions[valid_mask] == celltypist_indices[valid_mask]).mean()

    # Mean confidence
    mean_dapi_conf = dapi_confidence.mean()

    # Cross-modal correlation
    valid_ct_conf = celltypist_confidence[valid_mask]
    valid_dapi_conf = dapi_confidence[valid_mask]
    correlation = np.corrcoef(valid_ct_conf, valid_dapi_conf)[0, 1]

    # Per-class agreement
    per_class = {}
    for name, idx in label_to_idx.items():
        mask = celltypist_indices == idx
        if mask.sum() > 0:
            class_agreement = (all_predictions[mask] == idx).mean()
            per_class[name] = float(class_agreement)

    return float(agreement), float(mean_dapi_conf), float(correlation), per_class


def extract_morphology_embeddings(
    model: "CellTypeClassifier",
    patches: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """Extract backbone feature embeddings for morphology clustering.

    Args:
        model: Trained DAPI classifier
        patches: Tensor of DAPI patches (N, 1, H, W)
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        Embeddings array of shape (N, num_features)
    """
    model.set_mode_inference()
    model.to(device)

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size].to(device)
            features = model.get_features(batch)
            all_embeddings.append(features.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def cluster_morphology_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 3,
    method: str = "kmeans",
) -> np.ndarray:
    """Cluster cells based on morphology embeddings.

    Args:
        embeddings: Feature embeddings (N, num_features)
        n_clusters: Number of clusters
        method: Clustering method ("kmeans", "leiden", or "hdbscan")

    Returns:
        Cluster labels (N,)
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans

        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return clusterer.fit_predict(embeddings)

    elif method == "hdbscan":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
        return clusterer.fit_predict(embeddings)

    elif method == "leiden":
        # Use scanpy's Leiden on a kNN graph of embeddings
        import scanpy as sc
        import anndata as ad

        adata_temp = ad.AnnData(X=embeddings)
        sc.pp.neighbors(adata_temp, n_neighbors=15, use_rep="X")
        sc.tl.leiden(adata_temp, resolution=0.5)
        return adata_temp.obs["leiden"].astype(int).values

    else:
        raise ValueError(f"Unknown clustering method: {method}")


def compute_confidence_tiers(
    celltypist_confidence: np.ndarray,
    dapi_confidence: np.ndarray | None = None,
    consensus_score: np.ndarray | None = None,
    dapi_agreement: np.ndarray | None = None,
) -> pl.DataFrame:
    """Assign confidence tiers to cells based on multiple validation signals.

    Args:
        celltypist_confidence: CellTypist confidence scores
        dapi_confidence: DAPI model confidence scores (optional)
        consensus_score: Multi-model consensus scores (optional)
        dapi_agreement: Boolean array of DAPI agreement (optional)

    Returns:
        DataFrame with confidence tier assignments
    """
    n_cells = len(celltypist_confidence)

    # Start with CellTypist confidence
    scores = celltypist_confidence.copy()

    # Boost/penalize based on DAPI agreement
    if dapi_agreement is not None:
        # Cells where DAPI agrees get a boost
        scores = np.where(dapi_agreement, scores * 1.1, scores * 0.8)

    # Factor in consensus
    if consensus_score is not None:
        # Weight by sqrt of consensus (softer penalty)
        scores = scores * np.sqrt(consensus_score)

    # Normalize to 0-1
    scores = np.clip(scores, 0, 1)

    # Assign tiers
    tiers = np.where(
        scores >= 0.7,
        "high",
        np.where(scores >= 0.4, "medium", "low"),
    )

    return pl.DataFrame(
        {
            "cell_idx": range(n_cells),
            "combined_score": scores,
            "confidence_tier": tiers,
            "celltypist_confidence": celltypist_confidence,
            "dapi_confidence": dapi_confidence if dapi_confidence is not None else [None] * n_cells,
            "consensus_score": consensus_score if consensus_score is not None else [None] * n_cells,
            "dapi_agrees": dapi_agreement if dapi_agreement is not None else [None] * n_cells,
        }
    )


def quick_validate(
    celltypist_labels: np.ndarray,
    leiden_labels: np.ndarray,
    dapi_predictions: np.ndarray | None = None,
) -> dict:
    """Quick validation without full AnnData/model setup.

    Args:
        celltypist_labels: CellTypist predictions
        leiden_labels: Leiden cluster assignments
        dapi_predictions: DAPI model predictions (optional)

    Returns:
        Dictionary with validation metrics
    """
    results = {
        "leiden_ari": adjusted_rand_score(celltypist_labels, leiden_labels),
        "leiden_nmi": normalized_mutual_info_score(celltypist_labels, leiden_labels),
    }

    if dapi_predictions is not None:
        results["dapi_agreement"] = (celltypist_labels == dapi_predictions).mean()
        results["dapi_celltypist_ari"] = adjusted_rand_score(celltypist_labels, dapi_predictions)

    return results


def extract_pretrained_features(
    patches: np.ndarray | torch.Tensor,
    backbone_name: str = "efficientnetv2_rw_s",
    device: str = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """Extract features using a PRETRAINED backbone (no CellTypist training).

    This provides TRULY INDEPENDENT morphology features for unsupervised
    clustering, avoiding the circularity of using a model trained on CellTypist.

    Args:
        patches: DAPI patches, shape (N, H, W) or (N, 1, H, W), values in [0, 1]
        backbone_name: Pretrained backbone to use. Options:
            - "efficientnetv2_rw_s" (ImageNet, default)
            - "phikon" (histopathology-pretrained ViT from Owkin)
            - Any timm model name
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        Feature embeddings of shape (N, num_features)
    """
    # Prepare patches
    if isinstance(patches, np.ndarray):
        patches = torch.from_numpy(patches)

    if patches.ndim == 3:
        patches = patches.unsqueeze(1)  # Add channel dim

    # Handle Phikon (histopathology-pretrained ViT)
    if backbone_name.lower() == "phikon":
        return _extract_phikon_features(patches, device, batch_size)

    # Standard timm model
    import timm

    logger.info(f"Loading pretrained backbone: {backbone_name} (ImageNet weights)")
    model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    model.to(device)
    model.train(False)

    # Normalize for ImageNet (replicate grayscale to RGB)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_features = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size].float().to(device)

            # Replicate single channel to 3 channels
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # Apply ImageNet normalization
            batch = (batch - imagenet_mean) / imagenet_std

            features = model(batch)
            all_features.append(features.cpu())

            if (i + batch_size) % 5000 < batch_size:
                logger.info(f"  Extracted features: {min(i + batch_size, len(patches))}/{len(patches)}")

    return torch.cat(all_features, dim=0).numpy()


def _extract_phikon_features(
    patches: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract features using Phikon (histopathology-pretrained ViT).

    Phikon was trained on 40M histopathology tiles and should be better
    suited for nuclear morphology than ImageNet-pretrained models.

    Args:
        patches: DAPI patches, shape (N, 1, H, W), values in [0, 1]
        device: Device for inference
        batch_size: Batch size (smaller due to ViT memory)

    Returns:
        Feature embeddings of shape (N, 768)
    """
    from transformers import AutoModel

    logger.info("Loading Phikon (histopathology-pretrained ViT from Owkin)")
    model = AutoModel.from_pretrained("owkin/phikon", trust_remote_code=True)
    model.to(device)
    model.train(False)

    # Phikon expects 224x224 RGB images
    # Normalize similar to ImageNet but may need adjustment
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_features = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size].float().to(device)

            # Replicate single channel to 3 channels
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)

            # Resize to 224x224 if needed (Phikon expects this)
            if batch.shape[-1] != 224:
                batch = torch.nn.functional.interpolate(
                    batch, size=(224, 224), mode="bilinear", align_corners=False
                )

            # Apply normalization
            batch = (batch - imagenet_mean) / imagenet_std

            # Get CLS token features
            outputs = model(batch)
            # Phikon returns last_hidden_state, take CLS token (first token)
            features = outputs.last_hidden_state[:, 0, :]
            all_features.append(features.cpu())

            if (i + batch_size) % 2000 < batch_size:
                logger.info(f"  Extracted Phikon features: {min(i + batch_size, len(patches))}/{len(patches)}")

    return torch.cat(all_features, dim=0).numpy()


def unsupervised_morphology_validation(
    patches: np.ndarray | torch.Tensor,
    celltypist_labels: np.ndarray,
    n_clusters: int | None = None,
    backbone_name: str = "efficientnetv2_rw_s",
    clustering_method: str = "kmeans",
    device: str = "cuda",
    batch_size: int = 128,
) -> dict:
    """Validate CellTypist using UNSUPERVISED morphology clustering.

    This is TRULY INDEPENDENT validation because:
    1. Features come from ImageNet-pretrained backbone (no CellTypist training)
    2. Clustering is unsupervised (no labels used)
    3. Only compare clusters to CellTypist AFTER clustering

    High ARI/NMI means CellTypist labels correspond to real morphological
    differences visible in DAPI images.

    Args:
        patches: DAPI patches, shape (N, H, W) or (N, 1, H, W)
        celltypist_labels: CellTypist predictions (string labels)
        n_clusters: Number of clusters (default: number of unique labels)
        backbone_name: Pretrained backbone for feature extraction
        clustering_method: "kmeans", "hdbscan", or "leiden"
        device: Device for inference
        batch_size: Batch size for feature extraction

    Returns:
        Dictionary with validation metrics including ARI, NMI, and cluster mapping
    """
    # Determine number of clusters from labels if not specified
    unique_labels = np.unique(celltypist_labels)
    if n_clusters is None:
        n_clusters = len(unique_labels)

    logger.info(f"Unsupervised morphology validation: {len(patches)} cells, {n_clusters} clusters")

    # Step 1: Extract features using PRETRAINED backbone (no CellTypist training)
    logger.info("Step 1: Extracting pretrained features (ImageNet backbone)...")
    features = extract_pretrained_features(
        patches, backbone_name=backbone_name, device=device, batch_size=batch_size
    )
    logger.info(f"  Feature shape: {features.shape}")

    # Step 2: Reduce dimensionality with PCA (helps clustering)
    logger.info("Step 2: PCA dimensionality reduction...")
    from sklearn.decomposition import PCA

    n_components = min(50, features.shape[1], features.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features)
    variance_captured = float(pca.explained_variance_ratio_.sum())
    logger.info(f"  Reduced to {n_components} components, variance captured: {variance_captured:.1%}")

    # Step 3: Cluster in morphology space
    logger.info(f"Step 3: Clustering with {clustering_method}...")
    cluster_labels = cluster_morphology_embeddings(
        features_pca, n_clusters=n_clusters, method=clustering_method
    )
    n_found_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
    logger.info(f"  Found {n_found_clusters} clusters")

    # Step 4: Compare clusters to CellTypist labels
    logger.info("Step 4: Computing agreement metrics...")

    # Filter out noise points (cluster = -1 for HDBSCAN)
    valid_mask = cluster_labels >= 0
    valid_clusters = cluster_labels[valid_mask]
    valid_ct_labels = celltypist_labels[valid_mask]

    ari = adjusted_rand_score(valid_ct_labels, valid_clusters)
    nmi = normalized_mutual_info_score(valid_ct_labels, valid_clusters)

    logger.info(f"  ARI (unsupervised): {ari:.3f}")
    logger.info(f"  NMI (unsupervised): {nmi:.3f}")

    # Compute cluster-to-label mapping (majority vote)
    cluster_to_label = {}

    for cluster_id in np.unique(valid_clusters):
        cluster_mask = valid_clusters == cluster_id
        cluster_ct_labels = valid_ct_labels[cluster_mask]

        # Count labels in this cluster
        unique, counts = np.unique(cluster_ct_labels, return_counts=True)

        # Majority label
        majority_label = unique[counts.argmax()]
        purity = counts.max() / counts.sum()
        cluster_to_label[int(cluster_id)] = {
            "majority_label": str(majority_label),
            "purity": float(purity),
            "size": int(cluster_mask.sum()),
        }

    # Overall purity (weighted by cluster size)
    total_correct = sum(
        info["purity"] * info["size"] for info in cluster_to_label.values()
    )
    overall_purity = total_correct / len(valid_clusters)

    results = {
        "ari": float(ari),
        "nmi": float(nmi),
        "overall_purity": float(overall_purity),
        "n_clusters_found": n_found_clusters,
        "n_clusters_requested": n_clusters,
        "n_cells_clustered": int(valid_mask.sum()),
        "n_cells_noise": int((~valid_mask).sum()),
        "cluster_to_label": cluster_to_label,
        "clustering_method": clustering_method,
        "backbone": backbone_name,
        "pca_variance_captured": variance_captured,
    }

    # Interpretation
    if ari >= 0.6:
        results["interpretation"] = "STRONG: CellTypist labels reflect clear morphological differences"
    elif ari >= 0.3:
        results["interpretation"] = "MODERATE: Some morphological basis for CellTypist labels"
    else:
        results["interpretation"] = "WEAK: CellTypist labels may not correlate with morphology"

    return results
