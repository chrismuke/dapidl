"""STHELAR Dataset Reader.

Reads SpatialData zarr objects from the STHELAR dataset
(Giraud-Sauveur et al., 2025) and extracts:
- DAPI morphology images
- Nucleus centroids (in pixel coordinates)
- Cell type labels at multiple hierarchy levels
- Confidence scores

STHELAR provides co-registered DAPI + H&E + spatial transcriptomics
for 31 Xenium FFPE sections across 16 tissue types.

The DAPI is at 0.2125 um/px (same as all Xenium data).
Coordinates in the SpatialData object are in microns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


# STHELAR group label mapping (from cells_final_label_group)
# WARNING: These group labels are noisy — Leiden refinement misassigns cells.
# Prefer TANGRAM_TO_COARSE for direct fine-grained mapping.
STHELAR_TO_DAPIDL_COARSE = {
    "Epithelial": "Epithelial",
    "Blood_vessel": "Endothelial",
    "Endothelial": "Endothelial",
    "Fibroblast_Myofibroblast": "Stromal",
    "Fibroblast": "Stromal",
    "CAF": "Stromal",
    "Myeloid": "Immune",
    "Monocyte/Macrophage": "Immune",
    "T_NK": "Immune",
    "T": "Immune",
    "B_Plasma": "Immune",
    "B": "Immune",
    "NK": "Immune",
    "Specialized": "Unknown",
    "Melanocyte": "Unknown",
    "Glioblastoma": "Unknown",
    "Adipocyte": "Unknown",
    "Mast": "Immune",
    "Other": "Unknown",
    "Less10": "Unknown",
    "less10": "Unknown",
    "Unknown": "Unknown",
}

# Biologically correct mapping from raw Tangram fine-grained labels
# These are more accurate than STHELAR's Leiden-refined group labels
TANGRAM_TO_COARSE = {
    # Epithelial (mammary)
    "CXCL14 mammary basal cell": "Epithelial",
    "SFN mammary luminal progenitor": "Epithelial",
    "KRT6B mammary basal cell": "Epithelial",
    "Secretoglobin mammary luminal progenitor": "Epithelial",
    "Cycling mammary luminal progenitor": "Epithelial",
    "CCSER1 mammary basal cell": "Epithelial",
    "SCGB3A1 mammary luminal progenitor": "Epithelial",
    "PIP mammary luminal cell": "Epithelial",
    "SAA2 mammary luminal progenitor": "Epithelial",
    "KRT17 mammary luminal cell": "Epithelial",
    "Secretoglobin mammary luminal cell": "Epithelial",
    "Lactocyte": "Epithelial",
    # Endothelial (true vascular endothelium only)
    "Venous EC": "Endothelial",
    "Capillary EC": "Endothelial",
    "Arterial EC": "Endothelial",
    "Lymphatic EC": "Endothelial",
    # Stromal (fibroblasts, pericytes, smooth muscle)
    "CXCL+ fibroblast": "Stromal",
    "IGFBP6+APOD+ fibroblast": "Stromal",
    "IGFBP6+SFRP4+ fibroblast": "Stromal",
    "CCL19/21 pericyte": "Stromal",
    "Pericyte": "Stromal",
    "CXCL+ pericyte": "Stromal",
    "CREB+MT1A+ vascular smooth muscle cell": "Stromal",
    "Vascular smooth muscle cell": "Stromal",
    # Immune
    "Monocyte": "Immune",
    "Dendritic cell": "Immune",
    "pDC": "Immune",
    "Macrophage": "Immune",
    "M1 macrophage": "Immune",
    "LYVE1 macrophage": "Immune",
    "CD4 T cell": "Immune",
    "GZMK CD8 T cell": "Immune",
    "GZMB CD8 T cell": "Immune",
    "NK cell": "Immune",
    "B cell": "Immune",
    "Plasma cell": "Immune",
    "Treg cell": "Immune",
    "ILC": "Immune",
    "Mast cell": "Immune",
}

XENIUM_PIXEL_SIZE_UM = 0.2125


@dataclass
class STHELARSlide:
    """Extracted data from one STHELAR slide."""

    name: str
    dapi: np.ndarray  # (H, W) uint16
    centroids_px: np.ndarray  # (N, 2) float64, (x_px, y_px)
    cell_ids: list[str]

    # Labels at multiple hierarchy levels
    label_coarse: list[str]  # 4 DAPIDL classes
    label_medium: list[str]  # STHELAR 7-10 categories (label1)
    label_fine: list[str]  # STHELAR detailed (label2)
    label_tangram: list[str]  # Original Tangram predictions

    # Confidence
    confidence: np.ndarray  # (N,) float32

    # Optional
    h_and_e: np.ndarray | None = None  # (3, H, W) uint8 if loaded

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.dapi.shape


def load_sthelar_slide(
    zarr_path: str | Path,
    metadata_path: str | Path | None = None,
    load_he: bool = False,
    dapi_level: int = 0,
) -> STHELARSlide:
    """Load a STHELAR slide from SpatialData zarr.

    Args:
        zarr_path: Path to extracted zarr directory (e.g., sdata_breast_s3.zarr)
        metadata_path: Path to cell_metadata.parquet (optional, for extra annotations)
        load_he: Whether to also load the H&E image
        dapi_level: Pyramid level for DAPI (0=full res, 1=half, etc.)

    Returns:
        STHELARSlide with extracted DAPI, centroids, and labels
    """
    import spatialdata as sd

    zarr_path = Path(zarr_path)
    slide_name = zarr_path.stem  # e.g., "sdata_breast_s3"

    logger.info(f"Loading STHELAR slide: {slide_name}")
    sdata = sd.read_zarr(str(zarr_path))

    # Extract DAPI image
    morpho = sdata.images["morpho"]
    scale_key = f"scale{dapi_level}"
    dapi_da = morpho[scale_key].ds["image"]
    logger.info(f"DAPI: {dapi_da.shape}, dtype={dapi_da.dtype}, level={dapi_level}")

    # Load full DAPI into memory (channel 0)
    dapi = dapi_da[0].values  # (H, W) uint16
    img_h, img_w = dapi.shape
    logger.info(f"DAPI loaded: {img_h}x{img_w}, range=[{dapi.min()}, {dapi.max()}]")

    # Extract nucleus centroids from boundaries
    nuc_boundaries = sdata.shapes["nucleus_boundaries"]
    centroids_um = np.column_stack([
        nuc_boundaries.geometry.centroid.x.values,
        nuc_boundaries.geometry.centroid.y.values,
    ])  # (N, 2) in microns

    # Convert microns to pixels
    scale_factor = 2 ** dapi_level  # Account for pyramid level
    centroids_px = centroids_um / (XENIUM_PIXEL_SIZE_UM * scale_factor)
    logger.info(
        f"Centroids: {len(centroids_px)} nuclei, "
        f"x=[{centroids_px[:,0].min():.0f}, {centroids_px[:,0].max():.0f}], "
        f"y=[{centroids_px[:,1].min():.0f}, {centroids_px[:,1].max():.0f}]"
    )

    cell_ids = nuc_boundaries.index.tolist()

    # Extract labels from table_cells
    cells_table = sdata.tables["table_cells"]
    obs = cells_table.obs

    # label1 = coarse/medium (Epithelial, Endothelial, Fibroblast, T, B, etc.)
    label_medium = obs["label1"].tolist() if "label1" in obs.columns else ["Unknown"] * len(obs)
    # label2 = fine (Mammary_luminal_cell, CAF, Angiogenic_endothelial_cell, etc.)
    label_fine = obs["label2"].tolist() if "label2" in obs.columns else ["Unknown"] * len(obs)

    # Get original Tangram predictions + confidence from metadata
    label_tangram = ["Unknown"] * len(obs)
    confidence = np.ones(len(obs), dtype=np.float32)

    if metadata_path:
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            meta = pl.read_parquet(metadata_path)
            logger.info(f"Loaded metadata: {meta.shape}")

            meta_dict = {
                row["cell_id"]: row
                for row in meta.iter_rows(named=True)
            }

            tangram_labels = []
            conf_vals = []
            for cid in cell_ids:
                row = meta_dict.get(cid, {})
                tangram_labels.append(row.get("nuclei_ct_tangram", "Unknown"))
                conf = row.get("cells_final_label_confidence")
                conf_vals.append(conf if conf is not None else 1.0)

            label_tangram = tangram_labels
            confidence = np.array(conf_vals, dtype=np.float32)

    # Map to DAPIDL coarse: prefer Tangram-direct (biologically correct)
    # over STHELAR group labels (Leiden-refined, noisy)
    from collections import Counter
    label_coarse = [TANGRAM_TO_COARSE.get(lab, "Unknown") for lab in label_tangram]
    # Fallback to group labels for cells without Tangram predictions
    for i, lab in enumerate(label_coarse):
        if lab == "Unknown" and label_medium[i] != "Unknown":
            label_coarse[i] = STHELAR_TO_DAPIDL_COARSE.get(label_medium[i], "Unknown")

    coarse_dist = Counter(label_coarse)
    logger.info(f"Coarse labels (Tangram-direct): {dict(sorted(coarse_dist.items(), key=lambda x: -x[1]))}")

    medium_dist = Counter(label_medium)
    logger.info(f"Medium labels ({len(medium_dist)} types): {dict(sorted(medium_dist.items(), key=lambda x: -x[1])[:8])}")

    # Load H&E if requested
    h_and_e = None
    if load_he:
        he_da = sdata.images["he"][f"scale{dapi_level}"].ds["image"]
        h_and_e = he_da.values  # (3, H, W) uint8
        logger.info(f"H&E loaded: {h_and_e.shape}")

    return STHELARSlide(
        name=slide_name,
        dapi=dapi,
        centroids_px=centroids_px,
        cell_ids=cell_ids,
        label_coarse=label_coarse,
        label_medium=label_medium,
        label_fine=label_fine,
        label_tangram=label_tangram,
        confidence=confidence,
        h_and_e=h_and_e,
    )
