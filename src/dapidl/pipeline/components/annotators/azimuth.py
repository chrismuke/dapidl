"""Azimuth-based cell type annotation component.

Azimuth is a Seurat-based reference mapping tool that uses anchor-based
integration to transfer labels from reference to query data.

Reference: Hao et al. Cell 2021
https://github.com/satijalab/azimuth
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pandas as pd
import numpy as np
from loguru import logger

from dapidl.pipeline.base import AnnotationConfig, AnnotationResult
from dapidl.pipeline.components.annotators.mapping import (
    get_class_names,
    map_to_broad_category,
)
from dapidl.pipeline.registry import register_annotator


# Azimuth references available
AZIMUTH_REFERENCES = {
    "pbmc": "pbmcref",
    "adipose": "adiposeref",
    "bonemarrow": "bonemarrowref",
    "fetus": "fetusref",
    "heart": "heartref",
    "kidney": "kidneyref",
    "lung": "lungref",
    "motor_cortex": "motorcortexref",
    "pancreas": "pancreasref",
    "tonsil": "tonsilref",
}

# Label mapping to broad categories
AZIMUTH_TO_BROAD = {
    # Common PBMC types
    "CD4 T": "Immune",
    "CD8 T": "Immune",
    "B cell": "Immune",
    "NK": "Immune",
    "Mono": "Immune",
    "DC": "Immune",
    "Platelet": "Immune",
    # Epithelial
    "Epithelial": "Epithelial",
    "AT1": "Epithelial",
    "AT2": "Epithelial",
    "Basal": "Epithelial",
    "Club": "Epithelial",
    # Stromal
    "Fibroblast": "Stromal",
    "Smooth Muscle": "Stromal",
    "Pericyte": "Stromal",
    "Adipocyte": "Stromal",
    # Endothelial
    "Endothelial": "Endothelial",
    "Capillary": "Endothelial",
    "Arterial": "Endothelial",
    "Venous": "Endothelial",
}


def map_to_broad(label: str) -> str:
    """Map Azimuth prediction to broad category."""
    if label in AZIMUTH_TO_BROAD:
        return AZIMUTH_TO_BROAD[label]

    label_lower = label.lower()
    if any(x in label_lower for x in ["t cell", "cd4", "cd8", "nk", "b cell", "mono", "dc", "macro", "dendritic", "mast", "neutro"]):
        return "Immune"
    if any(x in label_lower for x in ["epithelial", "at1", "at2", "basal", "club", "goblet", "ciliated"]):
        return "Epithelial"
    if any(x in label_lower for x in ["fibroblast", "smooth muscle", "pericyte", "adipocyte", "stromal", "mesenchymal"]):
        return "Stromal"
    if any(x in label_lower for x in ["endothelial", "capillary", "arterial", "venous", "vascular"]):
        return "Endothelial"

    return map_to_broad_category(label)


@register_annotator
class AzimuthAnnotator:
    """Cell type annotation using Azimuth reference mapping.

    Azimuth uses Seurat's anchor-based integration to map query cells
    to reference atlases and transfer cell type labels.

    Requires R with Seurat and SeuratData packages installed.
    """

    name = "azimuth"

    def __init__(
        self,
        config: AnnotationConfig | None = None,
        reference: str = "pbmc",
        annotation_level: str = "celltype.l2",
    ):
        """Initialize the Azimuth annotator.

        Args:
            config: Annotation configuration
            reference: Name of reference atlas (pbmc, lung, etc.)
            annotation_level: Level of annotation detail (l1, l2, l3)
        """
        self.config = config or AnnotationConfig()
        self.reference = reference
        self.annotation_level = annotation_level

    def annotate(
        self,
        config: AnnotationConfig | None = None,
        adata: Any | None = None,
        expression_path: Path | None = None,
        cells_df: pl.DataFrame | None = None,
    ) -> AnnotationResult:
        """Annotate cells using Azimuth reference mapping.

        Args:
            config: Override config for this call
            adata: AnnData object with expression data (required)
            expression_path: Path to expression matrix (h5, h5ad)
            cells_df: Cell metadata DataFrame (not used)

        Returns:
            Annotation results with cell types and confidence
        """
        cfg = config or self.config

        # Need either adata or expression_path
        if adata is None and expression_path is None:
            raise ValueError("AzimuthAnnotator requires either adata or expression_path")

        # Load adata if needed
        if adata is None and expression_path is not None:
            adata = self._load_expression(expression_path)

        logger.info(f"Running Azimuth with {self.reference} reference...")
        logger.info(f"  Query: {adata.n_obs} cells x {adata.n_vars} genes")

        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save expression matrix for R (genes x cells)
            expr_df = pd.DataFrame(
                adata.X.toarray().T if hasattr(adata.X, 'toarray') else adata.X.T,
                index=adata.var_names,
                columns=adata.obs_names
            )
            expr_path = temp_path / "expr_matrix.csv"
            expr_df.to_csv(expr_path)
            logger.info(f"Saved {expr_df.shape[0]} genes x {expr_df.shape[1]} cells")

            # Create R script
            results_path = temp_path / "azimuth_results.csv"
            r_script = self._generate_r_script(expr_path, results_path)

            script_path = temp_path / "run_azimuth.R"
            with open(script_path, 'w') as f:
                f.write(r_script)

            # Run R script
            logger.info("Running Azimuth R script...")
            result = subprocess.run(
                ["Rscript", str(script_path)],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
                cwd=str(temp_path)
            )

            if result.returncode != 0:
                logger.error(f"Azimuth failed: {result.stderr}")
                # Return empty result on failure
                return self._empty_result(cfg)

            # Read results
            if not results_path.exists():
                logger.error("Azimuth output file not found")
                return self._empty_result(cfg)

            results_df = pd.read_csv(results_path)

        # Process results
        annotations_data = []
        for _, row in results_df.iterrows():
            pred = row.get("predicted_type", "Unknown")
            conf = row.get("confidence", 0.5)
            broad = map_to_broad(pred)

            annotations_data.append({
                "cell_id": str(row["cell_id"]),
                "predicted_type": pred,
                "broad_category": broad,
                "confidence": float(conf),
            })

        annotations_df = pl.DataFrame(annotations_data)

        # Filter out Unknown if configured
        include_unknown = getattr(cfg, "include_unknown", True)
        if not include_unknown:
            annotations_df = annotations_df.filter(pl.col("broad_category") != "Unknown")

        # Build class mapping
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        # Stats
        n_annotated = annotations_df.height
        class_dist = (
            annotations_df.group_by("broad_category")
            .count()
            .to_pandas()
            .set_index("broad_category")["count"]
            .to_dict()
        )

        logger.info(f"Azimuth annotation complete: {n_annotated} cells annotated")
        logger.info(f"  Class distribution: {class_dist}")

        return AnnotationResult(
            annotations_df=annotations_df,
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={
                "n_annotated": n_annotated,
                "class_distribution": class_dist,
                "reference": self.reference,
            },
        )

    def _generate_r_script(self, expr_path: Path, results_path: Path) -> str:
        """Generate the R script for running Azimuth."""
        ref_name = AZIMUTH_REFERENCES.get(self.reference, self.reference)

        return f'''
library(Seurat)
library(SeuratData)
library(Azimuth)

# Load expression matrix
cat("Loading expression data...\\n")
expr <- as.matrix(read.csv("{expr_path}", row.names=1, check.names=FALSE))
cat("Dimensions:", dim(expr), "\\n")

# Create Seurat object
seurat_obj <- CreateSeuratObject(counts = expr, project = "query")

# Normalize
seurat_obj <- NormalizeData(seurat_obj)
seurat_obj <- FindVariableFeatures(seurat_obj)
seurat_obj <- ScaleData(seurat_obj)
seurat_obj <- RunPCA(seurat_obj)

# Try to run Azimuth
tryCatch({{
    # Install reference if needed
    if (!"{ref_name}" %in% InstalledData()) {{
        InstallData("{ref_name}")
    }}

    # Run Azimuth annotation
    seurat_obj <- RunAzimuth(seurat_obj, reference = "{ref_name}")

    # Extract predictions
    predictions <- seurat_obj@meta.data$predicted.{self.annotation_level}
    scores <- seurat_obj@meta.data$predicted.{self.annotation_level}.score

    # Handle missing predictions
    if (is.null(predictions)) {{
        predictions <- rep("Unknown", ncol(seurat_obj))
        scores <- rep(0.5, ncol(seurat_obj))
    }}

    # Save results
    results <- data.frame(
        cell_id = colnames(seurat_obj),
        predicted_type = predictions,
        confidence = scores
    )
    write.csv(results, "{results_path}", row.names = FALSE)

    cat("Successfully annotated", nrow(results), "cells\\n")

}}, error = function(e) {{
    cat("Azimuth error:", conditionMessage(e), "\\n")

    # Fall back to simple label transfer
    cat("Attempting fallback with SingleR...\\n")

    library(SingleR)
    library(celldex)

    # Use BlueprintEncode as fallback reference
    ref <- BlueprintEncodeData()
    common <- intersect(rownames(expr), rownames(ref))

    if (length(common) >= 50) {{
        results_sr <- SingleR(test = expr[common,], ref = ref[common,], labels = ref$label.main)

        results <- data.frame(
            cell_id = colnames(expr),
            predicted_type = results_sr$labels,
            confidence = ifelse(is.na(results_sr$pruned.labels), 0.3, 0.7)
        )
        write.csv(results, "{results_path}", row.names = FALSE)
        cat("Fallback annotated", nrow(results), "cells\\n")
    }} else {{
        # Complete failure
        results <- data.frame(
            cell_id = colnames(expr),
            predicted_type = rep("Unknown", ncol(expr)),
            confidence = rep(0.0, ncol(expr))
        )
        write.csv(results, "{results_path}", row.names = FALSE)
        cat("Annotation failed, returning Unknown\\n")
    }}
}})
'''

    def _empty_result(self, cfg: AnnotationConfig) -> AnnotationResult:
        """Return empty result on failure."""
        class_names = get_class_names(cfg.fine_grained)
        class_mapping = {name: i for i, name in enumerate(class_names)}
        index_to_class = {i: name for i, name in enumerate(class_names)}

        return AnnotationResult(
            annotations_df=pl.DataFrame(),
            class_mapping=class_mapping,
            index_to_class=index_to_class,
            stats={"n_annotated": 0, "error": "Azimuth failed"},
        )

    def _load_expression(self, path: Path) -> Any:
        """Load expression data from file."""
        import anndata as ad

        path = Path(path)

        if path.suffix in [".h5", ".h5ad"]:
            return ad.read_h5ad(path)
        elif path.suffix == ".zarr" or path.is_dir():
            return ad.read_zarr(path)
        else:
            raise ValueError(f"Unsupported expression format: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azimuth cell type annotation")
    parser.add_argument("data_path", type=Path, help="Path to expression data")
    parser.add_argument("--reference", "-r", default="pbmc", help="Azimuth reference")
    parser.add_argument("--output", "-o", type=Path, help="Output parquet path")

    args = parser.parse_args()

    annotator = AzimuthAnnotator(reference=args.reference)

    import anndata as ad
    adata = ad.read_h5ad(args.data_path) if args.data_path.suffix == ".h5ad" else None

    if adata is None:
        # Try loading as Xenium
        import h5py
        from scipy.sparse import csc_matrix

        with h5py.File(args.data_path, 'r') as f:
            data = f['matrix/data'][:]
            indices = f['matrix/indices'][:]
            indptr = f['matrix/indptr'][:]
            shape = f['matrix/shape'][:]
            barcodes = [b.decode() for b in f['matrix/barcodes'][:]]
            genes = [g.decode() for g in f['matrix/features/name'][:]]

        X = csc_matrix((data, indices, indptr), shape=shape).T
        adata = ad.AnnData(X=X)
        adata.obs_names = barcodes
        adata.var_names = genes

    result = annotator.annotate(adata=adata)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.annotations_df.write_parquet(args.output)

    print(result.annotations_df.head(10))
