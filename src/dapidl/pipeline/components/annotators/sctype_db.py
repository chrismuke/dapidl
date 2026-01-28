"""ScTypeDB Marker Database Loader.

Loads tissue-specific marker gene sets from the ScTypeDB database
(Ianevski et al., Nature Communications 2022).

Database URL: https://github.com/IanevskiAleksandr/sc-type

Available Tissues:
    - Immune system (34 cell types)
    - Brain (24 cell types)
    - Hippocampus (27 cell types)
    - Eye (24 cell types)
    - Kidney (20 cell types)
    - Lung (16 cell types)
    - Heart (16 cell types)
    - Stomach (16 cell types)
    - Pancreas (15 cell types)
    - Liver (13 cell types)
    - Intestine (12 cell types)
    - Adrenal (12 cell types)
    - Placenta (12 cell types)
    - Muscle (11 cell types)
    - Spleen (9 cell types)
    - Thymus (5 cell types)

Usage:
    from dapidl.pipeline.components.annotators.sctype_db import (
        load_sctype_db,
        get_tissue_markers,
        AVAILABLE_TISSUES,
    )

    # Get markers for a specific tissue
    markers = get_tissue_markers("Liver")

    # Get markers for multiple tissues
    markers = get_tissue_markers(["Immune system", "Liver"])

    # Get all available tissues
    print(AVAILABLE_TISSUES)
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import polars as pl
from loguru import logger


# Path to bundled ScTypeDB
SCTYPE_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "resources" / "ScTypeDB_full.xlsx"

# Available tissues in ScTypeDB
AVAILABLE_TISSUES = [
    "Immune system",
    "Pancreas",
    "Liver",
    "Eye",
    "Kidney",
    "Brain",
    "Lung",
    "Adrenal",
    "Heart",
    "Intestine",
    "Muscle",
    "Placenta",
    "Spleen",
    "Stomach",
    "Thymus",
    "Hippocampus",
]

# Tissue aliases for common naming variations
TISSUE_ALIASES = {
    # Common variations
    "breast": "Immune system",  # No breast-specific, use immune
    "mammary": "Immune system",
    "blood": "Immune system",
    "pbmc": "Immune system",
    "immune": "Immune system",
    "colon": "Intestine",
    "gut": "Intestine",
    "small intestine": "Intestine",
    "large intestine": "Intestine",
    "cardiac": "Heart",
    "hepatic": "Liver",
    "renal": "Kidney",
    "pulmonary": "Lung",
    "neural": "Brain",
    "cns": "Brain",
    "retina": "Eye",
    "cornea": "Eye",
    "gastric": "Stomach",
    "skeletal muscle": "Muscle",
    "adrenal gland": "Adrenal",
    # Exact matches (lowercase)
    "immune system": "Immune system",
    "pancreas": "Pancreas",
    "liver": "Liver",
    "eye": "Eye",
    "kidney": "Kidney",
    "brain": "Brain",
    "lung": "Lung",
    "adrenal": "Adrenal",
    "heart": "Heart",
    "intestine": "Intestine",
    "muscle": "Muscle",
    "placenta": "Placenta",
    "spleen": "Spleen",
    "stomach": "Stomach",
    "thymus": "Thymus",
    "hippocampus": "Hippocampus",
}


@lru_cache(maxsize=1)
def load_sctype_db(db_path: Path | None = None) -> pl.DataFrame:
    """Load the ScTypeDB database.

    Args:
        db_path: Path to ScTypeDB Excel file. Defaults to bundled database.

    Returns:
        Polars DataFrame with columns:
        - tissueType: Tissue name
        - cellName: Cell type name
        - geneSymbolmore1: Positive markers (comma-separated)
        - geneSymbolmore2: Negative markers (comma-separated)
        - shortName: Abbreviated cell type name
    """
    db_path = db_path or SCTYPE_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(
            f"ScTypeDB not found at {db_path}. "
            "Download from: https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"
        )

    # Read Excel with pandas then convert to polars (polars Excel support is limited)
    import pandas as pd
    df_pd = pd.read_excel(db_path)
    df = pl.from_pandas(df_pd)

    logger.debug(f"Loaded ScTypeDB: {len(df)} entries, {df['tissueType'].n_unique()} tissues")

    return df


def resolve_tissue(tissue: str) -> str:
    """Resolve tissue name to ScTypeDB tissue name.

    Args:
        tissue: User-provided tissue name (case-insensitive)

    Returns:
        Canonical ScTypeDB tissue name

    Raises:
        ValueError: If tissue not found in database or aliases
    """
    tissue_lower = tissue.lower().strip()

    # Check aliases
    if tissue_lower in TISSUE_ALIASES:
        return TISSUE_ALIASES[tissue_lower]

    # Check exact match (case-insensitive)
    for available in AVAILABLE_TISSUES:
        if available.lower() == tissue_lower:
            return available

    # Not found
    raise ValueError(
        f"Unknown tissue: '{tissue}'. "
        f"Available tissues: {AVAILABLE_TISSUES}"
    )


def get_tissue_markers(
    tissue: str | list[str],
    include_immune: bool = True,
) -> dict[str, dict[str, list[str]]]:
    """Get marker gene sets for a tissue.

    Args:
        tissue: Tissue name or list of tissues. Case-insensitive.
        include_immune: Always include immune cell markers (default True).
                       Useful since immune cells are present in most tissues.

    Returns:
        Dictionary mapping cell type names to marker sets:
        {
            "Hepatocyte": {
                "positive": ["ALB", "HNF4A", ...],
                "negative": ["PTPRC", ...]
            },
            ...
        }
    """
    db = load_sctype_db()

    # Normalize tissue input to list
    if isinstance(tissue, str):
        tissues = [tissue]
    else:
        tissues = list(tissue)

    # Resolve tissue names
    resolved_tissues = set()
    for t in tissues:
        try:
            resolved_tissues.add(resolve_tissue(t))
        except ValueError as e:
            logger.warning(str(e))

    # Always include immune system if requested
    if include_immune and "Immune system" not in resolved_tissues:
        resolved_tissues.add("Immune system")
        logger.debug("Added 'Immune system' markers (present in most tissues)")

    if not resolved_tissues:
        raise ValueError(f"No valid tissues found in: {tissues}")

    # Filter database to selected tissues
    df_filtered = db.filter(pl.col("tissueType").is_in(list(resolved_tissues)))

    # Build marker dictionary
    markers = {}
    for row in df_filtered.iter_rows(named=True):
        cell_name = row["cellName"]

        # Parse positive markers
        pos_str = row["geneSymbolmore1"]
        if pos_str and isinstance(pos_str, str):
            positive = [g.strip().upper() for g in pos_str.split(",") if g.strip()]
        else:
            positive = []

        # Parse negative markers
        neg_str = row["geneSymbolmore2"]
        if neg_str and isinstance(neg_str, str):
            negative = [g.strip().upper() for g in neg_str.split(",") if g.strip()]
        else:
            negative = []

        # Skip if no positive markers
        if not positive:
            continue

        # Handle duplicate cell names across tissues
        if cell_name in markers:
            # Merge markers (union)
            markers[cell_name]["positive"] = list(set(markers[cell_name]["positive"] + positive))
            markers[cell_name]["negative"] = list(set(markers[cell_name]["negative"] + negative))
        else:
            markers[cell_name] = {
                "positive": positive,
                "negative": negative,
            }

    logger.info(
        f"Loaded {len(markers)} cell types from ScTypeDB for tissues: {list(resolved_tissues)}"
    )

    return markers


def get_all_tissues() -> list[str]:
    """Get list of all available tissues in ScTypeDB."""
    return AVAILABLE_TISSUES.copy()


def get_tissue_cell_types(tissue: str) -> list[str]:
    """Get list of cell types available for a tissue."""
    db = load_sctype_db()
    resolved = resolve_tissue(tissue)
    df_filtered = db.filter(pl.col("tissueType") == resolved)
    return df_filtered["cellName"].unique().to_list()


def print_database_summary():
    """Print summary of ScTypeDB contents."""
    db = load_sctype_db()

    print("=" * 60)
    print("ScTypeDB Summary")
    print("=" * 60)
    print(f"Total entries: {len(db)}")
    print(f"Total tissues: {db['tissueType'].n_unique()}")
    print()

    for tissue in AVAILABLE_TISSUES:
        df_tissue = db.filter(pl.col("tissueType") == tissue)
        n_types = df_tissue["cellName"].n_unique()
        print(f"  {tissue}: {n_types} cell types")

    print()
    print("Tissue aliases available:")
    for alias, canonical in sorted(TISSUE_ALIASES.items()):
        if alias != canonical.lower():
            print(f"  '{alias}' → '{canonical}'")


if __name__ == "__main__":
    print_database_summary()

    print()
    print("=" * 60)
    print("Example: Liver markers")
    print("=" * 60)

    markers = get_tissue_markers("Liver", include_immune=False)
    for cell_type, marker_info in list(markers.items())[:5]:
        print(f"\n{cell_type}:")
        print(f"  Positive: {marker_info['positive'][:5]}...")
        print(f"  Negative: {marker_info['negative'][:3] if marker_info['negative'] else 'None'}")
