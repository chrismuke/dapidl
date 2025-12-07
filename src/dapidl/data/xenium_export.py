"""Export annotated Xenium datasets with hardlinks for space efficiency."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger


def create_hardlink_dataset(
    source_dir: Path,
    output_dir: Path,
    modified_files: dict[str, Path],
    copy_if_hardlink_fails: bool = True,
) -> Path:
    """Create a new Xenium dataset directory using hardlinks for unchanged files.

    This function creates a space-efficient copy of a Xenium dataset by using
    hardlinks for files that haven't changed (large image files, transcripts, etc.)
    and only copying/replacing the modified files (cells.parquet, analysis.zarr.zip).

    Args:
        source_dir: Path to original Xenium output directory
        output_dir: Path for the new annotated dataset
        modified_files: Dict mapping relative paths to new file paths to copy
        copy_if_hardlink_fails: If True, copy files when hardlink fails (e.g., cross-device)

    Returns:
        Path to the created output directory

    Note:
        Hardlinks share the same inode as the original file, so they:
        - Take zero additional disk space
        - Remain valid even if the original is moved (within the same filesystem)
        - Are deleted independently from the original
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        logger.warning(f"Output directory exists, will be removed: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of modified files (normalized paths)
    modified_set = {str(Path(p)) for p in modified_files.keys()}

    # Walk source directory and create structure
    files_linked = 0
    files_copied = 0
    files_replaced = 0
    total_size_saved = 0

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(source_dir)

        # Create subdirectories
        for d in dirs:
            (output_dir / rel_root / d).mkdir(parents=True, exist_ok=True)

        # Process files
        for f in files:
            src_file = root_path / f
            rel_path = rel_root / f
            dst_file = output_dir / rel_path

            # Check if this file should be replaced
            if str(rel_path) in modified_set:
                # Copy the modified version
                new_file = modified_files[str(rel_path)]
                shutil.copy2(new_file, dst_file)
                files_replaced += 1
                logger.debug(f"Replaced: {rel_path}")
            else:
                # Try to create hardlink
                try:
                    os.link(src_file, dst_file)
                    files_linked += 1
                    total_size_saved += src_file.stat().st_size
                except OSError as e:
                    if copy_if_hardlink_fails:
                        shutil.copy2(src_file, dst_file)
                        files_copied += 1
                        logger.debug(f"Copied (hardlink failed): {rel_path}")
                    else:
                        raise OSError(f"Cannot create hardlink for {rel_path}: {e}")

    logger.info(f"Created dataset at {output_dir}")
    logger.info(f"  Files hardlinked: {files_linked} (saved {total_size_saved / 1e9:.2f} GB)")
    logger.info(f"  Files replaced: {files_replaced}")
    if files_copied > 0:
        logger.info(f"  Files copied (fallback): {files_copied}")

    return output_dir


def export_cell_groups_csv(
    annotations_df: pl.DataFrame,
    model_name: str,
    output_path: Path,
    celltype_column: str = "predicted_type",
    add_colors: bool = True,
) -> Path:
    """Export CellTypist annotations to CSV format for Xenium Explorer.

    Creates a CSV file with the format required by Xenium Explorer:
    - Column 1: cell_id
    - Column 2: group (cell type annotation)
    - Column 3 (optional): color (HEX color code)

    Args:
        annotations_df: DataFrame with cell_id and cell type columns
        model_name: Name of the model (used for filename)
        output_path: Directory where CSV file will be written
        celltype_column: Name of the column containing cell types
        add_colors: Whether to add auto-generated color column

    Returns:
        Path to the created CSV file
    """
    import colorsys

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Slugify model name for filename
    model_slug = model_name.replace(".pkl", "").replace("_", "_").lower()

    # Create output dataframe
    csv_df = annotations_df.select([
        pl.col("cell_id").cast(pl.Utf8),
        pl.col(celltype_column).alias("group"),
    ]).drop_nulls()

    if add_colors:
        # Get unique cell types and generate colors
        unique_types = csv_df["group"].unique().sort().to_list()
        n_types = len(unique_types)

        # Generate visually distinct colors using golden angle
        colors = []
        golden_angle = 137.508
        for i in range(n_types):
            hue = (i * golden_angle) % 360
            saturation = 0.7 + (i % 3) * 0.125
            value = 0.8 + (i % 2) * 0.15
            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            hex_color = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
            colors.append(hex_color)

        # Create color mapping
        color_map = dict(zip(unique_types, colors))

        # Add color column
        csv_df = csv_df.with_columns(
            pl.col("group").replace(color_map).alias("color")
        )

    # Write CSV
    csv_path = output_path / f"celltypist_{model_slug}.csv"
    csv_df.write_csv(csv_path)

    logger.info(f"Exported cell groups CSV: {csv_path}")
    return csv_path


def update_cells_parquet(
    original_parquet: Path,
    annotations_df: pl.DataFrame,
    output_path: Path,
    create_backup: bool = True,
) -> Path:
    """Update cells.parquet with CellTypist annotations.

    Args:
        original_parquet: Path to original cells.parquet
        annotations_df: DataFrame with cell_id and annotation columns
        output_path: Path for output parquet file
        create_backup: Whether to create backup of original

    Returns:
        Path to the updated parquet file
    """
    output_path = Path(output_path)

    # Load original cells
    cells_df = pl.read_parquet(original_parquet)

    # Ensure cell_id types match
    cells_df = cells_df.with_columns(pl.col("cell_id").cast(pl.Utf8))
    annotations_df = annotations_df.with_columns(pl.col("cell_id").cast(pl.Utf8))

    # Join annotations
    updated = cells_df.join(annotations_df, on="cell_id", how="left")

    # Create backup if requested
    if create_backup and output_path.exists():
        backup_path = output_path.with_name(
            f"{output_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_path.suffix}"
        )
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    # Write updated parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.write_parquet(output_path, compression="zstd")

    logger.info(f"Updated cells parquet: {output_path}")
    return output_path


def create_annotated_xenium_dataset(
    xenium_dir: Path,
    output_dir: Path,
    annotations_df: pl.DataFrame,
    model_names: list[str],
    output_format: str = "csv",
    add_colors: bool = True,
) -> Path:
    """Create an annotated Xenium dataset for viewing in Xenium Explorer.

    This creates a space-efficient copy of the Xenium dataset using hardlinks
    for unchanged files, and adds CellTypist annotations in the specified format.

    Args:
        xenium_dir: Path to original Xenium output directory (containing cells.parquet, etc.)
        output_dir: Path for the annotated dataset
        annotations_df: DataFrame with CellTypist annotations
        model_names: List of model names used for annotation
        output_format: Output format - 'csv' (recommended), 'parquet', or 'both'
        add_colors: Whether to add auto-generated colors (CSV format only)

    Returns:
        Path to the created annotated dataset
    """
    xenium_dir = Path(xenium_dir)
    output_dir = Path(output_dir)

    # Find the outs directory if needed
    if (xenium_dir / "outs").exists():
        xenium_outs = xenium_dir / "outs"
    elif (xenium_dir / "cells.parquet").exists():
        xenium_outs = xenium_dir
    else:
        raise FileNotFoundError(f"Cannot find Xenium output files in {xenium_dir}")

    modified_files: dict[str, Path] = {}

    # Create temp directory for modified files
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Generate outputs based on format
        if output_format in ("parquet", "both"):
            # Create modified cells.parquet
            temp_cells = temp_path / "cells.parquet"
            update_cells_parquet(
                xenium_outs / "cells.parquet",
                annotations_df,
                temp_cells,
                create_backup=False,
            )
            modified_files["cells.parquet"] = temp_cells

        # Create hardlinked dataset
        if (xenium_dir / "outs").exists():
            output_outs = output_dir / "outs"
        else:
            output_outs = output_dir

        create_hardlink_dataset(
            source_dir=xenium_outs,
            output_dir=output_outs,
            modified_files=modified_files,
        )

        # Generate CSV files (after creating dataset structure)
        if output_format in ("csv", "both"):
            for i, model_name in enumerate(model_names):
                suffix = "" if len(model_names) == 1 else f"_{i + 1}"
                celltype_col = f"predicted_type{suffix}"

                if celltype_col in annotations_df.columns:
                    export_cell_groups_csv(
                        annotations_df=annotations_df,
                        model_name=model_name,
                        output_path=output_outs,
                        celltype_column=celltype_col,
                        add_colors=add_colors,
                    )

    logger.info(f"Created annotated Xenium dataset: {output_dir}")
    return output_dir
