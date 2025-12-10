"""DAPIDL Command Line Interface."""

import sys

import click
from pathlib import Path
from typing import Tuple
from rich.console import Console
from rich.table import Table

console = Console()


def _capture_cli_command(ctx: click.Context) -> None:
    """Capture the full CLI command for reproducibility logging."""
    from dapidl.tracking.reproducibility import set_cli_command

    # Reconstruct the command from sys.argv
    # Format: dapidl <subcommand> <args>
    command = " ".join(sys.argv)
    set_cli_command(command)


@click.group()
@click.version_option(version="0.1.0")
@click.pass_context
def main(ctx: click.Context) -> None:
    """DAPIDL: Deep learning for cell type prediction from DAPI nuclear staining."""
    # Capture CLI command for reproducibility
    _capture_cli_command(ctx)


@main.command(name="list-backbones")
def list_backbones_cmd() -> None:
    """List available CNN backbone architectures.

    Shows all backbone options for training, including pretrained ImageNet models
    and custom microscopy-optimized architectures.
    """
    from dapidl.models.backbone import BACKBONE_PRESETS

    console.print("[bold blue]Available Backbone Architectures[/bold blue]\n")

    table = Table(title="Backbone Options")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Channels", style="yellow")

    for name, info in BACKBONE_PRESETS.items():
        channels = "1 (native)" if info["native_channels"] == 1 else "3 (adapted)"
        table.add_row(name, info["description"], channels)

    console.print(table)
    console.print("\n[dim]Use --backbone <name> with train or pipeline commands[/dim]")
    console.print("[dim]Custom microscopy models don't need channel adaptation[/dim]")


@main.command(name="list-models")
@click.option(
    "--downloaded-only",
    is_flag=True,
    default=False,
    help="Only show locally downloaded models",
)
@click.option(
    "--update",
    is_flag=True,
    default=False,
    help="Fetch latest model list from server",
)
def list_models(downloaded_only: bool, update: bool) -> None:
    """List available CellTypist models.

    Shows all available models from the CellTypist repository.
    Use --downloaded-only to show only locally cached models.
    """
    from dapidl.data.annotation import list_available_models, get_downloaded_models

    console.print("[bold blue]Available CellTypist Models[/bold blue]\n")

    if downloaded_only:
        models = get_downloaded_models()
        if not models:
            console.print("[yellow]No models downloaded yet.[/yellow]")
            console.print("Run 'dapidl prepare' or use celltypist to download models.")
            return

        table = Table(title="Downloaded Models")
        table.add_column("Model Name", style="cyan")
        for model in sorted(models):
            table.add_row(model)
        console.print(table)
        console.print(f"\n[green]Total: {len(models)} model(s)[/green]")
    else:
        models_df = list_available_models(force_update=update)

        table = Table(title="All Available Models")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        for _, row in models_df.iterrows():
            # Truncate long descriptions
            desc = row["description"][:80] + "..." if len(row["description"]) > 80 else row["description"]
            table.add_row(row["model"], desc)

        console.print(table)
        console.print(f"\n[green]Total: {len(models_df)} model(s)[/green]")
        console.print("\n[dim]Use --downloaded-only to see locally cached models[/dim]")
        console.print("[dim]Visit https://www.celltypist.org/models for full descriptions[/dim]")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to spatial transcriptomics data (Xenium or MERSCOPE, auto-detected)",
)
@click.option(
    "--xenium-path",
    "-x",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to Xenium output directory (also accepts MERSCOPE)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for prepared dataset",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    default=("Cells_Adult_Breast.pkl",),
    show_default=True,
    help="CellTypist model(s) to use. Repeat for multiple models.",
)
@click.option(
    "--patch-size",
    "-p",
    type=int,
    default=128,
    help="Size of extracted patches (default: 128)",
)
@click.option(
    "--confidence-threshold",
    "-c",
    type=float,
    default=0.5,
    help="Minimum confidence for cell type predictions (default: 0.5)",
)
@click.option(
    "--majority-voting/--no-majority-voting",
    default=True,
    help="Use majority voting for predictions (default: True)",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["consensus", "hierarchical", "popv", "single", "ground_truth"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy), ground_truth (from file)",
)
@click.option(
    "--fine-grained/--no-fine-grained",
    default=False,
    help="Use fine-grained cell types instead of broad categories (default: False)",
)
@click.option(
    "--filter-category",
    type=str,
    default=None,
    help="Filter to only cells of this broad category (e.g., 'Immune' for fine-grained immune classification)",
)
@click.option(
    "--ground-truth-file",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to Excel file with ground truth annotations (required for ground_truth strategy)",
)
@click.option(
    "--ground-truth-sheet",
    type=str,
    default="Xenium R1 Fig1-5 (supervised)",
    show_default=True,
    help="Sheet name in ground truth Excel file",
)
@click.option(
    "--z-index",
    type=int,
    default=3,
    help="Z-slice index for MERSCOPE DAPI images (default: 3)",
)
def prepare(
    input_path: Path | None,
    xenium_path: Path | None,
    output: Path,
    models: Tuple[str, ...],
    patch_size: int,
    confidence_threshold: float,
    majority_voting: bool,
    strategy: str,
    fine_grained: bool,
    filter_category: str | None,
    ground_truth_file: Path | None,
    ground_truth_sheet: str,
    z_index: int,
) -> None:
    """Prepare dataset from spatial transcriptomics data.

    Supports both 10x Xenium and Vizgen MERSCOPE platforms (auto-detected).
    Extracts nucleus patches, generates cell type labels using CellTypist,
    and saves the prepared dataset in Zarr format.

    Annotation strategies:
        - consensus: Voting across multiple models (default, recommended)
        - hierarchical: Tissue-specific model + specialized refinement
        - popv: popV ensemble prediction (requires popv package)
        - single: Legacy single-model mode

    Fine-grained classification:
        Use --fine-grained to classify using detailed cell types (e.g., "CD4+ T cells")
        instead of broad categories (e.g., "Immune"). Combine with --filter-category
        to focus on a specific cell population:

        # Fine-grained immune cell classification
        dapidl prepare -i /path -o /out -m Immune_All_Low.pkl \\
            --fine-grained --filter-category Immune

    For multiple models, use --model multiple times:
        dapidl prepare -i /path -o /out -m Model1.pkl -m Model2.pkl
    """
    from dapidl.data.merscope import create_reader, detect_platform
    from dapidl.data.patches import PatchExtractor
    from dapidl.data.annotation import CellTypeAnnotator

    # Handle deprecated --xenium-path option
    data_path = input_path or xenium_path
    if data_path is None:
        console.print("[red]Error: --input is required[/red]")
        raise click.Abort()
    if xenium_path is not None:
        console.print("[yellow]Warning: --xenium-path is deprecated, use --input instead[/yellow]")

    # Detect platform
    try:
        platform = detect_platform(data_path)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()

    console.print("[bold blue]DAPIDL Dataset Preparation[/bold blue]")
    console.print(f"Platform: {platform.upper()}")
    console.print(f"Input path: {data_path}")
    console.print(f"Output path: {output}")
    console.print(f"Model(s): {', '.join(models)}")
    console.print(f"Strategy: {strategy}")
    console.print(f"Patch size: {patch_size}")
    console.print(f"Confidence threshold: {confidence_threshold}")
    console.print(f"Majority voting: {majority_voting}")
    console.print(f"Fine-grained: {fine_grained}")
    if filter_category:
        console.print(f"Filter category: {filter_category}")
    if strategy.lower() == "ground_truth":
        console.print(f"Ground truth file: {ground_truth_file}")
        console.print(f"Ground truth sheet: {ground_truth_sheet}")
    if platform == "merscope":
        console.print(f"Z-index: {z_index}")

    # Validate ground_truth strategy requirements
    if strategy.lower() == "ground_truth" and ground_truth_file is None:
        console.print("[red]Error: --ground-truth-file is required for ground_truth strategy[/red]")
        raise click.Abort()

    # Load data using auto-detected reader
    console.print(f"\n[yellow]Loading {platform.upper()} data...[/yellow]")
    if platform == "merscope":
        reader = create_reader(data_path, z_index=z_index)
    else:
        reader = create_reader(data_path)

    # Create annotator with specified models and strategy
    annotator = CellTypeAnnotator(
        model_names=list(models),
        confidence_threshold=confidence_threshold,
        majority_voting=majority_voting,
        strategy=strategy,
        fine_grained=fine_grained,
        filter_category=filter_category,
        ground_truth_file=str(ground_truth_file) if ground_truth_file else None,
        ground_truth_sheet=ground_truth_sheet,
    )

    # Extract patches
    console.print("[yellow]Extracting patches...[/yellow]")
    extractor = PatchExtractor(
        reader=reader,
        patch_size=patch_size,
        confidence_threshold=confidence_threshold,
        annotator=annotator,
    )
    extractor.extract_and_save(output)

    console.print(f"\n[bold green]Dataset saved to {output}[/bold green]")


@main.command()
@click.option(
    "--xenium-path",
    "-x",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to Xenium output directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for annotated Xenium dataset",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    default=("Cells_Adult_Breast.pkl",),
    show_default=True,
    help="CellTypist model(s) to use. Repeat for multiple models.",
)
@click.option(
    "--output-format",
    type=click.Choice(["csv", "parquet", "both"], case_sensitive=False),
    default="csv",
    show_default=True,
    help="Output format: 'csv' for Xenium Explorer import (recommended), 'parquet' to update cells.parquet, 'both' for both.",
)
@click.option(
    "--majority-voting/--no-majority-voting",
    default=True,
    help="Use majority voting for predictions (default: True)",
)
@click.option(
    "--add-colors/--no-colors",
    default=True,
    help="Add auto-generated HEX colors to CSV exports (default: True)",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["consensus", "hierarchical", "popv", "single"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy)",
)
def annotate(
    xenium_path: Path,
    output: Path,
    models: Tuple[str, ...],
    output_format: str,
    majority_voting: bool,
    add_colors: bool,
    strategy: str,
) -> None:
    """Annotate Xenium dataset and create copy for Xenium Explorer.

    Creates a space-efficient copy of the Xenium dataset using hardlinks
    for unchanged files, and adds CellTypist annotations in the specified format.

    The CSV format (default) is recommended for Xenium Explorer - import the
    generated CSV files via Cells -> Cell Groups -> Upload.

    Annotation strategies:
        - consensus: Voting across multiple models (default, recommended)
        - hierarchical: Tissue-specific model + specialized refinement
        - popv: popV ensemble prediction (requires popv package)
        - single: Legacy single-model mode

    Examples:
        # Annotate with default breast model
        dapidl annotate -x /path/to/xenium -o /path/to/output

        # Use multiple models with consensus voting
        dapidl annotate -x /path/to/xenium -o /path/to/output \\
            -m Cells_Adult_Breast.pkl -m Immune_All_High.pkl
    """
    from dapidl.data.xenium import XeniumDataReader
    from dapidl.data.annotation import CellTypeAnnotator
    from dapidl.data.xenium_export import create_annotated_xenium_dataset

    console.print("[bold blue]DAPIDL Xenium Annotation[/bold blue]")
    console.print(f"Xenium path: {xenium_path}")
    console.print(f"Output path: {output}")
    console.print(f"Model(s): {', '.join(models)}")
    console.print(f"Strategy: {strategy}")
    console.print(f"Output format: {output_format}")
    console.print(f"Majority voting: {majority_voting}")
    console.print(f"Add colors: {add_colors}")

    # Load Xenium data
    console.print("\n[yellow]Loading Xenium data...[/yellow]")
    reader = XeniumDataReader(xenium_path)

    # Create annotator and run annotation
    console.print("[yellow]Running cell type annotation...[/yellow]")
    annotator = CellTypeAnnotator(
        model_names=list(models),
        confidence_threshold=0.0,  # Keep all cells for visualization
        majority_voting=majority_voting,
        strategy=strategy,
    )
    annotations = annotator.annotate_from_reader(reader)

    # Create annotated dataset
    console.print("[yellow]Creating annotated dataset...[/yellow]")
    create_annotated_xenium_dataset(
        xenium_dir=xenium_path,
        output_dir=output,
        annotations_df=annotations,
        model_names=list(models),
        output_format=output_format.lower(),
        add_colors=add_colors,
    )

    console.print(f"\n[bold green]Annotated dataset created at {output}[/bold green]")

    if output_format.lower() in ("csv", "both"):
        console.print("\n[cyan]To view in Xenium Explorer:[/cyan]")
        console.print("  1. Open the annotated dataset in Xenium Explorer")
        console.print("  2. Go to: Cells -> Cell Groups -> Upload")
        console.print("  3. Select the generated CSV file(s)")


@main.command(name="create-dataset")
@click.option(
    "--xenium-path",
    "-x",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to source Xenium output directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for new hardlinked dataset",
)
@click.option(
    "--csv",
    "-c",
    "csv_files",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="CSV file(s) to include in dataset. Repeat for multiple files.",
)
@click.option(
    "--replace",
    "-r",
    "replace_files",
    multiple=True,
    nargs=2,
    type=(str, click.Path(exists=True, path_type=Path)),
    help="Replace a file: --replace <relative_path> <new_file_path>",
)
def create_dataset(
    xenium_path: Path,
    output: Path,
    csv_files: Tuple[Path, ...],
    replace_files: Tuple[Tuple[str, Path], ...],
) -> None:
    """Create a hardlinked Xenium dataset with custom CSV files.

    Creates a space-efficient copy of a Xenium dataset using hardlinks for
    all unchanged files, and adds the specified CSV file(s) for Xenium Explorer.

    This is useful when you have pre-computed cell annotations in CSV format
    and want to create a viewable dataset without duplicating the large files.

    Examples:
        # Add a single CSV file
        dapidl create-dataset -x /path/to/xenium -o /output -c annotations.csv

        # Add multiple CSV files
        dapidl create-dataset -x /path/to/xenium -o /output \\
            -c celltypes.csv -c clusters.csv

        # Replace cells.parquet with a modified version
        dapidl create-dataset -x /path/to/xenium -o /output \\
            -c annotations.csv --replace cells.parquet /path/to/new_cells.parquet
    """
    from dapidl.data.xenium_export import create_hardlink_dataset
    import shutil

    console.print("[bold blue]DAPIDL Create Hardlinked Dataset[/bold blue]")
    console.print(f"Source Xenium: {xenium_path}")
    console.print(f"Output path: {output}")
    console.print(f"CSV files: {', '.join(str(f) for f in csv_files)}")
    if replace_files:
        console.print(f"Files to replace: {len(replace_files)}")

    # Find the outs directory
    if (xenium_path / "outs").exists():
        xenium_outs = xenium_path / "outs"
        output_outs = output / "outs"
    elif (xenium_path / "cells.parquet").exists():
        xenium_outs = xenium_path
        output_outs = output
    else:
        console.print("[red]Error: Cannot find Xenium output files[/red]")
        raise click.Abort()

    # Build modified files dict from --replace options
    modified_files: dict[str, Path] = {}
    for rel_path, new_file in replace_files:
        modified_files[rel_path] = Path(new_file)
        console.print(f"  Will replace: {rel_path} -> {new_file}")

    # Create hardlinked dataset
    console.print("\n[yellow]Creating hardlinked dataset...[/yellow]")
    create_hardlink_dataset(
        source_dir=xenium_outs,
        output_dir=output_outs,
        modified_files=modified_files,
    )

    # Copy CSV files to output
    console.print("[yellow]Copying CSV files...[/yellow]")
    for csv_file in csv_files:
        dest = output_outs / csv_file.name
        shutil.copy2(csv_file, dest)
        console.print(f"  Copied: {csv_file.name}")

    console.print(f"\n[bold green]Dataset created at {output}[/bold green]")
    console.print("\n[cyan]To view in Xenium Explorer:[/cyan]")
    console.print(f"  1. Open {output_outs} in Xenium Explorer")
    console.print("  2. Go to: Cells -> Cell Groups -> Upload")
    console.print("  3. Select the CSV file(s) to import")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to Hydra config file",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to prepared dataset",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    help="Output directory for checkpoints and logs",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=100,
    help="Number of training epochs (default: 100)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=64,
    help="Batch size (default: 64)",
)
@click.option(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate (default: 1e-4)",
)
@click.option(
    "--wandb/--no-wandb",
    default=True,
    help="Enable Weights & Biases logging",
)
@click.option(
    "--backbone",
    type=str,
    default="efficientnetv2_rw_s",
    show_default=True,
    help="CNN backbone architecture. Use 'dapidl list-backbones' for options.",
)
@click.option(
    "--max-weight-ratio",
    type=float,
    default=10.0,
    show_default=True,
    help="Max class weight ratio to prevent mode collapse (0 to disable capping)",
)
@click.option(
    "--min-samples",
    type=int,
    default=None,
    help="Filter classes with fewer than this many samples",
)
@click.option(
    "--backend",
    type=click.Choice(["pytorch", "dali", "dali-lmdb"], case_sensitive=False),
    default="pytorch",
    show_default=True,
    help="Data loading backend: pytorch (CPU), dali (GPU with Zarr), or dali-lmdb (GPU with LMDB, fastest)",
)
def train(
    config: Path | None,
    data: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    wandb: bool,
    backbone: str,
    max_weight_ratio: float,
    min_samples: int | None,
    backend: str,
) -> None:
    """Train cell type classifier.

    Trains a CNN model to classify cell types from DAPI patches.

    Backbone options:
        - efficientnetv2_rw_s: EfficientNetV2-S (default, ImageNet pretrained)
        - resnet18/resnet34: ResNet variants (lighter, ImageNet pretrained)
        - convnext_tiny/convnext_small: Modern ConvNeXt (ImageNet pretrained)
        - microscopy_cnn: Custom lightweight CNN (~1M params, no pretraining)
        - microscopy_cnn_deep: Deeper custom CNN (~4M params, no pretraining)

    Use 'dapidl list-backbones' to see all available options.
    """
    from dapidl.training.trainer import Trainer

    console.print(f"[bold blue]DAPIDL Training[/bold blue]")
    console.print(f"Data path: {data}")
    console.print(f"Output path: {output}")
    console.print(f"Backbone: {backbone}")
    console.print(f"Backend: {backend}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {lr}")
    console.print(f"Max weight ratio: {max_weight_ratio}")
    if min_samples:
        console.print(f"Min samples per class: {min_samples}")
    console.print(f"W&B logging: {wandb}")

    # Check DALI availability if requested
    if backend.lower() == "dali":
        from dapidl.data.dali_pipeline import is_dali_available
        if not is_dali_available():
            console.print("[red]Error: DALI backend requested but nvidia-dali is not installed.[/red]")
            console.print("[yellow]Install with: pip install nvidia-dali-cuda120[/yellow]")
            console.print("[yellow]Or use: uv add nvidia-dali-cuda120[/yellow]")
            raise click.Abort()

    trainer = Trainer(
        data_path=data,
        output_path=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        use_wandb=wandb,
        backbone_name=backbone,
        max_weight_ratio=max_weight_ratio,
        min_samples_per_class=min_samples,
        backend=backend.lower(),
    )
    trainer.train()

    console.print(f"\n[bold green]Training complete![/bold green]")


@main.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to test dataset",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for evaluation results",
)
def evaluate(checkpoint: Path, data: Path, output: Path | None) -> None:
    """Evaluate trained model.

    Computes metrics (F1, accuracy, confusion matrix) on test set.
    """
    from dapidl.evaluation.metrics import compute_metrics, plot_confusion_matrix

    console.print(f"[bold blue]DAPIDL Evaluation[/bold blue]")
    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Data path: {data}")

    # TODO: Implement evaluation
    console.print("\n[yellow]Evaluation not yet implemented[/yellow]")


@main.command(name="compare-labels")
@click.option(
    "--predictions",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to predictions CSV/parquet with cell_id and predicted_type columns",
)
@click.option(
    "--ground-truth",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to ground truth Excel file (Xenium Cell_Barcode_Type_Matrices.xlsx)",
)
@click.option(
    "--sheet",
    "-s",
    type=str,
    default="Xenium R1 Fig1-5 (supervised)",
    help="Sheet name in Excel file with ground truth labels",
)
@click.option(
    "--prediction-source",
    type=str,
    default="celltypist_breast",
    help="Prediction source name for mapping lookup (e.g., celltypist_breast, popv_immune)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output path for detailed results CSV",
)
def compare_labels(
    predictions: Path,
    ground_truth: Path,
    sheet: str,
    prediction_source: str,
    output: Path | None,
) -> None:
    """Compare prediction labels against ground truth using harmonization.

    Evaluates cell type predictions against ground truth at multiple
    hierarchy levels (broad, mid, fine) using label harmonization to
    handle differences in label vocabularies.

    The harmonization system maps labels from different sources to a
    common hierarchy, enabling meaningful comparison between:
    - CellTypist predictions (e.g., CD4-Tem, Macro-m1)
    - Ground truth labels (e.g., CD4+_T_Cells, Macrophages_1)

    Examples:
        # Compare CellTypist predictions to ground truth
        dapidl compare-labels -p annotations.parquet \\
            -g Cell_Barcode_Type_Matrices.xlsx

        # Specify prediction source for better mapping
        dapidl compare-labels -p predictions.csv \\
            -g ground_truth.xlsx --prediction-source popv_immune

        # Save detailed results
        dapidl compare-labels -p pred.parquet -g gt.xlsx -o results.csv
    """
    import pandas as pd
    import polars as pl
    from dapidl.harmonization import (
        LabelHarmonizer,
        evaluate_annotations_df,
        print_evaluation_report,
    )
    from dapidl.data.annotation import GROUND_TRUTH_MAPPING

    console.print("[bold blue]DAPIDL Label Comparison with Harmonization[/bold blue]")
    console.print()
    console.print(f"Predictions:       {predictions}")
    console.print(f"Ground truth:      {ground_truth}")
    console.print(f"Sheet:             {sheet}")
    console.print(f"Prediction source: {prediction_source}")
    console.print()

    # Load predictions
    console.print("[yellow]Loading predictions...[/yellow]")
    if predictions.suffix == ".parquet":
        pred_df = pl.read_parquet(predictions)
    elif predictions.suffix == ".csv":
        pred_df = pl.read_csv(predictions)
    else:
        console.print(f"[red]Unsupported format: {predictions.suffix}[/red]")
        raise click.Abort()

    # Determine prediction column name
    pred_col = None
    for col in ["predicted_type", "predicted_type_1", "cell_type", "label"]:
        if col in pred_df.columns:
            pred_col = col
            break
    if pred_col is None:
        console.print(f"[red]No prediction column found. Have: {pred_df.columns}[/red]")
        raise click.Abort()

    console.print(f"  Found {len(pred_df)} predictions (column: {pred_col})")

    # Load ground truth
    console.print("[yellow]Loading ground truth...[/yellow]")
    gt_pd = pd.read_excel(ground_truth, sheet_name=sheet)
    gt_df = pl.DataFrame({
        "cell_id": gt_pd["Barcode"].values,
        "predicted_type": gt_pd["Cluster"].values,
    })
    console.print(f"  Found {len(gt_df)} ground truth cells")

    # Run harmonized evaluation
    console.print("[yellow]Running harmonized evaluation...[/yellow]")
    harmonizer = LabelHarmonizer()
    result, joined_df = evaluate_annotations_df(
        predictions_df=pred_df,
        ground_truth_df=gt_df,
        pred_label_col=pred_col,
        gt_label_col="predicted_type",
        prediction_source=prediction_source,
        ground_truth_source="xenium_breast",
        harmonizer=harmonizer,
    )

    # Print report
    print_evaluation_report(result, title="Cell Type Prediction Evaluation")

    # Save detailed results if requested
    if output:
        console.print(f"[yellow]Saving detailed results to {output}...[/yellow]")
        joined_df.write_csv(output)
        console.print(f"[green]✓ Results saved to {output}[/green]")

    # Summary table
    console.print("[bold cyan]Summary Metrics:[/bold cyan]")
    table = Table()
    table.add_column("Level", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Macro F1", style="green")
    table.add_column("Samples", style="yellow")

    for level in ["broad", "mid", "fine"]:
        if level in result.metrics:
            m = result.metrics[level]
            table.add_row(
                level.upper(),
                f"{m['accuracy']:.3f}",
                f"{m['f1_macro']:.3f}",
                str(m['n_samples']),
            )
    console.print(table)


@main.command()
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model",
)
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to DAPI image",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for predictions",
)
@click.option(
    "--segmentation",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to pre-computed segmentation (optional)",
)
def predict(
    model: Path,
    image: Path,
    output: Path,
    segmentation: Path | None,
) -> None:
    """Predict cell types from DAPI image.

    Runs inference on a new DAPI image, optionally using pre-computed segmentation.
    """
    console.print(f"[bold blue]DAPIDL Prediction[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Image: {image}")
    console.print(f"Output: {output}")

    # TODO: Implement prediction
    console.print("\n[yellow]Prediction not yet implemented[/yellow]")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to spatial transcriptomics data (Xenium or MERSCOPE, auto-detected)",
)
@click.option(
    "--xenium-path",
    "-x",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to Xenium output directory (also accepts MERSCOPE)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for dataset and model",
)
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    default=("Cells_Adult_Breast.pkl",),
    show_default=True,
    help="CellTypist model(s) to use. Repeat for multiple models.",
)
@click.option(
    "--patch-size",
    "-p",
    type=int,
    default=128,
    help="Size of extracted patches (default: 128)",
)
@click.option(
    "--confidence-threshold",
    "-c",
    type=float,
    default=0.5,
    help="Minimum confidence for cell type predictions (default: 0.5)",
)
@click.option(
    "--majority-voting/--no-majority-voting",
    default=True,
    help="Use majority voting for predictions (default: True)",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=50,
    help="Number of training epochs (default: 50)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=64,
    help="Batch size (default: 64)",
)
@click.option(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate (default: 1e-4)",
)
@click.option(
    "--wandb/--no-wandb",
    default=True,
    help="Enable Weights & Biases logging",
)
@click.option(
    "--skip-prepare",
    is_flag=True,
    default=False,
    help="Skip dataset preparation (use existing dataset)",
)
@click.option(
    "--skip-train",
    is_flag=True,
    default=False,
    help="Skip training (prepare dataset only)",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["consensus", "hierarchical", "popv", "single"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy)",
)
@click.option(
    "--fine-grained/--no-fine-grained",
    default=False,
    help="Use fine-grained cell types instead of broad categories (default: False)",
)
@click.option(
    "--filter-category",
    type=str,
    default=None,
    help="Filter to only cells of this broad category (e.g., 'Immune' for fine-grained immune classification)",
)
@click.option(
    "--backbone",
    type=str,
    default="efficientnetv2_rw_s",
    show_default=True,
    help="CNN backbone architecture. Use 'dapidl list-backbones' for options.",
)
@click.option(
    "--z-index",
    type=int,
    default=3,
    help="Z-slice index for MERSCOPE DAPI images (default: 3)",
)
def pipeline(
    input_path: Path | None,
    xenium_path: Path | None,
    output: Path,
    models: Tuple[str, ...],
    patch_size: int,
    confidence_threshold: float,
    majority_voting: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    wandb: bool,
    skip_prepare: bool,
    skip_train: bool,
    strategy: str,
    fine_grained: bool,
    filter_category: str | None,
    backbone: str,
    z_index: int,
) -> None:
    """Run the complete DAPIDL pipeline: prepare + train.

    Supports both 10x Xenium and Vizgen MERSCOPE platforms (auto-detected).

    This supercommand executes the full workflow:
    1. Prepare dataset from spatial data (extract patches, annotate with CellTypist)
    2. Train the CNN classifier on the prepared dataset

    Output structure:
        <output>/
        ├── dataset/          # Prepared training data
        │   ├── patches.zarr/ # DAPI patches
        │   ├── labels.npy
        │   └── metadata.parquet
        └── training/         # Model outputs
            ├── checkpoints/
            └── logs/

    Annotation strategies:
        - consensus: Voting across multiple models (default, recommended)
        - hierarchical: Tissue-specific model + specialized refinement
        - popv: popV ensemble prediction (requires popv package)
        - single: Legacy single-model mode

    Examples:
        # Run full pipeline with Xenium data
        dapidl pipeline -i /path/to/xenium -o ./experiment

        # Run pipeline with MERSCOPE data
        dapidl pipeline -i /path/to/merscope -o ./experiment

        # Use custom model and more epochs
        dapidl pipeline -i /path/to/data -o ./experiment \\
            -m Immune_All_High.pkl --epochs 100

        # Only prepare dataset, skip training
        dapidl pipeline -i /path/to/data -o ./experiment --skip-train

        # Only train, using existing dataset
        dapidl pipeline -i /path/to/data -o ./experiment --skip-prepare
    """
    from dapidl.data.merscope import create_reader, detect_platform
    from dapidl.data.patches import PatchExtractor
    from dapidl.data.annotation import CellTypeAnnotator
    from dapidl.training.trainer import Trainer

    # Handle both -i and -x options
    data_path = input_path or xenium_path
    if data_path is None and not skip_prepare:
        console.print("[red]Error: --input or --xenium-path is required[/red]")
        raise click.Abort()

    # Detect platform if we have a path
    platform = None
    if data_path:
        try:
            platform = detect_platform(data_path)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.Abort()

    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]                    DAPIDL PIPELINE                           [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print()

    # Define paths
    dataset_path = output / "dataset"
    training_path = output / "training"

    # Display configuration
    console.print("[bold cyan]Configuration:[/bold cyan]")
    if platform:
        console.print(f"  Platform:       {platform.upper()}")
    console.print(f"  Input:          {data_path}")
    console.print(f"  Output dir:     {output}")
    console.print(f"  Dataset dir:    {dataset_path}")
    console.print(f"  Training dir:   {training_path}")
    console.print(f"  Model(s):       {', '.join(models)}")
    console.print(f"  Strategy:       {strategy}")
    console.print(f"  Patch size:     {patch_size}")
    console.print(f"  Confidence:     {confidence_threshold}")
    console.print(f"  Majority vote:  {majority_voting}")
    console.print(f"  Fine-grained:   {fine_grained}")
    if filter_category:
        console.print(f"  Filter cat:     {filter_category}")
    console.print(f"  Backbone:       {backbone}")
    console.print(f"  Epochs:         {epochs}")
    console.print(f"  Batch size:     {batch_size}")
    console.print(f"  Learning rate:  {lr}")
    console.print(f"  W&B logging:    {wandb}")
    console.print()

    # Step 1: Prepare dataset
    if not skip_prepare:
        console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
        console.print("[bold blue]  STEP 1: Dataset Preparation                                  [/bold blue]")
        console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
        console.print()

        console.print(f"[yellow]Loading {platform.upper()} data...[/yellow]")
        if platform == "merscope":
            reader = create_reader(data_path, z_index=z_index)
        else:
            reader = create_reader(data_path)

        console.print("[yellow]Initializing CellTypist annotator...[/yellow]")
        annotator = CellTypeAnnotator(
            model_names=list(models),
            confidence_threshold=confidence_threshold,
            majority_voting=majority_voting,
            strategy=strategy,
            fine_grained=fine_grained,
            filter_category=filter_category,
        )

        console.print("[yellow]Extracting patches and generating labels...[/yellow]")
        extractor = PatchExtractor(
            reader=reader,
            patch_size=patch_size,
            confidence_threshold=confidence_threshold,
            annotator=annotator,
        )
        extractor.extract_and_save(dataset_path)

        console.print(f"\n[green]✓ Dataset prepared at {dataset_path}[/green]")
        console.print()
    else:
        console.print("[yellow]Skipping dataset preparation (--skip-prepare)[/yellow]")
        if not dataset_path.exists():
            console.print(f"[red]Error: Dataset not found at {dataset_path}[/red]")
            console.print("[red]Remove --skip-prepare or prepare dataset first.[/red]")
            raise click.Abort()
        console.print()

    # Step 2: Train model
    if not skip_train:
        console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
        console.print("[bold blue]  STEP 2: Model Training                                       [/bold blue]")
        console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
        console.print()

        trainer = Trainer(
            data_path=dataset_path,
            output_path=training_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            use_wandb=wandb,
            backbone_name=backbone,
        )
        trainer.train()

        console.print(f"\n[green]✓ Training complete![/green]")
        console.print()
    else:
        console.print("[yellow]Skipping training (--skip-train)[/yellow]")
        console.print()

    # Summary
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]                    PIPELINE COMPLETE                          [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print()
    console.print("[bold green]Output locations:[/bold green]")
    console.print(f"  Dataset:     {dataset_path}")
    if not skip_train:
        console.print(f"  Model:       {training_path}/checkpoints/")
        console.print(f"  Logs:        {training_path}/")


@main.command(name="export-lmdb")
@click.option(
    "-d", "--data",
    "data_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to prepared dataset directory (with patches.zarr)",
)
@click.option(
    "-o", "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output path for LMDB database (default: <data>/patches.lmdb)",
)
@click.option(
    "--map-size",
    type=float,
    default=50.0,
    show_default=True,
    help="Maximum LMDB database size in GB",
)
@click.option(
    "-w", "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of parallel workers for reading Zarr data",
)
def export_lmdb(data_path: str, output_path: str | None, map_size: float, workers: int) -> None:
    """Convert Zarr patches to LMDB format for faster DALI training.

    This command converts the patches.zarr file to LMDB format, which can
    be read much faster by NVIDIA DALI. Use with --backend dali-lmdb in train.

    Example:
        dapidl export-lmdb -d ./dataset
        dapidl train -d ./dataset --backend dali-lmdb --batch-size 256
    """
    from dapidl.data.dali_native import convert_dataset_to_lmdb, is_lmdb_available

    if not is_lmdb_available():
        console.print("[red]Error: LMDB is not installed.[/red]")
        console.print("[yellow]Install with: pip install lmdb[/yellow]")
        raise click.Abort()

    console.print("[bold blue]Converting Zarr to LMDB Format[/bold blue]")
    console.print(f"Input: {data_path}")
    console.print(f"Max size: {map_size} GB")

    data_path_obj = Path(data_path)

    # Check that Zarr exists
    zarr_path = data_path_obj / "patches.zarr"
    if not zarr_path.exists():
        console.print(f"[red]Error: patches.zarr not found at {zarr_path}[/red]")
        raise click.Abort()

    console.print(f"Workers: {workers}")

    try:
        lmdb_path = convert_dataset_to_lmdb(
            data_path=data_path,
            output_path=output_path,
            map_size_gb=map_size,
            num_workers=workers,
        )
        console.print(f"\n[green]✓ LMDB export complete![/green]")
        console.print(f"  Output: {lmdb_path}")
        console.print("\n[dim]Use with: dapidl train -d <dataset> --backend dali-lmdb[/dim]")
    except Exception as e:
        console.print(f"[red]Error during conversion: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    main()
