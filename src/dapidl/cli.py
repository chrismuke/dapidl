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
    type=click.Choice(["consensus", "hierarchical", "popv", "single", "ground_truth", "singler"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy), ground_truth (from file), singler (R-based reference)",
)
@click.option(
    "--singler-ref",
    type=click.Choice(["blueprint", "hpca", "monaco", "novershtern"], case_sensitive=False),
    default="blueprint",
    show_default=True,
    help="SingleR reference dataset (only used with --strategy singler). blueprint=BlueprintEncodeData (best), hpca=HumanPrimaryCellAtlas",
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
    singler_ref: str,
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
        singler_reference=singler_ref,
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
    type=click.Choice(["consensus", "hierarchical", "popv", "single", "singler"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy), singler (R-based reference)",
)
@click.option(
    "--singler-ref",
    type=click.Choice(["blueprint", "hpca", "monaco", "novershtern"], case_sensitive=False),
    default="blueprint",
    show_default=True,
    help="SingleR reference dataset (only used with --strategy singler)",
)
def annotate(
    xenium_path: Path,
    output: Path,
    models: Tuple[str, ...],
    output_format: str,
    majority_voting: bool,
    add_colors: bool,
    strategy: str,
    singler_ref: str,
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
        singler_reference=singler_ref,
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
@click.option(
    "--heavy-aug/--no-heavy-aug",
    default=False,
    help="Use heavy augmentation for rare classes (<5%% of data)",
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
    heavy_aug: bool,
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
    console.print(f"Heavy augmentation: {heavy_aug}")
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
        use_heavy_aug=heavy_aug,
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
    type=click.Choice(["consensus", "hierarchical", "popv", "single", "singler"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy), singler (R-based reference)",
)
@click.option(
    "--singler-ref",
    type=click.Choice(["blueprint", "hpca", "monaco", "novershtern"], case_sensitive=False),
    default="blueprint",
    show_default=True,
    help="SingleR reference dataset (only used with --strategy singler)",
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
    singler_ref: str,
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

        console.print("[yellow]Initializing annotator...[/yellow]")
        annotator = CellTypeAnnotator(
            model_names=list(models),
            confidence_threshold=confidence_threshold,
            majority_voting=majority_voting,
            strategy=strategy,
            fine_grained=fine_grained,
            filter_category=filter_category,
            singler_reference=singler_ref,
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


# ─────────────────────────────────────────────────────────────────────────────
# HEIST Commands
# ─────────────────────────────────────────────────────────────────────────────


@main.command(name="heist-prepare")
@click.option(
    "-x",
    "--xenium-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to Xenium output directory",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for HEIST data",
)
@click.option(
    "--model",
    "celltypist_model",
    default="Cells_Adult_Breast.pkl",
    help="CellTypist model for annotations",
)
@click.option(
    "--ground-truth",
    "ground_truth_file",
    default=None,
    type=click.Path(exists=True),
    help="Path to ground truth Excel file (overrides CellTypist)",
)
@click.option(
    "--mi-threshold",
    default=0.35,
    type=float,
    help="Mutual information threshold for GRN edges",
)
@click.option(
    "--spatial-k",
    default=10,
    type=int,
    help="Number of spatial neighbors",
)
@click.option(
    "--max-distance-um",
    default=50.0,
    type=float,
    help="Maximum distance for spatial edges (micrometers)",
)
@click.option(
    "--max-cells-per-type",
    default=5000,
    type=int,
    help="Maximum cells to sample per type for GRN computation (for speed)",
)
def heist_prepare(
    xenium_path: str,
    output_dir: str,
    celltypist_model: str,
    ground_truth_file: str | None,
    mi_threshold: float,
    spatial_k: int,
    max_distance_um: float,
    max_cells_per_type: int,
) -> None:
    """Prepare data for HEIST training.

    Exports expression matrix, builds spatial graph and cell-type GRNs.

    Example:
        dapidl heist-prepare -x /path/to/xenium -o ./heist_data
    """
    import numpy as np
    import torch
    from loguru import logger

    from dapidl.data.annotation import CellTypeAnnotator
    from dapidl.data.xenium import XeniumDataReader
    from dapidl.models.heist import CellTypeGRNBuilder, SpatialGraphBuilder

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]HEIST Data Preparation[/bold blue]")
    console.print(f"Xenium path: {xenium_path}")
    console.print(f"Output: {output_dir}")

    # Load Xenium data
    console.print("\n[cyan]Loading Xenium data...[/cyan]")
    reader = XeniumDataReader(xenium_path)
    cells_df = reader.cells_df  # Property, not method
    expression_matrix, gene_names, cell_ids = reader.load_expression_matrix()

    console.print(f"  Cells: {len(cells_df)}")
    console.print(f"  Genes: {expression_matrix.shape[1]}")

    # Get coordinates (in microns for spatial graph)
    coords = reader.get_centroids_microns()

    # Annotate cells
    import anndata as ad
    import pandas as pd
    from dapidl.data.annotation import AnnotationStrategy

    # Create AnnData for annotation
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame({"cell_id": cell_ids}),
        var=pd.DataFrame(index=gene_names),
    )

    if ground_truth_file:
        console.print(f"\n[cyan]Loading ground truth from {ground_truth_file}...[/cyan]")
        annotator = CellTypeAnnotator(
            strategy=AnnotationStrategy.GROUND_TRUTH,
            ground_truth_file=ground_truth_file,
            fine_grained=True,
        )
    else:
        console.print(f"\n[cyan]Annotating cells with {celltypist_model}...[/cyan]")
        annotator = CellTypeAnnotator(
            model_names=celltypist_model,
            fine_grained=True,
        )

    annotations = annotator.annotate(adata)

    # Get annotated cell IDs and filter data to match
    # Ensure consistent types (convert to strings for comparison)
    annotated_cell_ids = annotations["cell_id"].to_numpy()
    annotated_cell_ids_set = set(str(cid) for cid in annotated_cell_ids)
    cell_id_to_idx = {str(cid): i for i, cid in enumerate(cell_ids)}

    # Create valid indices matching annotation order
    valid_indices = np.array([cell_id_to_idx[str(cid)] for cid in annotated_cell_ids
                              if str(cid) in cell_id_to_idx])

    # Filter expression and coordinates to annotated cells
    expression_matrix = expression_matrix[valid_indices]
    coords = coords[valid_indices]

    # Also filter labels to match valid_indices order
    valid_cids = [str(cid) for cid in annotated_cell_ids if str(cid) in cell_id_to_idx]

    console.print(f"  Annotated cells: {len(valid_indices)} / {len(cell_ids)}")

    # Get fine-grained labels - filter to match valid_indices
    # Create a mapping from cell_id to label
    labels_full = annotations["predicted_type"].to_numpy()
    cid_to_label = {str(cid): label for cid, label in zip(annotated_cell_ids, labels_full, strict=True)}
    labels = np.array([cid_to_label[cid] for cid in valid_cids])

    label_to_idx = {label: i for i, label in enumerate(sorted(set(labels)))}
    labels_encoded = np.array([label_to_idx[l] for l in labels])

    console.print(f"  Cell types: {len(label_to_idx)}")

    # Build spatial graph
    console.print(f"\n[cyan]Building spatial graph (k={spatial_k})...[/cyan]")
    spatial_builder = SpatialGraphBuilder(k=spatial_k, max_distance_um=max_distance_um)
    spatial_edge_index, _ = spatial_builder.build_graph(coords)

    console.print(f"  Edges: {spatial_edge_index.shape[1]}")

    # Build cell-type GRNs
    console.print(f"\n[cyan]Building cell-type GRNs (MI > {mi_threshold})...[/cyan]")
    grn_builder = CellTypeGRNBuilder(
        mi_threshold=mi_threshold,
        max_cells_per_type=max_cells_per_type,
    )

    # Log-normalize expression for GRN computation
    expr_log = np.log1p(expression_matrix)
    cell_type_names = list(label_to_idx.keys())
    cell_type_grns = grn_builder.build_grns(
        expr_log, labels_encoded, cell_type_names
    )

    # Build universal GRN for inference
    universal_grn = grn_builder.build_universal_grn(cell_type_grns)

    # Save everything
    console.print("\n[cyan]Saving data...[/cyan]")
    np.save(output_path / "expression.npy", expr_log.astype(np.float32))
    np.save(output_path / "coords.npy", coords.astype(np.float32))
    np.save(output_path / "labels.npy", labels_encoded)
    torch.save(spatial_edge_index, output_path / "spatial_graph.pt")
    torch.save(cell_type_grns, output_path / "cell_type_grns.pt")
    torch.save(universal_grn, output_path / "universal_grn.pt")

    # Save metadata
    import json
    metadata = {
        "n_cells": len(valid_indices),
        "n_genes": expression_matrix.shape[1],
        "n_cell_types": len(label_to_idx),
        "cell_type_map": label_to_idx,
        "gene_names": gene_names,
        "annotation_source": ground_truth_file if ground_truth_file else celltypist_model,
        "mi_threshold": mi_threshold,
        "spatial_k": spatial_k,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"\n[green]✓ HEIST data prepared![/green]")
    console.print(f"  Expression: {output_path / 'expression.npy'}")
    console.print(f"  Spatial graph: {spatial_edge_index.shape[1]} edges")
    console.print(f"  Cell-type GRNs: {len(cell_type_grns)} types")
    console.print("\n[dim]Next: dapidl heist-train -d {output_dir}[/dim]")


@main.command(name="heist-train")
@click.option(
    "-d",
    "--data-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to HEIST data directory (from heist-prepare)",
)
@click.option(
    "-o",
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for model (default: data_dir/training)",
)
@click.option("--epochs", default=50, type=int, help="Number of training epochs")
@click.option("--batch-size", default=4, type=int, help="Partitions per batch")
@click.option("--partition-size", default=128, type=int, help="Cells per partition")
@click.option("--hidden-dim", default=128, type=int, help="Hidden dimension")
@click.option("--n-layers", default=4, type=int, help="Number of HEIST layers")
@click.option("--learning-rate", default=1e-3, type=float, help="Learning rate")
@click.option("--patience", default=10, type=int, help="Early stopping patience")
@click.option("--no-wandb", is_flag=True, help="Disable W&B logging")
@click.option("--wandb-project", default="dapidl-heist", help="W&B project name")
def heist_train(
    data_dir: str,
    output_dir: str | None,
    epochs: int,
    batch_size: int,
    partition_size: int,
    hidden_dim: int,
    n_layers: int,
    learning_rate: float,
    patience: int,
    no_wandb: bool,
    wandb_project: str,
) -> None:
    """Train HEIST classifier.

    Trains the HEIST model for cell type classification using
    expression data prepared with heist-prepare.

    Example:
        dapidl heist-train -d ./heist_data --epochs 50
    """
    import json
    import numpy as np
    import torch
    from loguru import logger

    from dapidl.data.heist_dataset import create_heist_data_splits, load_heist_data
    from dapidl.models.heist import HEISTClassifier
    from dapidl.training.heist_trainer import HEISTTrainer, compute_class_weights

    data_path = Path(data_dir)
    if output_dir is None:
        output_path = data_path / "training"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]HEIST Training[/bold blue]")
    console.print(f"Data: {data_dir}")
    console.print(f"Output: {output_path}")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        console.print(f"[green]Using GPU: {torch.cuda.get_device_name()}[/green]")
    else:
        console.print("[yellow]Warning: Training on CPU (slow)[/yellow]")

    # Load data
    console.print("\n[cyan]Loading data...[/cyan]")
    expression, coords, labels, spatial_edge_index, cell_type_grns = load_heist_data(
        data_path
    )

    # Load metadata
    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    n_genes = expression.shape[1]
    n_classes = metadata["n_cell_types"]

    console.print(f"  Cells: {len(expression)}")
    console.print(f"  Genes: {n_genes}")
    console.print(f"  Classes: {n_classes}")

    # Create data splits
    console.print("\n[cyan]Creating train/val/test splits...[/cyan]")
    train_ds, val_ds, test_ds = create_heist_data_splits(
        expression=expression,
        coords=coords,
        labels=labels,
        spatial_edge_index=spatial_edge_index,
        partition_size=partition_size,
    )

    console.print(f"  Train: {len(train_ds)} partitions")
    console.print(f"  Val: {len(val_ds)} partitions")
    console.print(f"  Test: {len(test_ds)} partitions")

    # Compute class weights
    class_weights = compute_class_weights(labels)

    # Load universal GRN
    universal_grn = torch.load(data_path / "universal_grn.pt", weights_only=True)

    # Create model
    console.print("\n[cyan]Creating model...[/cyan]")
    model = HEISTClassifier(
        num_classes=n_classes,
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Parameters: {total_params:,}")

    # Create trainer
    # NOTE: GRN is currently disabled because it's a gene-gene graph (nodes 0-540)
    # but the model expects cell-cell edges. This is a fundamental architecture
    # issue that needs proper redesign. For now, we use spatial graph only.
    trainer = HEISTTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        grn_edge_index=None,  # Disabled until proper GRN integration
        output_dir=output_path,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        class_weights=class_weights,
        device=device,
        use_wandb=not no_wandb,
        wandb_project=wandb_project,
    )

    # Train
    console.print("\n[cyan]Starting training...[/cyan]")
    results = trainer.train()

    console.print(f"\n[green]✓ Training complete![/green]")
    console.print(f"  Best val F1: {results['best_val_f1']:.4f} (epoch {results['best_epoch']})")
    if "test_metrics" in results:
        console.print(f"  Test F1: {results['test_metrics']['f1']:.4f}")
        console.print(f"  Test accuracy: {results['test_metrics']['accuracy']:.4f}")
    console.print(f"\n  Model saved to: {output_path / 'best_model.pt'}")


# =============================================================================
# ClearML Pipeline Commands
# =============================================================================


@main.group(name="clearml-pipeline")
def clearml_pipeline_group() -> None:
    """ClearML Pipeline commands for distributed processing.

    Run spatial transcriptomics pipelines using ClearML for orchestration,
    experiment tracking, and remote execution.
    """
    pass


@clearml_pipeline_group.command(name="run")
@click.option(
    "--dataset-id",
    help="ClearML Dataset ID for input data",
)
@click.option(
    "--local-path",
    type=click.Path(exists=True, path_type=Path),
    help="Local path to Xenium/MERSCOPE data (alternative to dataset-id)",
)
@click.option(
    "--platform",
    type=click.Choice(["auto", "xenium", "merscope"]),
    default="auto",
    help="Platform type",
)
@click.option(
    "--segmenter",
    type=click.Choice(["cellpose", "native"]),
    default="cellpose",
    help="Segmentation method",
)
@click.option(
    "--annotator",
    type=click.Choice(["celltypist", "ground_truth", "popv"]),
    default="celltypist",
    help="Annotation method",
)
@click.option(
    "--ground-truth-file",
    type=click.Path(exists=False, path_type=Path),
    help="Path to ground truth file (for ground_truth annotator). For remote execution, use filename from dataset.",
)
@click.option(
    "--patch-size",
    type=click.Choice(["32", "64", "128", "256"]),
    default="128",
    help="Patch size in pixels",
)
@click.option(
    "--backbone",
    default="efficientnetv2_rw_s",
    help="CNN backbone for training",
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    help="Training epochs",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run locally (without ClearML agents)",
)
@click.option(
    "--project",
    default="DAPIDL/pipelines",
    help="ClearML project name",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Run cross-modal validation after training (Leiden + DAPI + consensus)",
)
@click.option(
    "--extended-consensus",
    is_flag=True,
    help="Use 6 CellTypist models for extended consensus (better coverage)",
)
@click.option(
    "--compare-ground-truth",
    is_flag=True,
    help="Compare CellTypist annotations to ground truth file",
)
@click.option(
    "--gt-comparison-file",
    type=click.Path(exists=False, path_type=Path),
    help="Ground truth file for comparison (Excel/CSV). Different from --ground-truth-file which is for annotation.",
)
@click.option(
    "--fine-grained",
    is_flag=True,
    help="Use fine-grained cell types (~20 classes) instead of broad categories (3 classes)",
)
def run_pipeline(
    dataset_id: str | None,
    local_path: Path | None,
    platform: str,
    segmenter: str,
    annotator: str,
    ground_truth_file: Path | None,
    patch_size: str,
    backbone: str,
    epochs: int,
    local: bool,
    project: str,
    validate: bool,
    extended_consensus: bool,
    compare_ground_truth: bool,
    gt_comparison_file: Path | None,
    fine_grained: bool,
) -> None:
    """Run the DAPIDL ClearML pipeline.

    Process spatial transcriptomics data through segmentation, annotation,
    patch extraction, and model training.

    Examples:

        # Run with ClearML Dataset
        dapidl clearml-pipeline run --dataset-id abc123 --epochs 50

        # Run locally with Xenium data
        dapidl clearml-pipeline run --local-path /path/to/xenium --local

        # Use ground truth annotations
        dapidl clearml-pipeline run --dataset-id abc123 \\
            --annotator ground_truth \\
            --ground-truth-file annotations.xlsx
    """
    from dapidl.pipeline import PipelineConfig, create_pipeline

    if not dataset_id and not local_path:
        console.print("[red]Error: Either --dataset-id or --local-path required[/red]")
        raise click.Abort()

    config = PipelineConfig(
        project=project,
        dataset_id=dataset_id,
        local_path=str(local_path) if local_path else None,
        platform=platform,
        segmenter=segmenter,
        annotator=annotator,
        ground_truth_file=str(ground_truth_file) if ground_truth_file else None,
        patch_size=int(patch_size),
        backbone=backbone,
        epochs=epochs,
        execute_remotely=not local,
        run_validation=validate or compare_ground_truth,  # Enable validation for GT comparison
        extended_consensus=extended_consensus,
        run_ground_truth_comparison=compare_ground_truth,
        gt_comparison_file=str(gt_comparison_file) if gt_comparison_file else None,
        fine_grained=fine_grained,
    )

    console.print("[bold blue]DAPIDL ClearML Pipeline[/bold blue]\n")
    console.print(f"  Platform: {platform}")
    console.print(f"  Segmenter: {segmenter}")
    console.print(f"  Annotator: {annotator}")
    console.print(f"  Fine-grained: {fine_grained}")
    console.print(f"  Extended consensus: {extended_consensus}")
    console.print(f"  Patch size: {patch_size}px")
    console.print(f"  Backbone: {backbone}")
    console.print(f"  Epochs: {epochs}")
    if compare_ground_truth:
        console.print(f"  Ground truth comparison: {gt_comparison_file}")
    console.print(f"  Execution: {'local' if local else 'ClearML agents'}")
    console.print()

    pipeline = create_pipeline(config)

    if local:
        console.print("[cyan]Running pipeline locally...[/cyan]\n")
        results = pipeline.run_locally()
        console.print("\n[green]✓ Pipeline completed![/green]")
        if "training" in results:
            test_metrics = results["training"].get("test_metrics", {})
            console.print(f"  Test F1: {test_metrics.get('f1', 'N/A'):.4f}")
            console.print(f"  Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
    else:
        console.print("[cyan]Creating ClearML pipeline...[/cyan]")
        pipeline.create_pipeline()
        console.print("[cyan]Starting remote execution...[/cyan]")
        pipeline_id = pipeline.run(wait=False)
        console.print(f"\n[green]✓ Pipeline started: {pipeline_id}[/green]")
        console.print("  Monitor at: https://app.clear.ml")


@clearml_pipeline_group.command(name="list-components")
def list_components() -> None:
    """List available pipeline components.

    Shows all registered segmenters and annotators that can be used
    in the pipeline.
    """
    from dapidl.pipeline import list_segmenters, list_annotators

    console.print("[bold blue]Pipeline Components[/bold blue]\n")

    table = Table(title="Segmenters")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    segmenter_desc = {
        "cellpose": "GPU-accelerated nucleus segmentation with Cellpose 2.0",
        "native": "Use platform-provided cell boundaries (pass-through)",
    }

    for name in list_segmenters():
        table.add_row(name, segmenter_desc.get(name, ""))
    console.print(table)

    console.print()

    table = Table(title="Annotators")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    annotator_desc = {
        "celltypist": "Gene expression-based annotation with CellTypist models",
        "ground_truth": "Load annotations from curated Excel/CSV/Parquet files",
        "popv": "Ensemble prediction using HuggingFace Tabula Sapiens models",
    }

    for name in list_annotators():
        table.add_row(name, annotator_desc.get(name, ""))
    console.print(table)


@clearml_pipeline_group.command(name="universal")
@click.option(
    "--tissue",
    "-t",
    multiple=True,
    nargs=4,
    type=(str, str, str, int),
    help="Add tissue: TISSUE DATASET_ID PLATFORM TIER (e.g., 'breast abc123 xenium 1')",
)
@click.option(
    "--tissue-local",
    multiple=True,
    nargs=4,
    type=(str, click.Path(exists=True), str, int),
    help="Add local tissue: TISSUE PATH PLATFORM TIER (e.g., 'lung /data/lung xenium 2')",
)
@click.option(
    "--sampling",
    type=click.Choice(["equal", "proportional", "sqrt"]),
    default="sqrt",
    help="Tissue sampling strategy",
)
@click.option(
    "--backbone",
    default="efficientnetv2_rw_s",
    help="CNN backbone for training",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    help="Training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Training batch size",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run locally (without ClearML agents)",
)
@click.option(
    "--project",
    default="DAPIDL/universal",
    help="ClearML project name",
)
@click.option(
    "--name",
    default="universal-dapi",
    help="Pipeline name",
)
def run_universal_pipeline(
    tissue: tuple,
    tissue_local: tuple,
    sampling: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    local: bool,
    project: str,
    name: str,
) -> None:
    """Run universal cross-tissue DAPI training pipeline.

    Combines multiple tissue datasets for training a universal classifier
    with Cell Ontology standardized labels and tissue-balanced sampling.

    Examples:

        # Train with multiple ClearML datasets
        dapidl clearml-pipeline universal \\
            -t breast abc123 xenium 1 \\
            -t lung def456 xenium 2 \\
            -t liver ghi789 merscope 2

        # Train locally with multiple local datasets
        dapidl clearml-pipeline universal --local \\
            --tissue-local breast /data/breast xenium 1 \\
            --tissue-local lung /data/lung xenium 2

        # Use equal sampling instead of sqrt-balanced
        dapidl clearml-pipeline universal \\
            -t breast abc123 xenium 1 \\
            -t lung def456 xenium 2 \\
            --sampling equal

    Confidence tiers:
        1 = Ground truth labels (highest weight)
        2 = Consensus annotations (medium weight)
        3 = Single predictor (lowest weight)
    """
    from dapidl.pipeline.universal_controller import (
        UniversalPipelineConfig,
        UniversalDAPIPipelineController,
    )

    # Validate inputs
    if not tissue and not tissue_local:
        console.print("[red]Error: At least one tissue dataset required[/red]")
        console.print("[dim]Use --tissue or --tissue-local to add datasets[/dim]")
        raise click.Abort()

    # Build configuration
    config = UniversalPipelineConfig(
        name=name,
        project=project,
        sampling_strategy=sampling,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        execute_remotely=not local,
    )

    # Add ClearML datasets
    for tissue_name, dataset_id, platform, tier in tissue:
        config.add_tissue(
            tissue=tissue_name,
            dataset_id=dataset_id,
            platform=platform,
            confidence_tier=int(tier),
        )

    # Add local datasets
    for tissue_name, path, platform, tier in tissue_local:
        config.add_tissue(
            tissue=tissue_name,
            local_path=str(path),
            platform=platform,
            confidence_tier=int(tier),
        )

    console.print("[bold blue]DAPIDL Universal Cross-Tissue Pipeline[/bold blue]\n")
    console.print(f"  Name: {name}")
    console.print(f"  Sampling: {sampling}")
    console.print(f"  Backbone: {backbone}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Tissues: {len(config.tissues)}")
    for tc in config.tissues:
        tier_label = {1: "ground truth", 2: "consensus", 3: "predicted"}
        console.print(f"    - {tc.tissue}/{tc.platform}: tier {tc.confidence_tier} ({tier_label.get(tc.confidence_tier, 'unknown')})")
    console.print(f"  Execution: {'local' if local else 'ClearML agents'}")
    console.print()

    controller = UniversalDAPIPipelineController(config)

    if local:
        console.print("[cyan]Running universal pipeline locally...[/cyan]\n")
        results = controller.run_locally()

        console.print("\n[green]✓ Universal pipeline completed![/green]")
        if "universal_training" in results:
            train_results = results["universal_training"]
            test_metrics = train_results.get("test_metrics", {})
            console.print(f"  Test F1 (fine): {test_metrics.get('f1_fine', 'N/A'):.4f}")
            console.print(f"  Test F1 (coarse): {test_metrics.get('f1_coarse', 'N/A'):.4f}")
            console.print(f"  Model: {train_results.get('model_path', 'N/A')}")

            # Per-tissue metrics
            tissue_metrics = train_results.get("tissue_metrics", {})
            if tissue_metrics:
                console.print("\n  Per-tissue F1 (fine):")
                for tissue_key, metrics in tissue_metrics.items():
                    console.print(f"    {tissue_key}: {metrics.get('f1_fine', 'N/A'):.4f}")
    else:
        console.print("[cyan]Creating ClearML universal pipeline...[/cyan]")
        controller.create_pipeline()
        console.print("[cyan]Starting remote execution...[/cyan]")
        pipeline_id = controller.run(wait=False)
        console.print(f"\n[green]✓ Universal pipeline started: {pipeline_id}[/green]")
        console.print("  Monitor at: https://app.clear.ml")


@clearml_pipeline_group.command(name="enhanced")
@click.option(
    "--raw-dataset-id",
    default=None,
    help="ClearML Dataset ID for raw data",
)
@click.option(
    "--s3-uri",
    default=None,
    help="S3 URI for raw data (e.g., s3://dapidl/raw-data/xenium-breast/)",
)
@click.option(
    "--platform",
    type=click.Choice(["auto", "xenium", "merscope"]),
    default="auto",
    help="Platform type (auto-detected if not specified)",
)
@click.option(
    "--celltypist-models",
    multiple=True,
    default=["Cells_Adult_Breast.pkl", "Immune_All_High.pkl"],
    help="CellTypist models for ensemble annotation (can specify multiple)",
)
@click.option(
    "--include-singler/--no-singler",
    default=True,
    help="Include SingleR in ensemble annotation",
)
@click.option(
    "--patch-sizes",
    multiple=True,
    type=int,
    default=[128],
    help="Patch sizes to generate (can specify multiple: --patch-sizes 64 --patch-sizes 128)",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    help="Training epochs",
)
@click.option(
    "--backbone",
    type=click.Choice(["efficientnetv2_rw_s", "resnet50", "convnext_tiny"]),
    default="efficientnetv2_rw_s",
    help="CNN backbone architecture",
)
@click.option(
    "--training-mode",
    type=click.Choice(["hierarchical", "flat"]),
    default="hierarchical",
    help="Training mode (hierarchical uses curriculum learning)",
)
@click.option(
    "--coarse-only-epochs",
    type=int,
    default=20,
    help="Epochs for Phase 1 (coarse only) in hierarchical mode",
)
@click.option(
    "--coarse-medium-epochs",
    type=int,
    default=50,
    help="Epochs for Phase 2 (coarse + medium) in hierarchical mode",
)
@click.option(
    "--local/--remote",
    default=False,
    help="Run locally or on ClearML agents",
)
@click.option(
    "--upload-to-s3/--no-s3",
    default=True,
    help="Upload datasets and models to S3",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./enhanced_pipeline_output",
    help="Local output directory",
)
def run_enhanced_pipeline(
    raw_dataset_id: str | None,
    s3_uri: str | None,
    platform: str,
    celltypist_models: tuple,
    include_singler: bool,
    patch_sizes: tuple,
    epochs: int,
    backbone: str,
    training_mode: str,
    coarse_only_epochs: int,
    coarse_medium_epochs: int,
    local: bool,
    upload_to_s3: bool,
    output_dir: str,
) -> None:
    """Run enhanced DAPIDL pipeline with GUI-configurable parameters.

    This pipeline features:
    - Ensemble annotation with multiple methods (CellTypist, SingleR)
    - Multi-patch-size LMDB creation
    - Dataset lineage for space efficiency
    - Smart step skipping when outputs exist

    Examples:

        # Remote execution on ClearML
        dapidl clearml-pipeline enhanced --raw-dataset-id abc123 --epochs 100

        # Local execution with S3 data
        dapidl clearml-pipeline enhanced --s3-uri s3://dapidl/raw-data/xenium-breast/ --local

        # Multiple patch sizes and CellTypist models
        dapidl clearml-pipeline enhanced \\
            --raw-dataset-id abc123 \\
            --celltypist-models Cells_Adult_Breast.pkl \\
            --celltypist-models Immune_All_High.pkl \\
            --patch-sizes 64 --patch-sizes 128 --patch-sizes 256 \\
            --local
    """
    from dapidl.pipeline import GUIPipelineConfig, EnhancedDAPIDLPipelineController

    # Build configuration
    config = GUIPipelineConfig(
        raw_dataset_id=raw_dataset_id,
        s3_data_uri=s3_uri,
        platform=platform,
        celltypist_models=list(celltypist_models),
        include_singler=include_singler,
        patch_sizes=list(patch_sizes),
        epochs=epochs,
        backbone=backbone,
        training_mode=training_mode,
        coarse_only_epochs=coarse_only_epochs,
        coarse_medium_epochs=coarse_medium_epochs,
        upload_to_s3=upload_to_s3,
        output_dir=output_dir,
    )

    controller = EnhancedDAPIDLPipelineController(config)

    console.print("[bold blue]Enhanced DAPIDL Pipeline[/bold blue]\n")
    console.print(f"Platform: {platform}")
    console.print(f"Annotation: Ensemble with {len(celltypist_models)} CellTypist models")
    if include_singler:
        console.print("  + SingleR (blueprint reference)")
    console.print(f"Patch sizes: {list(patch_sizes)}")
    console.print(f"Training: {training_mode} mode, {epochs} epochs")
    console.print(f"Backbone: {backbone}")
    console.print()

    if local:
        console.print("[cyan]Running locally...[/cyan]\n")
        result = controller.run_locally()

        if result.success:
            console.print("\n[green]✓ Pipeline complete![/green]")
            console.print(f"  Model: {result.model_path}")
            if result.training_metrics:
                console.print("  Test metrics:")
                for key, value in result.training_metrics.items():
                    if isinstance(value, float):
                        console.print(f"    {key}: {value:.4f}")
        else:
            console.print(f"\n[red]✗ Pipeline failed: {result.error}[/red]")
            raise click.Abort()
    else:
        console.print("[cyan]Creating ClearML enhanced pipeline...[/cyan]")
        controller.create_pipeline()
        console.print("[cyan]Starting remote execution...[/cyan]")
        pipeline_id = controller.run(wait=False)
        console.print(f"\n[green]✓ Enhanced pipeline started: {pipeline_id}[/green]")
        console.print("  Monitor at: https://app.clear.ml")


@clearml_pipeline_group.command(name="create-base-tasks")
@click.option(
    "--project",
    default="DAPIDL/pipelines",
    help="ClearML project name",
)
@click.option(
    "--include-universal",
    is_flag=True,
    help="Include universal training step tasks",
)
def create_base_tasks(project: str, include_universal: bool) -> None:
    """Create ClearML base tasks for pipeline steps.

    This registers each pipeline step as a ClearML Task, which is required
    before running the pipeline remotely. Only needs to be run once.
    """
    from dapidl.pipeline import PipelineConfig, create_pipeline

    console.print("[bold blue]Creating ClearML Base Tasks[/bold blue]\n")

    config = PipelineConfig(project=project)
    pipeline = create_pipeline(config)

    console.print("[cyan]Registering pipeline steps...[/cyan]")
    pipeline.create_base_tasks()

    console.print("\n[green]✓ Base tasks created![/green]")
    console.print(f"  Project: {project}")
    console.print("  You can now run the pipeline with: dapidl clearml-pipeline run")


# =============================================================================
# PopV Commands
# =============================================================================


@main.group(name="popv")
def popv_group() -> None:
    """PopV annotation commands for universal cell type prediction.

    PopV uses 8+ annotation methods with ontology-based voting for
    high-confidence, tissue-agnostic cell type annotation.
    """
    pass


@popv_group.command(name="annotate")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to expression data (H5AD, H5, or directory)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for annotation results",
)
@click.option(
    "--organ",
    type=click.Choice([
        "auto", "Mammary", "Lung", "Liver", "Heart", "Kidney", "Brain",
        "Pancreas", "Spleen", "Small_Intestine", "Large_Intestine",
        "Bladder", "Eye", "Skin", "Fat", "Prostate", "Uterus",
        "Muscle", "Bone_Marrow", "Blood", "Thymus"
    ]),
    default="auto",
    show_default=True,
    help="Organ-specific Tabula Sapiens reference (auto-detected by default)",
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "inference", "retrain"]),
    default="fast",
    show_default=True,
    help="Annotation mode: fast (5min), inference (30min), retrain (1hr)",
)
@click.option(
    "--min-consensus",
    type=int,
    default=6,
    show_default=True,
    help="Minimum methods agreeing (0-8). 6=90%%, 7=95%% accuracy",
)
@click.option(
    "--n-top-genes",
    type=int,
    default=2000,
    show_default=True,
    help="Number of highly variable genes to use",
)
@click.option(
    "--fine-grained/--coarse",
    default=True,
    help="Output fine-grained cell types (default) or coarse categories",
)
@click.option(
    "--include-methods/--no-methods",
    default=True,
    help="Include per-method prediction columns in output",
)
@click.option(
    "--wandb/--no-wandb",
    default=False,
    help="Log to Weights & Biases",
)
@click.option(
    "--clearml/--no-clearml",
    default=False,
    help="Log to ClearML",
)
@click.option(
    "--project",
    default="dapidl-popv",
    help="Project name for WandB/ClearML logging",
)
def popv_annotate(
    input_path: Path,
    output: Path,
    organ: str,
    mode: str,
    min_consensus: int,
    n_top_genes: int,
    fine_grained: bool,
    include_methods: bool,
    wandb: bool,
    clearml: bool,
    project: str,
) -> None:
    """Annotate cells using popV ensemble prediction.

    PopV combines 8+ annotation methods with Tabula Sapiens reference
    for universal, high-confidence cell type prediction.

    Methods include: Random Forest, SVM, XGBoost, CellTypist, OnClass,
    scVI+kNN, scANVI, BBKNN+kNN, Scanorama+kNN, Harmony+kNN.

    Examples:

        # Quick annotation with auto organ detection
        dapidl popv annotate -i expression.h5ad -o ./results

        # Breast-specific annotation with high confidence filter
        dapidl popv annotate -i expression.h5ad -o ./results \\
            --organ Mammary --min-consensus 7

        # Full retraining mode with WandB logging
        dapidl popv annotate -i expression.h5ad -o ./results \\
            --mode retrain --wandb --project my-project
    """
    # Check popV availability
    try:
        from dapidl.pipeline.components.annotators.popv import (
            annotate_with_popv,
            is_popv_available,
        )
    except ImportError:
        console.print("[red]Error: popV annotator module not found[/red]")
        console.print("[yellow]This may be an installation issue.[/yellow]")
        raise click.Abort()

    if not is_popv_available():
        console.print("[red]Error: popV package is not installed[/red]")
        console.print("[yellow]Install with: pip install popv[/yellow]")
        raise click.Abort()

    console.print("[bold blue]DAPIDL PopV Annotation[/bold blue]\n")
    console.print(f"Input: {input_path}")
    console.print(f"Output: {output}")
    console.print(f"Organ: {organ}")
    console.print(f"Mode: {mode}")
    console.print(f"Min consensus: {min_consensus}/8")
    console.print(f"HVGs: {n_top_genes}")
    console.print(f"Fine-grained: {fine_grained}")
    console.print(f"Include methods: {include_methods}")
    if wandb:
        console.print(f"WandB project: {project}")
    if clearml:
        console.print(f"ClearML project: {project}")
    console.print()

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Running popV annotation...[/cyan]")
    console.print("[dim]This may take 5-60 minutes depending on mode and data size[/dim]\n")

    try:
        result = annotate_with_popv(
            expression_path=input_path,
            output_path=output,
            organ=organ,
            mode=mode,
            min_consensus=min_consensus,
            use_wandb=wandb,
            use_clearml=clearml,
            project_name=project,
        )

        console.print(f"\n[green]✓ Annotation complete![/green]")
        console.print(f"  Cells annotated: {result.stats.get('n_cells', 'N/A')}")
        console.print(f"  High-confidence cells: {result.stats.get('n_high_confidence', 'N/A')}")
        console.print(f"  Cell types: {len(result.class_mapping)}")
        console.print(f"  Output: {output}")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise click.Abort()


@popv_group.command(name="list-references")
def popv_list_references() -> None:
    """List available Tabula Sapiens organ references.

    Shows all pre-trained popV models available from Tabula Sapiens.
    These references cover 20+ human organs without manual tuning.
    """
    console.print("[bold blue]Available PopV References (Tabula Sapiens)[/bold blue]\n")

    # Tabula Sapiens organs with popV models
    organs = {
        "Mammary": "Breast/mammary tissue (recommended for breast cancer)",
        "Lung": "Lung tissue",
        "Liver": "Liver tissue",
        "Heart": "Heart/cardiac tissue",
        "Kidney": "Kidney tissue",
        "Brain": "Brain tissue",
        "Pancreas": "Pancreatic tissue",
        "Spleen": "Spleen tissue",
        "Small_Intestine": "Small intestine",
        "Large_Intestine": "Large intestine/colon",
        "Bladder": "Bladder tissue",
        "Eye": "Eye/ocular tissue",
        "Skin": "Skin tissue",
        "Fat": "Adipose tissue",
        "Prostate": "Prostate tissue",
        "Uterus": "Uterine tissue",
        "Muscle": "Muscle tissue",
        "Bone_Marrow": "Bone marrow",
        "Blood": "Blood/PBMCs",
        "Thymus": "Thymus tissue",
    }

    table = Table(title="Organ References")
    table.add_column("Organ", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for organ, desc in organs.items():
        table.add_row(organ, desc)

    console.print(table)
    console.print(f"\n[green]Total: {len(organs)} organ references[/green]")
    console.print("\n[dim]Use --organ <name> with 'dapidl popv annotate'[/dim]")
    console.print("[dim]Use --organ auto for automatic tissue detection[/dim]")


@popv_group.command(name="info")
def popv_info() -> None:
    """Show popV package information and availability.

    Displays popV installation status, version, and available methods.
    """
    console.print("[bold blue]PopV Package Information[/bold blue]\n")

    # Check installation
    try:
        import popv
        console.print(f"[green]✓ popV installed[/green]: version {popv.__version__}")
    except ImportError:
        console.print("[red]✗ popV not installed[/red]")
        console.print("  Install with: [cyan]pip install popv[/cyan]")
        console.print("  Or: [cyan]uv add popv[/cyan]")
        return

    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    deps = {
        "scvi-tools": "scVI/scANVI methods",
        "celltypist": "CellTypist method",
        "scanpy": "Data processing",
        "anndata": "Data format",
    }

    for pkg, desc in deps.items():
        try:
            mod = __import__(pkg.replace("-", "_"))
            version = getattr(mod, "__version__", "unknown")
            console.print(f"  [green]✓[/green] {pkg}: {version} ({desc})")
        except ImportError:
            console.print(f"  [yellow]✗[/yellow] {pkg}: not installed ({desc})")

    # Show available methods
    console.print("\n[bold]Available Methods:[/bold]")
    methods = [
        ("Random Forest", "Classical ML classifier"),
        ("SVM", "Support Vector Machine"),
        ("XGBoost", "Gradient boosting"),
        ("CellTypist", "Logistic regression"),
        ("OnClass", "Cell Ontology-aware"),
        ("scVI + kNN", "VAE embedding + neighbors"),
        ("scANVI", "Semi-supervised VAE"),
        ("BBKNN + kNN", "Batch-balanced neighbors"),
        ("Scanorama + kNN", "MNN integration"),
        ("Harmony + kNN", "Harmony integration"),
    ]

    for method, desc in methods:
        console.print(f"  • {method}: {desc}")

    console.print("\n[bold]Consensus Score Interpretation:[/bold]")
    console.print("  8/8: 98% accuracy (perfect agreement)")
    console.print("  7/8: 95% accuracy (near-perfect)")
    console.print("  6/8: 90% accuracy (strong)")
    console.print("  5/8: 80% accuracy (moderate)")
    console.print("  ≤4/8: <65% accuracy (low confidence)")


if __name__ == "__main__":
    main()
