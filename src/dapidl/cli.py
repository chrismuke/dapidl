"""DAPIDL Command Line Interface."""

import sys

import click
from pathlib import Path
from typing import Tuple
from rich.console import Console
from rich.table import Table

console = Console()


def _get_clearml_web_url() -> str:
    """Get the ClearML web server URL from active config."""
    try:
        from clearml.backend_api import Session
        host = Session.get_api_server_host()
        return host.replace("api.", "app.").replace("://api", "://app")
    except Exception:
        return "https://app.clear.ml"


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


@main.command(name="train-multi")
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    nargs=5,
    type=(click.Path(exists=True), str, str, int, float),
    required=True,
    help="Dataset spec: PATH TISSUE PLATFORM CONFIDENCE WEIGHT (e.g., '/lmdb/breast breast xenium 2 1.0')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for model",
)
@click.option(
    "--sampling",
    type=click.Choice(["equal", "proportional", "sqrt"]),
    default="sqrt",
    show_default=True,
    help="Tissue sampling strategy",
)
@click.option(
    "--standardize-labels/--no-standardize-labels",
    default=True,
    help="Use Cell Ontology label standardization",
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    show_default=True,
    help="Training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Batch size",
)
@click.option(
    "--backbone",
    default="efficientnetv2_rw_s",
    show_default=True,
    help="CNN backbone architecture",
)
@click.option(
    "--lr",
    type=float,
    default=3e-4,
    show_default=True,
    help="Learning rate",
)
@click.option(
    "--max-weight-ratio",
    type=float,
    default=10.0,
    show_default=True,
    help="Max class weight ratio to prevent mode collapse",
)
@click.option(
    "--wandb/--no-wandb",
    default=True,
    help="Enable Weights & Biases logging",
)
def train_multi(
    dataset: tuple,
    output: Path,
    sampling: str,
    standardize_labels: bool,
    epochs: int,
    batch_size: int,
    backbone: str,
    lr: float,
    max_weight_ratio: float,
    wandb: bool,
) -> None:
    """Train on multiple LMDB datasets with runtime combination.

    Combines multiple LMDB datasets at training time without creating a merged file.
    Supports tissue-balanced sampling, confidence weighting, and Cell Ontology
    label standardization.

    Examples:

        # Train on two datasets with equal sampling
        dapidl train-multi \\
            -d ./xenium-breast breast xenium 2 1.0 \\
            -d ./merscope-liver liver merscope 2 0.8 \\
            --sampling equal \\
            -o ./model

        # Train with confidence weighting (higher tier = more trusted)
        dapidl train-multi \\
            -d ./ground-truth breast xenium 1 1.0 \\  # tier 1 = ground truth
            -d ./consensus lung xenium 2 1.0 \\       # tier 2 = consensus
            -o ./model

    Confidence tiers:
        1 = Ground truth (full weight)
        2 = Consensus annotation (80% weight)
        3 = Predicted/uncertain (50% weight)
    """
    from dapidl.data.multi_tissue_dataset import (
        MultiTissueConfig,
        create_multi_tissue_splits,
    )
    from dapidl.training.trainer import Trainer

    console.print("[bold blue]DAPIDL Multi-Dataset Training[/bold blue]\n")

    # Build multi-tissue config
    config = MultiTissueConfig(
        sampling_strategy=sampling,
        standardize_labels=standardize_labels,
    )

    for path, tissue, platform, confidence, weight in dataset:
        config.add_dataset(
            path=path,
            tissue=tissue,
            platform=platform,
            confidence_tier=int(confidence),
            weight_multiplier=float(weight),
        )
        console.print(f"  + {tissue} ({platform}): {path}")
        console.print(f"    Confidence tier: {confidence}, Weight: {weight}")

    console.print(f"\n  Sampling: {sampling}")
    console.print(f"  Label standardization: {standardize_labels}")
    console.print(f"  Backbone: {backbone}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print()

    # Create train/val/test splits
    console.print("[cyan]Creating data splits...[/cyan]")
    train_ds, val_ds, test_ds = create_multi_tissue_splits(
        config,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by="both",  # Stratify by tissue and label
    )

    console.print(f"  Train: {len(train_ds)} samples")
    console.print(f"  Val: {len(val_ds)} samples")
    console.print(f"  Test: {len(test_ds)} samples")
    console.print(f"  Classes: {train_ds.num_classes}")

    # Create trainer with multi-tissue dataset
    console.print("\n[cyan]Creating trainer...[/cyan]")
    trainer = Trainer(
        data_path=None,  # Not used for multi-tissue
        multi_tissue_config=config,
        output_path=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        use_wandb=wandb,
        backbone_name=backbone,
        max_weight_ratio=max_weight_ratio,
    )

    # Train
    console.print("[cyan]Starting training...[/cyan]\n")
    trainer.train()

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Model saved to: {output / 'best_model.pt'}")


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


@main.command()
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint (trained on source platform)",
)
@click.option(
    "--target-data",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to target platform LMDB/Zarr dataset for adaptation",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for adapted model (default: <checkpoint>_adapted.pt)",
)
@click.option(
    "--num-batches",
    "-n",
    type=int,
    default=20,
    show_default=True,
    help="Number of batches for BN statistics adaptation",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=64,
    show_default=True,
    help="Batch size for adaptation data loader",
)
@click.option(
    "--compute-metrics/--no-compute-metrics",
    default=True,
    help="Compute domain shift metrics before/after adaptation",
)
def adapt(
    checkpoint: Path,
    target_data: Path,
    output: Path | None,
    num_batches: int,
    batch_size: int,
    compute_metrics: bool,
) -> None:
    """Adapt trained model to a new platform using AdaBN.

    Uses Adaptive Batch Normalization (AdaBN) to update the BatchNorm
    layer statistics on target domain data without retraining. This is
    essential for cross-platform transfer (e.g., Xenium → MERSCOPE).

    The adaptation process:
      1. Loads the checkpoint trained on source platform
      2. Runs forward passes on target data to update BN statistics
      3. Saves the adapted model

    Expected improvement: 5-15% accuracy gain on target platform.

    Examples:
        # Adapt Xenium-trained model to MERSCOPE
        dapidl adapt -c xenium_model.pt -t merscope-data/ -o adapted.pt

        # Quick adaptation with fewer batches
        dapidl adapt -c model.pt -t target/ -n 10

        # Adapt with domain shift analysis
        dapidl adapt -c model.pt -t target/ --compute-metrics
    """
    import torch
    from dapidl.models import (
        adapt_batch_norm,
        create_adaptation_loader,
        compute_domain_shift_metrics,
    )

    console.print("[bold blue]DAPIDL Domain Adaptation (AdaBN)[/bold blue]")
    console.print(f"Source checkpoint: {checkpoint}")
    console.print(f"Target data: {target_data}")
    console.print(f"Num batches: {num_batches}")
    console.print(f"Batch size: {batch_size}")

    # Determine output path
    if output is None:
        output = checkpoint.parent / f"{checkpoint.stem}_adapted.pt"
    console.print(f"Output: {output}")

    # Load model - detect model type from checkpoint
    console.print("\n[yellow]Loading model...[/yellow]")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint to detect model type
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint_data.get("model_type")
    hparams = checkpoint_data.get("hparams", {})

    # Detect model type from hparams if not explicitly set
    if model_type is None:
        if "num_coarse" in hparams and "num_medium" in hparams:
            model_type = "HierarchicalClassifier"
        elif "seg_hidden_dim" in hparams:
            model_type = "MultiTaskClassifier"
        else:
            model_type = "CellTypeClassifier"

    console.print(f"  Model type: {model_type}")

    # Load appropriate model class
    if model_type == "HierarchicalClassifier":
        from dapidl.models.hierarchical import HierarchicalClassifier
        model = HierarchicalClassifier.from_checkpoint(str(checkpoint))
    elif model_type == "MultiTaskClassifier":
        from dapidl.models.multitask import MultiTaskClassifier
        model = MultiTaskClassifier.from_checkpoint(str(checkpoint))
    else:
        from dapidl.models.classifier import CellTypeClassifier
        model = CellTypeClassifier.from_checkpoint(str(checkpoint))

    model = model.to(device)
    console.print(f"  Device: {device}")
    console.print(f"  Classes: {getattr(model, 'num_classes', 'N/A')}")

    # Create adaptation data loader
    console.print("[yellow]Creating adaptation loader...[/yellow]")
    target_loader = create_adaptation_loader(
        data_path=target_data,
        batch_size=batch_size,
    )
    console.print(f"  Loaded target dataset from {target_data}")

    # Optionally compute domain shift metrics before adaptation
    if compute_metrics:
        console.print("\n[yellow]Computing domain shift metrics (before)...[/yellow]")
        metrics_before = compute_domain_shift_metrics(
            model, target_loader, device=device, num_batches=5
        )
        console.print(f"  Feature mean: {metrics_before['feature_mean']:.4f}")
        console.print(f"  Feature std: {metrics_before['feature_std']:.4f}")

    # Perform AdaBN adaptation
    console.print("\n[yellow]Adapting BatchNorm statistics...[/yellow]")
    model = adapt_batch_norm(
        model=model,
        target_loader=target_loader,
        num_batches=num_batches,
        device=device,
    )
    console.print(f"  Adapted using {num_batches} batches")

    # Optionally compute domain shift metrics after adaptation
    if compute_metrics:
        console.print("\n[yellow]Computing domain shift metrics (after)...[/yellow]")
        # Recreate loader since we consumed it
        target_loader = create_adaptation_loader(
            data_path=target_data,
            batch_size=batch_size,
        )
        metrics_after = compute_domain_shift_metrics(
            model, target_loader, device=device, num_batches=5
        )
        console.print(f"  Feature mean: {metrics_after['feature_mean']:.4f}")
        console.print(f"  Feature std: {metrics_after['feature_std']:.4f}")

        # Show comparison
        console.print("\n[cyan]Domain Shift Analysis:[/cyan]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Before", style="yellow")
        table.add_column("After", style="green")
        table.add_row(
            "Feature Mean",
            f"{metrics_before['feature_mean']:.4f}",
            f"{metrics_after['feature_mean']:.4f}",
        )
        table.add_row(
            "Feature Std",
            f"{metrics_before['feature_std']:.4f}",
            f"{metrics_after['feature_std']:.4f}",
        )
        console.print(table)

    # Save adapted model
    console.print(f"\n[yellow]Saving adapted model to {output}...[/yellow]")
    # Save the full checkpoint with adapted state dict
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_data["model_state_dict"] = model.state_dict()
    checkpoint_data["adapted"] = True
    checkpoint_data["adaptation_batches"] = num_batches
    checkpoint_data["target_data"] = str(target_data)
    torch.save(checkpoint_data, output)

    console.print(f"\n[bold green]Adaptation complete![/bold green]")
    console.print(f"Adapted model saved to: {output}")
    console.print("\n[dim]Use the adapted model for inference on target platform data.[/dim]")


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


@main.command(name="clean-dataset")
@click.option(
    "-d", "--data",
    "data_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to dataset directory (with patches.zarr)",
)
@click.option(
    "-o", "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output path for cleaned dataset (default: <data>_cleaned)",
)
@click.option(
    "--coherence-threshold",
    type=float,
    default=0.20,
    show_default=True,
    help="Minimum neighbor agreement rate (0-1). Cells below this are filtered.",
)
@click.option(
    "-k", "--neighbors",
    type=int,
    default=20,
    show_default=True,
    help="Number of neighbors for coherence computation",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Only compute stats, don't write filtered dataset",
)
def clean_dataset(
    data_path: str,
    output_path: str | None,
    coherence_threshold: float,
    neighbors: int,
    dry_run: bool,
) -> None:
    """Apply spatial consistency filtering to remove noisy annotations.

    Filters cells whose labels disagree with their spatial neighbors.
    This removes likely segmentation artifacts and annotation errors.

    Cells of certain types (dendritic cells, mast cells, etc.) are exempt
    from filtering since they naturally appear in isolation.

    Example:
        dapidl clean-dataset -d ./dataset --coherence-threshold 0.2

        # Dry run to see statistics only
        dapidl clean-dataset -d ./dataset --dry-run
    """
    import json
    import numpy as np
    import polars as pl

    from dapidl.data.cleaning import (
        compute_spatial_coherence,
        filter_spatially_inconsistent,
        clean_dataset_spatial,
    )

    console.print("[bold blue]Spatial Consistency Filtering[/bold blue]")
    console.print(f"Dataset: {data_path}")
    console.print(f"Coherence threshold: {coherence_threshold:.0%}")
    console.print(f"k-neighbors: {neighbors}")

    data_path_obj = Path(data_path)

    if dry_run:
        # Just compute and display stats
        console.print("\n[yellow]Dry run - computing statistics only[/yellow]\n")

        # Load data
        metadata = pl.read_parquet(data_path_obj / "metadata.parquet")
        labels = np.load(data_path_obj / "labels.npy")
        with open(data_path_obj / "class_mapping.json") as f:
            class_mapping = json.load(f)
        label_to_name = {v: k for k, v in class_mapping.items()}

        # Find coordinate columns
        x_col = y_col = None
        for col in metadata.columns:
            if "x" in col.lower() and "centroid" in col.lower():
                x_col = col
            if "y" in col.lower() and "centroid" in col.lower():
                y_col = col
        if x_col is None:
            for col in metadata.columns:
                if col.lower() in ("x", "x_location"):
                    x_col = col
                if col.lower() in ("y", "y_location"):
                    y_col = col

        if x_col is None or y_col is None:
            console.print(f"[red]Error: Could not find coordinate columns[/red]")
            console.print(f"Available columns: {metadata.columns}")
            raise click.Abort()

        coordinates = np.column_stack([
            metadata[x_col].to_numpy(),
            metadata[y_col].to_numpy(),
        ])

        # Compute stats
        keep_mask, stats = filter_spatially_inconsistent(
            metadata=metadata,
            cell_coordinates=coordinates,
            cell_labels=labels,
            coherence_threshold=coherence_threshold,
            k=neighbors,
            label_to_name=label_to_name,
        )

        # Display results
        console.print(f"\n[bold]Overall Statistics:[/bold]")
        console.print(f"  Original cells: {stats['n_cells_original']:,}")
        console.print(f"  Would keep:     {stats['n_cells_kept']:,}")
        console.print(f"  Would filter:   {stats['n_cells_filtered']:,} ({stats['filter_rate']*100:.1f}%)")
        console.print(f"  Exempt cells:   {stats['n_cells_exempt']:,}")

        console.print(f"\n[bold]Coherence Distribution:[/bold]")
        p = stats["coherence_percentiles"]
        console.print(f"  p5:  {p['p5']:.2f}  p25: {p['p25']:.2f}  p50: {p['p50']:.2f}  p75: {p['p75']:.2f}  p95: {p['p95']:.2f}")

        console.print(f"\n[bold]Per-Class Breakdown:[/bold]")
        per_class = stats.get("per_class", {})
        # Sort by filter rate descending
        sorted_classes = sorted(per_class.items(), key=lambda x: 1 - x[1]["keep_rate"])
        for class_name, cs in sorted_classes[:10]:
            keep_pct = cs["keep_rate"] * 100
            console.print(
                f"  {class_name:<25} {cs['kept']:>6}/{cs['original']:<6} "
                f"({keep_pct:5.1f}% kept, coherence={cs['mean_coherence']:.2f})"
            )
        if len(sorted_classes) > 10:
            console.print(f"  ... and {len(sorted_classes) - 10} more classes")

    else:
        # Actually filter the dataset
        try:
            stats = clean_dataset_spatial(
                data_path=data_path_obj,
                output_path=Path(output_path) if output_path else None,
                coherence_threshold=coherence_threshold,
                k=neighbors,
            )

            output = output_path if output_path else f"{data_path}_cleaned"
            console.print(f"\n[green]✓ Dataset cleaned![/green]")
            console.print(f"  Original:  {stats['n_cells_original']:,} cells")
            console.print(f"  Cleaned:   {stats['n_cells_kept']:,} cells")
            console.print(f"  Filtered:  {stats['n_cells_filtered']:,} ({stats['filter_rate']*100:.1f}%)")
            console.print(f"  Output:    {output}")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
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
    "--tissue", "-t",
    multiple=True,
    nargs=4,
    type=(str, str, str, int),
    help="Add dataset: TISSUE SOURCE PLATFORM TIER. SOURCE is ClearML dataset ID or local path.",
)
@click.option(
    "--sampling",
    type=click.Choice(["equal", "proportional", "sqrt"]),
    default="sqrt",
    help="Tissue sampling strategy",
)
@click.option(
    "--segmenter",
    type=click.Choice(["cellpose", "native"]),
    default="native",
    help="Segmentation method",
)
@click.option(
    "--annotator",
    type=click.Choice(["celltypist", "ground_truth", "popv"]),
    default="celltypist",
    help="Annotation method",
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
    default="DAPIDL/pipelines",
    help="ClearML project name",
)
@click.option(
    "--skip-training",
    is_flag=True,
    help="Skip training step (prepare-only mode for creating LMDB datasets)",
)
@click.option(
    "--fine-grained",
    is_flag=True,
    help="Use fine-grained cell types (~20 classes) instead of broad categories (3 classes)",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Run cross-modal validation after training",
)
@click.option(
    "--ground-truth-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to ground truth annotations file (for recipe=gt)",
)
@click.option(
    "--recipe",
    type=str,
    default="default",
    help="Processing recipe (built-in: default, gt, no_cl, annotate_only; or custom from configs/recipes.yaml)",
)
@click.option(
    "--orchestrator/--legacy",
    default=False,
    help="Use task-based orchestrator (supports per-dataset recipes) or legacy PipelineController",
)
@click.option(
    "--gpu-queue",
    default="gpu",
    help="ClearML queue for GPU steps (e.g., gpu-local, gpu-cloud)",
)
@click.option(
    "--default-queue",
    default="default",
    help="ClearML queue for CPU steps (e.g., cpu-local)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable step caching (force all steps to re-run)",
)
def run_pipeline(
    tissue: tuple,
    sampling: str,
    segmenter: str,
    annotator: str,
    patch_size: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    local: bool,
    project: str,
    skip_training: bool,
    fine_grained: bool,
    validate: bool,
    ground_truth_file: str | None,
    recipe: str,
    orchestrator: bool,
    gpu_queue: str,
    default_queue: str,
    no_cache: bool,
) -> None:
    """Run the unified DAPIDL pipeline (supports 1-N datasets).

    Each dataset is specified with -t: TISSUE SOURCE PLATFORM TIER.
    SOURCE can be a ClearML dataset ID or a local path.

    Examples:

        # Single dataset (N=1)
        dapidl clearml-pipeline run -t lung bf8f913f xenium 2 --local --epochs 10

        # Multiple datasets (N=2)
        dapidl clearml-pipeline run \\
            -t lung bf8f913f xenium 2 \\
            -t heart 482be038 xenium 2 \\
            --epochs 50 --sampling sqrt

        # Local path instead of dataset ID
        dapidl clearml-pipeline run -t breast /data/xenium/breast xenium 1 --local

        # Prepare-only (no training)
        dapidl clearml-pipeline run -t lung bf8f913f xenium 2 --skip-training --local

        # Ground truth recipe with GT file
        dapidl clearml-pipeline run -t breast eedc831b xenium 1 \\
            --recipe gt --ground-truth-file /path/to/ground_truth.xlsx --orchestrator

    Confidence tiers:
        1 = Ground truth labels (highest weight)
        2 = Consensus annotations (medium weight)
        3 = Single predictor (lowest weight)
    """
    from dapidl.pipeline.unified_config import (
        AnnotationConfig,
        BackboneType,
        DAPIDLPipelineConfig,
        ExecutionConfig,
        LMDBConfig,
        Platform,
        SamplingStrategy,
        SegmenterType,
        SegmentationConfig,
        TrainingConfig,
        ValidationConfig,
    )
    from dapidl.pipeline.unified_controller import UnifiedPipelineController

    if not tissue:
        console.print("[red]Error: At least one -t/--tissue required[/red]")
        console.print("[dim]Usage: -t TISSUE SOURCE PLATFORM TIER[/dim]")
        console.print("[dim]Example: -t lung bf8f913f xenium 2[/dim]")
        raise click.Abort()

    # Build unified config
    config = DAPIDLPipelineConfig(
        project=project,
        training=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            backbone=BackboneType(backbone),
            sampling_strategy=SamplingStrategy(sampling),
        ),
        segmentation=SegmentationConfig(segmenter=SegmenterType(segmenter)),
        annotation=AnnotationConfig(
            fine_grained=fine_grained,
            ground_truth_file=ground_truth_file,
        ),
        lmdb=LMDBConfig(patch_sizes=[int(patch_size)]),
        execution=ExecutionConfig(execute_remotely=not local, skip_training=skip_training, gpu_queue=gpu_queue, default_queue=default_queue, cache_data_steps=not no_cache),
        validation=ValidationConfig(enabled=validate),
    )

    # Auto-enable orchestrator for non-default recipes
    if recipe != "default" and not orchestrator:
        orchestrator = True

    # Add each tissue dataset
    for tissue_name, source, platform, tier in tissue:
        # Auto-detect: if source is a path that exists, use as local_path
        source_path = Path(source)
        if source_path.exists():
            config.input.add_tissue(
                tissue=tissue_name,
                local_path=str(source_path),
                platform=Platform(platform),
                confidence_tier=int(tier),
            )
        else:
            config.input.add_tissue(
                tissue=tissue_name,
                dataset_id=source,
                platform=Platform(platform),
                confidence_tier=int(tier),
            )

    # Apply recipe to all datasets added via -t
    if recipe != "default":
        for tc in config.input.tissues:
            tc.recipe = recipe

    # Print summary
    n_datasets = len(config.input.tissues)
    console.print("[bold blue]DAPIDL Unified Pipeline[/bold blue]\n")
    console.print(f"  Datasets: {n_datasets}")
    for tc in config.input.tissues:
        tier_label = {1: "ground truth", 2: "consensus", 3: "predicted"}
        source = tc.local_path or tc.dataset_id
        recipe_str = f", recipe={tc.recipe}" if tc.recipe != "default" else ""
        console.print(f"    - {tc.tissue}/{tc.platform.value}: {source} (tier {tc.confidence_tier}, {tier_label.get(tc.confidence_tier, 'unknown')}{recipe_str})")
    if ground_truth_file:
        console.print(f"  Ground truth file: {ground_truth_file}")
    console.print(f"  Segmenter: {segmenter}")
    console.print(f"  Sampling: {sampling}")
    console.print(f"  Fine-grained: {fine_grained}")
    console.print("  Cell Ontology: always on")
    console.print(f"  Patch size: {patch_size}px")
    if not skip_training:
        console.print(f"  Backbone: {backbone}")
        console.print(f"  Epochs: {epochs}")
    else:
        console.print("  [yellow]Training: SKIPPED (prepare-only mode)[/yellow]")
    console.print(f"  Execution: {'local' if local else 'ClearML agents'}")
    if not local:
        console.print(f"  GPU queue: {gpu_queue}")
    console.print()

    if orchestrator:
        from dapidl.pipeline.orchestrator import PipelineOrchestrator

        console.print("[dim]Using task-based orchestrator[/dim]")
        orch = PipelineOrchestrator(config)
        result = orch.run()

        if result.success:
            console.print("\n[green]✓ Pipeline completed![/green]")
            if result.lmdb_paths:
                console.print(f"  LMDB datasets: {len(result.lmdb_paths)}")
            if result.training_metrics:
                test_metrics = result.training_metrics
                if "f1_fine" in test_metrics:
                    console.print(f"  Test F1 (fine): {test_metrics.get('f1_fine', 'N/A'):.4f}")
                    console.print(f"  Test F1 (coarse): {test_metrics.get('f1_coarse', 'N/A'):.4f}")
                elif "f1" in test_metrics:
                    console.print(f"  Test F1: {test_metrics.get('f1', 'N/A'):.4f}")
            if result.model_path:
                console.print(f"  Model: {result.model_path}")
            if skip_training:
                console.print("[yellow]  Training was skipped (prepare-only mode)[/yellow]")
        else:
            console.print(f"\n[red]✗ Pipeline failed: {result.error}[/red]")
    else:
        controller = UnifiedPipelineController(config)

        if local:
            console.print("[cyan]Running pipeline locally...[/cyan]\n")
            result = controller.run_locally()

            if result.success:
                console.print("\n[green]✓ Pipeline completed![/green]")
                if result.training_metrics:
                    test_metrics = result.training_metrics
                    if "f1_fine" in test_metrics:
                        console.print(f"  Test F1 (fine): {test_metrics.get('f1_fine', 'N/A'):.4f}")
                        console.print(f"  Test F1 (coarse): {test_metrics.get('f1_coarse', 'N/A'):.4f}")
                    elif "f1" in test_metrics:
                        console.print(f"  Test F1: {test_metrics.get('f1', 'N/A'):.4f}")
                if result.model_path:
                    console.print(f"  Model: {result.model_path}")
                if skip_training:
                    console.print("[yellow]  Training was skipped (prepare-only mode)[/yellow]")
            else:
                console.print(f"\n[red]✗ Pipeline failed: {result.error}[/red]")
        else:
            console.print("[cyan]Creating ClearML pipeline...[/cyan]")
            controller.create_pipeline()
            console.print("[cyan]Starting pipeline (controller local, steps on agents)...[/cyan]")
            console.print(f"  Monitor at: {_get_clearml_web_url()}")
            pipeline_id = controller.run()
            console.print(f"\n[green]✓ Pipeline completed: {pipeline_id}[/green]")


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


@clearml_pipeline_group.command(name="universal", deprecated=True)
@click.option("--tissue", "-t", multiple=True, nargs=4, type=(str, str, str, int))
@click.option("--tissue-local", multiple=True, nargs=4, type=(str, str, str, int))
@click.option("--sampling", type=click.Choice(["equal", "proportional", "sqrt"]), default="sqrt")
@click.option("--backbone", default="efficientnetv2_rw_s")
@click.option("--epochs", type=int, default=100)
@click.option("--batch-size", type=int, default=64)
@click.option("--local", is_flag=True)
@click.option("--project", default="DAPIDL/pipelines")
@click.option("--name", default="universal-dapi")
def run_universal_pipeline(tissue, tissue_local, sampling, backbone, epochs, batch_size, local, project, name):
    """[DEPRECATED] Use 'clearml-pipeline run -t ...' instead.

    This command is kept for backward compatibility. It delegates to the
    unified 'run' command which supports 1-N datasets.
    """
    import warnings
    warnings.warn(
        "The 'universal' command is deprecated. Use 'clearml-pipeline run -t ...' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    console.print("[yellow]⚠ DEPRECATED: Use 'clearml-pipeline run -t ...' instead.[/yellow]\n")

    # Combine tissue and tissue_local into unified -t tuples
    all_tissues = list(tissue)
    for tissue_name, path, platform, tier in tissue_local:
        all_tissues.append((tissue_name, path, platform, tier))

    # Delegate to run_pipeline via Click context
    ctx = click.get_current_context()
    ctx.invoke(
        run_pipeline,
        tissue=tuple(all_tissues),
        sampling=sampling,
        segmenter="native",
        annotator="celltypist",
        patch_size="128",
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        local=local,
        project=project,
        skip_training=False,
        fine_grained=False,
        validate=False,
    )


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
        console.print("[cyan]Starting pipeline (controller local, steps on agents)...[/cyan]")
        console.print(f"  Monitor at: {_get_clearml_web_url()}")
        pipeline_id = controller.run()
        console.print(f"\n[green]✓ Enhanced pipeline completed: {pipeline_id}[/green]")


@clearml_pipeline_group.command(name="create-base-tasks")
@click.option(
    "--project",
    default="DAPIDL/pipelines",
    help="ClearML project name",
)
def create_base_tasks(project: str) -> None:
    """Create ClearML base tasks for pipeline steps.

    This registers each pipeline step as a ClearML Task, which is required
    before running the pipeline remotely. Only needs to be run once.
    """
    from dapidl.pipeline.unified_config import DAPIDLPipelineConfig
    from dapidl.pipeline.unified_controller import UnifiedPipelineController

    console.print("[bold blue]Creating ClearML Base Tasks[/bold blue]\n")

    config = DAPIDLPipelineConfig(project=project)
    controller = UnifiedPipelineController(config)

    console.print("[cyan]Registering all pipeline steps...[/cyan]")
    controller.create_base_tasks()

    console.print("\n[green]✓ Base tasks created![/green]")
    console.print(f"  Project: {project}")
    console.print("  You can now run the pipeline with: dapidl clearml-pipeline run")


@clearml_pipeline_group.command(name="create-controller-task")
@click.option(
    "--tissue", "-t",
    multiple=True,
    nargs=4,
    type=(str, str, str, int),
    help="Add dataset: TISSUE SOURCE PLATFORM TIER. SOURCE is ClearML dataset ID or local path.",
)
@click.option(
    "--sampling",
    type=click.Choice(["equal", "proportional", "sqrt"]),
    default="sqrt",
    help="Tissue sampling strategy",
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    help="Training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Training batch size",
)
@click.option(
    "--backbone",
    default="efficientnetv2_rw_s",
    help="CNN backbone for training",
)
@click.option(
    "--fine-grained",
    is_flag=True,
    help="Use fine-grained cell types (~20 classes) instead of broad categories (3 classes)",
)
@click.option(
    "--project",
    default="DAPIDL/pipelines",
    help="ClearML project name",
)
@click.option(
    "--queue",
    default="services",
    help="ClearML queue for the controller task",
)
@click.option(
    "--enqueue/--no-enqueue",
    default=False,
    help="Immediately enqueue the task (default: create only)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable step caching (force all steps to re-run)",
)
def create_controller_task(
    tissue: tuple,
    sampling: str,
    epochs: int,
    batch_size: int,
    backbone: str,
    fine_grained: bool,
    project: str,
    queue: str,
    enqueue: bool,
    no_cache: bool,
) -> None:
    """Create a ClearML controller task launchable from the web UI.

    This registers a pipeline controller as a ClearML task with editable
    parameters. You can then clone and enqueue it from app.clear.ml.

    Examples:

        # Create controller task (single dataset)
        dapidl clearml-pipeline create-controller-task \\
            -t lung bf8f913f xenium 2 --epochs 10

        # Create and immediately enqueue
        dapidl clearml-pipeline create-controller-task \\
            -t lung bf8f913f xenium 2 --epochs 10 --enqueue

        # Multi-tissue controller
        dapidl clearml-pipeline create-controller-task \\
            -t lung bf8f913f xenium 2 \\
            -t heart 482be038 xenium 2 \\
            --epochs 50 --sampling sqrt
    """
    from clearml import Task
    from dapidl.pipeline.unified_config import (
        AnnotationConfig,
        BackboneType,
        DAPIDLPipelineConfig,
        ExecutionConfig,
        LMDBConfig,
        Platform,
        SamplingStrategy,
        TrainingConfig,
    )

    if not tissue:
        console.print("[red]Error: At least one -t/--tissue required[/red]")
        console.print("[dim]Usage: -t TISSUE SOURCE PLATFORM TIER[/dim]")
        console.print("[dim]Example: -t lung bf8f913f xenium 2[/dim]")
        raise click.Abort()

    # Build unified config
    config = DAPIDLPipelineConfig(
        project=project,
        training=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            backbone=BackboneType(backbone),
            sampling_strategy=SamplingStrategy(sampling),
        ),
        annotation=AnnotationConfig(fine_grained=fine_grained),
        lmdb=LMDBConfig(patch_sizes=[128]),
        execution=ExecutionConfig(execute_remotely=True, cache_data_steps=not no_cache),
    )

    # Add tissues
    for tissue_name, source, platform, tier in tissue:
        source_path = Path(source)
        if source_path.exists():
            config.input.add_tissue(
                tissue=tissue_name,
                local_path=str(source_path),
                platform=Platform(platform),
                confidence_tier=int(tier),
            )
        else:
            config.input.add_tissue(
                tissue=tissue_name,
                dataset_id=source,
                platform=Platform(platform),
                confidence_tier=int(tier),
            )

    # Create a ClearML task pointing to the controller script
    console.print("[bold blue]Creating ClearML Controller Task[/bold blue]\n")

    task = Task.create(
        project_name=project,
        task_name="dapidl-pipeline-controller",
        task_type=Task.TaskTypes.controller,
        script="scripts/clearml_pipeline_controller.py",
        repo=".",
        add_task_init_call=False,
    )

    # Set all pipeline parameters (editable in the UI)
    clearml_params = config.to_clearml_parameters()
    task.set_parameters(clearml_params)

    console.print(f"  Task ID: {task.id}")
    console.print(f"  Project: {project}")
    console.print(f"  Tissues: {len(config.input.tissues)}")
    for tc in config.input.tissues:
        source = tc.local_path or tc.dataset_id
        console.print(f"    - {tc.tissue}/{tc.platform.value}: {source} (tier {tc.confidence_tier})")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Parameters: {len(clearml_params)} editable fields")

    if enqueue:
        Task.enqueue(task, queue_name=queue)
        console.print(f"\n[green]Task enqueued to '{queue}' queue[/green]")
        console.print(f"  Monitor at: {_get_clearml_web_url()}")
    else:
        console.print(f"\n[green]Task created (not enqueued)[/green]")
        console.print(f"  To launch: clone in {_get_clearml_web_url()} -> edit params -> enqueue to '{queue}'")
        console.print(f"  Or enqueue via CLI: clearml-task enqueue --id {task.id} --queue {queue}")


@clearml_pipeline_group.command(name="sota")
@click.option(
    "--dataset-id",
    type=str,
    help="ClearML Dataset ID for raw spatial data",
)
@click.option(
    "--local-path",
    "-l",
    type=click.Path(exists=True, path_type=Path),
    help="Local path to Xenium/MERSCOPE output directory",
)
@click.option(
    "--s3-uri",
    type=str,
    help="S3 URI for raw data (e.g., s3://dapidl/raw-data/xenium-breast/)",
)
@click.option(
    "--platform",
    type=click.Choice(["auto", "xenium", "merscope"]),
    default="auto",
    show_default=True,
    help="Spatial platform (auto-detected by default)",
)
@click.option(
    "--fine-grained/--coarse",
    default=False,
    help="Use fine-grained classification (default: coarse 3-class)",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    show_default=True,
    help="Training epochs",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run locally (no ClearML agents, no caching)",
)
@click.option(
    "--local-cached",
    is_flag=True,
    help="Run locally WITH ClearML caching (recommended for development)",
)
@click.option(
    "--create-tasks",
    is_flag=True,
    help="Create base tasks before running (first time only)",
)
def sota_pipeline(
    dataset_id: str | None,
    local_path: Path | None,
    s3_uri: str | None,
    platform: str,
    fine_grained: bool,
    epochs: int,
    local: bool,
    local_cached: bool,
    create_tasks: bool,
) -> None:
    """Run state-of-the-art pipeline with best practices from benchmarking.

    This pipeline uses optimal settings discovered through comprehensive
    benchmarking (Jan 2025):

    \b
    Annotation (F1=0.844):
      - PopV Ensemble with 3 CellTypist + SingleR (HPCA + Blueprint)
      - UNWEIGHTED voting (beats confidence-weighted by 15-22%)
      - Blueprint reference CRITICAL for Stromal (+117% F1)

    \b
    Training (F1=0.8481):
      - EfficientNetV2-S backbone, 256px patches
      - max_weight_ratio=10.0 (CRITICAL: prevents mode collapse)
      - WeightedRandomSampler + weighted loss

    \b
    Examples:

        # Remote execution on ClearML
        dapidl clearml-pipeline sota --dataset-id abc123

        # Local execution with ClearML caching (recommended for development)
        dapidl clearml-pipeline sota --local-path /path/to/xenium --local-cached

        # Local execution without caching (fastest, no ClearML dependency)
        dapidl clearml-pipeline sota --local-path /path/to/xenium --local

        # Fine-grained classification with more epochs
        dapidl clearml-pipeline sota --dataset-id abc123 --fine-grained --epochs 150
    """
    from dapidl.pipeline import SOTAPipelineController, create_sota_config

    # Validate input - but allow --create-tasks without data source
    if not create_tasks and not dataset_id and not local_path and not s3_uri:
        console.print("[red]Error: Must specify --dataset-id, --local-path, or --s3-uri[/red]")
        raise click.Abort()

    # Create configuration with SOTA settings
    config = create_sota_config(
        dataset_id=dataset_id,
        local_path=str(local_path) if local_path else None,
        s3_uri=s3_uri,
        platform=platform,
        fine_grained=fine_grained,
        epochs=epochs,
    )

    controller = SOTAPipelineController(config)

    console.print("[bold blue]DAPIDL State-of-the-Art Pipeline[/bold blue]\n")
    console.print("[dim]Best practices from Jan 2025 benchmarking[/dim]\n")

    # Show SOTA settings
    console.print("[cyan]Annotation (SOTA):[/cyan]")
    console.print(f"  Models: {config.annotation.celltypist_models}")
    console.print(f"  SingleR: {config.annotation.singler_reference} (CRITICAL for Stromal)")
    console.print(f"  Voting: UNWEIGHTED (beats confidence-weighted by 15-22%)")

    console.print("\n[cyan]Training (SOTA):[/cyan]")
    console.print(f"  Backbone: {config.training.backbone.value}")
    console.print(f"  Patch sizes: {config.lmdb.patch_sizes} (primary: {config.lmdb.primary_patch_size})")
    console.print(f"  Epochs: {config.training.epochs}")
    console.print(f"  max_weight_ratio: {config.training.max_weight_ratio} (CRITICAL: prevents mode collapse)")
    console.print()

    # Create base tasks if requested
    if create_tasks:
        console.print("[cyan]Creating SOTA base tasks...[/cyan]")
        controller.create_base_tasks()
        console.print("[green]✓ Base tasks created[/green]\n")

        # If only creating tasks (no data source), exit now
        if not dataset_id and not local_path and not s3_uri:
            console.print("[dim]Base tasks created. Run with --dataset-id, --local-path, or --s3-uri to execute pipeline.[/dim]")
            return

    if local_cached:
        console.print("[cyan]Running locally WITH ClearML caching...[/cyan]")
        console.print("[dim]Steps will run as subprocesses with cache_executed_step support[/dim]\n")
        controller.create_pipeline()
        pipeline_id = controller.run_locally_with_caching()
        console.print(f"\n[green]✓ SOTA Pipeline complete![/green]")
        console.print(f"  Pipeline ID: {pipeline_id}")
    elif local:
        console.print("[cyan]Running locally (no caching)...[/cyan]\n")
        results = controller.run_locally()

        if "training" in results:
            console.print("\n[green]✓ SOTA Pipeline complete![/green]")
            if "model_path" in results["training"]:
                console.print(f"  Model: {results['training']['model_path']}")
            if "metrics" in results["training"]:
                metrics = results["training"]["metrics"]
                console.print("  Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        console.print(f"    {key}: {value:.4f}")
        else:
            console.print("\n[yellow]Pipeline completed but training results not found[/yellow]")
    else:
        console.print("[cyan]Creating ClearML SOTA pipeline...[/cyan]")
        controller.create_pipeline()
        console.print("[cyan]Starting pipeline (controller local, steps on agents)...[/cyan]")
        console.print(f"  Monitor at: {_get_clearml_web_url()}")
        pipeline_id = controller.run()
        console.print(f"\n[green]✓ SOTA pipeline completed: {pipeline_id}[/green]")


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


# =============================================================================
# Standalone Step Commands
# =============================================================================


@main.group(name="step")
def step_group() -> None:
    """Run individual pipeline steps standalone.

    Each step can run independently with its own config. Pass artifacts
    between steps using --input-artifacts and --output-artifacts JSON files.

    Examples:
        # Run data loader
        dapidl step data-loader --dataset-id bf8f913f --platform xenium \
            --output-artifacts /tmp/step1.json

        # Run annotation using previous step's output
        dapidl step annotate --input-artifacts /tmp/step1.json \
            --output-artifacts /tmp/step2.json

        # Run LMDB creation
        dapidl step lmdb --input-artifacts /tmp/step2.json --patch-size 128 \
            --output-artifacts /tmp/step3.json
    """
    pass


def _load_artifacts(path: str | None) -> dict:
    """Load artifacts from JSON file."""
    if path is None:
        return {}
    import json

    with open(path) as f:
        return json.load(f)


def _save_artifacts(artifacts: dict, path: str | None) -> None:
    """Save artifacts to JSON file."""
    if path is None:
        return
    import json

    # Filter to JSON-serializable values
    serializable = {}
    for k, v in artifacts.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable[k] = v
        elif hasattr(v, "__str__"):
            serializable[k] = str(v)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    console.print(f"  Artifacts saved: {path}")


@step_group.command(name="data-loader")
@click.option("--dataset-id", default=None, help="ClearML dataset ID")
@click.option("--local-path", "-l", default=None, help="Local data path")
@click.option("--platform", default="auto", help="Platform: auto, xenium, merscope")
@click.option("--output-artifacts", "-o", default=None, help="Save artifacts JSON")
def step_data_loader(dataset_id, local_path, platform, output_artifacts):
    """Load raw spatial transcriptomics data."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.data_loader import DataLoaderConfig, DataLoaderStep

    config = DataLoaderConfig(
        dataset_id=dataset_id,
        local_path=local_path,
        platform=platform,
    )
    step = DataLoaderStep(config)
    console.print("[cyan]Running DataLoader step...[/cyan]")
    result = step.execute(StepArtifacts())

    console.print("[green]✓ DataLoader complete[/green]")
    for k, v in result.outputs.items():
        console.print(f"  {k}: {v}")

    _save_artifacts(result.outputs, output_artifacts)


@step_group.command(name="annotate")
@click.option("--annotator", default="celltypist", help="Method: celltypist, ground_truth, popv")
@click.option("--strategy", default="consensus", help="Strategy: single, consensus")
@click.option("--model-names", default="Cells_Adult_Breast.pkl,Immune_All_High.pkl", help="CellTypist models (comma-separated)")
@click.option("--fine-grained/--coarse", default=False, help="Fine-grained vs broad categories")
@click.option("--input-artifacts", "-i", default=None, help="Input artifacts JSON from data-loader")
@click.option("--output-artifacts", "-o", default=None, help="Save artifacts JSON")
def step_annotate(annotator, strategy, model_names, fine_grained, input_artifacts, output_artifacts):
    """Annotate cell types from gene expression."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.annotation import AnnotationStep, AnnotationStepConfig

    prev = _load_artifacts(input_artifacts)
    config = AnnotationStepConfig(
        annotator=annotator,
        strategy=strategy,
        model_names=model_names.split(","),
        fine_grained=fine_grained,
    )
    step = AnnotationStep(config)
    console.print("[cyan]Running Annotation step...[/cyan]")
    result = step.execute(StepArtifacts(inputs={}, outputs=prev))

    console.print("[green]✓ Annotation complete[/green]")
    for k, v in result.outputs.items():
        if not isinstance(v, dict) or len(str(v)) < 200:
            console.print(f"  {k}: {v}")

    combined = {**prev, **result.outputs}
    _save_artifacts(combined, output_artifacts)


@step_group.command(name="segment")
@click.option("--segmenter", default="native", help="Method: cellpose, native")
@click.option("--diameter", default=40, help="Nucleus diameter in pixels")
@click.option("--input-artifacts", "-i", default=None, help="Input artifacts JSON from data-loader")
@click.option("--output-artifacts", "-o", default=None, help="Save artifacts JSON")
def step_segment(segmenter, diameter, input_artifacts, output_artifacts):
    """Segment nuclei from DAPI images."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.segmentation import SegmentationStep, SegmentationStepConfig

    prev = _load_artifacts(input_artifacts)
    config = SegmentationStepConfig(
        segmenter=segmenter,
        diameter=diameter,
    )
    step = SegmentationStep(config)
    console.print("[cyan]Running Segmentation step...[/cyan]")
    result = step.execute(StepArtifacts(inputs={}, outputs=prev))

    console.print("[green]✓ Segmentation complete[/green]")
    for k, v in result.outputs.items():
        if not isinstance(v, dict) or len(str(v)) < 200:
            console.print(f"  {k}: {v}")

    combined = {**prev, **result.outputs}
    _save_artifacts(combined, output_artifacts)


@step_group.command(name="lmdb")
@click.option("--patch-size", default=128, help="Patch size: 32, 64, 128, 256")
@click.option("--normalization", default="adaptive", help="Method: adaptive, percentile, minmax")
@click.option("--input-artifacts", "-i", default=None, help="Input artifacts JSON from annotate")
@click.option("--output-artifacts", "-o", default=None, help="Save artifacts JSON")
def step_lmdb(patch_size, normalization, input_artifacts, output_artifacts):
    """Create LMDB training dataset from annotated data."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.lmdb_creation import LMDBCreationConfig, LMDBCreationStep

    prev = _load_artifacts(input_artifacts)
    config = LMDBCreationConfig(
        patch_size=patch_size,
        normalization_method=normalization,
    )
    step = LMDBCreationStep(config)
    console.print("[cyan]Running LMDB Creation step...[/cyan]")
    result = step.execute(StepArtifacts(inputs={}, outputs=prev))

    console.print("[green]✓ LMDB creation complete[/green]")
    for k, v in result.outputs.items():
        if not isinstance(v, dict) or len(str(v)) < 200:
            console.print(f"  {k}: {v}")

    combined = {**prev, **result.outputs}
    _save_artifacts(combined, output_artifacts)


@step_group.command(name="train")
@click.option("--backbone", default="efficientnetv2_rw_s", help="CNN backbone")
@click.option("--epochs", default=50, help="Training epochs")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--learning-rate", default=3e-4, help="Learning rate")
@click.option("--input-artifacts", "-i", default=None, help="Input artifacts JSON from lmdb")
@click.option("--output-artifacts", "-o", default=None, help="Save artifacts JSON")
def step_train(backbone, epochs, batch_size, learning_rate, input_artifacts, output_artifacts):
    """Train DAPI cell type classifier."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.training import TrainingStep, TrainingStepConfig

    prev = _load_artifacts(input_artifacts)
    config = TrainingStepConfig(
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    step = TrainingStep(config)
    console.print("[cyan]Running Training step...[/cyan]")
    result = step.execute(StepArtifacts(inputs={}, outputs=prev))

    console.print("[green]✓ Training complete[/green]")
    for k, v in result.outputs.items():
        if not isinstance(v, dict) or len(str(v)) < 200:
            console.print(f"  {k}: {v}")

    combined = {**prev, **result.outputs}
    _save_artifacts(combined, output_artifacts)


@step_group.command(name="cl-standardize")
@click.option("--target-level", default="coarse", help="Level: broad, coarse, fine")
@click.option("--fuzzy-threshold", default=0.85, help="Fuzzy matching threshold")
@click.option("--input-artifacts", "-i", default=None, help="Input artifacts JSON from annotate")
@click.option("--output-artifacts", "-o", default=None, help="Save artifacts JSON")
def step_cl_standardize(target_level, fuzzy_threshold, input_artifacts, output_artifacts):
    """Standardize annotations to Cell Ontology terms."""
    from dapidl.pipeline.base import StepArtifacts
    from dapidl.pipeline.steps.cl_standardization import (
        CLStandardizationConfig,
        CLStandardizationStep,
    )

    prev = _load_artifacts(input_artifacts)
    config = CLStandardizationConfig(target_level=target_level, fuzzy_threshold=fuzzy_threshold)
    step = CLStandardizationStep(config=config)
    console.print("[cyan]Running CL Standardization step...[/cyan]")
    result = step.execute(StepArtifacts(inputs={}, outputs=prev))

    console.print("[green]✓ CL Standardization complete[/green]")
    for k, v in result.outputs.items():
        if not isinstance(v, dict) or len(str(v)) < 200:
            console.print(f"  {k}: {v}")

    combined = {**prev, **result.outputs}
    _save_artifacts(combined, output_artifacts)


if __name__ == "__main__":
    main()
