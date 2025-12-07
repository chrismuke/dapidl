"""DAPIDL Command Line Interface."""

import click
from pathlib import Path
from typing import Tuple
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """DAPIDL: Deep learning for cell type prediction from DAPI nuclear staining."""
    pass


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
    type=click.Choice(["consensus", "hierarchical", "popv", "single"], case_sensitive=False),
    default="consensus",
    show_default=True,
    help="Annotation strategy: consensus (voting), hierarchical (primary+refinement), popv (ensemble), single (legacy)",
)
def prepare(
    xenium_path: Path,
    output: Path,
    models: Tuple[str, ...],
    patch_size: int,
    confidence_threshold: float,
    majority_voting: bool,
    strategy: str,
) -> None:
    """Prepare dataset from Xenium output.

    Extracts nucleus patches, generates cell type labels using CellTypist,
    and saves the prepared dataset in Zarr format.

    Annotation strategies:
        - consensus: Voting across multiple models (default, recommended)
        - hierarchical: Tissue-specific model + specialized refinement
        - popv: popV ensemble prediction (requires popv package)
        - single: Legacy single-model mode

    For multiple models, use --model multiple times:
        dapidl prepare -x /path -o /out -m Model1.pkl -m Model2.pkl
    """
    from dapidl.data.xenium import XeniumDataReader
    from dapidl.data.patches import PatchExtractor
    from dapidl.data.annotation import CellTypeAnnotator

    console.print("[bold blue]DAPIDL Dataset Preparation[/bold blue]")
    console.print(f"Xenium path: {xenium_path}")
    console.print(f"Output path: {output}")
    console.print(f"Model(s): {', '.join(models)}")
    console.print(f"Strategy: {strategy}")
    console.print(f"Patch size: {patch_size}")
    console.print(f"Confidence threshold: {confidence_threshold}")
    console.print(f"Majority voting: {majority_voting}")

    # Load Xenium data
    console.print("\n[yellow]Loading Xenium data...[/yellow]")
    reader = XeniumDataReader(xenium_path)

    # Create annotator with specified models and strategy
    annotator = CellTypeAnnotator(
        model_names=list(models),
        confidence_threshold=confidence_threshold,
        majority_voting=majority_voting,
        strategy=strategy,
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
def train(
    config: Path | None,
    data: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    wandb: bool,
) -> None:
    """Train cell type classifier.

    Trains an EfficientNetB2 model to classify cell types from DAPI patches.
    """
    from dapidl.training.trainer import Trainer

    console.print(f"[bold blue]DAPIDL Training[/bold blue]")
    console.print(f"Data path: {data}")
    console.print(f"Output path: {output}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {lr}")
    console.print(f"W&B logging: {wandb}")

    trainer = Trainer(
        data_path=data,
        output_path=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        use_wandb=wandb,
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
def pipeline(
    xenium_path: Path,
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
) -> None:
    """Run the complete DAPIDL pipeline: prepare + train.

    This supercommand executes the full workflow:
    1. Prepare dataset from Xenium output (extract patches, annotate with CellTypist)
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
        # Run full pipeline with defaults
        dapidl pipeline -x /path/to/xenium -o ./experiment

        # Use custom model and more epochs
        dapidl pipeline -x /path/to/xenium -o ./experiment \\
            -m Immune_All_High.pkl --epochs 100

        # Only prepare dataset, skip training
        dapidl pipeline -x /path/to/xenium -o ./experiment --skip-train

        # Only train, using existing dataset
        dapidl pipeline -x /path/to/xenium -o ./experiment --skip-prepare
    """
    from dapidl.data.xenium import XeniumDataReader
    from dapidl.data.patches import PatchExtractor
    from dapidl.data.annotation import CellTypeAnnotator
    from dapidl.training.trainer import Trainer

    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]                    DAPIDL PIPELINE                           [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════════════[/bold magenta]")
    console.print()

    # Define paths
    dataset_path = output / "dataset"
    training_path = output / "training"

    # Display configuration
    console.print("[bold cyan]Configuration:[/bold cyan]")
    console.print(f"  Xenium input:   {xenium_path}")
    console.print(f"  Output dir:     {output}")
    console.print(f"  Dataset dir:    {dataset_path}")
    console.print(f"  Training dir:   {training_path}")
    console.print(f"  Model(s):       {', '.join(models)}")
    console.print(f"  Strategy:       {strategy}")
    console.print(f"  Patch size:     {patch_size}")
    console.print(f"  Confidence:     {confidence_threshold}")
    console.print(f"  Majority vote:  {majority_voting}")
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

        console.print("[yellow]Loading Xenium data...[/yellow]")
        reader = XeniumDataReader(xenium_path)

        console.print("[yellow]Initializing CellTypist annotator...[/yellow]")
        annotator = CellTypeAnnotator(
            model_names=list(models),
            confidence_threshold=confidence_threshold,
            majority_voting=majority_voting,
            strategy=strategy,
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


if __name__ == "__main__":
    main()
