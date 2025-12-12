#!/usr/bin/env python
"""Download essential files from Vizgen MERSCOPE datasets.

Downloads only the files needed for DAPIDL training:
- DAPI mosaic image (single z-plane)
- Cell metadata (coordinates)
- Cell-by-gene expression matrix
- Coordinate transform

Skips large unnecessary files like detected_transcripts.csv (~20-50GB each).

Usage:
    # List available datasets
    python download_vizgen.py --list

    # Download specific dataset by short name
    python download_vizgen.py -d breast -o ~/datasets/vizgen/
    python download_vizgen.py -d colon1 -o ~/datasets/vizgen/

    # Download multiple datasets
    python download_vizgen.py -d breast -d lung1 -d lung2 -o ~/datasets/vizgen/

    # Download all datasets
    python download_vizgen.py --all --output ~/datasets/vizgen/

    # Dry run (show what would be downloaded)
    python download_vizgen.py -d breast --dry-run

    # Include expression matrix (for CellTypist annotation)
    python download_vizgen.py -d breast --include-expression

    # Download specific z-plane for DAPI
    python download_vizgen.py -d breast --z-plane 3

Requirements:
    - gsutil (Google Cloud SDK): https://cloud.google.com/sdk/docs/install
    - Access to Vizgen FFPE IO datasets (register at info.vizgen.com/ffpe-showcase)
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Vizgen FFPE IO datasets on Google Cloud Storage
# Bucket: gs://vz-ffpe-showcase
VIZGEN_DATASETS = {
    # Breast Cancer
    "breast": {
        "full_name": "HumanBreastCancerPatient1",
        "description": "Human breast cancer FFPE (500 genes, IO panel)",
        "tissue": "breast",
        "aliases": ["breast1", "breast_cancer"],
    },

    # Colon Cancer
    "colon1": {
        "full_name": "HumanColonCancerPatient1",
        "description": "Human colon cancer FFPE sample 1",
        "tissue": "colon",
        "aliases": ["colon_cancer_1"],
    },
    "colon2": {
        "full_name": "HumanColonCancerPatient2",
        "description": "Human colon cancer FFPE sample 2",
        "tissue": "colon",
        "aliases": ["colon_cancer_2"],
    },

    # Liver Cancer
    "liver1": {
        "full_name": "HumanLiverCancerPatient1",
        "description": "Human liver cancer FFPE sample 1",
        "tissue": "liver",
        "aliases": ["liver_cancer_1"],
    },
    "liver2": {
        "full_name": "HumanLiverCancerPatient2",
        "description": "Human liver cancer FFPE sample 2",
        "tissue": "liver",
        "aliases": ["liver_cancer_2"],
    },

    # Lung Cancer
    "lung1": {
        "full_name": "HumanLungCancerPatient1",
        "description": "Human lung cancer FFPE sample 1",
        "tissue": "lung",
        "aliases": ["lung_cancer_1"],
    },
    "lung2": {
        "full_name": "HumanLungCancerPatient2",
        "description": "Human lung cancer FFPE sample 2",
        "tissue": "lung",
        "aliases": ["lung_cancer_2"],
    },

    # Melanoma
    "melanoma1": {
        "full_name": "HumanMelanomaPatient1",
        "description": "Human melanoma/skin cancer FFPE sample 1",
        "tissue": "skin",
        "aliases": ["skin1", "melanoma_1"],
    },
    "melanoma2": {
        "full_name": "HumanMelanomaPatient2",
        "description": "Human melanoma/skin cancer FFPE sample 2",
        "tissue": "skin",
        "aliases": ["skin2", "melanoma_2"],
    },

    # Ovarian Cancer (4 samples)
    "ovarian1": {
        "full_name": "HumanOvarianCancerPatient1",
        "description": "Human ovarian cancer FFPE sample 1",
        "tissue": "ovarian",
        "aliases": ["ovarian_cancer_1"],
    },
    "ovarian2a": {
        "full_name": "HumanOvarianCancerPatient2Slice1",
        "description": "Human ovarian cancer patient 2, slice 1",
        "tissue": "ovarian",
        "aliases": ["ovarian_cancer_2a", "ovarian2_slice1"],
    },
    "ovarian2b": {
        "full_name": "HumanOvarianCancerPatient2Slice2",
        "description": "Human ovarian cancer patient 2, slice 2",
        "tissue": "ovarian",
        "aliases": ["ovarian_cancer_2b", "ovarian2_slice2"],
    },
    "ovarian2c": {
        "full_name": "HumanOvarianCancerPatient2Slice3",
        "description": "Human ovarian cancer patient 2, slice 3",
        "tissue": "ovarian",
        "aliases": ["ovarian_cancer_2c", "ovarian2_slice3"],
    },

    # Prostate Cancer
    "prostate1": {
        "full_name": "HumanProstateCancerPatient1",
        "description": "Human prostate cancer FFPE sample 1",
        "tissue": "prostate",
        "aliases": ["prostate_cancer_1"],
    },
    "prostate2": {
        "full_name": "HumanProstateCancerPatient2",
        "description": "Human prostate cancer FFPE sample 2",
        "tissue": "prostate",
        "aliases": ["prostate_cancer_2"],
    },

    # Uterine Cancer
    "uterine1": {
        "full_name": "HumanUterineCancerPatient1",
        "description": "Human uterine cancer FFPE sample 1",
        "tissue": "uterine",
        "aliases": ["uterine_cancer_1"],
    },
    "uterine2_ra": {
        "full_name": "HumanUterineCancerPatient2-RACostain",
        "description": "Human uterine cancer patient 2, RA costain",
        "tissue": "uterine",
        "aliases": ["uterine_cancer_2_ra", "uterine2a"],
    },
    "uterine2_ro": {
        "full_name": "HumanUterineCancerPatient2-ROCostain",
        "description": "Human uterine cancer patient 2, RO costain",
        "tissue": "uterine",
        "aliases": ["uterine_cancer_2_ro", "uterine2b"],
    },
}

# Build alias lookup
ALIAS_MAP = {}
for name, info in VIZGEN_DATASETS.items():
    ALIAS_MAP[name] = name
    for alias in info.get("aliases", []):
        ALIAS_MAP[alias] = name

# Bucket configuration
GCS_BUCKET = "gs://vz-ffpe-showcase"

# Files to download (patterns)
ESSENTIAL_FILES = [
    # Cell metadata with coordinates
    "cell_metadata*.csv",
    # Coordinate transform (in images folder)
    "images/micron_to_mosaic_pixel_transform.csv",
    # Manifest for image info
    "images/manifest.json",
]

# DAPI image pattern (will be formatted with z-plane)
DAPI_PATTERN = "images/mosaic_DAPI_z{z}.tif"

# Optional files
EXPRESSION_FILES = [
    "cell_by_gene*.csv",
]

# Cell type annotations (if available)
ANNOTATION_FILES = [
    "cell_types*.csv",
    "*cell_type*.csv",
    "*annotations*.csv",
]

# Files to explicitly SKIP (large and unnecessary)
SKIP_PATTERNS = [
    "detected_transcripts*.csv",  # 20-50 GB each!
    "*.vzg",  # Proprietary viewer format
    "mosaic_PolyT*.tif",  # PolyT stain (not needed)
    "mosaic_Cellbound*.tif",  # Cell boundary stain (not needed)
]


def resolve_dataset_name(name: str) -> str | None:
    """Resolve dataset name or alias to canonical name."""
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    return ALIAS_MAP.get(name_lower)


def run_gsutil(args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Run gsutil command."""
    cmd = ["gsutil"] + args
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd)


def check_gsutil() -> bool:
    """Check if gsutil is available."""
    try:
        result = run_gsutil(["version"], capture=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def list_bucket_contents(bucket_path: str) -> list[str]:
    """List contents of a GCS bucket path."""
    result = run_gsutil(["ls", "-r", bucket_path], capture=True)
    if result.returncode != 0:
        print(f"Error listing {bucket_path}: {result.stderr}")
        return []
    return [line.strip() for line in result.stdout.split("\n") if line.strip()]


def get_file_size(gcs_path: str) -> int:
    """Get size of a GCS file in bytes."""
    result = run_gsutil(["du", "-s", gcs_path], capture=True)
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.split()[0])
        except (ValueError, IndexError):
            pass
    return 0


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def find_dapi_z_planes(bucket_path: str) -> list[int]:
    """Find available DAPI z-planes in dataset."""
    result = run_gsutil(["ls", f"{bucket_path}/images/mosaic_DAPI_z*.tif"], capture=True)
    if result.returncode != 0:
        return []

    z_planes = []
    for line in result.stdout.split("\n"):
        if "mosaic_DAPI_z" in line:
            # Extract z number from filename
            import re
            match = re.search(r"mosaic_DAPI_z(\d+)\.tif", line)
            if match:
                z_planes.append(int(match.group(1)))

    return sorted(z_planes)


def download_file(gcs_path: str, local_path: Path, dry_run: bool = False) -> bool:
    """Download a single file from GCS."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        size = get_file_size(gcs_path)
        print(f"  [DRY RUN] Would download: {gcs_path}")
        print(f"            Size: {format_size(size)}")
        print(f"            To: {local_path}")
        return True

    print(f"  Downloading: {gcs_path}")
    result = run_gsutil(["-m", "cp", gcs_path, str(local_path)])
    return result.returncode == 0


def download_pattern(bucket_path: str, pattern: str, local_dir: Path, dry_run: bool = False) -> list[Path]:
    """Download files matching a pattern."""
    gcs_pattern = f"{bucket_path}/{pattern}"

    # List matching files
    result = run_gsutil(["ls", gcs_pattern], capture=True)
    if result.returncode != 0:
        # Pattern didn't match anything
        return []

    downloaded = []
    for line in result.stdout.split("\n"):
        if not line.strip():
            continue

        gcs_path = line.strip()
        # Get relative path from bucket
        rel_path = gcs_path.replace(bucket_path + "/", "")
        local_path = local_dir / rel_path

        if download_file(gcs_path, local_path, dry_run):
            downloaded.append(local_path)

    return downloaded


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    z_plane: int = 3,
    include_expression: bool = False,
    include_annotations: bool = True,
    dry_run: bool = False,
) -> bool:
    """Download essential files for a dataset."""
    # Resolve alias
    canonical_name = resolve_dataset_name(dataset_name)
    if canonical_name is None:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Use --list to see available datasets")
        return False

    dataset = VIZGEN_DATASETS[canonical_name]
    full_name = dataset["full_name"]
    bucket_path = f"{GCS_BUCKET}/{full_name}"
    local_dir = output_dir / canonical_name

    print(f"\n{'='*60}")
    print(f"Dataset: {canonical_name} ({full_name})")
    print(f"Description: {dataset['description']}")
    print(f"Tissue: {dataset['tissue']}")
    print(f"Source: {bucket_path}")
    print(f"Destination: {local_dir}")
    print(f"{'='*60}")

    # Check available z-planes
    print("\nChecking available DAPI z-planes...")
    z_planes = find_dapi_z_planes(bucket_path)
    if z_planes:
        print(f"  Available z-planes: {z_planes}")
        if z_plane not in z_planes:
            # Use middle z-plane if requested one not available
            z_plane = z_planes[len(z_planes) // 2]
            print(f"  Using z-plane {z_plane} (middle)")
    else:
        print("  Warning: Could not determine z-planes, trying z=3")

    total_downloaded = []

    # Download essential files
    print("\nDownloading essential files...")
    for pattern in ESSENTIAL_FILES:
        files = download_pattern(bucket_path, pattern, local_dir, dry_run)
        total_downloaded.extend(files)

    # Download DAPI image (single z-plane)
    print(f"\nDownloading DAPI image (z={z_plane})...")
    dapi_pattern = DAPI_PATTERN.format(z=z_plane)
    files = download_pattern(bucket_path, dapi_pattern, local_dir, dry_run)
    total_downloaded.extend(files)

    # Download expression matrix if requested
    if include_expression:
        print("\nDownloading expression matrix (for CellTypist)...")
        for pattern in EXPRESSION_FILES:
            files = download_pattern(bucket_path, pattern, local_dir, dry_run)
            total_downloaded.extend(files)

    # Download cell type annotations if available
    if include_annotations:
        print("\nLooking for cell type annotations...")
        for pattern in ANNOTATION_FILES:
            files = download_pattern(bucket_path, pattern, local_dir, dry_run)
            total_downloaded.extend(files)

    # Summary
    print(f"\n{'='*60}")
    print(f"Download summary for {canonical_name}:")
    print(f"  Files downloaded: {len(total_downloaded)}")
    if not dry_run and total_downloaded:
        total_size = sum(f.stat().st_size for f in total_downloaded if f.exists())
        print(f"  Total size: {format_size(total_size)}")
    print(f"{'='*60}")

    return len(total_downloaded) > 0


def list_datasets():
    """List all available datasets."""
    print("\nAvailable Vizgen FFPE IO Datasets (gs://vz-ffpe-showcase)")
    print("=" * 80)
    print(f"{'Short Name':<15} {'Tissue':<10} {'GCS Path':<40} {'Description'}")
    print("-" * 80)

    # Group by tissue
    by_tissue = {}
    for name, info in VIZGEN_DATASETS.items():
        tissue = info["tissue"]
        if tissue not in by_tissue:
            by_tissue[tissue] = []
        by_tissue[tissue].append((name, info))

    for tissue in sorted(by_tissue.keys()):
        for name, info in by_tissue[tissue]:
            print(f"{name:<15} {tissue:<10} {info['full_name']:<40} {info['description'][:30]}")

    print("=" * 80)
    print(f"\nTotal datasets: {len(VIZGEN_DATASETS)}")
    print("\nUsage examples:")
    print("  python download_vizgen.py -d breast                  # Download breast cancer")
    print("  python download_vizgen.py -d lung1 -d lung2          # Download both lung samples")
    print("  python download_vizgen.py -d breast -e               # Include expression matrix")
    print("  python download_vizgen.py --all                      # Download all datasets")
    print("\nNote: Register at info.vizgen.com/ffpe-showcase to access these datasets.")


def discover_bucket_structure(bucket: str):
    """Discover the actual structure of a bucket."""
    print(f"\nDiscovering bucket structure: {bucket}")
    print("=" * 60)

    result = run_gsutil(["ls", bucket], capture=True)
    if result.returncode != 0:
        print(f"Error accessing bucket: {result.stderr}")
        print("\nMake sure you have:")
        print("  1. Registered at info.vizgen.com/ffpe-showcase")
        print("  2. Authenticated with: gcloud auth login")
        return

    print("Top-level contents:")
    for line in result.stdout.split("\n"):
        if line.strip():
            print(f"  {line.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download essential files from Vizgen MERSCOPE datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset", "-d",
        action="append",
        dest="datasets",
        help="Dataset name(s) to download (can specify multiple times, use --list to see available)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.home() / "datasets" / "vizgen",
        help="Output directory (default: ~/datasets/vizgen)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--z-plane", "-z",
        type=int,
        default=3,
        help="DAPI z-plane to download (default: 3, middle focus)",
    )
    parser.add_argument(
        "--include-expression", "-e",
        action="store_true",
        help="Include expression matrix (cell_by_gene.csv) for CellTypist",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip downloading cell type annotations",
    )
    parser.add_argument(
        "--discover",
        help="Discover structure of a specific bucket (for debugging)",
    )

    args = parser.parse_args()

    # Check gsutil
    if not check_gsutil():
        print("Error: gsutil not found!")
        print("Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        print("Then authenticate: gcloud auth login")
        sys.exit(1)

    # Handle different modes
    if args.list:
        list_datasets()
        return

    if args.discover:
        discover_bucket_structure(args.discover)
        return

    if not args.datasets and not args.all:
        parser.print_help()
        print("\nError: Please specify --dataset or --all")
        sys.exit(1)

    # Download datasets
    datasets_to_download = list(VIZGEN_DATASETS.keys()) if args.all else args.datasets

    success_count = 0
    for dataset_name in datasets_to_download:
        try:
            if download_dataset(
                dataset_name,
                args.output,
                z_plane=args.z_plane,
                include_expression=args.include_expression,
                include_annotations=not args.no_annotations,
                dry_run=args.dry_run,
            ):
                success_count += 1
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(datasets_to_download)} datasets")
    if args.dry_run:
        print("(Dry run - no files were actually downloaded)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
