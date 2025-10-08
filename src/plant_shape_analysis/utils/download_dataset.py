"""
Utility functions for downloading datasets.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


DATASETS = {
    "trackplant3d_v1": {
        "url": "https://polybox.ethz.ch/index.php/s/mxiZwKfCfd39Rxx/download",
        "filename": "v1.zip",
        "extract_dir": "v1",
        "size_mb": 751,
        "description": "TrackPlant3D v1 dataset with leaf keypoint annotations and dense point clouds",
        "expected_dirs": ["gt_corrected_v1", "dense"],  # keypoints are in PLY files
    }
}


def download_trackplant3d(
    target_dir: Optional[Path] = None,
    version: str = "v1",
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download TrackPlant3D dataset if it doesn't exist.

    Args:
        target_dir: Directory to download to. Defaults to data/TrackPlant3D/versions/
        version: Dataset version (default: "v1")
        force: If True, download even if dataset exists
        verbose: Print progress messages

    Returns:
        Path to downloaded dataset directory

    Example:
        >>> from plant_shape_analysis.utils.download_dataset import download_trackplant3d
        >>> dataset_path = download_trackplant3d()
        >>> print(f"Dataset available at: {dataset_path}")
    """
    # Determine target directory
    if target_dir is None:
        # Default to data/TrackPlant3D/versions/ relative to package root
        package_root = Path(__file__).parent.parent.parent.parent
        target_dir = package_root / "data" / "TrackPlant3D" / "versions"
    else:
        target_dir = Path(target_dir)

    dataset_path = target_dir / version

    # Check if dataset already exists
    if dataset_path.exists() and not force:
        if verbose:
            print(f"Dataset already exists at: {dataset_path}")
            print("Use force=True to re-download")
        return dataset_path

    # Get dataset info
    dataset_key = f"trackplant3d_{version}"
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset version: {version}. Available: {list(DATASETS.keys())}"
        )

    dataset_info = DATASETS[dataset_key]
    url = dataset_info["url"]
    filename = dataset_info["filename"]
    size_mb = dataset_info["size_mb"]
    expected_dirs = dataset_info["expected_dirs"]

    if verbose:
        print(f"Downloading TrackPlant3D {version}...")
        print(f"  Description: {dataset_info['description']}")
        print(f"  Size: ~{size_mb} MB")
        print(f"  Target: {target_dir}")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download file
    zip_path = target_dir / filename

    if verbose:
        print(f"\n  Downloading from {url}...")

    def show_progress(block_num, block_size, total_size):
        """Progress callback for urlretrieve"""
        if verbose and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=show_progress)
        if verbose:
            print("\n  ✓ Download complete")
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Failed to download dataset: {e}") from e

    # Extract zip file
    if verbose:
        print(f"  Extracting to {dataset_path}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        if verbose:
            print("  ✓ Extraction complete")

    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {e}") from e
    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
            if verbose:
                print("  ✓ Cleaned up temporary files")

    # Verify dataset structure
    missing_dirs = [d for d in expected_dirs if not (dataset_path / d).exists()]

    if missing_dirs:
        raise RuntimeError(
            f"Dataset extraction incomplete. Missing directories: {missing_dirs}"
        )

    if verbose:
        print(f"\n✓ Dataset ready at: {dataset_path}")

    return dataset_path


def check_dataset_exists(dataset_path: Path, version: str = "v1") -> bool:
    """
    Check if TrackPlant3D dataset exists at given path.

    Args:
        dataset_path: Path to dataset directory
        version: Dataset version to check structure for (default: "v1")

    Returns:
        True if dataset exists and has expected structure
    """
    if not dataset_path.exists():
        return False

    # Get expected directories for this version
    dataset_key = f"trackplant3d_{version}"
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset version: {version}")

    expected_dirs = DATASETS[dataset_key]["expected_dirs"]
    return all((dataset_path / d).exists() for d in expected_dirs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TrackPlant3D dataset")
    parser.add_argument(
        "--target-dir",
        type=str,
        help="Target directory (default: data/TrackPlant3D/versions/)",
    )
    parser.add_argument(
        "--version", type=str, default="v1", help="Dataset version (default: v1)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download if exists"
    )

    args = parser.parse_args()

    target_dir = Path(args.target_dir) if args.target_dir else None
    dataset_path = download_trackplant3d(
        target_dir=target_dir, version=args.version, force=args.force
    )

    print(f"\nDataset available at: {dataset_path}")
