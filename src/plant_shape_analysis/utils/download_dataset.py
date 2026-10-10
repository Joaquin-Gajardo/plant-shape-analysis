"""
Utility functions for downloading datasets.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


def _get_dataset_config(version: str):
    """
    Get dataset configuration from PlantSequencesDataset.

    Args:
        version: Dataset version (e.g., "v1")

    Returns:
        Dictionary with download_info and expected_dirs
    """
    # Import here to avoid circular dependency
    from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset

    if version not in PlantSequencesDataset.DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset version: {version}. "
            f"Available: {list(PlantSequencesDataset.DATASET_CONFIGS.keys())}"
        )

    config = PlantSequencesDataset.DATASET_CONFIGS[version]
    return {
        "url": config["download_info"]["url"],
        "filename": config["download_info"]["filename"],
        "extract_dir": config["download_info"]["extract_dir"],
        "size_mb": config["download_info"]["size_mb"],
        "description": config["download_info"]["description"],
        "expected_dirs": [d for d in config["data_dirs"].values() if d is not None],
    }


def download_trackplant3d(
    target_dir: Optional[Path] = None,
    version: str = "v2",
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download TrackPlant3D dataset if it doesn't exist.

    Args:
        target_dir: Directory to download to. Defaults to data/TrackPlant3D/versions/,
            relative to the working directory, like the loader's dataset_path
        version: Dataset version (default: "v2", the loader's default)
        force: If True, download even if dataset exists
        verbose: Print progress messages

    Returns:
        Path to downloaded dataset directory

    Example:
        >>> from plant_shape_analysis.utils.download_dataset import download_trackplant3d
        >>> dataset_path = download_trackplant3d()
        >>> print(f"Dataset available at: {dataset_path}")
    """
    # Get dataset info from PlantSequencesDataset config
    dataset_info = _get_dataset_config(version)
    extract_dir = dataset_info["extract_dir"]

    # Determine target directory
    if target_dir is None:
        # Relative to the working directory, as TrackPlant3D's dataset_path is. The
        # package root is wrong for a pip install, where it lands inside site-packages.
        target_dir = Path("data") / "TrackPlant3D" / "versions"
    else:
        target_dir = Path(target_dir)

    dataset_path = target_dir / extract_dir

    # Check if dataset already exists. A directory alone is not enough: an empty or
    # partial one (an interrupted download, or an older archive) must not pass.
    if dataset_path.exists() and not force:
        if check_dataset_exists(dataset_path, version):
            if verbose:
                print(f"Dataset already exists at: {dataset_path}")
                print("Use force=True to re-download")
            return dataset_path
        missing = [d for d in dataset_info["expected_dirs"] if not (dataset_path / d).exists()]
        raise RuntimeError(
            f"{dataset_path} exists but is incomplete for version {version!r} "
            f"(missing: {missing}). Re-download with force=True, which adds the missing "
            "files and keeps any of your own."
        )

    # Extract remaining info
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

    # Extract into a temporary directory and verify there, so a failed or partial
    # extraction never leaves a dataset_path behind that a later call would accept.
    if verbose:
        print(f"  Extracting to {dataset_path}...")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{extract_dir}_", dir=target_dir))
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        extracted = tmp_dir / extract_dir
        missing_dirs = [d for d in expected_dirs if not (extracted / d).exists()]
        if missing_dirs:
            raise RuntimeError(
                f"Dataset extraction incomplete. Missing directories: {missing_dirs}"
            )

        if dataset_path.exists():  # force=True: update in place, keep extra files
            shutil.copytree(extracted, dataset_path, dirs_exist_ok=True)
        else:
            extracted.rename(dataset_path)

        if verbose:
            print("  ✓ Extraction complete")

    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Failed to extract dataset: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink()
            if verbose:
                print("  ✓ Cleaned up temporary files")

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
    dataset_info = _get_dataset_config(version)
    expected_dirs = dataset_info["expected_dirs"]
    return all((dataset_path / d).exists() for d in expected_dirs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download TrackPlant3D dataset")
    parser.add_argument(
        "--target-dir",
        type=str,
        help="Target directory (default: data/TrackPlant3D/versions/ in the working directory)",
    )
    parser.add_argument(
        "--version", type=str, default="v2", help="Dataset version (default: v2)"
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
