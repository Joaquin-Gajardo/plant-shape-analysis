import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from plant_shape_analysis.data_preprocessing.detect_leaf_tips_trackplant3D import (
    load_point_cloud_from_txt,
    visualize_plant_open3d,
)
from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset
from plant_shape_analysis.utils.metrics import chamfer_distance


def rigid_registration_translation_only(
    source_points, target_points, max_iterations=200, tolerance=1e-6
):
    """
    Rigid registration finding optimal translation between two point clouds.
    Assumes rotation is already aligned, only estimates translation.

    Args:
        source_points: numpy array of shape (N, 3) - smaller point cloud
        target_points: numpy array of shape (M, 3) - larger point cloud
        max_iterations: maximum number of ICP iterations
        tolerance: convergence tolerance for translation change

    Returns:
        translation: optimal translation vector (3,)
        final_error: final mean squared error
        iterations: number of iterations used
    """

    source = np.array(source_points, dtype=np.float64)
    target = np.array(target_points, dtype=np.float64)

    print("Max iterations:", max_iterations)

    # Initialize translation to zero
    translation = np.zeros(3)
    prev_error = float("inf")

    # Build KD-tree for fast nearest neighbor search on larger target cloud
    kdtree = cKDTree(target)

    for iteration in range(max_iterations):
        # Apply current translation to source points
        transformed_source = source + translation

        # Find nearest neighbors in target for each transformed source point
        distances, indices = kdtree.query(transformed_source, k=1)
        closest_target_points = target[indices]

        # Calculate current mean squared error
        current_error = np.mean(distances**2)

        # Check for convergence
        if abs(prev_error - current_error) < tolerance:
            print(
                f"Change in error is less than tolerance: {abs(prev_error - current_error)} < {tolerance}. Stopping optimization",
            )
            break

        # Compute optimal translation using centroids
        # The optimal translation minimizes sum of squared distances
        source_centroid = np.mean(transformed_source, axis=0)
        target_centroid = np.mean(closest_target_points, axis=0)

        # Update translation
        translation_update = target_centroid - source_centroid
        translation += translation_update

        prev_error = current_error

        # Early stopping if translation update is very small
        if np.linalg.norm(translation_update) < tolerance:
            print(
                f"Norm of translation update lower than tolerance: {np.linalg.norm(translation_update)} < {tolerance}. Stopping optimization",
            )
            break

    return translation, current_error, iteration + 1


def evaluate_registration(source_points, target_points, translation):
    """
    Evaluate registration quality by computing mean distance to nearest neighbors and Chamfer distance.

    Args:
        source_points: original source points
        target_points: target points
        translation: computed translation vector

    Returns:
        mean_distance: mean distance from transformed source to nearest target points
        max_distance: maximum distance
        chamfer: symmetric Chamfer distance
    """
    transformed_source = source_points + translation
    kdtree = cKDTree(target_points)
    distances, _ = kdtree.query(transformed_source, k=1)
    chamfer = chamfer_distance(transformed_source, target_points)
    return np.mean(distances), np.max(distances), chamfer


def transfer_semantic_labels_knn(
    source_points,
    target_points,
    target_labels,
    k=1,
    max_distance=None,
    default_label=0,
):
    """
    Transfer semantic labels from labeled target cloud to unlabeled source cloud.
    We stay consistent with the terminology used for alignment where the target is
    the small labeled point cloud. Uses nearest neighbor to assign labels.

    Args:
        source_points: numpy array (N,) - unlabeled point cloud (larger, e.g., 250k)
        target_points: numpy array (M, 3) - labeled point cloud (smaller, e.g., 10k)
        target_labels: numpy array (N,) - semantic labels for target points
        k: int - number of nearest neighbors to consider (1 for closest, >1 for voting)
        max_distance: float - maximum distance to consider for labeling (None = no limit)
        default_label: int/str - label for points too far from any source point

    Returns:
        target_labels: numpy array (M,) - predicted labels for target points
        distances: numpy array (M,) - distance to nearest labeled point
    """

    # Build KD-tree on the smaller labeled cloud for efficiency
    kdtree = cKDTree(target_points)

    if k == 1:
        # Simple nearest neighbor
        distances, indices = kdtree.query(source_points, k=1)
        source_labels = target_labels[indices]
    else:
        raise NotImplementedError(
            "k > 1 not implemented yet, we recommend using 1 neighbor to avoid ambiguity"
        )

    # Apply distance threshold if specified
    if max_distance is not None:
        far_mask = distances > max_distance
        target_labels[far_mask] = default_label
        print(
            f"Set {np.sum(far_mask)} points to default label (too far from labeled points)"
        )

    return source_labels, distances


def get_trackplant3D_scans_paths(dataset_path: Path, crop: str) -> list[Path]:
    """
    Get all TrackPlant3D scan paths for a specific crop.

    Args:
        dataset_path: Path to TrackPlant3D dataset
        crop: Crop name ('maize' or 'tomato2')

    Returns:
        List of paths to scan files
    """
    plant_dataset = PlantSequencesDataset(dataset_path)
    trackplant3D = []
    for _, path in sorted(
        plant_dataset.get_sequences_by_crop(crop).items(), key=lambda x: x[0]
    ):
        trackplant3D.extend(path)
    return trackplant3D


def get_4dplantreg_scans_paths(base_path: Path, crop: str) -> list[Path]:
    """
    Get all 4D plant registration scan paths for a specific crop.

    Args:
        base_path: Path to 4d_plant_registration_data directory
        crop: Crop name ('maize' or 'tomato')

    Returns:
        List of paths to scan files
    """
    crop_path = base_path / crop
    scans_list = []
    for plant in sorted(crop_path.iterdir()):
        for seq in sorted(plant.iterdir()):
            scans_list.append(seq)
    return scans_list


def process_pairs(
    source_files: list[Path],
    target_files: list[Path],
    save_path: str | Path,
) -> dict:
    """
    Align source points to target points in batches using ICP, followed by
    evaluation and transfering of semantic labels.

    Args:
        source_files: List of paths to 4D plant registration files (dense)
        target_files: List of paths to TrackPlant3D files (sparse, labeled)
        save_path: Directory to save the dense labeled point clouds
        save_combined: Whether to save combined visualization files

    Returns:
        Dictionary with alignment results and metadata
    """
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    save_path.mkdir(parents=True, exist_ok=True)

    results_report = {}

    for source_path, target_path in zip(source_files, target_files):
        target_file_name = target_path.name
        print(f"Processing: {source_path} -> {target_path}")

        # Load point clouds
        source_points, _ = load_point_cloud_from_txt(source_path)
        target_points, target_labels = load_point_cloud_from_txt(target_path)

        print(f"  Source: {source_points.shape[0]} points")
        print(
            f"  Target: {target_points.shape[0]} points, {len(np.unique(target_labels))} leaf instances"
        )

        # Find and perform ICP transformation
        print(f"  Running ICP alignment...")
        translation, final_error, iters = rigid_registration_translation_only(
            source_points, target_points
        )
        transformed_source = source_points + translation

        # Evaluate registration quality
        icp_mean_dist, icp_max_dist, icp_chamfer = evaluate_registration(
            source_points, target_points, translation
        )

        print(f"  ICP converged in {iters} iterations")
        print(f"  Mean distance: {icp_mean_dist:.4f}, Chamfer: {icp_chamfer:.4f}")

        # Transfer semantic labels
        print(f"  Transferring labels...")
        predicted_labels, _ = transfer_semantic_labels_knn(
            transformed_source, target_points, target_labels
        )

        # Save transformed file with labels
        save_file_path = save_path / target_file_name
        np.savetxt(
            save_file_path,
            np.hstack([transformed_source, predicted_labels]),
            fmt="%f %f %f %d",
        )
        print(f"  Saved: {save_file_path}")

        # Store results
        results_report[target_file_name] = {
            "file_path": str(save_file_path),
            "source_path": str(source_path),
            "reference_path": str(target_path),
            "source_points": int(source_points.shape[0]),
            "target_points": int(target_points.shape[0]),
            "num_labels": int(target_labels.max()),
            "final_error": float(final_error),
            "iterations": int(iters),
            "icp_mean_dist": float(icp_mean_dist),
            "icp_max_dist": float(icp_max_dist),
            "icp_chamfer": float(icp_chamfer),
            "translation": translation.tolist(),
        }

    # Save report
    report_name = "alignment_results.json"
    if (save_path.parent / report_name).exists():
        with open(f"{save_path.parent / report_name}", "r") as f:
            existing_report = json.load(f)
            combined_results_report = {**existing_report, **results_report.copy()}

    with open(save_path.parent / report_name, "w") as f:
        json.dump(combined_results_report, f, indent=2)

    return results_report


def main():
    """
    This function will process all matched maize and tomato files for ICP alignment and label transfer,
    from the labeled but downsampled TrackPlant3D to the dense but unlabeled 4D plant registration datasets.
    Note that only maize and tomato are present in the 4d Plant Registration dataset. In total 26 maize and 27
    tomato plant scans will be processed, from 3 different plant sequences of each crop.
    """
    parser = argparse.ArgumentParser(
        description="Process all matched maize and tomato files for ICP alignment and label transfer"
    )
    parser.add_argument(
        "--trackplant3d-path",
        type=str,
        default="data/TrackPlant3D",
        help="Path to TrackPlant3D dataset directory. Downloaded from"
        " https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping",
    )
    parser.add_argument(
        "--plantreg4d-path",
        type=str,
        default="data/4d_plant_registration_data",
        help="Path to 4d_plant_registration_data directory. Downloaded from https://www.ipb.uni-bonn.de/data/4d-plant-registration/",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/TrackPlant3D/dense",
        help="Output directory for dense labeled point clouds",
    )

    args = parser.parse_args()

    trackplant3d_path = Path(args.trackplant3d_path)
    plantreg4d_path = Path(args.plantreg4d_path)
    output_path = Path(args.output_path)

    # Now we get the dense

    print("=== Processing Maize Files ===")
    maize_trackplant3d = get_trackplant3D_scans_paths(trackplant3d_path, "maize")
    maize_4dplantreg = get_4dplantreg_scans_paths(plantreg4d_path, "maize")

    print(f"Found {len(maize_trackplant3d)} TrackPlant3D maize files")
    print(f"Found {len(maize_4dplantreg)} 4D plant registration maize files")

    # Process maize files
    maize_output = output_path / "maize"
    maize_results = process_pairs(maize_4dplantreg, maize_trackplant3d, maize_output)

    print("\n=== Processing Tomato Files ===")
    tomato_trackplant3d = get_trackplant3D_scans_paths(trackplant3d_path, "tomato2")
    tomato_4dplantreg = get_4dplantreg_scans_paths(plantreg4d_path, "tomato")

    print(f"Found {len(tomato_trackplant3d)} TrackPlant3D tomato files")
    print(f"Found {len(tomato_4dplantreg)} 4D plant registration tomato files")

    # Process tomato files
    tomato_output = output_path / "tomato"
    tomato_results = process_pairs(
        tomato_4dplantreg, tomato_trackplant3d, tomato_output
    )

    print(f"\n=== Processing Complete ===")
    print(f"Processed {len(maize_results)} maize files")
    print(f"Processed {len(tomato_results)} tomato files")
    print(f"Total: {len(maize_results) + len(tomato_results)} files")


if __name__ == "__main__":
    main()
