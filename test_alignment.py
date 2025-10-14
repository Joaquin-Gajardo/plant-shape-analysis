import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset


def stem_based_alignment(
    source_points,
    source_labels,
    target_points,
    target_labels,
    use_rotation=False,
    max_iterations=200,
):
    """
    Align plant point clouds using only stem points (label 0)

    Args:
        source_points: Points to align (N, 3)
        source_labels: Labels for source points (N,)
        target_points: Reference points (M, 3)
        target_labels: Labels for target points (M,)
        use_rotation: Whether to include rotation (False = translation only)

    Returns:
        translation: Translation vector (3,)
        rotation: Rotation matrix (3, 3) if use_rotation=True, else identity
        error: Final alignment error
    """
    # Extract stem points (label 0)
    source_stem = source_points[source_labels == 0]
    target_stem = target_points[target_labels == 0]

    print(f"Stem points - Source: {len(source_stem)}, Target: {len(target_stem)}")

    if len(source_stem) == 0 or len(target_stem) == 0:
        print("Warning: No stem points found in one or both point clouds")
        return np.zeros(3), np.eye(3), float("inf")

    if use_rotation:
        # Use full ICP with rotation on stem points
        from plant_shape_analysis.alignment.icp_alignment import align_plant_pair_icp

        aligned_stem, rotation, translation = align_plant_pair_icp(
            source_stem,
            target_stem,
            max_iterations=max_iterations,
            convergence_threshold=0.01,
        )

        # Compute error
        kdtree = cKDTree(target_stem)
        distances, _ = kdtree.query(aligned_stem, k=1)
        error = np.mean(distances**2)
    else:
        # Use translation-only ICP on stem points
        # Simple translation-only ICP
        translation = np.zeros(3)
        prev_error = float("inf")

        kdtree = cKDTree(target_stem)

        for iteration in range(max_iterations):
            transformed_source = source_stem + translation
            distances, indices = kdtree.query(transformed_source, k=1)
            closest_target_points = target_stem[indices]

            current_error = np.mean(distances**2)

            tolerance = 1e-6
            if abs(prev_error - current_error) < tolerance:
                break

            source_centroid = np.mean(transformed_source, axis=0)
            target_centroid = np.mean(closest_target_points, axis=0)

            translation_update = target_centroid - source_centroid
            translation += translation_update

            prev_error = current_error

            if np.linalg.norm(translation_update) < tolerance:
                break

        rotation = np.eye(3)
        print(f"Translation-only ICP converged in {iteration} iterations")

    return translation, rotation, error


def save_aligned_sequence(results, sequence_name, output_dir, method_name="aligned"):
    """
    Save aligned point clouds to files for manual inspection.

    Args:
        results: List of alignment results (each with 'aligned_points', 'aligned_labels', 'day')
        sequence_name: Name of the sequence
        output_dir: Directory to save files
        method_name: Name of the alignment method (for subdirectory)
    """
    output_path = Path(output_dir) / method_name / sequence_name
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for result in results:
        day = result["day"]
        points = result["aligned_points"]
        labels = result["aligned_labels"]

        # Save as PLY file with Open3D
        filename = f"day_{day:02d}_aligned.ply"
        filepath = output_path / filename

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color by label
        colors = np.zeros((len(points), 3))
        unique_labels = np.unique(labels)

        # Generate colors for each label
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == 0:  # Stem
                colors[mask] = [0.6, 0.4, 0.2]  # Brown
            else:
                # Random color for each leaf
                np.random.seed(int(label))
                colors[mask] = np.random.rand(3)

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save
        o3d.io.write_point_cloud(str(filepath), pcd)
        saved_files.append(filepath)

        print(f"  Saved Day {day}: {filepath.name}")

    # Save transformation info as JSON
    transform_info = []
    for result in results:
        info = {
            "day": int(result["day"]),
            "translation": result["translation"].tolist(),
            "rotation": (
                result["rotation"].tolist()
                if isinstance(result["rotation"], np.ndarray)
                else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ),
            "error": float(result["error"]),
        }
        transform_info.append(info)

    json_path = output_path / "transformations.json"
    with open(json_path, "w") as f:
        json.dump(transform_info, f, indent=2)

    print(f"  Saved transformation info: {json_path}")
    print(f"  Total files saved: {len(saved_files)} PLY + 1 JSON")

    return output_path


def stem_alignment(
    timepoints, use_rotation=True, max_iterations=200, align_to_first=False
):
    """
    Align timepoints using stem-based ICP.

    Args:
        timepoints: List of timepoint dictionaries
        use_rotation: Whether to use rotation
        max_iterations: Max ICP iterations
        align_to_first: If True, align all to first frame. If False, sequential (to previous)

    Returns:
        results: List of alignment results
    """
    results = []

    # First timepoint is reference (identity transform)
    ref_tp = timepoints[0]
    results.append(
        {
            "day": ref_tp["day"],
            "translation": np.zeros(3),
            "rotation": np.eye(3),
            "error": 0.0,
            "aligned_points": ref_tp["points"],
            "aligned_labels": ref_tp["labels"],
        }
    )

    for i in range(1, len(timepoints)):
        curr_tp = timepoints[i]

        if align_to_first:
            # Align to first frame (reference-based)
            target_tp = results[0]
            print(f"\nAligning Day {curr_tp['day']} to Day {target_tp['day']} (reference)")
        else:
            # Align to previous frame (sequential)
            target_tp = results[i - 1]
            print(f"\nAligning Day {curr_tp['day']} to Day {target_tp['day']} (sequential)")

        # Align current to target (using target aligned points)
        trans, rot, error = stem_based_alignment(
            curr_tp["points"],
            curr_tp["labels"],
            target_tp["aligned_points"],
            target_tp["aligned_labels"],
            use_rotation=use_rotation,
            max_iterations=max_iterations,
        )

        # Apply transformation to current points
        if use_rotation:
            aligned_points = curr_tp["points"] @ rot.T + trans
        else:
            aligned_points = curr_tp["points"] + trans

        results.append(
            {
                "day": curr_tp["day"],
                "translation": trans,
                "rotation": rot,
                "error": error,
                "aligned_points": aligned_points,
                "aligned_labels": curr_tp["labels"],
            }
        )

    return results


def main():
    """
    Test sequential stem-based ICP alignment on TrackPlant3D dataset.
    """

    # Load dataset without alignment
    dataset_path = Path("data/TrackPlant3D/versions")

    plant_dataset = PlantSequencesDataset(
        dataset_path,
        version="v1",
        use_ply=True,
        alignment_method=None,
        auto_download=True,
    )

    # Select a sequence to test
    sequence_name = "maize_control_plant1"

    # Get sequence data
    timeseries = plant_dataset.get_timeseries_by_sequence_name(sequence_name)
    timepoints = timeseries["timepoints"]

    print(f"Sequence: {sequence_name}")
    print(f"Number of timepoints: {len(timepoints)}")
    print(f"Days: {[tp['day'] for tp in timepoints]}")

    # Sequential stem alignment (align to previous)
    print("\n" + "=" * 80)
    print("SEQUENTIAL STEM ALIGNMENT (to previous frame)")
    print("=" * 80)

    sequential_results = stem_alignment(
        timepoints, use_rotation=True, align_to_first=False
    )

    # Reference-based stem alignment (align to first)
    print("\n" + "=" * 80)
    print("REFERENCE-BASED STEM ALIGNMENT (to first frame)")
    print("=" * 80)

    reference_results = stem_alignment(
        timepoints, use_rotation=True, align_to_first=True
    )

    # Save aligned sequences
    print("\n" + "=" * 80)
    print("SAVING ALIGNED SEQUENCES")
    print("=" * 80)

    print("\nSaving sequential alignment...")
    output_dir1 = save_aligned_sequence(
        sequential_results,
        sequence_name,
        output_dir="output/aligned_sequences",
        method_name="sequential_stem",
    )

    print("\nSaving reference-based alignment...")
    output_dir2 = save_aligned_sequence(
        reference_results,
        sequence_name,
        output_dir="output/aligned_sequences",
        method_name="reference_stem",
    )

    print(f"\n{'='*80}")
    print("✓ Aligned sequences saved!")
    print(f"  Sequential (to previous): {output_dir1}")
    print(f"  Reference (to first): {output_dir2}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
