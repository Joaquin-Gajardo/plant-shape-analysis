import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from plant_shape_analysis.data_preprocessing.detect_leaf_tips_trackplant3D import (
    load_point_cloud_from_txt,
    visualize_plant_open3d,
)


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
            print(abs(prev_error - current_error))
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


def simple_centroid_registration(source_points, target_points):
    """
    Simple registration by aligning centroids.
    Fast but less accurate than ICP.

    Args:
        source_points: numpy array of shape (N, 3)
        target_points: numpy array of shape (M, 3)

    Returns:
        translation: translation vector to align centroids
    """
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    translation = target_centroid - source_centroid
    return translation


def chamfer_distance(pc1, pc2):
    """
    Compute the symmetric Chamfer distance between two point clouds.
    Args:
        pc1: numpy array of shape (N, 3)
        pc2: numpy array of shape (M, 3)
    Returns:
        chamfer: float
    """
    kdtree1 = cKDTree(pc1)
    kdtree2 = cKDTree(pc2)
    dist1, _ = kdtree2.query(pc1, k=1)
    dist2, _ = kdtree1.query(pc2, k=1)
    chamfer = np.mean(dist1) + np.mean(dist2)
    return chamfer


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


if __name__ == "__main__":

    source_path = "data/4d_plant_registration_data/maize/plant1/03-13.txt"
    target_path = (
        "data/TrackPlant3D/gt_corrected_v1/maize/1_maize_control_plant1_D00.txt"
    )
    source, _ = load_point_cloud_from_txt(source_path)
    target, target_labels = load_point_cloud_from_txt(target_path)

    print("Point cloud sizes:")
    print(f"Source: {source.shape[0]} points")
    print(f"Target: {target.shape[0]} points")

    # # Method 1: Simple centroid alignment
    # print("\n=== Centroid Registration ===")
    # centroid_translation = simple_centroid_registration(source, target)
    # centroid_mean_dist, centroid_max_dist, centroid_chamfer = evaluate_registration(
    #     source, target, centroid_translation
    # )
    # print(f"Centroid translation: {centroid_translation}")
    # print(f"Mean distance: {centroid_mean_dist:.4f}")
    # print(f"Max distance: {centroid_max_dist:.4f}")
    # print(f"Chamfer distance: {centroid_chamfer:.4f}")

    # Method 2: ICP-based registration
    print("\n=== ICP Registration ===")
    icp_translation, final_error, iterations = rigid_registration_translation_only(
        source, target
    )
    icp_mean_dist, icp_max_dist, icp_chamfer = evaluate_registration(
        source, target, icp_translation
    )
    print(f"ICP translation: {icp_translation}")
    print(f"Final MSE: {final_error:.4f}")
    print(f"Mean distance: {icp_mean_dist:.4f}")
    print(f"Max distance: {icp_max_dist:.4f}")
    print(f"Chamfer distance: {icp_chamfer:.4f}")
    print(f"Converged in {iterations} iterations")

    # Apply translations to source points
    # transformed_source_centroid = source + centroid_translation
    transformed_source_ICP = source + icp_translation

    # # Save transformed source point cloud
    # combined_pc = np.vstack([transformed_source_centroid, target])
    # target_colors = np.tile([0, 0, 1], (target.shape[0], 1))  # Blue
    # source_colors = np.tile(
    #     [0, 1, 0], (transformed_source_centroid.shape[0], 1)
    # )  # Green
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(combined_pc)
    # pcd.colors = o3d.utility.Vector3dVector(np.vstack([source_colors, target_colors]))
    # o3d.io.write_point_cloud("aligned_and_target_centroid_translation.ply", pcd)

    # # Save transformed source point cloud
    # combined_pc = np.vstack([transformed_source_ICP, target])
    # target_colors = np.tile([0, 0, 1], (target.shape[0], 1))  # Blue
    # source_colors = np.tile([0, 1, 0], (transformed_source_ICP.shape[0], 1))  # Green
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(combined_pc)
    # pcd.colors = o3d.utility.Vector3dVector(np.vstack([source_colors, target_colors]))
    # o3d.io.write_point_cloud("aligned_and_target_ICP.ply", pcd)

    # Transfer semantic label
    predicted_labels, distances = transfer_semantic_labels_knn(
        transformed_source_ICP, target, target_labels
    )
    visualize_plant_open3d(transformed_source_ICP, predicted_labels)

    # Save point cloud with labels
    np.savetxt(
        "test_transformed.txt",
        np.hstack([transformed_source_ICP, predicted_labels]),
        fmt="%f %f %f %d",
    )
