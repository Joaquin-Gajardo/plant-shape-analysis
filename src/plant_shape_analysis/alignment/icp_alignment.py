"""
ICP (Iterative Closest Point) alignment for plant point clouds.

Adapted from: https://github.com/romi/4d_plant_analysis
Original file: registration/registration_icp.py

This implementation provides point-to-point ICP registration for aligning
plant point clouds across time, handling growth and topology changes.
Note that the original implemented was designed to align skeletons.
"""

import numpy as np
from sklearn.neighbors import KDTree


def get_best_fit_transform(A, B):
    """
    Calculate the least-squares best-fit transform between corresponding 3D points A->B.

    Args:
        A: (N, 3) numpy array of corresponding 3D points
        B: (N, 3) numpy array of corresponding 3D points

    Returns:
        T: (4, 4) homogeneous transformation matrix
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert len(A) == len(B), "Point clouds must have same number of points"

    # Translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute rotation matrix using SVD
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Handle special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # Homogeneous transformation matrix
    T = np.identity(4)
    T[0:3, 0:3] = R.T
    T[3, 0:3] = t.T

    return T, R, t


def get_nearest_neighbors(points_source, points_target):
    """
    Find nearest neighbors in target point cloud for each source point.

    Args:
        points_source: (N, 3) source points
        points_target: (M, 3) target points

    Returns:
        distances: (N,) distances to nearest neighbors
        indices: (N,) indices of nearest neighbors in target
    """
    tree = KDTree(points_target)
    distances, indices = tree.query(points_source, k=1, return_distance=True)
    indices = indices.flatten()
    distances = distances.flatten()

    return distances, indices


def align_plant_pair_icp(
    points1,
    points2,
    max_iterations=500,
    convergence_threshold=0.01,
    initial_alignment="centroid",
    return_correspondences=False,
    use_bidirectional=True,
):
    """
    Align two plant point clouds using Iterative Closest Point (ICP) registration.

    This function aligns points1 (source) to points2 (target/reference) by iteratively
    finding point correspondences and computing optimal transformations.

    Args:
        points1: (N, 3) numpy array - source point cloud to be aligned
        points2: (M, 3) numpy array - target/reference point cloud
        max_iterations: Maximum number of ICP iterations (default: 500)
        convergence_threshold: Stop when cost change is below this (default: 0.01)
        initial_alignment: Initial alignment strategy - "centroid" or "identity"
        return_correspondences: If True, return final point correspondences
        use_bidirectional: If True, use bidirectional correspondences (Chamfer distance)
                          instead of unidirectional (default: True)

    Returns:
        If return_correspondences=False:
            aligned_points1: (N, 3) transformed source points aligned to target
            rotation_matrix: (3, 3) rotation matrix
            translation: (3,) translation vector

        If return_correspondences=True:
            aligned_points1, rotation_matrix, translation, correspondences
            where correspondences is (N,) array of indices into points2
    """
    # Work with copies
    points1 = np.copy(points1).T  # (3, N)
    points2 = np.copy(points2).T  # (3, M)

    # Initialize transformation
    if initial_alignment == "centroid":
        R_init = np.eye(3)
        t_init = (np.mean(points2, axis=1) - np.mean(points1, axis=1)).reshape(-1, 1)
    else:  # identity
        R_init = np.eye(3)
        t_init = np.zeros((3, 1))

    # Accumulate transformation
    R_accum = R_init
    t_accum = t_init
    R_total = R_init
    t_total = np.zeros((3, 1))

    costs = []
    correspondences = None

    for iteration in range(max_iterations):
        # Apply current transformation
        t_accum = t_accum.reshape(-1, 1)
        points1 = np.dot(R_accum, points1) + t_accum
        R_total = np.dot(R_accum, R_total)
        t_total = np.dot(R_accum, t_total) + t_accum

        if use_bidirectional:
            # Bidirectional correspondences (Chamfer distance minimization)
            # Find nearest neighbors both ways
            distances_1to2, indices_1to2 = get_nearest_neighbors(points1.T, points2.T)
            distances_2to1, indices_2to1 = get_nearest_neighbors(points2.T, points1.T)

            # Compute Chamfer distance (symmetric)
            cost = (np.mean(distances_1to2) + np.mean(distances_2to1)) / 2
            costs.append(cost)

            # Create bidirectional correspondence pairs
            # Source->Target: each source point to its nearest target
            source_points_1to2 = points1.T  # (N, 3)
            target_points_1to2 = points2.T[indices_1to2]  # (N, 3)

            # Target->Source: each target point to its nearest source (reversed)
            source_points_2to1 = points1.T[indices_2to1]  # (M, 3)
            target_points_2to1 = points2.T  # (M, 3)

            # Combine both sets of correspondences
            all_source_points = np.vstack([source_points_1to2, source_points_2to1])
            all_target_points = np.vstack([target_points_1to2, target_points_2to1])

            correspondences = indices_1to2  # Keep for backward compatibility
        else:
            # Unidirectional correspondences (original ICP)
            distances, correspondences = get_nearest_neighbors(points1.T, points2.T)
            cost = np.mean(distances)
            costs.append(cost)

            all_source_points = points1.T
            all_target_points = points2.T[correspondences]

        # Check convergence
        if iteration > 1:
            cost_change = np.abs(costs[-1] - costs[-2])
            if cost_change < convergence_threshold or costs[-1] > costs[-2] + 1e-4:
                break

        # Compute incremental transformation using all correspondences
        _, R_accum, t_accum = get_best_fit_transform(
            all_source_points, all_target_points
        )

    # Final aligned points
    aligned_points1 = points1.T

    if return_correspondences:
        return aligned_points1, R_total, t_total.flatten(), correspondences
    else:
        return aligned_points1, R_total, t_total.flatten()


def align_sequence_pairwise_icp(
    sequence_points,
    reference_idx=0,
    max_iterations=500,
    convergence_threshold=0.01,
):
    """
    Align a sequence of point clouds using pairwise ICP registration.

    Each point cloud is aligned to the reference point cloud.

    Args:
        sequence_points: List of (N_i, 3) point cloud arrays
        reference_idx: Index of reference point cloud (default: 0)
        max_iterations: Maximum ICP iterations per alignment
        convergence_threshold: ICP convergence threshold

    Returns:
        aligned_sequence: List of aligned point clouds
        transformations: List of dicts containing rotation_matrix and translation
    """
    if len(sequence_points) < 2:
        return sequence_points, []

    reference_points = sequence_points[reference_idx]
    aligned_sequence = []
    transformations = []

    for i, points in enumerate(sequence_points):
        if i == reference_idx:
            # Reference stays unchanged
            aligned_sequence.append(points)
            transformations.append(
                {
                    "rotation_matrix": np.eye(3),
                    "translation": np.zeros(3),
                    "is_reference": True,
                }
            )
        else:
            # Align to reference
            aligned_points, R, t = align_plant_pair_icp(
                points,
                reference_points,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
            )
            aligned_sequence.append(aligned_points)
            transformations.append(
                {
                    "rotation_matrix": R,
                    "translation": t,
                    "is_reference": False,
                }
            )

    return aligned_sequence, transformations
