"""
Normal estimation and consistency enforcement for plant point clouds.

This module provides utilities for estimating surface normals and ensuring
they remain consistent across temporal sequences and organ boundaries.
"""

from typing import Any, Dict, List

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def estimate_normals_for_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Estimate normals for a single point cloud using Open3D.

    Args:
        points: (N, 3) array of point coordinates

    Returns:
        normals: (N, 3) array of estimated normals
    """
    if points is None or len(points) == 0:
        return None

    # Estimate normals using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    pcd.orient_normals_consistent_tangent_plane(k=30)
    normals = np.asarray(pcd.normals)
    return normals


def estimate_normals_for_timeseries(
    timeseries: List[Dict[str, Any]], object_type: str = "plant"
) -> None:
    """
    Estimate normals for all timepoints in a timeseries (modifies in-place).

    Args:
        timeseries: List of timeseries dictionaries, each containing a list of timepoints
        object_type: Type of object ("plant" or "leaf") for logging purposes
    """
    # Check if normals are already loaded
    has_normals = any(
        timepoint.get("normals") is not None
        for ts in timeseries
        for timepoint in ts["timepoints"]
    )

    if has_normals:
        print(
            f"Warning: Normals already present in {object_type} dataset. Re-estimating anyway."
        )
        print(
            "  (Set estimate_normals=False to skip estimation and use loaded normals)"
        )

    print(f"Estimating normals for all {object_type}s in the dataset...")
    for ts in timeseries:
        for timepoint in ts["timepoints"]:
            points = timepoint["points"]
            normals = estimate_normals_for_point_cloud(points)
            if normals is not None:
                timepoint["normals"] = normals

    print("Normal estimation complete.")


def enforce_temporal_normal_consistency(
    timepoints: List[Dict[str, Any]], reference_idx: int = 0
) -> None:
    """
    Enforce temporal consistency of normals across timepoints for a sequence.
    Uses sequential propagation: compares each timepoint to the previous one.
    Processes each organ (label) separately to ensure normals remain consistent
    on the same face of each organ over time.

    This function is used for plant sequences where multiple organs exist.

    Args:
        timepoints: List of timepoint dictionaries containing 'points', 'labels', and 'normals'
        reference_idx: Index of reference timepoint (default: 0, not used in sequential mode)
    """
    if len(timepoints) < 2:
        return

    # Check if first timepoint has normals
    if "normals" not in timepoints[0] or timepoints[0]["normals"] is None:
        return

    # Sequential propagation: compare each timepoint to the previous one
    for i in range(1, len(timepoints)):
        prev_tp = timepoints[i - 1]  # Previous timepoint as reference
        curr_tp = timepoints[i]

        if "normals" not in curr_tp or curr_tp["normals"] is None:
            continue

        prev_points = prev_tp["points"]
        prev_labels = prev_tp["labels"]
        prev_normals = prev_tp["normals"]

        curr_points = curr_tp["points"]
        curr_labels = curr_tp["labels"]
        curr_normals = curr_tp["normals"]

        # Get unique organ labels from current timepoint
        unique_labels = np.unique(curr_labels)

        # Process each organ separately
        for organ_label in unique_labels:
            # Skip if this organ doesn't exist in current timepoint
            curr_organ_mask = curr_labels == organ_label
            if not np.any(curr_organ_mask):
                continue

            # Skip if organ doesn't exist in previous timepoint
            prev_organ_mask = prev_labels == organ_label
            if not np.any(prev_organ_mask):
                continue

            # Get points and normals for this organ
            curr_organ_points = curr_points[curr_organ_mask]
            curr_organ_normals = curr_normals[curr_organ_mask]
            prev_organ_points = prev_points[prev_organ_mask]
            prev_organ_normals = prev_normals[prev_organ_mask]

            # Build KD-tree for previous organ points
            prev_tree = cKDTree(prev_organ_points)

            # Find nearest neighbors in previous frame
            distances, indices = prev_tree.query(curr_organ_points, k=1)

            # For each point, check if normal should be flipped
            matched_prev_normals = prev_organ_normals[indices]

            # Compute dot product between current and previous normals
            dot_products = np.sum(curr_organ_normals * matched_prev_normals, axis=1)

            # Use majority voting: if most normals point in wrong direction, flip ALL
            # Only consider points with close matches (within reasonable distance)
            max_distance = np.percentile(
                distances, 75
            )  # Use 75th percentile as threshold
            reliable_matches = distances < max_distance

            if np.any(reliable_matches):
                avg_dot_product = np.mean(dot_products[reliable_matches])

                # If average dot product is negative, flip ALL normals for this organ
                if avg_dot_product < 0:
                    curr_organ_normals = -curr_organ_normals

            # Update normals in the full array
            curr_normals[curr_organ_mask] = curr_organ_normals

        # Update normals in timepoint
        curr_tp["normals"] = curr_normals
