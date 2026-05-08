"""
Utility functions for loading and manipulating plant point clouds.

This module provides I/O operations for point clouds in various formats
used by the TrackPlant3D dataset (PLY and TXT formats).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


def load_point_cloud(
    file_path: Path, use_ply: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load point cloud from txt or ply file, including normals if available.

    Args:
        file_path: Path to point cloud file
        use_ply: If True, load from PLY file; otherwise load from TXT file

    Returns:
        points: (N, 3) array of point coordinates
        labels: (N,) array of organ labels
        normals: (N, 3) array of normals, or None if not available
        predicted_labels: (N,) array of predicted organ labels, or None if field absent
    """
    if use_ply:
        pcd = o3d.t.io.read_point_cloud(str(file_path))
        points = pcd.point.positions.numpy()
        labels = pcd.point.organ_label.numpy().flatten().astype(int)
        normals = pcd.point.normals.numpy() if "normals" in pcd.point else None
        predicted_labels = (
            pcd.point.predicted_organ_label.numpy().flatten().astype(int)
            if "predicted_organ_label" in pcd.point
            else None
        )
        return points, labels, normals, predicted_labels
    else:
        data = np.loadtxt(file_path)
        points = data[:, :3]
        labels = data[:, 3].astype(int)
        return points, labels, None, None


def load_dense_point_cloud(
    sparse_file_path: Path, dense_path: Optional[Path], use_ply: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load dense point cloud for a specific sequence and day if available.

    Args:
        sparse_file_path: Path to the sparse point cloud file (used to determine crop name)
        dense_path: Path to dense point cloud directory, or None if not available
        use_ply: If True, load from PLY file; otherwise load from TXT file

    Returns:
        points: (N, 3) array of point coordinates, or None if not found
        labels: (N,) array of organ labels, or None if not found
        normals: (N, 3) array of normals, or None if not available/found
        predicted_labels: (N,) array of predicted organ labels, or None
    """
    if dense_path is None:
        return None, None, None, None
    crop_name = sparse_file_path.parent.name
    dense_file_path = dense_path / crop_name / sparse_file_path.name
    if dense_file_path.exists():
        return load_point_cloud(dense_file_path, use_ply=use_ply)
    return None, None, None, None


def load_leaf_tips(
    point_cloud_path: Path,
    leaf_tips_path: Optional[Path] = None,
    use_ply: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load leaf tips for a specific sequence and day.

    Args:
        point_cloud_path: Path to the point cloud file
        leaf_tips_path: Path to leaf tips directory (only used for TXT format)
        use_ply: If True, load from PLY scalar field; otherwise from TXT file

    Returns:
        global_indices: (M,) array of indices into the point cloud
        coordinates: (M, 3) array of leaf tip coordinates
    """
    crop_name = point_cloud_path.parent.name

    if use_ply:
        # For PLY: read is_leaf_tip scalar field from main point cloud
        pcd = o3d.t.io.read_point_cloud(str(point_cloud_path))
        points = pcd.point.positions.numpy()
        is_leaf_tip = pcd.point.is_leaf_tip.numpy().flatten().astype(bool)

        # Get indices where is_leaf_tip is True
        global_indices = np.where(is_leaf_tip)[0]
        coordinates = points[global_indices]

        return global_indices, coordinates
    else:
        # For TXT: load from comma-delimited txt file
        if leaf_tips_path is None:
            return np.array([]), np.array([])

        file_path = leaf_tips_path / crop_name / point_cloud_path.name

        # Load with comma delimiter - format: global_index, x, y, z
        global_indices = np.array([])
        coordinates = np.array([])
        if file_path.exists():
            data = np.loadtxt(file_path, delimiter=",")
            if data.ndim == 1:  # Single row
                data = data.reshape(1, -1)

            # Extract global indices and coordinates
            global_indices = data[:, 0].astype(int)
            coordinates = data[:, 1:4]
        # Note: Leaf tips may not exist for all files, which is expected

        return global_indices, coordinates


def needs_orientation_correction(sequence_name: str) -> bool:
    """
    Check if sequence needs Y->Z orientation correction.

    Some crops in the dataset (sorghum, tobacco, tomato1) use Y as the
    vertical axis instead of Z, requiring correction.

    Args:
        sequence_name: Name of the sequence (e.g., "sorghum_control_01")

    Returns:
        True if correction is needed, False otherwise
    """
    # Sorghum, tobacco, and tomato1 have Y as up instead of Z
    crops_to_correct = ["sorghum", "tobacco", "tomato1"]
    return any(sequence_name.startswith(crop) for crop in crops_to_correct)


def correct_orientation(
    points: np.ndarray, normals: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply 90° rotation around X axis to make Z up instead of Y.

    Args:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of normals, or None

    Returns:
        rotated_points: (N, 3) array of rotated point coordinates
        rotated_normals: (N, 3) array of rotated normals, or None if input was None
    """
    # Rotation matrix for 90° around X: Y -> Z, Z -> -Y, X -> X
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    rotated_points = points @ R.T
    rotated_normals = None
    if normals is not None:
        rotated_normals = normals @ R.T

    return rotated_points, rotated_normals
