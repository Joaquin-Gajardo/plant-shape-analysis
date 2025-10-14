"""
Stem-based alignment for plant point clouds.

This module provides alignment methods that use only stem points (label 0)
for registration, followed by optional vertical correction using the stem base.
This approach is particularly effective for temporal plant sequences where
only translation and small rotations are expected.
"""

import numpy as np
from scipy.spatial import cKDTree


def stem_based_icp(
    source_points,
    source_labels,
    target_points,
    target_labels,
    use_rotation=True,
    max_iterations=200,
):
    """
    Align plant point clouds using only stem points (label 0).

    Args:
        source_points: Points to align (N, 3)
        source_labels: Labels for source points (N,) - stem points have label 0
        target_points: Reference points (M, 3)
        target_labels: Labels for target points (M,) - stem points have label 0
        use_rotation: Whether to include rotation (False = translation only)
        max_iterations: Maximum ICP iterations

    Returns:
        translation: Translation vector (3,)
        rotation: Rotation matrix (3, 3) if use_rotation=True, else identity
        error: Final alignment error
    """
    # Extract stem points (label 0)
    source_stem = source_points[source_labels == 0]
    target_stem = target_points[target_labels == 0]

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
        # Translation-only ICP on stem points
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
        error = current_error

    return translation, rotation, error


def align_plant_sequence_stem_based(
    timepoints,
    use_rotation=True,
    max_iterations=200,
    vertical_correction=True,
    base_height_mm=10,
):
    """
    Align plant sequence using stem-based ICP with optional vertical correction.

    This is a two-stage alignment approach:
    - Stage 1: Stem-based ICP alignment (rotation + translation)
    - Stage 2 (optional): Vertical shift correction using stem base centroid

    Args:
        timepoints: List of timepoint dictionaries with 'points' and 'labels' keys
        use_rotation: Whether to use rotation in stage 1
        max_iterations: Maximum ICP iterations
        vertical_correction: If True, apply vertical correction in stage 2
        base_height_mm: Height in mm from stem base to use for vertical correction (default: 10mm)

    Returns:
        aligned_timepoints: List of aligned timepoint data
        transformations: List of transformation dictionaries
    """
    aligned_timepoints = []
    transformations = []

    # First timepoint is reference (identity transform)
    ref_tp = timepoints[0]
    aligned_timepoints.append(ref_tp.copy())
    transformations.append(
        {
            "day": ref_tp["day"],
            "translation": np.zeros(3),
            "rotation": np.eye(3),
            "error": 0.0,
            "stage": "reference",
            "is_reference": True,
        }
    )

    # Stage 1: Stem-based alignment
    for i in range(1, len(timepoints)):
        curr_tp = timepoints[i]
        prev_aligned = aligned_timepoints[i - 1]

        # Align current to previous (sequential alignment)
        trans, rot, error = stem_based_icp(
            curr_tp["points"],
            curr_tp["labels"],
            prev_aligned["points"],
            prev_aligned["labels"],
            use_rotation=use_rotation,
            max_iterations=max_iterations,
        )

        # Apply transformation to current points
        if use_rotation:
            aligned_points = curr_tp["points"] @ rot.T + trans
        else:
            aligned_points = curr_tp["points"] + trans

        # Create aligned timepoint
        aligned_tp = curr_tp.copy()
        aligned_tp["points"] = aligned_points

        aligned_timepoints.append(aligned_tp)
        transformations.append(
            {
                "day": curr_tp["day"],
                "translation": trans,
                "rotation": rot,
                "error": error,
                "stage": "stage1_stem_icp",
                "is_reference": False,
            }
        )

    # Stage 2: Vertical correction (if enabled)
    if vertical_correction:
        for i in range(1, len(aligned_timepoints)):
            curr_aligned = aligned_timepoints[i]
            prev_aligned = aligned_timepoints[i - 1]

            # Get stem points from current and previous aligned results
            curr_stem_mask = curr_aligned["labels"] == 0
            curr_stem = curr_aligned["points"][curr_stem_mask]

            prev_stem_mask = prev_aligned["labels"] == 0
            prev_stem = prev_aligned["points"][prev_stem_mask]

            if len(curr_stem) == 0 or len(prev_stem) == 0:
                continue

            # Get first X mm of stem from the base (using 1% percentile to avoid outliers)
            curr_min_z = np.percentile(curr_stem[:, 2], 1)
            prev_min_z = np.percentile(prev_stem[:, 2], 1)

            curr_z_threshold = curr_min_z + base_height_mm
            prev_z_threshold = prev_min_z + base_height_mm

            curr_base = curr_stem[curr_stem[:, 2] <= curr_z_threshold]
            prev_base = prev_stem[prev_stem[:, 2] <= prev_z_threshold]

            if len(curr_base) < 5 or len(prev_base) < 5:
                continue

            # Compute centroids of stem base
            curr_base_centroid = np.mean(curr_base, axis=0)
            prev_base_centroid = np.mean(prev_base, axis=0)

            # Compute vertical shift only
            shift = prev_base_centroid - curr_base_centroid

            # Apply shift to current aligned points
            curr_aligned["points"] = curr_aligned["points"] + shift

            # Update transformation record
            transformations[i]["vertical_shift"] = shift
            transformations[i]["stage"] = "stage2_vertical_correction"

    return aligned_timepoints, transformations
