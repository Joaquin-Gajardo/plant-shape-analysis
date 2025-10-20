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
    manual_z_rotations=None,
    verbose=False,
):
    """
    Align plant sequence using stem-based ICP with optional vertical correction and manual Z-rotations.

    Multi-stage alignment approach:
    - Stage 1: Stem-based ICP alignment (rotation + translation using only stem points)
    - Stage 2: Orient stem main axis parallel to Z-direction (using lower 50% of stem points)
    - Stage 3: Vertical shift correction to align stem base centroid to origin
    - Stage 4 (optional): Manual Z-axis rotations for specific timesteps

    Args:
        timepoints: List of timepoint dictionaries with 'points' and 'labels' keys
        use_rotation: Whether to use rotation in stage 1
        max_iterations: Maximum ICP iterations for stage 1
        manual_z_rotations: Dict mapping timepoint index to rotation angle in degrees
                           Example: {6: 144.0} applies 144° rotation to timepoint 6
        verbose: If True, print debug information

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

    # Stage 2: Orient stem main axis to Z-direction (using only lower 50% of stem points)
    target_axis = np.array([0, 0, 1])  # Z-axis
    for i in range(len(aligned_timepoints)):
        points = aligned_timepoints[i]["points"]
        labels = aligned_timepoints[i]["labels"]

        # Use only stem points (label 0) to compute main axis
        stem_mask = labels == 0
        stem_points = points[stem_mask]

        if len(stem_points) < 3:
            # Not enough stem points, skip rotation for this timepoint
            continue

        # Use only lower 50% of stem points (more stable base, less affected by bending)
        stem_z = stem_points[:, 2]
        z_median = np.median(stem_z)
        lower_stem_mask = stem_z <= z_median
        lower_stem_points = stem_points[lower_stem_mask]

        if len(lower_stem_points) < 3:
            # Fallback to all stem points if not enough lower points
            lower_stem_points = stem_points

        # Get stem main axis via PCA
        stem_center = np.mean(lower_stem_points, axis=0)
        stem_centered = lower_stem_points - stem_center
        _, _, Vt = np.linalg.svd(stem_centered, full_matrices=False)
        main_axis = Vt[0]  # First principal component
        main_axis = main_axis / np.linalg.norm(main_axis)

        # Ensure main axis points upward
        if np.dot(main_axis, target_axis) < 0:
            main_axis = -main_axis

        # Compute rotation to align main_axis to Z-axis
        v = np.cross(main_axis, target_axis)
        c = np.dot(main_axis, target_axis)

        if np.abs(c - 1.0) < 1e-8:
            # Already aligned
            rot_matrix = np.eye(3)
        elif np.abs(c + 1.0) < 1e-8:
            # Opposite direction, rotate 180° around X
            rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        else:
            # Rodrigues' formula
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

        # Apply rotation to all points (around origin)
        aligned_timepoints[i]["points"] = points @ rot_matrix.T

        # Update transformation - compose with existing rotation from Stage 1
        prev_rot = transformations[i]["rotation"]
        total_rot = rot_matrix @ prev_rot
        transformations[i]["rotation"] = total_rot
        transformations[i]["stage"] = "stage2_orient_to_z"

    # Stage 3: Vertical correction - align stem base centroid to origin
    for i in range(len(aligned_timepoints)):
        points = aligned_timepoints[i]["points"]
        labels = aligned_timepoints[i]["labels"]

        # Use only stem points (label 0)
        stem_mask = labels == 0
        stem_points = points[stem_mask]

        if len(stem_points) < 3:
            # Not enough stem points, use lowest point as fallback
            min_z = np.min(points[:, 2])
            shift = np.array([0, 0, -min_z])
        else:
            # Use lowest 10% of stem points to define base layer
            stem_z = stem_points[:, 2]
            z_percentile_10 = np.percentile(stem_z, 10)
            base_mask = stem_z <= z_percentile_10
            base_stem_points = stem_points[base_mask]

            if len(base_stem_points) < 1:
                # Fallback to lowest point
                min_z = np.min(stem_z)
                shift = np.array([0, 0, -min_z])
            else:
                # Compute centroid of base layer
                base_centroid = np.mean(base_stem_points, axis=0)
                # Shift to align base centroid to XY plane (z=0)
                shift = np.array([0, 0, -base_centroid[2]])

        # Apply shift to all points
        aligned_timepoints[i]["points"] = points + shift

        # Update transformation record
        transformations[i]["vertical_shift"] = shift
        transformations[i]["stage"] = "stage3_vertical_correction"

    # Stage 4: Manual Z-axis rotations (if specified)
    if manual_z_rotations:
        if verbose:
            print(
                f"\n  Stage 4: Applying manual Z-axis rotations to {len(manual_z_rotations)} timepoints"
            )

        for timepoint_idx, angle_deg in manual_z_rotations.items():
            if timepoint_idx < 0 or timepoint_idx >= len(aligned_timepoints):
                if verbose:
                    print(
                        f"    Warning: Skipping invalid timepoint index {timepoint_idx}"
                    )
                continue

            curr_aligned = aligned_timepoints[timepoint_idx]
            angle_rad = np.radians(angle_deg)

            if verbose:
                print(
                    f"    Timepoint {timepoint_idx} (Day {curr_aligned['day']}): applying {angle_deg}° Z-rotation"
                )

            # Create Z-rotation matrix
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)
            z_rot = np.array(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                dtype=np.float64,
            )

            # Apply rotation to all points
            curr_aligned["points"] = curr_aligned["points"] @ z_rot.T

            # Update transformation if not reference
            if timepoint_idx > 0:
                # Compose with existing transformation
                prev_rot = transformations[timepoint_idx]["rotation"]
                total_rot = z_rot @ prev_rot
                transformations[timepoint_idx]["rotation"] = total_rot
                transformations[timepoint_idx]["manual_z_rotation_deg"] = angle_deg
                transformations[timepoint_idx]["stage"] = "stage4_manual_z_rotation"

    return aligned_timepoints, transformations
