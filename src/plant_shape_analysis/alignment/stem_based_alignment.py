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
    manual_z_rotations=None,
    verbose=False,
):
    """
    Align plant sequence using stem-based ICP with optional vertical correction and manual Z-rotations.

    Multi-stage alignment approach:
    - Stage 1: Stem-based ICP alignment (rotation + translation using only stem points)
    - Stage 2 (optional): Manual Z-axis rotations for specific timesteps
    - Stage 3 (optional): Vertical shift correction to align lowest points to origin

    Args:
        timepoints: List of timepoint dictionaries with 'points' and 'labels' keys
        use_rotation: Whether to use rotation in stage 1
        max_iterations: Maximum ICP iterations for stage 1
        vertical_correction: If True, apply vertical correction to align lowest points (stage 3)
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

    # Stage 2: Manual Z-axis rotations (if specified)
    if manual_z_rotations:
        if verbose:
            print(
                f"\n  Stage 2: Applying manual Z-axis rotations to {len(manual_z_rotations)} timepoints"
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
                transformations[timepoint_idx]["stage"] = "stage2_manual_z_rotation"

    # Stage 3: Vertical correction - align lowest points to origin (if enabled)
    if vertical_correction:
        # For reference timepoint (first), shift so its lowest point is at z=0
        ref_min_z = np.min(aligned_timepoints[0]["points"][:, 2])
        ref_shift = np.array([0, 0, -ref_min_z])
        aligned_timepoints[0]["points"] = aligned_timepoints[0]["points"] + ref_shift
        transformations[0]["vertical_shift"] = ref_shift
        transformations[0]["stage"] = "stage3_vertical_correction"

        # For all other timepoints, shift independently so each lowest point is at z=0
        for i in range(1, len(aligned_timepoints)):
            curr_aligned = aligned_timepoints[i]

            # Find minimum z-coordinate (lowest point)
            curr_min_z = np.min(curr_aligned["points"][:, 2])

            # Shift to align lowest point to z=0
            shift = np.array([0, 0, -curr_min_z])

            # Apply shift to current aligned points
            curr_aligned["points"] = curr_aligned["points"] + shift

            # Update transformation record
            transformations[i]["vertical_shift"] = shift
            # Update stage based on whether manual rotations were applied
            if manual_z_rotations and i in manual_z_rotations:
                transformations[i]["stage"] = "stage3_vertical_correction"
            elif not transformations[i].get("manual_z_rotation_deg"):
                transformations[i]["stage"] = "stage2_vertical_correction"
            else:
                transformations[i]["stage"] = "stage3_vertical_correction"

    return aligned_timepoints, transformations
