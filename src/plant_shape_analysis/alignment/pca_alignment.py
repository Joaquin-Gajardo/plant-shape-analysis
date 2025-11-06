"""
PCA-based alignment methods for plant point clouds.

This module provides alignment strategies based on Principal Component Analysis (PCA),
useful for aligning point clouds with similar shapes but different orientations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def compute_pca_basis(points: np.ndarray) -> np.ndarray:
    """
    Compute PCA basis vectors for a point cloud.

    Args:
        points: (N, 3) array of point coordinates

    Returns:
        basis: (3, 3) array where rows are principal components (sorted by variance)
    """
    centered = points - points.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    is_sorted = np.all(S[:-1] >= S[1:])
    if not is_sorted:
        print("Warning: Singular values not sorted descending")
    return Vt  # rows are components


def align_components_to_reference(
    components: np.ndarray, reference_components: np.ndarray
) -> np.ndarray:
    """
    Align principal components to match reference orientation.

    For each component, checks the dot product with the corresponding reference component.
    If negative, flips the component to align with the reference.

    Args:
        components: (3, 3) array of principal components (rows)
        reference_components: (3, 3) array of reference principal components (rows)

    Returns:
        aligned_components: (3, 3) array of aligned principal components
    """
    aligned_components = components.copy()

    for i in range(min(len(components), len(reference_components))):
        # Check dot product to determine if we need to flip
        dot_product = np.dot(components[i], reference_components[i])

        # If dot product is negative, flip the component to align with reference
        if dot_product < 0:
            aligned_components[i] = -aligned_components[i]

    return aligned_components


def pca_align_pair(pc1: np.ndarray, pc2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA-based alignment that doesn't require same number of points.
    Ensures consistent orientation by aligning to reference (pc2) principal components.

    Args:
        pc1: First point cloud (N, 3) - to be aligned
        pc2: Second point cloud (M, 3) - reference

    Returns:
        rotation: (3, 3) rotation matrix to align pc1 to pc2's coordinate system
        basis: (3, 3) basis vectors (principal components) for pc1 after alignment
    """
    # Get PCA components for both point clouds
    R1_raw = compute_pca_basis(pc1)
    R2 = compute_pca_basis(pc2)

    # Align pc1 components to match pc2 reference orientation
    R1 = align_components_to_reference(R1_raw, R2)

    # Compute rotation matrix to align pc1 to pc2
    R = R2.T @ R1

    # Additional check: ensure rotation doesn't introduce a flip
    # Check if determinant is negative (indicates reflection/flip)
    if np.linalg.det(R) < 0:
        print("Reflection detected: flipping first principal component")
        # If we have a reflection, flip the first principal component
        R1[0] = -R1[0]
        R = R2.T @ R1

    return R, R1


# def align_sequence_pairwise_pca(
#     timepoints: List[Dict[str, Any]],
#     normal_matching_percentile_threshold: int = 75,
#     initial_basis: Optional[List[np.ndarray]] = None,
# ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     """
#     Align sequence using PCA-based registration with sequential alignment.
#     Preserves scale differences to show growth over time.

#     Two-stage alignment process:
#     1. Sequential PCA alignment: each timepoint is aligned to the previous one (i→i-1)
#        rather than all to the first reference. This prevents alternating alignment behavior
#        for objects with changing shape.
#     2. Normal-aware Z-rotation: if normals are available, computes optimal rotation angle
#        around Z-axis to align normal directions, ensuring consistent face orientation
#        across time (e.g., upper surface always points in same direction).

#     Args:
#         timepoints: List of timepoint dictionaries containing 'points', 'normals', etc.
#         normal_matching_percentile_threshold: Distance threshold percentile for matching normals
#         initial_basis: Optional list of pre-computed basis matrices (3x3) for each timepoint.
#                       If provided, uses these instead of computing PCA from scratch.

#     Returns:
#         aligned_timepoints: List of aligned timepoint dictionaries
#         transformations: List of transformation dictionaries for each timepoint
#     """
#     if len(timepoints) < 2:
#         return timepoints, []

#     aligned_timepoints = []
#     transformations = []

#     # Sequential alignment: align each timepoint to the previous one
#     for i, tp in enumerate(timepoints):
#         points = tp["points"]
#         center = np.mean(points, axis=0)
#         centered = points - center

#         if i == 0:
#             # First timepoint stays as reference (just centered to its own center)
#             ref_center = center
#             aligned_points = points  # Keep at original position
#             rotation_matrix = np.eye(3)
#             # Use pre-computed basis if provided, otherwise use identity
#             if initial_basis is not None and i < len(initial_basis):
#                 basis = initial_basis[i]
#             else:
#                 basis = np.eye(3)
#         else:
#             # Align to previous aligned timepoint
#             prev_aligned_tp = aligned_timepoints[i - 1]
#             ref_points = prev_aligned_tp["points"]
#             ref_center = np.mean(ref_points, axis=0)
#             ref_centered = ref_points - ref_center

#             # Use pre-computed basis if provided, otherwise compute PCA
#             if initial_basis is not None and i < len(initial_basis):
#                 # Use pre-computed basis for current timepoint
#                 curr_basis = initial_basis[i]
#                 # Get reference basis from previous transformation
#                 ref_basis = transformations[i - 1]["basis"]
#                 # Compute rotation to align current basis to reference
#                 rotation_matrix = ref_basis.T @ curr_basis
#                 basis = curr_basis  # Store the basis for this timepoint
#             else:
#                 # Find optimal rotation using PCA alignment
#                 rotation_matrix, basis = pca_align_pair(centered, ref_centered)

#             # Use normals to compute optimal Z-axis rotation (if available)
#             # PCA aligns the object plane correctly, but we need to align the normal direction
#             if (
#                 tp.get("normals") is not None
#                 and prev_aligned_tp.get("normals") is not None
#             ):
#                 curr_normals = tp["normals"]
#                 prev_normals = prev_aligned_tp["normals"]

#                 # Rotate normals using the computed rotation
#                 rotated_normals = curr_normals @ rotation_matrix.T

#                 # Find nearest neighbors to match normals
#                 ref_tree = cKDTree(ref_centered + ref_center)
#                 aligned_points_temp = centered @ rotation_matrix.T + ref_center
#                 distances, indices = ref_tree.query(aligned_points_temp, k=1)

#                 # Use reliable matches for computing average direction
#                 max_distance = np.percentile(
#                     distances, normal_matching_percentile_threshold
#                 )
#                 reliable_matches = distances < max_distance

#                 if np.any(reliable_matches):
#                     # Compute average normal direction for current timepoint (after PCA rotation)
#                     curr_avg_normal = np.mean(rotated_normals[reliable_matches], axis=0)
#                     curr_avg_normal = curr_avg_normal / (
#                         np.linalg.norm(curr_avg_normal) + 1e-8
#                     )

#                     # Compute average normal direction for previous timepoint
#                     matched_prev_normals = prev_normals[indices[reliable_matches]]
#                     prev_avg_normal = np.mean(matched_prev_normals, axis=0)
#                     prev_avg_normal = prev_avg_normal / (
#                         np.linalg.norm(prev_avg_normal) + 1e-8
#                     )

#                     # Project normals onto XY plane (since we only rotate around Z)
#                     curr_xy = curr_avg_normal[:2]
#                     prev_xy = prev_avg_normal[:2]

#                     # Compute rotation angle around Z-axis to align current to previous
#                     # Using atan2 to get the angle between the two vectors in XY plane
#                     curr_angle = np.arctan2(curr_xy[1], curr_xy[0])
#                     prev_angle = np.arctan2(prev_xy[1], prev_xy[0])
#                     rotation_angle = prev_angle - curr_angle

#                     # Create rotation matrix around Z-axis
#                     cos_theta = np.cos(rotation_angle)
#                     sin_theta = np.sin(rotation_angle)
#                     rot_z = np.array(
#                         [
#                             [cos_theta, -sin_theta, 0],
#                             [sin_theta, cos_theta, 0],
#                             [0, 0, 1],
#                         ],
#                         dtype=np.float64,
#                     )

#                     # Apply Z-rotation to the PCA rotation
#                     rotation_matrix = rot_z @ rotation_matrix

#             # Apply rotation and translate to previous timepoint's center
#             aligned_points = centered @ rotation_matrix.T + ref_center

#         # Transform dense points if they exist
#         # Use same center as sparse points for perfect alignment
#         aligned_dense_points = None
#         if tp.get("dense_points") is not None:
#             dense_points = tp["dense_points"]
#             dense_centered = dense_points - center  # Use sparse points center!
#             if i == 0:
#                 aligned_dense_points = dense_points  # Keep at original position
#             else:
#                 aligned_dense_points = dense_centered @ rotation_matrix.T + ref_center

#         # Transform leaf tip if it exists
#         aligned_leaf_tip = None
#         if tp.get("leaf_tip") is not None:
#             original_tip = tp["leaf_tip"]
#             # Apply same transformation as points: center, rotate, translate
#             centered_tip = original_tip - center
#             if i == 0:
#                 aligned_leaf_tip = original_tip  # Keep at original position
#             else:
#                 aligned_leaf_tip = centered_tip @ rotation_matrix.T + ref_center

#         # Transform normals if they exist (normals are direction vectors, don't translate)
#         aligned_normals = None
#         if tp.get("normals") is not None:
#             if i == 0:
#                 aligned_normals = tp["normals"]  # First timepoint normals stay as is
#             else:
#                 # Rotate normals (no translation for direction vectors)
#                 aligned_normals = tp["normals"] @ rotation_matrix.T

#         # Preserve original timepoint structure
#         aligned_tp = tp.copy()
#         aligned_tp["points"] = aligned_points
#         aligned_tp["dense_points"] = aligned_dense_points
#         aligned_tp["leaf_tip"] = aligned_leaf_tip
#         aligned_tp["normals"] = aligned_normals
#         aligned_timepoints.append(aligned_tp)

#         # Store transformation in standard format
#         # Transform: p' = (p - center) @ rotation.T + ref_center
#         # Standard: p' = (p - center) @ rotation.T + center + translation
#         # Therefore: translation = ref_center - center
#         transformations.append(
#             {
#                 "stage": "pca_align",
#                 "day": tp["day"],
#                 "rotation": rotation_matrix,
#                 "translation": ref_center - center,
#                 "center": center,
#                 "basis": basis,
#                 "original_center": center,
#                 "reference_center": ref_center,
#                 "is_reference": i == 0,
#             }
#         )

#     return aligned_timepoints, transformations


def correct_pca_axis_with_stem(
    leaf_timepoints: List[Dict[str, Any]], stem_timepoints: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Correct PCA axis orientation using stem information to determine leaf insertion point.

    This ensures the main PCA axis consistently points from base (insertion) to tip,
    fixing sign ambiguity issues that can cause flipped leaves even without leaf tips.

    The algorithm:
    1. Compute PCA for each leaf timepoint
    2. Find closest stem point to leaf centroid (insertion point)
    3. Find closest leaf point to that stem point (base)
    4. Ensure PCA axis points from base to tip (base has negative projection, tip positive)
    5. Apply temporal consistency check using majority voting for reliability

    Args:
        leaf_timepoints: List of leaf timepoint dictionaries
        stem_timepoints: List of stem timepoint dictionaries (same days as leaves)

    Returns:
        List of transformation dictionaries with corrected "basis" field
    """
    transformations = []
    corrected_bases = []

    # First pass: correct each timepoint independently using stem information
    for idx, (leaf_tp, stem_tp) in enumerate(zip(leaf_timepoints, stem_timepoints)):
        points = leaf_tp["points"]
        center = np.mean(points, axis=0)
        centered = points - center

        # Compute fresh PCA axes
        pca_axes = compute_pca_basis(centered)
        pca_axis = pca_axes[0] / np.linalg.norm(pca_axes[0])

        # Find closest stem point to leaf centroid (insertion point approximation)
        distances_to_stem = np.linalg.norm(stem_tp["points"] - center, axis=1)
        closest_stem_point = stem_tp["points"][np.argmin(distances_to_stem)]

        # Find closest leaf point to the closest stem point (base/insertion)
        distances_to_leaf = np.linalg.norm(points - closest_stem_point, axis=1)
        closest_leaf_point = points[np.argmin(distances_to_leaf)]

        # Project the base point onto the PCA axis
        base_centered = closest_leaf_point - center
        base_projection = np.dot(base_centered, pca_axis)

        # Determine growth direction: PCA axis should point from base → tip
        # If base has positive projection, flip the axis so base has negative projection
        if base_projection > 0:
            growth_direction = -pca_axis
            flipped = True
        else:
            growth_direction = pca_axis
            flipped = False

        # Store corrected basis (all three principal components)
        corrected_basis = np.array(
            [
                growth_direction,
                pca_axes[1] / np.linalg.norm(pca_axes[1]),
                pca_axes[2] / np.linalg.norm(pca_axes[2]),
            ]
        )

        # Verify with leaf tip if available
        verification_passed = True
        if leaf_tp.get("leaf_tip") is not None:
            tip_centered = leaf_tp["leaf_tip"] - center
            tip_projection = np.dot(tip_centered, growth_direction)
            base_projection_corrected = np.dot(base_centered, growth_direction)

            # Check if base is behind origin and tip is ahead
            correctly_oriented = base_projection_corrected < 0 and tip_projection > 0

            if not correctly_oriented:
                # Flip again if verification fails
                corrected_basis[0] = -corrected_basis[0]
                verification_passed = False

        corrected_bases.append(corrected_basis)
        transformations.append(
            {
                "stage": "correct_pca_stem",
                "day": leaf_tp["day"],
                "rotation": np.eye(3),  # No rotation applied yet, just storing basis
                "translation": np.zeros(3),
                "center": center,
                "basis": corrected_basis,
                "flipped": flipped,
                "verification_passed": verification_passed,
                "base_projection": base_projection,
            }
        )

    # Second pass: temporal consistency check using majority voting
    # This is especially important for timepoints without tips
    timepoints_without_tips = [
        i for i, tp in enumerate(leaf_timepoints) if tp.get("leaf_tip") is None
    ]

    if len(timepoints_without_tips) > 0 and len(timepoints_without_tips) < len(
        leaf_timepoints
    ):
        # Get all growth directions
        growth_directions = np.array([basis[0] for basis in corrected_bases])

        # Use timepoints WITH tips as reference (more reliable)
        timepoints_with_tips = [
            i for i, tp in enumerate(leaf_timepoints) if tp.get("leaf_tip") is not None
        ]

        if len(timepoints_with_tips) > 0:
            # Use average direction of timepoints with tips as reference
            reference_direction = np.mean(
                growth_directions[timepoints_with_tips], axis=0
            )
            reference_direction = reference_direction / np.linalg.norm(
                reference_direction
            )
        else:
            # Fallback: use majority voting across all timepoints
            # Compute median direction by checking alignment between all pairs
            dot_products = growth_directions @ growth_directions.T
            # For each timepoint, count how many agree with it (dot > 0)
            agreement_counts = (dot_products > 0).sum(axis=1)
            # Use the direction with most agreement as reference
            reference_idx = np.argmax(agreement_counts)
            reference_direction = growth_directions[reference_idx]

        # Check each timepoint without tip
        for idx in timepoints_without_tips:
            current_direction = corrected_bases[idx][0]

            # Check alignment with reference
            dot_product = np.dot(current_direction, reference_direction)

            if dot_product < 0:
                # Pointing opposite direction, flip it
                corrected_bases[idx][0] = -current_direction
                transformations[idx]["basis"][0] = -current_direction
                transformations[idx]["temporal_flip"] = True

    return transformations


def align_main_axis_to_z(
    timepoints: List[Dict[str, Any]],
    basis_from_pca: Optional[List[Dict[str, Any]]] = None,
    seq_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Align the main PCA axis of each timepoint to the positive Z-axis (upward).
    This makes all objects have the same inclination (parallel to each other)
    and ensures they are not upside down.

    The main axis direction is checked first - if it points more downward than
    upward, it is flipped before computing the rotation. This ensures the rotation
    is always less than 90° and objects remain right-side up.

    Args:
        timepoints: List of aligned timepoint dictionaries (modified in-place)
        basis_from_pca: Optional list of transformation dicts from previous stage
                      with "basis" field. Used for computing main axis direction.
        seq_name: Optional sequence name for logging purposes

    Returns:
        List of transformation dictionaries for each timepoint
    """
    target_axis = np.array([0, 0, 1])  # Z-axis
    transformations = []

    for idx, tp in enumerate(timepoints):
        if tp["points"] is None or len(tp["points"]) < 3:
            transformations.append(
                {
                    "stage": "align_to_z",
                    "day": tp["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": np.zeros(3),
                }
            )
            continue

        points = tp["points"]
        center = np.mean(points, axis=0)
        centered = points - center  # Always compute centered for later rotation

        # Get main axis: either from provided basis or compute PCA
        basis = None
        if basis_from_pca is not None and idx < len(basis_from_pca):
            # Try to reuse basis from previous stage (e.g., Stage 1 stem correction)
            basis = basis_from_pca[idx].get("basis", None)

        if basis is not None:
            # Reuse the first principal component from previous stage
            main_axis = basis[0]  # First row is first principal component
        else:
            # Compute PCA from scratch
            main_axis = compute_pca_basis(centered)[0]  # First principal component

        # Normalize
        main_axis = main_axis / np.linalg.norm(main_axis)

        # Compute rotation matrix to align main_axis to Z-axis
        # Using Rodrigues' rotation formula
        v = np.cross(main_axis, target_axis)
        c = np.dot(main_axis, target_axis)

        # Check if vectors are already aligned or opposite
        if np.abs(c - 1.0) < 1e-8:
            # Already aligned, no rotation needed
            transformations.append(
                {
                    "stage": "align_to_z",
                    "day": tp["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": center,
                }
            )
            continue
        elif np.abs(c + 1.0) < 1e-8:
            # Vectors are opposite, rotate 180° around any perpendicular axis
            # Use X-axis as rotation axis
            rot_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        else:
            # General case: use Rodrigues' formula
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

        # Apply rotation (centered around object centroid)
        tp["points"] = centered @ rot_matrix.T + center

        # Rotate normals if they exist
        if tp.get("normals") is not None:
            tp["normals"] = tp["normals"] @ rot_matrix.T

        # Rotate dense points if they exist
        if tp.get("dense_points") is not None:
            centered_dense = tp["dense_points"] - center
            tp["dense_points"] = centered_dense @ rot_matrix.T + center

        # Rotate leaf tip if it exists
        if tp.get("leaf_tip") is not None:
            centered_tip = tp["leaf_tip"] - center
            tp["leaf_tip"] = centered_tip @ rot_matrix.T + center

        transformations.append(
            {
                "stage": "align_to_z",
                "day": tp["day"],
                "rotation": rot_matrix,
                "translation": np.zeros(3),
                "center": center,
            }
        )

    return transformations


def align_with_normal_frame(
    timepoints: List[Dict[str, Any]],
    basis_from_stage1: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Construct coordinate frames using PCA main axis and normals.

    Creates a consistent right-handed coordinate system where:
    - 1st axis: Main PCA axis (base→tip direction from Stage 1)
    - 3rd axis: Average normal direction (leaf surface normal)
    - 2nd axis: Cross product of 1st and 3rd axes

    This ensures all leaves have the same orientation convention based on
    their intrinsic geometry (growth direction + surface normal).

    Args:
        timepoints: List of timepoint dictionaries (modified in-place)
        basis_from_stage1: Optional list of transformations from Stage 1 with 'basis' field

    Returns:
        List of transformation dictionaries for each timepoint
    """
    transformations = []

    for idx, tp in enumerate(timepoints):
        if tp["points"] is None or len(tp["points"]) < 3:
            transformations.append(
                {
                    "stage": "normal_frame",
                    "day": tp["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": np.zeros(3),
                }
            )
            continue

        points = tp["points"]
        center = np.mean(points, axis=0)

        # Get PCA basis from Stage 1 (already corrected main axis to point base→tip)
        if basis_from_stage1 is not None and idx < len(basis_from_stage1):
            pca_basis = basis_from_stage1[idx]["basis"]  # All three principal components
            main_axis = pca_basis[0]
            pca_axis2 = pca_basis[1]
            pca_axis3 = pca_basis[2]
        else:
            # Fallback: compute PCA
            centered = points - center
            pca_basis = compute_pca_basis(centered)
            main_axis = pca_basis[0]
            pca_axis2 = pca_basis[1]
            pca_axis3 = pca_basis[2]

        main_axis = main_axis / np.linalg.norm(main_axis)
        pca_axis2 = pca_axis2 / np.linalg.norm(pca_axis2)
        pca_axis3 = pca_axis3 / np.linalg.norm(pca_axis3)

        # Get average normal direction
        if tp.get("normals") is not None and len(tp["normals"]) > 0:
            normals = tp["normals"]
            avg_normal = np.mean(normals, axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)

            # Correct sign of 3rd PCA axis to align with normal direction
            # The 3rd PCA axis should point in a direction consistent with normals
            # Make avg_normal perpendicular to main_axis first (Gram-Schmidt)
            normal_perp = avg_normal - np.dot(avg_normal, main_axis) * main_axis
            normal_perp_norm = np.linalg.norm(normal_perp)

            if normal_perp_norm > 1e-6:
                normal_perp = normal_perp / normal_perp_norm

                # Check alignment of pca_axis3 with the perpendicular component of avg_normal
                # If they point in opposite directions, flip pca_axis3
                dot_product = np.dot(pca_axis3, normal_perp)
                if dot_product < 0:
                    pca_axis3 = -pca_axis3
                    # Also flip axis2 to maintain right-handed system
                    pca_axis2 = -pca_axis2

        # Construct orthogonal frame using corrected PCA axes
        axis1 = main_axis
        axis2 = pca_axis2
        axis3 = pca_axis3

        # Create target basis (where we want these axes to point)
        # Map: axis1 → Z, axis2 → X, axis3 → Y
        target_frame = np.array([
            [0, 0, 1],  # axis1 → Z (vertical, main growth)
            [1, 0, 0],  # axis2 → X
            [0, 1, 0],  # axis3 → Y (normal direction)
        ])

        # Current frame (rows are the axes)
        current_frame = np.array([axis1, axis2, axis3])

        # Rotation matrix to align current_frame to target_frame
        # target = current @ R.T  =>  R.T = current^-1 @ target  =>  R = target.T @ current
        rotation_matrix = target_frame.T @ current_frame

        # Apply rotation around centroid
        centered = points - center
        tp["points"] = centered @ rotation_matrix.T + center

        # Rotate normals
        if tp.get("normals") is not None:
            tp["normals"] = tp["normals"] @ rotation_matrix.T

        # Rotate dense points
        if tp.get("dense_points") is not None:
            centered_dense = tp["dense_points"] - center
            tp["dense_points"] = centered_dense @ rotation_matrix.T + center

        # Rotate leaf tip
        if tp.get("leaf_tip") is not None:
            centered_tip = tp["leaf_tip"] - center
            tp["leaf_tip"] = centered_tip @ rotation_matrix.T + center

        transformations.append(
            {
                "stage": "normal_frame",
                "day": tp["day"],
                "rotation": rotation_matrix,
                "translation": np.zeros(3),
                "center": center,
                "constructed_frame": current_frame,
            }
        )

    return transformations


def align_z_rotation_sequential(
    timepoints: List[Dict[str, Any]],
    normal_matching_percentile_threshold: int = 75,
    use_normals: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sequential alignment with Z-axis rotation only.

    Aligns each timepoint to the previous one by computing optimal rotation
    around Z-axis. Can use either normals or point-based matching.

    This assumes:
    - Main PCA axis is already aligned to Z-axis
    - We only want rotation around Z, preserving vertical alignment

    Args:
        timepoints: List of timepoint dictionaries (modified in-place)
        normal_matching_percentile_threshold: Distance threshold percentile for matching
        use_normals: If True, use normals for alignment. If False, use point matching.

    Returns:
        List of transformation dictionaries for each timepoint
    """
    from scipy.spatial import cKDTree

    if len(timepoints) < 2:
        return [
            {
                "stage": "z_rotation_sequential",
                "day": timepoints[0]["day"],
                "rotation": np.eye(3),
                "translation": np.zeros(3),
                "center": (
                    np.mean(timepoints[0]["points"], axis=0)
                    if timepoints[0]["points"] is not None
                    else np.zeros(3)
                ),
                "rotation_angle_deg": 0.0,
            }
        ]

    transformations = []

    # First timepoint is reference
    center0 = np.mean(timepoints[0]["points"], axis=0)
    transformations.append(
        {
            "stage": "z_rotation_sequential",
            "day": timepoints[0]["day"],
            "rotation": np.eye(3),
            "translation": np.zeros(3),
            "center": center0,
            "rotation_angle_deg": 0.0,
        }
    )

    # Sequential alignment: align each timepoint to previous one
    for i in range(1, len(timepoints)):
        prev_tp = timepoints[i - 1]
        curr_tp = timepoints[i]

        curr_points = curr_tp["points"]
        prev_points = prev_tp["points"]
        center = np.mean(curr_points, axis=0)

        rotation_angle = 0.0

        # Try to use normals if available and requested
        if (
            use_normals
            and curr_tp.get("normals") is not None
            and prev_tp.get("normals") is not None
            and len(curr_tp["normals"]) > 0
            and len(prev_tp["normals"]) > 0
        ):
            curr_normals = curr_tp["normals"]
            prev_normals = prev_tp["normals"]

            # Find nearest neighbors to match normals
            prev_tree = cKDTree(prev_points)
            distances, indices = prev_tree.query(curr_points, k=1)

            # Use reliable matches for computing average direction
            max_distance = np.percentile(
                distances, normal_matching_percentile_threshold
            )
            reliable_matches = distances < max_distance

            if np.any(reliable_matches):
                # Compute average normal direction for current timepoint
                curr_avg_normal = np.mean(curr_normals[reliable_matches], axis=0)
                curr_avg_normal = curr_avg_normal / (
                    np.linalg.norm(curr_avg_normal) + 1e-8
                )

                # Compute average normal direction for previous timepoint
                matched_prev_normals = prev_normals[indices[reliable_matches]]
                prev_avg_normal = np.mean(matched_prev_normals, axis=0)
                prev_avg_normal = prev_avg_normal / (
                    np.linalg.norm(prev_avg_normal) + 1e-8
                )

                # Project normals onto XY plane (since we only rotate around Z)
                curr_xy = curr_avg_normal[:2]
                prev_xy = prev_avg_normal[:2]

                # Compute rotation angle around Z-axis to align current to previous
                curr_angle = np.arctan2(curr_xy[1], curr_xy[0])
                prev_angle = np.arctan2(prev_xy[1], prev_xy[0])
                rotation_angle = prev_angle - curr_angle

        # Create rotation matrix around Z-axis
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rot_z = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
            dtype=np.float64,
        )

        # Apply rotation around centroid
        centered = curr_points - center
        curr_tp["points"] = centered @ rot_z.T + center

        # Rotate normals
        if curr_tp.get("normals") is not None:
            curr_tp["normals"] = curr_tp["normals"] @ rot_z.T

        # Rotate dense points if they exist
        if curr_tp.get("dense_points") is not None:
            centered_dense = curr_tp["dense_points"] - center
            curr_tp["dense_points"] = centered_dense @ rot_z.T + center

        # Rotate leaf tip if it exists
        if curr_tp.get("leaf_tip") is not None:
            centered_tip = curr_tp["leaf_tip"] - center
            curr_tp["leaf_tip"] = centered_tip @ rot_z.T + center

        transformations.append(
            {
                "stage": "z_rotation_sequential",
                "day": curr_tp["day"],
                "rotation": rot_z,
                "translation": np.zeros(3),
                "center": center,
                "rotation_angle_deg": np.degrees(rotation_angle),
            }
        )

    return transformations


def align_z_rotation_with_normals(
    timepoints: List[Dict[str, Any]], normal_matching_percentile_threshold: int = 75
) -> List[Dict[str, Any]]:
    """
    Rotate around Z-axis to align normal directions across timepoints.

    This assumes:
    - Main PCA axis is already aligned to Z-axis
    - Normals are available and consistent across time (from plant alignment)

    Uses sequential alignment: each timepoint is aligned to the previous one
    by computing the optimal rotation angle around Z-axis based on average normal direction.

    Args:
        timepoints: List of timepoint dictionaries (modified in-place)
        normal_matching_percentile_threshold: Distance threshold percentile for matching normals

    Returns:
        List of transformation dictionaries for each timepoint
    """
    from scipy.spatial import cKDTree

    if len(timepoints) < 2:
        center0 = np.mean(timepoints[0]["points"], axis=0) if timepoints[0]["points"] is not None else np.zeros(3)
        return [
            {
                "stage": "z_rotation_normals",
                "day": timepoints[0]["day"],
                "rotation": np.eye(3),
                "translation": np.zeros(3),
                "center": center0,
                "rotation_angle_deg": 0.0,
            }
        ]

    transformations = []

    # First timepoint is reference
    center0 = np.mean(timepoints[0]["points"], axis=0)
    transformations.append(
        {
            "stage": "z_rotation_normals",
            "day": timepoints[0]["day"],
            "rotation": np.eye(3),
            "translation": np.zeros(3),
            "center": center0,
            "rotation_angle_deg": 0.0,
        }
    )

    # Sequential alignment: align each timepoint to previous one
    for i in range(1, len(timepoints)):
        prev_tp = timepoints[i - 1]
        curr_tp = timepoints[i]

        # Check if both have normals
        if (
            curr_tp.get("normals") is None
            or prev_tp.get("normals") is None
            or len(curr_tp["normals"]) == 0
            or len(prev_tp["normals"]) == 0
        ):
            center = np.mean(curr_tp["points"], axis=0)
            transformations.append(
                {
                    "stage": "z_rotation_normals",
                    "day": curr_tp["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": center,
                    "rotation_angle_deg": 0.0,
                }
            )
            continue

        curr_points = curr_tp["points"]
        curr_normals = curr_tp["normals"]
        prev_points = prev_tp["points"]
        prev_normals = prev_tp["normals"]

        # Find nearest neighbors to match normals
        prev_tree = cKDTree(prev_points)
        distances, indices = prev_tree.query(curr_points, k=1)

        # Use reliable matches for computing average direction
        max_distance = np.percentile(distances, normal_matching_percentile_threshold)
        reliable_matches = distances < max_distance

        if not np.any(reliable_matches):
            center = np.mean(curr_points, axis=0)
            transformations.append(
                {
                    "stage": "z_rotation_normals",
                    "day": curr_tp["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": center,
                    "rotation_angle_deg": 0.0,
                }
            )
            continue

        # Compute average normal direction for current timepoint
        curr_avg_normal = np.mean(curr_normals[reliable_matches], axis=0)
        curr_avg_normal = curr_avg_normal / (np.linalg.norm(curr_avg_normal) + 1e-8)

        # Compute average normal direction for previous timepoint
        matched_prev_normals = prev_normals[indices[reliable_matches]]
        prev_avg_normal = np.mean(matched_prev_normals, axis=0)
        prev_avg_normal = prev_avg_normal / (np.linalg.norm(prev_avg_normal) + 1e-8)

        # Project normals onto XY plane (since we only rotate around Z)
        curr_xy = curr_avg_normal[:2]
        prev_xy = prev_avg_normal[:2]

        # Compute rotation angle around Z-axis to align current to previous
        curr_angle = np.arctan2(curr_xy[1], curr_xy[0])
        prev_angle = np.arctan2(prev_xy[1], prev_xy[0])
        rotation_angle = prev_angle - curr_angle

        # Create rotation matrix around Z-axis
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rot_z = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
            dtype=np.float64,
        )

        # Apply rotation around centroid
        center = np.mean(curr_points, axis=0)
        centered = curr_points - center
        curr_tp["points"] = centered @ rot_z.T + center

        # Rotate normals
        if curr_tp.get("normals") is not None:
            curr_tp["normals"] = curr_tp["normals"] @ rot_z.T

        # Rotate dense points if they exist
        if curr_tp.get("dense_points") is not None:
            centered_dense = curr_tp["dense_points"] - center
            curr_tp["dense_points"] = centered_dense @ rot_z.T + center

        # Rotate leaf tip if it exists
        if curr_tp.get("leaf_tip") is not None:
            centered_tip = curr_tp["leaf_tip"] - center
            curr_tp["leaf_tip"] = centered_tip @ rot_z.T + center

        transformations.append(
            {
                "stage": "z_rotation_normals",
                "day": curr_tp["day"],
                "rotation": rot_z,
                "translation": np.zeros(3),
                "rotation_angle_deg": np.degrees(rotation_angle),
                "center": center,
            }
        )

    return transformations


def align_base_to_origin(
    timepoints: List[Dict[str, Any]], base_percentile: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Align each timepoint by bringing the base (insertion point) to the origin.

    Uses the lowest percentile of points along the main axis to robustly estimate
    the base location, similar to stem-based alignment. This is more robust than
    using just the single lowest point.

    Args:
        timepoints: List of aligned timepoint dictionaries (modified in-place)
        base_percentile: Percentile of lowest points to use for base estimation (default: 1.0)

    Returns:
        List of transformation dictionaries for each timepoint
    """
    transformations = []

    for tp in timepoints:
        if tp["points"] is not None and len(tp["points"]) > 0:
            points = tp["points"]

            # Use lowest percentile of points along Z-axis to define base
            z_coords = points[:, 2]
            z_threshold = np.percentile(z_coords, base_percentile)
            base_mask = z_coords <= z_threshold

            if np.sum(base_mask) < 1:
                # Fallback: use lowest point
                min_z = np.min(z_coords)
                shift = np.array([0, 0, -min_z])
            else:
                # Compute centroid of base points
                base_points = points[base_mask]
                base_centroid = np.mean(base_points, axis=0)
                # Shift to bring base centroid to origin
                shift = -base_centroid

            # Shift points
            tp["points"] = tp["points"] + shift

            # Shift dense points
            if tp.get("dense_points") is not None:
                tp["dense_points"] = tp["dense_points"] + shift

            # Shift leaf tip
            if tp.get("leaf_tip") is not None:
                tp["leaf_tip"] = tp["leaf_tip"] + shift

            # Normals are direction vectors - no translation needed

            transformations.append(
                {
                    "stage": "align_base",
                    "day": tp["day"],
                    "rotation": np.eye(3),
                    "translation": shift,
                    "center": np.zeros(3),
                    "base_percentile": base_percentile,
                }
            )
        else:
            transformations.append(
                {
                    "stage": "align_base",
                    "day": tp["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": np.zeros(3),
                    "base_percentile": base_percentile,
                }
            )

    return transformations
