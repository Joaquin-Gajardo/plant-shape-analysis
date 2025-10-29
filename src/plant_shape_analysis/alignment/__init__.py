"""Alignment methods for plant point clouds."""

from plant_shape_analysis.alignment.icp_alignment import (
    align_plant_pair_icp,
    align_sequence_pairwise_icp,
)
from plant_shape_analysis.alignment.pca_alignment import (  # align_sequence_pairwise_pca,
    align_base_to_origin,
    align_main_axis_to_z,
    align_with_normal_frame,
    align_z_rotation_sequential,
    align_z_rotation_with_normals,
    compute_pca_basis,
    correct_pca_axis_with_stem,
    pca_align_pair,
)
from plant_shape_analysis.alignment.stem_based_alignment import (
    align_plant_sequence_stem_based,
    stem_based_icp,
)

__all__ = [
    "align_plant_pair_icp",
    "align_sequence_pairwise_icp",
    "stem_based_icp",
    "align_plant_sequence_stem_based",
    "pca_align_pair",
    # "align_sequence_pairwise_pca",
    "align_main_axis_to_z",
    "align_z_rotation_with_normals",
    "align_z_rotation_sequential",
    "align_with_normal_frame",
    "align_base_to_origin",
    "compute_pca_basis",
    "correct_pca_axis_with_stem",
]
