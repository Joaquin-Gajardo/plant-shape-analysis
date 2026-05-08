"""
Organ matching via Hungarian assignment (adapted from TrackPlant3D).

Original: TrackPlant3D/tracking/utils.py and tracking/organ_matching.py.
Filesystem I/O removed; all functions operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1000.0
_SML = 1e-6


def _array2samples_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean minimum distance from b to a (one direction of Chamfer)."""
    # For each point in b, find the closest point in a
    diffs = a[None, :, :] - b[:, None, :]           # (|b|, |a|, 3)
    dists = np.linalg.norm(diffs, axis=-1).min(axis=1)  # (|b|,)
    return float(dists.mean())


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point sets."""
    return _array2samples_distance(a, b) + _array2samples_distance(b, a)


def build_matrix(pc_bf: np.ndarray, pc_af: np.ndarray) -> np.ndarray:
    """
    Build a (af_class × bf_class) organ matching cost matrix.

    Cost = 0.7 * chamfer_distance + 0.1 * centroid_distance + 0.2 * relative_distance.

    Args:
        pc_bf: (N, 4) array [x, y, z, tracked_label] for the previous frame.
        pc_af: (M, 4) array [x, y, z, raw_label] for the current (registered) frame.

    Returns:
        cost_array: (af_class, bf_class) float array.
    """
    bf_class = int(pc_bf[:, -1].max()) + 1
    af_class = int(pc_af[:, -1].max()) + 1
    bf_center = pc_bf[:, :3].mean(axis=0)
    af_center = pc_af[:, :3].mean(axis=0)
    cost = np.zeros((af_class, bf_class))
    for ca in range(af_class):
        for cb in range(bf_class):
            bf_pts = pc_bf[pc_bf[:, -1] == cb, :3]
            af_pts = pc_af[pc_af[:, -1] == ca, :3]
            if len(bf_pts) == 0 or len(af_pts) == 0:
                cost[ca, cb] = _BIG
                continue
            cd = chamfer_distance(bf_pts, af_pts)
            centroid_d = np.linalg.norm(bf_pts.mean(0) - af_pts.mean(0))
            bf_rel = np.linalg.norm(bf_pts - bf_center, axis=1).mean()
            af_rel = np.linalg.norm(af_pts - af_center, axis=1).mean()
            cost[ca, cb] = 0.7 * cd + 0.1 * centroid_d + 0.2 * abs(bf_rel - af_rel)
    return cost


def extend_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Extend cost matrix with dummy rows/cols to handle organ births and deaths.

    Returns square (m+n) × (m+n) matrix suitable for linear_sum_assignment.
    """
    m, n = matrix.shape
    B = np.full((m, m), _BIG)
    C = np.full((n, n), _BIG)
    D = np.full((n, m), _SML)
    for r in range(m):
        B[r, r] = _SML if matrix[r].max() == matrix[r].min() else 0.9 * matrix[r].min()
    for c in range(n):
        C[c, c] = _SML if matrix[:, c].max() == matrix[:, c].min() else 0.9 * matrix[:, c].min()
    return np.block([[matrix, B], [C, D]])


def assign_new_label(
    init_label: np.ndarray,
    old_label: np.ndarray,
    match_index: np.ndarray,
    start_lab: int,
) -> tuple[int, np.ndarray]:
    """
    Assign temporally consistent labels to the current frame.

    Args:
        init_label: tracked labels of the previous downsampled frame.
        old_label: raw predicted labels of the current downsampled frame.
        match_index: col_ind from linear_sum_assignment on the extended cost matrix.
        start_lab: current highest label value used so far.

    Returns:
        next_start: updated highest label value.
        pseudo_lab: (M,) new consistent labels for the current frame.
    """
    pseudo_lab = np.full_like(old_label, -1, dtype=np.int64)
    next_start = max(int(init_label.max()), start_lab)
    for i in range(len(match_index)):
        if i not in old_label:
            continue
        if match_index[i] > int(init_label.max()):
            next_start += 1
            pseudo_lab[old_label == i] = next_start
        else:
            pseudo_lab[old_label == i] = match_index[i]
    assert -1 not in pseudo_lab, (
        "Some current-frame organs were not assigned a label. "
        "Check that all label values in old_label appear in the matching."
    )
    return next_start, pseudo_lab


def match_organs(
    points_prev: np.ndarray,
    labels_prev: np.ndarray,
    points_curr_reg: np.ndarray,
    labels_curr: np.ndarray,
    start_label: int,
) -> tuple[np.ndarray, int]:
    """
    Match organs between two consecutive downsampled frames.

    Args:
        points_prev: (K, 3) 3DEPS-normalized coords for the previous frame.
        labels_prev: (K,) tracked labels for the previous frame.
        points_curr_reg: (K, 3) CPD-registered coords for the current frame.
        labels_curr: (K,) raw predicted labels for the current frame.
        start_label: current highest label value used so far.

    Returns:
        new_labels: (K,) temporally consistent labels for the current frame.
        next_start: updated highest label value.
    """
    d_before = np.column_stack([points_prev, labels_prev])
    d_after = np.column_stack([points_curr_reg, labels_curr])
    cost_matrix = build_matrix(d_before, d_after)
    final_matrix = extend_matrix(cost_matrix)
    _, col_ind = linear_sum_assignment(final_matrix)
    next_start, pseudo_label = assign_new_label(labels_prev, labels_curr, col_ind, start_label)
    return pseudo_label, next_start
