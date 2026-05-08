"""
High-level TrackPlant3D tracking pipeline for a labeled point cloud sequence.
"""

from __future__ import annotations

import numpy as np

from plant_shape_analysis.tracking.trackplant3d._cpd import DeformableRegistration
from plant_shape_analysis.tracking.trackplant3d.downsampling import downsample_3deps
from plant_shape_analysis.tracking.trackplant3d.organ_matching import match_organs


def run_tracking_pipeline(
    sequence_points: list[np.ndarray],
    sequence_labels: list[np.ndarray],
    seed: int = 0,
) -> list[np.ndarray]:
    """
    Run the TrackPlant3D tracking pipeline on a sequence of labeled point clouds.

    Produces temporally consistent organ labels (0=stem, 1,2,...=individual leaves)
    by combining 3DEPS downsampling, CPD non-rigid registration, and Hungarian
    organ matching across consecutive frames.

    Args:
        sequence_points: list of (N_t, 3) point clouds, one per timepoint.
        sequence_labels: list of (N_t,) predicted organ labels per timepoint
                        (e.g. from PSegNet; 0=stem, 1,2,...=leaves).
        seed: random seed passed to 3DEPS downsampling (incremented per timepoint).

    Returns:
        tracked_labels: list of (N_t,) int64 arrays with temporally consistent labels.
    """
    T = len(sequence_points)
    assert T == len(sequence_labels), "points and labels must have the same length"
    assert T > 0, "Empty sequence"

    if T == 1:
        return [sequence_labels[0].copy()]

    # ── Step 1: Downsample all timepoints ────────────────────────────────────
    ds_pts: list[np.ndarray] = []   # (256, 3) normalized, per timepoint
    ds_labs: list[np.ndarray] = []  # (256,)  PSegNet labels for selected points

    for t in range(T):
        pts_norm, labs = downsample_3deps(
            sequence_points[t], sequence_labels[t], seed=seed + t
        )
        ds_pts.append(pts_norm)
        ds_labs.append(labs)

    # ── Step 2: Track across consecutive frames ───────────────────────────────
    tracked_ds_labs: list[np.ndarray | None] = [None] * T
    tracked_ds_labs[0] = ds_labs[0].copy()
    start_label = int(np.max(ds_labs[0]))

    for t in range(1, T):
        # CPD: deform current frame (Y) toward previous frame (X)
        reg = DeformableRegistration(X=ds_pts[t - 1], Y=ds_pts[t], low_rank=False)
        TY, _ = reg.register()  # TY: (256, 3) CPD-deformed current coords

        new_ds_labs, start_label = match_organs(
            points_prev=ds_pts[t - 1],
            labels_prev=tracked_ds_labs[t - 1],
            points_curr_reg=TY,
            labels_curr=ds_labs[t],
            start_label=start_label,
        )
        tracked_ds_labs[t] = new_ds_labs

    # ── Step 3: Propagate labels to full resolution ───────────────────────────
    tracked_labels: list[np.ndarray] = []
    for t in range(T):
        full_tracked = _propagate_labels(
            full_labels=sequence_labels[t],
            ds_old_labels=ds_labs[t],
            ds_new_labels=tracked_ds_labs[t],
        )
        tracked_labels.append(full_tracked)

    return tracked_labels


def _propagate_labels(
    full_labels: np.ndarray,
    ds_old_labels: np.ndarray,
    ds_new_labels: np.ndarray,
) -> np.ndarray:
    """
    Map tracked labels from the downsampled cloud to the full-resolution cloud.

    For each unique organ ID in ds_old_labels, finds its tracked counterpart in
    ds_new_labels and assigns it to every full-res point that shares that organ ID.
    Any organ present in full_labels but absent from ds_old_labels is assigned
    the tracked label of the numerically closest organ in ds_old_labels.
    """
    # Build old_id → new_tracked_id mapping from the downsampled cloud
    old_to_new: dict[int, int] = {}
    for organ_id in np.unique(ds_old_labels):
        idx = np.where(ds_old_labels == organ_id)[0]
        old_to_new[int(organ_id)] = int(ds_new_labels[idx[0]])

    ds_keys = sorted(old_to_new.keys())
    full_tracked = np.empty(len(full_labels), dtype=np.int64)

    for organ_id in np.unique(full_labels):
        mask = full_labels == organ_id
        if organ_id in old_to_new:
            full_tracked[mask] = old_to_new[organ_id]
        else:
            # Organ was present in full-res but missed by 3DEPS — use nearest by ID
            nearest = ds_keys[int(np.argmin([abs(organ_id - k) for k in ds_keys]))]
            full_tracked[mask] = old_to_new[nearest]

    return full_tracked
