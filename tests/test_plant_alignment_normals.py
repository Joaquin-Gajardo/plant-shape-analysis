"""Plant-level PCA/ICP alignment must rotate normals along with the points."""

import numpy as np
import pytest

from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset

Z = np.array([0.0, 0.0, 1.0])


def _timepoint(day, rotation):
    """A flat, elongated cloud in the xy-plane (normals +z), then rotated."""
    rng = np.random.default_rng(0)
    n = 2000
    points = np.column_stack(
        [rng.uniform(-4, 4, n), rng.uniform(-1, 1, n), rng.uniform(-0.01, 0.01, n)]
    )
    normals = np.tile(Z, (n, 1))
    return {
        "day": day,
        "points": points @ rotation.T,
        "normals": normals @ rotation.T,
        "dense_points": None,
        "dense_labels": None,
        "leaf_tip_idxs": np.array([], dtype=int),
    }


@pytest.mark.parametrize("method", ["pca", "icp"])
def test_alignment_rotates_normals(method):
    # Day 1 is day 0 turned 90 degrees about x: the plane stands up, normals point along y.
    rot_x = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    sequence = [_timepoint(0, np.eye(3)), _timepoint(1, rot_x)]
    ds = object.__new__(PlantSequencesDataset)  # the method needs no loaded data
    aligned, _ = ds.align_plant_sequence(sequence, method=method)

    moved = aligned[1]
    assert np.std(moved["points"][:, 2]) < 0.05, "alignment did not lay the plane flat"
    # The plane is back in xy, so its normals must be back along z (up to sign).
    assert np.all(np.abs(moved["normals"] @ Z) > 0.99)
