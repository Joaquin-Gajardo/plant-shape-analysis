"""Stem-based plant alignment must record the transform it applies to the points."""

import json

import numpy as np

from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset


def _rotation(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _plant(day, rotation, offset):
    """A vertical stem (label 0) with two leaves (label 1), then moved rigidly."""
    rng = np.random.default_rng(day)
    z = rng.uniform(0, 10, 300)
    stem = np.column_stack([rng.normal(0, 0.1, 300), rng.normal(0, 0.1, 300), z])
    u = rng.uniform(0, 4, 200)
    leaves = np.vstack([
        np.column_stack([u, 0.2 * rng.standard_normal(200), 5 + 0.5 * u]),
        np.column_stack([-u, 0.2 * rng.standard_normal(200), 7 + 0.3 * u]),
    ])
    points = np.vstack([stem, leaves]) @ rotation.T + offset
    normals = rng.standard_normal(points.shape)
    return {
        "day": day,
        "points": points,
        "labels": np.r_[np.zeros(300, int), np.ones(400, int)],
        "normals": normals / np.linalg.norm(normals, axis=1, keepdims=True),
        "dense_points": points.copy(),  # dense = sparse, so both must land alike
        "dense_labels": None,
        "leaf_tip_idxs": np.array([], dtype=int),
    }


def test_recorded_transform_matches_applied(tmp_path):
    # Tilted and shifted differently per day, and a manual Z rotation on the
    # reference frame too, which used to go unrecorded.
    sequence = [
        _plant(0, _rotation("x", 8), np.array([1.0, 2.0, 3.0])),
        _plant(1, _rotation("z", 40) @ _rotation("x", -5), np.array([4.0, -1.0, 0.5])),
        _plant(2, _rotation("z", -70) @ _rotation("x", 12), np.array([-2.0, 3.0, 6.0])),
    ]
    ds = object.__new__(PlantSequencesDataset)  # the methods need no loaded data
    ds.manual_z_rotations = {"seq": {0: 30.0, 2: -45.0}}
    ds.verbose = False
    ds.orientation_corrected = True
    ds.dataset_path = tmp_path
    aligned, transforms = ds._align_stem_based(sequence, sequence_name="seq")

    for original, out, trans in zip(sequence, aligned, transforms):
        R = trans["rotation_matrix"]
        recorded = original["points"] @ R.T + trans["translation"] + trans["vertical_shift"]
        np.testing.assert_allclose(recorded, out["points"], atol=1e-9)
        np.testing.assert_allclose(out["dense_points"], out["points"], atol=1e-9)
        np.testing.assert_allclose(out["normals"], original["normals"] @ R.T, atol=1e-12)

    ds.save_transformations("seq", transforms)
    saved = json.loads((tmp_path / "transformations" / "seq_transformations.json").read_text())
    for original, out, tp in zip(sequence, aligned, saved["timepoints"]):
        M = np.array(tp["composed_matrix"])
        np.testing.assert_allclose(original["points"] @ M[:3, :3].T + M[:3, 3], out["points"], atol=1e-9)
        Minv = np.array(tp["inverse_matrix"])
        np.testing.assert_allclose(out["points"] @ Minv[:3, :3].T + Minv[:3, 3], original["points"], atol=1e-9)
