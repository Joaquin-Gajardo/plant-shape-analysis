"""Leaf PCA-axis orientation (`alignment.correct_pca_axis_with_stem`) on synthetic leaves."""

import numpy as np

from plant_shape_analysis.alignment.pca_alignment import correct_pca_axis_with_stem

X = np.array([1.0, 0.0, 0.0])


def _leaf(day, tip=False):
    """A flat leaf from its base at x=1 to its tip at x=3."""
    rng = np.random.default_rng(day)
    n = 400
    points = np.column_stack(
        [rng.uniform(1, 3, n), rng.uniform(-0.3, 0.3, n), rng.uniform(-0.01, 0.01, n)]
    )
    return {"day": day, "points": points, "leaf_tip": np.array([3.0, 0, 0]) if tip else None}


def _stem(day, x):
    """A vertical stem at x."""
    z = np.linspace(-1, 1, 50)
    return {"day": day, "points": np.column_stack([np.full_like(z, x), 0 * z, z])}


def test_stem_is_matched_by_day_not_position():
    # The leaf emerges on day 1. Day 0's stem lies past the tip, so pairing the first
    # leaf frame with the first stem frame would orient it tip -> base.
    leaves = [_leaf(1), _leaf(2)]
    stems = [_stem(0, 5.0), _stem(1, 0.5), _stem(2, 0.5)]
    for t in correct_pca_axis_with_stem(leaves, stems):
        assert np.dot(t["basis"][0], X) > 0.99, t["day"]


def test_missing_stem_day_uses_nearest_day():
    leaves = [_leaf(1), _leaf(2)]
    stems = [_stem(0, 5.0), _stem(1, 0.5), {"day": 2, "points": np.empty((0, 3))}]
    for t in correct_pca_axis_with_stem(leaves, stems):
        assert np.dot(t["basis"][0], X) > 0.99, t["day"]


def test_temporal_pass_flips_tipless_frame():
    # Day 2 has no tip and its stem lies past the tip, so the stem check orients it
    # backwards; the temporal pass must flip it to agree with the tipped day 1.
    leaves = [_leaf(1, tip=True), _leaf(2)]
    stems = [_stem(1, 0.5), _stem(2, 5.0)]
    trans = correct_pca_axis_with_stem(leaves, stems)
    assert np.dot(trans[1]["basis"][0], X) > 0.99
    assert trans[1].get("temporal_flip") is True
    assert "temporal_flip" not in trans[0]
