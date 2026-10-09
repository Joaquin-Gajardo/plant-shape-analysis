"""Delaunay triangles must index the caller's points, so point order can't change the area."""

import numpy as np

from plant_shape_analysis.reconstruction.classical_methods import compute_leaf_area_delaunay


def test_area_ignores_point_order():
    # A flat leaf: a jittered grid (an exact grid makes Delaunay degenerate).
    rng = np.random.default_rng(0)
    g = np.linspace(0, 0.5, 21)
    xx, yy = np.meshgrid(g, g)
    xy = np.column_stack([xx.ravel(), yy.ravel()]) + rng.normal(0, 0.003, (xx.size, 2))
    points = np.column_stack([xy, np.zeros(xx.size)])
    shuffled = points[rng.permutation(len(points))]
    assert np.isclose(compute_leaf_area_delaunay(shuffled), compute_leaf_area_delaunay(points))
