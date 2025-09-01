import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def chamfer_distance(pc1: NDArray, pc2: NDArray) -> np.float64:
    """
    Compute the symmetric Chamfer distance between two point clouds.
    Args:
        pc1: numpy array of shape (N, 3)
        pc2: numpy array of shape (M, 3)
    Returns:
        chamfer: float
    """
    assert pc1.shape[1] == 3
    assert pc2.shape[1] == 3
    kdtree1 = cKDTree(pc1)
    kdtree2 = cKDTree(pc2)
    dist1, _ = kdtree2.query(pc1, k=1)
    dist2, _ = kdtree1.query(pc2, k=1)
    chamfer = np.mean(dist1) + np.mean(dist2)
    return chamfer
