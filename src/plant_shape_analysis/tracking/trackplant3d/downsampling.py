"""
3DEPS boundary-preserving downsampling (adapted from TrackPlant3D).

Original: TrackPlant3D/downsampling/3DEPS(python).py
All filesystem I/O and directory management removed; operates on numpy arrays.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


class _FarthestSampler:
    """Greedy farthest-point sampler operating on (N, 4) arrays (xyz + label)."""

    def _calc_distances(self, p0: np.ndarray, points: np.ndarray) -> np.ndarray:
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts: np.ndarray, k: int) -> np.ndarray:
        farthest_pts = np.zeros((k, 4), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3])
            )
        return farthest_pts


def _pointnet_norm(data: np.ndarray) -> np.ndarray:
    """Center and scale to unit sphere."""
    centroid = np.mean(data, axis=0)
    data = data - centroid
    scale = np.max(np.sqrt(np.sum(data ** 2, axis=1)))
    return data / scale


def downsample_3deps(
    points: np.ndarray,
    labels: np.ndarray,
    n_merge: int = 4096,
    n_out: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    3DEPS boundary-preserving downsampling.

    Separates boundary (edge) and interior (core) points using Open3D, then:
    1. Randomly samples n_merge points from each group.
    2. Applies FPS separately to core (n_out/2) and edge (n_out/2) portions.
    3. Returns PointNet-normalized coordinates + labels.

    Args:
        points: (N, 3) point cloud.
        labels: (N,) integer organ labels (0=stem, 1+=leaves).
        n_merge: size of random pre-sample from each group before FPS.
        n_out: total output points (split 50/50 between core and edge).
        seed: random seed.

    Returns:
        ds_points: (n_out, 3) PointNet-normalized coordinates.
        ds_labels: (n_out,) organ labels for each downsampled point.
    """
    rng = np.random.default_rng(seed)

    # 1. Boundary / core separation
    pcd = o3d.t.geometry.PointCloud(points.astype(np.float64))
    pcd.estimate_normals(max_nn=20)
    _, mask = pcd.compute_boundary_points(radius=10, max_nn=20, angle_threshold=90)
    edge_mask = mask.numpy().astype(bool)

    pts4 = np.column_stack([points, labels])  # (N, 4)
    edge_pts = pts4[edge_mask]
    core_pts = pts4[~edge_mask]

    # Fallback: if boundary detection yields an empty group, use all points
    if len(edge_pts) == 0:
        edge_pts = pts4
    if len(core_pts) == 0:
        core_pts = pts4

    # 2. Random sampling n_merge from each group (with replacement if needed)
    core_idx = rng.integers(0, len(core_pts), n_merge)
    edge_idx = rng.integers(0, len(edge_pts), n_merge)
    merged = np.vstack([core_pts[core_idx], edge_pts[edge_idx]]).astype(np.float32)

    # 3. FPS on core and edge halves separately
    n_core_out = n_out - n_out // 2
    n_edge_out = n_out // 2
    sampler = _FarthestSampler()
    sample_pts = np.vstack([
        sampler(merged[:n_merge], n_core_out),
        sampler(merged[n_merge:], n_edge_out),
    ])  # (n_out, 4)

    # 4. PointNet normalize coordinates
    return _pointnet_norm(sample_pts[:, :3]), sample_pts[:, 3].astype(np.int64)
