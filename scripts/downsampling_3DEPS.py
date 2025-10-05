"""
3DEPS-based edge-preserving downsampling of point clouds.
Taken from: https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping/blob/main/downsampling/3DEPS(python).py
"""

import numpy as np
import open3d as o3d


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        if k >= len(pts):
            return pts  # no need to downsample
        farthest_pts = np.zeros((k, pts.shape[1]), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3])
            )
        return farthest_pts


def threeDEPS_clean(points, total_samples=None, edge_ratio=0.3, filter_edges=False):
    """
    points: np.ndarray [N,3] or [N,4]
    total_samples: if None -> keep all points (only filtering), else resample
    edge_ratio: proportion of edge points when resampling
    filter_edges: if True, removes a fraction of detected edge points (outlier cleaning)
    """
    if points.shape[1] == 3:
        points = np.hstack((points, np.zeros((len(points), 1))))  # add dummy label

    # Edge detection with Open3D
    pcd = o3d.t.geometry.PointCloud(points[:, :3])
    pcd.estimate_normals(max_nn=30)
    _, mask = pcd.compute_boundary_points(radius=10, max_nn=30, angle_threshold=90)
    edge_idx = mask.numpy()
    core_idx = ~edge_idx

    edge_points = points[edge_idx, :]
    core_points = points[core_idx, :]

    # Optional filter: drop some edge points (often noisy at borders)
    if filter_edges:
        keep_frac = 0.7  # keep 70% of edge points, tune as needed
        n_keep = int(len(edge_points) * keep_frac)
        edge_points = edge_points[
            np.random.choice(len(edge_points), n_keep, replace=False)
        ]

    if total_samples is None:
        # Just return cleaned sets (all points kept)
        merged = np.vstack((core_points, edge_points))
    else:
        n_edge = int(total_samples * edge_ratio)
        n_core = total_samples - n_edge
        fps = FarthestSampler()
        edge_sample = fps(edge_points, n_edge) if n_edge > 0 else np.empty((0, 4))
        core_sample = fps(core_points, n_core) if n_core > 0 else np.empty((0, 4))
        merged = np.vstack((core_sample, edge_sample))

    return merged, edge_points, core_points


def visualize_points(core_points, edge_points, sampled=None):
    """
    Show core vs edge (and sampled if given) in Open3D.
    """
    geoms = []

    if len(core_points) > 0:
        pcd_core = o3d.geometry.PointCloud()
        pcd_core.points = o3d.utility.Vector3dVector(core_points[:, :3])
        pcd_core.paint_uniform_color([0.2, 0.8, 0.2])  # green
        geoms.append(pcd_core)

    if len(edge_points) > 0:
        pcd_edge = o3d.geometry.PointCloud()
        pcd_edge.points = o3d.utility.Vector3dVector(edge_points[:, :3])
        pcd_edge.paint_uniform_color([0.8, 0.2, 0.2])  # red
        geoms.append(pcd_edge)

    if sampled is not None:
        pcd_sampled = o3d.geometry.PointCloud()
        pcd_sampled.points = o3d.utility.Vector3dVector(sampled[:, :3])
        pcd_sampled.paint_uniform_color([0.2, 0.2, 0.8])  # blue
        geoms.append(pcd_sampled)

    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    # Load a dense cloud (Nx3 or Nx4)
    pts = np.loadtxt("data/TrackPlant3D/dense/maize/1_maize_control_plant1_D00.txt")

    # --- Option A: keep 90% (resampled) ---
    cleaned, edge, core = threeDEPS_clean(
        pts, total_samples=int(len(pts) * 0.9), edge_ratio=0.3, filter_edges=False
    )
    print("Original:", pts.shape, "-> Resampled:", cleaned.shape)

    visualize_points(core, edge, sampled=cleaned)

    # # --- Option B: just filter edges, no resampling ---
    # cleaned_all, edge_filt, core_filt = threeDEPS_clean(
    #     pts, total_samples=None, edge_ratio=0.3, filter_edges=True
    # )
    # print("Original:", pts.shape, "-> After filtering:", cleaned_all.shape)

    # visualize_points(core_filt, edge_filt)
