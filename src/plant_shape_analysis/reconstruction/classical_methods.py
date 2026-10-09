"""
Classical surface reconstruction methods for leaf point clouds.

Reimplemented from lidar_leaf_properties (https://github.com/mattbv/lidar_leaf_properties) and
surface-models-source-codes (https://gitlab.unistra.fr/plant-leaf-area-estimation-gvc-2022/surface-models-source-codes)
for benchmarking against learning-based approaches.
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def remove_duplicates(
    points: np.ndarray, tolerance: float = 1e-6, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Remove duplicate points from point cloud.

    The result is in lexicographic order, not input order. With `return_indices`, also
    returns where each kept point sits in the input.
    """
    if len(points) == 0:
        return (points, np.arange(0)) if return_indices else points

    # Use lexicographic sorting to find duplicates
    sorted_indices = np.lexsort(points.T)
    sorted_points = points[sorted_indices]

    # Find unique points
    unique_mask = np.ones(len(sorted_points), dtype=bool)
    unique_mask[1:] = np.any(np.abs(np.diff(sorted_points, axis=0)) > tolerance, axis=1)

    unique_indices = sorted_indices[unique_mask]
    if return_indices:
        return points[unique_indices], unique_indices
    return points[unique_indices]


def triangulate_delaunay(
    points: np.ndarray,
    knn: int = 15,
    eval_threshold: float = 0.1,
    dist_threshold: float = 0.04,
) -> np.ndarray:
    """
    Triangulate point cloud using Delaunay triangulation in local neighborhoods.

    Adapted from lidar_leaf_properties/leafproperties/leaf_geometry.py

    Args:
        points: Nx3 point cloud
        knn: Number of nearest neighbors to consider
        eval_threshold: Eigenvalue threshold for planarity filtering
        dist_threshold: Distance threshold for neighborhood inclusion

    Returns:
        Triangles as Mx3 array of indices into `points` (duplicate points are left
        unreferenced)
    """
    # Triangulate the deduplicated, reordered points, then map the triangles back to
    # the caller's indices: callers index their own `points` with the result.
    points, kept = remove_duplicates(points[:, :3], return_indices=True)

    # Build KNN index
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm="kd_tree").fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Compute local normal vectors and planarity
    normals = np.full([len(points), 3], np.nan)
    eigenvalues = np.full([len(points), 3], np.nan)

    for i, (idx, dist) in enumerate(zip(indices, distances)):
        neighborhood = points[idx]

        # Compute PCA for normal estimation
        centroid = np.mean(neighborhood, axis=0)
        centered = neighborhood - centroid

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]  # Smallest eigenvalue corresponds to normal

            # Compute eigenvalues for planarity check
            cov_matrix = np.cov(centered.T)
            evals, _ = np.linalg.eigh(cov_matrix)
            evals = np.sort(evals)[::-1]  # Sort descending

            normals[i] = normal
            eigenvalues[i] = evals

        except np.linalg.LinAlgError:
            continue

    # Orient normals consistently (pointing up)
    normal_mask = normals[:, 2] < 0
    normals[normal_mask] = -normals[normal_mask]

    # Filter points based on planarity (eigenvalue ratios)
    evals_normalized = eigenvalues / np.sum(eigenvalues, axis=1, keepdims=True)
    planarity_mask = (
        evals_normalized[:, 2] <= eval_threshold
    )  # Small third eigenvalue = planar
    valid_indices = np.where(planarity_mask)[0]

    triangles = []

    for i in valid_indices:
        nbr_indices = indices[i]
        nbr_distances = distances[i]
        close_neighbors = nbr_indices[nbr_distances <= dist_threshold]

        if len(close_neighbors) < 3:
            continue

        try:
            # Project to 2D using PCA
            neighborhood_3d = points[close_neighbors]
            pca = PCA(n_components=2).fit(neighborhood_3d)
            neighborhood_2d = pca.transform(neighborhood_3d)

            # Delaunay triangulation in 2D
            tri = Delaunay(neighborhood_2d)

            # Only keep triangles that include the center point (index 0)
            center_triangles = tri.simplices[np.any(tri.simplices == 0, axis=1)]

            for triangle in center_triangles:
                triangle_indices = close_neighbors[triangle]
                triangles.append(np.sort(triangle_indices))

        except Exception:
            continue

    # Remove duplicate triangles
    if triangles:
        triangles = np.array(triangles)
        triangles = np.unique(triangles, axis=0)
        return kept[triangles].astype(int)
    else:
        return np.array([]).reshape(0, 3).astype(int)


def alpha_shape_reconstruction(
    points: np.ndarray, alpha_multiplier: float = 15.0
) -> o3d.geometry.TriangleMesh:
    """
    Alpha shape reconstruction using Open3D.

    Adapted from surface-models-source-codes/02_processing_surface_model_alpha_shape/

    Args:
        points: Nx3 point cloud
        alpha_multiplier: Multiplier for median nearest neighbor distance

    Returns:
        Open3D triangle mesh
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Compute median nearest neighbor distance
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    nn_distances = []

    for point in pcd.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 2)
        if k >= 2:
            nn_distances.append(
                np.linalg.norm(
                    np.asarray(pcd.points)[idx[1]] - np.asarray(pcd.points)[idx[0]]
                )
            )

    if nn_distances:
        median_distance = np.median(nn_distances)
        alpha = alpha_multiplier * median_distance
    else:
        alpha = 0.1  # fallback

    # Generate alpha shape
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh


def ball_pivoting_reconstruction(
    points: np.ndarray, radius_multiplier: float = 2.0, estimate_normals: bool = True
) -> o3d.geometry.TriangleMesh:
    """
    Ball pivoting algorithm reconstruction using Open3D.

    Args:
        points: Nx3 point cloud
        radius_multiplier: Multiplier for median nearest neighbor distance
        estimate_normals: Whether to estimate normals

    Returns:
        Open3D triangle mesh
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if estimate_normals:
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Estimate ball radius from nearest neighbor distances
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    nn_distances = []

    for point in pcd.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 2)
        if k >= 2:
            nn_distances.append(
                np.linalg.norm(
                    np.asarray(pcd.points)[idx[1]] - np.asarray(pcd.points)[idx[0]]
                )
            )

    if nn_distances:
        median_distance = np.median(nn_distances)
        radius = radius_multiplier * median_distance
    else:
        radius = 0.1  # fallback

    # Ball pivoting reconstruction
    radii = [radius, radius * 2, radius * 4]  # Multiple radii for robustness
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    return mesh


def poisson_reconstruction(
    points: np.ndarray, depth: int = 9, estimate_normals: bool = True
) -> o3d.geometry.TriangleMesh:
    """
    Poisson surface reconstruction using Open3D.

    Args:
        points: Nx3 point cloud
        depth: Octree depth for Poisson reconstruction
        estimate_normals: Whether to estimate normals

    Returns:
        Open3D triangle mesh
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if estimate_normals or not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

    return mesh


def compute_triangle_area(triangle_points: np.ndarray) -> float:
    """Compute area of a triangle using cross product."""
    if triangle_points.shape != (3, 3):
        return 0.0

    v1 = triangle_points[1] - triangle_points[0]
    v2 = triangle_points[2] - triangle_points[0]
    cross = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross)


def compute_mesh_area(points: np.ndarray, triangles: np.ndarray) -> float:
    """
    Compute total surface area from triangulated mesh.

    Args:
        points: Nx3 point cloud
        triangles: Mx3 triangle indices

    Returns:
        Total surface area
    """
    if len(triangles) == 0:
        return 0.0

    total_area = 0.0
    for triangle in triangles:
        triangle_points = points[triangle]
        total_area += compute_triangle_area(triangle_points)

    return total_area


def delaunay_mesh_reconstruction(points: np.ndarray, **kwargs) -> o3d.geometry.TriangleMesh:
    """
    Create Open3D mesh from Delaunay triangulation.
    
    Args:
        points: Nx3 point cloud
        **kwargs: Arguments passed to triangulate_delaunay
        
    Returns:
        Open3D triangle mesh
    """
    triangles = triangulate_delaunay(points, **kwargs)
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
    
    if len(triangles) > 0:
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
    
    return mesh

def compute_leaf_area_delaunay(points: np.ndarray, **kwargs) -> float:
    """
    Compute leaf area using Delaunay triangulation method.

    Adapted from lidar_leaf_properties/leafproperties/leaf_area.py
    """
    triangles = triangulate_delaunay(points, **kwargs)
    return compute_mesh_area(points, triangles)


def compute_normals(points: np.ndarray, knn: int = 15) -> np.ndarray:
    """Compute point normals using PCA on local neighborhoods."""
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm="kd_tree").fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points)

    for i, idx in enumerate(indices):
        neighborhood = points[idx]
        centroid = np.mean(neighborhood, axis=0)
        centered = neighborhood - centroid

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]  # Normal is direction of smallest variance
            normals[i] = normal
        except np.linalg.LinAlgError:
            normals[i] = [0, 0, 1]  # fallback

    # Orient normals consistently (pointing up)
    normal_mask = normals[:, 2] < 0
    normals[normal_mask] = -normals[normal_mask]

    return normals


def bezier_surface_reconstruction(
    points: np.ndarray,
    degree_u: int = 3,
    degree_v: int = 3,
    control_points_u: int = 6,
    control_points_v: int = 6,
) -> o3d.geometry.TriangleMesh:
    """
    Bezier surface fitting using simplified parametric approach.

    Simplified implementation inspired by surface-models-source-codes Bezier leaf model.
    Fits a Bezier surface patch to the point cloud.

    Args:
        points: Nx3 point cloud
        degree_u: Degree in u direction
        degree_v: Degree in v direction
        control_points_u: Number of control points in u
        control_points_v: Number of control points in v

    Returns:
        Open3D triangle mesh of fitted Bezier surface
    """
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    # Create Open3D point cloud for normal estimation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.estimate_normals()

    # Parameterize points in UV space using PCA projection
    centered_points = points[:, :3] - np.mean(points[:, :3], axis=0)
    pca = PCA(n_components=2).fit(centered_points)
    points_2d = pca.transform(centered_points)

    # Normalize to [0,1] parameter space
    u_coords = (points_2d[:, 0] - points_2d[:, 0].min()) / (
        points_2d[:, 0].max() - points_2d[:, 0].min()
    )
    v_coords = (points_2d[:, 1] - points_2d[:, 1].min()) / (
        points_2d[:, 1].max() - points_2d[:, 1].min()
    )

    # Initialize control points grid
    control_mesh = np.zeros((control_points_u, control_points_v, 3))

    # Rough initial guess by fitting control points to point cloud regions
    for i in range(control_points_u):
        for j in range(control_points_v):
            u_center = i / (control_points_u - 1)
            v_center = j / (control_points_v - 1)

            # Find closest points in parameter space
            u_dist = np.abs(u_coords - u_center)
            v_dist = np.abs(v_coords - v_center)
            combined_dist = u_dist + v_dist
            closest_indices = np.argsort(combined_dist)[: min(20, len(points))]

            if len(closest_indices) > 0:
                control_mesh[i, j] = np.mean(points[closest_indices, :3], axis=0)
            else:
                # Fallback - interpolate from existing points
                control_mesh[i, j] = np.mean(points[:, :3], axis=0)

    def bernstein_poly(n: int, i: int, t: np.ndarray) -> np.ndarray:
        """Bernstein polynomial basis function."""
        from math import comb

        return comb(n, i) * (t**i) * ((1 - t) ** (n - i))

    def evaluate_bezier_surface(
        control_points: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Evaluate Bezier surface at parameter coordinates."""
        nu, nv = control_points.shape[:2]
        surface_points = np.zeros((len(u), 3))

        for i in range(nu):
            for j in range(nv):
                basis_u = bernstein_poly(nu - 1, i, u)
                basis_v = bernstein_poly(nv - 1, j, v)
                basis_combined = basis_u * basis_v

                for dim in range(3):
                    surface_points[:, dim] += basis_combined * control_points[i, j, dim]

        return surface_points

    # Create mesh by sampling surface
    u_samples = np.linspace(0, 1, 50)
    v_samples = np.linspace(0, 1, 50)
    u_grid, v_grid = np.meshgrid(u_samples, v_samples)
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    surface_points = evaluate_bezier_surface(control_mesh, u_flat, v_flat)

    # Create triangular mesh
    vertices = surface_points
    triangles = []

    rows, cols = len(v_samples), len(u_samples)
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Two triangles per quad
            v0 = i * cols + j
            v1 = i * cols + (j + 1)
            v2 = (i + 1) * cols + j
            v3 = (i + 1) * cols + (j + 1)

            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    return mesh


def bspline_surface_reconstruction(
    points: np.ndarray,
    degree_u: int = 3,
    degree_v: int = 3,
    num_control_points_u: int = 8,
    num_control_points_v: int = 8,
) -> o3d.geometry.TriangleMesh:
    """
    B-spline surface fitting using simplified approach.

    Simplified implementation inspired by surface-models-source-codes B-spline methods.
    Note: This is a basic implementation without full NURBS optimization.

    Args:
        points: Nx3 point cloud
        degree_u: B-spline degree in u direction
        degree_v: B-spline degree in v direction
        num_control_points_u: Number of control points in u
        num_control_points_v: Number of control points in v

    Returns:
        Open3D triangle mesh
    """
    try:
        from scipy.interpolate import BSpline
    except ImportError:
        warnings.warn(
            "scipy.interpolate.BSpline not available, falling back to Bezier surface"
        )
        return bezier_surface_reconstruction(
            points, degree_u, degree_v, num_control_points_u, num_control_points_v
        )

    # Create Open3D point cloud for normal estimation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Parameterize points in UV space using PCA
    centered_points = points[:, :3] - np.mean(points[:, :3], axis=0)
    pca = PCA(n_components=2).fit(centered_points)
    points_2d = pca.transform(centered_points)

    # Normalize to parameter space
    u_coords = (points_2d[:, 0] - points_2d[:, 0].min()) / (
        points_2d[:, 0].max() - points_2d[:, 0].min()
    )
    v_coords = (points_2d[:, 1] - points_2d[:, 1].min()) / (
        points_2d[:, 1].max() - points_2d[:, 1].min()
    )

    # Create knot vectors (clamped)
    knots_u = np.concatenate(
        [
            np.zeros(degree_u),
            np.linspace(0, 1, num_control_points_u - degree_u + 1),
            np.ones(degree_u),
        ]
    )
    knots_v = np.concatenate(
        [
            np.zeros(degree_v),
            np.linspace(0, 1, num_control_points_v - degree_v + 1),
            np.ones(degree_v),
        ]
    )

    # Initialize control points using least squares fitting
    from scipy.interpolate import griddata

    # Create regular grid for control points
    u_control = np.linspace(0, 1, num_control_points_u)
    v_control = np.linspace(0, 1, num_control_points_v)
    u_ctrl_grid, v_ctrl_grid = np.meshgrid(u_control, v_control)

    # Interpolate point cloud data to control grid
    control_points = np.zeros((num_control_points_u, num_control_points_v, 3))

    for dim in range(3):
        try:
            # Interpolate each coordinate dimension
            coord_values = griddata(
                np.column_stack([u_coords, v_coords]),
                points[:, dim],
                (u_ctrl_grid, v_ctrl_grid),
                method="linear",
                fill_value=np.mean(points[:, dim]),
            )
            control_points[:, :, dim] = coord_values.T
        except Exception:
            # Fallback - use mean
            control_points[:, :, dim] = np.mean(points[:, dim])

    # Sample surface for mesh generation
    u_samples = np.linspace(0, 1, 40)
    v_samples = np.linspace(0, 1, 40)

    # Simplified B-spline evaluation using basis functions
    def b_spline_basis(t: float, i: int, k: int, knots: np.ndarray) -> float:
        """Evaluate B-spline basis function using Cox-de Boor recursion."""
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

        # Handle division by zero
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = (
            0.0
            if denom1 == 0
            else ((t - knots[i]) / denom1) * b_spline_basis(t, i, k - 1, knots)
        )
        term2 = (
            0.0
            if denom2 == 0
            else ((knots[i + k + 1] - t) / denom2)
            * b_spline_basis(t, i + 1, k - 1, knots)
        )

        return term1 + term2

    def evaluate_bspline_surface(u: float, v: float) -> np.ndarray:
        """Evaluate B-spline surface at parameter (u,v)."""
        point = np.zeros(3)

        for i in range(num_control_points_u):
            for j in range(num_control_points_v):
                basis_u = b_spline_basis(u, i, degree_u, knots_u)
                basis_v = b_spline_basis(v, j, degree_v, knots_v)
                weight = basis_u * basis_v

                point += weight * control_points[i, j]

        return point

    # Generate mesh vertices
    vertices = []
    for v in v_samples:
        for u in u_samples:
            try:
                vertex = evaluate_bspline_surface(u, v)
                vertices.append(vertex)
            except:
                # Fallback for evaluation errors
                vertices.append(np.mean(points[:, :3], axis=0))

    vertices = np.array(vertices)

    # Create triangles
    triangles = []
    rows, cols = len(v_samples), len(u_samples)
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = i * cols + (j + 1)
            v2 = (i + 1) * cols + j
            v3 = (i + 1) * cols + (j + 1)

            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    return mesh


def compute_leaf_angles(points: np.ndarray, knn: int = 15) -> np.ndarray:
    """
    Compute leaf angle distribution from point cloud normals.

    Adapted from lidar_leaf_properties/leafproperties/leaf_angle.py

    Args:
        points: Nx3 point cloud
        knn: Number of nearest neighbors for normal estimation

    Returns:
        Array of angles in degrees between normals and horizontal plane
    """
    points = remove_duplicates(points[:, :3])
    normals = compute_normals(points, knn)

    # Compute angles with horizontal plane (z=1 vector)
    horizontal = np.array([0, 0, 1])
    angles = np.arccos(np.clip(np.abs(np.dot(normals, horizontal)), 0, 1))
    angles_degrees = np.degrees(angles)

    return angles_degrees
