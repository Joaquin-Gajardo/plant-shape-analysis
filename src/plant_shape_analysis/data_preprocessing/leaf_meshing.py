from argparse import ArgumentParser
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import pymeshlab


def calculate_ball_pivoting_radius(
    pcd: o3d.geometry.PointCloud, scaling_factor: int = 5
) -> list:
    """
    Ball-Pivoting-Radius for Mesh calculation
    Inspired by: https://github.com/Engineering-Geodesy-Bonn/CropMesh
    """
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist / 2
    radii = [(1 / scaling_factor) * radius, radius, scaling_factor * radius]
    return radii


def mesh_ball_pivoting_o3d(leaf_points: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Function to create a mesh from point cloud using Ball Pivoting algorithm"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(leaf_points)

    # Estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    pcd.orient_normals_consistent_tangent_plane(
        k=30
    )  # This significantly improved mesh quality!

    # Create mesh using Ball Pivoting algorithm
    radii = calculate_ball_pivoting_radius(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    mesh.vertex_colors = mesh.vertex_normals

    # # Clean up (just in case but doesn't seem to help much) # NOTE: this messed up with point cloud original indices
    # mesh.remove_unreferenced_vertices()  # Clean up the mesh
    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_degenerate_triangles()  # Remove degenerate triangles
    # mesh.remove_non_manifold_edges()

    return pcd, mesh


def mesh_poisson_o3d(
    leaf_points: np.ndarray,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """Function to create a mesh from point cloud using Poisson Surface Reconstruction"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(leaf_points)

    # Estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    pcd.orient_normals_consistent_tangent_plane(k=30)  # This is very important!!

    # Create mesh using Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False
    )
    mesh.vertex_colors = mesh.vertex_normals

    # # Clean up (just in case but doesn't seem to help much) # NOTE: this messed up with point cloud original indices
    # mesh.remove_unreferenced_vertices()  # Clean up the mesh
    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_degenerate_triangles()  # Remove degenerate triangles
    # mesh.remove_non_manifold_edges()

    return pcd, mesh


def mesh_ball_pivoting_pymeshlab(
    leaf_points: np.ndarray,
) -> Tuple[Optional[o3d.geometry.PointCloud], o3d.geometry.TriangleMesh]:
    """Function to create a mesh from point cloud using Ball Pivoting algorithm with PyMeshLab"""
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=leaf_points)
    ms.add_mesh(mesh, "leaf")
    ms.generate_surface_reconstruction_ball_pivoting()
    return None, mesh_pymeshlab_to_o3d(ms.current_mesh())


def mesh_poisson_pymeshlab(
    leaf_points: np.ndarray,
) -> Tuple[Optional[o3d.geometry.PointCloud], o3d.geometry.TriangleMesh]:
    """Function to create a mesh from point cloud using Poisson Surface Reconstruction with PyMeshLab"""
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=leaf_points)
    ms.add_mesh(mesh, "leaf")
    ms.compute_normal_for_point_clouds(k=20, smoothiter=10)
    ms.generate_surface_reconstruction_screened_poisson(depth=8)
    return None, mesh_pymeshlab_to_o3d(ms.current_mesh())


def mesh_pymeshlab_to_o3d(mesh: pymeshlab.Mesh) -> o3d.geometry.TriangleMesh:
    """Convert a PyMeshLab mesh to Open3D TriangleMesh"""
    vertices = np.asarray(mesh.vertex_matrix())
    triangles = np.asarray(mesh.face_matrix())
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def mesh_leaf(
    leaf_points: np.ndarray,
    method: str = "ball_pivoting",
    backend: str = "open3d",
    visualize: bool = True,
) -> None:
    """Create a mesh from leaf point cloud using specified method and backend.
    Valid methods: 'ball_pivoting', 'poisson'.
    Valid backends: 'open3d', 'pymeshlab'."""

    if method == "ball_pivoting":
        if backend == "open3d":
            pcd, mesh = mesh_ball_pivoting_o3d(leaf_points)
        elif backend == "pymeshlab":
            pcd, mesh = mesh_ball_pivoting_pymeshlab(leaf_points)
        else:
            raise ValueError("Invalid backend. Choose 'open3d' or 'pymeshlab'.")
    elif method == "poisson":
        if backend == "open3d":
            pcd, mesh = mesh_poisson_o3d(leaf_points)
        elif backend == "pymeshlab":
            pcd, mesh = mesh_poisson_pymeshlab(leaf_points)
        else:
            raise ValueError("Invalid backend. Choose 'open3d' or 'pymeshlab'.")
    else:
        raise ValueError("Invalid method. Choose 'ball_pivoting' or 'poisson'.")

    # Visualize the mesh
    if visualize:
        if backend == "open3d":
            o3d.visualization.draw_geometries(
                [pcd],
                point_show_normal=True,
                window_name="Leaf Point Cloud and normals",
            )
        o3d.visualization.draw_geometries(
            [mesh], mesh_show_back_face=True, window_name="Leaf Mesh"
        )

    return pcd, mesh


def parse_args():
    parser = ArgumentParser(
        description="Mesh leaf point cloud using different methods."
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        choices=["ball_pivoting", "poisson"],
        default="ball_pivoting",
        help="Method to use for meshing: 'ball_pivoting' or 'poisson'.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["open3d", "pymeshlab"],
        default="open3d",
        help="Backend to use for meshing: 'open3d' or 'pymeshlab'.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    file_path = "data/TrackPlant3D/processed/single_leafs/maize/1_maize_control_plant1_D00_leaf2.ply"
    leaf_points = o3d.io.read_point_cloud(file_path)
    leaf_points = np.asarray(leaf_points.points)

    mesh_leaf(leaf_points, method=args.method, backend=args.backend)
