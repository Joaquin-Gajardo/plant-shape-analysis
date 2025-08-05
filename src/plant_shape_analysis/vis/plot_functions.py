import colorsys
from typing import Optional

import numpy as np
import open3d as o3d
import pymeshlab


def visualize_pymeshlab_mesh(
    ms: pymeshlab.MeshSet, window_name: str = "Mesh Visualization"
):
    """
    Visualize a PyMeshLab mesh using Open3D

    Args:
        ms: PyMeshLab MeshSet containing the mesh
        window_name: Name for the visualization window
    """
    # Get the current mesh from MeshSet
    mesh = ms.current_mesh()

    # Extract vertices and faces
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute normals for better visualization
    o3d_mesh.compute_vertex_normals()
    # Convert normals to colors (normal mapping visualization)
    normals = np.asarray(o3d_mesh.vertex_normals)

    # Map normals from [-1, 1] to [0, 1] for RGB values
    # This creates the typical blue-green normal map appearance
    colors = (normals + 1.0) / 2.0

    # Apply the colors to the mesh
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries(
        [o3d_mesh], window_name=window_name, mesh_show_back_face=True
    )


def visualize_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    window_name: str = "Plant Visualization",
):
    """
    Visualize plant/leaf points and keypoints using Open3D

    Args:
        points: numpy array of shape (n_points, 3)
        colors: optional numpy array of shape (n_points, 3) for custom colors
        window_name: name for the Open3D visualization window
    """

    # Create point cloud
    plant_pcd = o3d.geometry.PointCloud()
    plant_pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        plant_pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries = [plant_pcd]

    # Visualize
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def create_correspondence_lines(
    correspondences: list,
) -> Optional[o3d.geometry.LineSet]:
    """
    Create line connections between consecutive corresponding points

    Args:
        correspondences: List of point correspondences in time series

    Returns:
        Open3D LineSet or None
    """
    # Filter out None positions and keep track of indices
    valid_points = []
    valid_indices = []

    for i, pos in enumerate(correspondences):
        if pos is not None:
            valid_points.append(pos)
            valid_indices.append(i)

    if len(valid_points) < 2:
        return None

    # Create line set
    line_set = o3d.geometry.LineSet()

    # Points are the valid tip positions
    line_set.points = o3d.utility.Vector3dVector(valid_points)

    # Lines connect consecutive tips
    lines = []
    colors = []

    for i in range(len(valid_points) - 1):
        lines.append([i, i + 1])
        # Color based on timepoint progression
        hue = valid_indices[i] / max(1, len(valid_points) - 1)
        color = colorsys.hsv_to_rgb(hue * 0.8, 0.7, 0.8)
        colors.append(color)

    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def visualize_leaf_sequence(
    leaf_timeseries: dict,
    spacing: float = 100.0,
    show_leaf_tips: bool = True,
    show_connections: bool = True,
    window_name: str = "Leaf Growth Sequence",
):
    """
    Visualize a single leaf's growth over time using Open3D

    Args:
        leaf_timeseries: Dictionary with 'sequence_name', 'leaf_id', and 'timepoints' keys
                        from LeafSequencesDataset
        spacing: Distance between timepoints in the visualization
        show_leaf_tips: Whether to highlight leaf tips as red spheres
        show_connections: Whether to show connections between corresponding points
        connection_samples: Number of connection lines to draw between timepoints
        window_name: Name for the visualization window
    """
    timepoints = leaf_timeseries["timepoints"]
    if len(timepoints) < 2:
        print("At least two timepoints are needed for visualization")
        return

    # Generate colors for each timepoint using HSV colorspace
    colors = [
        np.array(colorsys.hsv_to_rgb(i / len(timepoints), 0.8, 0.9))
        for i in range(len(timepoints))
    ]

    geometries = []
    all_points = []
    leaf_tip_positions = []

    # Process each timepoint
    for i, timepoint in enumerate(timepoints):
        points = timepoint["points"]
        day = timepoint["day"]
        leaf_tip = timepoint["leaf_tip"]

        # Create point cloud for this timepoint
        pcd = o3d.geometry.PointCloud()

        # Translate each timepoint along Y axis for spacing
        translated_points = points.copy()
        translated_points[:, 1] += i * spacing

        pcd.points = o3d.utility.Vector3dVector(translated_points)

        # Color the point cloud
        point_colors = np.tile(colors[i], (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

        geometries.append(pcd)
        all_points.append(translated_points)

        # Add leaf tip visualization if available
        if leaf_tip is None:
            leaf_tip_positions.append(leaf_tip)
        else:
            tip_position = leaf_tip.copy()
            tip_position[1] += i * spacing
            leaf_tip_positions.append(tip_position)

            # Add leaf tip sphere
            if show_leaf_tips:
                tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                tip_sphere.translate(tip_position)
                tip_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
                geometries.append(tip_sphere)

    if show_connections:
        # Visualize connections between leaf tips
        line_set = create_correspondence_lines(leaf_tip_positions)
        geometries.append(line_set) if line_set is not None else None

    # Visualize all geometries
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{window_name} - {leaf_timeseries["sequence_name"]} Leaf {leaf_timeseries["leaf_id"]}",
    )
