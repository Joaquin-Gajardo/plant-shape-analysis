import colorsys
from typing import Optional

import matplotlib.pyplot as plt
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
    spacing: float = 30.0,
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

        # Suppose you have: transformations = [...]  # list of dicts, each with "basis"

        # Draw principal axes from transformation["basis"] if available
        transformations = leaf_timeseries["transformations"][i]
        if transformations is not None:
            basis = transformations.get(
                "basis", None
            )  # NOTE: can change to "rotation_matrix" too
            if basis is not None:
                arrow_origin = points.mean(axis=0)
                arrow_origin[1] += i * spacing
                axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                for j in range(3):
                    axis_vec = basis[j]  # or basis[:, j] depending on your convention
                    length = 10  # or scale as you wish
                    arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.3,
                        cone_radius=0.6,
                        cylinder_height=length * 0.8,
                        cone_height=length * 0.2,
                    )
                    # Align arrow with axis_vec
                    z_axis = np.array([0, 0, 1])
                    axis_vec_norm = axis_vec / np.linalg.norm(axis_vec)
                    v = np.cross(z_axis, axis_vec_norm)
                    c = np.dot(z_axis, axis_vec_norm)
                    if np.linalg.norm(v) < 1e-8:
                        R = np.eye(3)
                    else:
                        vx = np.array(
                            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
                        )
                        R = (
                            np.eye(3)
                            + vx
                            + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
                        )
                    arrow.rotate(R, center=np.zeros(3))
                    arrow.translate(arrow_origin)
                    arrow.paint_uniform_color(axis_colors[j])
                    geometries.append(arrow)

        # # --- Add principal axes (eigenvectors) visualization as arrows ---
        # # Compute PCA (eigenvectors of covariance)
        # pc_centered = points - points.mean(axis=0)
        # cov = np.cov(pc_centered, rowvar=False)
        # eigvals, eigvecs = np.linalg.eigh(cov)
        # idx = np.argsort(eigvals)[::-1]
        # eigvecs = eigvecs[:, idx]
        # eigvals = eigvals[idx]
        # # Arrow origin: mean of translated points
        # arrow_origin = points.mean(axis=0)
        # arrow_origin[1] += i * spacing
        # # Draw 3 principal axes as arrows (quivers)
        # for j in range(3):
        #     axis_vec = eigvecs[:, j]
        #     # Scale for visualization (length proportional to sqrt eigenvalue)
        #     length = np.sqrt(np.abs(eigvals[j])) * 5
        #     arrow = o3d.geometry.TriangleMesh.create_arrow(
        #         cylinder_radius=0.3,
        #         cone_radius=0.6,
        #         cylinder_height=length * 0.8,
        #         cone_height=length * 0.2,
        #     )
        #     # Align arrow with axis_vec
        #     # Default arrow points in +Z, so rotate to axis_vec
        #     z_axis = np.array([0, 0, 1])
        #     axis_vec_norm = axis_vec / np.linalg.norm(axis_vec)
        #     v = np.cross(z_axis, axis_vec_norm)
        #     c = np.dot(z_axis, axis_vec_norm)
        #     if np.linalg.norm(v) < 1e-8:
        #         R = np.eye(3)
        #     else:
        #         vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        #         R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
        #     arrow.rotate(R, center=np.zeros(3))
        #     arrow.translate(arrow_origin)
        #     # Color: RGB for axes
        #     axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        #     arrow.paint_uniform_color(axis_colors[j])
        #     geometries.append(arrow)

    if show_connections:
        # Visualize connections between leaf tips
        line_set = create_correspondence_lines(leaf_tip_positions)
        geometries.append(line_set) if line_set is not None else None

    # Visualize all geometries
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{window_name} - {leaf_timeseries["sequence_name"]} Leaf {leaf_timeseries["leaf_id"]}",
    )


def compare_original_vs_aligned_compact(
    leaf_dataset,
    leaf_timeseries: dict,
    reference_idx: int = 0,
    temporal_spacing: float = 30.0,
    horizontal_spacing: float = 30.0,
    show_leaf_tips: bool = True,
    show_connections: bool = True,
):
    """
    Show original and aligned sequences side by side with compact spacing.

    Args:
        leaf_dataset: LeafSequencesDataset instance
        leaf_timeseries: Dictionary with sequence data
        reference_idx: Index of reference timepoint for alignment
        spacing: Compact spacing between timepoints
        show_leaf_tips: Whether to show leaf tip markers
        show_connections: Whether to show tip connections
    """
    # Get aligned sequence
    aligned_timepoints, transformations = leaf_dataset.align_leaf_sequence(
        leaf_timeseries, reference_idx
    )

    if len(aligned_timepoints) < 2:
        print("At least two timepoints are needed for visualization")
        return

    timepoints = leaf_timeseries["timepoints"]
    colors = [
        np.array(colorsys.hsv_to_rgb(i / len(timepoints), 0.8, 0.9))
        for i in range(len(timepoints))
    ]

    geometries = []

    # Original sequence (left side)
    original_leaf_tips = []
    for i, timepoint in enumerate(timepoints):
        points = timepoint["points"]
        leaf_tip = timepoint["leaf_tip"]

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        translated_points = points.copy()
        translated_points[:, 1] += i * temporal_spacing
        translated_points[:, 0] -= horizontal_spacing  # Move to left side

        pcd.points = o3d.utility.Vector3dVector(translated_points)
        point_colors = np.tile(colors[i], (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        geometries.append(pcd)

        # Leaf tip handling
        if leaf_tip is not None:
            tip_position = leaf_tip.copy()
            tip_position[1] += i * temporal_spacing
            tip_position[0] -= horizontal_spacing
            original_leaf_tips.append(tip_position)

            if show_leaf_tips:
                tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
                tip_sphere.translate(tip_position)
                tip_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(tip_sphere)
        else:
            original_leaf_tips.append(None)

    # Aligned sequence (right side)
    aligned_leaf_tips = []
    for i, timepoint in enumerate(aligned_timepoints):
        points = timepoint["points"]
        leaf_tip = timepoint["leaf_tip"]

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        translated_points = points.copy()
        translated_points[:, 1] += i * temporal_spacing
        translated_points[:, 0] += horizontal_spacing  # Move to right side

        pcd.points = o3d.utility.Vector3dVector(translated_points)
        point_colors = np.tile(colors[i], (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        geometries.append(pcd)

        # Leaf tip handling
        if leaf_tip is not None:
            tip_position = leaf_tip.copy()
            tip_position[1] += i * temporal_spacing
            tip_position[0] += horizontal_spacing
            aligned_leaf_tips.append(tip_position)

            if show_leaf_tips:
                tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.8)
                tip_sphere.translate(tip_position)
                tip_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(tip_sphere)
        else:
            aligned_leaf_tips.append(None)

    # Add connection lines
    if show_connections:
        # Original connections (red lines)
        orig_lines = create_correspondence_lines(original_leaf_tips)
        if orig_lines is not None:
            orig_lines.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(orig_lines)

        # Aligned connections (green lines)
        aligned_lines = create_correspondence_lines(aligned_leaf_tips)
        if aligned_lines is not None:
            aligned_lines.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(aligned_lines)

    # Visualize
    window_name = f"Original vs Aligned - {leaf_timeseries['sequence_name']} Leaf {leaf_timeseries['leaf_id']}"
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def plot_sequence_projections(
    leaf_dataset,
    leaf_timeseries: dict,
    reference_idx: int = 0,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
):
    """
    Create matplotlib figure showing projected views of original and aligned sequences.

    Args:
        leaf_dataset: LeafSequencesDataset instance
        leaf_timeseries: Dictionary with sequence data
        reference_idx: Index of reference timepoint for alignment
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    # Get aligned sequence
    aligned_timepoints, transformations = leaf_dataset.align_leaf_sequence(
        leaf_timeseries, reference_idx
    )

    timepoints = leaf_timeseries["timepoints"]
    n_timepoints = len(timepoints)

    if n_timepoints < 2:
        print("Need at least 2 timepoints for projection visualization")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(
        f'{leaf_timeseries["sequence_name"]} Leaf {leaf_timeseries["leaf_id"]} - Growth Analysis',
        fontsize=16,
    )

    # Define colors for each timepoint
    colors = plt.cm.viridis(np.linspace(0, 1, n_timepoints))

    # Projection views: XY, XZ, YZ
    projections = [
        (0, 1, "XY", "X", "Y"),
        (0, 2, "XZ", "X", "Z"),
        (1, 2, "YZ", "Y", "Z"),
    ]

    for proj_idx, (dim1, dim2, name, label1, label2) in enumerate(projections):
        # Original sequence (left column)
        ax_orig = axes[proj_idx, 0]
        ax_orig.set_title(f"Original - {name} Projection")
        ax_orig.set_xlabel(label1)
        ax_orig.set_ylabel(label2)

        for i, timepoint in enumerate(timepoints):
            points = timepoint["points"]
            leaf_tip = timepoint["leaf_tip"]

            # Plot point cloud projection
            ax_orig.scatter(
                points[:, dim1],
                points[:, dim2],
                c=[colors[i]],
                alpha=0.6,
                s=2,
                label=f'Day {timepoint["day"]}',
            )

            # Plot leaf tip if available
            if leaf_tip is not None:
                ax_orig.scatter(
                    leaf_tip[dim1],
                    leaf_tip[dim2],
                    c=[colors[i]],
                    s=50,
                    marker="*",
                    edgecolors="red",
                    linewidths=1,
                )

        # Aligned sequence (right column)
        ax_aligned = axes[proj_idx, 1]
        ax_aligned.set_title(f"PCA Aligned - {name} Projection")
        ax_aligned.set_xlabel(label1)
        ax_aligned.set_ylabel(label2)

        for i, timepoint in enumerate(aligned_timepoints):
            points = timepoint["points"]
            leaf_tip = timepoint["leaf_tip"]

            # Plot aligned point cloud projection
            ax_aligned.scatter(
                points[:, dim1],
                points[:, dim2],
                c=[colors[i]],
                alpha=0.6,
                s=2,
                label=f'Day {timepoint["day"]}',
            )

            # Plot aligned leaf tip if available (only add label once)
            leaf_tip_label = ""
            if i == len(aligned_timepoints) - 1:
                leaf_tip_label = "Leaf tip"
            if leaf_tip is not None:
                ax_aligned.scatter(
                    leaf_tip[dim1],
                    leaf_tip[dim2],
                    c=[colors[i]],
                    s=50,
                    marker="*",
                    edgecolors="red",
                    linewidths=1,
                    label=leaf_tip_label,
                )

        # Set equal aspect ratio
        ax_orig.set_aspect("equal", adjustable="box")
        ax_aligned.set_aspect("equal", adjustable="box")

    # Add legend to the last subplot
    axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def plot_pairwise_alignment_with_quivers(timepoints, transformations, idx1, idx2):
    """
    Plots two timepoints' point clouds and their principal axes as quivers.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r", "g", "b"]
    for idx, color, label in zip([idx1, idx2], ["blue", "orange"], ["ref", "other"]):
        points = timepoints[idx]["points"]
        R = transformations[idx]["rotation_matrix"]
        center = transformations[idx]["reference_center"]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            alpha=0.3,
            label=f"Timepoint {idx} ({label})",
        )
        # Plot principal axes as quivers
        for i in range(3):
            ax.quiver(
                center[0],
                center[1],
                center[2],
                R[i, 0],
                R[i, 1],
                R[i, 2],
                length=10,
                color=colors[i],
                linewidth=2,
                arrow_length_ratio=0.2,
            )

    ax.legend()
    ax.set_title("Pairwise Alignment with Principal Axes (Quivers)")
    plt.show()
