import colorsys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pymeshlab
import torch


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

    has_normals = points.shape[-1] == 6
    if has_normals:
        xyz = points[:, :3]
        normals = points[:, 3:]
    else:
        xyz = points

    plant_pcd = o3d.geometry.PointCloud()
    plant_pcd.points = o3d.utility.Vector3dVector(xyz)
    if has_normals:
        plant_pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        plant_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([plant_pcd], window_name=window_name)


def visualize_mesh_open3d(filename):
    """Visualize a PLY mesh file"""

    try:
        mesh = o3d.io.read_triangle_mesh(filename)

        if len(mesh.vertices) == 0:
            print(f"Mesh {filename} is empty!")
            return

        print(f"Loaded mesh {filename}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.triangles)}")

        # Compute vertex normals
        mesh.compute_vertex_normals()

        # Print mesh statistics
        vertices = np.asarray(mesh.vertices)
        print(f"  Vertex range: {vertices.min(axis=0)} to {vertices.max(axis=0)}")
        print(f"  Mesh is watertight: {mesh.is_watertight()}")
        print(f"  Mesh is manifold: {mesh.is_vertex_manifold()}")

        # Visualize
        print("Launching Open3D visualizer...")
        o3d.visualization.draw_geometries(
            [mesh], window_name=f"Mesh: {filename}", width=800, height=600
        )

    except Exception as e:
        print(f"Error loading {filename}: {e}")


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
    dense_points: bool = False,
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
    points_key = "dense_points" if dense_points else "points"
    for i, timepoint in enumerate(timepoints):
        points = timepoint[points_key]
        day = timepoint["day"]
        leaf_tip = timepoint["leaf_tip"]

        if points is None:
            raise ValueError(
                f"Leaf sequence {leaf_timeseries['sequence_name']}, timepoint {day} does not have '{points_key}' data."
            )

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
        window_name=f"{window_name} - {leaf_timeseries['sequence_name']} Leaf {leaf_timeseries['leaf_id']}",
    )


def visualize_plant_sequence(
    plant_sequence_data: dict,
    spacing: float = 50.0,
    show_leaf_tips: bool = True,
    show_connections: bool = False,
    dense_points: bool = False,
    color_by_organ: bool = True,
    window_name: str = "Plant Growth Sequence",
    offscreen: bool = False,
):
    """
    Visualize a plant's growth over time using Open3D

    Args:
        plant_sequence_data: Dictionary with 'data', 'is_aligned', and 'transformations' keys
                            from PlantSequencesDataset when using alignment,
                            or just the sequence data list when not using alignment
        spacing: Distance between timepoints in the visualization
        show_leaf_tips: Whether to highlight leaf tips as red spheres
        show_connections: Whether to show connections between leaf tips across time
        dense_points: Whether to use dense point clouds if available
        color_by_organ: If True, color by organ labels. If False, color by timepoint
        window_name: Name for the visualization window
        offscreen: If True, run visualization in offscreen mode (no GUI and save image)
    """
    # Handle both aligned and non-aligned data formats
    if isinstance(plant_sequence_data, dict) and "timepoints" in plant_sequence_data:
        # New structure: {"sequence_name": str, "timepoints": list, "is_aligned": bool, "transformations": list}
        timepoints = plant_sequence_data["timepoints"]
        transformations = plant_sequence_data.get("transformations", [])
        is_aligned = plant_sequence_data.get("is_aligned", False)
    elif isinstance(plant_sequence_data, dict) and "data" in plant_sequence_data:
        # Old structure (backward compatibility)
        timepoints = plant_sequence_data["data"]
        transformations = plant_sequence_data.get("transformations", [])
        is_aligned = plant_sequence_data.get("is_aligned", False)
    else:
        timepoints = plant_sequence_data
        transformations = []
        is_aligned = False

    if len(timepoints) < 1:
        print("At least one timepoint is needed for visualization")
        return

    # Generate colors for each timepoint using HSV colorspace
    timepoint_colors = [
        np.array(colorsys.hsv_to_rgb(i / max(1, len(timepoints) - 1), 0.8, 0.9))
        for i in range(len(timepoints))
    ]

    # Define colors for different organ types
    organ_colors = {
        0: np.array([0.6, 0.4, 0.2]),  # Stem - brown
        1: np.array([0.2, 0.8, 0.2]),  # Leaf 1 - green
        2: np.array([0.3, 0.9, 0.3]),  # Leaf 2 - lighter green
        3: np.array([0.4, 0.85, 0.4]),  # Leaf 3
        4: np.array([0.5, 0.9, 0.5]),  # Leaf 4
        5: np.array([0.25, 0.95, 0.25]),  # Leaf 5
    }

    geometries = []
    all_leaf_tips = []  # Track all leaf tips across timepoints for connections

    # Process each timepoint
    points_key = "dense_points" if dense_points else "points"
    labels_key = "dense_labels" if dense_points else "labels"

    for i, timepoint in enumerate(timepoints):
        points = timepoint.get(points_key)
        labels = timepoint.get(labels_key)
        day = timepoint["day"]
        leaf_tip_idxs = timepoint.get("leaf_tip_idxs", np.array([]))

        if points is None:
            print(f"Timepoint day {day} does not have '{points_key}' data.")
            continue

        # Create point cloud for this timepoint
        pcd = o3d.geometry.PointCloud()

        # Translate each timepoint along Y axis for spacing
        translated_points = points.copy()
        translated_points[:, 1] += i * spacing

        pcd.points = o3d.utility.Vector3dVector(translated_points)

        # Color the point cloud
        if color_by_organ and labels is not None:
            # Color by organ label
            point_colors = np.zeros((len(points), 3))
            for organ_id in np.unique(labels):
                mask = labels == organ_id
                # Get color for this organ, or use a default if not in our palette
                if organ_id < len(organ_colors):
                    color = organ_colors.get(organ_id, np.array([0.5, 0.5, 0.5]))
                else:
                    # Generate color for higher leaf IDs
                    hue = ((organ_id - 1) * 0.15) % 1.0
                    color = np.array(colorsys.hsv_to_rgb(hue, 0.7, 0.9))
                point_colors[mask] = color
        else:
            # Color by timepoint
            point_colors = np.tile(timepoint_colors[i], (len(points), 1))

        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        geometries.append(pcd)

        # Add leaf tip visualization if available
        timepoint_tips = []
        if leaf_tip_idxs.size > 0 and show_leaf_tips:
            tip_coords = translated_points[leaf_tip_idxs]
            for tip_coord in tip_coords:
                tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
                tip_sphere.translate(tip_coord)
                tip_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
                geometries.append(tip_sphere)
                timepoint_tips.append(tip_coord)

        all_leaf_tips.append(timepoint_tips)

        # Draw principal axes from transformation basis if available
        if transformations and i < len(transformations):
            trans = transformations[i]
            basis = trans.get("basis", None)
            if basis is not None:
                arrow_origin = points.mean(axis=0)
                arrow_origin[1] += i * spacing
                axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                for j in range(3):
                    axis_vec = basis[j]
                    length = 15  # Arrow length
                    arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cylinder_radius=0.5,
                        cone_radius=1.0,
                        cylinder_height=length * 0.8,
                        cone_height=length * 0.2,
                    )
                    # Align arrow with axis_vec
                    z_axis = np.array([0, 0, 1])
                    axis_vec_norm = axis_vec / np.linalg.norm(axis_vec)
                    v = np.cross(z_axis, axis_vec_norm)
                    c = np.dot(z_axis, axis_vec_norm)
                    if np.linalg.norm(v) < 1e-8:
                        R = np.eye(3) if c > 0 else -np.eye(3)
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

    # Add connections between corresponding leaf tips if requested
    if show_connections and len(all_leaf_tips) > 1:
        # Try to match leaf tips across timepoints (simple nearest neighbor for now)
        for tip_idx in range(max(len(tips) for tips in all_leaf_tips)):
            tip_trajectory = []
            for timepoint_tips in all_leaf_tips:
                if tip_idx < len(timepoint_tips):
                    tip_trajectory.append(timepoint_tips[tip_idx])

            if len(tip_trajectory) > 1:
                line_set = create_correspondence_lines(tip_trajectory)
                if line_set is not None:
                    geometries.append(line_set)

    # Get sequence name if available
    if isinstance(plant_sequence_data, dict):
        sequence_info = plant_sequence_data.get("sequence_name", "Unknown")
    else:
        sequence_info = "Unknown"

    alignment_status = " (Aligned)" if is_aligned else ""

    # Visualize all geometries
    if offscreen:
        visualize_multiple_plants(
            geometries,
            output_path=f"{window_name} - {sequence_info}{alignment_status}.png",
        )
    else:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"{window_name} - {sequence_info}{alignment_status}",
        )


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


def visualize_sdf_slices(model, z_levels=[-0.05, 0.0, 0.02, 0.05]):
    """Create 2D slices of the SDF field for visualization"""

    print("\n=== Creating SDF visualization ===")

    # Create 2D slices at different Z levels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, z_val in enumerate(z_levels):
        # Create a 2D grid at this Z level
        xy_grid = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(xy_grid, xy_grid)
        Z = np.full_like(X, z_val)

        coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        coords_tensor = torch.from_numpy(coords).float()

        with torch.no_grad():
            output = model(coords_tensor)
            sdf_values = output["model_out"].squeeze().numpy()

        sdf_grid = sdf_values.reshape(X.shape)

        # Plot the SDF field
        im = axes[i].contourf(X, Y, sdf_grid, levels=20, cmap="RdBu_r")
        axes[i].contour(X, Y, sdf_grid, levels=[0], colors="black", linewidths=2)
        axes[i].set_title(f"SDF at Z={z_val:.1f}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].set_aspect("equal")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("sdf_slices.png", dpi=150, bbox_inches="tight")
    print("Saved SDF slices to 'sdf_slices.png'")


def create_sdf_cross_section(sdf_decoder, device, slice_position=0.0, resolution=256):
    """Create cross-section visualizations of the SDF along the three orthogonal planes"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Collect all SDF values to determine global min/max for consistent colorbar
    all_sdf_values = []

    # XY plane (constant Z)
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = torch.full_like(X, slice_position)
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).to(device)

    with torch.no_grad():
        sdf_values_xy = sdf_decoder(coords).cpu().reshape(resolution, resolution)
    all_sdf_values.append(sdf_values_xy)

    # XZ plane (constant Y)
    x = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    X, Z = torch.meshgrid(x, z, indexing="ij")
    Y = torch.full_like(X, slice_position)
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).to(device)

    with torch.no_grad():
        sdf_values_xz = sdf_decoder(coords).cpu().reshape(resolution, resolution)
    all_sdf_values.append(sdf_values_xz)

    # YZ plane (constant X)
    y = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    Y, Z = torch.meshgrid(y, z, indexing="ij")
    X = torch.full_like(Y, slice_position)
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).to(device)

    with torch.no_grad():
        sdf_values_yz = sdf_decoder(coords).cpu().reshape(resolution, resolution)
    all_sdf_values.append(sdf_values_yz)

    # Determine global min/max for consistent colorbar
    vmin = min(sdf.min().item() for sdf in all_sdf_values)
    vmax = max(sdf.max().item() for sdf in all_sdf_values)

    # Plot XY plane
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    axes[0].contour(
        X.cpu(), Y.cpu(), sdf_values_xy, levels=[0], colors="red", linewidths=2
    )
    im0 = axes[0].contourf(
        X.cpu(),
        Y.cpu(),
        sdf_values_xy,
        levels=20,
        cmap="RdBu_r",
        alpha=0.6,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"XY plane (z = {slice_position})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # Plot XZ plane
    x = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    X, Z = torch.meshgrid(x, z, indexing="ij")
    axes[1].contour(
        X.cpu(), Z.cpu(), sdf_values_xz, levels=[0], colors="red", linewidths=2
    )
    im1 = axes[1].contourf(
        X.cpu(),
        Z.cpu(),
        sdf_values_xz,
        levels=20,
        cmap="RdBu_r",
        alpha=0.6,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"XZ plane (y = {slice_position})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].set_aspect("equal")

    # Plot YZ plane
    y = torch.linspace(-1, 1, resolution)
    z = torch.linspace(-1, 1, resolution)
    Y, Z = torch.meshgrid(y, z, indexing="ij")
    axes[2].contour(
        Y.cpu(), Z.cpu(), sdf_values_yz, levels=[0], colors="red", linewidths=2
    )
    im2 = axes[2].contourf(
        Y.cpu(),
        Z.cpu(),
        sdf_values_yz,
        levels=20,
        cmap="RdBu_r",
        alpha=0.6,
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title(f"YZ plane (x = {slice_position})")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    axes[2].set_aspect("equal")

    # Add a single colorbar for all subplots with consistent range
    fig.colorbar(im0, ax=axes, shrink=0.8, aspect=30)

    return fig


def visualize_multiple_plants(
    plant_geometries, output_path="all_plants.png", camera_angle=0, fov=60
):
    """Visualize multiple plants in one scene"""

    width, height = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    # Add all plants and compute combined bounding box
    combined_bbox = None
    for i, plant in enumerate(plant_geometries):
        renderer.scene.add_geometry(f"plant_{i}", plant, material)
        if combined_bbox is None:
            combined_bbox = plant.get_axis_aligned_bounding_box()
        else:
            combined_bbox += plant.get_axis_aligned_bounding_box()

    # Calculate camera to see ALL plants
    bounds = combined_bbox.get_extent()
    centroid = combined_bbox.get_center()
    extent = np.linalg.norm(bounds)
    camera_distance = extent * 0.5

    # Camera position rotating around the object (in horizontal plane, looking sideways)
    cam_x = centroid[0] + camera_distance * np.cos(camera_angle)
    cam_z = centroid[2] + camera_distance * np.sin(camera_angle)
    cam_y = centroid[1]  # At same height as centroid for sideways view

    # Set view control
    camera_position = np.array([cam_x, cam_y, cam_z])
    up_vector = [0, 0, 1]  # Z-axis is up

    renderer.setup_camera(fov, centroid, camera_position, up_vector)

    # Lighting
    renderer.scene.scene.set_sun_light([0.707, 0.707, 0], [1.0, 1.0, 1.0], 100000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.set_background([1, 1, 1, 1])

    img = renderer.render_to_image()
    o3d.io.write_image(output_path, img)
    print(f"Saved {len(plant_geometries)} plants to {output_path}")

    return img
