import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import pymeshlab
from leaf_meshing import mesh_leaf

STEM_LABEL = 0  # Label for stem points in the point cloud
COLOR_MAP = {
    0: np.array([0.0, 1.0, 0.0]),  # Green
    1: np.array([1.0, 0.0, 0.0]),  # Red
    2: np.array([1.0, 1.0, 0.0]),  # Yellow
    3: np.array([0.0, 0.0, 1.0]),  # Blue
    4: np.array([1.0, 0.65, 0.0]),  # Orange
    5: np.array([0.5, 0.0, 0.5]),  # Purple
    6: np.array([0.0, 1.0, 1.0]),  # Cyan
    7: np.array([1.0, 0.0, 1.0]),  # Magenta
    8: np.array([0.5, 1.0, 0.0]),  # Lime (Light Green)
    9: np.array([1.0, 0.75, 0.8]),  # Pink
}
DEFAULTS = {
    "meshing_method": "ball_pivoting",  # Default meshing method, options: "ball_pivoting", "poisson"
    "meshing_backend": "open3d",  # Default backend for meshing, options: "open3d", "pymeshlab"
    "visualize": False,  # Whether to visualize the meshes and keypoints
    "save_keypoints": True,  # Whether to save extracted keypoints
    "save_labelcloud_format": True,  # Whether to save keypoints in labelCloud format
    "sphere_radius": 0.5,  # Radius for spheres in labelCloud format
}


def load_point_cloud_from_txt(file_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to load point cloud data from a text file.
    Assuming the text file has columns: x, y, z, label
    """
    data = np.loadtxt(file_path)
    return data[:, :3], data[:, 3:]  # Assuming last columns are labels


def extract_leaf_keypoints(
    leaf_points: np.ndarray,
    stem_points: np.ndarray,
    meshing_method: str = DEFAULTS["meshing_method"],
    meshing_backend: str = DEFAULTS["meshing_backend"],
    visualize: bool = DEFAULTS["visualize"],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Extract insertion point and tip point for a leaf using geodesic distance.

    Args:
        leaf_points: numpy array of shape (n_leaf_points, 3)
        stem_points: numpy array of shape (n_stem_points, 3)

    Returns:
        insertion_point: numpy array of shape (3,) - leaf insertion point
        tip_point: numpy array of shape (3,) - leaf tip point
        insertion_index: index of insertion point in leaf_points
        tip_index: index of tip point in leaf_points
    """
    # Create mesh for geodesic distance computation
    _, mesh = mesh_leaf(
        leaf_points, method=meshing_method, backend=meshing_backend, visualize=False
    )
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)

    # Transform into PyMeshLab format
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(
        vertex_matrix=points, face_matrix=faces, v_normals_matrix=normals
    )
    ms.add_mesh(mesh, "leaf")

    if visualize:
        visualize_mesh_open3d(ms, f"Leaf Mesh")

    # Find leaf centroid
    centroid = np.mean(leaf_points, axis=0)
    centroid_index = np.argmin(np.linalg.norm(leaf_points - centroid, axis=1))
    centroid_point = np.array(leaf_points[centroid_index], dtype=np.float32)

    # Get farthest point from centroid (candidate A)
    ms.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
        startpoint=centroid_point
    )
    geodist1 = ms.current_mesh().vertex_scalar_array()
    index_A = np.argmax(geodist1)
    point_A = leaf_points[index_A]

    # Get farthest point from point_A (candidate B)
    ms.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
        startpoint=np.array(point_A, dtype=np.float32)
    )
    geodist2 = ms.current_mesh().vertex_scalar_array()
    index_B = np.argmax(geodist2)
    point_B = leaf_points[index_B]

    # Determine which is insertion point (closer to stem) and which is tip
    distances = [
        np.linalg.norm(stem_points - point_A, axis=1).min(),
        np.linalg.norm(stem_points - point_B, axis=1).min(),
    ]
    points = [point_A, point_B]
    indices = [index_A, index_B]

    insertion_idx = np.argmin(distances)
    tip_idx = 1 - insertion_idx

    insertion_point = points[insertion_idx]
    insertion_index = indices[insertion_idx]
    tip_point = points[tip_idx]
    tip_index = indices[tip_idx]

    return insertion_point, tip_point, insertion_index, tip_index


def visualize_mesh_open3d(
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


def process_plant_keypoints(
    points: np.ndarray,
    labels: np.ndarray,
    meshing_method: str = DEFAULTS["meshing_method"],
    meshing_backend: str = DEFAULTS["meshing_backend"],
    visualize: bool = DEFAULTS["visualize"],
) -> dict:
    """
    Process all leaves in a plant to extract keypoints

    Args:
        points: numpy array of shape (n_points, 3) - all plant points
        labels: numpy array of shape (n_points,) - point labels (0=stem, 1+=leaves)
        visualize: bool - whether to show interactive plots

    Returns:
        keypoints_data: dict - keypoint information for each leaf
    """
    unique_labels = np.unique(labels)
    stem_indices = np.where(labels == STEM_LABEL)[0]
    stem_points = points[stem_indices]

    keypoints_data = {}

    for leaf_id in sorted(unique_labels):
        if leaf_id == STEM_LABEL:  # Skip stem
            continue

        # Extract leaf points
        leaf_indices = np.where(labels == leaf_id)[0]
        leaf_points = points[leaf_indices]

        if len(leaf_points) < 10:  # Skip very small leaves
            continue

        try:
            # Extract keypoints
            insertion_point, tip_point, insertion_index, tip_index = (
                extract_leaf_keypoints(
                    leaf_points, stem_points, meshing_method, meshing_backend, visualize
                )
            )

            keypoints_data[int(leaf_id)] = {
                "insertion_point": insertion_point,
                "tip_point": tip_point,
                "insertion_index": insertion_index,
                "tip_index": tip_index,
                "leaf_points": leaf_points,
                "num_points": len(leaf_points),
            }

            print(
                f"Leaf {int(leaf_id)}: {len(leaf_points)} points, "
                f"insertion at index {insertion_index}, tip at index {tip_index}"
            )

            # Visualize if requested
            if visualize:
                visualize_plant_open3d(
                    leaf_points,
                    keypoint_indices=[tip_index],
                    window_name=f"Leaf {int(leaf_id)} and leaf tip (red highlight)",
                )

        except Exception as e:
            print(f"Failed to process leaf {int(leaf_id)}: {e}")

    return keypoints_data


def map_labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """
    Map semantic labels to colors using the COLOR_MAP.

    Args:
        labels: numpy array of shape (n_points,) with semantic labels

    Returns:
        colors: numpy array of shape (n_points, 3) with RGB colors
    """
    colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        if label in COLOR_MAP:
            colors[i] = COLOR_MAP[label]
        else:
            # Default color for unknown labels (black)
            colors[i] = np.array([0.0, 0.0, 0.0])
    return colors


def visualize_plant_open3d(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    keypoint_indices: Optional[list] = None,
    window_name: str = "Plant Keypoints Visualization",
):
    """
    Visualize plant/leaf points and keypoints using Open3D

    Args:
        points: numpy array of shape (n_points, 3)
        labels: numpy array of shape (n_points,) with semantic labels
        tip_indices: indices of keypoint to highlight
    """

    # Array of colors for visualization
    if labels is not None:
        colors = map_labels_to_colors(labels)
    else:
        colors = np.tile(
            [0.5, 0.5, 0.5], (len(points), 1)
        )  # Gray for single leaf points
        if keypoint_indices is not None:
            colors[keypoint_indices] = [1.0, 0.0, 0.0]  # Red for key points

    # Create point cloud
    plant_pcd = o3d.geometry.PointCloud()
    plant_pcd.points = o3d.utility.Vector3dVector(points)
    plant_pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries = [plant_pcd]

    # Visualize
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def process_single_plant(
    file_path: Path,
    output_folder: Path,
    meshing_method: str = DEFAULTS["meshing_method"],
    meshing_backend: str = DEFAULTS["meshing_backend"],
    visualize: bool = DEFAULTS["visualize"],
    save_keypoints: bool = DEFAULTS,
    save_labelcloud_format: bool = DEFAULTS["save_labelcloud_format"],
):

    print(f"\nProcessing: {file_path.name}")

    points, labels = load_point_cloud_from_txt(file_path)
    print(f"Loaded {len(points)} points with {len(np.unique(labels))} unique labels")

    # Visualize the whole plant point cloud
    if visualize:
        visualize_plant_open3d(
            points, labels=labels.flatten(), window_name="Plant with semantic labels"
        )

    # Extract keypoints for all leaves
    keypoints_data = process_plant_keypoints(
        points,
        labels.flatten(),
        meshing_method=meshing_method,
        meshing_backend=meshing_backend,
        visualize=visualize,
    )

    # Print summary
    print(f"\nKeypoints extracted for {len(keypoints_data)} leaves:")
    for leaf_id, data in keypoints_data.items():
        print(f"  Leaf {leaf_id}: {data['num_points']} points")
        print(f"    Insertion: {data['insertion_point']}")
        print(f"    Tip: {data['tip_point']}")

    # Optionally save keypoints
    if save_keypoints:
        # Save in original format
        output_path = output_folder / "leaf_tips" / f"{file_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for leaf_id, data in keypoints_data.items():
            json_data[leaf_id] = {
                "insertion_point": data["insertion_point"].tolist(),
                "tip_point": data["tip_point"].tolist(),
                "insertion_index": int(data["insertion_index"]),
                "tip_index": int(data["tip_index"]),
                "num_points": data["num_points"],
            }

        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nKeypoints saved to {output_path}")

        # Also save in labelCloud format if requested
        if save_labelcloud_format:
            save_keypoints_labelcloud_format(keypoints_data, file_path, output_folder)


def save_keypoints_labelcloud_format(
    keypoints_data: dict,
    file_path: Path,
    output_folder: Path,
    sphere_radius: float = DEFAULTS["sphere_radius"],
    round_decimals: int = 6,
):
    """
    Save keypoints as spheres in json format for inspection and manual editing in labelCloud.

    Args:
        keypoints_data: dict - keypoint information for each leaf
        file_path: Path - original plant data file path (used for naming)
        output_folder: Path - folder to save the labelCloud format file
        sphere_radius: float - radius for the annotation spheres
        round_decimals: int - number of decimal places to round coordinates
    """

    def round_dec(value: float) -> float:
        """Round value to specified decimal places"""
        return round(float(value), round_decimals)

    # Initialize labelCloud format structure
    labelcloud_data = {
        "folder": file_path.parent.name,
        "filename": str(file_path.name),
        "objects": [],
        "spheres": [],
    }

    # Add spheres for each leaf's keypoints
    for leaf_id, data in keypoints_data.items():
        tip_sphere = {
            "name": f"leaf_tip",
            "center": {
                "x": round_dec(data["tip_point"][0]),
                "y": round_dec(data["tip_point"][1]),
                "z": round_dec(data["tip_point"][2]),
            },
            "radius": round_dec(sphere_radius),
        }
        labelcloud_data["spheres"].append(tip_sphere)

    # Save to file
    output_path = output_folder / "leaf_tip_spheres" / f"{file_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(labelcloud_data, f, indent=2)

    print(f"LabelCloud format saved to {output_path}")
    return output_path


def process_all_plants(
    crop: str = "sorghum",
    visualize: bool = DEFAULTS["visualize"],
    meshing_method: str = DEFAULTS["meshing_method"],
    meshing_backend: str = DEFAULTS["meshing_backend"],
    save_keypoints: bool = DEFAULTS["save_keypoints"],
    save_labelcloud_format: bool = DEFAULTS["save_labelcloud_format"],
):
    data_folder = Path(f"data/TrackPlant3D/gt/{crop}").resolve()
    file_paths = sorted(data_folder.glob("*"), key=lambda x: int(x.stem.split("_")[0]))
    print(f"Data folder: {data_folder}")
    print(f"Found {len(file_paths)} files")

    output_folder = (
        data_folder.parent.parent
        / "keypoints_autolabel"
        / f"{meshing_method}_{meshing_backend}"
        / crop
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in file_paths:
        process_single_plant(
            file_path,
            output_folder=output_folder,
            meshing_method=meshing_method,
            meshing_backend=meshing_backend,
            visualize=visualize,
            save_keypoints=save_keypoints,
            save_labelcloud_format=save_labelcloud_format,
        )


def main():
    # Process all plants and save keypoints as points and as spheres (for verification in labelCloud)
    crop = "maize"  # Change to "maize" or other crops as needed
    print(f"Processing all plants for crop: {crop}")
    process_all_plants(
        crop=crop,
        visualize=DEFAULTS["visualize"],
        save_keypoints=DEFAULTS["save_keypoints"],
        save_labelcloud_format=DEFAULTS["save_labelcloud_format"],
    )


if __name__ == "__main__":
    main()
