import json
from pathlib import Path
from typing import Optional

import numpy as np
import pymeshlab

from data_preprocessing.leaf_meshing import mesh_leaf
from vis.plot_functions import visualize_point_cloud, visualize_pymeshlab_mesh

STEM_LABEL = 0  # Label for stem points in the point cloud
COLOR_MAP = {  # Color mapping for semantic labels to match roughly the TrackPlant3D paper figures
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
    "output_folder": "keypoints_autolabel",  # Default output folder for keypoints,
    "meshing_method": "ball_pivoting",  # Default meshing method, options: "ball_pivoting", "poisson"
    "meshing_backend": "pymeshlab",  # Default backend for meshing, options: "open3d", "pymeshlab"
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
        visualize_pymeshlab_mesh(ms, f"Leaf Mesh")

    # Find leaf centroid
    centroid = np.median(leaf_points, axis=0)
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
            insertion_point, tip_point, insertion_local_index, tip_local_index = (
                extract_leaf_keypoints(
                    leaf_points, stem_points, meshing_method, meshing_backend, visualize
                )
            )

            # Map local indices to global indices
            insertion_global_index = leaf_indices[insertion_local_index]
            tip_global_index = leaf_indices[tip_local_index]

            keypoints_data[int(leaf_id)] = {
                "insertion_point": insertion_point,
                "tip_point": tip_point,
                "insertion_local_index": insertion_local_index,
                "tip_local_index": tip_local_index,
                "insertion_global_index": insertion_global_index,
                "tip_global_index": tip_global_index,
                "leaf_points": leaf_points,
                "num_points": len(leaf_points),
            }

            print(
                f"Leaf {int(leaf_id)}: {len(leaf_points)} points, "
                f"insertion at index {insertion_local_index}, tip at index {tip_local_index}"
            )

            # Visualize if requested
            if visualize:
                visualize_plant_open3d(
                    leaf_points,
                    keypoint_indices=[tip_local_index],
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
    keypoint_indices: Optional[list[int]] = None,
    colors: Optional[np.ndarray] = None,
    window_name: str = "Plant Keypoints Visualization",
):
    """
    Visualize plant/leaf points and keypoints using Open3D

    Args:
        points: numpy array of shape (n_points, 3)
        labels: numpy array of shape (n_points,) with semantic labels
        keypoint_indices: indices of keypoint to highlight
        colors: optional numpy array of shape (n_points, 3) for custom colors
        window_name: name for the Open3D visualization window
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

    visualize_point_cloud(
        points,
        colors=colors,
        window_name=window_name,
    )


def process_single_plant(
    file_path: Path,
    output_folder: Optional[Path] = None,
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
                "insertion_local_index": int(data["insertion_local_index"]),
                "tip_local_index": int(data["tip_local_index"]),
                "insertion_global_index": int(data["insertion_global_index"]),
                "tip_global_index": int(data["tip_global_index"]),
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
    output_folder: str = DEFAULTS["output_folder"],
    visualize: bool = DEFAULTS["visualize"],
    meshing_method: str = DEFAULTS["meshing_method"],
    meshing_backend: str = DEFAULTS["meshing_backend"],
    save_keypoints: bool = DEFAULTS["save_keypoints"],
    save_labelcloud_format: bool = DEFAULTS["save_labelcloud_format"],
):
    print(f"Processing all plants for crop: {crop}")
    data_folder = Path(f"data/TrackPlant3D/gt/{crop}").resolve()
    assert data_folder.exists(), f"Data folder {data_folder} does not exist"

    file_paths = sorted(data_folder.glob("*"), key=lambda x: int(x.stem.split("_")[0]))
    print(f"Data folder: {data_folder}")
    print(f"Found {len(file_paths)} files")

    output_folder_path = (
        data_folder.parent.parent
        / output_folder
        / f"{meshing_method}_{meshing_backend}"
        / crop
    )
    output_folder_path.mkdir(parents=True, exist_ok=True)

    for file_path in file_paths:
        process_single_plant(
            file_path,
            output_folder=output_folder_path,
            meshing_method=meshing_method,
            meshing_backend=meshing_backend,
            visualize=visualize,
            save_keypoints=save_keypoints,
            save_labelcloud_format=save_labelcloud_format,
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect leaf tips in TrackPlant3D data"
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="maize",
        choices=["maize", "sorghum", "tobacco", "tomato"],
        help="Crop type to process (default: maize)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=DEFAULTS["output_folder"],
        help=f"Output folder to save results (default: {DEFAULTS['output_folder']})",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=f"Visualize the point clouds and keypoints",
    )
    parser.add_argument(
        "--meshing_method",
        type=str,
        default=DEFAULTS["meshing_method"],
        choices=["ball_pivoting", "poisson"],
        help=f"Meshing method to use (default: {DEFAULTS['meshing_method']})",
    )
    parser.add_argument(
        "--meshing_backend",
        type=str,
        default=DEFAULTS["meshing_backend"],
        choices=["open3d", "pymeshlab"],
        help=f"Meshing backend to use (default: {DEFAULTS['meshing_backend']})",
    )
    parser.add_argument(
        "--no_save_keypoints",
        action="store_false",
        dest="save_keypoints",
        help="Do not save extracted keypoints to JSON files",
    )
    parser.add_argument(
        "--no_save_labelcloud_format",
        action="store_false",
        dest="save_labelcloud_format",
        help="Do not save keypoints in labelCloud format for manual editing",
    )
    parser.add_argument(
        "--sphere_radius",
        type=float,
        default=DEFAULTS["sphere_radius"],
        help=f"Radius for spheres in labelCloud format (default: {DEFAULTS['sphere_radius']})",
    )
    parser.add_argument(
        "--single_plant_path",
        type=str,
        default=None,
        help="Path to a single plant file to process instead of all plants. Overrides all other arguments"
        " except --meshing_backend and --meshing_method and visualizes only the specified plant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.single_plant_path:
        file_path = Path(args.single_plant_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Single plant file {file_path} does not exist")
        # Process the single plant file
        process_single_plant(
            file_path,
            meshing_method=args.meshing_method,
            meshing_backend=args.meshing_backend,
            visualize=True,  # Always visualize single plant
            save_keypoints=False,  # Disable saving keypoints for single plant processing
            save_labelcloud_format=False,  # Disable saving labelCloud format for single plant processing
        )
    else:
        # Process all plants and save keypoints as points and as spheres (for verification in labelCloud)
        process_all_plants(
            crop=args.crop,
            output_folder=args.output_folder,
            visualize=args.visualize,
            meshing_method=args.meshing_method,
            meshing_backend=args.meshing_backend,
            save_keypoints=args.save_keypoints,
            save_labelcloud_format=args.save_labelcloud_format,
        )


if __name__ == "__main__":
    main()
