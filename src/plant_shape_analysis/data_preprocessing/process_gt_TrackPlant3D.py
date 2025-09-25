import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from plant_shape_analysis.dataloaders.trackplant3D import LeafSequencesDataset


def extract_dense_leaves(base_path: Path = Path("data/TrackPlant3D")):
    # Extract leaf point clouds from TrackPlant3D dataset and save them as individual PLY files

    dataset_path = base_path
    leaf_dataset = LeafSequencesDataset(
        dataset_path, min_timepoints=1, apply_pca_alignment=False  # we get all leaves
    )
    for seq in leaf_dataset:
        for scan in seq["timepoints"]:
            if scan["dense_points"] is not None:
                # save leaf
                crop = "".join(
                    [c for c in seq["sequence_name"].split("_")[0] if c.isalpha()]
                )
                file_name = scan["file_path"].name.replace(
                    ".txt", f"_leaf{scan['leaf_id']}_dense.ply"
                )
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(scan["dense_points"])
                save_dir = base_path / "dense_leaves" / crop
                save_dir.mkdir(parents=True, exist_ok=True)
                o3d.io.write_point_cloud(save_dir / file_name, pcd)


def custom_trimmed_poisson(
    pcd, poisson_depth=8, density_percentile=10, distance_multiplier=10.0
):
    """
    Improve Poisson mesh boundary by filtering vertices that are too far from original points
    """

    # Step 1: Poisson reconstruction
    print("Creating Poisson mesh...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth
    )

    # Step 2: Filter by density
    density_threshold = np.percentile(densities, density_percentile)
    density_mask = densities >= density_threshold
    print(
        f"Density filtering keeps: {np.sum(density_mask)}/{len(density_mask)} vertices"
    )

    # Step 3: Calculate distance threshold based on point cloud spacing
    pcd_points = np.asarray(pcd.points)

    # Find average spacing in point cloud
    nbrs = NearestNeighbors(n_neighbors=5, algorithm="kd_tree").fit(pcd_points)
    distances_pcd, _ = nbrs.kneighbors(pcd_points)
    avg_spacing = np.mean(distances_pcd[:, 1])
    distance_threshold = distance_multiplier * avg_spacing

    print(f"Average point cloud spacing: {avg_spacing:.6f}")
    print(f"Using distance threshold: {distance_threshold:.6f}")

    # Step 4: Filter mesh vertices by distance to original points
    mesh_vertices = np.asarray(mesh.vertices)
    nbrs_mesh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pcd_points)
    distances_to_pcd, _ = nbrs_mesh.kneighbors(mesh_vertices)
    distances_to_pcd = distances_to_pcd.flatten()

    distance_mask = distances_to_pcd <= distance_threshold
    print(
        f"Distance filtering keeps: {np.sum(distance_mask)}/{len(distance_mask)} vertices"
    )

    # Step 5: Combine filters and remove vertices
    vertices_to_remove = ~(density_mask & distance_mask)
    print(f"Removing {np.sum(vertices_to_remove)}/{len(vertices_to_remove)} vertices")

    # Step 6: Apply filtering
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

    return mesh, distances_to_pcd


def load_leaf_point_cloud(file_path):
    """Load leaf point cloud without visualization for batch processing"""

    leaf_pcd = o3d.io.read_point_cloud(str(file_path))

    leaf_pcd.estimate_normals()
    leaf_pcd.orient_normals_to_align_with_direction()
    leaf_pcd.orient_normals_consistent_tangent_plane(k=50)

    return leaf_pcd


def parse_mesh_filename(filename):
    """Extract sequence info from mesh filename."""
    # Pattern: {id}_{crop}_{treatment}_{plant}_D{day:02d}_leaf{leaf_id}_dense_mesh.ply
    pattern = r"(\d+)_([^_]+)_([^_]+)_([^_]+)_D(\d+)_leaf(\d+)_dense_mesh\.ply"
    match = re.match(pattern, filename)
    if match:
        return {
            "id": int(match.group(1)),
            "crop": match.group(2),
            "treatment": match.group(3),
            "plant": match.group(4),
            "day": int(match.group(5)),
            "leaf_id": int(match.group(6)),
        }
    return None


def find_transformation_file(mesh_info, transformations_dir):
    """Find corresponding transformation file for a mesh."""
    trans_filename = f"{mesh_info['crop']}_{mesh_info['treatment']}_{mesh_info['plant']}_leaf{mesh_info['leaf_id']}_transformations.json"
    trans_path = transformations_dir / trans_filename
    return trans_path if trans_path.exists() else None


def apply_transformation_to_mesh(mesh, transformation):
    """Apply transformation to mesh vertices."""
    if transformation is None:
        return mesh

    # Get transformation parameters
    rotation_matrix = np.array(transformation["rotation_matrix"])
    original_center = np.array(transformation["original_center"])
    reference_center = np.array(transformation["reference_center"])

    # Apply transformation to mesh vertices
    vertices = np.asarray(mesh.vertices)

    # Center at original center
    centered_vertices = vertices - original_center

    # Apply rotation
    rotated_vertices = centered_vertices @ rotation_matrix.T

    # Translate to reference center
    transformed_vertices = rotated_vertices + reference_center

    # Update mesh
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

    return mesh


def transform_existing_meshes(
    input_version: str,
    output_version: str,
    transformations_dir: str = None,
    base_path: Path = Path("data/TrackPlant3D"),
):
    """Transform existing meshes using saved transformations."""

    if transformations_dir is None:
        transformations_dir = base_path / "transformations"
    else:
        transformations_dir = Path(transformations_dir)

    input_dir = base_path / "dense_leaf_meshes" / "versions" / input_version
    output_dir = base_path / "dense_leaf_meshes" / "versions" / output_version

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    if not transformations_dir.exists():
        print(f"Error: Transformations directory does not exist: {transformations_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transforming meshes from {input_version} to {output_version}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Transformations: {transformations_dir}")

    # Process each crop directory
    for crop_dir in input_dir.iterdir():
        if not crop_dir.is_dir():
            continue

        crop_output_dir = output_dir / crop_dir.name
        crop_output_dir.mkdir(exist_ok=True)

        # Get all mesh files
        mesh_files = list(crop_dir.glob("*_dense_mesh.ply"))

        print(f"\nProcessing {len(mesh_files)} meshes for {crop_dir.name}...")

        processed_count = 0
        transformed_count = 0

        for mesh_file in tqdm(mesh_files, desc=f"Processing {crop_dir.name}"):
            try:
                # Parse mesh filename
                mesh_info = parse_mesh_filename(mesh_file.name)
                if mesh_info is None:
                    print(f"Could not parse filename: {mesh_file.name}")
                    continue

                # Find transformation file
                trans_file = find_transformation_file(mesh_info, transformations_dir)

                # Load mesh
                mesh = o3d.io.read_triangle_mesh(str(mesh_file))
                if len(mesh.vertices) == 0:
                    print(f"Could not load mesh: {mesh_file.name}")
                    continue

                # Apply transformation if available
                if trans_file:
                    with open(trans_file, "r") as f:
                        transformations = json.load(f)

                    # Find transformation for this day
                    transformation = None
                    for trans in transformations:
                        if trans["day"] == mesh_info["day"]:
                            transformation = trans
                            break

                    if transformation:
                        mesh = apply_transformation_to_mesh(mesh, transformation)
                        transformed_count += 1
                    else:
                        print(
                            f"No transformation found for day {mesh_info['day']} in {mesh_file.name}"
                        )

                # Save transformed mesh
                output_file = crop_output_dir / mesh_file.name
                success = o3d.io.write_triangle_mesh(str(output_file), mesh)

                if success:
                    processed_count += 1
                else:
                    print(f"Failed to save: {output_file}")

            except Exception as e:
                print(f"Error processing {mesh_file.name}: {e}")

        print(
            f"  {crop_dir.name}: {processed_count}/{len(mesh_files)} meshes saved, {transformed_count} transformed"
        )

    print(f"\nTransformation complete! Results saved to {output_dir}")


def process_all_dense_leaves(
    base_path: Path = Path("data/TrackPlant3D"),
    poisson_depth=8,
    density_percentile=10,
    distance_multiplier=10.0,
    version="v1",
):
    """Process all dense leaves with trimmed Poisson reconstruction"""

    print(f"Batch processing with parameters:")
    print(f"  version={version}")
    print(f"  poisson_depth={poisson_depth}")
    print(f"  density_percentile={density_percentile}")
    print(f"  distance_multiplier={distance_multiplier}")

    # Extract dense leaves if not already done
    if not (base_path / "dense_leaves").exists():
        extract_dense_leaves(base_path)

    # Create versioned output directory for processed meshes and results
    output_dir = base_path / "dense_leaf_meshes" / "versions" / version
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all crops
    for crop_dir in (base_path / "dense_leaves").iterdir():
        if not crop_dir.is_dir():
            continue

        crop_name = crop_dir.name
        print(f"\nProcessing crop: {crop_name}")

        # Create crop-specific output directory
        crop_output_dir = output_dir / crop_name
        crop_output_dir.mkdir(exist_ok=True)

        leaf_files = sorted(
            crop_dir.glob("*_dense.ply"), key=lambda x: int(x.stem.split("_")[0])
        )
        print(f"Found {len(leaf_files)} leaves to process")

        results = []

        for i, leaf_path in enumerate(leaf_files):
            try:
                print(f"  Processing {i+1}/{len(leaf_files)}: {leaf_path.name}")

                # Load leaf point cloud without visualization
                pcd = load_leaf_point_cloud(leaf_path)

                # Apply trimmed Poisson reconstruction
                mesh, _ = custom_trimmed_poisson(
                    pcd, poisson_depth, density_percentile, distance_multiplier
                )

                # Calculate surface area
                surface_area = mesh.get_surface_area()

                # Save mesh
                mesh_filename = leaf_path.stem.replace("_dense", "_dense_mesh.ply")
                mesh_path = crop_output_dir / mesh_filename
                o3d.io.write_triangle_mesh(str(mesh_path), mesh)

                # Store results
                results.append(
                    {
                        "leaf_file": leaf_path.name,
                        "mesh_file": mesh_filename,
                        "surface_area": surface_area,
                        "num_vertices": len(mesh.vertices),
                        "num_triangles": len(mesh.triangles),
                    }
                )

                print(f"    Surface area: {surface_area:.2f}")

            except Exception as e:
                print(f"    Error processing {leaf_path.name}: {e}")
                continue

        # Save results summary
        results_file = crop_output_dir / "processing_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Completed {crop_name}: {len(results)} leaves processed successfully")
        print(f"Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process TrackPlant3D dense leaves with trimmed Poisson reconstruction"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand for processing leaves to meshes
    process_parser = subparsers.add_parser(
        "process", help="Process dense leaves to meshes"
    )
    process_parser.add_argument(
        "--depth", type=int, default=8, help="Poisson depth (default: 8)"
    )
    process_parser.add_argument(
        "--density",
        type=float,
        default=10,
        help="Density percentile threshold (default: 10)",
    )
    process_parser.add_argument(
        "--distance",
        type=float,
        default=10.0,
        help="Distance multiplier (default: 10.0)",
    )
    process_parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version name for the output directory (default: v1)",
    )

    # Subcommand for transforming existing meshes
    transform_parser = subparsers.add_parser(
        "transform", help="Transform existing meshes using saved transformations"
    )
    transform_parser.add_argument(
        "--input-version",
        type=str,
        required=True,
        help="Input mesh version to transform (e.g., v1)",
    )
    transform_parser.add_argument(
        "--output-version",
        type=str,
        required=True,
        help="Output version name for transformed meshes (e.g., v1_transformed)",
    )
    transform_parser.add_argument(
        "--transformations-dir",
        type=str,
        help="Directory containing transformation JSON files (default: data/TrackPlant3D/transformations)",
    )

    args = parser.parse_args()

    base_path = Path("data/TrackPlant3D")

    if args.command == "process":
        process_all_dense_leaves(
            base_path, args.depth, args.density, args.distance, args.version
        )
    elif args.command == "transform":
        transform_existing_meshes(
            args.input_version, args.output_version, args.transformations_dir, base_path
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
