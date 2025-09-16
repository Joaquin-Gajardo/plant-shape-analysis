import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from plant_shape_analysis.dataloaders.trackplant3D import LeafSequencesDataset


def extract_dense_leaves(base_path: Path = Path("data/TrackPlant3D")):
    # Extract leaf point clouds from TrackPlant3D dataset and save them as individual PLY files

    dataset_path = base_path
    leaf_dataset = LeafSequencesDataset(
        dataset_path,
        min_timepoints=1,  # we get all leaves
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


def process_all_dense_leaves(
    base_path: Path = Path("data/TrackPlant3D"),
    poisson_depth=8,
    density_percentile=10,
    distance_multiplier=10.0,
):
    """Process all dense leaves with trimmed Poisson reconstruction"""

    print(f"Batch processing with parameters:")
    print(f"  poisson_depth={poisson_depth}")
    print(f"  density_percentile={density_percentile}")
    print(f"  distance_multiplier={distance_multiplier}")

    # Extract dense leaves if not already done
    if not (base_path / "dense_leaves").exists():
        extract_dense_leaves(base_path)

    # Create output directory for processed meshes and results
    output_dir = base_path / "dense_leaf_meshes"
    output_dir.mkdir(exist_ok=True)

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


def inspect_leaf_quality(
    base_path: Path = Path("data/TrackPlant3D"), crop_name="maize", max_samples=10
):
    """
    Manual quality inspection tool for checking mesh reconstruction results.
    Shows original point cloud and mesh side-by-side for visual assessment.
    """

    output_dir = base_path / "dense_leaf_meshes" / crop_name
    dense_dir = base_path / "dense_leaves" / crop_name

    if not output_dir.exists():
        print(f"No processed meshes found in {output_dir}")
        return

    mesh_files = sorted(output_dir.glob("*_dense_mesh.ply"))
    print(f"Found {len(mesh_files)} processed meshes")

    for i, mesh_path in enumerate(mesh_files[:max_samples]):
        # Find corresponding dense point cloud
        dense_name = mesh_path.name.replace("_dense_mesh.ply", "_dense.ply")
        dense_path = dense_dir / dense_name

        if not dense_path.exists():
            print(f"Warning: Could not find original point cloud for {mesh_path.name}")
            continue

        print(
            f"\n=== Inspecting {i+1}/{min(max_samples, len(mesh_files))}: {mesh_path.name} ==="
        )

        # Load point cloud and mesh
        pcd = o3d.io.read_point_cloud(str(dense_path))
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        # Color them differently for visualization
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for points
        mesh.paint_uniform_color([0.0, 0.7, 0.0])  # Green for mesh

        print(f"Point cloud: {len(pcd.points)} points")
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        print(f"Surface area: {mesh.get_surface_area():.2f}")

        # Show visualization
        print("Press Q to close and continue to next leaf...")
        o3d.visualization.draw_geometries(
            [pcd, mesh],
            window_name=f"Quality Check: {mesh_path.name}",
            point_show_normal=False,
        )

        # Ask for user feedback
        response = input(
            "Quality assessment (g=good, b=bad, s=skip remaining): "
        ).lower()
        if response == "s":
            break
        elif response == "b":
            print(f"  ❌ Marked {mesh_path.name} as poor quality")
        else:
            print(f"  ✅ Marked {mesh_path.name} as good quality")


def reprocess_with_custom_params(
    file_path: Path,
    poisson_depth=8,
    density_percentile=10,
    distance_multiplier=10.0,
    save_result=True,
):
    """
    Reprocess a specific leaf with custom parameters.
    Useful for fine-tuning parameters on problematic leaves.
    """

    print(f"Reprocessing {file_path.name} with custom parameters:")
    print(f"  poisson_depth={poisson_depth}")
    print(f"  density_percentile={density_percentile}")
    print(f"  distance_multiplier={distance_multiplier}")

    # Load and process
    pcd = load_leaf_point_cloud(file_path)
    mesh, distances = custom_trimmed_poisson(
        pcd, poisson_depth, density_percentile, distance_multiplier
    )

    surface_area = mesh.get_surface_area()
    print(f"Surface area: {surface_area:.2f}")

    # Visualize result
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.paint_uniform_color([0.0, 0.7, 0.0])
    o3d.visualization.draw_geometries([pcd, mesh])

    if save_result:
        # Save with parameter suffix
        output_path = (
            file_path.parent.parent.parent / "dense_leaf_meshes" / file_path.parent.name
        )
        output_path.mkdir(parents=True, exist_ok=True)

        mesh_name = file_path.stem.replace(
            "_dense",
            f"_mesh_d{poisson_depth}_p{density_percentile}_m{distance_multiplier:.1f}.ply",
        )
        mesh_path = output_path / mesh_name

        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Saved custom mesh to: {mesh_path}")

    return mesh, surface_area


def interactive_parameter_tuning(file_path: Path):
    """
    Interactive tool for finding optimal parameters for a specific leaf.
    """

    print(f"=== Interactive Parameter Tuning for {file_path.name} ===")
    print("Commands:")
    print("  process <depth> <density_pct> <dist_mult> - Process with custom params")
    print("  show - Show current result")
    print("  save - Save current result")
    print("  quit - Exit")

    current_mesh = None
    current_params = None

    while True:
        cmd = input("\n> ").strip().split()

        if not cmd:
            continue

        if cmd[0] == "quit":
            break

        elif cmd[0] == "process" and len(cmd) == 4:
            try:
                depth = int(cmd[1])
                density_pct = float(cmd[2])
                dist_mult = float(cmd[3])

                current_mesh, area = reprocess_with_custom_params(
                    file_path, depth, density_pct, dist_mult, save_result=False
                )
                current_params = (depth, density_pct, dist_mult)
                print(
                    f"Parameters: depth={depth}, density_pct={density_pct}, dist_mult={dist_mult}"
                )

            except (ValueError, IndexError):
                print("Usage: process <depth> <density_pct> <dist_mult>")

        elif cmd[0] == "show":
            if current_mesh is not None:
                pcd = load_leaf_point_cloud(file_path)
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                current_mesh.paint_uniform_color([0.0, 0.7, 0.0])
                o3d.visualization.draw_geometries([pcd, current_mesh])
            else:
                print("No mesh processed yet. Use 'process' command first.")

        elif cmd[0] == "save":
            if current_mesh is not None and current_params is not None:
                depth, density_pct, dist_mult = current_params
                reprocess_with_custom_params(
                    file_path, depth, density_pct, dist_mult, save_result=True
                )
            else:
                print("No mesh to save. Use 'process' command first.")

        else:
            print("Unknown command. Available: process, show, save, quit")


def main():
    parser = argparse.ArgumentParser(
        description="Process TrackPlant3D dense leaves with trimmed Poisson reconstruction"
    )

    # Global parameters
    parser.add_argument(
        "--depth", type=int, default=8, help="Poisson depth (default: 8)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=10,
        help="Density percentile threshold (default: 10)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=10.0,
        help="Distance multiplier (default: 10.0)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quality inspection
    inspect_parser = subparsers.add_parser("inspect", help="Quality inspection tool")
    inspect_parser.add_argument(
        "--crop", default="maize", help="Crop name (default: maize)"
    )
    inspect_parser.add_argument(
        "--samples", type=int, default=10, help="Max samples to inspect (default: 10)"
    )

    # Interactive tuning
    tune_parser = subparsers.add_parser("tune", help="Interactive parameter tuning")
    tune_parser.add_argument("leaf_file", help="Path to leaf PLY file")

    # Single reprocessing
    reprocess_parser = subparsers.add_parser("reprocess", help="Reprocess single leaf")
    reprocess_parser.add_argument("leaf_file", help="Path to leaf PLY file")

    args = parser.parse_args()

    base_path = Path("data/TrackPlant3D")

    if args.command == "inspect":
        inspect_leaf_quality(base_path, args.crop, args.samples)
    elif args.command == "tune":
        interactive_parameter_tuning(Path(args.leaf_file))
    elif args.command == "reprocess":
        reprocess_with_custom_params(
            Path(args.leaf_file), args.depth, args.density, args.distance
        )
    else:  # Default to batch processing
        process_all_dense_leaves(base_path, args.depth, args.density, args.distance)


if __name__ == "__main__":
    main()
