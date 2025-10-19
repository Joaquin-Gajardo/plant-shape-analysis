"""
Save aligned plant sequences with corrected normals as v2 PLY files.

This script:
1. Loads plant sequences with alignment
2. Creates leaf dataset to get temporally consistent normals
3. Maps corrected normals from leaves back to plant point clouds
4. Saves aligned plants as PLY files with normals
5. Saves transformation matrices for reproducibility

The resulting v2 dataset can be loaded directly without expensive alignment and normal computation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from plant_shape_analysis.dataloaders.trackplant3D import (
    LeafSequencesDataset,
    PlantSequencesDataset,
)


def map_leaf_normals_to_plant(plant_timeseries, leaf_dataset):
    """
    Map corrected normals from leaf sequences back to plant point clouds.

    After leaf alignment, normals have temporal consistency. This function
    extracts those corrected normals, transforms them back to plant-aligned
    coordinate system, and assigns them to the plant point clouds.

    Args:
        plant_timeseries: Plant timeseries dict from PlantSequencesDataset
        leaf_dataset: LeafSequencesDataset with corrected normals

    Returns:
        Updated plant timeseries with corrected normals
    """
    sequence_name = plant_timeseries["sequence_name"]

    # Get all leaf sequences for this plant
    leaf_sequences = leaf_dataset.get_timeseries_by_plant_sequence(sequence_name)

    if not leaf_sequences:
        # No leaves found, keep original normals (if any)
        return plant_timeseries

    # Process each timepoint
    for timepoint in plant_timeseries["timepoints"]:
        day = timepoint["day"]
        points = timepoint["points"]
        labels = timepoint["labels"]

        # Initialize normals array (zeros if not present)
        if "normals" not in timepoint or timepoint["normals"] is None:
            normals = np.zeros_like(points)
        else:
            normals = timepoint["normals"].copy()

        # Map normals from each leaf
        for leaf_seq in leaf_sequences:
            leaf_id = leaf_seq["leaf_id"]

            # Find matching timepoint in leaf sequence
            leaf_timepoint = None
            leaf_tp_idx = None
            for idx, leaf_tp in enumerate(leaf_seq["timepoints"]):
                if leaf_tp["day"] == day:
                    leaf_timepoint = leaf_tp
                    leaf_tp_idx = idx
                    break

            if leaf_timepoint is None or leaf_timepoint.get("normals") is None:
                continue

            # Get leaf normals (in leaf-aligned coordinate system)
            leaf_normals_aligned = leaf_timepoint["normals"]

            # Transform normals back to plant-aligned coordinate system
            # Apply inverse of leaf alignment transformations (rotation only for normals)
            if leaf_seq.get("transformation_stages") is not None:
                # Compose all transformation stages for this timepoint
                composed_transforms = LeafSequencesDataset.compose_transformations(
                    leaf_seq["transformation_stages"]
                )

                if leaf_tp_idx < len(composed_transforms):
                    # Get inverse transformation matrix
                    composed_matrix = composed_transforms[leaf_tp_idx]["composed_matrix"]
                    inv_matrix = LeafSequencesDataset.invert_transformation(
                        composed_matrix
                    )

                    # Extract rotation part only (upper-left 3x3)
                    R_inv = inv_matrix[:3, :3]

                    # Apply inverse rotation to normals
                    leaf_normals_plant_space = leaf_normals_aligned @ R_inv.T
                else:
                    # No transformation, use normals as-is
                    leaf_normals_plant_space = leaf_normals_aligned
            else:
                # No transformation stages (alignment not applied)
                leaf_normals_plant_space = leaf_normals_aligned

            # Get mask for this leaf in plant point cloud
            leaf_mask = labels == leaf_id

            # Assign corrected normals from leaf to plant
            if np.sum(leaf_mask) == len(leaf_normals_plant_space):
                normals[leaf_mask] = leaf_normals_plant_space
            else:
                print(
                    f"Warning: Size mismatch for {sequence_name} day {day} leaf {leaf_id}"
                )

        # Update timepoint with corrected normals
        timepoint["normals"] = normals

    return plant_timeseries


def save_as_ply(
    points, labels, leaf_tip_idxs, normals=None, output_path=None, include_leaf_tips=True
):
    """
    Save point cloud with labels, normals, and leaf tips as PLY file.

    Args:
        points: (N, 3) array of point coordinates
        labels: (N,) array of organ instance labels
        leaf_tip_idxs: array of indices marking leaf tips
        normals: (N, 3) array of normal vectors (optional)
        output_path: Path to save PLY file
        include_leaf_tips: Whether to include is_leaf_tip scalar field
    """
    # Create a tensor-based point cloud
    pcd = o3d.t.geometry.PointCloud(points)

    # Attach labels as a point property
    pcd.point["organ_label"] = o3d.core.Tensor(
        labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
    )

    # Attach normals if available
    if normals is not None:
        pcd.point["normals"] = o3d.core.Tensor(
            normals.astype(np.float32), dtype=o3d.core.Dtype.Float32
        )

    # Create leaf tip mask and attach as property
    if include_leaf_tips:
        leaf_tip_mask = np.zeros(len(points), dtype=np.uint8)
        if len(leaf_tip_idxs) > 0:
            leaf_tip_mask[leaf_tip_idxs] = 1
        pcd.point["is_leaf_tip"] = o3d.core.Tensor(
            leaf_tip_mask.reshape(-1, 1), dtype=o3d.core.Dtype.UInt8
        )

    # Save as PLY
    o3d.t.io.write_point_cloud(str(output_path), pcd)


def save_transformations(plant_dataset, output_dir):
    """
    Save transformation matrices for all plant sequences.

    Args:
        plant_dataset: PlantSequencesDataset with transformations
        output_dir: Output directory for transformation files
    """
    transform_dir = output_dir / "transformations"
    transform_dir.mkdir(parents=True, exist_ok=True)

    for sequence_name, transformations in plant_dataset.transformations.items():
        filename = f"{sequence_name}_transformations.json"
        filepath = transform_dir / filename

        json_data = {"sequence_name": sequence_name, "timepoints": []}

        # Convert numpy arrays to lists for JSON serialization
        for trans in transformations:
            json_trans = {}

            # Copy all fields
            for key, value in trans.items():
                if isinstance(value, np.ndarray):
                    json_trans[key] = value.tolist()
                else:
                    json_trans[key] = value

            # Build 4x4 homogeneous transformation matrix
            R = trans.get("rotation_matrix", np.eye(3))
            t = trans.get("translation", np.zeros(3))
            orig_c = trans.get("original_center", np.zeros(3))
            ref_c = trans.get("reference_center", np.zeros(3))
            v_shift = trans.get("vertical_shift", np.zeros(3))

            # Build transformation matrix: T = T(ref_c + t + v_shift) @ R @ T(-orig_c)
            T_neg_c = np.eye(4, dtype=np.float64)
            T_neg_c[:3, 3] = -orig_c

            T_R = np.eye(4, dtype=np.float64)
            T_R[:3, :3] = R.T  # Transpose for row vectors

            T_c_plus_t = np.eye(4, dtype=np.float64)
            T_c_plus_t[:3, 3] = ref_c + t + v_shift

            composed_matrix = T_c_plus_t @ T_R @ T_neg_c

            # Add composed matrices
            json_trans["composed_matrix"] = composed_matrix.tolist()
            json_trans["inverse_matrix"] = np.linalg.inv(composed_matrix).tolist()

            json_data["timepoints"].append(json_trans)

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Save aligned plant sequences with corrected normals as v2 PLY files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="data/TrackPlant3D/versions",
        help="Input directory containing the TrackPlant3D dataset (default: data/TrackPlant3D/versions)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/TrackPlant3D/versions/v2",
        help="Output directory for v2 PLY files (default: data/TrackPlant3D/versions/v2)",
    )
    parser.add_argument(
        "--alignment-method",
        type=str,
        choices=["pca", "icp", "stem_based"],
        default="stem_based",
        help="Plant alignment method (default: stem_based)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        default=None,
        help="List of specific sequence names to process. If not specified, all sequences will be processed.",
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        help="Also save dense point clouds (default: False, only sparse)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.input_dir)
    output_base_dir = Path(args.output_dir)

    print("=" * 80)
    print("Saving aligned plant sequences with corrected normals as v2")
    print("=" * 80)
    print(f"Input directory: {dataset_path}")
    print(f"Output directory: {output_base_dir}")
    print(f"Alignment method: {args.alignment_method}")
    print(f"Save dense: {args.save_dense}")
    print()

    # Step 1: Create leaf dataset to get corrected normals
    # This internally creates plant_dataset with alignment and normals
    print("Step 1/2: Creating leaf dataset to compute corrected normals...")
    print("  (This loads plants with alignment and estimates temporally consistent normals)")
    leaf_dataset = LeafSequencesDataset(
        dataset_path,
        version="v1",
        use_ply=True,
        plant_alignment_method=args.alignment_method,
        estimate_plant_normals=True,  # Estimate normals for plants
        apply_alignment=True,  # Apply leaf alignment to correct normals
        save_transformations=False,
    )

    # Access the plant dataset from leaf_dataset
    plant_dataset = leaf_dataset.plant_dataset

    # Determine which sequences to process
    if args.sequences:
        available_sequences = plant_dataset.get_sequence_names()
        sequences_to_process = []
        for seq in args.sequences:
            if seq in available_sequences:
                sequences_to_process.append(seq)
            else:
                print(f"Warning: Sequence '{seq}' not found. Skipping.")

        if not sequences_to_process:
            print("Error: No valid sequences to process.")
            return
    else:
        sequences_to_process = plant_dataset.get_sequence_names()

    print(f"\nStep 2/2: Processing and saving {len(sequences_to_process)} sequence(s)...")

    # Create output directories
    gt_corrected_dir = output_base_dir / "gt_corrected_v2"
    dense_dir = output_base_dir / "dense"

    # Count total files for progress bar
    total_files = 0
    for sequence_name in sequences_to_process:
        plant_ts = plant_dataset.get_timeseries_by_sequence_name(sequence_name)
        total_files += len(plant_ts["timepoints"])

    # Process sequences
    with tqdm(total=total_files, desc="Saving v2 PLY files") as pbar:
        for sequence_name in sequences_to_process:
            # Get plant timeseries
            plant_ts = plant_dataset.get_timeseries_by_sequence_name(sequence_name)

            # Map corrected normals from leaves back to plant
            plant_ts = map_leaf_normals_to_plant(plant_ts, leaf_dataset)

            # Save each timepoint
            for timepoint in plant_ts["timepoints"]:
                file_path = timepoint["file_path"]
                points = timepoint["points"]
                labels = timepoint["labels"]
                leaf_tip_idxs = timepoint["leaf_tip_idxs"]
                normals = timepoint.get("normals", None)
                dense_points = timepoint["dense_points"]
                dense_labels = timepoint["dense_labels"]

                crop_name = file_path.parent.name
                original_filename = file_path.stem + ".ply"

                # Save sparse point cloud with normals
                gt_crop_dir = gt_corrected_dir / crop_name
                gt_crop_dir.mkdir(parents=True, exist_ok=True)
                save_as_ply(
                    points,
                    labels,
                    leaf_tip_idxs,
                    normals=normals,
                    output_path=gt_crop_dir / original_filename,
                    include_leaf_tips=True,
                )

                # Save dense point clouds if requested
                if args.save_dense and dense_points is not None:
                    dense_crop_dir = dense_dir / crop_name
                    dense_crop_dir.mkdir(parents=True, exist_ok=True)
                    # Note: Dense points don't have normals or leaf tips
                    save_as_ply(
                        dense_points,
                        dense_labels,
                        np.array([]),
                        normals=None,
                        output_path=dense_crop_dir / original_filename,
                        include_leaf_tips=False,
                    )

                pbar.update(1)

    # Save transformation matrices
    print("\nSaving transformation matrices...")
    save_transformations(plant_dataset, output_base_dir)

    print("\n" + "=" * 80)
    print("Success! v2 dataset saved to:", output_base_dir)
    print("=" * 80)
    print(f"  Aligned point clouds: {gt_corrected_dir}")
    print(f"  Transformations: {output_base_dir / 'transformations'}")
    if args.save_dense:
        print(f"  Dense point clouds: {dense_dir}")
    print("\nNow you can load v2 with:")
    print('  PlantSequencesDataset(dataset_path, version="v2", alignment_method=None, estimate_normals=False)')


if __name__ == "__main__":
    main()
