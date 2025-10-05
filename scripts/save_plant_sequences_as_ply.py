"""
Load plant sequences with leaf tips and save as PLY files.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset


def save_as_ply(points, labels, leaf_tip_idxs, output_path, include_leaf_tips=True):
    """
    Save point cloud with labels and leaf tips as PLY file with scalar fields.

    Args:
        points: (N, 3) array of point coordinates
        labels: (N,) array of organ instance labels
        leaf_tip_idxs: array of indices marking leaf tips
        output_path: Path to save PLY file
        include_leaf_tips: Whether to include is_leaf_tip scalar field (only for sparse GT)
    """
    # Create a tensor-based point cloud
    pcd = o3d.t.geometry.PointCloud(points)

    # Attach labels as a point property
    pcd.point["organ_label"] = o3d.core.Tensor(
        labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
    )

    # Create leaf tip mask and attach as property (only for sparse GT files)
    if include_leaf_tips:
        leaf_tip_mask = np.zeros(len(points), dtype=np.uint8)
        if len(leaf_tip_idxs) > 0:
            leaf_tip_mask[leaf_tip_idxs] = 1
        pcd.point["is_leaf_tip"] = o3d.core.Tensor(
            leaf_tip_mask.reshape(-1, 1), dtype=o3d.core.Dtype.UInt8
        )

    # Save as PLY
    o3d.t.io.write_point_cloud(str(output_path), pcd)


def main():
    parser = argparse.ArgumentParser(
        description="Load plant sequences with leaf tips and save as PLY files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="data/TrackPlant3D",
        help="Input directory containing the TrackPlant3D dataset (default: data/TrackPlant3D)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/TrackPlant3D/versions/v1",
        help="Output directory for PLY files (default: data/TrackPlant3D/versions/v1)",
    )
    parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        nargs="*",
        default=None,
        help="List of specific sequence names to process (e.g., 'maize_control_plant2 tomato2_control_plant1'). If not specified, all sequences will be processed.",
    )
    parser.add_argument(
        "--use-ply",
        action="store_true",
        help="Load from PLY files instead of TXT files (default: False, loads from TXT)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["crop", "sequence"],
        default="crop",
        help="Output directory structure: 'crop' groups by crop type (default), 'sequence' creates per-plant directories with sparse/dense subdirectories",
    )
    parser.add_argument(
        "--align-method",
        type=str,
        choices=["pca", "icp"],
        default=None,
        help="Alignment method to apply (default: None)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.input_dir)
    output_base_dir = Path(args.output_dir)

    # Load dataset
    print("Loading plant sequences dataset...")
    print(f"  Input directory: {dataset_path}")
    print(f"  Output directory: {output_base_dir}")
    print(f"  Using PLY input: {args.use_ply}")
    print(f"  Output format: {args.output_format}")

    plant_dataset = PlantSequencesDataset(
        dataset_path,
        use_ply=args.use_ply,
        alignment_method=args.align_method,
    )

    # Determine which sequences to process
    if args.sequences:
        # Validate requested sequences
        available_sequences = plant_dataset.get_sequence_names()
        sequences_to_process = []
        for seq in args.sequences:
            if seq in available_sequences:
                sequences_to_process.append(seq)
            else:
                print(f"Warning: Sequence '{seq}' not found in dataset. Skipping.")

        if not sequences_to_process:
            print("Error: No valid sequences to process.")
            return

        print(f"Processing {len(sequences_to_process)} specified sequence(s):")
        for seq in sequences_to_process:
            print(f"  - {seq}")
    else:
        sequences_to_process = plant_dataset.get_sequence_names()
        print(f"Processing all {len(sequences_to_process)} sequences")

    # Count total files
    total_files = 0
    for sequence_name in sequences_to_process:
        total_files += len(plant_dataset.get_sequence(sequence_name))

    # Process sequences
    with tqdm(total=total_files, desc="Saving PLY files") as pbar:
        for sequence_name in sequences_to_process:
            sequence_data = plant_dataset.get_sequence_data(sequence_name)

            for timepoint_data in sequence_data:
                file_path = timepoint_data["file_path"]
                points = timepoint_data["points"]
                labels = timepoint_data["labels"]
                dense_points = timepoint_data["dense_points"]
                dense_labels = timepoint_data["dense_labels"]
                leaf_tip_idxs = timepoint_data["leaf_tip_idxs"]

                crop_name = file_path.parent.name
                original_filename = file_path.stem + ".ply"

                if args.output_format == "crop":
                    # Original format: group by crop type
                    gt_corrected_dir = output_base_dir / "gt_corrected_v1"
                    dense_dir = output_base_dir / "dense"

                    # 1. Save gt_corrected_v1 (sparse, with leaf tips)
                    gt_crop_dir = gt_corrected_dir / crop_name
                    gt_crop_dir.mkdir(parents=True, exist_ok=True)
                    save_as_ply(
                        points,
                        labels,
                        leaf_tip_idxs,
                        gt_crop_dir / original_filename,
                        include_leaf_tips=True,
                    )

                    # 2. Save dense point clouds if available (without leaf tips)
                    if dense_points is not None and dense_labels is not None:
                        dense_crop_dir = dense_dir / crop_name
                        dense_crop_dir.mkdir(parents=True, exist_ok=True)
                        save_as_ply(
                            dense_points,
                            dense_labels,
                            leaf_tip_idxs,
                            dense_crop_dir / original_filename,
                            include_leaf_tips=False,
                        )

                elif args.output_format == "sequence":
                    # Per-sequence format: each plant gets its own directory
                    sequence_dir = output_base_dir / sequence_name
                    sparse_dir = sequence_dir / "sparse_pc"
                    dense_dir = sequence_dir / "dense_pc"

                    # 1. Save sparse (with leaf tips)
                    sparse_dir.mkdir(parents=True, exist_ok=True)
                    save_as_ply(
                        points,
                        labels,
                        leaf_tip_idxs,
                        sparse_dir / original_filename,
                        include_leaf_tips=True,
                    )

                    # 2. Save dense point clouds if available (without leaf tips)
                    if dense_points is not None and dense_labels is not None:
                        dense_dir.mkdir(parents=True, exist_ok=True)
                        save_as_ply(
                            dense_points,
                            dense_labels,
                            leaf_tip_idxs,
                            dense_dir / original_filename,
                            include_leaf_tips=False,
                        )

                pbar.update(1)

    print(f"\nSuccessfully saved all PLY files to {output_base_dir}")
    if args.output_format == "crop":
        print(f"  Structure: crop-based (gt_corrected_v1/{{crop}}, dense/{{crop}})")
    elif args.output_format == "sequence":
        print(f"  Structure: sequence-based ({{sequence}}/sparse, {{sequence}}/dense)")


if __name__ == "__main__":
    main()
