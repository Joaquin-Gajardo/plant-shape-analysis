"""
Load plant sequences with leaf tips and save as PLY files.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from src.plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset


def save_as_ply(points, labels, leaf_tip_idxs, output_path):
    """
    Save point cloud with labels and leaf tips as PLY file with scalar fields.

    Args:
        points: (N, 3) array of point coordinates
        labels: (N,) array of organ instance labels
        leaf_tip_idxs: array of indices marking leaf tips
        output_path: Path to save PLY file
    """
    # Create a tensor-based point cloud
    pcd = o3d.t.geometry.PointCloud(points)

    # Attach labels as a point property
    pcd.point["organ_label"] = o3d.core.Tensor(
        labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
    )

    # Create leaf tip mask and attach as property
    leaf_tip_mask = np.zeros(len(points), dtype=np.uint8)
    if len(leaf_tip_idxs) > 0:
        leaf_tip_mask[leaf_tip_idxs] = 1
    pcd.point["is_leaf_tip"] = o3d.core.Tensor(
        leaf_tip_mask.reshape(-1, 1), dtype=o3d.core.Dtype.UInt8
    )

    # Save as PLY
    o3d.t.io.write_point_cloud(str(output_path), pcd)


def main():
    dataset_path = Path("data/TrackPlant3D")
    output_base_dir = Path("data/TrackPlant3D/versions/v1")

    # Create output directories
    gt_corrected_dir = output_base_dir / "gt_corrected_v1"
    dense_dir = output_base_dir / "dense"
    leaf_tips_dir = output_base_dir / "keypoints" / "leaf_tips"

    gt_corrected_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    leaf_tips_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading plant sequences dataset...")
    plant_dataset = PlantSequencesDataset(dataset_path, use_ply=False)

    print(f"Found {len(plant_dataset)} sequences")

    # Process all sequences
    total_files = sum(len(files) for files in plant_dataset.sequences.values())

    with tqdm(total=total_files, desc="Saving PLY files") as pbar:
        for sequence_name in plant_dataset.get_sequence_names():
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

                # 1. Save gt_corrected_v1
                gt_crop_dir = gt_corrected_dir / crop_name
                gt_crop_dir.mkdir(exist_ok=True)
                save_as_ply(
                    points, labels, leaf_tip_idxs, gt_crop_dir / original_filename
                )

                # 2. Save dense point clouds if available
                if dense_points is not None and dense_labels is not None:
                    dense_crop_dir = dense_dir / crop_name
                    dense_crop_dir.mkdir(exist_ok=True)
                    save_as_ply(
                        dense_points,
                        dense_labels,
                        leaf_tip_idxs,
                        dense_crop_dir / original_filename,
                    )

                # 3. Save leaf tips as separate PLY files
                if len(leaf_tip_idxs) > 0:
                    leaf_tips_crop_dir = leaf_tips_dir / crop_name
                    leaf_tips_crop_dir.mkdir(exist_ok=True)

                    tip_coords = points[leaf_tip_idxs]
                    tip_labels = labels[leaf_tip_idxs]

                    # Create point cloud with just leaf tips
                    pcd = o3d.t.geometry.PointCloud(tip_coords)
                    pcd.point["organ_label"] = o3d.core.Tensor(
                        tip_labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
                    )
                    o3d.t.io.write_point_cloud(
                        str(leaf_tips_crop_dir / original_filename), pcd
                    )

                pbar.update(1)

    print(f"\nSaved all PLY files to {output_base_dir}")


if __name__ == "__main__":
    main()
