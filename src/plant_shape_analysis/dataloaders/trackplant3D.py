import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

from plant_shape_analysis.vis.plot_functions import plot_pairwise_alignment_with_quivers


class PlantSequencesDataset(Dataset):
    def __init__(self, dataset_path):
        self.point_clouds_path = Path(dataset_path) / "gt_corrected_v1"
        self.leaf_tips_path = Path(dataset_path) / "keypoints" / "leaf_tips"
        self.dense_path = Path(dataset_path) / "dense"

        # Organize files into sequences
        self.sequences = self._organize_sequences()

    def _organize_sequences(self):
        """Organize files into sequences based on crop type, treatment, and plant number"""
        sequences = defaultdict(list)

        # Get all crop folders
        crop_folders = [d for d in self.point_clouds_path.iterdir() if d.is_dir()]

        for crop_folder in crop_folders:
            crop_name = crop_folder.name

            # Get all txt files in the crop folder
            txt_files = list(crop_folder.glob("*.txt"))

            # Group files by full sequence identifier (crop_treatment_plant)
            sequence_groups = defaultdict(list)

            for txt_file in txt_files:
                # Extract sequence identifier from filename
                # Example: "1_maize_control_plant1_D00.txt" -> "maize_control_plant1"
                # Pattern: number_crop_treatment_plantX_DXX.txt
                match = re.search(r"\d+_(.+)_D\d+\.txt$", txt_file.name)
                if match:
                    # Extract everything between the initial number and the day
                    sequence_part = match.group(1)  # e.g., "maize_control_plant1"
                    sequence_groups[sequence_part].append(txt_file)
                else:
                    # Fallback: try to extract manually
                    # Remove the leading number and trailing day part
                    name_parts = txt_file.stem.split("_")
                    if len(name_parts) >= 4:  # At least: number, crop, treatment, plant
                        sequence_part = "_".join(
                            name_parts[1:-1]
                        )  # Skip first number and last day
                        sequence_groups[sequence_part].append(txt_file)

            # Sort files within each sequence group by day
            for sequence_id, files in sequence_groups.items():
                # Sort by day number (D00, D01, etc.)
                files.sort(key=lambda x: int(re.search(r"D(\d+)", x.name).group(1)))
                sequences[sequence_id] = files

        return dict(sequences)

    def get_sequence_names(self):
        """Get all sequence names"""
        return list(self.sequences.keys())

    def get_sequences_by_crop(self, crop_name):
        """Get all sequences for a specific crop"""
        return {k: v for k, v in self.sequences.items() if k.startswith(crop_name)}

    def get_sequences_by_treatment(self, treatment):
        """Get all sequences for a specific treatment"""
        return {k: v for k, v in self.sequences.items() if treatment in k}

    def get_sequence(self, sequence_name):
        """Get all files for a specific sequence"""
        return self.sequences.get(sequence_name, [])

    def load_point_cloud(self, file_path):
        """Load point cloud from txt file"""
        # File format: x, y, z, organ_instance_label
        data = np.loadtxt(file_path)
        points = data[:, :3]  # x, y, z coordinates
        labels = data[:, 3].astype(int)  # labels
        return points, labels

    def load_dense_point_cloud(self, file_path):
        """Load dense point cloud for a specific sequence and day if available"""
        crop_name = file_path.parent.name
        dense_file_path = self.dense_path / crop_name / file_path.name
        if dense_file_path.exists():
            return self.load_point_cloud(dense_file_path)
        return None, None

    def load_leaf_tips(self, point_cloud_path):
        """Load leaf tips for a specific sequence and day"""
        crop_name = point_cloud_path.parent.name
        file_path = self.leaf_tips_path / crop_name / point_cloud_path.name

        # Load with comma delimiter - format: global_index, x, y, z
        global_indices = np.array([])
        coordinates = np.array([])
        if file_path.exists():
            data = np.loadtxt(file_path, delimiter=",")
            if data.ndim == 1:  # Single row
                data = data.reshape(1, -1)

            # Extract global indices and coordinates
            global_indices = data[:, 0].astype(int)
            coordinates = data[:, 1:4]
        # Note: Leaf tips may not exist for all files, which is expected

        return global_indices, coordinates

    def get_sequence_data(self, sequence_name):
        """Get all point clouds and leaf tips for a sequence"""
        files = self.get_sequence(sequence_name)
        sequence_data = []

        for file_path in files:
            # Extract day from filename
            day_match = re.search(r"D(\d+)", file_path.name)
            day = int(day_match.group(1)) if day_match else 0

            # Load point cloud
            points, labels = self.load_point_cloud(file_path)

            # Load dense point cloud if available
            dense_points, dense_labels = self.load_dense_point_cloud(file_path)

            # Load leaf tips if available
            leaf_tip_idxs, leaf_tip_coordinates = self.load_leaf_tips(file_path)
            if leaf_tip_idxs.size > 0:
                # Check if leaf tip coordinates match (with tolerance for floating point)
                try:
                    assert np.allclose(
                        points[leaf_tip_idxs],
                        leaf_tip_coordinates,
                        rtol=1e-5,
                        atol=1e-8,
                    )
                except (AssertionError, IndexError) as e:
                    print(
                        f"Warning: Leaf tip coordinate mismatch in {file_path.name}: {e}"
                    )
                    print("Leaf tip coordinates: \n", leaf_tip_coordinates)
                    print("Point cloud coordinates: \n", points[leaf_tip_idxs])
                    leaf_tip_idxs = np.array([])  # Clear invalid leaf tips

            sequence_data.append(
                {
                    "day": day,
                    "points": points,
                    "labels": labels,
                    "dense_points": dense_points,
                    "dense_labels": dense_labels,
                    "leaf_tip_idxs": leaf_tip_idxs,
                    "file_path": file_path,
                }
            )

        return sequence_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_name = list(self.sequences.keys())[idx]
        return self.get_sequence_data(sequence_name)


class LeafSequencesDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        min_timepoints: int = 3,
        max_timepoints: Optional[int] = None,
        apply_pca_alignment: bool = False,
        save_transformations: bool = False,
        estimate_normals: bool = False,
    ):
        self.plant_dataset = PlantSequencesDataset(dataset_path)
        self.dataset_path = Path(dataset_path)
        self.min_timepoints = min_timepoints
        self.max_timepoints = max_timepoints
        self.apply_pca_alignment = apply_pca_alignment
        self._save_transformations = save_transformations
        self.estimate_normals = estimate_normals

        # Build leaf timeseries samples
        self.leaf_timeseries = self._build_leaf_timeseries()

        # Apply PCA alignment if requested
        if self.apply_pca_alignment:
            print(
                f"Applying PCA alignment to {len(self.leaf_timeseries)} leaf sequences..."
            )
            self._apply_pca_alignment_to_dataset()

        if self.estimate_normals:
            self._estimate_normals()

    def get_sequence_names(self, sequence_list: Optional[list] = None):
        """Get all sequence names"""
        sequence = sequence_list or self.leaf_timeseries
        return [ts["sequence_name"] for ts in sequence]

    def get_sequences_by_treatment(
        self, treatment: str, sequence_list: Optional[list] = None
    ):
        """Get all sequences for a specific treatment"""
        sequences = []
        for ts in sequence_list or self.leaf_timeseries:
            if treatment in ts["sequence_name"]:
                sequences.append(ts)
        return sequences

    def _build_leaf_timeseries(self):
        """Build individual leaf timeseries from plant sequences"""
        leaf_timeseries = []

        for sequence_name in self.plant_dataset.get_sequence_names():
            sequence_data = self.plant_dataset.get_sequence_data(sequence_name)

            # Track leaves across time points
            leaf_tracks = self._track_leaves_across_time(sequence_data)

            # Filter by minimum timepoints requirement
            for leaf_id, timepoints in leaf_tracks.items():
                if len(timepoints) >= self.min_timepoints:
                    if (
                        self.max_timepoints is None
                        or len(timepoints) <= self.max_timepoints
                    ):
                        leaf_timeseries.append(
                            {
                                "sequence_name": sequence_name,
                                "leaf_id": int(leaf_id),
                                "timepoints": timepoints,
                            }
                        )

        return leaf_timeseries

    def _track_leaves_across_time(self, sequence_data):
        """Track individual leaves across time points in a sequence"""
        leaf_tracks = defaultdict(list)

        for timepoint_data in sequence_data:
            day = timepoint_data["day"]
            points = timepoint_data["points"]
            labels = timepoint_data["labels"]
            dense_points = timepoint_data["dense_points"]
            dense_labels = timepoint_data["dense_labels"]
            leaf_tip_idxs = timepoint_data["leaf_tip_idxs"]

            # Get unique leaf labels (excluding stem label 0)
            unique_leaves = np.unique(labels[labels > 0])

            for leaf_label in unique_leaves:
                # Extract points for this leaf
                leaf_mask = labels == leaf_label
                leaf_points = points[leaf_mask]

                # Extract dense points for this leaf if available
                dense_leaf_points = None
                if dense_points is not None and dense_labels is not None:
                    dense_leaf_mask = dense_labels == leaf_label
                    if np.any(dense_leaf_mask):
                        dense_leaf_points = dense_points[dense_leaf_mask]

                # Find leaf tip if available
                leaf_tip_coords = None
                if leaf_tip_idxs.size > 0:
                    # Map global indices to leaf-local indices
                    leaf_global_idxs = np.where(leaf_mask)[0]
                    tip_in_leaf = np.isin(leaf_tip_idxs, leaf_global_idxs)
                    if np.any(tip_in_leaf):
                        tip_global_idx = leaf_tip_idxs[tip_in_leaf][0]
                        leaf_tip_coords = points[tip_global_idx]

                leaf_tracks[leaf_label].append(
                    {
                        "day": day,
                        "points": leaf_points,
                        "dense_points": dense_leaf_points,
                        "leaf_tip": leaf_tip_coords,
                        "file_path": timepoint_data["file_path"],
                    }
                )

        # Sort each leaf track by day
        for leaf_id in leaf_tracks:
            leaf_tracks[leaf_id].sort(key=lambda x: x["day"])

        return dict(leaf_tracks)

    def _estimate_normals(self):
        print("Estimating normals for all leaves in the dataset...")
        for leaf in self.leaf_timeseries:
            for timepoint in leaf["timepoints"]:
                points = timepoint["points"]
                if points is not None:
                    # Estimate normals using Open3D
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.estimate_normals()
                    pcd.orient_normals_to_align_with_direction()
                    pcd.orient_normals_consistent_tangent_plane(k=30)
                    normals = np.asarray(pcd.normals)
                    timepoint["points"] = np.hstack([timepoint["points"], normals])

    def _apply_pca_alignment_to_dataset(self):
        """Apply PCA alignment to all leaf sequences in the dataset."""

        aligned_timeseries = []
        for i, leaf_ts in enumerate(self.leaf_timeseries):
            if len(leaf_ts["timepoints"]) >= 2:
                # Apply PCA alignment with first timepoint as reference
                aligned_timepoints, transformations = self.align_leaf_sequence(
                    leaf_ts, reference_idx=0
                )

                # Update the leaf timeseries with aligned data
                aligned_leaf_ts = leaf_ts.copy()
                aligned_leaf_ts["timepoints"] = aligned_timepoints
                aligned_leaf_ts["is_aligned"] = True
                aligned_leaf_ts["transformations"] = transformations

                aligned_timeseries.append(aligned_leaf_ts)

                # Save transformations if requested
                if self._save_transformations:
                    self.save_transformations(leaf_ts, transformations)

            else:
                # Keep original if insufficient timepoints for alignment
                leaf_ts["is_aligned"] = False
                aligned_timeseries.append(leaf_ts)

        # Replace original timeseries with aligned ones
        self.leaf_timeseries = aligned_timeseries
        print(
            f"PCA alignment complete. {len([ts for ts in self.leaf_timeseries if ts.get('is_aligned', False)])} sequences aligned."
        )

    def get_leaf_timeseries_info(self):
        """Get summary information about leaf timeseries"""
        info = {
            "total_timeseries": len(self.leaf_timeseries),
            "timepoint_distribution": defaultdict(int),
            "crop_distribution": defaultdict(int),
            "treatment_distribution": defaultdict(int),
        }

        for ts in self.leaf_timeseries:
            sequence_name = ts["sequence_name"]
            num_timepoints = len(ts["timepoints"])

            info["timepoint_distribution"][num_timepoints] += 1

            # Extract crop and treatment from sequence name
            parts = sequence_name.split("_")
            if len(parts) >= 2:
                crop = parts[0]
                treatment = parts[1]
                info["crop_distribution"][crop] += 1
                info["treatment_distribution"][treatment] += 1

        return dict(info)

    def get_timeseries_by_crop(self, crop_name):
        """Get leaf timeseries filtered by crop"""
        return [
            ts
            for ts in self.leaf_timeseries
            if ts["sequence_name"].startswith(crop_name)
        ]

    def get_timeseries_by_treatment(self, treatment):
        """Get leaf timeseries filtered by treatment"""
        return [ts for ts in self.leaf_timeseries if treatment in ts["sequence_name"]]

    def _pca_align(self, pc1, pc2):
        """
        PCA-based alignment that doesn't require same number of points.
        Ensures consistent orientation by aligning to reference (pc2) principal components.

        Args:
            pc1: First point cloud (centered) - to be aligned
            pc2: Second point cloud (centered) - reference

        Returns:
            Rotation matrix to align pc1 to pc2's coordinate system
        """

        def get_pca(pc):
            pc_centered = pc - pc.mean(axis=0)
            U, S, Vt = np.linalg.svd(pc_centered, full_matrices=False)
            is_sorted = np.all(S[:-1] >= S[1:])
            if not is_sorted:
                print("Singular values not sorted descending")
            return Vt  # rows are components

        def align_components_to_reference(components, reference_components):
            """Align principal components to match reference orientation"""
            aligned_components = components.copy()

            for i in range(min(len(components), len(reference_components))):
                # Check dot product to determine if we need to flip
                dot_product = np.dot(components[i], reference_components[i])

                # If dot product is negative, flip the component to align with reference
                if dot_product < 0:
                    aligned_components[i] = -aligned_components[i]

            return aligned_components

        # Get PCA components for both point clouds
        R1_raw = get_pca(pc1)
        R2 = get_pca(pc2)

        # Align pc1 components to match pc2 reference orientation
        R1 = align_components_to_reference(R1_raw, R2)

        # Compute rotation matrix to align pc1 to pc2
        R = R2.T @ R1

        # Additional check: ensure rotation doesn't introduce a flip
        # Check if determinant is negative (indicates reflection/flip)
        if np.linalg.det(R) < 0:
            print("Reflection detected: flipping first principal component")
            # If we have a reflection, flip the last principal component
            R1[0] = -R1[
                0
            ]  # NOTE: this helps for the first timeseries last leaf compared to flipping the third SV
            R = R2.T @ R1

        # self.check_singular_vector_consistency(R1, R2)

        return R, R1

    def align_leaf_sequence(self, leaf_timeseries, reference_idx=0):
        """
        Align leaf sequence using PCA-based registration, preserving scale differences.
        Only removes rotation and translation to show growth over time.
        Works with different numbers of points between timepoints.
        Ensures consistent orientation across all timepoints.

        Args:
            leaf_timeseries: Leaf timeseries dict from dataset
            reference_idx: Index of reference timepoint (default: 0)

        Returns:
            List of aligned point clouds and rotation matrices
        """
        timepoints = leaf_timeseries["timepoints"]
        if len(timepoints) < 2:
            return timepoints, []

        # Get reference timepoint
        ref_points = timepoints[reference_idx]["points"]
        ref_center = np.mean(ref_points, axis=0)
        ref_centered = ref_points - ref_center

        aligned_timepoints = []
        transformations = []

        for i, tp in enumerate(timepoints):
            points = tp["points"]
            center = np.mean(points, axis=0)
            centered = points - center

            if i == reference_idx:
                # Reference stays as is (just centered)
                aligned_points = centered + ref_center
                rotation_matrix = np.eye(3)
                basis = np.eye(3)
            else:
                # Find optimal rotation using PCA alignment
                rotation_matrix, basis = self._pca_align(centered, ref_centered)
                # Apply rotation and translate to reference center
                aligned_points = centered @ rotation_matrix.T + ref_center

            # Transform dense points if they exist
            aligned_dense_points = None
            if tp.get("dense_points") is not None:
                dense_points = tp["dense_points"]
                dense_center = np.mean(dense_points, axis=0)
                dense_centered = dense_points - dense_center
                if i == reference_idx:
                    aligned_dense_points = dense_centered + ref_center
                else:
                    aligned_dense_points = (
                        dense_centered @ rotation_matrix.T + ref_center
                    )

            # Transform leaf tip if it exists
            aligned_leaf_tip = None
            if tp["leaf_tip"] is not None:
                original_tip = tp["leaf_tip"]
                # Apply same transformation as points: center, rotate, translate
                centered_tip = original_tip - center
                if i == reference_idx:
                    aligned_leaf_tip = centered_tip + ref_center
                else:
                    aligned_leaf_tip = centered_tip @ rotation_matrix.T + ref_center

            # Preserve original timepoint structure
            aligned_tp = tp.copy()
            aligned_tp["points"] = aligned_points
            aligned_tp["dense_points"] = aligned_dense_points
            aligned_tp["leaf_tip"] = aligned_leaf_tip
            aligned_timepoints.append(aligned_tp)
            transformations.append(
                {
                    "day": tp["day"],
                    "rotation_matrix": rotation_matrix,
                    "basis": basis,
                    "original_center": center,
                    "reference_center": ref_center,
                    "is_reference": i == reference_idx,
                }
            )

            # # Plot pairwise alignment
            # if i == reference_idx:
            #     continue
            # plot_pairwise_alignment_with_quivers(
            #     aligned_timepoints, transformations, idx1=reference_idx, idx2=i
            # )

        return aligned_timepoints, transformations

    def save_transformations(
        self, leaf_timeseries, transformations, save_dir="transformations"
    ):
        """Save transformation matrices to JSON file for later use."""
        save_path = self.dataset_path / save_dir
        save_path.mkdir(parents=True, exist_ok=True)

        sequence_name = leaf_timeseries["sequence_name"]
        leaf_id = leaf_timeseries["leaf_id"]

        filename = f"{sequence_name}_leaf{leaf_id}_transformations.json"
        filepath = save_path / filename

        # Convert numpy arrays to lists for JSON serialization
        json_transformations = []
        for trans in transformations:
            json_trans = trans.copy()
            if isinstance(json_trans.get("rotation_matrix"), np.ndarray):
                json_trans["rotation_matrix"] = json_trans["rotation_matrix"].tolist()
            if isinstance(json_trans.get("original_center"), np.ndarray):
                json_trans["original_center"] = json_trans["original_center"].tolist()
            if isinstance(json_trans.get("reference_center"), np.ndarray):
                json_trans["reference_center"] = json_trans["reference_center"].tolist()
            json_transformations.append(json_trans)

        with open(filepath, "w") as f:
            json.dump(json_transformations, f, indent=2)

        print(f"Transformations saved to {filepath}")

    def __len__(self):
        return len(self.leaf_timeseries)

    def __getitem__(self, idx):
        return self.leaf_timeseries[idx]


if __name__ == "__main__":

    dataset_path = Path("data/TrackPlant3D")

    # # Example usage for PlantSequencesDataset
    # plant_dataset = PlantSequencesDataset(dataset_path)
    # print("Plant sequences dataset:")
    # print(f"Number of sequences: {len(plant_dataset)}")
    # print(
    #     "Available sequences (first 5):", plant_dataset.get_sequence_names()[:5]
    # )  # Show first 5
    # print("\n")

    # Example usage for LeafSequencesDataset
    print("Creating regular leaf dataset...")
    leaf_dataset = LeafSequencesDataset(
        dataset_path, min_timepoints=3, apply_pca_alignment=True
    )
    # print("Leaf timeseries dataset:")
    # print(f"Number of leaf timeseries: {len(leaf_dataset)}")

    # info = leaf_dataset.get_leaf_timeseries_info()
    # print("Dataset info:", info)
    # print("\n")

    # # Visualize one sequence
    # from plant_shape_analysis.vis.plot_functions import visualize_leaf_sequence

    # for i, sample in enumerate(leaf_dataset):
    #     if i <= 2:
    #         print(f"Visualizing leaf sequence {i}")
    #         visualize_leaf_sequence(sample)

    sample = leaf_dataset[0]
