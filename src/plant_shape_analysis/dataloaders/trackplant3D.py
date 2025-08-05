import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class PlantSequencesDataset(Dataset):
    def __init__(self, dataset_path):
        self.point_clouds_path = Path(dataset_path) / "gt_corrected_v1"
        self.leaf_tips_path = Path(dataset_path) / "keypoints" / "leaf_tips"

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
    def __init__(self, dataset_path, min_timepoints=3, max_timepoints=None):
        self.plant_dataset = PlantSequencesDataset(dataset_path)
        self.min_timepoints = min_timepoints
        self.max_timepoints = max_timepoints

        # Build leaf timeseries samples
        self.leaf_timeseries = self._build_leaf_timeseries()

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
                                "leaf_id": leaf_id,
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
            leaf_tip_idxs = timepoint_data["leaf_tip_idxs"]

            # Get unique leaf labels (excluding stem label 0)
            unique_leaves = np.unique(labels[labels > 0])

            for leaf_label in unique_leaves:
                # Extract points for this leaf
                leaf_mask = labels == leaf_label
                leaf_points = points[leaf_mask]

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
                        "leaf_tip": leaf_tip_coords,
                        "file_path": timepoint_data["file_path"],
                    }
                )

        # Sort each leaf track by day
        for leaf_id in leaf_tracks:
            leaf_tracks[leaf_id].sort(key=lambda x: x["day"])

        return dict(leaf_tracks)

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

    def __len__(self):
        return len(self.leaf_timeseries)

    def __getitem__(self, idx):
        return self.leaf_timeseries[idx]


if __name__ == "__main__":
    # Example usage for PlantSequencesDataset
    dataset_path = Path("data/TrackPlant3D")
    plant_dataset = PlantSequencesDataset(dataset_path)
    print("Plant sequences dataset:")
    print(f"Number of sequences: {len(plant_dataset)}")
    print(
        "Available sequences:", plant_dataset.get_sequence_names()[:5]
    )  # Show first 5
    print("\n")

    # Example usage for LeafTimeseriesDataset
    leaf_dataset = LeafSequencesDataset(dataset_path, min_timepoints=3)
    print("Leaf timeseries dataset:")
    print(f"Number of leaf timeseries: {len(leaf_dataset)}")

    info = leaf_dataset.get_leaf_timeseries_info()
    print("Dataset info:", info)
    print("\n")

    # Get first leaf timeseries sample
    if len(leaf_dataset) > 0:
        sample = leaf_dataset[100]
        print("Sample leaf timeseries:")
        print(f"Sequence: {sample['sequence_name']}")
        print(f"Leaf ID: {sample['leaf_id']}")
        print(f"Number of timepoints: {len(sample['timepoints'])}")
        print(f"Days: {[tp['day'] for tp in sample['timepoints']]}")
        print(f"Point counts: {[len(tp['points']) for tp in sample['timepoints']]}")
        print(
            f"Has leaf tips: {[tp['leaf_tip'] is not None for tp in sample['timepoints']]}"
        )

        # Example visualization
        from plant_shape_analysis.vis.plot_functions import visualize_leaf_sequence

        visualize_leaf_sequence(sample)
