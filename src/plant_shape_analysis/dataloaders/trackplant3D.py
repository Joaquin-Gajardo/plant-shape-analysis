import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class PlantSequencesDataset(Dataset):
    def __init__(self, dataset_path):
        self.point_clouds_path = Path(dataset_path) / "gt"
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
        else:
            print(f"Leaf tips file not found: {file_path}")

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
                assert np.isclose(points[leaf_tip_idxs], leaf_tip_coordinates).all()

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


if __name__ == "__main__":

    # Example usage
    dataset_path = Path("data/TrackPlant3D")
    dataset = PlantSequencesDataset(dataset_path)
    print(dataset[0])  # Get first sequence data
    print("\n")

    # Get all sequence names
    print("Available sequences:", dataset.get_sequence_names())
    print("\n")

    # Get data for a specific sequence
    sequence_name = "tomato2_control_plant1"
    sequence_data = dataset.get_sequence_data(sequence_name)
    print(f"Data for {sequence_name}:", sequence_data)
