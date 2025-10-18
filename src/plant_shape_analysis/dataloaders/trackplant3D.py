import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset


class PlantSequencesDataset(Dataset):
    DATASET_CONFIGS = {
        "v1": {
            "data_dirs": {"sparse": "gt_corrected_v1", "dense": "dense"},
            "download_info": {
                "url": "https://polybox.ethz.ch/index.php/s/mxiZwKfCfd39Rxx/download",
                "filename": "v1.zip",
                "extract_dir": "v1",
                "size_mb": 751,
                "description": "TrackPlant3D v1 dataset with leaf keypoint annotations and dense point clouds",
            },
        }
    }

    def __init__(
        self,
        dataset_path,
        version="v1",
        use_ply=False,
        alignment_method=None,
        save_transformations=False,
        estimate_normals=False,
        auto_download=True,
    ):
        """
        Initialize PlantSequencesDataset.

        Args:
            dataset_path: Path to TrackPlant3D dataset
            version: Dataset version (default: "v1")
            use_ply: If True, load from PLY files instead of TXT files. Keeping both options for compatibility to original dataset format.
            alignment_method: Alignment method - None (no alignment), 'pca' (fast, approximate), 'icp' (slower, more accurate), or 'stem_based' (uses only stem points with sequential alignment and vertical correction)
            save_transformations: If True, save transformation matrices when applying alignment
            estimate_normals: If True, estimate normals for all plant point clouds
            auto_download: If True, automatically download dataset if not found (default: True)
        """
        # Validate version and get config
        if version not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown version: {version}. Available: {list(self.DATASET_CONFIGS.keys())}"
            )

        config = self.DATASET_CONFIGS[version]
        data_dirs = config["data_dirs"]
        extract_dir = config["download_info"]["extract_dir"]

        # Always expect dataset at dataset_path/extract_dir
        self.dataset_path = Path(dataset_path) / extract_dir
        self.version = version

        # Auto-download if version directory doesn't exist
        if not self.dataset_path.exists():
            if auto_download:
                from plant_shape_analysis.utils.download_dataset import (
                    download_trackplant3d,
                )

                print(f"Dataset not found at {self.dataset_path}")
                print("Attempting to download...")
                self.dataset_path = download_trackplant3d(
                    target_dir=dataset_path, version=version, verbose=True
                )
            else:
                raise FileNotFoundError(
                    f'Dataset not found in "{str(self.dataset_path.resolve())}" and auto-download is disabled.'
                )

        self.use_ply = use_ply
        self.file_extension = "*.ply" if use_ply else "*.txt"
        self.alignment_method = alignment_method
        self._save_transformations = save_transformations
        self.estimate_normals = estimate_normals

        # Set paths from config
        self.sparse_path = self.dataset_path / data_dirs["sparse"]
        self.dense_path = self.dataset_path / data_dirs["dense"]

        # This is only used when using txt files, as PLY files have leaf tips as a scalar field (sparse ones)
        self.leaf_tips_path = self.dataset_path / "keypoints" / "leaf_tips"

        # Organize files into sequences
        self.sequences = self._organize_sequences()
        if len(self.sequences) == 0:
            raise ValueError(
                "No sequences found in the dataset. Verify dataset path and file extension (use `use_ply=True` if loading PLY files)."
            )

        # Build plant timeseries
        self.plant_timeseries = self._build_plant_timeseries()

        # Initialize transformations dictionary (populated if alignment is applied)
        self.transformations = {}

        # Apply alignment if requested
        if self.alignment_method is not None:
            print(
                f"Applying {self.alignment_method.upper()} alignment to {len(self.plant_timeseries)} plant sequences..."
            )
            self._align_dataset()

        # Estimate normals if requested
        if self.estimate_normals:
            self._estimate_normals()

    def _organize_sequences(self):
        """Organize files into sequences based on crop type, treatment, and plant number"""
        sequences = defaultdict(list)

        # Get all crop folders
        crop_folders = [d for d in self.sparse_path.iterdir() if d.is_dir()]

        for crop_folder in crop_folders:
            crop_name = crop_folder.name

            # Get all files with the appropriate extension
            files = list(crop_folder.glob(self.file_extension))

            # Group files by full sequence identifier (crop_treatment_plant)
            sequence_groups = defaultdict(list)

            for file in files:
                # Extract sequence identifier from filename
                # Example: "1_maize_control_plant1_D00.txt" -> "maize_control_plant1"
                # Pattern: number_crop_treatment_plantX_DXX.(txt|ply)
                pattern = r"\d+_(.+)_D\d+\.(?:txt|ply)$"
                match = re.search(pattern, file.name)
                if match:
                    # Extract everything between the initial number and the day
                    sequence_part = match.group(1)  # e.g., "maize_control_plant1"
                    sequence_groups[sequence_part].append(file)
                else:
                    # Fallback: try to extract manually
                    # Remove the leading number and trailing day part
                    name_parts = file.stem.split("_")
                    if len(name_parts) >= 4:  # At least: number, crop, treatment, plant
                        sequence_part = "_".join(
                            name_parts[1:-1]
                        )  # Skip first number and last day
                        sequence_groups[sequence_part].append(file)

            # Sort files within each sequence group by day
            for sequence_id, files in sequence_groups.items():
                # Sort by day number (D00, D01, etc.)
                files.sort(key=lambda x: int(re.search(r"D(\d+)", x.name).group(1)))
                sequences[sequence_id] = files

        return dict(sequences)

    def _build_plant_timeseries(self):
        """Build plant timeseries from organized sequences."""
        plant_timeseries = []

        for sequence_name, files in self.sequences.items():
            sequence_data = self.get_sequence_data(sequence_name)
            plant_timeseries.append(
                {
                    "sequence_name": sequence_name,
                    "timepoints": sequence_data,
                    "is_aligned": False,
                    "transformations": [],
                }
            )

        return plant_timeseries

    def get_sequence_names(self):
        """Get all sequence names"""
        return [ts["sequence_name"] for ts in self.plant_timeseries]

    def get_sequences_by_crop(self, crop_name):
        """Get all sequences for a specific crop"""
        return [
            ts
            for ts in self.plant_timeseries
            if ts["sequence_name"].startswith(crop_name)
        ]

    def get_sequences_by_treatment(self, treatment):
        """Get all sequences for a specific treatment"""
        return [ts for ts in self.plant_timeseries if treatment in ts["sequence_name"]]

    def get_timeseries_by_sequence_name(self, sequence_name):
        """Get plant timeseries by sequence name (e.g., 'maize_control_plant2')"""
        for ts in self.plant_timeseries:
            if ts["sequence_name"] == sequence_name:
                return ts
        return None

    def _get_sequence(self, sequence_name):
        """Get all files for a specific sequence"""
        return self.sequences.get(sequence_name, [])

    def load_point_cloud(self, file_path):
        """Load point cloud from txt or ply file"""
        if self.use_ply:
            # Load from PLY file
            pcd = o3d.t.io.read_point_cloud(str(file_path))
            points = pcd.point.positions.numpy()
            labels = pcd.point.organ_label.numpy().flatten().astype(int)
            return points, labels
        else:
            # Load from TXT file
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

        if self.use_ply:
            # For PLY: read is_leaf_tip scalar field from main point cloud
            pcd = o3d.t.io.read_point_cloud(str(point_cloud_path))
            points = pcd.point.positions.numpy()
            is_leaf_tip = pcd.point.is_leaf_tip.numpy().flatten().astype(bool)

            # Get indices where is_leaf_tip is True
            global_indices = np.where(is_leaf_tip)[0]
            coordinates = points[global_indices]

            return global_indices, coordinates
        else:
            # For TXT: load from comma-delimited txt file
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

    def _needs_orientation_correction(self, sequence_name):
        """Check if sequence needs Y->Z orientation correction"""
        # Sorghum, tobacco, and tomato1 have Y as up instead of Z
        crops_to_correct = ["sorghum", "tobacco", "tomato1"]
        return any(sequence_name.startswith(crop) for crop in crops_to_correct)

    def _correct_orientation(self, points, normals=None):
        """Apply 90° rotation around X axis to make Z up instead of Y"""
        # Rotation matrix for 90° around X: Y -> Z, Z -> -Y, X -> X
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

        rotated_points = points @ R.T
        rotated_normals = None
        if normals is not None:
            rotated_normals = normals @ R.T

        return rotated_points, rotated_normals

    def get_sequence_data(self, sequence_name):
        """Get all point clouds and leaf tips for a sequence"""
        files = self._get_sequence(sequence_name)
        sequence_data = []
        needs_correction = self._needs_orientation_correction(sequence_name)

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

            # Apply orientation correction if needed (before validation)
            if needs_correction:
                points, _ = self._correct_orientation(points)
                if dense_points is not None:
                    dense_points, _ = self._correct_orientation(dense_points)
                if leaf_tip_coordinates.size > 0:
                    leaf_tip_coordinates, _ = self._correct_orientation(
                        leaf_tip_coordinates
                    )

            # Validate leaf tips after orientation correction
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
            R1[0] = -R1[0]
            R = R2.T @ R1

        return R, R1

    def align_plant_sequence(self, sequence_name, reference_idx=0, method=None):
        """
        Align plant sequence using PCA, ICP, or stem-based registration.
        Preserves scale differences to show growth over time.
        Only removes rotation and translation.
        Works with different numbers of points between timepoints.
        Ensures consistent orientation across all timepoints.

        Args:
            sequence_name: Name of the sequence to align
            reference_idx: Index of reference timepoint (default: 0, ignored for stem_based)
            method: Alignment method - 'pca', 'icp', or 'stem_based'. If None, uses self.alignment_method

        Returns:
            List of aligned timepoint data and transformation matrices
        """
        method = method or self.alignment_method

        # Stem-based alignment uses sequential approach
        if method == "stem_based":
            return self._align_stem_based(sequence_name)

        sequence_data = self.get_sequence_data(sequence_name)
        if len(sequence_data) < 2:
            return sequence_data, []

        # Get reference timepoint
        ref_points = sequence_data[reference_idx]["points"]
        ref_center = np.mean(ref_points, axis=0)
        ref_centered = ref_points - ref_center

        aligned_sequence = []
        transformations = []

        for i, timepoint_data in enumerate(sequence_data):
            points = timepoint_data["points"]
            center = np.mean(points, axis=0)
            centered = points - center

            if i == reference_idx:
                # Reference stays as is (just centered)
                aligned_points = centered + ref_center
                rotation_matrix = np.eye(3)
                basis = np.eye(3)
                translation = np.zeros(3)
            else:
                # Find optimal rotation using selected method
                if method == "icp":
                    # Lazy load to avoid sklearn dependency if not using ICP
                    try:
                        from plant_shape_analysis.alignment.icp_alignment import (
                            align_plant_pair_icp,
                        )
                    except ImportError:
                        raise ImportError(
                            "ICP alignment requires 'scikit-learn'. Install it with `pip install scikit-learn`."
                        )

                    aligned_centered, rotation_matrix, translation = (
                        align_plant_pair_icp(
                            centered,
                            ref_centered,
                            max_iterations=500,
                            convergence_threshold=0.01,
                        )
                    )
                    # ICP already returns aligned points, just translate to reference center
                    aligned_points = aligned_centered + ref_center
                    basis = rotation_matrix  # For ICP, basis is same as rotation
                else:  # pca
                    rotation_matrix, basis = self._pca_align(centered, ref_centered)
                    aligned_points = centered @ rotation_matrix.T + ref_center
                    translation = np.zeros(3)

            # Transform dense points if they exist
            aligned_dense_points = None
            if timepoint_data["dense_points"] is not None:
                dense_points = timepoint_data["dense_points"]
                dense_centered = dense_points - center
                if i == reference_idx:
                    aligned_dense_points = dense_centered + ref_center
                else:
                    if method == "icp":
                        # Apply same ICP transformation
                        aligned_dense_points = (
                            dense_centered @ rotation_matrix.T
                            + translation
                            + ref_center
                        )
                    else:
                        aligned_dense_points = (
                            dense_centered @ rotation_matrix.T + ref_center
                        )

            # Transform leaf tips if they exist
            aligned_leaf_tip_idxs = timepoint_data["leaf_tip_idxs"]
            aligned_leaf_tip_coords = None
            if timepoint_data["leaf_tip_idxs"].size > 0:
                # Get original leaf tip coordinates
                original_tip_coords = points[timepoint_data["leaf_tip_idxs"]]
                # Apply same transformation
                centered_tips = original_tip_coords - center
                if i == reference_idx:
                    aligned_leaf_tip_coords = centered_tips + ref_center
                else:
                    if method == "icp":
                        aligned_leaf_tip_coords = (
                            centered_tips @ rotation_matrix.T + translation + ref_center
                        )
                    else:
                        aligned_leaf_tip_coords = (
                            centered_tips @ rotation_matrix.T + ref_center
                        )

            # Preserve original timepoint structure
            aligned_tp = timepoint_data.copy()
            aligned_tp["points"] = aligned_points
            aligned_tp["dense_points"] = aligned_dense_points
            aligned_tp["dense_labels"] = timepoint_data["dense_labels"]
            # Note: leaf_tip_idxs remain the same (indices into aligned points)
            aligned_sequence.append(aligned_tp)

            transformations.append(
                {
                    "day": timepoint_data["day"],
                    "rotation_matrix": rotation_matrix,
                    "basis": basis,
                    "translation": translation,
                    "original_center": center,
                    "reference_center": ref_center,
                    "is_reference": i == reference_idx,
                    "method": method,
                }
            )

        return aligned_sequence, transformations

    def _align_stem_based(self, sequence_name):
        """
        Align plant sequence using stem-based ICP with vertical correction.

        This is a two-stage sequential alignment approach:
        - Stage 1: Stem-based ICP alignment (rotation + translation)
        - Stage 2: Vertical shift correction using stem base centroid

        Args:
            sequence_name: Name of the sequence to align

        Returns:
            List of aligned timepoint data and transformation matrices
        """
        from plant_shape_analysis.alignment.stem_based_alignment import (
            align_plant_sequence_stem_based,
        )

        sequence_data = self.get_sequence_data(sequence_name)
        if len(sequence_data) < 2:
            return sequence_data, []

        # Run stem-based alignment
        aligned_timepoints, transformations = align_plant_sequence_stem_based(
            sequence_data,
            use_rotation=True,
            max_iterations=200,
            vertical_correction=True,
            base_height_mm=10,
        )

        # Transform dense points and leaf tips using the same transformations
        aligned_sequence = []
        for i, (aligned_tp, timepoint_data, trans_info) in enumerate(
            zip(aligned_timepoints, sequence_data, transformations)
        ):
            # Get transformation parameters
            rotation = trans_info["rotation"]
            translation = trans_info["translation"]
            vertical_shift = trans_info.get("vertical_shift", np.zeros(3))

            # Start with aligned points from stem-based alignment
            result_tp = timepoint_data.copy()
            result_tp["points"] = aligned_tp["points"]

            # Transform dense points if they exist
            if timepoint_data["dense_points"] is not None:
                dense_points = timepoint_data["dense_points"]

                if i == 0:
                    # Reference frame - no transformation
                    aligned_dense_points = dense_points
                else:
                    # Apply same transformation as sparse points
                    if trans_info.get("stage") == "stage2_vertical_correction":
                        # Apply rotation, translation, and vertical shift
                        aligned_dense_points = (
                            dense_points @ rotation.T + translation + vertical_shift
                        )
                    else:
                        # Apply rotation and translation only
                        aligned_dense_points = dense_points @ rotation.T + translation

                result_tp["dense_points"] = aligned_dense_points

            # Note: leaf_tip_idxs remain the same (indices into aligned points)
            # The aligned coordinates are automatically correct since points are aligned

            aligned_sequence.append(result_tp)

        # Format transformations to match other alignment methods
        formatted_transformations = []
        for trans_info in transformations:
            formatted_transformations.append(
                {
                    "day": trans_info["day"],
                    "rotation_matrix": trans_info["rotation"],
                    "basis": trans_info["rotation"],
                    "translation": trans_info["translation"],
                    "vertical_shift": trans_info.get("vertical_shift", np.zeros(3)),
                    "original_center": np.zeros(3),  # Not used in stem-based
                    "reference_center": np.zeros(3),  # Not used in stem-based
                    "is_reference": trans_info["is_reference"],
                    "method": "stem_based",
                    "stage": trans_info["stage"],
                }
            )

        return aligned_sequence, formatted_transformations

    def _align_dataset(self):
        """Apply alignment (PCA, ICP, or stem-based) to all plant sequences in the dataset."""
        aligned_timeseries = []

        for plant_ts in self.plant_timeseries:
            sequence_name = plant_ts["sequence_name"]
            timepoints = plant_ts["timepoints"]

            if len(timepoints) >= 2:
                # Apply alignment with first timepoint as reference
                aligned_data, transformations = self.align_plant_sequence(
                    sequence_name, reference_idx=0, method=self.alignment_method
                )

                # Update the timeseries with aligned data
                aligned_plant_ts = plant_ts.copy()
                aligned_plant_ts["timepoints"] = aligned_data
                aligned_plant_ts["is_aligned"] = True
                aligned_plant_ts["transformations"] = transformations

                aligned_timeseries.append(aligned_plant_ts)

                # Store transformations in dataset-level dictionary for easy access
                self.transformations[sequence_name] = transformations

                # Save transformations to file if requested
                if self._save_transformations:
                    self.save_transformations(sequence_name, transformations)
            else:
                # Keep original if insufficient timepoints for alignment
                plant_ts["is_aligned"] = False
                aligned_timeseries.append(plant_ts)

        # Replace with aligned timeseries
        self.plant_timeseries = aligned_timeseries
        print(
            f"{self.alignment_method.upper()} alignment complete. {sum(1 for ts in self.plant_timeseries if ts['is_aligned'])} sequences aligned."
        )

    def _estimate_normals(self):
        """Estimate normals for all plants in the dataset."""
        # NOTE: these are only estimated for sparse points, as dense points is takes a long time
        print("Estimating normals for all sparse plants in the dataset...")
        for plant_ts in self.plant_timeseries:
            for timepoint in plant_ts["timepoints"]:
                # Estimate normals for sparse points
                points = timepoint["points"]
                if points is not None and len(points) > 0:
                    # Estimate normals using Open3D
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.estimate_normals()
                    pcd.orient_normals_to_align_with_direction()
                    pcd.orient_normals_consistent_tangent_plane(k=30)
                    normals = np.asarray(pcd.normals)
                    timepoint["normals"] = normals

            # Ensure temporal consistency of normals across timepoints
            # self._enforce_temporal_normal_consistency_plant(plant_ts["timepoints"])
        print("Normal estimation complete.")

    def _enforce_temporal_normal_consistency_plant(self, timepoints, reference_idx=0):
        """
        Enforce temporal consistency of normals across timepoints for a plant sequence.
        Uses sequential propagation: compares each timepoint to the previous one.
        Processes each leaf (organ label) separately to ensure normals remain consistent
        on the same face of each leaf over time.

        Args:
            timepoints: List of timepoint dictionaries containing 'points', 'labels', and 'normals'
            reference_idx: Index of reference timepoint (default: 0, not used in sequential mode)
        """
        if len(timepoints) < 2:
            return

        # Check if first timepoint has normals
        if "normals" not in timepoints[0] or timepoints[0]["normals"] is None:
            return

        # Sequential propagation: compare each timepoint to the previous one
        for i in range(1, len(timepoints)):
            prev_tp = timepoints[i - 1]  # Previous timepoint as reference
            curr_tp = timepoints[i]

            if "normals" not in curr_tp or curr_tp["normals"] is None:
                continue

            prev_points = prev_tp["points"]
            prev_labels = prev_tp["labels"]
            prev_normals = prev_tp["normals"]

            curr_points = curr_tp["points"]
            curr_labels = curr_tp["labels"]
            curr_normals = curr_tp["normals"]

            # Get unique organ labels from current timepoint
            unique_labels = np.unique(curr_labels)

            # Process each organ separately
            for organ_label in unique_labels:
                # Skip if this organ doesn't exist in current timepoint
                curr_organ_mask = curr_labels == organ_label
                if not np.any(curr_organ_mask):
                    continue

                # Skip if organ doesn't exist in previous timepoint
                prev_organ_mask = prev_labels == organ_label
                if not np.any(prev_organ_mask):
                    continue

                # Get points and normals for this organ
                curr_organ_points = curr_points[curr_organ_mask]
                curr_organ_normals = curr_normals[curr_organ_mask]
                prev_organ_points = prev_points[prev_organ_mask]
                prev_organ_normals = prev_normals[prev_organ_mask]

                # Build KD-tree for previous organ points
                from scipy.spatial import cKDTree

                prev_tree = cKDTree(prev_organ_points)

                # Find nearest neighbors in previous frame
                distances, indices = prev_tree.query(curr_organ_points, k=1)

                # For each point, check if normal should be flipped
                matched_prev_normals = prev_organ_normals[indices]

                # Compute dot product between current and previous normals
                dot_products = np.sum(curr_organ_normals * matched_prev_normals, axis=1)

                # Use majority voting: if most normals point in wrong direction, flip ALL
                # Only consider points with close matches (within reasonable distance)
                max_distance = np.percentile(
                    distances, 75
                )  # Use 75th percentile as threshold
                reliable_matches = distances < max_distance

                if np.any(reliable_matches):
                    avg_dot_product = np.mean(dot_products[reliable_matches])

                    # If average dot product is negative, flip ALL normals for this organ
                    if avg_dot_product < 0:
                        curr_organ_normals = -curr_organ_normals

                # Update normals in the full array
                curr_normals[curr_organ_mask] = curr_organ_normals

            # Update normals in timepoint
            curr_tp["normals"] = curr_normals

    def save_transformations(
        self, sequence_name, transformations, save_dir="transformations"
    ):
        """
        Save transformation matrices to JSON file for later use.
        Includes both original transformation info and 4x4 homogeneous matrices.

        Args:
            sequence_name: Name of the sequence
            transformations: List of transformation dictionaries
            save_dir: Directory to save transformations
        """
        save_path = self.dataset_path / save_dir
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{sequence_name}_transformations.json"
        filepath = save_path / filename

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
            # For plant alignment: p' = (p - original_center) @ R.T + reference_center + translation
            R = trans.get("rotation_matrix", np.eye(3))
            t = trans.get("translation", np.zeros(3))
            orig_c = trans.get("original_center", np.zeros(3))
            ref_c = trans.get("reference_center", np.zeros(3))
            v_shift = trans.get("vertical_shift", np.zeros(3))

            # Build transformation matrix
            # T = T(ref_c + t + v_shift) @ R @ T(-orig_c)
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

        print(f"Transformations saved to {filepath}")

    def __len__(self):
        return len(self.plant_timeseries)

    def __getitem__(self, idx):
        return self.plant_timeseries[idx]


class LeafSequencesDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        version: str = "v1",
        min_timepoints: int = 3,
        max_timepoints: Optional[int] = None,
        apply_alignment: bool = False,
        plant_alignment_method: Optional[str] = None,
        estimate_plant_normals: bool = False,
        estimate_normals: bool = False,
        use_ply: bool = False,
        save_transformations: bool = False,
        auto_download: bool = True,
    ):
        """
        Initialize LeafSequencesDataset.

        Args:
            dataset_path: Path to TrackPlant3D dataset
            version: Dataset version (default: "v1")
            min_timepoints: Minimum number of timepoints for a leaf sequence
            max_timepoints: Maximum number of timepoints (None = no limit)
            apply_alignment: If True, align leaf sequences individually (multi-state approach)
            plant_alignment_method: Alignment method for plants when tracking leaves
            estimate_plant_normals: If True, estimate normals for plants before tracking leaves. Normals will be included in leaf data.
            estimate_normals: If True, estimate normals for all leaves. Ignored if plant normals are estimated.
            use_ply: If True, load from PLY files instead of TXT files
            save_transformations: If True, save transformation matrices when applying alignment
            auto_download: If True, automatically download dataset if not found (default: True)
        """
        self.plant_dataset = PlantSequencesDataset(
            dataset_path,
            version=version,
            use_ply=use_ply,
            alignment_method=plant_alignment_method,
            estimate_normals=estimate_plant_normals,
            auto_download=auto_download,
        )
        self.dataset_path = Path(dataset_path)
        self.min_timepoints = min_timepoints
        self.max_timepoints = max_timepoints
        self.apply_alignment = apply_alignment
        self.plant_alignment_method = plant_alignment_method
        self._save_transformations = save_transformations
        self.estimate_normals = estimate_normals

        # Build leaf timeseries samples
        self.leaf_timeseries = self._build_leaf_timeseries()

        # Initialize transformations dictionary (populated if alignment is applied)
        self.transformations = {}

        if len(self.leaf_timeseries) == 0:
            raise ValueError(
                "No leaf timeseries found with the given parameters. "
                "Verify dataset path and file extension (use `use_ply=True` if loading PLY files)."
            )

        # Estimate normals for leaves if requested (before alignment because alignment uses normals)
        if estimate_normals and not estimate_plant_normals:
            self._estimate_normals()

        # Apply alignment if requested
        if self.apply_alignment:
            print(
                f"Apply multi-stage PCA-based alignment to {len(self.leaf_timeseries)} leaf sequences..."
            )
            self._align_dataset()

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

    def get_timeseries_by_sequence_name(self, sequence_name):
        """Get leaf timeseries by unique sequence name (e.g., 'tomato2_control_plant2_leaf1')"""
        for ts in self.leaf_timeseries:
            if ts["sequence_name"] == sequence_name:
                return ts
        return None

    def get_timeseries_by_plant_sequence(self, plant_sequence_name):
        """Get all leaf timeseries for a plant sequence (e.g., 'tomato2_control_plant2')"""
        return [
            ts
            for ts in self.leaf_timeseries
            if ts["plant_sequence_name"] == plant_sequence_name
        ]

    def _build_leaf_timeseries(self):
        """Build individual leaf timeseries from plant sequences"""
        leaf_timeseries = []

        # Iterate through plant_timeseries directly instead of reloading from files
        # This ensures we use the data with normals if they were estimated
        for plant_ts in self.plant_dataset.plant_timeseries:
            sequence_name = plant_ts["sequence_name"]
            sequence_data = plant_ts[
                "timepoints"
            ]  # Use already-loaded data with normals!

            # Track leaves across time points
            leaf_tracks = self._track_leaves_across_time(sequence_data)

            # Filter by minimum timepoints requirement
            for leaf_id, timepoints in leaf_tracks.items():
                if len(timepoints) >= self.min_timepoints:
                    if (
                        self.max_timepoints is None
                        or len(timepoints) <= self.max_timepoints
                    ):
                        # Create unique sequence name with leaf ID
                        unique_sequence_name = f"{sequence_name}_leaf{leaf_id}"
                        leaf_timeseries.append(
                            {
                                "sequence_name": unique_sequence_name,
                                "plant_sequence_name": sequence_name,
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
            normals = timepoint_data.get(
                "normals", None
            )  # Extract normals if available

            # Get unique leaf labels (excluding stem label 0)
            unique_leaves = np.unique(labels[labels > 0])

            for leaf_label in unique_leaves:
                # Extract points for this leaf
                leaf_mask = labels == leaf_label
                leaf_points = points[leaf_mask]

                # Extract normals for this leaf if available
                leaf_normals = None
                if normals is not None:
                    leaf_normals = normals[leaf_mask]

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
                        "leaf_id": int(leaf_label),
                        "points": leaf_points,
                        "normals": leaf_normals,  # Include normals
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
                if points is not None and len(points) > 0:
                    # Estimate normals using Open3D
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.estimate_normals()
                    pcd.orient_normals_to_align_with_direction()
                    pcd.orient_normals_consistent_tangent_plane(k=30)
                    normals = np.asarray(pcd.normals)
                    timepoint["normals"] = normals

            # Ensure temporal consistency of normals across timepoints
            # self._enforce_temporal_normal_consistency(leaf["timepoints"])
        print("Normal estimation complete.")

    def _prealign_with_leaf_tips(self, timepoints):
        """
        Pre-align rotation around Y-axis using leaf tip directions.
        This is done on UNALIGNED data to roughly orient leaves correctly
        before normal consistency and PCA alignment.

        Uses sequential propagation: aligns each timepoint to the previous one
        based on the direction from centroid to leaf tip projected onto XZ plane.

        Args:
            timepoints: List of timepoint dictionaries (modified in-place)

        Returns:
            List of transformation dictionaries for each timepoint
        """
        if len(timepoints) < 2:
            return [
                {
                    "stage": "prealign_tip",
                    "day": timepoints[0]["day"],
                    "rotation": np.eye(3),
                    "translation": np.zeros(3),
                    "center": np.zeros(3),
                }
            ]

        transformations = []

        # First timepoint is reference
        transformations.append(
            {
                "stage": "prealign_tip",
                "day": timepoints[0]["day"],
                "rotation": np.eye(3),
                "translation": np.zeros(3),
                "center": np.mean(timepoints[0]["points"], axis=0),
            }
        )

        # Sequential propagation: align each timepoint to previous one
        for i in range(1, len(timepoints)):
            prev_tp = timepoints[i - 1]
            curr_tp = timepoints[i]

            curr_center = np.mean(curr_tp["points"], axis=0)

            # Check if both have leaf tips
            if prev_tp.get("leaf_tip") is None or curr_tp.get("leaf_tip") is None:
                # No transformation
                transformations.append(
                    {
                        "stage": "prealign_tip",
                        "day": curr_tp["day"],
                        "rotation": np.eye(3),
                        "translation": np.zeros(3),
                        "center": curr_center,
                    }
                )
                continue

            # Get centroids
            prev_center = np.mean(prev_tp["points"], axis=0)

            # Compute direction from centroid to tip
            prev_direction = prev_tp["leaf_tip"] - prev_center
            curr_direction = curr_tp["leaf_tip"] - curr_center

            # Project onto XZ plane (we rotate around Y-axis)
            prev_xz = np.array([prev_direction[0], prev_direction[2]])
            curr_xz = np.array([curr_direction[0], curr_direction[2]])

            # Compute rotation angle in XZ plane
            prev_angle = np.arctan2(prev_xz[1], prev_xz[0])  # Z, X
            curr_angle = np.arctan2(curr_xz[1], curr_xz[0])
            rotation_angle = prev_angle - curr_angle

            # Create Y-rotation matrix
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)
            rot_y = np.array(
                [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]],
                dtype=np.float64,
            )

            # Apply rotation to current timepoint (centered around its own centroid)
            # Rotate points
            centered_points = curr_tp["points"] - curr_center
            curr_tp["points"] = centered_points @ rot_y.T + curr_center

            # Rotate normals if they exist (direction vectors, no translation)
            if curr_tp.get("normals") is not None:
                curr_tp["normals"] = curr_tp["normals"] @ rot_y.T

            # Rotate dense points if they exist
            if curr_tp.get("dense_points") is not None:
                centered_dense = curr_tp["dense_points"] - curr_center
                curr_tp["dense_points"] = centered_dense @ rot_y.T + curr_center

            # Rotate leaf tip
            centered_tip = curr_tp["leaf_tip"] - curr_center
            curr_tp["leaf_tip"] = centered_tip @ rot_y.T + curr_center

            # Track transformation
            transformations.append(
                {
                    "stage": "prealign_tip",
                    "day": curr_tp["day"],
                    "rotation": rot_y,
                    "translation": np.zeros(3),
                    "center": curr_center,
                }
            )

        return transformations

    def _enforce_temporal_normal_consistency(self, timepoints):
        """
        Enforce temporal consistency of normals across timepoints for a leaf sequence.
        Uses sequential propagation: compares each timepoint to the previous one.
        This ensures normals point in the same direction over time.

        Args:
            timepoints: List of timepoint dictionaries containing 'points' and 'normals'
        """
        if len(timepoints) < 2:
            return

        # Check if first timepoint has normals
        if "normals" not in timepoints[0] or timepoints[0]["normals"] is None:
            return

        # Sequential propagation: compare each timepoint to the previous one
        from scipy.spatial import cKDTree

        for i in range(1, len(timepoints)):
            prev_tp = timepoints[i - 1]  # Previous timepoint as reference
            curr_tp = timepoints[i]

            if "normals" not in curr_tp or curr_tp["normals"] is None:
                continue

            prev_points = prev_tp["points"]
            prev_normals = prev_tp["normals"]

            curr_points = curr_tp["points"]
            curr_normals = curr_tp["normals"]

            # Build KD-tree for previous points
            prev_tree = cKDTree(prev_points)

            # Find nearest neighbors in previous frame
            distances, indices = prev_tree.query(curr_points, k=1)

            # For each point, check if normal should be flipped
            matched_prev_normals = prev_normals[indices]

            # Compute dot product between current and previous normals
            dot_products = np.sum(curr_normals * matched_prev_normals, axis=1)

            # Use majority voting: if most normals point in wrong direction, flip ALL
            # Only consider points with close matches (within reasonable distance)
            max_distance = np.percentile(
                distances, 75
            )  # Use 75th percentile as threshold
            reliable_matches = distances < max_distance

            if np.any(reliable_matches):
                avg_dot_product = np.mean(dot_products[reliable_matches])

                # If average dot product is negative, flip ALL normals
                if avg_dot_product < 0:
                    curr_normals = -curr_normals

            # Update normals in timepoint
            curr_tp["normals"] = curr_normals
        return timepoints

    def _align_dataset(self):
        """
        Apply PCA-based alignment to all leaf sequences in the dataset.

        Multi-stage approach:
        0. Pre-align rotation using leaf tips (on original unaligned data)
           - Uses leaf tip direction to roughly align Y-rotation before other processing
           - Critical for decaying/problematic leaves where normals might not be reliable
        1. Fix temporal normal consistency (on pre-aligned data)
           - Uses sequential propagation with nearest neighbor matching
           - Ensures normals point to same face consistently across time
        2. Apply PCA alignment with orientation corrections (using corrected normals)
           - Sequential alignment: each timepoint aligned to previous one
           - Z-axis rotation: computes optimal angle based on normal directions
        3. Align main PCA axis to Z-axis
           - Rotates each leaf so its main axis is parallel to Z-axis
           - Makes all leaves have the same inclination for easy comparison
        4. Vertical alignment to z=0 plane
           - Shifts each timepoint independently so its lowest point is at z=0
        """

        aligned_timeseries = []
        for i, leaf_ts in enumerate(self.leaf_timeseries):
            sequence_name = leaf_ts["sequence_name"]

            if len(leaf_ts["timepoints"]) >= 2:
                all_transformations = []

                # Stage 0: Pre-align rotation using leaf tips BEFORE everything else
                # This roughly aligns the leaves so the rest of the pipeline works better
                # In edge cases such as wilting or decaying leaves
                if leaf_ts["timepoints"][0].get("leaf_tip") is not None:
                    stage0_trans = self._prealign_with_leaf_tips(leaf_ts["timepoints"])
                    all_transformations.append(("prealign_tip", stage0_trans))
                else:
                    all_transformations.append(("prealign_tip", None))

                # Stage 1: Enforce temporal normal consistency
                # This ensures all normals point to the same face (inner/outer) consistently
                if leaf_ts["timepoints"][0].get("normals") is not None:
                    self._enforce_temporal_normal_consistency(leaf_ts["timepoints"])
                # Note: Normal consistency doesn't change geometry, only flips normal vectors

                # Stage 2: Apply PCA alignment with normal-aware Z-rotation
                # Now the alignment can use the corrected normals to determine proper orientation
                aligned_timepoints, stage2_trans = self.align_leaf_sequence(leaf_ts)
                all_transformations.append(("pca_align", stage2_trans))

                # Stage 3: Align main PCA axis to Z-axis (make all leaves parallel)
                # This aligns the leaf's main direction with the vertical axis
                stage3_trans = self._align_pca_to_z_axis(aligned_timepoints)
                all_transformations.append(("align_to_z", stage3_trans))

                # Stage 4: Vertical alignment - shift each timepoint so its lowest point is at z=0
                stage4_trans = self._align_leaves_to_xy_plane(aligned_timepoints)
                all_transformations.append(("vertical_align", stage4_trans))

                # Update the leaf timeseries with aligned data
                aligned_leaf_ts = leaf_ts.copy()
                aligned_leaf_ts["timepoints"] = aligned_timepoints
                aligned_leaf_ts["is_aligned"] = True
                aligned_leaf_ts["transformation_stages"] = all_transformations

                aligned_timeseries.append(aligned_leaf_ts)

                # Store transformations in dataset-level dictionary for easy access
                self.transformations[sequence_name] = all_transformations

                # Save transformations to file if requested
                if self._save_transformations:
                    self.save_transformations(sequence_name, all_transformations)

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

    def _align_pca_to_z_axis(self, timepoints):
        """
        Align the main PCA axis of each timepoint to the positive Z-axis (upward).
        This makes all leaves have the same inclination (parallel to each other)
        and ensures leaves are not upside down.

        The main axis direction is checked first - if it points more downward than
        upward, it is flipped before computing the rotation. This ensures the rotation
        is always less than 90° and leaves remain right-side up.

        Args:
            timepoints: List of aligned timepoint dictionaries (modified in-place)

        Returns:
            List of transformation dictionaries for each timepoint
        """

        # TODO: reuse PCA from previous alignment step to avoid recomputing

        target_axis = np.array([0, 0, 1])  # Z-axis
        transformations = []

        for tp in timepoints:
            if tp["points"] is None or len(tp["points"]) < 3:
                transformations.append(
                    {
                        "stage": "align_to_z",
                        "day": tp["day"],
                        "rotation": np.eye(3),
                        "translation": np.zeros(3),
                        "center": np.zeros(3),
                    }
                )
                continue

            # Compute PCA on the leaf points
            points = tp["points"]
            center = np.mean(points, axis=0)
            centered = points - center

            # Get principal components
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            main_axis = Vt[0]  # First principal component (main axis)

            # Normalize
            main_axis = main_axis / np.linalg.norm(main_axis)

            # Ensure main axis points upward (positive Z direction)
            # If it points more downward than upward, flip it
            if np.dot(main_axis, target_axis) < 0:
                main_axis = -main_axis

            # Compute rotation matrix to align main_axis to Z-axis
            # Using Rodrigues' rotation formula
            v = np.cross(main_axis, target_axis)
            c = np.dot(main_axis, target_axis)

            # Check if vectors are already aligned or opposite
            if np.abs(c - 1.0) < 1e-8:
                # Already aligned, no rotation needed
                transformations.append(
                    {
                        "stage": "align_to_z",
                        "day": tp["day"],
                        "rotation": np.eye(3),
                        "translation": np.zeros(3),
                        "center": center,
                    }
                )
                continue
            elif np.abs(c + 1.0) < 1e-8:
                # Vectors are opposite, rotate 180° around any perpendicular axis
                # Use X-axis as rotation axis
                rot_matrix = np.array(
                    [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
                )
            else:
                # General case: use Rodrigues' formula
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rot_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

            # Apply rotation (centered around leaf centroid)
            tp["points"] = centered @ rot_matrix.T + center

            # Rotate normals if they exist
            if tp.get("normals") is not None:
                tp["normals"] = tp["normals"] @ rot_matrix.T

            # Rotate dense points if they exist
            if tp.get("dense_points") is not None:
                centered_dense = tp["dense_points"] - center
                tp["dense_points"] = centered_dense @ rot_matrix.T + center

            # Rotate leaf tip if it exists
            if tp.get("leaf_tip") is not None:
                centered_tip = tp["leaf_tip"] - center
                tp["leaf_tip"] = centered_tip @ rot_matrix.T + center

            transformations.append(
                {
                    "stage": "align_to_z",
                    "day": tp["day"],
                    "rotation": rot_matrix,
                    "translation": np.zeros(3),
                    "center": center,
                }
            )

        return transformations

    def _align_leaves_to_xy_plane(self, timepoints):
        """
        Align each timepoint independently so its lowest point is at z=0.
        This ensures leaves are positioned at a consistent height for visualization
        while preserving vertical growth differences between timepoints.

        Args:
            timepoints: List of aligned timepoint dictionaries (modified in-place)

        Returns:
            List of transformation dictionaries for each timepoint
        """
        transformations = []

        for tp in timepoints:
            if tp["points"] is not None and len(tp["points"]) > 0:
                # Find minimum z-coordinate for this timepoint
                min_z = np.min(tp["points"][:, 2])
                vertical_shift = np.array([0, 0, -min_z])

                # Shift points
                tp["points"] = tp["points"] + vertical_shift

                # Shift dense points
                if tp.get("dense_points") is not None:
                    tp["dense_points"] = tp["dense_points"] + vertical_shift

                # Shift leaf tip
                if tp.get("leaf_tip") is not None:
                    tp["leaf_tip"] = tp["leaf_tip"] + vertical_shift

                # Normals are direction vectors - no translation needed

                transformations.append(
                    {
                        "stage": "vertical_align",
                        "day": tp["day"],
                        "rotation": np.eye(3),
                        "translation": vertical_shift,
                        "center": np.zeros(3),
                    }
                )
            else:
                transformations.append(
                    {
                        "stage": "vertical_align",
                        "day": tp["day"],
                        "rotation": np.eye(3),
                        "translation": np.zeros(3),
                        "center": np.zeros(3),
                    }
                )

        return transformations

    def align_leaf_sequence(
        self,
        leaf_timeseries,
        normal_matching_percentile_threshold=75,
        plot_pairwise_alignment=False,
    ):
        """
        Align leaf sequence using PCA-based registration with sequential alignment.
        Preserves scale differences to show growth over time.
        Only removes rotation and translation.
        Works with different numbers of points between timepoints.

        Two-stage alignment process (after pre-alignment):
        1. Sequential PCA alignment: each timepoint is aligned to the previous one (i→i-1)
           rather than all to the first reference. This prevents alternating alignment behavior
           for leaves with changing shape.
        2. Normal-aware Z-rotation: if normals are available, computes optimal rotation angle
           around Z-axis to align normal directions, ensuring leaves maintain consistent
           face orientation across time (e.g., upper surface always points in same direction).

        Note: This assumes timepoints have been pre-aligned using leaf tips (Stage 0) and
        have consistent normals (Stage 1).
        Args:
            leaf_timeseries: Leaf timeseries dict from dataset
            normal_matching_percentile_threshold: Distance threshold for matching normals (if None, uses 75th percentile)
            plot_pairwise_alignment: If True, plot pairwise alignment for each timepoint to reference (for debugging)

        Returns:
            List of aligned point clouds and rotation matrices
        """
        timepoints = leaf_timeseries["timepoints"]
        if len(timepoints) < 2:
            return timepoints, []

        aligned_timepoints = []
        transformations = []

        # Sequential alignment: align each timepoint to the previous one
        for i, tp in enumerate(timepoints):
            points = tp["points"]
            center = np.mean(points, axis=0)
            centered = points - center

            if i == 0:
                # First timepoint stays as reference (just centered to its own center)
                ref_center = center
                aligned_points = points  # Keep at original position
                rotation_matrix = np.eye(3)
                basis = np.eye(3)
            else:
                # Align to previous aligned timepoint
                prev_aligned_tp = aligned_timepoints[i - 1]
                ref_points = prev_aligned_tp["points"]
                ref_center = np.mean(ref_points, axis=0)
                ref_centered = ref_points - ref_center

                # Find optimal rotation using PCA alignment
                rotation_matrix, basis = self._pca_align(centered, ref_centered)

                # Use normals to compute optimal Z-axis rotation (if available)
                # PCA aligns the leaf plane correctly, but we need to align the normal direction
                if (
                    tp.get("normals") is not None
                    and prev_aligned_tp.get("normals") is not None
                ):
                    curr_normals = tp["normals"]
                    prev_normals = prev_aligned_tp["normals"]

                    # Rotate normals using the computed rotation
                    rotated_normals = curr_normals @ rotation_matrix.T

                    # Find nearest neighbors to match normals
                    from scipy.spatial import cKDTree

                    ref_tree = cKDTree(ref_centered + ref_center)
                    aligned_points_temp = centered @ rotation_matrix.T + ref_center
                    distances, indices = ref_tree.query(aligned_points_temp, k=1)

                    # Use reliable matches for computing average direction
                    max_distance = np.percentile(
                        distances, normal_matching_percentile_threshold
                    )
                    reliable_matches = distances < max_distance

                    if np.any(reliable_matches):
                        # Compute average normal direction for current timepoint (after PCA rotation)
                        curr_avg_normal = np.mean(
                            rotated_normals[reliable_matches], axis=0
                        )
                        curr_avg_normal = curr_avg_normal / (
                            np.linalg.norm(curr_avg_normal) + 1e-8
                        )

                        # Compute average normal direction for previous timepoint
                        matched_prev_normals = prev_normals[indices[reliable_matches]]
                        prev_avg_normal = np.mean(matched_prev_normals, axis=0)
                        prev_avg_normal = prev_avg_normal / (
                            np.linalg.norm(prev_avg_normal) + 1e-8
                        )

                        # Project normals onto XY plane (since we only rotate around Z)
                        curr_xy = curr_avg_normal[:2]
                        prev_xy = prev_avg_normal[:2]

                        # Compute rotation angle around Z-axis to align current to previous
                        # Using atan2 to get the angle between the two vectors in XY plane
                        curr_angle = np.arctan2(curr_xy[1], curr_xy[0])
                        prev_angle = np.arctan2(prev_xy[1], prev_xy[0])
                        rotation_angle = prev_angle - curr_angle

                        # Create rotation matrix around Z-axis
                        cos_theta = np.cos(rotation_angle)
                        sin_theta = np.sin(rotation_angle)
                        rot_z = np.array(
                            [
                                [cos_theta, -sin_theta, 0],
                                [sin_theta, cos_theta, 0],
                                [0, 0, 1],
                            ],
                            dtype=np.float64,
                        )

                        # Apply Z-rotation to the PCA rotation
                        rotation_matrix = rot_z @ rotation_matrix

                # Apply rotation and translate to previous timepoint's center
                aligned_points = centered @ rotation_matrix.T + ref_center

            # Transform dense points if they exist
            # Use same center as sparse points for perfect alignment
            aligned_dense_points = None
            if tp.get("dense_points") is not None:
                dense_points = tp["dense_points"]
                dense_centered = dense_points - center  # Use sparse points center!
                if i == 0:
                    aligned_dense_points = dense_points  # Keep at original position
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
                if i == 0:
                    aligned_leaf_tip = original_tip  # Keep at original position
                else:
                    aligned_leaf_tip = centered_tip @ rotation_matrix.T + ref_center

            # Transform normals if they exist (normals are direction vectors, don't translate)
            aligned_normals = None
            if tp.get("normals") is not None:
                if i == 0:
                    aligned_normals = tp[
                        "normals"
                    ]  # First timepoint normals stay as is
                else:
                    # Rotate normals (no translation for direction vectors)
                    aligned_normals = tp["normals"] @ rotation_matrix.T

            # Preserve original timepoint structure
            aligned_tp = tp.copy()
            aligned_tp["points"] = aligned_points
            aligned_tp["dense_points"] = aligned_dense_points
            aligned_tp["leaf_tip"] = aligned_leaf_tip
            aligned_tp["normals"] = aligned_normals
            aligned_timepoints.append(aligned_tp)

            # Store transformation in standard format for compose_transformations
            # Transform: p' = (p - center) @ rotation.T + ref_center
            # Standard: p' = (p - center) @ rotation.T + center + translation
            # Therefore: translation = ref_center - center
            transformations.append(
                {
                    "stage": "pca_align",
                    "day": tp["day"],
                    "rotation": rotation_matrix,
                    "translation": ref_center - center,
                    "center": center,
                    "basis": basis,
                    "original_center": center,
                    "reference_center": ref_center,
                    "is_reference": i == 0,
                }
            )

        # Plot pairwise alignment (for sequential alignment, plot each pair i-1 -> i)
        if plot_pairwise_alignment:
            # Lazy import for visualization to avoid heavy dependencies
            from plant_shape_analysis.vis.plot_functions import (
                plot_pairwise_alignment_with_quivers,
            )

            for i in range(1, len(aligned_timepoints)):
                plot_pairwise_alignment_with_quivers(
                    aligned_timepoints, transformations, idx1=i - 1, idx2=i
                )

        return aligned_timepoints, transformations

    @staticmethod
    def compose_transformations(transformation_stages):
        """
        Compose all transformation stages into per-timepoint 4x4 homogeneous matrices.

        Args:
            transformation_stages: List of (stage_name, stage_transformations) tuples
                Each stage_transformations is a list of dicts with keys:
                - rotation: 3x3 rotation matrix
                - translation: 3D translation vector
                - center: 3D center point (for rotations around a point)

        Returns:
            List of dicts, one per timepoint, containing:
            - composed_matrix: 4x4 homogeneous transformation matrix
            - individual_stages: list of individual transformations in order
        """
        # Get number of timepoints from first non-None stage
        num_timepoints = 0
        for stage_name, stage_trans in transformation_stages:
            if stage_trans is not None:
                num_timepoints = len(stage_trans)
                break

        if num_timepoints == 0:
            return []

        composed_transforms = []

        for tp_idx in range(num_timepoints):
            # Start with identity
            composed_matrix = np.eye(4, dtype=np.float64)
            individual_stages_list = []
            day = None

            # Apply each stage in order
            for stage_name, stage_trans in transformation_stages:
                if stage_trans is None:
                    continue

                trans_info = stage_trans[tp_idx]
                R = trans_info["rotation"]
                t = trans_info["translation"]
                c = trans_info["center"]

                # Extract day from first non-None stage
                if day is None and "day" in trans_info:
                    day = trans_info["day"]

                # Build 4x4 transformation matrix for this stage
                # Transform is: p' = (p - c) @ R.T + c + t
                # In homogeneous coords: T = T(c+t) @ R @ T(-c)

                # T(-c): translate to origin
                T_neg_c = np.eye(4, dtype=np.float64)
                T_neg_c[:3, 3] = -c

                # R: rotation
                T_R = np.eye(4, dtype=np.float64)
                T_R[:3, :3] = R.T  # Note: we use R.T because points are row vectors

                # T(c+t): translate back and apply translation
                T_c_plus_t = np.eye(4, dtype=np.float64)
                T_c_plus_t[:3, 3] = c + t

                # Compose: first translate to origin, then rotate, then translate back
                stage_matrix = T_c_plus_t @ T_R @ T_neg_c

                # Compose with previous transformations
                # For row vectors (p @ M.T), we compose right-to-left: M = Sn @ ... @ S2 @ S1
                # This way: p @ M.T = p @ S1.T @ S2.T @ ... @ Sn.T (applies S1 first)
                composed_matrix = stage_matrix @ composed_matrix

                # Store individual stage info
                individual_stages_list.append(
                    {
                        "stage": trans_info.get("stage", stage_name),
                        "matrix": stage_matrix,
                        "rotation": R,
                        "translation": t,
                        "center": c,
                    }
                )

            composed_transforms.append(
                {
                    "day": day,
                    "composed_matrix": composed_matrix,
                    "individual_stages": individual_stages_list,
                }
            )

        return composed_transforms

    @staticmethod
    def invert_transformation(transform_matrix):
        """
        Compute inverse of a 4x4 homogeneous transformation matrix.

        Args:
            transform_matrix: 4x4 homogeneous transformation matrix

        Returns:
            4x4 inverse transformation matrix
        """
        return np.linalg.inv(transform_matrix)

    @staticmethod
    def apply_transformation(points, transform_matrix):
        """
        Apply a 4x4 homogeneous transformation to points.

        Args:
            points: Nx3 array of points
            transform_matrix: 4x4 homogeneous transformation matrix

        Returns:
            Nx3 array of transformed points
        """
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        # Apply transformation (row vectors, so points @ T.T)
        transformed_homo = points_homo @ transform_matrix.T
        # Convert back to 3D
        return transformed_homo[:, :3]

    @staticmethod
    def apply_inverse_transformations(leaf_timeseries, transformation_stages):
        """
        Apply inverse transformations to aligned leaf sequence to reconstruct original positions.

        This reverses all alignment stages (tip pre-alignment, PCA, align to Z, vertical shift)
        to place the leaf back in its original plant coordinate space.

        Args:
            leaf_timeseries: Leaf timeseries dict with aligned timepoints
            transformation_stages: Transformation stages from self.transformations[sequence_name]

        Returns:
            List of timepoint dicts with points in original (unaligned) space
        """
        # Compose all transformation stages
        composed = LeafSequencesDataset.compose_transformations(transformation_stages)

        original_timepoints = []
        for tp, comp in zip(leaf_timeseries["timepoints"], composed):
            # Get inverse matrix (aligned -> original)
            inv_matrix = LeafSequencesDataset.invert_transformation(comp["composed_matrix"])

            # Apply inverse to points
            original_points = LeafSequencesDataset.apply_transformation(tp["points"], inv_matrix)

            # Create new timepoint dict with original points
            original_tp = tp.copy()
            original_tp["points"] = original_points

            # Also transform dense points if they exist
            if tp.get("dense_points") is not None:
                original_tp["dense_points"] = LeafSequencesDataset.apply_transformation(
                    tp["dense_points"], inv_matrix
                )

            # Transform leaf tip if it exists
            if tp.get("leaf_tip") is not None:
                tip_as_array = tp["leaf_tip"].reshape(1, -1)
                original_tp["leaf_tip"] = LeafSequencesDataset.apply_transformation(
                    tip_as_array, inv_matrix
                )[0]

            # Note: Normals are direction vectors, need rotation-only transform
            if tp.get("normals") is not None:
                # Extract rotation part of inverse (upper-left 3x3)
                R_inv = inv_matrix[:3, :3]
                original_tp["normals"] = tp["normals"] @ R_inv.T

            original_timepoints.append(original_tp)

        return original_timepoints

    def save_transformations(
        self, sequence_name, transformation_stages, save_dir="transformations"
    ):
        """
        Save transformation matrices to JSON file for later use.
        Includes both individual transformation stages and composed transformations.

        Args:
            sequence_name: Name of the sequence
            transformation_stages: List of (stage_name, stage_transformations) tuples
            save_dir: Directory to save transformations
        """
        save_path = self.dataset_path / save_dir
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{sequence_name}_transformations.json"
        filepath = save_path / filename

        # Compose transformations
        composed_transforms = self.compose_transformations(transformation_stages)

        # Prepare JSON-serializable data
        json_data = {"sequence_name": sequence_name, "stages": [], "timepoints": []}

        # Save individual stages
        for stage_name, stage_trans in transformation_stages:
            if stage_trans is None:
                json_data["stages"].append(
                    {"stage_name": stage_name, "transformations": None}
                )
                continue

            stage_list = []
            for trans in stage_trans:
                json_trans = {}
                json_trans["stage"] = trans.get("stage", stage_name)
                if isinstance(trans.get("rotation"), np.ndarray):
                    json_trans["rotation"] = trans["rotation"].tolist()
                if isinstance(trans.get("translation"), np.ndarray):
                    json_trans["translation"] = trans["translation"].tolist()
                if isinstance(trans.get("center"), np.ndarray):
                    json_trans["center"] = trans["center"].tolist()
                # Include any other fields (like day, is_reference, etc.)
                for key, value in trans.items():
                    if key not in ["rotation", "translation", "center", "stage"]:
                        if isinstance(value, np.ndarray):
                            json_trans[key] = value.tolist()
                        else:
                            json_trans[key] = value
                stage_list.append(json_trans)

            json_data["stages"].append(
                {"stage_name": stage_name, "transformations": stage_list}
            )

        # Save composed transformations per timepoint
        for tp_idx, comp_trans in enumerate(composed_transforms):
            tp_data = {
                "timepoint_index": tp_idx,
                "day": comp_trans["day"],
                "composed_matrix": comp_trans["composed_matrix"].tolist(),
                "inverse_matrix": self.invert_transformation(
                    comp_trans["composed_matrix"]
                ).tolist(),
                "individual_stages": [],
            }

            for stage_info in comp_trans["individual_stages"]:
                tp_data["individual_stages"].append(
                    {
                        "stage": stage_info["stage"],
                        "matrix": stage_info["matrix"].tolist(),
                        "rotation": stage_info["rotation"].tolist(),
                        "translation": stage_info["translation"].tolist(),
                        "center": stage_info["center"].tolist(),
                    }
                )

            json_data["timepoints"].append(tp_data)

        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Transformations saved to {filepath}")

    def __len__(self):
        return len(self.leaf_timeseries)

    def __getitem__(self, idx):
        return self.leaf_timeseries[idx]


if __name__ == "__main__":

    dataset_path = Path("data/TrackPlant3D/versions")

    # Example usage for PlantSequencesDataset with alignment
    print("Creating plant dataset...")
    plant_dataset = PlantSequencesDataset(
        dataset_path,
        version="v1",
        alignment_method="stem_based",
        estimate_normals=True,
        save_transformations=False,
        use_ply=True,
    )
    print("Plant sequences dataset:")
    print(f"Number of sequences: {len(plant_dataset)}")
    print(
        "Available sequences (first 5):", plant_dataset.get_sequence_names()[:5]
    )  # Show first 5
    print("\n")

    # Visualize some plant sequences
    from plant_shape_analysis.vis.plot_functions import visualize_plant_sequence

    plant_sequences = [
        "maize_control_plant2",
        "tomato2_control_plant2",
    ]

    for i, seq in enumerate(plant_sequences):
        sample = plant_dataset.get_timeseries_by_sequence_name(seq)
        print(f"Visualizing plant sequence {seq} ({i+1}/{len(plant_sequences)})...")
        visualize_plant_sequence(
            sample,
            dense_points=False,
            color_by_organ=True,
            show_leaf_tips=True,
            spacing=100.0,
        )

    # # Example usage for LeafSequencesDataset
    # print("Creating leaf dataset with PCA alignment...")
    # leaf_dataset = LeafSequencesDataset(
    #     dataset_path, min_timepoints=3, apply_pca_alignment=True
    # )
    # # print("Leaf timeseries dataset:")
    # # print(f"Number of leaf timeseries: {len(leaf_dataset)}")

    # # info = leaf_dataset.get_leaf_timeseries_info()
    # # print("Dataset info:", info)
    # # print("\n")

    # # sample = leaf_dataset[0]

    # # Visualize some leaf sequences
    # print("Visualizing some leaf sequences...")

    # from plant_shape_analysis.vis.plot_functions import visualize_leaf_sequence

    # sequences = [
    #     "maize_control_plant2_leaf2",
    #     "tomato2_control_plant2_leaf1",
    #     "tomato2_control_plant3_leaf2",
    # ]

    # for i, seq in enumerate(sequences):
    #     sample = leaf_dataset.get_timeseries_by_sequence_name(seq)
    #     print(f"Visualizing leaf sequence {seq} ({i+1}/{len(sequences)})...")
    #     visualize_leaf_sequence(sample, dense_points=True)
