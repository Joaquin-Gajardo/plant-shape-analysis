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

        # Apply alignment if requested
        if self.alignment_method is not None:
            print(
                f"Applying {self.alignment_method.upper()} alignment to {len(self.plant_timeseries)} plant sequences..."
            )
            self._apply_alignment_to_dataset()

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

    def _apply_alignment_to_dataset(self):
        """Apply alignment (PCA or ICP) to all plant sequences in the dataset."""
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

                # Save transformations if requested
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
        Processes each leaf (organ label) separately to ensure normals remain consistent
        on the same face of each leaf over time.

        Args:
            timepoints: List of timepoint dictionaries containing 'points', 'labels', and 'normals'
            reference_idx: Index of reference timepoint (default: 0)
        """
        if len(timepoints) < 2:
            return

        # Reference timepoint
        ref_tp = timepoints[reference_idx]
        if "normals" not in ref_tp or ref_tp["normals"] is None:
            return

        ref_points = ref_tp["points"]
        ref_labels = ref_tp["labels"]
        ref_normals = ref_tp["normals"]

        # Get unique organ labels (excluding stem label 0 if desired, but include all for now)
        unique_labels = np.unique(ref_labels)

        # Process each subsequent timepoint
        for i, tp in enumerate(timepoints):
            if i == reference_idx:
                continue

            if "normals" not in tp or tp["normals"] is None:
                continue

            curr_points = tp["points"]
            curr_labels = tp["labels"]
            curr_normals = tp["normals"]

            # Process each organ separately
            for organ_label in unique_labels:
                # Skip if this organ doesn't exist in current timepoint
                curr_organ_mask = curr_labels == organ_label
                if not np.any(curr_organ_mask):
                    continue

                # Skip if organ doesn't exist in reference
                ref_organ_mask = ref_labels == organ_label
                if not np.any(ref_organ_mask):
                    continue

                # Get points and normals for this organ
                curr_organ_points = curr_points[curr_organ_mask]
                curr_organ_normals = curr_normals[curr_organ_mask]
                ref_organ_points = ref_points[ref_organ_mask]
                ref_organ_normals = ref_normals[ref_organ_mask]

                # Build KD-tree for reference organ points
                from scipy.spatial import cKDTree

                ref_tree = cKDTree(ref_organ_points)

                # Find nearest neighbors in reference frame
                distances, indices = ref_tree.query(curr_organ_points, k=1)

                # For each point, check if normal should be flipped
                matched_ref_normals = ref_organ_normals[indices]

                # Compute dot product between current and reference normals
                dot_products = np.sum(curr_organ_normals * matched_ref_normals, axis=1)

                # Flip normals where dot product is negative
                flip_mask = dot_products < 0
                curr_organ_normals[flip_mask] = -curr_organ_normals[flip_mask]

                # Update normals in the full array
                curr_normals[curr_organ_mask] = curr_organ_normals

            # Update normals in timepoint
            tp["normals"] = curr_normals

    def save_transformations(
        self, sequence_name, transformations, save_dir="transformations"
    ):
        """Save transformation matrices to JSON file for later use."""
        save_path = self.dataset_path / save_dir
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{sequence_name}_transformations.json"
        filepath = save_path / filename

        # Convert numpy arrays to lists for JSON serialization
        json_transformations = []
        for trans in transformations:
            json_trans = trans.copy()
            if isinstance(json_trans.get("rotation_matrix"), np.ndarray):
                json_trans["rotation_matrix"] = json_trans["rotation_matrix"].tolist()
            if isinstance(json_trans.get("basis"), np.ndarray):
                json_trans["basis"] = json_trans["basis"].tolist()
            if isinstance(json_trans.get("original_center"), np.ndarray):
                json_trans["original_center"] = json_trans["original_center"].tolist()
            if isinstance(json_trans.get("reference_center"), np.ndarray):
                json_trans["reference_center"] = json_trans["reference_center"].tolist()
            json_transformations.append(json_trans)

        with open(filepath, "w") as f:
            json.dump(json_transformations, f, indent=2)

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
        alignment_method: Optional[str] = None,
        plant_alignment_method: Optional[str] = None,
        estimate_normals: bool = False,
        estimate_plant_normals: bool = False,
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
            alignment_method: Alignment method - None (no alignment) or 'pca' (PCA alignment)
            plant_alignment_method: Alignment method for plants when tracking leaves
            estimate_normals: If True, estimate normals for all leaves
            estimate_plant_normals: If True, estimate normals for plants before tracking leaves
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
        self.alignment_method = alignment_method
        self.plant_alignment_method = plant_alignment_method
        self._save_transformations = save_transformations
        self.estimate_normals = estimate_normals

        # Build leaf timeseries samples
        self.leaf_timeseries = self._build_leaf_timeseries()

        if len(self.leaf_timeseries) == 0:
            raise ValueError(
                "No leaf timeseries found with the given parameters. "
                "Verify dataset path and file extension (use `use_ply=True` if loading PLY files)."
            )
        # Apply alignment if requested
        if self.alignment_method is not None:
            print(
                f"Applying {self.alignment_method.upper()} alignment to {len(self.leaf_timeseries)} leaf sequences..."
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

        # Iterate through plant_timeseries directly instead of reloading from files
        # This ensures we use the data with normals if they were estimated
        for plant_ts in self.plant_dataset.plant_timeseries:
            sequence_name = plant_ts["sequence_name"]
            sequence_data = plant_ts["timepoints"]  # Use already-loaded data with normals!

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
            normals = timepoint_data.get("normals", None)  # Extract normals if available

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
            self._enforce_temporal_normal_consistency(leaf["timepoints"])
        print("Normal estimation complete.")

    def _enforce_temporal_normal_consistency(self, timepoints, reference_idx=0):
        """
        Enforce temporal consistency of normals across timepoints for a leaf sequence.
        Uses nearest neighbor matching to ensure normals point in the same direction over time.

        Args:
            timepoints: List of timepoint dictionaries containing 'points' and 'normals'
            reference_idx: Index of reference timepoint (default: 0)
        """
        if len(timepoints) < 2:
            return

        # Reference timepoint normals
        ref_tp = timepoints[reference_idx]
        if "normals" not in ref_tp or ref_tp["normals"] is None:
            return

        ref_points = ref_tp["points"]
        ref_normals = ref_tp["normals"]

        # Build KD-tree for reference points
        from scipy.spatial import cKDTree

        ref_tree = cKDTree(ref_points)

        # Process each subsequent timepoint
        for i, tp in enumerate(timepoints):
            if i == reference_idx:
                continue

            if "normals" not in tp or tp["normals"] is None:
                continue

            curr_points = tp["points"]
            curr_normals = tp["normals"]

            # Find nearest neighbors in reference frame
            distances, indices = ref_tree.query(curr_points, k=1)

            # For each point, check if normal should be flipped
            # by comparing with nearest neighbor's normal in reference
            matched_ref_normals = ref_normals[indices]

            # Compute dot product between current and reference normals
            dot_products = np.sum(curr_normals * matched_ref_normals, axis=1)

            # Flip normals where dot product is negative (pointing opposite direction)
            flip_mask = dot_products < 0
            curr_normals[flip_mask] = -curr_normals[flip_mask]

            # Update normals in timepoint
            tp["normals"] = curr_normals

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

    def align_leaf_sequence(
        self, leaf_timeseries, reference_idx=0, plot_pairwise_alignment=False
    ):
        """
        Align leaf sequence using PCA-based registration, preserving scale differences.
        Only removes rotation and translation to show growth over time.
        Works with different numbers of points between timepoints.
        Ensures consistent orientation across all timepoints.

        Args:
            leaf_timeseries: Leaf timeseries dict from dataset
            reference_idx: Index of reference timepoint (default: 0)
            plot_pairwise_alignment: If True, plot pairwise alignment for each timepoint to reference (for debugging)

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
            # Use same center as sparse points for perfect alignment
            aligned_dense_points = None
            if tp.get("dense_points") is not None:
                dense_points = tp["dense_points"]
                dense_centered = dense_points - center  # Use sparse points center!
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

        # Plot pairwise alignment
        if plot_pairwise_alignment and i != reference_idx:
            # Lazy import for visualization to avoid heavy dependencies
            from plant_shape_analysis.vis.plot_functions import (
                plot_pairwise_alignment_with_quivers,
            )

            plot_pairwise_alignment_with_quivers(
                aligned_timepoints, transformations, idx1=reference_idx, idx2=i
            )

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
            if isinstance(json_trans.get("basis"), np.ndarray):
                json_trans["basis"] = json_trans["basis"].tolist()
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
