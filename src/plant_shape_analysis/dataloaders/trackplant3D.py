import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

from plant_shape_analysis import alignment
from plant_shape_analysis.dataloaders import normal_estimation, point_cloud_utils


class PlantSequencesDataset(Dataset):
    DATASET_CONFIGS = {
        "v1": {
            "data_dirs": {"sparse": "gt_corrected_v1", "dense": "dense"},
            "download_info": {
                "url": "https://polybox.ethz.ch/index.php/s/mxiZwKfCfd39Rxx/download",
                "filename": "v1.zip",
                "extract_dir": "v1",
                "size_mb": 660,
                "description": "TrackPlant3D v1 dataset with leaf keypoint annotations and dense point clouds",
            },
            "orientation_corrected": False,  # Needs Y->Z correction for some crops
        },
        "v2": {
            "data_dirs": {"sparse": "gt_corrected_v2", "dense": None},
            "download_info": {
                "url": "https://polybox.ethz.ch/index.php/s/7XwferiX92aogn5/download",
                "filename": "v2.zip",
                "extract_dir": "v2",
                "size_mb": 130,
                "description": "TrackPlant3D v2 with pre-aligned point clouds and corrected normals (faster loading)",
            },
            "orientation_corrected": True,  # PLY files have been corrected Y→Z rotation for some sequences during preprocessing, set to False when loading raw data in TXT format
        },
        "v3": {
            "data_dirs": {
                # "sparse": "predicted_labels_provided_checkpoint",
                # "sparse": "predicted_labels", # retrained PSegNet
                # Directory name only determines which PLY files are loaded. The "predicted_labels"
                # timepoint key comes from reading the `predicted_organ_label` PLY field inside
                # those files (see point_cloud_utils.load_point_cloud).
                # "sparse": "predicted_labels_combined",  # Sorghum come from old checkpoint preds (more consistent stem=0)
                "sparse": "autoseg_tracking_allsequences",  # Sorghum come from old checkpoint preds (more consistent stem=0)
                "dense": None,
            },
            "download_info": {
                "url": None,
                "filename": None,
                "extract_dir": "v2",  # predicted_labels/ lives inside the v2 directory
                "size_mb": None,
                "description": "Model-predicted organ labels (PSegNet + TrackPlant3D). Generate with scripts/run_autoseg_pipeline.py",
            },
            "orientation_corrected": True,
            "no_auto_download": True,
        },
    }

    def __init__(
        self,
        dataset_path,
        version="v2",
        use_ply=True,
        alignment_method=None,
        save_transformations=False,
        estimate_normals=False,
        auto_download=True,
        manual_z_rotations=None,
        verbose=False,
        exclude_sequences=None,
    ):
        """
        Initialize PlantSequencesDataset.

        Args:
            dataset_path: Path to TrackPlant3D dataset
            version: Dataset version (default: "v2")
            use_ply: If True, load from PLY files instead of TXT files. Keeping both options for compatibility to original dataset format.
            alignment_method: Alignment method - None (no alignment), 'pca' (fast, approximate), 'icp' (slower, more accurate), or 'stem_based' (uses only stem points with sequential alignment and vertical correction)
            save_transformations: If True, save transformation matrices when applying alignment
            estimate_normals: If True, estimate normals for all plant point clouds
            auto_download: If True, automatically download dataset if not found (default: True)
            manual_z_rotations: Dict mapping sequence_name -> {timepoint_idx: angle_deg}
                               Example: {"tobacco_control_plant1": {6: 144.0}}
            selected_sequences: Optional list of sequence names to process (default: None = all sequences)
            verbose: If True, print debug information during stem alignment (default: False)

        Important:
            When using 'stem_based' alignment with TXT files from the original dataset (v1 or custom versions),
            ensure that orientation_corrected=False in the version config. The stem-based alignment expects
            Z-axis to be the vertical axis. For crops like tobacco, tomato1, and sorghum that originally use
            Y-axis as vertical, orientation correction must be applied BEFORE stem alignment to avoid plants
            appearing upside down. The v1 config has orientation_corrected=False by default.
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
        self.orientation_corrected = config.get("orientation_corrected", False)

        # Auto-download if version directory doesn't exist
        if not self.dataset_path.exists():
            if config.get("no_auto_download", False):
                raise FileNotFoundError(
                    f'Predicted labels not found at "{self.dataset_path.resolve()}". '
                    "Run scripts/run_autoseg_pipeline.py first to generate them."
                )
            elif auto_download:
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
        self.verbose = verbose
        self.manual_z_rotations = manual_z_rotations or {}
        self.exclude_sequences = set(exclude_sequences or [])

        # Set paths from config
        self.sparse_path = self.dataset_path / data_dirs["sparse"]
        self.dense_path = (
            self.dataset_path / data_dirs["dense"] if data_dirs["dense"] else None
        )

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

        # Check for problematic configuration: stem_based alignment with TXT files and orientation_corrected=True
        if (
            self.alignment_method == "stem_based"
            and self.orientation_corrected
            and not self.use_ply
        ):
            raise ValueError(
                f"Invalid configuration: stem_based alignment with orientation_corrected=True and use_ply=False.\n"
                f"The stem-based alignment expects Z-axis to be vertical, but TXT files for tobacco, tomato1, "
                f"and sorghum use Y-axis in the original data.\n"
                f"Solution: Set orientation_corrected=False in the DATASET_CONFIGS['{version}'] configuration,"
                f"or ensure your TXT data has been geometrically corrected (Y→Z rotation applied) during preprocessing."
            )

        # Apply alignment if requested
        if self.alignment_method is not None:
            print(
                f"Applying {self.alignment_method.upper()} alignment to {len(self.plant_timeseries)} plant sequences..."
            )
            self._align_dataset()

        # Estimate normals if requested
        if self.estimate_normals:
            # Done after alignment since we don't use them for alignment
            # like in leaves, and to avoid extra computation to rotate them
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
            if sequence_name in self.exclude_sequences:
                continue
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
        """Load point cloud (wrapper for point_cloud_utils.load_point_cloud)"""
        return point_cloud_utils.load_point_cloud(file_path, use_ply=self.use_ply)

    def load_dense_point_cloud(self, file_path):
        """Load dense point cloud (wrapper for point_cloud_utils.load_dense_point_cloud)"""
        return point_cloud_utils.load_dense_point_cloud(
            file_path, self.dense_path, use_ply=self.use_ply
        )

    def load_leaf_tips(self, point_cloud_path):
        """Load leaf tips (wrapper for point_cloud_utils.load_leaf_tips)"""
        return point_cloud_utils.load_leaf_tips(
            point_cloud_path, self.leaf_tips_path, use_ply=self.use_ply
        )

    def get_sequence_data(self, sequence_name):
        """Get all point clouds and leaf tips for a sequence"""
        files = self._get_sequence(sequence_name)
        sequence_data = []
        # Only apply orientation correction if dataset version needs it
        needs_correction = (
            not self.orientation_corrected
            and point_cloud_utils.needs_orientation_correction(sequence_name)
        )

        for file_path in files:
            # Extract day from filename
            day_match = re.search(r"D(\d+)", file_path.name)
            day = int(day_match.group(1)) if day_match else 0

            # Load point cloud
            points, labels, normals, predicted_labels = self.load_point_cloud(file_path)

            # Load dense point cloud if available
            dense_points, dense_labels, dense_normals, _ = self.load_dense_point_cloud(
                file_path
            )

            # Load leaf tips if available
            leaf_tip_idxs, leaf_tip_coordinates = self.load_leaf_tips(file_path)

            # Apply orientation correction if needed (before validation)
            if needs_correction:
                points, normals = point_cloud_utils.correct_orientation(points, normals)
                if dense_points is not None:
                    dense_points, dense_normals = point_cloud_utils.correct_orientation(
                        dense_points, dense_normals
                    )
                if leaf_tip_coordinates.size > 0:
                    leaf_tip_coordinates, _ = point_cloud_utils.correct_orientation(
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
                    "normals": normals,
                    "predicted_labels": predicted_labels,
                    "dense_points": dense_points,
                    "dense_labels": dense_labels,
                    "leaf_tip_idxs": leaf_tip_idxs,
                    "file_path": file_path,
                }
            )

        return sequence_data

    def _pca_align(self, pc1, pc2):
        """PCA alignment (wrapper for alignment.pca_align_pair)"""
        return alignment.pca_align_pair(pc1, pc2)

    def align_plant_sequence(self, sequence_data, reference_idx=0, method=None):
        """
        Align plant sequence using PCA, ICP, or stem-based registration.
        Preserves scale differences to show growth over time.
        Only removes rotation and translation.
        Works with different numbers of points between timepoints.
        Ensures consistent orientation across all timepoints.

        Args:
            sequence_data: List of timepoint dictionaries with 'points', 'labels', etc.
            reference_idx: Index of reference timepoint (default: 0, ignored for stem_based)
            method: Alignment method - 'pca', 'icp', or 'stem_based'. If None, uses self.alignment_method

        Returns:
            List of aligned timepoint data and transformation matrices
        """
        method = method or self.alignment_method

        # Stem-based alignment uses sequential approach
        if method == "stem_based":
            return self._align_stem_based(sequence_data)

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

    def _align_stem_based(self, sequence_data, sequence_name=None):
        """
        Align plant sequence using stem-based ICP with vertical correction.

        This is a two-stage sequential alignment approach:
        - Stage 1: Stem-based ICP alignment (rotation + translation)
        - Stage 2: Vertical shift correction using stem base centroid

        Args:
            sequence_data: List of timepoint dictionaries with 'points', 'labels', etc.
            sequence_name: Optional sequence name for manual Z-rotation lookup

        Returns:
            List of aligned timepoint data and transformation matrices
        """
        from plant_shape_analysis.alignment.stem_based_alignment import (
            align_plant_sequence_stem_based,
        )

        if len(sequence_data) < 2:
            return sequence_data, []

        # Get manual Z-rotations for this sequence (if any)
        sequence_manual_rotations = None
        if sequence_name is not None:
            sequence_manual_rotations = self.manual_z_rotations.get(sequence_name, None)

        # Run stem-based alignment
        aligned_timepoints, transformations = align_plant_sequence_stem_based(
            sequence_data,
            use_rotation=True,
            max_iterations=200,
            manual_z_rotations=sequence_manual_rotations,
            verbose=self.verbose,
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
                    # Reference frame - only apply vertical shift if present
                    if vertical_shift is not None and np.any(vertical_shift != 0):
                        aligned_dense_points = dense_points + vertical_shift
                    else:
                        aligned_dense_points = dense_points
                else:
                    # Apply same transformation as sparse points
                    # rotation and translation are cumulative from all stages
                    # vertical_shift is applied last (stage 2 or 3)
                    if vertical_shift is not None and np.any(vertical_shift != 0):
                        # Apply rotation, translation, and vertical shift
                        aligned_dense_points = (
                            dense_points @ rotation.T + translation + vertical_shift
                        )
                    else:
                        # Apply rotation and translation only
                        aligned_dense_points = dense_points @ rotation.T + translation

                result_tp["dense_points"] = aligned_dense_points

            # Transform normals if they exist (normals are direction vectors, only rotate)
            if timepoint_data.get("normals") is not None:
                normals = timepoint_data["normals"]

                if i == 0:
                    # Reference frame - no rotation
                    result_tp["normals"] = normals
                else:
                    # Apply cumulative rotation (no translation for direction vectors)
                    result_tp["normals"] = normals @ rotation.T

            # Note: leaf_tip_idxs remain the same (indices into aligned points)
            # The aligned coordinates are automatically correct since points are aligned

            aligned_sequence.append(result_tp)

        # Format transformations to match other alignment methods
        formatted_transformations = []
        for trans_info in transformations:
            formatted_trans = {
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
            # Add manual Z-rotation details if present
            if "manual_z_rotation_deg" in trans_info:
                formatted_trans["manual_z_rotation_deg"] = trans_info[
                    "manual_z_rotation_deg"
                ]

            formatted_transformations.append(formatted_trans)

        return aligned_sequence, formatted_transformations

    def _align_dataset(self):
        """Apply alignment (PCA, ICP, or stem-based) to all plant sequences in the dataset."""
        aligned_timeseries = []

        for plant_ts in self.plant_timeseries:
            sequence_name = plant_ts["sequence_name"]
            timepoints = plant_ts["timepoints"]

            if len(timepoints) >= 2:
                # Apply alignment with first timepoint as reference
                # Pass timepoints data directly to avoid reloading from files
                if self.alignment_method == "stem_based":
                    # Stem-based needs sequence_name for manual Z-rotations
                    aligned_data, transformations = self._align_stem_based(
                        timepoints, sequence_name=sequence_name
                    )
                else:
                    aligned_data, transformations = self.align_plant_sequence(
                        timepoints, reference_idx=0, method=self.alignment_method
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
        """Estimate normals (wrapper for normal_estimation.estimate_normals_for_timeseries)"""
        normal_estimation.estimate_normals_for_timeseries(
            self.plant_timeseries, object_type="plant"
        )

    def _enforce_temporal_normal_consistency_plant(self, timepoints, reference_idx=0):
        """Enforce temporal normal consistency (wrapper for normal_estimation.enforce_temporal_normal_consistency)"""
        normal_estimation.enforce_temporal_normal_consistency(timepoints, reference_idx)

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
        version: str = "v2",
        min_timepoints: int = 3,
        max_timepoints: Optional[int] = None,
        apply_alignment: bool = False,
        plant_alignment_method: Optional[str] = None,
        estimate_plant_normals: bool = False,
        estimate_normals: bool = False,
        use_ply: bool = True,
        save_transformations: bool = False,
        auto_download: bool = True,
        manual_z_rotations: Optional[dict] = None,
        exclude_sequences: Optional[list] = None,
    ):
        """
        Initialize LeafSequencesDataset.

        Args:
            dataset_path: Path to TrackPlant3D dataset
            version: Dataset version (default: "v2")
            min_timepoints: Minimum number of timepoints for a leaf sequence
            max_timepoints: Maximum number of timepoints (None = no limit)
            apply_alignment: If True, align leaf sequences individually (multi-state approach)
            plant_alignment_method: Alignment method for plants when tracking leaves
            estimate_plant_normals: If True, estimate normals for plants before tracking leaves. Normals will be included in leaf data.
            estimate_normals: If True, estimate normals for all leaves. Ignored if plant normals are estimated.
            use_ply: If True, load from PLY files instead of TXT files
            save_transformations: If True, save transformation matrices when applying alignment
            auto_download: If True, automatically download dataset if not found (default: True)
            manual_z_rotations: Dict mapping sequence_name -> {timepoint_idx: angle_deg}
                               for manual Z-axis rotations (passed to PlantSequencesDataset)
            exclude_sequences: Optional list of plant sequence names to skip entirely.
        """
        self.plant_dataset = PlantSequencesDataset(
            dataset_path,
            version=version,
            use_ply=use_ply,
            alignment_method=plant_alignment_method,
            estimate_normals=estimate_plant_normals,
            auto_download=auto_download,
            manual_z_rotations=manual_z_rotations,
            exclude_sequences=exclude_sequences,
        )
        self.dataset_path = Path(dataset_path)
        self.min_timepoints = min_timepoints
        self.max_timepoints = max_timepoints
        self.apply_alignment = apply_alignment
        self.plant_alignment_method = plant_alignment_method
        self._save_transformations = save_transformations
        self.estimate_normals = estimate_normals
        self.label_field = "predicted_labels" if version == "v3" else "labels"

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

            # Extract stem timepoints for this plant (needed for stem-based PCA correction)
            stem_timepoints = self._extract_stem_timepoints(sequence_data)

            # Track leaves across time points
            leaf_tracks = self._track_leaves_across_time(sequence_data, sequence_name)

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
                                "stem_timepoints": stem_timepoints,  # Store stem data
                            }
                        )

        return leaf_timeseries

    def _extract_stem_timepoints(self, sequence_data):
        """
        Extract stem points from plant sequence data.

        Args:
            sequence_data: List of plant timepoint dictionaries

        Returns:
            List of stem timepoint dictionaries (one per day)
        """
        stem_timepoints = []

        for timepoint in sequence_data:
            # Extract stem points (label 0)
            # stem_mask = timepoint["labels"] == 0
            stem_mask = timepoint[self.label_field] == 0
            stem_points = timepoint["points"][stem_mask]

            # Extract stem normals if available
            stem_normals = None
            if timepoint.get("normals") is not None:
                stem_normals = timepoint["normals"][stem_mask]

            stem_timepoints.append(
                {
                    "day": timepoint["day"],
                    "points": stem_points,
                    "normals": stem_normals,
                }
            )

        return stem_timepoints

    def _track_leaves_across_time(self, sequence_data, sequence_name="unknown"):
        """Track individual leaves across time points in a sequence"""
        leaf_tracks = defaultdict(list)

        for timepoint_data in sequence_data:
            day = timepoint_data["day"]
            points = timepoint_data["points"]
            labels = timepoint_data[self.label_field]
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

                # Skip degenerate fragments — PCA requires at least 3 points.
                # NOTE: this silently drops emerging leaves at their first appearance.
                # For predicted labels this is rare (2 cases in v3), but with noisier
                # segmentation models it could be more frequent and cause completeness issues.
                if len(leaf_points) < 3:
                    print(
                        f"Warning: dropping {sequence_name} day={day} leaf={leaf_label} "
                        f"({len(leaf_points)} pts < 3, PCA not possible)"
                    )
                    continue

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
        """Estimate normals (wrapper for normal_estimation.estimate_normals_for_timeseries)"""
        normal_estimation.estimate_normals_for_timeseries(
            self.leaf_timeseries, object_type="leaf"
        )

    def _align_dataset(self):
        """
        Apply PCA-based alignment to all leaf sequences in the dataset.

        New pipeline (uses normals inherited from plant alignment):
        1. Correct PCA axis sign using stem information
           - Determines leaf insertion point from stem
           - Ensures PCA axis points from base to tip
           - Fixes sign ambiguity even without leaf tips
        2. Align main PCA axis to Z-axis
           - Gets leaf standing vertically using corrected PCA basis
           - No additional flipping needed (orientation correct from Stage 1)
        3. Z-axis rotation using normals
           - Sequentially aligns each timepoint to previous using normals
           - Computes optimal rotation angle around Z-axis only
           - Preserves vertical alignment while fixing rotational orientation
        4. Align base to origin
           - Uses lowest 1% of points to robustly estimate base location
           - Brings base centroid to origin (0,0,0)
        """

        aligned_timeseries = []
        for i, leaf_ts in enumerate(self.leaf_timeseries):
            sequence_name = leaf_ts["sequence_name"]

            if len(leaf_ts["timepoints"]) >= 2:
                all_transformations = []

                # Start with leaf timepoints (normals inherited from plant dataset)
                aligned_timepoints = [tp.copy() for tp in leaf_ts["timepoints"]]

                # Stage 1: Correct PCA axis sign using stem information
                stage1_trans = self._correct_pca_axis_with_stem(
                    aligned_timepoints, leaf_ts["stem_timepoints"]
                )
                all_transformations.append(("correct_pca_stem", stage1_trans))

                # Stage 2: Align main PCA axis to Z-axis (make leaves vertical)
                # Reuse corrected PCA basis from Stage 1
                stage2_trans = self._align_pca_to_z_axis(
                    aligned_timepoints,
                    basis_from_pca=stage1_trans,
                    seq_name=sequence_name,
                )
                all_transformations.append(("align_to_z", stage2_trans))

                # Stage 3: Sequential Z-axis rotation alignment
                # stage3_trans = self._align_z_rotation_sequential(aligned_timepoints)
                # all_transformations.append(("z_rotation_sequential", stage3_trans))
                stage3_trans = self._align_z_rotation_with_normals(aligned_timepoints)
                all_transformations.append(("z_rotation_normals", stage3_trans))

                # Stage 4: Align base to origin
                stage4_trans = self._align_base_to_origin(aligned_timepoints)
                all_transformations.append(("align_base", stage4_trans))

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
        """PCA alignment (wrapper for alignment.pca_align_pair)"""
        return alignment.pca_align_pair(pc1, pc2)

    def _correct_pca_axis_with_stem(self, leaf_timepoints, stem_timepoints):
        """Correct PCA axis sign using stem (wrapper for alignment.correct_pca_axis_with_stem)"""
        return alignment.correct_pca_axis_with_stem(leaf_timepoints, stem_timepoints)

    def _align_pca_to_z_axis(self, timepoints, basis_from_pca=None, seq_name=None):
        """Align main axis to Z (wrapper for alignment.align_main_axis_to_z)"""
        return alignment.align_main_axis_to_z(
            timepoints, basis_from_pca=basis_from_pca, seq_name=seq_name
        )

    def _align_base_to_origin(self, timepoints, base_percentile=1.0):
        """Align base to origin (wrapper for alignment.align_base_to_origin)"""
        return alignment.align_base_to_origin(timepoints, base_percentile)

    def _align_z_rotation_sequential(
        self, timepoints, normal_matching_percentile_threshold=75, use_normals=True
    ):
        """Sequential Z-axis rotation (wrapper for alignment.align_z_rotation_sequential)"""
        return alignment.align_z_rotation_sequential(
            timepoints, normal_matching_percentile_threshold, use_normals
        )

    def _align_z_rotation_with_normals(
        self, timepoints, normal_matching_percentile_threshold=75
    ):
        """Align Z rotation with normals (wrapper for alignment.align_z_rotation_with_normals)"""
        return alignment.align_z_rotation_with_normals(
            timepoints, normal_matching_percentile_threshold
        )

    # def _align_sequence_pairwise_with_basis(
    #     self, timepoints, initial_basis_trans, normal_matching_percentile_threshold=75
    # ):
    #     """
    #     Align sequence using pairwise PCA with pre-computed basis from Stage 1.

    #     Args:
    #         timepoints: List of timepoint dictionaries
    #         initial_basis_trans: List of transformations from Stage 1 with 'basis' field
    #         normal_matching_percentile_threshold: Threshold for normal matching

    #     Returns:
    #         List of transformation dictionaries
    #     """
    #     # Extract basis matrices from Stage 1 transformations
    #     initial_basis = [trans["basis"] for trans in initial_basis_trans]

    #     # Call pairwise PCA alignment with pre-computed basis
    #     # Note: align_sequence_pairwise_pca modifies timepoints in-place
    #     _, transformations = alignment.align_sequence_pairwise_pca(
    #         timepoints, normal_matching_percentile_threshold, initial_basis
    #     )

    #     return transformations

    # def align_leaf_sequence(
    #     self,
    #     leaf_timeseries,
    #     normal_matching_percentile_threshold=75,
    #     plot_pairwise_alignment=False,
    # ):
    #     """
    #     Align leaf sequence using sequential PCA (wrapper for alignment.align_sequence_pairwise_pca).

    #     Args:
    #         leaf_timeseries: Leaf timeseries dict from dataset
    #         normal_matching_percentile_threshold: Distance threshold for matching normals
    #         plot_pairwise_alignment: If True, plot pairwise alignment (for debugging)

    #     Returns:
    #         aligned_timepoints: List of aligned timepoint dicts
    #         transformations: List of transformation dicts
    #     """
    #     timepoints = leaf_timeseries["timepoints"]

    #     # Use alignment module function
    #     aligned_timepoints, transformations = alignment.align_sequence_pairwise_pca(
    #         timepoints, normal_matching_percentile_threshold
    #     )

    #     # Plot pairwise alignment (for sequential alignment, plot each pair i-1 -> i)
    #     if plot_pairwise_alignment:
    #         # Lazy import for visualization to avoid heavy dependencies
    #         from plant_shape_analysis.vis.plot_functions import (
    #             plot_pairwise_alignment_with_quivers,
    #         )

    #         for i in range(1, len(aligned_timepoints)):
    #             plot_pairwise_alignment_with_quivers(
    #                 aligned_timepoints, transformations, idx1=i - 1, idx2=i
    #             )

    #     return aligned_timepoints, transformations

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

                # Extract day from first non-None stage
                if day is None and "day" in trans_info:
                    day = trans_info["day"]

                # Skip stages that only affect normals (not geometry)
                # e.g., enforce_downward_normals only flips normal vectors
                if stage_name == "enforce_downward_normals":
                    # This stage doesn't change point positions, skip for composed matrix
                    continue

                R = trans_info["rotation"]
                t = trans_info["translation"]
                c = trans_info["center"]

                # Build 4x4 transformation matrix for this stage
                # Transform is: p' = (p - c) @ R.T + c + t
                # For row vectors: [p,1] @ M.T gives [p @ M[:3,:3].T + M[:3,3], 1]
                # We want: p @ M[:3,:3].T = p @ R.T, so M[:3,:3] = R (not R.T!)

                # T(-c): translate to origin
                T_neg_c = np.eye(4, dtype=np.float64)
                T_neg_c[:3, 3] = -c

                # R: rotation matrix for row vectors
                T_R = np.eye(4, dtype=np.float64)
                T_R[:3, :3] = R  # Store R directly (will be transposed when applied)

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

        This reverses all alignment stages (downward normals, PCA, align to Z, vertical shift)
        to place the leaf back in its original plant coordinate space.

        Args:
            leaf_timeseries: Leaf timeseries dict with aligned timepoints
            transformation_stages: Transformation stages from self.transformations[sequence_name]

        Returns:
            List of timepoint dicts with points in original (unaligned) space
        """
        # Compose all transformation stages (skips normal-only stages)
        composed = LeafSequencesDataset.compose_transformations(transformation_stages)

        original_timepoints = []
        for tp_idx, (tp, comp) in enumerate(
            zip(leaf_timeseries["timepoints"], composed)
        ):
            # Get inverse matrix (aligned -> original)
            inv_matrix = LeafSequencesDataset.invert_transformation(
                comp["composed_matrix"]
            )

            # Apply inverse to points
            original_points = LeafSequencesDataset.apply_transformation(
                tp["points"], inv_matrix
            )

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

    def flip_normals_for_timepoints(self, sequence_name, timepoint_indices):
        """
        Manually flip normals for specific timepoints in a leaf sequence.
        Useful for interactive correction when automatic downward-facing detection fails.

        Args:
            sequence_name: Unique leaf sequence name (e.g., 'maize_control_plant1_leaf1')
            timepoint_indices: List of timepoint indices to flip (0-based)

        Returns:
            Updated leaf timeseries dict
        """
        leaf_seq = self.get_timeseries_by_sequence_name(sequence_name)

        if leaf_seq is None:
            raise ValueError(f"Sequence '{sequence_name}' not found")

        for idx in timepoint_indices:
            if idx < 0 or idx >= len(leaf_seq["timepoints"]):
                print(f"Warning: Invalid timepoint index {idx}, skipping")
                continue

            tp = leaf_seq["timepoints"][idx]
            if tp.get("normals") is not None:
                tp["normals"] = -tp["normals"]
                print(
                    f"✓ Flipped normals for {sequence_name} timepoint {idx} (day {tp['day']})"
                )

        return leaf_seq

    def save_plant_with_all_leaves(self, plant_sequence_name, output_dir=None):
        """
        Save entire plant with ALL corrected leaf normals from current dataset state.
        Use this when you've corrected multiple leaves to save them all at once.

        Args:
            plant_sequence_name: Plant sequence name (e.g., 'maize_control_plant1')
            output_dir: Directory to save PLY files (default: overwrites v2 dataset)

        Returns:
            Path to saved files directory
        """
        # Get plant timeseries
        plant_ts = self.plant_dataset.get_timeseries_by_sequence_name(
            plant_sequence_name
        )
        if plant_ts is None:
            raise ValueError(f"Plant sequence '{plant_sequence_name}' not found")

        # Get all leaves for this plant
        all_leaves = self.get_timeseries_by_plant_sequence(plant_sequence_name)

        # Set output directory - default to overwriting the source
        if output_dir is None:
            output_dir = self.plant_dataset.sparse_path.parent
        else:
            output_dir = Path(output_dir)

        # Process each timepoint
        saved_count = 0
        for timepoint in plant_ts["timepoints"]:
            day = timepoint["day"]
            points = timepoint["points"]
            labels = timepoint["labels"]
            leaf_tip_idxs = timepoint["leaf_tip_idxs"]

            # Get or initialize normals
            if "normals" not in timepoint or timepoint["normals"] is None:
                normals = np.zeros_like(points)
            else:
                normals = timepoint["normals"].copy()

            # Update normals for ALL leaves in this plant
            for leaf_seq in all_leaves:
                leaf_id = leaf_seq["leaf_id"]

                # Find matching leaf timepoint
                leaf_tp = None
                leaf_tp_idx = None
                for idx, ltp in enumerate(leaf_seq["timepoints"]):
                    if ltp["day"] == day:
                        leaf_tp = ltp
                        leaf_tp_idx = idx
                        break

                if leaf_tp is None or leaf_tp.get("normals") is None:
                    continue

                # Get leaf normals in aligned space
                leaf_normals_aligned = leaf_tp["normals"]

                # Transform normals back to plant-aligned coordinate system
                if leaf_seq.get("transformation_stages") is not None:
                    composed_transforms = LeafSequencesDataset.compose_transformations(
                        leaf_seq["transformation_stages"]
                    )

                    if leaf_tp_idx < len(composed_transforms):
                        # Get inverse transformation matrix
                        composed_matrix = composed_transforms[leaf_tp_idx][
                            "composed_matrix"
                        ]
                        inv_matrix = LeafSequencesDataset.invert_transformation(
                            composed_matrix
                        )

                        # Extract rotation part only (upper-left 3x3)
                        R_inv = inv_matrix[:3, :3]

                        # Apply inverse rotation to normals
                        leaf_normals_plant_space = leaf_normals_aligned @ R_inv.T
                    else:
                        leaf_normals_plant_space = leaf_normals_aligned
                else:
                    leaf_normals_plant_space = leaf_normals_aligned

                # Get mask for this leaf in plant point cloud
                leaf_mask = labels == leaf_id

                # Assign corrected normals from leaf to plant
                if np.sum(leaf_mask) == len(leaf_normals_plant_space):
                    normals[leaf_mask] = leaf_normals_plant_space

            # Save as PLY
            file_path = timepoint["file_path"]
            crop_name = file_path.parent.name
            filename = file_path.stem + ".ply"

            crop_dir = output_dir / "gt_corrected_v2" / crop_name
            crop_dir.mkdir(parents=True, exist_ok=True)

            # Create tensor-based point cloud
            pcd = o3d.t.geometry.PointCloud(points)
            pcd.point["organ_label"] = o3d.core.Tensor(
                labels.reshape(-1, 1), dtype=o3d.core.Dtype.Int32
            )
            pcd.point["normals"] = o3d.core.Tensor(
                normals.astype(np.float32), dtype=o3d.core.Dtype.Float32
            )

            # Add leaf tips
            leaf_tip_mask = np.zeros(len(points), dtype=np.uint8)
            if len(leaf_tip_idxs) > 0:
                leaf_tip_mask[leaf_tip_idxs] = 1
            pcd.point["is_leaf_tip"] = o3d.core.Tensor(
                leaf_tip_mask.reshape(-1, 1), dtype=o3d.core.Dtype.UInt8
            )

            # Save
            output_path = crop_dir / filename
            o3d.t.io.write_point_cloud(str(output_path), pcd)
            saved_count += 1

        output_path = output_dir / "gt_corrected_v2"
        print(f"✓ Saved plant '{plant_sequence_name}' with all corrected leaves")
        print(f"  {saved_count} timepoints written to: {output_path}")
        return output_path

    def __len__(self):
        return len(self.leaf_timeseries)

    def __getitem__(self, idx):
        return self.leaf_timeseries[idx]


if __name__ == "__main__":

    dataset_path = Path("data/TrackPlant3D/versions")

    # # Example usage for PlantSequencesDataset with alignment
    # print("Creating plant dataset...")
    # plant_dataset = PlantSequencesDataset(dataset_path)

    # # Visualize some plant sequences
    # from plant_shape_analysis.vis.plot_functions import visualize_plant_sequence

    # plant_sequences = [
    #     "maize_control_plant2",
    #     "tomato2_control_plant2",
    # ]

    # for i, seq in enumerate(plant_sequences):
    #     sample = plant_dataset.get_timeseries_by_sequence_name(seq)
    #     print(f"Visualizing plant sequence {seq} ({i+1}/{len(plant_sequences)})...")
    #     visualize_plant_sequence(
    #         sample,
    #         dense_points=False,
    #         color_by_organ=True,
    #         show_leaf_tips=True,
    #         spacing=100.0,
    #     )

    # Example usage for LeafSequencesDataset
    print("Creating leaf dataset with PCA alignment...")
    leaf_dataset = LeafSequencesDataset(dataset_path, apply_alignment=True)

    print("Visualizing some leaf sequences...")

    from plant_shape_analysis.vis.plot_functions import visualize_leaf_sequence

    sequences = [
        "maize_control_plant2_leaf2",
        "tomato2_control_plant2_leaf1",
        "tomato2_control_plant3_leaf2",
    ]

    for i, seq in enumerate(sequences):
        sample = leaf_dataset.get_timeseries_by_sequence_name(seq)
        print(f"Visualizing leaf sequence {seq} ({i+1}/{len(sequences)})...")
        visualize_leaf_sequence(sample)
