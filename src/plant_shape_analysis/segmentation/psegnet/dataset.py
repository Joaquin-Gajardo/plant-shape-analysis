"""
PyTorch Dataset for PSegNet training on PlantSequencesDataset v2.

Each sample is one point-cloud frame (single timepoint of one plant sequence),
downsampled to n_points via FPS and PointNet-normalised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset
from plant_shape_analysis.segmentation.psegnet.inference import _fps_numpy, _pointnet_norm

# Held-out sequences used only for final evaluation — never seen during training/val
TEST_SEQUENCES: list[str] = [
    "maize_control_plant2",
    "maize_control_plant3",
    "sorghum_control_plant2",
    "sorghum_highlight_plant2",
    "tobacco_control_plant1",
    "tobacco_shade_plant3",
    "tomato1_heat_plant3",
    "tomato1_shade_plant1",
]


def _species_from_name(seq_name: str) -> str:
    """Extract crop species prefix from sequence name, e.g. 'tomato1' → 'tomato'."""
    prefix = seq_name.split("_")[0]
    # Normalise tomato1/tomato2 → tomato so stratification treats them as one species
    return prefix.rstrip("0123456789")


class PSegNetDataset(Dataset):
    """
    Flat dataset of individual point-cloud frames for PSegNet training.

    Items are drawn from all non-test sequences of PlantSequencesDataset v2,
    one item per (sequence, timepoint) pair. A stratified 85/15 train/val split
    is performed at the frame level, with species as the stratification key.

    Args:
        dataset_path:   Root path passed to PlantSequencesDataset
                        (e.g. 'data/TrackPlant3D/versions').
        split:          'train', 'val', or 'all'.
        n_points:       Number of points per sample after FPS downsampling.
        val_fraction:   Fraction of frames held out for validation.
        random_state:   RNG seed for the train/val split (fixed for reproducibility).
        split_save_path: Optional path to save/load the split indices as JSON.
                         If the file exists the split is loaded from disk; otherwise
                         it is computed and saved. Pass None to skip persistence.
        augment:        Apply random Z-rotation and jitter (forced off for 'val').
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        n_points: int = 4096,
        n_repeats: int = 10,
        val_fraction: float = 0.15,
        random_state: int = 42,
        split_save_path: Optional[str | Path] = None,
        augment: bool = True,
    ):
        assert split in ("train", "val", "all"), f"split must be train/val/all, got {split!r}"
        self.split = split
        self.n_points = n_points
        self.n_repeats = n_repeats
        self.augment = augment and (split == "train")

        # Load base dataset (v2, Z-up, orientation already corrected)
        base = PlantSequencesDataset(dataset_path=str(dataset_path), version="v2")

        # Flatten to individual (points, labels) frames, skip test sequences
        frames: list[tuple[np.ndarray, np.ndarray]] = []
        species_per_frame: list[str] = []
        for seq in base.plant_timeseries:
            if seq["sequence_name"] in TEST_SEQUENCES:
                continue
            sp = _species_from_name(seq["sequence_name"])
            for tp in seq["timepoints"]:
                frames.append((tp["points"], tp["labels"]))
                species_per_frame.append(sp)

        # Build or load train/val split
        indices = np.arange(len(frames))
        train_idx, val_idx = self._get_split(
            indices, species_per_frame, val_fraction, random_state, split_save_path
        )

        if split == "train":
            selected = train_idx
        elif split == "val":
            selected = val_idx
        else:
            selected = indices

        self.frames = [frames[i] for i in selected]

    # ------------------------------------------------------------------
    def _get_split(
        self,
        indices: np.ndarray,
        species: list[str],
        val_fraction: float,
        random_state: int,
        save_path: Optional[str | Path],
    ) -> tuple[np.ndarray, np.ndarray]:
        if save_path is not None and Path(save_path).exists():
            with open(save_path) as f:
                d = json.load(f)
            return np.array(d["train"]), np.array(d["val"])

        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            stratify=species,
            random_state=random_state,
        )
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"train": train_idx.tolist(), "val": val_idx.tolist()}, f)
        return train_idx, val_idx

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frames) * self.n_repeats

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        points, inst_labels = self.frames[idx % len(self.frames)]  # (N, 3), (N,)

        # FPS downsample — random starting point acts as mild augmentation
        # NOTE: _fps_numpy is O(N²); use num_workers≥4 in the DataLoader
        ds_idx = _fps_numpy(points, self.n_points) if len(points) > self.n_points else np.arange(len(points))
        pts = points[ds_idx].copy()
        lbl = inst_labels[ds_idx].copy()

        # PointNet normalisation: centre + scale to unit sphere
        pts = _pointnet_norm(pts).astype(np.float32)

        if self.augment:
            pts = self._augment(pts)

        sem_labels = (lbl > 0).astype(np.int64)   # 0=stem, 1=leaf

        return {
            "points":     torch.from_numpy(pts),              # (n_points, 3)
            "sem_labels": torch.from_numpy(sem_labels),        # (n_points,)
            "inst_labels": torch.from_numpy(lbl.astype(np.int64)),  # (n_points,)
        }

    @staticmethod
    def _augment(pts: np.ndarray) -> np.ndarray:
        """Random Z-axis rotation + Gaussian jitter."""
        # Z-rotation
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ R.T
        # Jitter
        pts += np.clip(np.random.normal(0, 0.01, pts.shape), -0.05, 0.05).astype(np.float32)
        return pts
