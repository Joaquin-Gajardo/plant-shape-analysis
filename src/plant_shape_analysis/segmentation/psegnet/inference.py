"""
PSegNet inference: predict per-point organ labels from a plant point cloud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift

from plant_shape_analysis.segmentation.psegnet.model import PSegNet


def _fps_numpy(points: np.ndarray, k: int) -> np.ndarray:
    """Greedy farthest-point sampling; returns indices of k selected points."""
    N = points.shape[0]
    if k >= N:
        return np.arange(N)
    selected = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf)
    farthest = int(np.random.randint(0, N))
    for i in range(k):
        selected[i] = farthest
        d = np.sum((points - points[farthest]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return selected


def _pointnet_norm(points: np.ndarray) -> np.ndarray:
    """Center and scale to unit sphere (same normalization used during training)."""
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    scale = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    return pts / scale


def load_psegnet(
    checkpoint_path: str | Path,
    device: str = "cuda",
    num_classes: int = 6,
) -> PSegNet:
    """
    Load a pre-trained PSegNet model.

    Args:
        checkpoint_path: Path to .pth checkpoint saved by the original training script.
        device: Torch device string ('cuda' or 'cpu').
        num_classes: Number of semantic classes (default 6, as in the original paper).

    Returns:
        PSegNet model in eval mode on the requested device.
    """
    model = PSegNet(num_classes=num_classes)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_organ_labels(
    points: np.ndarray,
    model: PSegNet,
    device: str = "cuda",
    n_input_points: int = 4096,
    bandwidth: float = 0.6,
    stem_semantic_class: int = 0,
    pre_rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Predict per-point organ labels for a plant point cloud.

    Runs PSegNet inference and converts outputs to the organ_label convention
    used by PlantSequencesDataset: 0 = stem, 1, 2, 3, ... = individual leaves.

    Args:
        points: (N, 3) point cloud in original coordinates.
        model: Loaded PSegNet model (from load_psegnet).
        device: Torch device string.
        n_input_points: Points to subsample before inference (model trained on 4096).
        bandwidth: MeanShift bandwidth for instance clustering of non-stem embeddings.
        stem_semantic_class: PSegNet semantic class index for the stem organ.
            Class 0 is the default (matches the training label convention where
            organ_label 0 = stem). Verify against your checkpoint if unsure.
        pre_rotation: Optional (3, 3) rotation matrix applied to points before
            inference. Use to undo orientation correction — PSegNet was trained
            on Y-up data, so pass the inverse rotation for species whose v2 PLYs
            were corrected to Z-up (sorghum, tobacco, tomato1).

    Returns:
        organ_labels: (N,) int64 array. 0 = stem, 1+ = individual leaf instances.
    """
    N = points.shape[0]

    # Apply optional pre-rotation (e.g. Z-up → Y-up for PSegNet trained on Y-up)
    pts = points @ pre_rotation.T if pre_rotation is not None else points

    # 1. FPS to n_input_points (keep coords consistent for back-projection)
    ds_idx = _fps_numpy(pts, n_input_points) if N > n_input_points else np.arange(N)
    ds_points = pts[ds_idx]

    # 2. PointNet normalization (same as training preprocessing)
    ds_norm = _pointnet_norm(ds_points).astype(np.float32)

    # 3. Forward pass
    pts_tensor = torch.from_numpy(ds_norm).unsqueeze(0).to(device)  # (1, n, 3)
    with torch.no_grad():
        pred_sem, pred_ins, _ = model(pts_tensor)

    # 4. Semantic argmax → per-downsampled-point class
    sem_labels = torch.argmax(pred_sem, dim=2).squeeze(0).cpu().numpy()   # (n,)

    # 5. Instance embeddings (n, 5)
    ins_embed = pred_ins.squeeze(0).cpu().numpy()

    # 6. MeanShift clustering on non-stem embeddings → instance IDs starting at 1
    organ_labels_ds = np.zeros(len(ds_points), dtype=np.int64)  # stem = 0
    non_stem_idx = np.where(sem_labels != stem_semantic_class)[0]
    if len(non_stem_idx) > 0:
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        ms.fit(ins_embed[non_stem_idx])
        organ_labels_ds[non_stem_idx] = ms.labels_ + 1  # leaves start at 1

    # 7. KNN back-propagation to all N original points (in same rotated space)
    tree = cKDTree(ds_points)
    _, nn_idx = tree.query(pts, k=1)
    return organ_labels_ds[nn_idx].astype(np.int64)
