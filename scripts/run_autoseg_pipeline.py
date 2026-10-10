"""
Create automatic organ labels for the v2 point clouds (what `v2-autoseg` ships).

For each plant sequence in the v2 dataset:
  1. Run PSegNet inference on each timepoint to get per-point organ labels.
  2. Run the TrackPlant3D tracking pipeline to assign temporally consistent IDs.
  3. Write a new PLY file (copy of the v2 PLY) with an extra `predicted_organ_label`
     int32 scalar field.

Output directory: <dataset-path>/v2/predicted_labels/{crop}/ (or --output-dir).
Load the result with `PlantSequencesDataset(..., version="v2-autoseg",
sparse_dir=<output dir>)`; `version="v2-autoseg"` alone reads the deposited labels.

Usage (the recipe behind the deposited labels is in the README, *Automatic organ labels*):
    python scripts/run_autoseg_pipeline.py \\
        --checkpoint outputs/psegnet_trackplant3d/<run>/checkpoints/best_model.pth \\
        --num-classes 2

Notes:
  - PSegNet semantic class 0 = stem.
  - The original PSegNet checkpoint (model_epoch199.pth) needs `--num-classes 6
    --prerotation`: it was trained on Y-up data, and on tomato/tobacco/sorghum only, so
    maize is out of distribution for it. Models retrained with
    scripts/train_psegnet.py need neither.
  - Inference is not deterministic: reruns differ on about 1% of points.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from plant_shape_analysis.dataloaders.point_cloud_utils import needs_orientation_correction
from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset
from plant_shape_analysis.segmentation import load_psegnet, predict_organ_labels
from plant_shape_analysis.tracking import run_tracking_pipeline

# Inverse of the Z-up correction applied to sorghum/tobacco/tomato1 in v2 PLYs.
# PSegNet was trained on Y-up data, so we undo the correction before inference.
_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

# Trained with the upstream PSegNet code; pass --checkpoint or set PSEGNET_CHECKPOINT.
DEFAULT_CHECKPOINT = os.environ.get(
    "PSEGNET_CHECKPOINT",
    "models/checkpoints/psegnet_model_epoch199.pth",
)
ALL_SPECIES = ["maize", "sorghum", "tobacco", "tomato"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate v3 predicted-organ-label dataset")
    p.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to PSegNet checkpoint (.pth)",
    )
    p.add_argument(
        "--dataset-path",
        default="data/TrackPlant3D/versions",
        help="Root path passed to PlantSequencesDataset (default: data/TrackPlant3D/versions)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available)",
    )
    p.add_argument(
        "--species",
        nargs="+",
        default=ALL_SPECIES,
        help=f"Species to process (default: {ALL_SPECIES})",
    )
    p.add_argument(
        "--n-input-points",
        type=int,
        default=4096,
        help="Points fed into PSegNet per timepoint (default: 4096)",
    )
    p.add_argument(
        "--bandwidth",
        type=float,
        default=0.6,
        help="MeanShift bandwidth for instance clustering (default: 0.6)",
    )
    p.add_argument(
        "--stem-class",
        type=int,
        default=0,
        help="PSegNet semantic class index for the stem organ (default: 0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for 3DEPS downsampling (default: 0)",
    )
    p.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Run only on these specific sequence names (e.g. --sequences maize_control_plant1 maize_control_plant2). Overrides --species.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: <dataset-path>/v2/predicted_labels/)",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Semantic classes the checkpoint was trained with (default: 2 for retrained model). Pass 6 when using the original checkpoint (model_epoch199.pth).",
    )
    p.add_argument(
        "--prerotation",
        action="store_true",
        help="Apply Z-up→Y-up pre-rotation before inference. Only needed for the original checkpoint (model_epoch199.pth), which was trained on Y-up data. The retrained model does not require this.",
    )
    return p.parse_args()


def process_sequence(
    sequence_name: str,
    sequence_data: list[dict],
    model,
    device: str,
    output_dir: Path,
    n_input_points: int,
    bandwidth: float,
    stem_class: int,
    seed: int,
    apply_prerotation: bool = True,
):
    """Run the full pipeline for one plant sequence and save PLY files."""
    T = len(sequence_data)
    print(f"  [{sequence_name}] {T} timepoints")

    # Old checkpoint was trained on Y-up; undo the Z-up correction for those species.
    # Retrained checkpoint is trained on Z-up data — skip pre-rotation entirely.
    pre_rotation = (
        _R_ZUP_TO_YUP if (apply_prerotation and needs_orientation_correction(sequence_name))
        else None
    )

    # ── 1. PSegNet inference per timepoint ────────────────────────────────────
    psegnet_labels: list[np.ndarray] = []
    for t, tp in enumerate(sequence_data):
        labels = predict_organ_labels(
            points=tp["points"],
            model=model,
            device=device,
            n_input_points=n_input_points,
            bandwidth=bandwidth,
            stem_semantic_class=stem_class,
            pre_rotation=pre_rotation,
        )
        psegnet_labels.append(labels)
        n_unique = len(np.unique(labels))
        print(f"    t={t:02d} day={tp['day']:02d}: {tp['points'].shape[0]} pts, "
              f"{n_unique} predicted organs")

    # ── 2. TrackPlant3D tracking pipeline ────────────────────────────────────
    print(f"  [{sequence_name}] Running tracking pipeline...")
    tracked_labels = run_tracking_pipeline(
        sequence_points=[tp["points"] for tp in sequence_data],
        sequence_labels=psegnet_labels,
        seed=seed,
    )

    # ── 3. Save PLY files with predicted_organ_label field ───────────────────
    for t, (tp, pred_lab) in enumerate(zip(sequence_data, tracked_labels)):
        v2_path: Path = tp["file_path"]
        crop_name = v2_path.parent.name
        out_crop_dir = output_dir / crop_name
        out_crop_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_crop_dir / v2_path.name

        # Load original v2 PLY and add the predicted_organ_label field
        pcd = o3d.t.io.read_point_cloud(str(v2_path))
        pcd.point["predicted_organ_label"] = o3d.core.Tensor(
            pred_lab.astype(np.int32).reshape(-1, 1)
        )
        o3d.t.io.write_point_cloud(str(out_path), pcd)

    print(f"  [{sequence_name}] Saved {T} PLY files to {output_dir}")


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else dataset_path / "v2" / "predicted_labels"
    print(f"Output directory: {output_dir}")

    # Load v2 dataset (GT labels for file paths and point cloud coordinates)
    print("Loading PlantSequencesDataset v2...")
    dataset = PlantSequencesDataset(dataset_path, version="v2", use_ply=True, auto_download=False)

    # Load PSegNet model
    print(f"Loading PSegNet from {checkpoint_path} on {args.device}...")
    model = load_psegnet(checkpoint_path, device=args.device, num_classes=args.num_classes)

    # Process sequences by species
    sequences = dataset.get_sequence_names()
    total = len(sequences)
    processed = 0

    for seq_name in sequences:
        if args.sequences is not None:
            if seq_name not in args.sequences:
                continue
        elif not any(seq_name.startswith(sp) for sp in args.species):
            continue

        seq_data = dataset.get_sequence_data(seq_name)
        if not seq_data:
            print(f"  Skipping {seq_name}: no data found")
            continue

        try:
            process_sequence(
                sequence_name=seq_name,
                sequence_data=seq_data,
                model=model,
                device=args.device,
                output_dir=output_dir,
                n_input_points=args.n_input_points,
                bandwidth=args.bandwidth,
                stem_class=args.stem_class,
                seed=args.seed,
                apply_prerotation=args.prerotation,
            )
            processed += 1
        except Exception as e:
            print(f"  ERROR processing {seq_name}: {e}")
            raise

    print(f"\nDone. Processed {processed}/{total} sequences.")
    # version='v2-autoseg' alone reads the deposited labels, not these.
    print(
        f"Load with: PlantSequencesDataset('{args.dataset_path}', version='v2-autoseg', "
        f"sparse_dir='{output_dir.resolve()}')"
    )


if __name__ == "__main__":
    main()
