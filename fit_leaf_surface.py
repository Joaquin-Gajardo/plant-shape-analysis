import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from plant_shape_analysis.dataloaders.trackplant3D import LeafSequencesDataset
from plant_shape_analysis.models.siren import PointCloudSiren, Siren, sdf_loss
from plant_shape_analysis.utils import sdf_meshing
from plant_shape_analysis.vis.plot_functions import (
    create_sdf_cross_section,
    visualize_point_cloud,
)


def get_leaf(
    seq_idx: int = None,
    timepoint: int = None,
    with_normals: bool = True,
    visualize: bool = True,
    off_surface_points: Optional[int] = None,
    sampling_strategy: str = "mixed",
    dense_points: bool = False,
):
    """Get a leaf dataset with the fixed PointCloudSiren class"""

    # We just want a single leaf so we don't estimate normals for all to save time
    leaf_sequences = LeafSequencesDataset("data/TrackPlant3D", estimate_normals=False)

    if seq_idx is None:
        seq_idx = random.randint(0, len(leaf_sequences) - 1)
    if timepoint is None:
        timepoint = random.randint(0, len(leaf_sequences[seq_idx]["timepoints"]) - 1)

    points_key = "dense_points" if dense_points else "points"
    point_cloud = leaf_sequences[seq_idx]["timepoints"][timepoint][points_key]

    if point_cloud is None:
        raise ValueError(
            f"Leaf sequence {seq_idx}, timepoint {timepoint} does not have '{points_key}' points."
        )
    print(
        f"Loaded point cloud of leaf sequence {seq_idx}, timepoint {timepoint} with shape: {point_cloud.shape}"
    )
    leaf_name = f"{leaf_sequences[seq_idx]['sequence_name']}_leaf{int(leaf_sequences[seq_idx]['leaf_id'])}_day{timepoint}"

    if with_normals:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        point_cloud = np.hstack([point_cloud, np.asarray(pcd.normals)])

    if visualize:
        visualize_point_cloud(point_cloud)

    dataset = PointCloudSiren(
        point_cloud,
        len(point_cloud),
        off_surface_points=off_surface_points,
        sampling_strategy=sampling_strategy,
    )

    return dataset, leaf_name


def extract_mesh(
    checkpoint_path: str,
    out_mesh_path: str,
    N: int = 512,
    iso_level: float = 0.0,
    format: str = "ply",
):
    class SDFDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Siren(
                in_features=3,
                hidden_features=128,
                hidden_layers=3,
                out_features=1,
                outermost_linear=True,
                first_omega_0=30,
                hidden_omega_0=30,
            )
            if torch.cuda.is_available():
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu")
                )
                self.model.cuda()

            self.model.eval()

        def forward(self, coords):
            return self.model(coords)["model_out"]

    sdf_decoder = SDFDecoder()
    sdf_meshing.create_mesh(
        sdf_decoder, out_mesh_path, N=N, iso_level=iso_level, format=format
    )


def main(
    results_folder="results/static_leaves/siren/dry_runs",
    leaf_sequence=0,
    timepoint=0,
    epochs=100_000,
    off_surface_points=None,
    sampling_strategy="mixed",
    resolution=512,
    logger="wandb",
    dense_points=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get leaf point cloud to fit
    sample, leaf_name = get_leaf(
        seq_idx=leaf_sequence,
        timepoint=timepoint,
        with_normals=True,
        visualize=False,
        off_surface_points=off_surface_points,
        sampling_strategy=sampling_strategy,
        dense_points=dense_points,
    )

    # Model and optimizer
    model = Siren(
        in_features=3,
        hidden_features=128,
        hidden_layers=3,
        out_features=1,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30,
    ).to(device)
    print(model)
    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())

    dataloader = DataLoader(
        sample, shuffle=True, batch_size=1, pin_memory=True, num_workers=0
    )

    # Define paths
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{now}_{leaf_name}-{model.model_name}_{epochs}epochs_{off_surface_points}offpoints_{sampling_strategy}"
    experiment_path = Path(results_folder) / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_path / (experiment_name + ".pth")

    # Initialize logging
    writer = None
    if logger == "tensorboard":
        writer = SummaryWriter(log_dir=experiment_path)
    elif logger == "wandb":
        wandb.init(
            project="plant-shape-analysis",
            name=experiment_name,
            dir=experiment_path,
            config={
                "leaf_name": leaf_name,
                "model": model.model_name,
                "epochs": epochs,
                "off_surface_points": off_surface_points,
                "sampling_strategy": sampling_strategy,
                "hidden_features": 128,
                "hidden_layers": 3,
                "learning_rate": 1e-4,
            },
        )

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0

        for batch in dataloader:
            x, y = batch
            model.zero_grad()

            output = model(x["coords"].to(device))
            losses = sdf_loss(output, y["sdf"].to(device), y["normals"].to(device))

            train_loss = 0.0
            for _, loss in losses.items():
                train_loss += loss.mean()

            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

        if epoch % 100 == 0 and epoch > 0:
            avg_loss = total_train_loss / len(dataloader)
            print(f"Epoch {epoch}/{epochs}, training loss: {avg_loss}")
            if logger == "tensorboard" and writer:
                writer.add_scalar("total_train_loss", avg_loss, epoch)
            elif logger == "wandb":
                log_data = {"total_train_loss": avg_loss, "epoch": epoch}

                # Log SDF cross-sections every 5000 epochs
                if epoch % 5000 == 0 and epoch > 0:
                    model.eval()
                    with torch.no_grad():
                        # Create wrapper that extracts SDF values
                        def sdf_decoder(coords):
                            return model(coords)["model_out"]

                        fig = create_sdf_cross_section(
                            sdf_decoder, device, slice_position=0.0, resolution=128
                        )
                        log_data["sdf_cross_section"] = wandb.Image(fig)
                        plt.close(fig)
                    model.train()

                # Log intermediate mesh every 20000 epochs
                if epoch % 20000 == 0 and epoch > 0:
                    model.eval()

                    # Temporary checkpoint and mesh
                    temp_checkpoint = experiment_path / f"temp_epoch_{epoch}.pth"
                    temp_mesh_path = experiment_path / f"mesh_epoch_{epoch}_res128.obj"

                    torch.save(model.state_dict(), temp_checkpoint)
                    extract_mesh(
                        temp_checkpoint,
                        temp_mesh_path,
                        N=128,
                        iso_level=0.0,
                        format="obj",
                    )

                    if temp_mesh_path.exists():
                        log_data[f"mesh"] = wandb.Object3D(
                            str(temp_mesh_path), caption=f"Mesh at epoch {epoch}"
                        )

                    # Clean up temporary checkpoint
                    temp_checkpoint.unlink(missing_ok=True)
                    model.train()

                wandb.log(log_data)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete. Model saved to {checkpoint_path}")

    # Extract mesh
    print(f"Extracting mesh with marching cubes...")
    extract_mesh(
        checkpoint_path,
        out_mesh_path=experiment_path / f"mesh_res{resolution}.ply",
        N=resolution,
        format="ply",
    )

    # Log final mesh to wandb
    if logger == "wandb":
        wandb.finish()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--results_folder", type=str, default="results/static_leaves/siren/dry_runs"
    )
    parser.add_argument(
        "--leaf_sequence", type=int, default=0, help="Leaf sequence index"
    )
    parser.add_argument("--timepoint", type=int, default=0, help="Timepoint index")
    parser.add_argument(
        "--epochs", type=int, default=100_000, help="Number of training epochs"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="mixed",
        help="Sampling strategy for off-surface points. Mixed follows IGR with uniform sampling + gaussian sampling for near-surface points.",
        choices=["uniform", "mixed"],
    )
    parser.add_argument(
        "--off_surface_points",
        type=int,
        default=None,
        help="Number of off-surface points to sample. If using mixed strategy, this is the total number of points sampled (half from each).",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )
    parser.add_argument(
        "--dense_points", action="store_true", help="Use dense points if available"
    )
    args = parser.parse_args()

    main(
        results_folder=args.results_folder,
        leaf_sequence=args.leaf_sequence,
        timepoint=args.timepoint,
        epochs=args.epochs,
        sampling_strategy=args.sampling_strategy,
        off_surface_points=args.off_surface_points,
        resolution=args.resolution,
        logger=args.logger,
        dense_points=args.dense_points,
    )
