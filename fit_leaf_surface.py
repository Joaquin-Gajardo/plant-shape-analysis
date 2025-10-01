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
    data_path: str = "data/TrackPlant3D",
    sequence_name: str = None,
    timepoint: int = None,
    with_normals: bool = True,
    visualize: bool = True,
    sampling_strategy: str = "mixed",
    dense_points: bool = False,
):
    """Get a leaf dataset with the fixed PointCloudSiren class"""

    # We just want a single leaf so to save time we don't estimate normals of all leaves or align thems
    leaf_sequences = LeafSequencesDataset(
        data_path, estimate_normals=False, apply_pca_alignment=False
    )

    # Get leaf timeseries by sequence name
    if sequence_name is None:
        # If no sequence name provided, pick the first one
        leaf_timeseries = leaf_sequences[0]
    else:
        # Look up by sequence name
        leaf_timeseries = leaf_sequences.get_timeseries_by_sequence_name(sequence_name)
        if leaf_timeseries is None:
            raise ValueError(f"Sequence name '{sequence_name}' not found in dataset")

    if timepoint is None:
        timepoint = 0  # Use first timepoint if not specified

    points_key = "dense_points" if dense_points else "points"
    point_cloud = leaf_timeseries["timepoints"][timepoint][points_key]

    if point_cloud is None:
        raise ValueError(
            f"Leaf sequence {leaf_timeseries['sequence_name']}, timepoint {timepoint} does not have '{points_key}' points."
        )
    print(
        f"Loaded point cloud of sequence {leaf_timeseries['sequence_name']}, timepoint {timepoint} with shape: {point_cloud.shape}"
    )
    leaf_name = f"{leaf_timeseries['sequence_name']}_day{timepoint}"
    if dense_points:
        leaf_name += "-dense"

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
        sampling_strategy=sampling_strategy,
    )

    return dataset, leaf_name


def extract_mesh(
    checkpoint_path: str,
    out_mesh_path: str,
    model_kwargs: dict,
    N: int = 512,
    iso_level: float = 0.0,
    format: str = "ply",
):
    class SDFDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Siren(
                in_features=model_kwargs.get("in_features", 3),
                hidden_features=model_kwargs.get("hidden_features", 128),
                hidden_layers=model_kwargs.get("hidden_layers", 3),
                out_features=model_kwargs.get("out_features", 1),
                outermost_linear=model_kwargs.get("outermost_linear", True),
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
    data_path: str = "data/TrackPlant3D",
    results_folder="results/static_leaves/siren/dry_runs",
    sequence_name=None,
    timepoint=0,
    hidden_neurons=128,
    hidden_layers=3,
    lr=1e-4,
    epochs=100_000,
    sampling_strategy="mixed",
    resolution=512,
    logger="wandb",
    dense_points=False,
    w_sdf=1.0,
    w_inter=1.0,
    w_normal=1.0,
    w_grad=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get leaf point cloud to fit
    sample, leaf_name = get_leaf(
        data_path=data_path,
        sequence_name=sequence_name,
        timepoint=timepoint,
        with_normals=True,
        visualize=False,
        sampling_strategy=sampling_strategy,
        dense_points=dense_points,
    )

    # Model and optimizer
    model = Siren(
        in_features=3,
        hidden_features=hidden_neurons,
        hidden_layers=hidden_layers,
        out_features=1,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30,
    ).to(device)
    print(model)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    dataloader = DataLoader(
        sample, shuffle=True, batch_size=1, pin_memory=True, num_workers=0
    )

    # Define paths
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{now}_{leaf_name}-{model.model_name}-{hidden_layers}HL-{hidden_neurons}HU_{epochs}epochs_{sampling_strategy}"
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
                "sampling_strategy": sampling_strategy,
                "hidden_features": hidden_neurons,
                "hidden_layers": hidden_layers,
                "learning_rate": lr,
            },
        )
        wandb.watch(model, log="all", log_freq=100)

    # Training loop
    model.train()

    for epoch in range(epochs):
        total_train_loss = 0.0
        total_losses = {
            "sdf": 0.0,
            "inter": 0.0,
            "normal_constraint": 0.0,
            "grad_constraint": 0.0,
        }

        for batch in dataloader:
            x, y = batch
            model.zero_grad()

            output = model(x["coords"].to(device))
            losses = sdf_loss(output, y["sdf"].to(device), y["normals"].to(device))

            weighted_losses = {
                "sdf": losses["sdf"] * w_sdf,
                "inter": losses["inter"] * w_inter,
                "normal_constraint": losses["normal_constraint"] * w_normal,
                "grad_constraint": losses["grad_constraint"] * w_grad,
            }
            train_loss = sum(weighted_losses.values())

            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

            # For logging individual losses
            for k in total_losses:
                total_losses[k] += weighted_losses[k].item()

        if epoch % 100 == 0 and epoch > 0:
            avg_loss = total_train_loss / len(dataloader)
            avg_losses = {k: total_losses[k] / len(dataloader) for k in total_losses}
            print(f"Epoch {epoch}/{epochs}, training loss: {avg_loss}")
            print("  Loss breakdown:", {k: round(v, 4) for k, v in avg_losses.items()})
            if logger == "tensorboard" and writer:
                writer.add_scalar("total_train_loss", avg_loss, epoch)
                for k, v in avg_losses.items():
                    writer.add_scalar(f"loss/{k}", v, epoch)
            elif logger == "wandb":
                log_data = {"total_train_loss": avg_loss, "epoch": epoch}
                for k, v in avg_losses.items():
                    log_data[f"loss/{k}"] = v

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
                        model_kwargs={
                            "hidden_features": hidden_neurons,
                            "hidden_layers": hidden_layers,
                        },
                        N=128,
                        iso_level=0.0,
                        format="obj",
                    )

                    if temp_mesh_path.exists() and logger == "wandb":
                        log_data[f"mesh"] = wandb.Object3D(
                            str(temp_mesh_path), caption=f"Mesh at epoch {epoch}"
                        )

                    # Clean up temporary checkpoint
                    temp_checkpoint.unlink(missing_ok=True)
                    model.train()

                if logger == "wandb":
                    wandb.log(log_data)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete. Model saved to {checkpoint_path}")

    # Extract mesh
    print(f"Extracting mesh with marching cubes...")
    extract_mesh(
        checkpoint_path,
        out_mesh_path=experiment_path / f"mesh_res{resolution}.ply",
        model_kwargs={
            "hidden_features": hidden_neurons,
            "hidden_layers": hidden_layers,
        },
        N=resolution,
        format="ply",
    )

    # Log final mesh to wandb
    if logger == "wandb":
        wandb.finish()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p", "--dataset_path", type=str, default="data/TrackPlant3D")
    parser.add_argument(
        "--results_folder", type=str, default="results/static_leaves/siren/dry_runs"
    )
    parser.add_argument(
        "-s",
        "--sequence_name",
        type=str,
        default=None,
        help="Leaf sequence name (e.g., 'tomato2_control_plant2_leaf1')",
    )
    parser.add_argument(
        "-t", "--timepoint", type=int, default=0, help="Timepoint index"
    )
    parser.add_argument(
        "--epochs", type=int, default=100_000, help="Number of training epochs"
    )
    parser.add_argument(
        "--hidden_neurons",
        type=int,
        default=128,
        help="Number of hidden neurons per layer in MLP",
    )
    parser.add_argument(
        "--hidden_layers", type=int, default=3, help="Number of hidden layers in MLP"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="mixed",
        help="Sampling strategy for off-surface points. Uniform follow SIREN, mixed follows IGR (Gropp et al. 2020) sampling, and prasad follows Prasad, 2022 (https://openreview.net/forum?id=F4eTwol9qne).",
        choices=["mixed", "uniform", "prasad"],
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--logger", type=str, default="wandb", choices=["tensorboard", "wandb"]
    )
    parser.add_argument(
        "--dense_points", action="store_true", help="Use dense points if available"
    )
    parser.add_argument("--w_sdf", type=float, default=3e3, help="Weight for sdf loss")
    parser.add_argument(
        "--w_inter", type=float, default=1e2, help="Weight for inter loss"
    )
    parser.add_argument(
        "--w_normal", type=float, default=1e2, help="Weight for normal constraint loss"
    )
    parser.add_argument(
        "--w_grad", type=float, default=5e1, help="Weight for grad constraint loss"
    )

    args = parser.parse_args()

    main(
        data_path=args.dataset_path,
        results_folder=args.results_folder,
        sequence_name=args.sequence_name,
        timepoint=args.timepoint,
        hidden_neurons=args.hidden_neurons,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        sampling_strategy=args.sampling_strategy,
        resolution=args.resolution,
        logger=args.logger,
        dense_points=args.dense_points,
        w_sdf=args.w_sdf,
        w_inter=args.w_inter,
        w_normal=args.w_normal,
        w_grad=args.w_grad,
    )
