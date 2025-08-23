import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from urllib import parse

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from plant_shape_analysis.dataloaders.trackplant3D import LeafSequencesDataset
from plant_shape_analysis.models.siren import PointCloudSiren, Siren, sdf_loss
from plant_shape_analysis.utils import sdf_meshing
from plant_shape_analysis.vis.plot_functions import visualize_point_cloud


def get_leaf(
    seq_idx: int = None,
    timepoint: int = None,
    with_normals: bool = True,
    visualize: bool = True,
    off_surface_points: int = 10_000,
    sampling_strategy: str = "mixed",
):
    """Get a leaf dataset with the fixed PointCloudSiren class"""

    # We just want a single leaf so we don't estimate normals for all to save time
    leaf_sequences = LeafSequencesDataset("data/TrackPlant3D", estimate_normals=False)

    if seq_idx is None:
        seq_idx = random.randint(0, len(leaf_sequences) - 1)
    if timepoint is None:
        timepoint = random.randint(0, len(leaf_sequences[seq_idx]["timepoints"]) - 1)

    point_cloud = leaf_sequences[seq_idx]["timepoints"][timepoint]["points"]
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
    checkpoint_path: str, out_mesh_path: str, N: int = 512, iso_level: float = 0.0
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
    sdf_meshing.create_mesh(sdf_decoder, out_mesh_path, N=N, iso_level=iso_level)


def main(
    results_folder="results/static_leaves/siren/dry_runs",
    num_epochs=100_000,
    off_surface_points=10_000,
    sampling_strategy="mixed",
    resolution=512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get leaf point cloud to fit
    sample, leaf_name = get_leaf(
        seq_idx=0,
        timepoint=0,
        with_normals=True,
        visualize=False,
        off_surface_points=off_surface_points,
        sampling_strategy=sampling_strategy,
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
    experiment_name = f"{now}_{leaf_name}-{model.model_name}_{off_surface_points}offpoints_{sampling_strategy}"
    experiment_path = Path(results_folder) / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_path / (experiment_name + ".pth")

    # Train loop
    writer = SummaryWriter(log_dir=experiment_path)
    model.train()

    for epoch in range(num_epochs):
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

        if epoch % 100 == 0:
            avg_loss = total_train_loss / len(dataloader)
            print(f"Epoch {epoch}/{num_epochs}, training loss: {avg_loss}")
            writer.add_scalar("total_train_loss", avg_loss, epoch)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete. Model saved to {checkpoint_path}")

    # Extract mesh
    print(f"Extracting mesh with marching cubes...")
    extract_mesh(
        checkpoint_path,
        out_mesh_path=experiment_path / f"mesh_res{resolution}.ply",
        N=resolution,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--results_folder", type=str, default="results/static_leaves/siren/dry_runs"
    )
    parser.add_argument("--num_epochs", type=int, default=100_000)
    parser.add_argument("--sampling_strategy", type=str, default="mixed")
    parser.add_argument("--off_surface_points", type=int, default=10_000)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    main(
        results_folder=args.results_folder,
        num_epochs=args.num_epochs,
        sampling_strategy=args.sampling_strategy,
        off_surface_points=args.off_surface_points,
        resolution=args.resolution,
    )
