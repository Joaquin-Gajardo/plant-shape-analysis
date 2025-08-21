import random

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
        point_cloud, len(point_cloud), off_surface_points=off_surface_points
    )

    return dataset, leaf_name


def extract_mesh(
    checkpoint_path: str, mesh_filename: str, N: int = 512, iso_level: float = 0.0
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
        sdf_decoder, f"{mesh_filename}.ply", N=N, iso_level=iso_level
    )


def main(
    num_epochs=100_000,
    checkpoint_path="siren_model.pth",
    off_surface_points=10_000,
    resolution=512,
):
    # Get random leaf, instanciate model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, leaf_name = get_leaf(
        seq_idx=0,
        timepoint=0,
        with_normals=True,
        visualize=False,
        off_surface_points=off_surface_points,
    )
    print("Dataset size: ", len(dataset))

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
        dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0
    )

    # Train loop
    writer = SummaryWriter("logs")
    model.train()

    for epoch in range(num_epochs):
        total_train_loss = 0.0

        for batch in dataloader:
            x, y = batch
            model.zero_grad()

            output = model(x["coords"].to(device))
            losses = sdf_loss(output, y["sdf"].to(device), y["normals"].to(device))

            train_loss = 0.0
            for loss_name, loss in losses.items():
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
        mesh_filename=f"{leaf_name}-{model.model_name}_res{resolution}",
        N=resolution,
    )


if __name__ == "__main__":
    main()
