import numpy as np
import open3d as o3d

from plant_shape_analysis.dataloaders.trackplant3D import LeafSequencesDataset
from plant_shape_analysis.vis.plot_functions import visualize_point_cloud


def get_leaf(seq_idx=0, timepoint=0, with_normals=True, visualize=True):
    # We just want a single leaf so we don't estimate normals for all to save time
    dataset = LeafSequencesDataset("data/TrackPlant3D", estimate_normals=False)
    point_cloud = dataset[seq_idx]["timepoints"][timepoint]["points"]
    if with_normals:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        point_cloud = np.hstack([point_cloud, np.asarray(pcd.normals)])

    print(f"Loaded point cloud with shape: {point_cloud.shape}")

    if visualize:
        visualize_point_cloud(point_cloud)


def main():
    get_leaf(seq_idx=0, timepoint=0, with_normals=True, visualize=True)


if __name__ == "__main__":
    main()
