"""From the DeepSDF repository https://github.com/facebookresearch/DeepSDF"""

#!/usr/bin/env python3

import time

import numpy as np
import plyfile
import skimage.measure
import torch
from tqdm import tqdm


def create_mesh(
    decoder, filename, N=256, max_batch=64**3, offset=None, scale=None, iso_level=0.0, format="ply", inverse_transform=None
):
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 4)

    print(f"Sampling SDF with {N**3} voxels...")

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0
    with tqdm(total=num_samples, desc="Sampling SDF") as pbar:
        while head < num_samples:
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3]
            if torch.cuda.is_available():
                sample_subset = sample_subset.cuda()

            with torch.no_grad():
                sdf_values = decoder(sample_subset).squeeze().detach().cpu()

            samples[head : min(head + max_batch, num_samples), 3] = sdf_values
            processed = min(max_batch, num_samples - head)
            head += processed
            pbar.update(processed)

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    # Run marching cubes to get mesh
    mesh_points, faces = extract_mesh_from_sdf(
        sdf_values.numpy(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
        iso_level=iso_level,
        inverse_transform=inverse_transform,
    )

    # Export to the specified format
    if format.lower() == "ply":
        save_mesh_as_ply(mesh_points, faces, filename)
    elif format.lower() == "obj":
        save_mesh_as_obj(mesh_points, faces, filename)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'ply' or 'obj'.")


def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    iso_level=0.0,
):
    """
    Convert sdf samples to .ply

    :param numpy_3d_sdf_tensor: a numpy.ndarray of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    verts, faces, normals, values = (
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0),
    )
    try:
        print(f"Running marching cubes...")
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, spacing=[voxel_size] * 3, level=iso_level
        )
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3))

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # Write PLY file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    if num_verts > 0:
        verts_tuple = np.zeros(
            (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        )

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        ply_data.write(ply_filename_out)
        print(
            f"Saved mesh with {num_verts} vertices and {num_faces} faces to {ply_filename_out}"
        )

    else:
        print(f"No mesh generated for {ply_filename_out}")


def extract_mesh_from_sdf(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
    iso_level=0.0,
    inverse_transform=None,
):
    """
    Extract mesh from SDF using marching cubes

    Args:
        inverse_transform: Can be either:
                          - A callable that takes mesh_points and returns transformed points
                          - A dict with keys 'center', 'coord_min', 'coord_max', 'scale' for backwards compatibility
                          - None to use legacy offset/scale behavior

    Returns:
        mesh_points: numpy array of vertices
        faces: numpy array of face indices
    """
    try:
        print(f"Running marching cubes...")
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, spacing=[voxel_size] * 3, level=iso_level
        )
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return np.zeros((0, 3)), np.zeros((0, 3))

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # Apply inverse transformation
    if callable(inverse_transform):
        # Use the provided callable
        print("Applying inverse transformation to mesh vertices...")
        mesh_points = inverse_transform(mesh_points)
    elif isinstance(inverse_transform, dict):
        # Backwards compatibility: transform_params dict
        print("Applying inverse transformation to mesh vertices...")
        # Inverse transformation from normalized [-1, 1] to original scale
        mesh_points = mesh_points / 2.0  # [-1, 1] -> [-0.5, 0.5]
        mesh_points = mesh_points + 0.5  # [-0.5, 0.5] -> [0, 1]
        mesh_points = mesh_points * inverse_transform['scale'] + inverse_transform['coord_min']  # [0, 1] -> centered original scale
        mesh_points = mesh_points + inverse_transform['center']  # Add back original center
    else:
        # apply additional offset and scale (legacy behavior)
        if scale is not None:
            mesh_points = mesh_points / scale
        if offset is not None:
            mesh_points = mesh_points - offset

    return mesh_points, faces


def save_mesh_as_ply(mesh_points, faces, filename):
    """Save mesh as PLY file"""
    num_verts = mesh_points.shape[0]
    num_faces = faces.shape[0]

    if num_verts > 0:
        verts_tuple = np.zeros(
            (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        )

        for i in range(num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        ply_data.write(filename)
        print(f"Saved mesh with {num_verts} vertices and {num_faces} faces to {filename}")
    else:
        print(f"No mesh generated for {filename}")


def save_mesh_as_obj(mesh_points, faces, filename):
    """Save mesh as OBJ file"""
    num_verts = mesh_points.shape[0]
    num_faces = faces.shape[0]

    if num_verts > 0:
        with open(filename, 'w') as f:
            # Write vertices
            for i in range(num_verts):
                f.write(f"v {mesh_points[i, 0]} {mesh_points[i, 1]} {mesh_points[i, 2]}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for i in range(num_faces):
                f.write(f"f {faces[i, 0]+1} {faces[i, 1]+1} {faces[i, 2]+1}\n")
        
        print(f"Saved mesh with {num_verts} vertices and {num_faces} faces to {filename}")
    else:
        print(f"No mesh generated for {filename}")
