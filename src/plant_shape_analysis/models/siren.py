"""
https://github.com/vsitzmann/siren

MIT License

Copyright (c) 2020 Vincent Sitzmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    model_name = "siren"

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return {"model_in": coords, "model_out": output}


class PointCloudSiren(Dataset):
    def __init__(
        self,
        point_cloud,
        on_surface_points: int,
        off_surface_points: Optional[int] = None,
        keep_aspect_ratio=True,
        sampling_strategy="mixed",
        k_neighbors=50,
        off_surface_local_points_ratio=0.5,
    ):
        super().__init__()

        coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.0

        self.on_surface_points = on_surface_points
        self.off_surface_points = off_surface_points
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.off_surface_local_points_ratio = off_surface_local_points_ratio

        # Pre-compute k-nearest neighbor distances for Gaussian sampling
        if sampling_strategy == "mixed":
            self._compute_knn_distances()

        print("Off surface sampling strategy: ", self.sampling_strategy)

    def _compute_knn_distances(self):
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(self.coords)
        distances, _ = nbrs.kneighbors(self.coords)
        # Use k-th nearest neighbor distance (excluding self)
        self.knn_distances = distances[:, self.k_neighbors]

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]
        if self.off_surface_points is None:
            self.off_surface_points = self.on_surface_points
        total_samples = self.on_surface_points + self.off_surface_points

        # On surface points
        rand_idcs = np.random.choice(
            point_cloud_size, size=self.on_surface_points
        )  # on surface points is the total number of points anyways, since the leaves are very sparse
        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        # Off-surface points
        if self.sampling_strategy == "uniform":
            off_surface_coords = np.random.uniform(
                -1, 1, size=(self.off_surface_points, 3)
            )
            off_surface_normals = np.ones((self.off_surface_points, 3)) * -1

        elif self.sampling_strategy == "mixed":
            # Split off-surface points into local and global
            n_local = int(self.off_surface_points * self.off_surface_local_points_ratio)
            n_global = self.off_surface_points - n_local

            # Global off-surface points: uniform distribution
            global_coords = np.random.uniform(-1, 1, size=(n_global, 3))

            # Local off-surface points (gaussian distribution)
            center_indices = np.random.choice(point_cloud_size, size=n_local)
            centers = self.coords[center_indices]
            std_devs = self.knn_distances[center_indices]
            local_coords = np.random.normal(
                loc=centers, scale=std_devs[:, np.newaxis], size=(n_local, 3)
            )
            #  local_coords = global_coords + local_coords # NOTE: this seems like a bug

            off_surface_coords = np.concatenate([local_coords, global_coords], axis=0)
            off_surface_normals = np.ones((self.off_surface_points, 3)) * -1
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }


def get_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def sdf_loss(model_output, gt_sdf, gt_normals):
    """
    x: batch of input coordinates
    y: usually the output of the trial_soln function
    """
    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    gradient = get_gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf))
    )
    normal_constraint = torch.where(
        gt_sdf != -1,
        1 - cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient[..., :1]),
    )
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {
        "sdf": torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
        "inter": inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
        "normal_constraint": normal_constraint.mean() * 1e2,  # 1e2
        "grad_constraint": grad_constraint.mean() * 5e1,
    }  # 1e1      # 5e1
