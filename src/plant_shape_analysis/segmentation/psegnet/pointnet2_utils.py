"""
PointNet++ building blocks for PSegNet.

Adapted from PlantNet-and-PSegNet (PSegNet/PSegNet_pytorch/utils/pointnet2_util_pytorch.py
and utils/torch_util.py). All sys.path manipulation and external imports removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Inlined from torch_util ──────────────────────────────────────────────────

def dg_knn(adj_matrix: torch.Tensor, k: int = 20, d: int = 3) -> torch.Tensor:
    k1 = k * d + 1
    _, nn_idx = torch.topk(-adj_matrix, k=k1)
    return nn_idx[:, :, 1:k1:d]


def pairwise_distance(point_cloud: torch.Tensor) -> torch.Tensor:
    og_batch_size = point_cloud.shape[0]
    point_cloud = torch.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)
    pc_T = point_cloud.transpose(-1, -2)
    inner = -2 * torch.matmul(point_cloud, pc_T)
    sq = torch.sum(torch.square(point_cloud), dim=-1, keepdim=True)
    return sq + inner + sq.permute(0, 2, 1)


def get_edge_feature(point_cloud: torch.Tensor, nn_idx: torch.Tensor, k: int = 20) -> torch.Tensor:
    og_batch_size = point_cloud.shape[0]
    point_cloud = point_cloud.squeeze()
    if og_batch_size == 1:
        point_cloud = point_cloud.unsqueeze(0)
    batch_size, num_points, num_dims = point_cloud.shape
    device = point_cloud.device
    idx_ = (torch.arange(batch_size) * num_points).view(batch_size, 1, 1).to(device)
    flat = point_cloud.contiguous().view(-1, num_dims)
    neighbors = flat[nn_idx + idx_]
    central = point_cloud.unsqueeze(-2).repeat(1, 1, k, 1)
    return torch.cat([central, neighbors - central], dim=-1)


# ── Inlined from pointnet2_util_pytorch ─────────────────────────────────────

def square_distance(src: torch.Tensor, dst: torch.Tensor):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.sqrt(dist.clamp(min=0))
    idx = torch.argmin(dist, dim=-1)
    return dist, idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(npoint: int, xyz: torch.Tensor) -> torch.Tensor:
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def get_sample_points(xyz: torch.Tensor, points: torch.Tensor, npoints: int, use_xyz: bool = True):
    batch_size = xyz.shape[0]
    index = farthest_point_sample(npoints, xyz)
    new_xyz = index_points(xyz, index)
    if points.shape[2] > 0:
        sample_points = torch.stack([points[b].index_select(0, index[b]) for b in range(batch_size)])
        new_points = torch.cat([new_xyz, sample_points], dim=-1) if use_xyz else sample_points
    else:
        new_points = new_xyz
    return new_xyz, new_points


def relative_pos_encoding(xyz: torch.Tensor, neigh_idx: torch.Tensor) -> torch.Tensor:
    neighbor_xyz = index_points(xyz, neigh_idx)
    xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = torch.sqrt(torch.sum(relative_xyz ** 2, dim=-1, keepdim=True))
    return torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)


# ── Network modules ──────────────────────────────────────────────────────────

class position_encode(nn.Module):
    def __init__(self, k: int, d: int):
        super().__init__()
        self.k1 = k
        self.d1 = d

    def forward(self, new_xyz: torch.Tensor, new_points: torch.Tensor) -> torch.Tensor:
        adj = pairwise_distance(new_points)
        nn_idx = dg_knn(adj, k=self.k1, d=self.d1)
        return relative_pos_encoding(new_xyz, nn_idx)


class edge_conv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.CONV2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, new_points: torch.Tensor, k: int, d: int) -> torch.Tensor:
        adj = pairwise_distance(new_points)
        nn_idx = dg_knn(adj, k=k, d=d)
        feature_set = get_edge_feature(new_points, nn_idx=nn_idx, k=k)
        out = self.CONV2d(feature_set.transpose(0, 1).transpose(1, 3))
        return out.transpose(1, 3).transpose(0, 1)


class att_pooling(nn.Module):
    def __init__(self, out_channel: int):
        super().__init__()
        in_channel = out_channel + 10
        self.l1 = nn.Linear(in_channel, in_channel)
        self.CONV2d = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, new_xyz1: torch.Tensor, new_points1: torch.Tensor) -> torch.Tensor:
        points_xyz = torch.cat([new_points1, new_xyz1], dim=-1)
        B, N, K, D = points_xyz.shape
        f = points_xyz.reshape(-1, K, D)
        att = F.softmax(self.l1(f), dim=1)
        f_agg = (f * att).sum(dim=1).reshape(B, N, 1, D)
        f_agg = self.CONV2d(f_agg.transpose(0, 1).transpose(1, 3))
        f_agg = f_agg.transpose(0, 1).transpose(0, 3)
        og_batch = f_agg.shape[0]
        f_agg = f_agg.squeeze()
        if og_batch == 1:
            f_agg = f_agg.unsqueeze(0)
        return f_agg


class PointNetSetAbstraction(nn.Module):
    def __init__(self, mlp1: list[int], in_channel: int, k: int, d: int):
        super().__init__()
        self.k1 = k
        self.d1 = d
        self.mlp = mlp1
        self.PE = position_encode(k, d)
        self.mlp1_aps = nn.ModuleList()
        self.mlp1_ecs = nn.ModuleList()
        for i, out_ch in enumerate(mlp1):
            self.mlp1_aps.append(att_pooling(out_ch))
            if i == 0:
                self.mlp1_ecs.append(edge_conv(2 * in_channel, out_ch))
            elif i == 1:
                self.mlp1_ecs.append(edge_conv(2 * mlp1[0], out_ch))
            else:
                self.mlp1_ecs.append(edge_conv(4 * mlp1[1], out_ch))

    def forward(self, new_xyz: torch.Tensor, new_points: torch.Tensor) -> torch.Tensor:
        new_xyz1 = self.PE(new_xyz, new_points)
        f_agg1 = f_agg2 = f_agg3 = None
        for i in range(len(self.mlp)):
            AP, EC = self.mlp1_aps[i], self.mlp1_ecs[i]
            if i == 0:
                f_agg1 = AP(new_xyz1, EC(new_points, self.k1, self.d1))
            elif i == 1:
                f_agg2 = AP(new_xyz1, EC(f_agg1, self.k1, self.d1))
                f_agg2 = torch.cat([f_agg1, f_agg2], dim=-1)
            else:
                f_agg3 = AP(new_xyz1, EC(f_agg2, self.k1, self.d1))
                f_agg3 = f_agg2 + f_agg3
        return f_agg3


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, mlp: list[int], in_channel: int):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2)
            dists, idx = torch.sort(dists, dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            weight = (1.0 / (dists + 1e-8))
            weight = weight / weight.sum(dim=2, keepdim=True)
            interpolated = (index_points(points2, idx) * weight.unsqueeze(-1)).sum(dim=2)
        new_points = torch.cat([points1, interpolated], dim=-1) if points1 is not None else interpolated
        new_points = new_points.unsqueeze(2).transpose(0, 1).transpose(1, 3)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        og_batch = new_points.shape[-1]
        new_points = new_points.squeeze()
        if og_batch == 1:
            new_points = new_points.unsqueeze(-1)
        return new_points.transpose(0, 2).transpose(1, 2)


class SimmatModel(nn.Module):
    def forward(self, Fsim: torch.Tensor, batch_size: int) -> torch.Tensor:
        r = torch.sum(Fsim * Fsim, dim=2).view(batch_size, -1, 1)
        D = r - 2 * torch.matmul(Fsim, Fsim.transpose(1, 2)) + r.transpose(1, 2)
        return torch.maximum(10 * D, torch.zeros_like(D))
