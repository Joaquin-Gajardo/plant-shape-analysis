"""
PSegNet model architecture for simultaneous semantic and instance segmentation of plant organs.

Adapted from PlantNet-and-PSegNet (PSegNet/PSegNet_pytorch/models/model_pytorch.py).
Loss functions removed; only the forward pass is retained for inference.
"""

import torch
import torch.nn as nn

from plant_shape_analysis.segmentation.psegnet.pointnet2_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    SimmatModel,
    get_sample_points,
)

NUM_CLASSES = 6


class PSegNet(nn.Module):
    """
    PSegNet: dual-head PointNet++ for semantic + instance segmentation of plant organs.

    Input : (B, N, 3) point cloud (PointNet-normalized).
    Output: sem_logits (B, N, num_classes), ins_embed (B, N, 5), simmat (B, N, N).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        # Encoder
        self.sa1 = PointNetSetAbstraction(mlp1=[32, 32, 64],   in_channel=3 + 3,   k=16, d=1)
        self.sa2 = PointNetSetAbstraction(mlp1=[64, 64, 128],  in_channel=64 + 3,  k=16, d=1)
        self.sa3 = PointNetSetAbstraction(mlp1=[128, 128, 256],in_channel=128 + 3, k=8,  d=1)
        self.sa4 = PointNetSetAbstraction(mlp1=[256, 256, 512],in_channel=256 + 3, k=8,  d=1)
        # Semantic decoder
        self.fp4 = PointNetFeaturePropagation(mlp=[256, 256],      in_channel=512 + 256)
        self.fp3 = PointNetFeaturePropagation(mlp=[256, 256],      in_channel=128 + 256)
        self.fp2 = PointNetFeaturePropagation(mlp=[256, 128],      in_channel=64 + 256)
        self.fp1 = PointNetFeaturePropagation(mlp=[128, 128, 128], in_channel=3 + 128)
        # Instance decoder (uses different skip connections)
        self.fp8 = PointNetFeaturePropagation(mlp=[256, 256],      in_channel=512 + 256)
        self.fp6 = PointNetFeaturePropagation(mlp=[256, 128],      in_channel=64 + 256)
        self.fp5 = PointNetFeaturePropagation(mlp=[128, 128, 128], in_channel=3 + 128)
        # Feature fusion
        self.CONV1 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV2 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV3 = nn.Sequential(nn.Conv1d(129, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV4 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        # Instance head
        self.CONV5 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV6 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV9 = nn.Conv1d(128, 5, 1)
        self.drop3 = nn.Dropout(0.5)
        # Semantic head
        self.CONV7 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV8 = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.CONV10 = nn.Conv1d(128, self.num_classes, 1)
        self.drop4 = nn.Dropout(0.5)
        # Similarity matrix (used during training only, kept for checkpoint compatibility)
        self.Simmat_logits = SimmatModel()

    def forward(self, point_cloud: torch.Tensor):
        B = point_cloud.shape[0]
        l0_xyz = l0_points = point_cloud

        # Encoder
        l1_xyz, l1_points = get_sample_points(l0_xyz, l0_points, npoints=1024)
        l1_points = self.sa1(l1_xyz, l1_points)
        l2_xyz, l2_points = get_sample_points(l1_xyz, l1_points, npoints=256)
        l2_points = self.sa2(l2_xyz, l2_points)
        l3_xyz, l3_points = get_sample_points(l2_xyz, l2_points, npoints=128)
        l3_points = self.sa3(l3_xyz, l3_points)
        l4_xyz, l4_points = get_sample_points(l3_xyz, l3_points, npoints=128)
        l4_points = self.sa4(l4_xyz, l4_points)

        # Semantic decoder
        l0_sem = self.fp1(l0_xyz, l1_xyz, l0_points,
                  self.fp2(l1_xyz, l2_xyz, l1_points,
                  self.fp3(l2_xyz, l3_xyz, l2_points,
                  self.fp4(l3_xyz, l4_xyz, l3_points, l4_points))))
        net_sem = self.CONV1(l0_sem.transpose(1, 2)).transpose(1, 2)

        # Instance decoder (fp8 → fp6 → fp5, skipping l2)
        l0_ins = self.fp5(l0_xyz, l1_xyz, l0_points,
                  self.fp6(l1_xyz, l3_xyz, l1_points,
                  self.fp8(l3_xyz, l4_xyz, l3_points, l4_points)))
        net_ins = self.CONV2(l0_ins.transpose(1, 2)).transpose(1, 2)

        # Fusion
        ins_avg = net_ins.mean(dim=-1, keepdim=True)                      # (B, N, 1)
        fused = torch.cat([net_sem, ins_avg], dim=-1)                     # (B, N, 129)
        fused = self.CONV3(fused.transpose(1, 2)).transpose(1, 2)         # (B, N, 128)
        fused = self.CONV4(torch.cat([fused, net_ins], dim=-1).transpose(1, 2)).transpose(1, 2)

        simmat = self.Simmat_logits(fused, B)

        # Instance head with self-attention pooling
        net_ins3 = fused * torch.sigmoid(fused.mean(dim=-1, keepdim=True))
        max_i = self.CONV5(net_ins3.max(dim=-2, keepdim=True)[0].transpose(1, 2)).transpose(1, 2)
        avg_i = self.CONV6(net_ins3.mean(dim=-2, keepdim=True).transpose(1, 2)).transpose(1, 2)
        net_ins3 = net_ins3 * torch.sigmoid(max_i + avg_i)
        net_ins4 = self.CONV9(self.drop3(net_ins3).transpose(1, 2)).transpose(1, 2)

        # Semantic head with self-attention pooling
        net_sem3 = fused * torch.sigmoid(fused.mean(dim=-1, keepdim=True))
        max_s = self.CONV7(net_sem3.max(dim=-2, keepdim=True)[0].transpose(1, 2)).transpose(1, 2)
        avg_s = self.CONV8(net_sem3.mean(dim=-2, keepdim=True).transpose(1, 2)).transpose(1, 2)
        net_sem3 = net_sem3 * torch.sigmoid(max_s + avg_s)
        net_sem4 = self.CONV10(self.drop4(net_sem3).transpose(1, 2)).transpose(1, 2)

        return net_sem4, net_ins4, simmat
