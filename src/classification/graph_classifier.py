"""
Phase 3 Step 3.6: Model C — Spatio-Temporal Graph Convolutional Network (ST-GCN).
Ingests 25-node skeletal + board kinematic graph tensor (25, T=64, 5).
STRICT PROJECT MANDATE: ZERO raw RGB downstream of Phase 1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


def build_skate_kinematic_adjacency() -> torch.Tensor:
    """
    Constructs normalized spatial adjacency matrix (25 x 25):
    - Nodes 0-16: 17 COCO Human Pose joints
    - Nodes 17-24: 8 Semantic Board keypoints (nose, tail, corners, trucks)
    - Foot-to-board contact edges
    """
    N = 25
    A = np.zeros((N, N), dtype=np.float32)

    # 1. Human pose skeleton edges (COCO format)
    pose_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),               # torso
        (11, 13), (13, 15), (12, 14), (14, 16)    # legs to ankles
    ]
    for i, j in pose_edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # 2. Board perimeter and truck edges (Nodes 17 to 24)
    # 17: nose, 18: tail, 19: centroid, 20-23: corners, 24: trucks
    board_edges = [
        (17, 19), (18, 19),
        (17, 20), (17, 21), (18, 22), (18, 23),
        (20, 21), (22, 23), (19, 24)
    ]
    for i, j in board_edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # 3. Dynamic foot-to-board interaction edges
    # Left ankle (15) and Right ankle (16) to board nose, tail, and centroid
    foot_board_edges = [
        (15, 18), (15, 19),  # Left foot to tail / centroid
        (16, 17), (16, 19)   # Right foot to nose / centroid
    ]
    for i, j in foot_board_edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # 4. Self loops
    for i in range(N):
        A[i, i] = 1.0

    # Degree normalization: D^{-1/2} A D^{-1/2}
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv = np.diag(deg_inv_sqrt)
    A_norm = D_inv @ A @ D_inv

    return torch.from_numpy(A_norm)


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution operation."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V), A: (V, V)
        # Spatial aggregation over V
        x_a = torch.einsum('nctv,vw->nctw', (x, A))
        return self.conv(x_a)


class STGCNBlock(nn.Module):
    """Single Spatio-Temporal Graph Convolution Block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x + res)


class SkateSTGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for Skateboard Action Recognition.
    Ingests normalized coordinate tensors: (Batch, Channels=5, Time=64, Vertices=25).
    ZERO raw RGB inputs.
    """
    def __init__(self, in_channels: int = 5, num_classes: int = 9, sequence_length: int = 64):
        super().__init__()
        self.register_buffer('A', build_skate_kinematic_adjacency())

        self.data_bn = nn.BatchNorm2d(in_channels)
        self.block1 = STGCNBlock(in_channels, 32, stride=1)
        self.block2 = STGCNBlock(32, 64, stride=2)
        self.block3 = STGCNBlock(64, 128, stride=2)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channels, Time, Vertices)
        N, C, T, V = x.size()
        x = self.data_bn(x)

        x = self.block1(x, self.A)
        x = self.block2(x, self.A)
        x = self.block3(x, self.A)

        # Global average pool over Time and Vertices
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)

        return self.fc(x)


def build_compact_skate_adjacency() -> torch.Tensor:
    """
    Constructs normalized 6-node spatial adjacency:
    0: mid_hip, 1: left_ankle, 2: right_ankle
    3: board_nose, 4: board_tail, 5: board_centroid
    """
    V = 6
    A = np.eye(V, dtype=np.float32)
    edges = [
        (0, 1), (0, 2), (1, 2),         # Body triangle
        (3, 5), (4, 5), (3, 4),         # Board backbone
        (1, 4), (1, 5), (2, 3), (2, 5)  # Dynamic foot-board coupling
    ]
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    deg = np.sum(A, axis=1)
    deg_inv = np.power(deg, -0.5, where=deg > 0)
    A_norm = deg_inv[:, None] * A * deg_inv[None, :]
    return torch.from_numpy(A_norm).float()


class CompactSTGCN(nn.Module):
    """
    Compact regularized ST-GCN with Drop-Edge (p=0.2) and root-relative coords.
    Prevents catastrophic memorization of camera pan and skater identity.
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 9, drop_edge_p: float = 0.2):
        super().__init__()
        self.drop_edge_p = drop_edge_p
        self.register_buffer('A_base', build_compact_skate_adjacency())

        self.bn_in = nn.BatchNorm2d(in_channels)
        self.gcn1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.tcn1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2)
        )
        self.gcn2 = nn.Conv2d(32, 64, kernel_size=1)
        self.tcn2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        x = self.bn_in(x)

        # Drop-edge regularization during training
        A = self.A_base.clone()
        if self.training and self.drop_edge_p > 0.0:
            mask = (torch.rand_like(A) > self.drop_edge_p).float()
            A = A * mask

        # Block 1
        x = torch.einsum('nctv,vw->nctw', (x, A))
        x = self.gcn1(x)
        x = self.tcn1(x)

        # Block 2
        x = torch.einsum('nctv,vw->nctw', (x, A))
        x = self.gcn2(x)
        x = self.tcn2(x)

        # Global average pool
        x = F.avg_pool2d(x, x.size()[2:]).view(N, -1)
        return self.fc(x)

