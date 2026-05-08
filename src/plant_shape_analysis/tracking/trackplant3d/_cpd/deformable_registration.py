"""
Coherent Point Drift deformable registration.

Copied from the pycpd library bundled with TrackPlant3D (MIT licence).
Reference: Myronenko & Song, "Point Set Registration: Coherent Point Drift",
           IEEE TPAMI 2010. https://arxiv.org/abs/0905.2635
"""

import numpy as np

from plant_shape_analysis.tracking.trackplant3d._cpd.emregistration import EMRegistration
from plant_shape_analysis.tracking.trackplant3d._cpd.utility import gaussian_kernel, low_rank_eigen


class DeformableRegistration(EMRegistration):
    """
    Non-rigid (deformable) CPD registration.

    Args:
        alpha: Regularization strength (default 2).
        beta: Gaussian kernel width (default 2).
        low_rank: Use low-rank approximation (default False).
        num_eig: Number of eigenvectors for low-rank mode (default 100).
        X: (N, D) target point cloud.
        Y: (M, D) source point cloud to deform.
    """

    def __init__(self, alpha=2, beta=2, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1.0 / self.S)
            self.S = np.diag(self.S)
            self.E = 0.0

    def update_transform(self):
        if not self.low_rank:
            A = np.diag(self.P1) @ self.G + self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.diag(self.P1) @ self.Y
            self.W = np.linalg.solve(A, B)
        else:
            dP = np.diag(self.P1)
            dPQ = dP @ self.Q
            F = self.PX - dP @ self.Y
            self.W = (F - dPQ @ np.linalg.solve(
                self.alpha * self.sigma2 * self.inv_S + self.Q.T @ dPQ,
                self.Q.T @ F,
            )) / (self.alpha * self.sigma2)
            self.E += self.alpha / 2 * np.trace((self.Q.T @ self.W).T @ self.S @ (self.Q.T @ self.W))

    def transform_point_cloud(self, Y=None):
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + G @ self.W
        if not self.low_rank:
            self.TY = self.Y + self.G @ self.W
        else:
            self.TY = self.Y + self.Q @ self.S @ self.Q.T @ self.W

    def update_variance(self):
        qprev = self.sigma2
        self.q = np.inf
        xPx = self.Pt1 @ np.sum(self.X ** 2, axis=1)
        yPy = self.P1 @ np.sum(self.TY ** 2, axis=1)
        trPXY = np.sum(self.TY * self.PX)
        self.sigma2 = max((xPx - 2 * trPXY + yPy) / (self.Np * self.D), self.tolerance / 10)
        self.diff = abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        return self.G, self.W
