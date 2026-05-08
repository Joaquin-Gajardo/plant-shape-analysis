"""
EM-based point cloud registration base class.

Copied from the pycpd library bundled with TrackPlant3D (MIT licence).
"""

from __future__ import division

import numbers
from warnings import warn

import numpy as np


def _initialize_sigma2(X, Y):
    N, D = X.shape
    M = Y.shape[0]
    diff = X[None, :, :] - Y[:, None, :]
    return np.sum(diff ** 2) / (D * M * N)


class EMRegistration:
    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, w=None, *args, **kwargs):
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be a 2D numpy array.")
        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError("Y must be a 2D numpy array.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of dimensions.")

        if max_iterations is not None and not isinstance(max_iterations, int):
            warn(f"max_iterations {max_iterations} cast to int.")
            max_iterations = int(max_iterations)

        self.X = X
        self.Y = Y
        self.TY = Y.copy()
        self.sigma2 = _initialize_sigma2(X, Y) if sigma2 is None else sigma2
        self.N, self.D = X.shape
        self.M = Y.shape[0]
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf
        self.q = np.inf
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros(self.N)
        self.P1 = np.zeros(self.M)
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0.0

    def register(self, callback=None):
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                callback(iteration=self.iteration, error=self.q, X=self.X, Y=self.TY)
        return self.TY, self.get_registration_parameters()

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        P = np.exp(-np.sum((self.X[None] - self.TY[:, None]) ** 2, axis=2) / (2 * self.sigma2))
        c = (2 * np.pi * self.sigma2) ** (self.D / 2) * self.w / (1.0 - self.w) * self.M / self.N
        den = np.clip(P.sum(axis=0, keepdims=True), np.finfo(self.X.dtype).eps, None) + c
        self.P = P / den
        self.Pt1 = self.P.sum(axis=0)
        self.P1 = self.P.sum(axis=1)
        self.Np = self.P1.sum()
        self.PX = self.P @ self.X

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()

    def get_registration_parameters(self):
        raise NotImplementedError

    def update_transform(self):
        raise NotImplementedError

    def transform_point_cloud(self):
        raise NotImplementedError

    def update_variance(self):
        raise NotImplementedError
