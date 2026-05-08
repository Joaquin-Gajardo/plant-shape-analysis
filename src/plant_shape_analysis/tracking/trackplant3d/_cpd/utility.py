import numpy as np


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-np.sum(diff ** 2, axis=2) / (2 * beta ** 2))


def low_rank_eigen(G, num_eig):
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    return Q[:, eig_indices], S[eig_indices]
