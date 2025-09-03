# Examples of analytical SDF area calculation using the coarea formula and Monte Carlo approximation

import argparse

import numpy as np
import open3d as o3d


def run_sphere(N, eps=0.01, threshold=0.01, seed=0):
    """
    Numerical check of the delta-integral area formula on a sphere using Monte Carlo
    Sphere SDF: phi(x,y,z) = sqrt(x^2+y^2+z^2) - r
    Area formula for an exact SDF: A = ∫ δ(phi(x)) dx (since ||∇phi||=1 a.e.)
    We'll approximate δ with a narrow Gaussian mollifier δ_ε(t) = (1/(sqrt(pi) ε)) * exp(-(t/ε)^2)
    """
    r = 1.0  # radius
    A_exact = 4 * np.pi * r**2
    margin = 0.25
    L = r + margin
    box_min = np.array([-L, -L, -L])
    box_max = np.array([L, L, L])
    vol = np.prod(box_max - box_min)
    rng = np.random.default_rng(seed)
    X = rng.uniform(box_min[0], box_max[0], N)
    Y = rng.uniform(box_min[1], box_max[1], N)
    Z = rng.uniform(box_min[2], box_max[2], N)
    phi = np.sqrt(X**2 + Y**2 + Z**2) - r
    delta_eps = (1.0 / (np.sqrt(np.pi) * eps)) * np.exp(-((phi / eps) ** 2))
    A_mc = vol * np.mean(delta_eps)
    print(f"[Sphere] Exact area: {A_exact:.6f}, MC estimate: {A_mc:.6f}")
    mask = np.abs(phi) < threshold
    surface_points = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)
    o3d.visualization.draw_geometries([pcd])


def run_torus(N, eps=0.01, threshold=0.01, seed=0):
    """
    Numerical check of the delta-integral area formula on a torus using Monte Carlo
    Torus SDF: phi(x,y,z) = sqrt((sqrt(x^2+y^2) - R)^2 + z^2) - r
    Area formula for an exact SDF: A = ∫ δ(phi(x)) dx (since ||∇phi||=1 a.e.)
    We'll approximate δ with a narrow Gaussian mollifier δ_ε(t) = (1/(sqrt(pi) ε)) * exp(-(t/ε)^2)
    """
    R = 2.0  # major radius
    r = 0.5  # minor radius
    A_exact = 4 * np.pi**2 * R * r
    margin = 0.25
    Lxy = R + r
    Lz = r + margin
    box_min = np.array([-Lxy, -Lxy, -Lz])
    box_max = np.array([Lxy, Lxy, Lz])
    vol = np.prod(box_max - box_min)
    rng = np.random.default_rng(seed)
    X = rng.uniform(box_min[0], box_max[0], N)
    Y = rng.uniform(box_min[1], box_max[1], N)
    Z = rng.uniform(box_min[2], box_max[2], N)
    s = np.sqrt(X**2 + Y**2)
    phi = np.sqrt((s - R) ** 2 + Z**2) - r
    delta_eps = (1.0 / (np.sqrt(np.pi) * eps)) * np.exp(-((phi / eps) ** 2))
    A_mc = vol * np.mean(delta_eps)
    print(f"[Torus] Exact area: {A_exact:.6f}, MC estimate: {A_mc:.6f}")
    mask = np.abs(phi) < threshold
    surface_points = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo SDF area estimation for sphere or torus."
    )
    parser.add_argument(
        "--shape",
        choices=["sphere", "torus"],
        required=True,
        help="Shape to analyze (sphere or torus)",
    )
    parser.add_argument(
        "--samples", type=int, default=int(1e6), help="Number of Monte Carlo samples"
    )
    parser.add_argument(
        "--eps", type=float, default=0.01, help="Width of Gaussian delta approximation"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for surface visualization",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.shape == "sphere":
        run_sphere(args.samples, eps=args.eps, threshold=args.threshold, seed=args.seed)
    elif args.shape == "torus":
        run_torus(args.samples, eps=args.eps, threshold=args.threshold, seed=args.seed)
