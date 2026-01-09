from __future__ import annotations
import numpy as np


def _sample_gaussian_modes(
    n_modes: int,
    samples_per_mode: int,
    mean_low: float,
    mean_high: float,
    std_low: float,
    std_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pts: list[np.ndarray] = []
    for _ in range(int(n_modes)):
        mu = rng.uniform(mean_low, mean_high, size=(2,))
        std = rng.uniform(std_low, std_high, size=(2,))
        cov = np.diag(std ** 2)
        pts.append(rng.multivariate_normal(mean=mu, cov=cov, size=int(samples_per_mode)))

    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.vstack(pts)


def generate_two_class_data(
    modes0: int,
    modes1: int,
    samples_per_mode: int,
    mean_low: float = -1.0,
    mean_high: float = 1.0,
    std_low: float = 0.10,
    std_high: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:

    if mean_high <= mean_low:
        raise ValueError("mean_high must be > mean_low")
    if std_low <= 0 or std_high <= std_low:
        raise ValueError("std range must be positive and std_high > std_low")
    if modes0 == 0 and modes1 == 0:
        raise ValueError("At least one class must have ≥ 1 mode.")
    if samples_per_mode <= 0:
        raise ValueError("samples_per_mode must be > 0")

    rng = np.random.default_rng(seed)

    X0 = _sample_gaussian_modes(modes0, samples_per_mode, mean_low, mean_high, std_low, std_high, rng)
    X1 = _sample_gaussian_modes(modes1, samples_per_mode, mean_low, mean_high, std_low, std_high, rng)

    y0 = np.zeros((X0.shape[0],), dtype=float)
    y1 = np.ones((X1.shape[0],), dtype=float)

    if X0.size and X1.size:
        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])
    elif X0.size:
        X, y = X0, y0
    else:
        X, y = X1, y1

    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    return X[idx], y[idx]
