from __future__ import annotations
import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Append bias column of ones: (N,2) -> (N,3)."""
    if X.ndim != 2:
        raise ValueError("X must be 2D array")
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])


def as_int(value: str, name: str, min_value: int | None = None) -> int:
    try:
        v = int(float(value))
    except Exception as e:
        raise ValueError(f"{name} must be an integer") from e
    if min_value is not None and v < min_value:
        raise ValueError(f"{name} must be ≥ {min_value}")
    return v


def as_float(value: str, name: str) -> float:
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"{name} must be a number") from e

