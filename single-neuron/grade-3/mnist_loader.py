# mnist_loader.py
from __future__ import annotations
import numpy as np

def load_mnist_via_torch(train: bool = True):
    """
    Uses torchvision if available. Returns (X, y) with:
      X: float32 in [0,1], shape (N, 784)
      y: int64 in [0..9], shape (N,)
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError(
            "torch/torchvision not available. Install them or provide your own MNIST loader."
        ) from e

    ds = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

    X = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
    y = ds.targets.numpy().astype(np.int64)

    X = X.reshape(X.shape[0], -1)  # flatten -> (N, 784)
    return X, y
