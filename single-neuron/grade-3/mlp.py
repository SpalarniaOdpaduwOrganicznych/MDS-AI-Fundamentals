# mlp.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def sigmoid(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = np.clip(beta * z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime_from_activation(a: np.ndarray, beta: float = 1.0) -> np.ndarray:
    # if a = sigmoid(beta*z), then da/dz = beta*a*(1-a)
    return beta * a * (1.0 - a)


def softmax(logits: np.ndarray) -> np.ndarray:
    # stable softmax
    m = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - m)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int).ravel()
    oh = np.zeros((y.size, num_classes), dtype=float)
    oh[np.arange(y.size), y] = 1.0
    return oh


@dataclass(frozen=True)
class LRScheduleCosine:
    eta_min: float
    eta_max: float

    def value(self, n: int, n_max: int) -> float:
        if n_max <= 0:
            return float(self.eta_max)
        n = max(0, min(int(n), int(n_max)))
        return float(
            self.eta_min
            + (self.eta_max - self.eta_min) * (1.0 + np.cos(np.pi * n / n_max))
        )


class MLP:
    """
    Fully-connected MLP, up to 5 layers total (input + hidden(s) + output).
    Hidden activation: logistic (sigmoid). Output: softmax.
    Backprop is vectorized (NO individual neurons).
    """

    def __init__(
        self,
        layer_sizes: list[int],   # e.g. [2, 16, 16, 2] or [784, 128, 64, 10]
        lr: float = 0.05,
        beta: float = 1.0,
        seed: int = 0,
    ):
        if len(layer_sizes) < 3:
            raise ValueError("Need at least 3 layers: input, ≥1 hidden, output.")
        if len(layer_sizes) > 5:
            raise ValueError("Max 5 layers allowed.")
        if any(s <= 0 for s in layer_sizes):
            raise ValueError("All layer sizes must be positive.")

        self.layer_sizes = [int(s) for s in layer_sizes]
        self.lr = float(lr)
        self.beta = float(beta)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

        # Xavier/Glorot-ish init for sigmoid
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out), dtype=float))

    def set_params(self, lr: float | None = None, beta: float | None = None) -> None:
        if lr is not None:
            self.lr = float(lr)
        if beta is not None:
            self.beta = float(beta)

    def forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Returns:
          Zs: pre-activations per layer (excluding input)
          As: activations per layer including input A0 = X
        """
        A = X
        As = [A]
        Zs = []
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            A = sigmoid(Z, beta=self.beta)
            Zs.append(Z)
            As.append(A)

        # output layer (logits)
        ZL = A @ self.W[-1] + self.b[-1]
        Zs.append(ZL)
        AL = softmax(ZL)
        As.append(AL)
        return Zs, As

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, As = self.forward(X)
        return As[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = y.astype(int).ravel()
        pred = self.predict(X)
        return float(np.mean(pred == y))

    def loss_ce(self, probs: np.ndarray, y: np.ndarray) -> float:
        y = y.astype(int).ravel()
        eps = 1e-12
        p = np.clip(probs[np.arange(y.size), y], eps, 1.0)
        return float(np.mean(-np.log(p)))

    def train_minibatch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        shuffle: bool = True,
        lr_schedule: LRScheduleCosine | None = None,
    ) -> list[float]:
        """
        Vectorized backprop for softmax + cross-entropy:
          dZ_out = (P - Y_onehot)/B
        Hidden:
          dZ = (dA * sigmoid'(A))
        """
        X = np.asarray(X, dtype=float)
        y = y.astype(int).ravel()
        n = X.shape[0]
        if n == 0:
            return []

        num_classes = self.layer_sizes[-1]
        if np.max(y) >= num_classes or np.min(y) < 0:
            raise ValueError("y contains labels outside output dimension.")

        batch_size = max(1, int(batch_size))
        epochs = int(epochs)
        losses: list[float] = []

        rng = np.random.default_rng(1234)
        n_max = max(1, epochs - 1)

        for ep in range(epochs):
            if lr_schedule is not None:
                self.lr = lr_schedule.value(ep, n_max)

            idx = np.arange(n)
            if shuffle:
                rng.shuffle(idx)

            for start in range(0, n, batch_size):
                bi = idx[start : start + batch_size]
                Xb = X[bi]
                yb = y[bi]
                B = Xb.shape[0]

                Zs, As = self.forward(Xb)
                P = As[-1]  # softmax output
                Y = one_hot(yb, num_classes)

                # --- output layer gradient ---
                dZ = (P - Y) / B  # (B, C)
                dW = As[-2].T @ dZ
                db = np.sum(dZ, axis=0, keepdims=True)

                # store grads and backprop through hidden layers
                dWs = [None] * len(self.W)
                dbs = [None] * len(self.b)
                dWs[-1], dbs[-1] = dW, db

                dA_prev = dZ @ self.W[-1].T  # (B, H_last)

                # --- hidden layers (reverse) ---
                for li in range(len(self.W) - 2, -1, -1):
                    A = As[li + 1]  # activation at this hidden layer
                    dZ = dA_prev * sigmoid_prime_from_activation(A, beta=self.beta)
                    dW = As[li].T @ dZ
                    db = np.sum(dZ, axis=0, keepdims=True)

                    dWs[li], dbs[li] = dW, db

                    if li > 0:
                        dA_prev = dZ @ self.W[li].T

                # --- SGD step ---
                for li in range(len(self.W)):
                    self.W[li] -= self.lr * dWs[li]
                    self.b[li] -= self.lr * dbs[li]

            # epoch loss
            probs = self.predict_proba(X)
            losses.append(self.loss_ce(probs, y))

        return losses
