# neuron.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from utils import add_bias


def heaviside(s: np.ndarray) -> np.ndarray:
    return (s >= 0).astype(float)

def sigmoid(s: np.ndarray, beta: float) -> np.ndarray:
    z = np.clip(beta * s, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def tanh_act(s: np.ndarray) -> np.ndarray:
    return np.tanh(s)

def sin_act(s: np.ndarray) -> np.ndarray:
    return np.sin(s)

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


class SingleNeuron:

    def __init__(self, lr: float = 0.05, activation: str = "heaviside", beta: float = 2.0, seed: int = 0):
        self.lr = float(lr)
        self.activation = activation.lower()
        self.beta = float(beta)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.3, size=(3,))  # [w1, w2, b]

    def set_params(self, lr: float | None = None, activation: str | None = None, beta: float | None = None) -> None:
        if lr is not None:
            self.lr = float(lr)
        if activation is not None:
            self.activation = activation.lower()
        if beta is not None:
            self.beta = float(beta)

    def score(self, X: np.ndarray) -> np.ndarray:
        Xb = add_bias(X)
        return Xb @ self.w

    def forward(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        a = self.activation
        if a == "heaviside":
            return heaviside(s)
        if a in ("logistic", "sigmoid"):
            return sigmoid(s, beta=self.beta)
        if a == "tanh":
            return tanh_act(s)
        if a == "sin":
            return sin_act(s)
        raise ValueError("Unsupported activation (use heaviside/logistic/tanh/sin).")

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        a = self.activation
        if a == "heaviside":
            return (s >= 0).astype(float)
        if a in ("logistic", "sigmoid"):
            return (sigmoid(s, beta=self.beta) >= 0.5).astype(float)
        if a == "tanh":
            return (tanh_act(s) >= 0.0).astype(float)
        if a == "sin":
            return (sin_act(s) >= 0.0).astype(float)
        raise ValueError("Unsupported activation (use heaviside/logistic/tanh/sin).")

    def accuracy(self, X: np.ndarray, d01: np.ndarray) -> float:
        pred = self.predict_labels(X)
        return float(np.mean(pred == d01.astype(float)))

    def train_sgd(
        self,
        X: np.ndarray,
        d01: np.ndarray,
        epochs: int = 30,
        shuffle: bool = True,
        lr_schedule: LRScheduleCosine | None = None,
    ) -> list[float]:
        Xb = add_bias(X)
        d = d01.astype(float)

        rng = np.random.default_rng(1234)
        losses: list[float] = []

        epochs = int(epochs)
        if epochs <= 0:
            return losses

        n_max = max(1, epochs - 1)

        for ep in range(epochs):
            if lr_schedule is not None:
                self.lr = lr_schedule.value(ep, n_max)

            idx = np.arange(Xb.shape[0])
            if shuffle:
                rng.shuffle(idx)

            for i in idx:
                x_i = Xb[i]
                s = float(x_i @ self.w)
                a = self.activation

                if a == "heaviside":
                    y = 1.0 if s >= 0 else 0.0
                    fprime = 1.0
                elif a in ("logistic", "sigmoid"):
                    y = float(sigmoid(np.array([s]), beta=self.beta)[0])
                    fprime = self.beta * y * (1.0 - y)
                elif a == "tanh":
                    y = float(np.tanh(s))
                    fprime = 1.0 - y * y
                elif a == "sin":
                    y = float(np.sin(s))
                    fprime = float(np.cos(s))
                else:
                    raise ValueError("Unsupported activation for training.")

                self.w += self.lr * (d[i] - y) * fprime * x_i

            y_all = self.forward(X)
            losses.append(float(np.mean((d - y_all) ** 2)))

        return losses
