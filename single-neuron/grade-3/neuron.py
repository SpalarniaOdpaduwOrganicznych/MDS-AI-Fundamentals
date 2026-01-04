from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from utils import add_bias


# ----------------------------
# Activations
# ----------------------------
def heaviside(s: np.ndarray) -> np.ndarray:
    return (s >= 0).astype(float)

def sigmoid(s: np.ndarray, beta: float) -> np.ndarray:
    z = np.clip(beta * s, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def tanh_act(s: np.ndarray) -> np.ndarray:
    return np.tanh(s)

def sin_act(s: np.ndarray) -> np.ndarray:
    return np.sin(s)

def sign_act(s: np.ndarray) -> np.ndarray:
    # -1, 0, 1
    return np.sign(s)

def relu_act(s: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, s)

def leaky_relu_act(s: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(s > 0.0, s, alpha * s)


@dataclass(frozen=True)
class LRSchedule:
    """Cosine learning-rate schedule from the assignment sheet (cosine annealing)."""
    eta_min: float
    eta_max: float

    def value(self, n: int, n_max: int) -> float:
        # Typical cosine schedule:
        # eta(n) = eta_min + 0.5*(eta_max-eta_min)*(1+cos(pi*n/n_max))
        if n_max <= 0:
            return float(self.eta_max)
        n = max(0, min(n, n_max))
        return float(self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(np.pi * n / n_max)))


class SingleNeuron:
    """
    Implements:
      Δw = η (d - y) f'(s) x
      s = w^T x
    with bias via x=[x,y,1].

    Activations for evaluation:
      heaviside, logistic, tanh, sin, sign, relu, leaky_relu

    Training required by rubric:
      heaviside (assume f'(s)=1), logistic (true derivative)

    Grade 4 option:
      - variable learning rate (cosine schedule)
      OR
      - tanh/sin training (also supported here)
    """

    def __init__(
        self,
        lr: float = 0.05,
        activation: str = "heaviside",
        beta: float = 2.0,
        leaky_alpha: float = 0.01,
        seed: int = 0,
    ):
        self.lr = float(lr)
        self.activation = activation.lower()
        self.beta = float(beta)
        self.leaky_alpha = float(leaky_alpha)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.3, size=(3,))  # [w1, w2, b]

    def set_params(
        self,
        lr: float | None = None,
        activation: str | None = None,
        beta: float | None = None,
        leaky_alpha: float | None = None,
    ) -> None:
        if lr is not None:
            self.lr = float(lr)
        if activation is not None:
            self.activation = activation.lower()
        if beta is not None:
            self.beta = float(beta)
        if leaky_alpha is not None:
            self.leaky_alpha = float(leaky_alpha)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw linear score s = w^T x (with bias)."""
        Xb = add_bias(X)
        return Xb @ self.w

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Activation output y = f(s)."""
        s = self.score(X)
        a = self.activation
        if a == "heaviside":
            return heaviside(s)
        if a == "logistic":
            return sigmoid(s, beta=self.beta)
        if a == "tanh":
            return tanh_act(s)
        if a == "sin":
            return sin_act(s)
        if a == "sign":
            return sign_act(s)
        if a == "relu":
            return relu_act(s)
        if a in ("leaky_relu", "lrelu", "leaky"):
            return leaky_relu_act(s, alpha=self.leaky_alpha)
        raise ValueError("Unsupported activation.")

    # ----------------------------
    # Target encoding + classification
    # ----------------------------
    def _encode_targets_for_training(self, d01: np.ndarray) -> np.ndarray:
        """
        For tanh/sin/sign it is more natural to train with targets in {-1, +1}.
        For heaviside/logistic/relu/leaky we train with {0,1}.
        """
        a = self.activation
        if a in ("tanh", "sin", "sign"):
            return 2.0 * d01.astype(float) - 1.0
        return d01.astype(float)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """
        Convert activation outputs to class labels 0/1 for visualization & accuracy.
        """
        a = self.activation
        s = self.score(X)

        if a in ("heaviside",):
            return (s >= 0).astype(float)

        if a in ("logistic",):
            return (sigmoid(s, beta=self.beta) >= 0.5).astype(float)

        if a in ("tanh",):
            return (tanh_act(s) >= 0.0).astype(float)

        if a in ("sin",):
            # will create multiple stripes because sin crosses 0 repeatedly — that’s OK for “evaluation”
            return (sin_act(s) >= 0.0).astype(float)

        if a in ("sign",):
            return (sign_act(s) > 0.0).astype(float)

        if a in ("relu", "leaky_relu", "lrelu", "leaky"):
            # boundary is still s=0
            return (s >= 0.0).astype(float)

        raise ValueError("Unsupported activation.")

    # ----------------------------
    # Training
    # ----------------------------
    def train_sgd(
        self,
        X: np.ndarray,
        d01: np.ndarray,
        epochs: int = 30,
        shuffle: bool = True,
        lr_schedule: LRSchedule | None = None,
    ) -> list[float]:
        """
        SGD training using:
          Δw = η (d - y) f'(s) x

        - heaviside: assume f'(s)=1
        - logistic : f'(s)=beta*y*(1-y)

        Optional (grade 4):
          - lr_schedule: cosine η(epoch)

        Optional (grade 4 alternative):
          - tanh/sin training: derivative implemented (not required if you choose LR schedule method)
        """
        Xb = add_bias(X)
        d = self._encode_targets_for_training(d01)

        rng = np.random.default_rng(1234)
        losses: list[float] = []

        for ep in range(int(epochs)):
            # variable LR (Method A)
            if lr_schedule is not None:
                self.lr = lr_schedule.value(ep, epochs)

            idx = np.arange(Xb.shape[0])
            if shuffle:
                rng.shuffle(idx)

            for i in idx:
                x_i = Xb[i]
                s = float(x_i @ self.w)
                a = self.activation

                # forward + derivative
                if a == "heaviside":
                    y = 1.0 if s >= 0 else 0.0
                    fprime = 1.0  # per assignment
                elif a == "logistic":
                    y = float(sigmoid(np.array([s]), beta=self.beta)[0])
                    fprime = self.beta * y * (1.0 - y)
                elif a == "tanh":
                    y = float(np.tanh(s))
                    fprime = 1.0 - y * y
                elif a == "sin":
                    y = float(np.sin(s))
                    fprime = float(np.cos(s))
                else:
                    # For grade 5 activations training is not required.
                    raise ValueError(f"Training not enabled for activation '{a}' (rubric only needs evaluation).")

                self.w += self.lr * (d[i] - y) * fprime * x_i

            # track MSE on activation output vs encoded targets
            y_out = self.forward(X)
            losses.append(float(np.mean((d - y_out) ** 2)))

        return losses

    def accuracy(self, X: np.ndarray, d01: np.ndarray) -> float:
        pred = self.predict_label(X)
        return float(np.mean((pred == d01.astype(float)).astype(float)))
