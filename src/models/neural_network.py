"""A feed-forward neural network (MLP) with backpropagation, from scratch.

Supports an arbitrary stack of ReLU hidden layers followed by a softmax output,
trained with mini-batch stochastic gradient descent.  Forward activations are
cached per layer so the backward pass can reuse them, which is the whole point
of backprop -- gradients flow right-to-left through the cached graph.
"""
from __future__ import annotations

import numpy as np

from .base import BaseClassifier
from ..preprocessing import one_hot


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetwork(BaseClassifier):
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64),
        lr: float = 0.05,
        n_epochs: int = 60,
        batch_size: int = 128,
        l2: float = 1e-4,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history_: list[float] = []

    # ------------------------------------------------------------------ init
    def _init_params(self, n_features: int, n_classes: int) -> None:
        rng = np.random.default_rng(self.random_state)
        sizes = [n_features, *self.hidden_layers, n_classes]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation keeps the variance of activations stable
            # through ReLU layers.
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    # --------------------------------------------------------------- forward
    def _forward(self, X: np.ndarray) -> tuple[list, list]:
        """Return (pre-activations, activations) caches for backprop."""
        activations = [X]
        pre_acts = []
        a = X
        n_layers = len(self.W)
        for i in range(n_layers):
            z = a @ self.W[i] + self.b[i]
            pre_acts.append(z)
            if i < n_layers - 1:
                a = np.maximum(0.0, z)          # ReLU on hidden layers
            else:
                a = _softmax(z)                 # softmax on the output layer
            activations.append(a)
        return pre_acts, activations

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetwork":
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        Y = one_hot(y_idx, n_classes)

        n_samples, n_features = X.shape
        self._init_params(n_features, n_classes)

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            Xs, Ys = X[perm], Y[perm]
            epoch_loss = 0.0

            for start in range(0, n_samples, self.batch_size):
                xb = Xs[start : start + self.batch_size]
                yb = Ys[start : start + self.batch_size]
                bsz = len(xb)

                pre_acts, activations = self._forward(xb)
                probs = activations[-1]
                epoch_loss += -np.sum(yb * np.log(probs + 1e-12))

                # Backprop: dL/dz at the output is (P - Y) for softmax + CE.
                delta = (probs - yb) / bsz
                grads_W = [None] * len(self.W)
                grads_b = [None] * len(self.b)
                for i in reversed(range(len(self.W))):
                    a_prev = activations[i]
                    grads_W[i] = a_prev.T @ delta + self.l2 * self.W[i]
                    grads_b[i] = delta.sum(axis=0)
                    if i > 0:  # propagate through the ReLU of the layer below
                        relu_grad = (pre_acts[i - 1] > 0).astype(float)
                        delta = (delta @ self.W[i].T) * relu_grad

                for i in range(len(self.W)):
                    self.W[i] -= self.lr * grads_W[i]
                    self.b[i] -= self.lr * grads_b[i]

            self.loss_history_.append(epoch_loss / n_samples)
            if self.verbose and epoch % max(1, self.n_epochs // 10) == 0:
                print(f"[NN] epoch {epoch:3d}  loss={self.loss_history_[-1]:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, activations = self._forward(np.asarray(X, dtype=float))
        return activations[-1]
