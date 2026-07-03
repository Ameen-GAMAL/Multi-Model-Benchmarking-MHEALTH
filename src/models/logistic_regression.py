"""Multinomial logistic regression (softmax regression) via gradient descent.

Trained by minimising the cross-entropy loss with full-batch gradient descent
and optional L2 regularisation.  The gradient of softmax + cross-entropy has the
famously clean form ``X^T (P - Y) / n``, which is what the update below uses.
"""
from __future__ import annotations

import numpy as np

from .base import BaseClassifier
from ..preprocessing import one_hot


def _softmax(z: np.ndarray) -> np.ndarray:
    # Subtract the row max before exponentiating for numerical stability.
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class LogisticRegression(BaseClassifier):
    def __init__(
        self,
        lr: float = 0.1,
        n_iters: int = 500,
        l2: float = 1e-4,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        Y = one_hot(y_idx, n_classes)

        n_samples, n_features = X.shape
        # Small random init breaks symmetry between class weight vectors.
        self.W = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.b = np.zeros(n_classes)

        for it in range(self.n_iters):
            logits = X @ self.W + self.b
            P = _softmax(logits)

            # Cross-entropy loss (+ L2), tracked for diagnostics/plots.
            log_likelihood = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
            loss = log_likelihood + 0.5 * self.l2 * np.sum(self.W ** 2)
            self.loss_history_.append(float(loss))

            grad_logits = (P - Y) / n_samples
            grad_W = X.T @ grad_logits + self.l2 * self.W
            grad_b = grad_logits.sum(axis=0)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if self.verbose and it % max(1, self.n_iters // 10) == 0:
                print(f"[LogReg] iter {it:4d}  loss={loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _softmax(X @ self.W + self.b)
