"""k-Nearest Neighbours, implemented from scratch.

The only non-trivial part is doing the distance computation without a Python
loop over every train/test pair.  We use the identity

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a . b

so all pairwise squared distances come from a single matrix product, which
keeps KNN tractable on several thousand windowed samples.
"""
from __future__ import annotations

import numpy as np
from numbers import Integral

from .base import BaseClassifier


class KNNClassifier(BaseClassifier):
    def __init__(self, k: int = 5, weights: str = "distance") -> None:
        """
        Parameters
        ----------
        k:
            Number of neighbours that vote on each prediction.
        weights:
            ``'uniform'`` -- every neighbour votes equally.
            ``'distance'`` -- closer neighbours get proportionally more weight,
            which usually helps on noisy sensor features.
        """
        if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"Unknown weights={weights!r}")
        self.k = int(k)
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        # KNN is a lazy learner: "training" just memorises the data.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("Training features and labels must have compatible shapes.")
        if len(X) < self.k:
            raise ValueError(f"k={self.k} exceeds the {len(X)} training samples.")
        self._X, self._y = X, y
        self._train_norm = np.einsum("ij,ij->i", X, X)
        self.classes_ = np.unique(self._y)
        self._class_index = {c: i for i, c in enumerate(self.classes_)}
        return self

    def _squared_distances(self, X: np.ndarray) -> np.ndarray:
        """Pairwise squared Euclidean distances, shape (n_test, n_train)."""
        X = np.asarray(X, dtype=float)
        x_norm = np.sum(X ** 2, axis=1, keepdims=True)          # (n_test, 1)
        cross = X @ self._X.T                                   # (n_test, n_train)
        d2 = x_norm + self._train_norm[None, :] - 2.0 * cross
        return np.maximum(d2, 0.0)  # clip tiny negatives from float error

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Bound the distance/partition arrays on larger subject folds.
        X = np.asarray(X, dtype=float)
        if len(X) <= 256:
            return self._predict_batch(X)
        return np.concatenate([self._predict_batch(X[start:start + 256]) for start in range(0, len(X), 256)])

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        d2 = self._squared_distances(X)
        n_test = d2.shape[0]
        n_classes = len(self.classes_)

        # Indices of the k smallest distances per row (unordered is fine).
        knn_idx = np.argpartition(d2, kth=self.k - 1, axis=1)[:, : self.k]
        rows = np.arange(n_test)[:, None]
        neigh_labels = self._y[knn_idx]                         # (n_test, k)

        if self.weights == "distance":
            neigh_d2 = d2[rows, knn_idx]
            vote = 1.0 / (np.sqrt(neigh_d2) + 1e-12)
        else:
            vote = np.ones_like(knn_idx, dtype=float)

        proba = np.zeros((n_test, n_classes), dtype=float)
        # Accumulate each neighbour's vote into its class column.
        for i, c in enumerate(self.classes_):
            proba[:, i] = np.where(neigh_labels == c, vote, 0.0).sum(axis=1)

        proba /= proba.sum(axis=1, keepdims=True) + 1e-12
        return proba
