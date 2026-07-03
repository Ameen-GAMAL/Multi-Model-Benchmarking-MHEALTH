"""Preprocessing utilities (train/test splitting, scaling) -- NumPy only."""
from __future__ import annotations

import numpy as np

__all__ = ["train_test_split", "StandardScaler", "one_hot"]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    stratify: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/test partitions.

    With ``stratify=True`` the class proportions are preserved in both splits,
    which matters here because the activity classes are unbalanced.
    """
    rng = np.random.default_rng(random_state)
    X, y = np.asarray(X), np.asarray(y)
    n = len(y)

    if stratify:
        train_idx, test_idx = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            cut = int(round(len(idx) * (1.0 - test_size)))
            train_idx.append(idx[:cut])
            test_idx.append(idx[cut:])
        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(round(n * (1.0 - test_size)))
        train_idx, test_idx = idx[:cut], idx[cut:]

    # Shuffle within each split so classes are interleaved, not blocked.
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Statistics are learned on the training set only (``fit``) and reused on the
    test set (``transform``) to avoid leaking information across the split.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # guard against constant columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Encode integer labels as one-hot rows."""
    y = np.asarray(y).astype(int).ravel()
    if n_classes is None:
        n_classes = int(y.max() + 1)
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out
