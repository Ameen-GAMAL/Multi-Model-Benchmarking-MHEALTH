"""Shared estimator interface for the from-scratch classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """Minimal scikit-learn-style contract shared by every model.

    Concrete models implement ``fit`` and ``predict_proba``; ``predict`` is
    derived from the probabilities so the argmax logic lives in one place.
    """

    #: populated by ``fit`` -- the sorted array of class labels seen in training
    classes_: np.ndarray

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, n_classes)``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most probable class label for each sample."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def name(self) -> str:
        return type(self).__name__
