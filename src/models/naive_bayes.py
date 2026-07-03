"""Gaussian Naive Bayes, implemented from scratch.

Naive Bayes assumes the features are conditionally independent given the class,
so each class-conditional density factorises into a product of per-feature
Gaussians.  We work in log-space throughout to avoid underflow when multiplying
~180 small likelihoods together.
"""
from __future__ import annotations

import numpy as np

from .base import BaseClassifier


class GaussianNaiveBayes(BaseClassifier):
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        # A floor added to every variance, mirroring sklearn's var_smoothing,
        # so that near-constant features do not blow up the Gaussian term.
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        self.classes_ = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))   # per-class means
        self.var_ = np.zeros((n_classes, n_features))     # per-class variances
        self.log_prior_ = np.zeros(n_classes)

        epsilon = self.var_smoothing * X.var(axis=0).max()
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0) + epsilon
            self.log_prior_[i] = np.log(len(Xc) / len(y))  # P(class)
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(class) + log P(x | class) for every sample and class."""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        jll = np.zeros((n_samples, len(self.classes_)))
        for i in range(len(self.classes_)):
            # log of the Gaussian pdf, summed over independent features.
            log_det = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i]))
            sq = -0.5 * np.sum(((X - self.theta_[i]) ** 2) / self.var_[i], axis=1)
            jll[:, i] = self.log_prior_[i] + log_det + sq
        return jll

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        # Normalise via log-sum-exp to turn joint log-likelihoods into
        # posterior probabilities P(class | x).
        log_norm = jll.max(axis=1, keepdims=True)
        log_prob = jll - (log_norm + np.log(
            np.sum(np.exp(jll - log_norm), axis=1, keepdims=True)
        ))
        return np.exp(log_prob)
