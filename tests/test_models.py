"""Sanity tests for the from-scratch models and metrics.

Runnable either with pytest (``pytest -q``) or directly (``python tests/test_models.py``).
Each model is checked on an easy, linearly-separable synthetic problem where a
correct implementation must reach high accuracy; the metrics are checked against
hand-computable cases.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import accuracy, confusion_matrix, precision_recall_f1, roc_auc_ovr
from src.models import (
    GaussianNaiveBayes,
    KNNClassifier,
    LogisticRegression,
    NeuralNetwork,
)
from src.preprocessing import StandardScaler, one_hot, train_test_split


def _blobs(n=300, seed=0):
    """Three well-separated Gaussian blobs in 2-D."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 6], [0, 8]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.7, size=(n, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n)
    return X, y


def test_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])
    assert abs(accuracy(y_true, y_pred) - 5 / 6) < 1e-9
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.sum() == 6 and cm[1, 2] == 1
    _, _, f1 = precision_recall_f1(y_true, y_pred, 3, average="macro")
    assert 0.0 <= f1 <= 1.0


def test_roc_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert abs(roc_auc_ovr(y, perfect) - 1.0) < 1e-9
    tie = np.full((4, 2), 0.5)
    assert abs(roc_auc_ovr(y, tie) - 0.5) < 1e-9  # ties -> chance level


def _fit_score(model):
    X, y = _blobs()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
    scaler = StandardScaler().fit(X_tr)
    model.fit(scaler.transform(X_tr), y_tr)
    return accuracy(y_te, model.predict(scaler.transform(X_te)))


def test_knn():
    assert _fit_score(KNNClassifier(k=5)) > 0.9


def test_logistic_regression():
    assert _fit_score(LogisticRegression(lr=0.5, n_iters=400, random_state=0)) > 0.9


def test_naive_bayes():
    assert _fit_score(GaussianNaiveBayes()) > 0.9


def test_neural_network():
    assert _fit_score(
        NeuralNetwork(hidden_layers=(16,), n_epochs=80, lr=0.1, random_state=0)
    ) > 0.9


def test_one_hot_roundtrip():
    y = np.array([0, 2, 1, 2])
    oh = one_hot(y, 3)
    assert oh.shape == (4, 3)
    assert np.array_equal(oh.argmax(axis=1), y)


def test_probabilities_sum_to_one():
    X, y = _blobs(n=100)
    for model in (
        KNNClassifier(k=5),
        LogisticRegression(n_iters=100, random_state=0),
        GaussianNaiveBayes(),
        NeuralNetwork(hidden_layers=(16,), n_epochs=20, random_state=0),
    ):
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
