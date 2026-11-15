import numpy as np
import pytest

from src.metrics import accuracy, precision_recall_f1, roc_auc_ovr
from src.models import KNNClassifier, GaussianNaiveBayes, LogisticRegression
from src.models.logistic_regression import _softmax


@pytest.mark.parametrize("k", [0, -1, 2.5, True])
def test_invalid_neighbor_counts(k):
    with pytest.raises(ValueError, match="positive integer"):
        KNNClassifier(k=k)


def test_knn_rejects_fewer_training_samples_than_neighbors():
    with pytest.raises(ValueError, match="exceeds"):
        KNNClassifier(k=4).fit([[0], [1], [2]], [0, 1, 1])


def test_naive_bayes_constant_features_produce_finite_probabilities():
    X = np.ones((4, 2))
    probabilities = GaussianNaiveBayes().fit(X, [0, 0, 1, 1]).predict_proba(X)
    np.testing.assert_allclose(probabilities, np.full((4, 2), 0.5))


def test_logistic_weight_gradient_matches_finite_difference():
    X = np.array([[-1.0, 0.35], [1.0, -0.5], [0.5, 0.2], [0, 1]])
    y = np.array([0, 0, 1, 2])
    initial = np.random.default_rng(22).normal(0.0, 0.01, (X.shape[1], 3))
    regularization, lr, eps = 0.07, 0.1, 1e-5
    fitted = LogisticRegression(lr=lr, n_iters=1, l2=regularization, random_state=22).fit(X, y)

    def loss(weights):
        chosen = _softmax(X @ weights)[np.arange(len(y)), y]
        return -np.log(chosen).mean() + regularization * np.square(weights).sum() / 2

    numerical = np.empty_like(initial)
    for idx in np.ndindex(initial.shape):
        perturb = np.zeros_like(initial)
        perturb[idx] = eps
        numerical[idx] = (loss(initial + perturb) - loss(initial - perturb)) / (2 * eps)
    np.testing.assert_allclose((initial - fitted.W) / lr, numerical, atol=1e-8)


def test_metrics_match_scikit_learn_reference():
    reference = pytest.importorskip("sklearn.metrics")
    y = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array([[.1, .7, .2], [.5, .3, .2], [.5, .3, .2], [.2, .6, .2], [.3, .5, .2], [.2, .3, .5]])
    predicted = probabilities.argmax(axis=1)
    assert accuracy(y, predicted) == reference.accuracy_score(y, predicted)
    assert precision_recall_f1(y, predicted, 3)[2] == pytest.approx(reference.f1_score(y, predicted, average="macro"))
    assert roc_auc_ovr(y, probabilities) == pytest.approx(reference.roc_auc_score(y, probabilities, multi_class="ovr"))
