"""Classification metrics implemented from scratch with NumPy.

Everything a scikit-learn user would reach for -- accuracy, precision, recall,
F1 and ROC-AUC -- reimplemented here so the benchmark depends on nothing but
NumPy.  Keeping the metrics first-principles is deliberate: it forces the maths
(confusion matrices, one-vs-rest averaging, the rank formulation of AUC) to be
explicit and auditable.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "roc_auc_ovr",
    "classification_report",
]


def _as_int(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).astype(int).ravel()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that match the ground truth."""
    y_true, y_pred = _as_int(y_true), _as_int(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Return the ``n_classes x n_classes`` confusion matrix.

    Rows are the true class, columns are the predicted class.  Labels are
    assumed to be contiguous integers in ``[0, n_classes)``.
    """
    y_true, y_pred = _as_int(y_true), _as_int(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # np.add.at gives an in-place scatter-add, i.e. cm[t, p] += 1 vectorised.
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
    average: str = "macro",
) -> tuple[float, float, float]:
    """Precision, recall and F1.

    ``average='macro'`` treats every class equally (the right choice for an
    imbalanced activity-recognition problem), ``'micro'`` aggregates over all
    samples, and ``None`` returns the per-class arrays.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    if average is None:
        return precision, recall, f1
    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())
    if average == "micro":
        total_tp = tp.sum()
        p = total_tp / (total_tp + fp.sum()) if (total_tp + fp.sum()) else 0.0
        r = total_tp / (total_tp + fn.sum()) if (total_tp + fn.sum()) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return float(p), float(r), float(f)
    raise ValueError(f"Unknown average={average!r}")


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC for one binary problem via the Mann-Whitney U / rank statistic.

    AUC = P(score(positive) > score(negative)).  Ranking the scores and summing
    the ranks of the positive class gives this probability exactly, with the
    average-rank convention handling ties.
    """
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # AUC undefined when a class is absent

    order = np.argsort(scores, kind="mergesort")
    ranked = scores[order]
    ranks = np.empty(len(scores), dtype=float)

    # Assign average ranks to tied groups of scores.
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j] == ranked[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
        i = j

    sum_pos_ranks = ranks[labels].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_auc_ovr(
    y_true: np.ndarray, proba: np.ndarray, average: str = "macro"
) -> float:
    """One-vs-rest multiclass ROC-AUC.

    ``proba`` has shape ``(n_samples, n_classes)``.  Each column is scored
    against the rest; classes absent from ``y_true`` are skipped.
    """
    y_true = _as_int(y_true)
    proba = np.asarray(proba, dtype=float)
    n_classes = proba.shape[1]
    aucs = []
    for c in range(n_classes):
        auc = _binary_auc(proba[:, c], y_true == c)
        if not np.isnan(auc):
            aucs.append(auc)
    if not aucs:
        return float("nan")
    return float(np.mean(aucs)) if average == "macro" else float(aucs)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    n_classes: int | None = None,
) -> dict:
    """Bundle the headline metrics into a single dictionary."""
    acc = accuracy(y_true, y_pred)
    p, r, f1 = precision_recall_f1(y_true, y_pred, n_classes, average="macro")
    report = {
        "accuracy": acc,
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }
    if proba is not None:
        report["roc_auc_ovr_macro"] = roc_auc_ovr(y_true, proba)
    return report
