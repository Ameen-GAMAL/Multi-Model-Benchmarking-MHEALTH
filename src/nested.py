"""Nested subject evaluation for the four from-scratch classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .metrics import classification_report, confusion_matrix, precision_recall_f1
from .models import GaussianNaiveBayes, KNNClassifier, LogisticRegression, NeuralNetwork
from .models.base import BaseClassifier
from .preprocessing import StandardScaler

MODEL_TYPES = {
    "KNN": KNNClassifier,
    "Logistic Regression": LogisticRegression,
    "Naive Bayes": GaussianNaiveBayes,
    "Neural Network": NeuralNetwork,
}
MODEL_PARAMETERS = {
    "KNN": ({"k": 1}, {"k": 7}),
    "Logistic Regression": (
        {"lr": 0.35, "n_iters": 220, "l2": 0.001},
        {"lr": 0.35, "n_iters": 350, "l2": 0.0001},
    ),
    "Naive Bayes": ({"var_smoothing": 1e-9}, {"var_smoothing": 1e-4}),
    "Neural Network": (
        {"hidden_layers": (48,), "n_epochs": 20},
        {"hidden_layers": (64, 32), "n_epochs": 30},
    ),
}
METRICS = ("accuracy", "f1_macro", "roc_auc_ovr_macro")


@dataclass(frozen=True)
class Candidate:
    model: str
    feature_set: str
    parameters: dict

    def create(self, seed: int) -> BaseClassifier:
        parameters = dict(self.parameters)
        if self.model in ("Logistic Regression", "Neural Network"):
            parameters["random_state"] = seed
        return MODEL_TYPES[self.model](**parameters)

    def as_dict(self) -> dict:
        return {"model": self.model, "feature_set": self.feature_set, "parameters": self.parameters}


def grouped_folds(groups: np.ndarray, outer_subject: int, folds: int, seed: int):
    """Return outer-training indices and group-exclusive inner index pairs."""
    outer_train = np.flatnonzero(groups != outer_subject)
    subjects = np.unique(groups[outer_train])
    if len(subjects) < 2 or folds < 2:
        raise ValueError("Nested evaluation needs at least three subjects and two inner folds.")
    np.random.default_rng(seed).shuffle(subjects)
    inner = []
    for validation_subjects in np.array_split(subjects, min(folds, len(subjects))):
        validation = np.isin(groups[outer_train], validation_subjects)
        inner.append((outer_train[~validation], outer_train[validation]))
    return outer_train, inner


def _sample_train(indices: np.ndarray, y: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    if cap is None:
        return indices
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y[indices]):
        rows = indices[y[indices] == label]
        selected.append(rng.choice(rows, min(cap, len(rows)), replace=False))
    return np.sort(np.concatenate(selected))


def aligned_probabilities(model: BaseClassifier, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Project trained-class probabilities onto the benchmark's fixed class order."""
    probabilities = np.zeros((len(X), n_classes))
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(X)
    return probabilities


def _fit_and_predict(candidate, X_train, X_test, y_train, n_classes, seed):
    model = candidate.create(seed)
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probabilities = aligned_probabilities(model, X_test, n_classes)
    predicted = probabilities.argmax(axis=1)
    predict_seconds = perf_counter() - start
    return predicted, probabilities, fit_seconds, predict_seconds


def _scale(X: np.ndarray, train: np.ndarray, test: np.ndarray):
    scaler = StandardScaler().fit(X[train])
    return scaler.transform(X[train]), scaler.transform(X[test])


def _describe_split(ids, groups, y, classes):
    return {
        "subjects": np.unique(groups[ids]).tolist(),
        "windows": int(len(ids)),
        "class_counts": {str(label): int(np.sum(y[ids] == index)) for index, label in enumerate(classes)},
    }


def _tune(features, y, inner, n_classes, seed):
    # The cache is scoped to this outer fold, and every scaler only sees its inner train rows.
    scaled = {
        name: [_scale(X, train, val) for train, val in inner]
        for name, X in features.items()
    }
    candidates = []
    for model, choices in MODEL_PARAMETERS.items():
        for feature_set in features:
            for params in choices:
                candidate = Candidate(model, feature_set, params)
                if model == "KNN" and any(params["k"] > len(train) for train, _ in inner):
                    continue
                scores = []
                for fold, (train, validation) in enumerate(inner):
                    X_train, X_val = scaled[feature_set][fold]
                    predicted, _, _, _ = _fit_and_predict(
                        candidate, X_train, X_val, y[train], n_classes, seed + fold
                    )
                    scores.append(precision_recall_f1(y[validation], predicted, n_classes)[2])
                candidates.append((candidate, scores, float(np.mean(scores))))
    return candidates


def run_nested(features, y, groups, classes, *, inner_folds, repetitions, seed, cap=None, progress=None):
    """Tune on inner subjects, then report one outer score per held-out person."""
    subjects = np.unique(groups)
    if len(subjects) < 3 or repetitions < 1 or (cap is not None and cap < 1):
        raise ValueError("Need three subjects, at least one repetition, and a positive training cap if set.")
    n_classes = len(classes)
    n_rows = len(y)
    if n_classes < 2 or len(groups) != n_rows or any(len(X) != n_rows for X in features.values()):
        raise ValueError("Features, groups and labels must share rows and cover at least two activities.")

    rows, folds = [], []
    confusion = {name: np.zeros((n_classes, n_classes), dtype=int) for name in [*MODEL_TYPES, "Selected pipeline"]}
    for outer_subject in subjects:
        if progress:
            progress(int(outer_subject))
        outer_train, inner = grouped_folds(groups, int(outer_subject), inner_folds, seed)
        inner = [(_sample_train(train, y, cap, seed + fold), val) for fold, (train, val) in enumerate(inner)]
        outer_test = np.flatnonzero(groups == outer_subject)
        outer_train = _sample_train(outer_train, y, cap, seed)
        tuning = _tune(features, y, inner, n_classes, seed)

        # Stable ordering breaks exact ties in favour of the declared, simpler candidate.
        selected_pipeline = max(tuning, key=lambda item: item[2])[0]
        selected_by_model = {
            name: max((item for item in tuning if item[0].model == name), key=lambda item: item[2])[0]
            for name in MODEL_TYPES
        }
        scaled = {name: _scale(X, outer_train, outer_test) for name, X in features.items()}
        for name, candidate in selected_by_model.items():
            X_train, X_test = scaled[candidate.feature_set]
            fit_times, predict_times = [], []
            for _ in range(repetitions):
                predicted, prob, fit_s, predict_s = _fit_and_predict(
                    candidate, X_train, X_test, y[outer_train], n_classes, seed
                )
                fit_times.append(fit_s)
                predict_times.append(predict_s)

            report = classification_report(y[outer_test], predicted, prob, n_classes=n_classes)
            cm = confusion_matrix(y[outer_test], predicted, n_classes)
            record = {
                "subject": int(outer_subject),
                "model": name,
                "feature_set": candidate.feature_set,
                "parameters": candidate.parameters,
                "train_windows": len(outer_train),
                "test_windows": len(outer_test),
                **{key: float(report[key]) if np.isfinite(report[key]) else None for key in METRICS},
                "fit_seconds": fit_times,
                "predict_seconds": predict_times,
            }
            rows.append(record)
            confusion[name] += cm
            if candidate == selected_pipeline:
                rows.append({**record, "model": "Selected pipeline", "selected_model": name})
                confusion["Selected pipeline"] += cm

        folds.append({
            "held_out_subject": int(outer_subject),
            "outer_train": _describe_split(outer_train, groups, y, classes),
            "outer_test": _describe_split(outer_test, groups, y, classes),
            "inner": [
                {"train": _describe_split(train, groups, y, classes),
                 "validation": _describe_split(val, groups, y, classes)}
                for train, val in inner
            ],
            "inner_scores": [
                {**candidate.as_dict(), "fold_f1_macro": scores, "mean_f1_macro": mean}
                for candidate, scores, mean in tuning
            ],
            "selected_pipeline": selected_pipeline.as_dict(),
            "selected_by_model": {name: candidate.as_dict() for name, candidate in selected_by_model.items()},
        })
    return rows, folds, {name: matrix.tolist() for name, matrix in confusion.items()}


def summarize(rows, *, seed: int, bootstrap_replicates: int = 2000) -> dict:
    """Weight subjects equally and resample subjects rather than overlapping windows."""
    summaries = {}
    for name in dict.fromkeys(row["model"] for row in rows):
        group = [row for row in rows if row["model"] == name]
        report = {}
        for metric in METRICS:
            values = np.array([row[metric] for row in group if row[metric] is not None])
            if not len(values):
                report[metric] = {"mean": None, "bootstrap_95_ci": None, "subjects": 0}
                continue
            samples = np.random.default_rng(seed).choice(values, size=(bootstrap_replicates, len(values)))
            interval = np.quantile(samples.mean(axis=1), (0.025, 0.975))
            report[metric] = {"mean": float(values.mean()), "bootstrap_95_ci": interval.tolist(), "subjects": len(values)}
        for metric in ("fit_seconds", "predict_seconds"):
            times = [value for row in group for value in row[metric]]
            report[metric] = timing_summary(times)
        summaries[name] = report
    return summaries


def timing_summary(times: list[float]) -> dict:
    q1, median, q3 = np.quantile(times, (0.25, 0.5, 0.75))
    return {"median": float(median), "iqr": [float(q1), float(q3)], "samples": len(times)}
