"""End-to-end benchmark: build features, train four from-scratch classifiers,
and compare them on accuracy, macro-F1 and macro ROC-AUC.

Examples
--------
    python scripts/run_benchmark.py                 # full run, saves plots
    python scripts/run_benchmark.py --no-plots      # metrics only
    python scripts/run_benchmark.py --subjects 1 2 3  --max-per-class 400

Outputs (written to ``results/``):
    benchmark_results.csv / .md            headline metrics per model
    metrics_comparison.png                 grouped bar chart
    confusion_matrix_<best>.png            confusion matrix of the best model
    feature_engineering_comparison.png     basic vs full feature F1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make ``src`` importable when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (  # noqa: E402
    ACTIVITY_NAMES,
    build_windows,
    relabel_contiguous,
)
from src.metrics import (  # noqa: E402
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    roc_auc_ovr,
)
from src.models import (  # noqa: E402
    GaussianNaiveBayes,
    KNNClassifier,
    LogisticRegression,
    NeuralNetwork,
)
from src.preprocessing import StandardScaler, train_test_split  # noqa: E402


def build_models(seed: int) -> dict:
    """Instantiate the four classifiers with sensible defaults."""
    return {
        "KNN": KNNClassifier(k=7, weights="distance"),
        "Logistic Regression": LogisticRegression(
            lr=0.5, n_iters=800, l2=1e-4, random_state=seed
        ),
        "Naive Bayes": GaussianNaiveBayes(),
        "Neural Network": NeuralNetwork(
            hidden_layers=(128, 64), lr=0.05, n_epochs=60,
            batch_size=128, l2=1e-4, random_state=seed
        ),
    }


def make_split(X, y, groups, args, seed):
    """Split into train/test either by subject or by stratified random sampling.

    ``split='subject'`` (the default) is a *leave-subjects-out* evaluation: the
    test subjects are never seen in training.  This is the honest protocol for
    human-activity recognition -- a plain random split lets near-identical,
    overlapping windows from the same activity segment leak across the split and
    inflates every score toward 100%.
    """
    if args.split == "subject":
        subjects = np.unique(groups)
        if args.test_subjects:
            test_subjects = np.array(sorted(args.test_subjects))
        else:
            n_test = max(1, int(round(len(subjects) * args.test_size)))
            test_subjects = subjects[-n_test:]
        test_mask = np.isin(groups, test_subjects)
        train_mask = ~test_mask
        info = f"leave-subjects-out (held-out subjects: {list(test_subjects)})"
        return (X[train_mask], X[test_mask], y[train_mask], y[test_mask], info)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=True, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te, "stratified random split"


def subsample_per_class(X, y, groups, max_per_class, seed):
    """Cap the number of windows per class (keeps runs fast / balanced)."""
    if not max_per_class:
        return X, y, groups
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep], groups[keep]


def evaluate_models(models, X_tr, y_tr, X_te, y_te, n_classes):
    """Fit and score every model; return a results list and cached predictions."""
    results, cache = [], {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = model.predict_proba(X_te)
        pred = model.classes_[np.argmax(proba, axis=1)]
        pred_s = time.perf_counter() - t0

        acc = accuracy(y_te, pred)
        _, _, f1 = precision_recall_f1(y_te, pred, n_classes, average="macro")
        auc = roc_auc_ovr(y_te, proba)

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1,
            "roc_auc_ovr": auc,
            "fit_seconds": fit_s,
            "predict_seconds": pred_s,
        })
        cache[name] = {"pred": pred, "proba": proba}
        print(
            f"  {name:<20s}  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
            f"(fit {fit_s:5.2f}s)"
        )
    return results, cache


def feature_engineering_study(args, seed):
    """Compare the 'basic' (mean-only) vs 'full' feature sets across models.

    Both representations are built from identical windows and split with the same
    seed, so any difference is attributable to the feature engineering alone.
    """
    print("\n[2/2] Feature-engineering study (basic vs full)...")
    scores = {}
    for fs in ("basic", "full"):
        X, y, groups, _ = build_windows(
            data_dir=args.data_dir, subjects=args.subjects,
            window_size=args.window_size, stride=args.stride,
            feature_set=fs, verbose=False,
        )
        y, classes = relabel_contiguous(y)
        X, y, groups = subsample_per_class(X, y, groups, args.max_per_class, seed)
        X_tr, X_te, y_tr, y_te, _ = make_split(X, y, groups, args, seed)
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

        n_classes = len(classes)
        fs_scores = {}
        for name, model in build_models(seed).items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            _, _, f1 = precision_recall_f1(y_te, pred, n_classes, average="macro")
            fs_scores[name] = f1
        scores[fs] = fs_scores
        n_feat = X.shape[1]
        print(f"  {fs:>5s} features ({n_feat:3d} dims): "
              + "  ".join(f"{k}={v:.3f}" for k, v in fs_scores.items()))
    return scores


def save_tables(results, out_dir):
    import csv

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    fields = ["model", "accuracy", "f1_macro", "roc_auc_ovr",
              "fit_seconds", "predict_seconds"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fields})

    md_path = os.path.join(out_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("| Model | Accuracy | Macro-F1 | ROC-AUC (OvR) | Fit (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in sorted(results, key=lambda x: x["f1_macro"], reverse=True):
            f.write(
                f"| {r['model']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
                f"{r['roc_auc_ovr']:.4f} | {r['fit_seconds']:.2f} |\n"
            )
    print(f"\nSaved metrics to {csv_path} and {md_path}")


def make_plots(results, cache, feat_scores, y_te, class_names, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(Skipping plots -- matplotlib unavailable: {exc})")
        return

    names = [r["model"] for r in results]

    # 1) Grouped metric comparison.
    metrics = ["accuracy", "f1_macro", "roc_auc_ovr"]
    labels = ["Accuracy", "Macro-F1", "ROC-AUC"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1) * width, [r[m] for r in results], width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on MHEALTH (windowed features)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_comparison.png"), dpi=130)
    plt.close(fig)

    # 2) Confusion matrix of the best model (by macro-F1).
    best = max(results, key=lambda r: r["f1_macro"])["model"]
    cm = confusion_matrix(y_te, cache[best]["pred"], len(class_names))
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix -- {best} (row-normalised)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_matrix_{_slug(best)}.png"), dpi=130)
    plt.close(fig)

    # 3) Feature-engineering comparison (basic vs full, macro-F1).
    if feat_scores:
        model_names = list(feat_scores["full"].keys())
        x = np.arange(len(model_names))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width / 2, [feat_scores["basic"][m] for m in model_names],
               width, label="Basic (means only)")
        ax.bar(x + width / 2, [feat_scores["full"][m] for m in model_names],
               width, label="Full (8 stats/channel)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1")
        ax.set_title("Impact of feature engineering on macro-F1")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_engineering_comparison.png"), dpi=130)
        plt.close(fig)

    print(f"Saved plots to {out_dir}/")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join("data", "raw", "MHEALTHDATASET"))
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="Subset of subject ids (1..10). Default: all.")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--split", choices=["subject", "random"], default="subject",
                   help="'subject' = leave-subjects-out (honest for HAR); "
                        "'random' = stratified random split (leaks overlapping "
                        "windows, scores inflate toward 100%%).")
    p.add_argument("--test-subjects", type=int, nargs="+", default=None,
                   help="Explicit held-out subject ids for --split subject.")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Cap windows per class (speeds up KNN).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-feature-study", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed = args.seed
    np.random.seed(seed)

    print("=" * 70)
    print("Multi-Model Benchmarking on Time-Series Sensor Data (MHEALTH)")
    print("=" * 70)

    print("\n[1/2] Building windowed feature dataset (full features)...")
    X, y, groups, names = build_windows(
        data_dir=args.data_dir, subjects=args.subjects,
        window_size=args.window_size, stride=args.stride, feature_set="full",
    )
    y, classes = relabel_contiguous(y)
    class_names = [ACTIVITY_NAMES.get(int(c), str(c)) for c in classes]
    X, y, groups = subsample_per_class(X, y, groups, args.max_per_class, seed)

    X_tr, X_te, y_tr, y_te, split_info = make_split(X, y, groups, args, seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"  split: {split_info}")
    print(f"  train={X_tr.shape[0]}  test={X_te.shape[0]}  "
          f"features={X_tr.shape[1]}  classes={len(classes)}")

    print("\nTraining & evaluating models:")
    results, cache = evaluate_models(
        build_models(seed), X_tr, y_tr, X_te, y_te, len(classes)
    )

    feat_scores = None
    if not args.no_feature_study:
        feat_scores = feature_engineering_study(args, seed)

    save_tables(results, args.out_dir)
    if not args.no_plots:
        make_plots(results, cache, feat_scores, y_te, class_names, args.out_dir)

    best = max(results, key=lambda r: r["f1_macro"])
    print(f"\nBest model by macro-F1: {best['model']} "
          f"(F1={best['f1_macro']:.4f}, acc={best['accuracy']:.4f})")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
