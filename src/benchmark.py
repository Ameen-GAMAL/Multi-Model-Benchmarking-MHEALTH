"""Run nested subject evaluation and write an auditable MHEALTH result bundle."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from .data_loader import DEFAULT_DATA_DIR, SENSOR_COLUMNS, build_windows, feature_names, relabel_contiguous, subject_files
from .dataset import read_verified_source, sha256_file
from .nested import MODEL_PARAMETERS, run_nested, summarize, timing_summary


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        options = {"cwd": root, "capture_output": True, "text": True, "check": True}
        commit = subprocess.run(["git", "rev-parse", "HEAD"], **options).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], **options).stdout)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(file.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(file.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def _inputs(data_dir, subjects, synthetic):
    source = None if synthetic else read_verified_source(data_dir)
    files = []
    for sid, filename in subject_files(str(data_dir), subjects):
        path = Path(filename)
        record = {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        if source:
            expected = source["files"].get(path.name)
            if expected != {"sha256": record["sha256"], "bytes": record["bytes"]}:
                raise ValueError(f"Dataset input differed from the verified archive: {path.name}.")
        files.append({"subject": sid, **record})
    return {
        "kind": "synthetic" if synthetic else "uci_mhealth" if source else "unverified_local_logs",
        "archive": {key: value for key, value in source.items() if key != "files"} if source else None,
        "files": files,
    }


def _fixture(directory: Path, seed: int, window: int, stride: int) -> None:
    """Generate deterministic non-personal logs through the same parser as UCI data."""
    rng = np.random.default_rng(seed)
    samples = max(4 * window, 4 * stride)
    channels = np.arange(1, len(SENSOR_COLUMNS) + 1)
    for subject in range(1, 5):
        rows = []
        for label in (1, 2, 3):
            t = np.arange(samples) / max(1, window)
            sensor = (0.25 * label * channels[None, :] + 0.1 * subject
                      + np.sin(t[:, None] * label) + rng.normal(0, 0.2, (samples, len(channels))))
            rows.append(np.column_stack((sensor, np.full(samples, label))))
        np.savetxt(directory / f"mHealth_subject{subject}.log", np.vstack(rows), fmt="%.8g", delimiter="\t")


def _json(target: Path, data) -> None:
    target.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_results(out: Path, rows, summaries, confusion, manifest) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = ["outer_folds.csv", "summary.json", "confusion_matrices.json", "benchmark_results.md"]
    fields = ["subject", "model", "selected_model", "feature_set", "parameters", "train_windows", "test_windows",
              "accuracy", "f1_macro", "roc_auc_ovr_macro", "fit_seconds", "predict_seconds"]
    with (out / files[0]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            converted = {**row, **{name: json.dumps(row[name]) for name in ("parameters", "fit_seconds", "predict_seconds")}}
            writer.writerow(converted)
    _json(out / files[1], {"dataset_kind": manifest["dataset"]["kind"], "models": summaries})
    _json(out / files[2], {"class_order": manifest["class_order"], "matrices": confusion})

    def f1_interval(metrics):
        low, high = metrics["f1_macro"]["bootstrap_95_ci"]
        return f"{metrics['f1_macro']['mean']:.4f} ({low:.4f}–{high:.4f})"

    lines = [
        "# Nested subject evaluation", "",
        f"Dataset: `{manifest['dataset']['kind']}`. Each subject contributes one held-out score.", "",
        "| Model or selection policy | Accuracy | Macro-F1 (95% subject bootstrap) | Median fit (s) | Median predict (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summaries.items():
        lines.append(f"| {name} | {metrics['accuracy']['mean']:.4f} | {f1_interval(metrics)} | "
                     f"{metrics['fit_seconds']['median']:.4f} | {metrics['predict_seconds']['median']:.4f} |")
    feature_timing = manifest["timings_seconds"]["feature_build"]
    q1, q3 = feature_timing["iqr"]
    lines.extend(["", f"Feature construction median: {feature_timing['median']:.3f} s "
                  f"(Q1–Q3 {q1:.3f}–{q3:.3f} s, {feature_timing['samples']} repetitions).", "",
                  "See [the run manifest](run_manifest.json) for split membership, environment and output hashes.",
                  "Intervals are descriptive across the observed subjects with overlapping training folds."])
    (out / files[3]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["outputs"] = {name: sha256_file(out / name) for name in files}
    _json(out / "run_manifest.json", manifest)


def _run(args, data_dir, argv):
    dataset = _inputs(data_dir, args.subjects, args.synthetic)
    git = _git_state()
    source_hash = _source_fingerprint()
    started = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=args.threads, user_api="blas"):
        pool_details = [{k: v for k, v in info.items() if k != "filepath"} for info in threadpool_info()]
        feature_seconds = []
        for repeat in range(args.timing_repetitions):
            build_start = perf_counter()
            X, y, groups, names = build_windows(
                str(data_dir), args.subjects, args.window_size, args.stride, verbose=repeat == 0
            )
            y, classes = relabel_contiguous(y)
            mean_indices = [names.index(name) for name in feature_names("basic")]
            features = {"basic": X[:, mean_indices], "full": X}
            feature_seconds.append(perf_counter() - build_start)
        evaluation_start = perf_counter()
        rows, folds, confusion = run_nested(
            features, y, groups, classes,
            inner_folds=args.inner_folds, repetitions=args.timing_repetitions, seed=args.seed,
            cap=args.max_per_class, progress=lambda subject: print(f"Held-out subject {subject}", flush=True),
        )
        evaluation_seconds = perf_counter() - evaluation_start
    manifest = {
        "schema_version": 1,
        "started_utc": started,
        "command": ["mhealth-benchmark", *argv],
        "arguments": argv,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git": git,
        "source_sha256": source_hash,
        "dataset": dataset,
        "class_order": classes.tolist(),
        "feature_order": {"basic": feature_names("basic"), "full": names},
        "sample_count": len(y),
        "candidate_space": MODEL_PARAMETERS,
        "selection": "Mean inner-fold macro-F1, fixed class order, first declared candidate on exact ties",
        "outer": "Leave one subject out. Repetitions time the same fixed-seed fit and predictions.",
        "bootstrap": {"unit": "outer held-out subject", "replicates": 2000, "interval": "percentile 95%"},
        "cycles": folds,
        "timings_seconds": {
            "feature_build": {**timing_summary(feature_seconds), "observations": feature_seconds},
            "nested_evaluation": evaluation_seconds,
        },
        "environment": {
            "python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__,
            "threadpoolctl": __import__("threadpoolctl").__version__, "threadpools": pool_details,
        },
    }
    save_results(args.out_dir, rows, summarize(rows, seed=args.seed), confusion, manifest)
    print(f"Wrote nested evaluation to {args.out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, help="Optional cap applied to training rows only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1, help="BLAS thread limit for reproducible small-matrix timings.")
    parser.add_argument("--timing-repetitions", type=int, default=3)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="Run four synthetic subjects without downloading UCI data.")
    provided = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(provided)
    args.out_dir = args.out_dir or Path("results/synthetic" if args.synthetic else "results/nested")
    if args.out_dir.resolve() == Path("results").resolve():
        parser.error("Choose a result subdirectory to preserve the historical single-split artifacts.")
    if min(args.window_size, args.stride, args.threads, args.timing_repetitions, args.seed + 1) < 1 or args.inner_folds < 2:
        parser.error("Window size, stride, thread count and repetitions must be positive; seed >= 0 and inner folds >= 2.")
    if args.max_per_class is not None and args.max_per_class < 1:
        parser.error("--max-per-class requires a positive cap.")
    try:
        if args.synthetic:
            with tempfile.TemporaryDirectory(prefix="mhealth-synthetic-") as directory:
                path = Path(directory)
                _fixture(path, args.seed, args.window_size, args.stride)
                _run(args, path, provided)
        else:
            _run(args, args.data_dir, provided)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Benchmark failed. {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
