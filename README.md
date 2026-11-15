# From-scratch activity recognition on MHEALTH

[![Benchmark checks](https://github.com/Ameen-GAMAL/Multi-Model-Benchmarking-MHEALTH/actions/workflows/ci.yml/badge.svg)](https://github.com/Ameen-GAMAL/Multi-Model-Benchmarking-MHEALTH/actions/workflows/ci.yml)

Four classifiers implemented in NumPy predict 12 activities from short windows of 23-channel, 50 Hz wearable-sensor logs. The benchmark compares k-nearest neighbours, multinomial logistic regression, Gaussian naive Bayes, and a feed-forward neural network on people kept separate from model selection and training.

## Measured results

The [full nested run](results/nested/benchmark_results.md) evaluated 5,213 windows from all 10 UCI MHEALTH subjects. In each outer fold, the other nine subjects supplied three grouped development folds for selecting the feature set and hyperparameters. The pipeline that also selected the classifier family chose logistic regression with full features in all ten folds. Its held-out mean macro-F1 was **0.9577**, with a descriptive subject-bootstrap interval of **0.9302–0.9818**.

[Per-subject scores](results/nested/outer_folds.csv) · [Aggregate JSON](results/nested/summary.json) · [Confusion matrices](results/nested/confusion_matrices.json) · [Run manifest](results/nested/run_manifest.json)

The manifest records the run command, UCI archive and file checksums, fold membership and class counts, candidate scores, model and feature selections, Python and dependency versions, hardware and BLAS settings, the producing source commit and SHA-256, and output hashes.

The earlier fixed-split [table](results/benchmark_results.md) and the [feature study](results/feature_engineering_comparison.png) remain available as historical experiments. They describe subjects 8–10 only, and the earlier fit times lack a recorded hardware context.

## Run it

Python 3.9–3.12 is covered by the configured CI matrix. From the repository root:

```sh
python -m pip install -c constraints.txt .
mhealth-download
mhealth-benchmark
```

The downloader uses the pinned official archive and verifies its size, SHA-256, and exact member list. A separately obtained copy can be passed with `mhealth-download --archive /path/to/mhealth+dataset.zip`. Archive and sensor logs stay in `data/raw/`; generated evidence goes in `results/nested/`. Choose a different `--out-dir` to keep runs side by side.

The synthetic workflow exercises the same loader, feature, split, training and artifact code without fetching real data:

```sh
mhealth-benchmark --synthetic --inner-folds 2 --timing-repetitions 1
```

Without installing the project itself, `python scripts/download_data.py` and `python -m src.benchmark` use the same entry points once dependencies are installed. `make benchmark` and `make smoke` wrap the installed commands where Make is available. `--subjects` can shrink exploratory runs. `--max-per-class` caps only training rows, leaving validation and held-out distributions intact. See `mhealth-benchmark --help` for window, stride, seed, thread and timing controls.

## Method

1. Load and validate each sensor log separately. Form 128-sample windows with stride 64 by default. Retain windows with a single activity label and remove null windows.
2. Compute eight statistics per channel: mean, standard deviation, minimum, maximum, median, range, RMS, and mean absolute deviation. The basic feature set uses the same windows and their mean columns alone.
3. Hold out every person once. Tune features and model parameters on grouped inner folds among the other people. Each scaler is fit only on the corresponding training windows.
4. Refit the selected candidates on the outer training fold and score the held-out person. Export per-model results and a separate selection policy that picks the model family using only inner scores.
5. Weight subjects equally. Bootstrap held-out subjects for the descriptive 95% intervals. Report median and IQR across repeated feature-build, model-fit and model-prediction timings.

| Classifier | From-scratch implementation |
|---|---|
| KNN | Batched vectorized squared Euclidean distances and distance-weighted votes |
| Logistic regression | Multinomial softmax, cross-entropy, and L2-regularized gradient descent |
| Gaussian naive Bayes | Class-conditional Gaussians and stable log-space probabilities |
| Neural network | ReLU layers, He initialization, and minibatch backpropagation |

## Development

```sh
python -m pip install -c constraints.txt '.[dev]'
python -m pytest -q
ruff check .
python -m build --wheel
```

The legacy fixed-split CLI remains under `scripts/run_benchmark.py`; install `.[plots]` for its plots and use an explicit `--out-dir` to avoid replacing the historical examples. Random-window splitting is useful for illustrating overlap leakage and should be treated as a teaching diagnostic. The CI workflow builds the wheel, runs the suite and synthetic benchmark, and prepares a CycloneDX runtime SBOM for tag or manual builds.

## Claims and limits

The results describe ten volunteers performing the activities in this dataset. The intervals are descriptive across the observed subjects and reuse overlapping training sets. They do not validate other sensor placement, new populations, free-living activity detection, or clinical use. The model includes the two ECG channels as generic input signals, and the repository excludes raw traces and subject-level sensor exports.

## Dataset and license

Banos, O., Garcia, R., and Saez, A. (2014). *MHEALTH* [Dataset]. UCI Machine Learning Repository. [Dataset and citation](https://doi.org/10.24432/C5TW22). The dataset is CC BY 4.0. Project code is [MIT](LICENSE). See [data setup](data/README.md), [third-party notices](THIRD_PARTY_NOTICES.md), and [security](SECURITY.md).
