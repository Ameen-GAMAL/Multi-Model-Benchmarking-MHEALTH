"""Load the MHEALTH ``.log`` files and turn the raw 50 Hz sensor streams into a
supervised learning matrix.

The dataset is a *time series*: 23 sensor channels sampled at 50 Hz while the
subject performs one of 12 activities.  Classifying a single 20 ms sample in
isolation throws away the temporal structure, so the core of this module is a
sliding-window feature extractor that summarises each ~2.5 s window into a fixed
vector of statistics.  Choosing those statistics is exactly the feature
engineering that drives the benchmark's biggest accuracy gains.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- schema
# The 23 sensor channels, in the column order of the raw log files.
SENSOR_COLUMNS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_lead_1", "ecg_lead_2",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "arm_acc_x", "arm_acc_y", "arm_acc_z",
    "arm_gyro_x", "arm_gyro_y", "arm_gyro_z",
    "arm_mag_x", "arm_mag_y", "arm_mag_z",
]
COLUMNS = SENSOR_COLUMNS + ["label"]

# Activity labels 1..12 (0 is the "null" class recorded between activities).
ACTIVITY_NAMES = {
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back",
}

SAMPLING_RATE_HZ = 50
DEFAULT_DATA_DIR = os.path.join("data", "raw", "MHEALTHDATASET")


# ---------------------------------------------------------------- loading
def load_subject(path: str) -> pd.DataFrame:
    """Load one ``mHealth_subject*.log`` file into a labelled DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns in {path!s}; got {df.shape[1]}.")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Found a missing or non-finite sensor value or label in {path!s}.")
    df.columns = COLUMNS
    if not df["label"].isin([0, *ACTIVITY_NAMES]).all():
        raise ValueError(f"Expected activity labels in 0..12 in {path!s}.")
    df["label"] = df["label"].astype(int)
    return df


def load_raw(data_dir: str = DEFAULT_DATA_DIR, subjects=None) -> pd.DataFrame:
    """Concatenate the requested subjects into one DataFrame.

    A ``subject`` column is added so evaluation can, if desired, respect subject
    boundaries.  ``subjects`` is an iterable of integer ids (1..10); ``None``
    loads everyone found in ``data_dir``.
    """
    frames = []
    for sid, path in subject_files(data_dir, subjects):
        df = load_subject(path)
        df["subject"] = sid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def subject_files(data_dir: str = DEFAULT_DATA_DIR, subjects=None) -> list[tuple[int, str]]:
    """Return selected logs in subject order, rejecting ambiguous or missing IDs."""
    paths = glob.glob(os.path.join(data_dir, "mHealth_subject*.log"))
    found = {}
    for path in paths:
        match = re.fullmatch(r"mHealth_subject([0-9]+)\.log", os.path.basename(path))
        if match is None:
            continue
        sid = int(match[1])
        if sid in found:
            raise ValueError(f"Multiple input logs identify subject {sid}.")
        found[sid] = path
    if not found:
        raise FileNotFoundError(
            f"No 'mHealth_subject*.log' files in {data_dir!r}. "
            "Run `python scripts/download_data.py` first."
        )
    wanted = set(found) if subjects is None else set(subjects)
    if not wanted or wanted - found.keys():
        raise ValueError(f"No logs for requested subjects: {sorted(wanted - found.keys())}.")
    return [(sid, found[sid]) for sid in sorted(wanted)]


# --------------------------------------------------- feature engineering
# The per-channel summary statistics computed over each window.  Adding or
# removing entries here is the single most impactful knob in the pipeline.
_STAT_FUNCS = {
    "mean":   lambda w: w.mean(axis=1),
    "std":    lambda w: w.std(axis=1),
    "min":    lambda w: w.min(axis=1),
    "max":    lambda w: w.max(axis=1),
    "median": lambda w: np.median(w, axis=1),
    "range":  lambda w: w.max(axis=1) - w.min(axis=1),
    "rms":    lambda w: np.sqrt((w ** 2).mean(axis=1)),          # signal energy
    "mad":    lambda w: np.mean(np.abs(w - w.mean(axis=1, keepdims=True)), axis=1),
}

FEATURE_SETS = {
    # "basic": channel means only -- the naive representation.
    "basic": ["mean"],
    # "full": the eight-statistic set that captures shape and variability.
    "full": list(_STAT_FUNCS.keys()),
}


def feature_names(feature_set: str = "full") -> list[str]:
    stats = FEATURE_SETS[feature_set]
    return [f"{ch}_{s}" for ch in SENSOR_COLUMNS for s in stats]


def _window_features(windows: np.ndarray, stats: list[str]) -> np.ndarray:
    """Vectorised statistics for a stack of windows.

    ``windows`` has shape ``(n_windows, window_size, n_channels)``.  The stat
    functions reduce over ``axis=1`` -- the time axis -- giving
    ``(n_windows, n_channels)`` per statistic; the stats are then interleaved
    per channel so columns line up with ``feature_names``.
    """
    per_stat = [_STAT_FUNCS[s](windows) for s in stats]
    # per_stat[k] has shape (n_windows, n_channels); stack -> (n_windows, n_channels, n_stats)
    stacked = np.stack(per_stat, axis=2)
    n_windows = stacked.shape[0]
    return stacked.reshape(n_windows, -1)  # channel-major, matches feature_names


def build_windows(
    data_dir: str = DEFAULT_DATA_DIR,
    subjects=None,
    window_size: int = 128,
    stride: int = 64,
    feature_set: str = "full",
    drop_null: bool = True,
    verbose: bool = True,
):
    """Build a windowed, feature-engineered dataset.

    Windows are formed within each subject (never across subjects) and are kept
    only when every sample in the window shares the same activity label, so each
    window has an unambiguous target.  Returns ``(X, y, groups, names)`` where
    ``groups`` records the source subject of every window.
    """
    if window_size < 1 or stride < 1:
        raise ValueError("Window size and stride must be positive.")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set!r}.")
    stats = FEATURE_SETS[feature_set]
    feature_parts, label_list, group_list = [], [], []
    for sid, path in subject_files(data_dir, subjects):
        sub_df = load_subject(path)
        values = sub_df[SENSOR_COLUMNS].to_numpy(dtype=float)
        labels = sub_df["label"].to_numpy(dtype=int)
        n = len(values)
        win_list = []
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            wl = labels[start:end]
            # Keep only "pure" windows -- a single activity from start to end.
            if wl[0] == wl[-1] and np.all(wl == wl[0]):
                lab = wl[0]
                if drop_null and lab == 0:
                    continue
                win_list.append(values[start:end])
                label_list.append(lab)
                group_list.append(sid)
        if win_list:
            feature_parts.append(_window_features(np.asarray(win_list), stats))

    if not feature_parts:
        raise RuntimeError("No valid windows produced; check window_size/stride.")

    X = np.concatenate(feature_parts)
    y = np.asarray(label_list, dtype=int)
    groups = np.asarray(group_list, dtype=int)
    names = feature_names(feature_set)

    if verbose:
        print(
            f"Built {X.shape[0]} windows x {X.shape[1]} features "
            f"({feature_set}) from {len(np.unique(groups))} subjects; "
            f"{len(np.unique(y))} activity classes."
        )
    return X, y, groups, names


def relabel_contiguous(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary integer labels to 0..K-1 and return the mapping."""
    classes = np.unique(y)
    lookup = {c: i for i, c in enumerate(classes)}
    y_mapped = np.array([lookup[v] for v in y], dtype=int)
    return y_mapped, classes
