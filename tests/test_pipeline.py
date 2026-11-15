import json
from pathlib import Path
from zipfile import ZipInfo

import numpy as np
import pytest

from src.benchmark import main
from src.data_loader import SENSOR_COLUMNS, build_windows, load_subject
from src.dataset import PINNED_SOURCE, SOURCE_MANIFEST, read_verified_source, validate_archive, validate_members, sha256_file
from src.nested import aligned_probabilities, grouped_folds
from src.models import KNNClassifier


def _log(path: Path, labels, offset):
    values = np.arange(len(labels))[:, None] + np.arange(len(SENSOR_COLUMNS))[None, :] * 100 + offset
    np.savetxt(path, np.column_stack((values, labels)), fmt="%d")


def test_windows_drop_transitions_and_nulls_without_crossing_subjects(tmp_path):
    _log(tmp_path / "mHealth_subject10.log", [2, 2, 2, 1, 1, 0, 0, 0], 1000)
    _log(tmp_path / "mHealth_subject2.log", [1, 1, 1, 2, 2, 0, 0, 0], 0)
    features, y, groups, names = build_windows(str(tmp_path), window_size=3, stride=1, verbose=False)
    assert y.tolist() == [1, 2]
    assert groups.tolist() == [2, 10]
    assert names[:8] == [f"chest_acc_x_{stat}" for stat in ("mean", "std", "min", "max", "median", "range", "rms", "mad")]
    assert features[0, names.index("chest_acc_x_min")] == 0
    assert features[1, names.index("ecg_lead_1_mean")] == 1301
    _, labels_with_null, _, _ = build_windows(str(tmp_path), window_size=3, stride=1, drop_null=False, verbose=False)
    assert labels_with_null.tolist() == [1, 0, 2, 0]


def test_loader_rejects_missing_subject_and_invalid_data(tmp_path):
    log = tmp_path / "mHealth_subject1.log"
    _log(log, [1, 1, 1], 0)
    with pytest.raises(ValueError, match="requested subjects"):
        build_windows(str(tmp_path), subjects=[2])
    with pytest.raises(ValueError, match="positive"):
        build_windows(str(tmp_path), stride=0)
    _log(log, [1, 99, 1], 0)
    with pytest.raises(ValueError, match="activity labels"):
        load_subject(log)
    log.write_text("1 2\n3 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="24 columns"):
        load_subject(log)


def test_inner_train_and_validation_never_contain_outer_subject():
    groups = np.repeat([1, 2, 3, 4, 5, 6], 4)
    outer, folds = grouped_folds(groups, 3, 3, 42)
    assert set(groups[outer]) == {1, 2, 4, 5, 6}
    seen = set()
    for train, validation in folds:
        training_groups, validation_groups = set(groups[train]), set(groups[validation])
        assert training_groups.isdisjoint(validation_groups)
        assert 3 not in training_groups | validation_groups
        assert seen.isdisjoint(validation_groups)
        seen.update(validation_groups)
    assert set(groups[outer]) == seen


def test_missing_training_classes_keep_other_probability_columns_aligned():
    X = np.array([[0.0], [1.0]])
    model = KNNClassifier(k=1).fit(X, np.array([0, 2]))
    probabilities = aligned_probabilities(model, X, 3)
    np.testing.assert_allclose(probabilities, [[1, 0, 0], [0, 0, 1]])


@pytest.mark.parametrize("name", ["/MHEALTHDATASET/README.txt", "MHEALTHDATASET/../README.txt", "MHEALTHDATASET/unknown.txt"])
def test_archive_validator_rejects_noncanonical_members(name):
    with pytest.raises(ValueError, match="archive member"):
        validate_members([ZipInfo(name)])


def test_archive_validator_rejects_unverified_bytes(tmp_path):
    local = tmp_path / "dataset.zip"
    local.write_bytes(b"unrecognized archive")
    with pytest.raises(ValueError, match="size or SHA-256"):
        validate_archive(local)


def test_cache_cannot_override_pinned_file_fingerprints(tmp_path):
    records = dict(PINNED_SOURCE["files"])
    records["README.txt"] = {**records["README.txt"], "sha256": "0" * 64}
    marker = tmp_path / SOURCE_MANIFEST
    marker.write_text(json.dumps({**PINNED_SOURCE, "files": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="Extraction record differs"):
        read_verified_source(tmp_path)


def test_synthetic_benchmark_writes_verified_split_and_output_evidence(tmp_path):
    out = tmp_path / "evidence"
    assert main(["--synthetic", "--out-dir", str(out), "--window-size", "8", "--stride", "4",
                 "--inner-folds", "2", "--timing-repetitions", "1"]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["kind"] == "synthetic"
    assert len(manifest["cycles"]) == 4
    assert len(manifest["outputs"]) == 4
    assert len(manifest["timings_seconds"]["feature_build"]["observations"]) == 1
    for name, digest in manifest["outputs"].items():
        assert sha256_file(out / name) == digest
    for fold in manifest["cycles"]:
        subject = fold["held_out_subject"]
        assert fold["outer_test"]["subjects"] == [subject]
        assert subject not in fold["outer_train"]["subjects"]
        for inner in fold["inner"]:
            assert subject not in inner["train"]["subjects"] + inner["validation"]["subjects"]
