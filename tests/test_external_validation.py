from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.io import savemat
import h5py

from BrainTumorAI.external_validation import (
    _decode_cheng_mat,
    calculate_external_metrics,
    evaluate_external_manifest,
    prepare_brats_dataset,
    prepare_cheng_dataset,
)
from BrainTumorAI.classifier_pipeline import make_model, save_checkpoint


def test_cheng_preparation_preserves_patient_and_maps_all_subtypes_to_tumor(tmp_path) -> None:
    source_dir = tmp_path / "cheng_source"
    source_dir.mkdir()
    savemat(
        source_dir / "slice_001.mat",
        {
            "cjdata": {
                "label": np.asarray([[2]]),
                "PID": np.asarray([[17]]),
                "image": np.arange(64, dtype=np.uint8).reshape(8, 8),
            }
        },
    )

    output_dir = tmp_path / "prepared"
    metadata = prepare_cheng_dataset(source_dir, output_dir)

    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert metadata["sample_count"] == 1
    assert metadata["patient_count"] == 1
    assert str(manifest.loc[0, "patient_id"]) == "17"
    assert manifest.loc[0, "label_idx"] == 1
    assert manifest.loc[0, "subtype"] == "glioma"
    with Image.open(output_dir / manifest.loc[0, "source_file"]) as image:
        assert image.mode == "L"
        assert image.size == (8, 8)


def test_cheng_preparation_rejects_missing_patient_identifier(tmp_path) -> None:
    source_dir = tmp_path / "cheng_source"
    source_dir.mkdir()
    savemat(
        source_dir / "slice_001.mat",
        {
            "cjdata": {
                "label": np.asarray([[1]]),
                "PID": np.asarray([[]]),
                "image": np.arange(16, dtype=np.uint8).reshape(4, 4),
            }
        },
    )

    with pytest.raises(ValueError, match="PID"):
        prepare_cheng_dataset(source_dir, tmp_path / "prepared")


def test_cheng_v73_mat_decodes_hdf5_patient_id_and_image_orientation() -> None:
    mat_bytes = io.BytesIO()
    expected_image = np.arange(16, dtype=np.int16).reshape(4, 4)
    with h5py.File(mat_bytes, "w") as handle:
        record = handle.create_group("cjdata")
        record.create_dataset("label", data=np.asarray([[3.0]]))
        record.create_dataset("PID", data=np.asarray([[ord(char)] for char in "233"], dtype=np.uint16))
        record.create_dataset("image", data=expected_image.T)

    image, patient_id, tumor_label = _decode_cheng_mat(mat_bytes.getvalue(), "fixture.mat")

    assert patient_id == "233"
    assert tumor_label == 3
    assert np.array_equal(image, expected_image)


def test_patient_cluster_bootstrap_supports_tumor_only_external_dataset() -> None:
    metrics, intervals = calculate_external_metrics(
        true_labels=[1, 1, 1, 1],
        tumor_scores=[0.9, 0.2, 0.8, 0.7],
        patient_ids=["patient-a", "patient-a", "patient-b", "patient-b"],
        bootstrap_replicates=50,
        seed=13,
    )

    assert metrics["patient_count"] == 2
    assert metrics["tumor_sensitivity_recall"] == pytest.approx(0.75)
    assert "specificity" in metrics["metrics_not_estimable"]
    assert "no_tumor_specificity" not in metrics
    assert intervals["tumor_sensitivity_recall"]["lower_95"] <= 0.75


def test_patient_cluster_bootstrap_reports_both_slice_classes() -> None:
    metrics, intervals = calculate_external_metrics(
        true_labels=[0, 1, 0, 1],
        tumor_scores=[0.1, 0.9, 0.8, 0.2],
        patient_ids=["patient-a", "patient-a", "patient-b", "patient-b"],
        bootstrap_replicates=50,
        seed=13,
    )

    assert metrics["external_validation_scope"].startswith("Slice-level binary")
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["patient_counts_by_slice_class"] == {"no_tumor": 2, "tumor": 2}
    assert set(intervals) == {
        "accuracy",
        "balanced_accuracy",
        "tumor_sensitivity_recall",
        "no_tumor_specificity",
        "tumor_precision",
        "tumor_f1",
        "roc_auc",
        "average_precision",
    }


def test_external_evaluation_writes_predictions_without_selecting_threshold(tmp_path) -> None:
    image_root = tmp_path / "dataset"
    image_root.mkdir()
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_root / "negative.png")
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(image_root / "positive.png")
    manifest = pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "source_file": "negative.png",
                "patient_id": "patient-a",
                "label_idx": 0,
                "subtype": "no_tumor",
            },
            {
                "dataset": "fixture",
                "source_file": "positive.png",
                "patient_id": "patient-b",
                "label_idx": 1,
                "subtype": "glioma",
            },
        ]
    )
    manifest_path = image_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    overlap_audit = image_root / "overlap_audit.json"
    overlap_audit.write_text(
        '{"matches": {"high_similarity_matches": [{"patient_id": "patient-a"}]}}',
        encoding="utf-8",
    )
    checkpoint = tmp_path / "small_cnn.pth"
    save_checkpoint(make_model("small_cnn"), checkpoint, "small_cnn", {"seed": 42})

    report = evaluate_external_manifest(
        manifest_path,
        checkpoint,
        tmp_path / "results",
        bootstrap_replicates=5,
        overlap_audit_path=overlap_audit,
    )

    output_dir = tmp_path / "results" / "fixture"
    predictions = pd.read_csv(output_dir / "predictions.csv")
    assert report["this_evaluation_fits_no_parameters_or_thresholds"] is True
    assert report["checkpoint_training_independence_from_external_cohort"].startswith("Not verified")
    assert report["metrics"]["sample_count"] == 1
    assert report["metrics"]["tumor_subtype_sensitivity_patient_cluster_bootstrap_95"]["glioma"][
        "patient_count"
    ] == 1
    assert report["patient_overlap_exclusion"]["excluded_patient_count"] == 1
    assert report["patient_overlap_exclusion"]["excluded_slice_count"] == 1
    assert len(predictions) == 1
    assert (output_dir / "metrics.json").is_file()


def test_brats_preparation_links_flair_slices_to_patient_and_mask(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "brats_source"
    data_root.mkdir()
    flair_path = data_root / "BraTS-001_flair.nii.gz"
    seg_path = data_root / "BraTS-001_seg.nii.gz"
    flair_path.touch()
    seg_path.touch()
    volume = np.zeros((8, 8, 4), dtype=np.float32)
    volume[1:7, 1:7, :] = np.arange(1, 5, dtype=np.float32)
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    segmentation[3:5, 3:5, 1:3] = 2

    class FakeNifti:
        def __init__(self, data: np.ndarray) -> None:
            self.dataobj = data
            self.affine = np.eye(4)

    def fake_load(path: str) -> FakeNifti:
        return FakeNifti(segmentation if path.endswith("_seg.nii.gz") else volume)

    fake_nibabel = SimpleNamespace(load=fake_load, as_closest_canonical=lambda image: image)
    monkeypatch.setitem(sys.modules, "nibabel", fake_nibabel)

    output_dir = tmp_path / "prepared"
    metadata = prepare_brats_dataset(data_root, output_dir, min_brain_pixels=1)
    manifest = pd.read_csv(output_dir / "manifest.csv")

    assert metadata["patient_count"] == 1
    assert manifest["patient_id"].unique().tolist() == ["BraTS-001"]
    assert manifest["label_idx"].tolist() == [0, 1, 1, 0]
    assert manifest["slice_id"].tolist() == [0, 1, 2, 3]


def test_brats_preparation_accepts_t2f_alias_for_flair(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "brats_source"
    data_root.mkdir()
    flair_path = data_root / "BraTS-GLI-00001-000-t2f.nii.gz"
    seg_path = data_root / "BraTS-GLI-00001-000-seg.nii.gz"
    flair_path.touch()
    seg_path.touch()
    volume = np.zeros((8, 8, 3), dtype=np.float32)
    volume[2:6, 2:6, :] = np.arange(1, 4, dtype=np.float32)[None, None, :]
    segmentation = np.zeros_like(volume, dtype=np.uint8)
    segmentation[3:5, 3:5, 1] = 1

    class FakeNifti:
        def __init__(self, data: np.ndarray) -> None:
            self.dataobj = data
            self.affine = np.eye(4)

    def fake_load(path: str) -> FakeNifti:
        return FakeNifti(segmentation if path.endswith("-seg.nii.gz") else volume)

    fake_nibabel = SimpleNamespace(load=fake_load, as_closest_canonical=lambda image: image)
    monkeypatch.setitem(sys.modules, "nibabel", fake_nibabel)

    output_dir = tmp_path / "prepared"
    metadata = prepare_brats_dataset(data_root, output_dir, min_brain_pixels=1)
    manifest = pd.read_csv(output_dir / "manifest.csv")

    assert metadata["patient_count"] == 1
    assert manifest["patient_id"].unique().tolist() == ["BraTS-GLI-00001-000"]
    assert set(manifest["label_idx"].unique()) == {0, 1}
