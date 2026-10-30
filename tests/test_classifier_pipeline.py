from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from BrainTumorAI.classifier_pipeline import (
    CLASS_TO_IDX,
    CheckpointError,
    IDX_TO_CLASS,
    SmallCNN,
    load_checkpoint,
    make_inference_transform,
    make_model,
    predict_image,
    preprocess_image,
    save_checkpoint,
    seed_everything,
)
from BrainTumorAI.evaluate_classifier import (
    compute_metrics,
    stratified_bootstrap_intervals,
    stratified_group_bootstrap_intervals,
)
from BrainTumorAI.heatmap_gen import GradCAM, apply_heatmap
import BrainTumorAI.medication_report as medication_report
from BrainTumorAI.preprocess import _assign_splits, _group_near_duplicates
from BrainTumorAI.stereotactic_robot import (
    SIMULATED_COORDINATE_LABEL,
    illustrative_heatmap_mapping,
)
from BrainTumorAI.surgery_viz import create_3d_brain_model
from BrainTumorAI.surgical_task_queue import default_queue
from BrainTumorAI.train_classifier import _validate_manifest


def test_explicit_class_mapping_and_model_output_dimensions() -> None:
    assert CLASS_TO_IDX == {"no_tumor": 0, "tumor": 1}
    assert IDX_TO_CLASS == {0: "no_tumor", 1: "tumor"}
    for architecture in ("resnet18", "small_cnn"):
        model = make_model(architecture).eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 64, 64))
        assert logits.shape == (1, 2)


def test_inference_preprocessing_matches_evaluation_transform() -> None:
    image = Image.fromarray(np.full((31, 47, 4), 128, dtype=np.uint8), mode="RGBA")
    from_shared_preprocessor = preprocess_image(image)
    from_evaluation_transform = make_inference_transform()(image.convert("RGB"))
    assert isinstance(from_evaluation_transform, torch.Tensor)
    assert torch.equal(from_shared_preprocessor, from_evaluation_transform)
    assert from_shared_preprocessor.shape == (3, 224, 224)


def test_app_prediction_matches_shared_evaluation_preprocessing() -> None:
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (52, 81, 3), dtype=np.uint8))
    model = make_model("small_cnn").eval()
    app_logits = predict_image(model, image)
    with torch.inference_mode():
        evaluation_logits = model(preprocess_image(image).unsqueeze(0))[0]
    assert torch.equal(app_logits, evaluation_logits)


def test_prediction_generation_covers_both_binary_classes() -> None:
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    model = make_model("small_cnn").eval()
    assert isinstance(model, SmallCNN)
    output_layer = next(layer for layer in model.classifier.modules() if isinstance(layer, nn.Linear))
    predictions = []
    for logits in ([3.0, 0.0], [0.0, 3.0]):
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.copy_(torch.tensor(logits))
        scores = torch.softmax(predict_image(model, image), dim=0)
        predictions.append(int(scores.argmax().item()))
    assert predictions == [CLASS_TO_IDX["no_tumor"], CLASS_TO_IDX["tumor"]]


def test_checkpoint_round_trip_strict_and_missing_checkpoint(tmp_path) -> None:
    model = make_model("small_cnn")
    assert isinstance(model, SmallCNN)
    checkpoint = tmp_path / "valid.pth"
    save_checkpoint(model, checkpoint, "small_cnn", {"seed": 42, "torch": torch.__version__})
    reloaded, metadata = load_checkpoint(checkpoint)
    assert isinstance(reloaded, SmallCNN)
    assert metadata["seed"] == 42
    assert torch.equal(
        model.state_dict()["classifier.2.weight"],
        reloaded.state_dict()["classifier.2.weight"],
    )
    with pytest.raises(FileNotFoundError, match="Train one"):
        load_checkpoint(tmp_path / "missing.pth")


def test_checkpoint_rejects_incompatible_mapping_and_state(tmp_path) -> None:
    model = make_model("small_cnn")
    incompatible_mapping = tmp_path / "bad_mapping.pth"
    torch.save(
        {
            "format_version": 1,
            "architecture": "small_cnn",
            "class_to_idx": {"tumor": 0, "no_tumor": 1},
            "model_state_dict": model.state_dict(),
            "metadata": {},
        },
        incompatible_mapping,
    )
    with pytest.raises(CheckpointError, match="class mapping"):
        load_checkpoint(incompatible_mapping)

    incompatible_state = tmp_path / "bad_state.pth"
    state = model.state_dict()
    state.pop("classifier.2.bias")
    torch.save(
        {
            "format_version": 1,
            "architecture": "small_cnn",
            "class_to_idx": CLASS_TO_IDX,
            "model_state_dict": state,
            "metadata": {},
        },
        incompatible_state,
    )
    with pytest.raises(CheckpointError, match="incompatible"):
        load_checkpoint(incompatible_state)


def test_grouped_splits_keep_exact_duplicate_content_together() -> None:
    rows = []
    for label_idx in (0, 1):
        for index in range(40):
            group = f"{label_idx}-{index // 2}"
            rows.append({"label_idx": label_idx, "pixel_sha256": group})
    assignments = _assign_splits(rows, seed=42)
    split_groups = {
        split: {
            rows[index]["pixel_sha256"]
            for index, assigned_split in assignments.items()
            if assigned_split == split
        }
        for split in ("train", "validation", "test")
    }
    assert not split_groups["train"] & split_groups["validation"]
    assert not split_groups["train"] & split_groups["test"]
    assert not split_groups["validation"] & split_groups["test"]
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_manifest_rejects_near_duplicate_groups_crossing_splits(tmp_path) -> None:
    rows = []
    for split in ("train", "validation", "test"):
        for label, label_idx in (("no_tumor", 0), ("tumor", 1)):
            source_file = f"{split}-{label}.jpg"
            rows.append(
                {
                    "source_file": source_file,
                    "label": label,
                    "label_idx": label_idx,
                    "split": split,
                    "pixel_sha256": source_file,
                    "duplicate_group": (
                        "shared-near-duplicate-group"
                        if label_idx == 0 and split in {"train", "test"}
                        else source_file
                    ),
                    "status": "readable",
                }
            )
    manifest = tmp_path / "cross_split_near_duplicates.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="duplicate groups cross split boundaries"):
        _validate_manifest(manifest)


def test_near_duplicate_screen_groups_high_similarity_images() -> None:
    rows = [
        {
            "pixel_sha256": "pixel-a",
            "phash": "0000000000000000",
            "label": "tumor",
            "source_file": "a.jpg",
            "_gray_64": np.full((64, 64), 100, dtype=np.uint8),
        },
        {
            "pixel_sha256": "pixel-b",
            "phash": "0000000000000000",
            "label": "tumor",
            "source_file": "b.jpg",
            "_gray_64": np.full((64, 64), 102, dtype=np.uint8),
        },
        {
            "pixel_sha256": "pixel-c",
            "phash": "7fffffffffffffff",
            "label": "no_tumor",
            "source_file": "c.jpg",
            "_gray_64": np.full((64, 64), 220, dtype=np.uint8),
        },
    ]
    summary = _group_near_duplicates(rows)
    assert rows[0]["duplicate_group"] == rows[1]["duplicate_group"]
    assert rows[2]["duplicate_group"] != rows[0]["duplicate_group"]
    assert summary["high_similarity_edges_phash_le_4_ssim_ge_0_95"] == 1
    assert summary["high_similarity_cross_label_edges"] == 0


def test_evaluation_uses_continuous_tumor_scores_and_flags_degenerate_predictions() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["predicted_no_tumor_count"] == 2
    assert metrics["predicted_tumor_count"] == 2
    degenerate = compute_metrics([0, 0, 1, 1], [0.7, 0.8, 0.9, 1.0])
    assert degenerate["degenerate_predictions"] is True
    assert "only one class" in degenerate["degeneracy_warning"]
    with pytest.raises(ValueError, match="Both classes"):
        compute_metrics([1, 1], [0.2, 0.7])


def test_stratified_bootstrap_intervals_have_expected_metrics() -> None:
    intervals = stratified_bootstrap_intervals(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
        seed=9,
        replicates=30,
    )
    assert intervals["accuracy"]["lower_95"] == pytest.approx(1.0)
    assert intervals["roc_auc"]["upper_95"] == pytest.approx(1.0)


def test_duplicate_group_bootstrap_resamples_whole_groups() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])
    groups = np.asarray(["a", "a", "b", "c", "d", "d"])
    intervals = stratified_group_bootstrap_intervals(
        labels,
        scores,
        groups,
        threshold=0.5,
        seed=12,
        replicates=30,
    )
    assert intervals["accuracy"]["lower_95"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="conflicting class labels"):
        stratified_group_bootstrap_intervals(
            np.asarray([0, 1]),
            np.asarray([0.2, 0.8]),
            np.asarray(["same-group", "same-group"]),
            threshold=0.5,
            replicates=1,
        )


def test_gradcam_generates_both_target_classes_and_removes_hooks() -> None:
    class TwoChannelActivationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target = nn.Conv2d(3, 2, kernel_size=1, bias=False)
            self.classifier = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.target.weight.zero_()
                self.target.weight[0, 0, 0, 0] = 1.0
                self.target.weight[1, 1, 0, 0] = 1.0
                self.classifier.weight.copy_(torch.eye(2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.target(inputs).mean(dim=(2, 3)))

    model = TwoChannelActivationModel().eval()
    target_layer = model.target
    grad_cam = GradCAM(model, target_layer)
    pattern = torch.linspace(0.1, 1.0, 64).reshape(1, 1, 8, 8)
    input_tensor = torch.cat((pattern, pattern.flip(-1), torch.zeros_like(pattern)), dim=1)
    for target_class in (0, 1):
        heatmap = grad_cam.generate_heatmap(input_tensor, target_class=target_class)
        assert heatmap.ndim == 2
        assert np.isfinite(heatmap).all()
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1
        assert len(target_layer._forward_hooks) == 0


def test_gradcam_rejects_invalid_or_contrast_free_heatmaps(tmp_path) -> None:
    class ConstantActivationModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target = nn.Conv2d(3, 2, kernel_size=1)
            with torch.no_grad():
                self.target.weight.zero_()
                if self.target.bias is not None:
                    self.target.bias.fill_(1.0)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            activation_sum = self.target(inputs).sum(dim=(1, 2, 3))
            return torch.stack((activation_sum, -activation_sum), dim=1)

    constant_model = ConstantActivationModel().eval()
    grad_cam = GradCAM(constant_model, constant_model.target)
    with pytest.raises(ValueError, match="one image"):
        grad_cam.generate_heatmap(torch.zeros(2, 3, 64, 64))
    with pytest.raises(ValueError, match="finite"):
        grad_cam.generate_heatmap(torch.full((1, 3, 64, 64), float("nan")))
    with pytest.raises(RuntimeError, match="no class-specific activation contrast"):
        grad_cam.generate_heatmap(
            torch.zeros(1, 3, 8, 8),
            target_class=1,
        )

    flat_heatmap = np.ones((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="no activation contrast"):
        apply_heatmap(tmp_path / "not_read.jpg", flat_heatmap)


def test_illustrative_coordinate_mapping_is_deterministic_and_labeled() -> None:
    heatmap = np.zeros((3, 5), dtype=np.float32)
    heatmap[0, 4] = 1.0
    first = illustrative_heatmap_mapping(heatmap)
    second = illustrative_heatmap_mapping(heatmap.copy())
    assert first == second
    assert first == ((90, 90, 0), "upper image", "image-right")
    assert SIMULATED_COORDINATE_LABEL == "Illustrative / Simulated Coordinates"
    assert illustrative_heatmap_mapping(None) == (
        (0, 0, 0),
        "Unknown image region",
        "Unknown image side",
    )
    with pytest.raises(ValueError, match="finite"):
        illustrative_heatmap_mapping(np.asarray([[np.nan]], dtype=np.float32))


def test_3d_visualization_is_labeled_as_simulated() -> None:
    figure = create_3d_brain_model(
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.uint8),
    )
    figure_json = figure.to_json()
    assert figure_json is not None
    serialized = json.loads(figure_json)
    assert "Illustrative / Simulated" in serialized["layout"]["title"]["text"]
    assert any(
        "not a segmentation mask" in trace.get("name", "")
        for trace in serialized["data"]
    )


def test_demo_queue_does_not_claim_anatomical_locations() -> None:
    queue = default_queue()
    assert queue
    assert all(task["lobe"] == "Not anatomically assessed" for task in queue)
    assert all("illustrative_coordinates" in task for task in queue)


def test_report_outputs_are_nonclinical_and_pdf_generation_succeeds(tmp_path, monkeypatch) -> None:
    assert medication_report.suggest_treatment("Glioma") == medication_report.suggest_treatment("no_tumor")
    assert medication_report.suggest_treatment("Glioma")["treatments"] == [
        "No treatment advice is generated."
    ]
    monkeypatch.setattr(medication_report, "__file__", str(tmp_path / "medication_report.py"))
    report_path = medication_report.create_pdf_report(
        "Synthetic demo record",
        "SYNTHETIC-001",
        "tumor",
        90.0,
        "not assessed",
        "not measured",
        "Synthetic test fixture only",
    )
    assert report_path.endswith(".pdf")
    with open(report_path, "rb") as report_file:
        assert report_file.read(5) == b"%PDF-"

    text_report = medication_report.generate_report_text(
        "Synthetic demo record",
        "SYNTHETIC-001",
        "tumor",
    )
    assert "NOT FOR PATIENT-CARE DECISIONS" in text_report
    assert "SUGGESTED TREATMENT" not in text_report
    assert "Anatomical location: Not assessed" in text_report


def test_seed_reproduces_model_initialization() -> None:
    seed_everything(123)
    first = make_model("small_cnn").state_dict()
    seed_everything(123)
    second = make_model("small_cnn").state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)
