from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .classifier_pipeline import (
    CLASS_TO_IDX,
    DEFAULT_CHECKPOINT,
    DEFAULT_RESULTS_DIR,
    IDX_TO_CLASS,
    PROJECT_ROOT,
    load_checkpoint,
    load_legacy_resnet18,
    sha256_file,
)
from .train_classifier import ManifestDataset, _validate_manifest


def compute_metrics(
    true_labels: list[int] | np.ndarray,
    tumor_scores: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(true_labels, dtype=np.int64)
    scores = np.asarray(tumor_scores, dtype=np.float64)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores) or len(y_true) == 0:
        raise ValueError("true_labels and tumor_scores must be non-empty, one-dimensional arrays of equal length.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in the [0, 1] interval.")
    if not set(np.unique(y_true)).issubset(set(CLASS_TO_IDX.values())):
        raise ValueError("Labels must use the explicit class indices 0=no_tumor and 1=tumor.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Tumor scores must be finite values in the [0, 1] interval.")
    if set(np.unique(y_true)) != set(CLASS_TO_IDX.values()):
        raise ValueError("Both classes must be present to calculate sensitivity, specificity, and ROC-AUC.")
    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    return {
        "threshold": threshold,
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "tumor_sensitivity_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "no_tumor_specificity": float(true_negative / (true_negative + false_positive)),
        "tumor_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "tumor_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["no_tumor", "tumor"],
        "predicted_no_tumor_count": int(np.count_nonzero(predictions == 0)),
        "predicted_tumor_count": int(np.count_nonzero(predictions == 1)),
        "degenerate_predictions": bool(len(np.unique(predictions)) == 1),
        "degeneracy_warning": (
            "The model predicted only one class at the fixed threshold."
            if len(np.unique(predictions)) == 1
            else None
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["no_tumor", "tumor"],
            output_dict=True,
            zero_division=0,
        ),
    }


def stratified_bootstrap_intervals(
    true_labels: np.ndarray,
    tumor_scores: np.ndarray,
    threshold: float,
    seed: int = 42,
    replicates: int = 1000,
) -> dict[str, dict[str, float]]:
    if replicates < 1:
        raise ValueError("replicates must be a positive integer.")
    rng = np.random.default_rng(seed)
    indices_by_class = {
        label: np.flatnonzero(true_labels == label)
        for label in CLASS_TO_IDX.values()
    }
    if any(len(indices) == 0 for indices in indices_by_class.values()):
        raise ValueError("Both classes are required for stratified bootstrap confidence intervals.")
    metrics: dict[str, list[float]] = {
        key: []
        for key in (
            "accuracy",
            "balanced_accuracy",
            "tumor_sensitivity_recall",
            "no_tumor_specificity",
            "tumor_precision",
            "tumor_f1",
            "roc_auc",
            "average_precision",
        )
    }
    for _ in range(replicates):
        sampled = np.concatenate(
            [
                rng.choice(class_indices, size=len(class_indices), replace=True)
                for class_indices in indices_by_class.values()
            ]
        )
        result = compute_metrics(true_labels[sampled], tumor_scores[sampled], threshold)
        for key in metrics:
            metrics[key].append(float(result[key]))
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in metrics.items()
    }


def stratified_group_bootstrap_intervals(
    true_labels: np.ndarray,
    tumor_scores: np.ndarray,
    group_ids: list[str] | np.ndarray,
    threshold: float,
    seed: int = 42,
    replicates: int = 1000,
) -> dict[str, dict[str, float]]:
    if replicates < 1:
        raise ValueError("replicates must be a positive integer.")
    groups = np.asarray(group_ids, dtype=str)
    if len(groups) != len(true_labels):
        raise ValueError("group_ids must have the same length as true_labels.")
    for group in np.unique(groups):
        if len(np.unique(true_labels[groups == group])) != 1:
            raise ValueError("A duplicate group contains conflicting class labels.")
    rng = np.random.default_rng(seed)
    indices_by_class: dict[int, list[np.ndarray]] = {}
    for label in CLASS_TO_IDX.values():
        class_indices = np.flatnonzero(true_labels == label)
        if len(class_indices) == 0:
            raise ValueError("Both classes are required for stratified group bootstrap confidence intervals.")
        class_groups = np.unique(groups[class_indices])
        indices_by_class[label] = []
        for group in class_groups:
            group_indices = class_indices[groups[class_indices] == group]
            if not np.all(true_labels[group_indices] == label):
                raise ValueError("A duplicate group contains conflicting class labels.")
            indices_by_class[label].append(group_indices)
    metrics: dict[str, list[float]] = {
        key: []
        for key in (
            "accuracy",
            "balanced_accuracy",
            "tumor_sensitivity_recall",
            "no_tumor_specificity",
            "tumor_precision",
            "tumor_f1",
            "roc_auc",
            "average_precision",
        )
    }
    for _ in range(replicates):
        sampled_groups: list[np.ndarray] = []
        for class_groups in indices_by_class.values():
            sampled_indices = rng.integers(0, len(class_groups), size=len(class_groups))
            sampled_groups.extend(class_groups[index] for index in sampled_indices)
        sampled_indices = np.concatenate(sampled_groups)
        result = compute_metrics(true_labels[sampled_indices], tumor_scores[sampled_indices], threshold)
        for key in metrics:
            metrics[key].append(float(result[key]))
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for key, values in metrics.items()
    }


def _predict(
    model: torch.nn.Module,
    rows: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ManifestDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.to(device)
    model.eval()
    all_labels: list[int] = []
    all_scores: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            tumor_scores = torch.softmax(logits, dim=1)[:, CLASS_TO_IDX["tumor"]]
            all_labels.extend(labels.tolist())
            all_scores.extend(tumor_scores.cpu().tolist())
    return np.asarray(all_labels, dtype=np.int64), np.asarray(all_scores, dtype=np.float64)


def _code_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _save_plots(
    output_dir: Path,
    label: str,
    true_labels: np.ndarray,
    scores: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["no_tumor", "tumor"],
        yticklabels=["no_tumor", "tumor"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{label}: confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, scores)
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(false_positive_rate, true_positive_rate, label=f"AUC={metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={metrics['average_precision']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{label}_roc_pr_curves.png", dpi=160)
    plt.close(fig)


def evaluate(
    manifest_path: Path,
    checkpoint_path: Path,
    results_dir: Path,
    compare_checkpoints: list[Path] | None = None,
    legacy_checkpoint: Path | None = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    rows = _validate_manifest(manifest_path)
    train_rows = rows[rows["split"] == "train"]
    test_rows = rows[rows["split"] == "test"].copy()
    if test_rows.empty or set(test_rows["label"].unique()) != set(CLASS_TO_IDX):
        raise ValueError("The final test split must contain both classes.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = results_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, torch.nn.Module, dict[str, Any], Path | None]] = []

    model, metadata = load_checkpoint(checkpoint_path)
    candidates.append((str(metadata.get("architecture", "trained_model")), model, metadata, checkpoint_path))
    for path in compare_checkpoints or []:
        comparison_model, comparison_metadata = load_checkpoint(path)
        label = f"{comparison_metadata.get('architecture', 'model')}_{path.stem}"
        candidates.append((label, comparison_model, comparison_metadata, path))
    if legacy_checkpoint is not None:
        candidates.append(
            (
                "legacy_existing_checkpoint",
                load_legacy_resnet18(legacy_checkpoint),
                {
                    "architecture": "resnet18",
                    "provenance": "Local pre-pipeline state_dict; training and data provenance unverified.",
                },
                legacy_checkpoint,
            )
        )

    train_class_counts = np.bincount(
        train_rows["label_idx"].to_numpy(dtype=np.int64),
        minlength=len(CLASS_TO_IDX),
    )
    train_majority_class = int(np.argmax(train_class_counts))
    y_true = test_rows["label_idx"].to_numpy(dtype=np.int64)
    model_results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for label, candidate, metadata, model_path in candidates:
        predicted_labels, scores = _predict(candidate, test_rows, device, batch_size)
        if not np.array_equal(predicted_labels, y_true):
            raise RuntimeError(f"Evaluation loader order/labels changed unexpectedly for {label}.")
        metrics = compute_metrics(y_true, scores, threshold)
        metrics["confidence_intervals_image_stratified_bootstrap_95"] = stratified_bootstrap_intervals(
            y_true, scores, threshold, replicates=bootstrap_replicates
        )
        metrics["confidence_intervals_duplicate_group_stratified_bootstrap_95"] = (
            stratified_group_bootstrap_intervals(
                y_true,
                scores,
                test_rows["duplicate_group"].astype(str).to_numpy(),
                threshold,
                replicates=bootstrap_replicates,
            )
        )
        metrics["sample_counts_by_true_class"] = {
            IDX_TO_CLASS[index]: int(np.count_nonzero(y_true == index))
            for index in IDX_TO_CLASS
        }
        metrics["checkpoint_sha256"] = sha256_file(model_path) if model_path else None
        metrics["checkpoint_metadata"] = metadata
        metrics["threshold_selection"] = "Fixed at 0.5; no final-test tuning."
        metrics["confidence_interval_unit"] = (
            "Primary: duplicate/near-duplicate-group-stratified bootstrap. Secondary: image-stratified "
            "bootstrap. Neither is patient-clustered because patient/volume IDs are unavailable."
        )
        model_results[label] = metrics
        _save_plots(output_dir, label, y_true, scores, metrics)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": label,
                    "source_file": test_rows["source_file"].to_numpy(),
                    "duplicate_group": test_rows["duplicate_group"].astype(str).to_numpy(),
                    "true_label": [IDX_TO_CLASS[index] for index in y_true],
                    "true_label_idx": y_true,
                    "no_tumor_score": 1.0 - scores,
                    "tumor_score": scores,
                    "predicted_label": [
                        IDX_TO_CLASS[index] for index in (scores >= threshold).astype(np.int64)
                    ],
                    "threshold": threshold,
                }
            )
        )

    majority_scores = np.full(len(y_true), float(train_majority_class), dtype=np.float64)
    majority_metrics = compute_metrics(y_true, majority_scores, threshold)
    majority_metrics["confidence_intervals_image_stratified_bootstrap_95"] = stratified_bootstrap_intervals(
        y_true, majority_scores, threshold, replicates=bootstrap_replicates
    )
    majority_metrics["confidence_intervals_duplicate_group_stratified_bootstrap_95"] = (
        stratified_group_bootstrap_intervals(
            y_true,
            majority_scores,
            test_rows["duplicate_group"].astype(str).to_numpy(),
            threshold,
            replicates=bootstrap_replicates,
        )
    )
    majority_metrics["baseline"] = (
        f"Constant training-majority class: {IDX_TO_CLASS[train_majority_class]}"
    )
    majority_metrics["threshold_selection"] = "No threshold tuning; constant training-majority prediction."
    model_results["training_majority_class"] = majority_metrics
    prediction_frames.append(
        pd.DataFrame(
            {
                "model": "training_majority_class",
                "source_file": test_rows["source_file"].to_numpy(),
                "duplicate_group": test_rows["duplicate_group"].astype(str).to_numpy(),
                "true_label": [IDX_TO_CLASS[index] for index in y_true],
                "true_label_idx": y_true,
                "no_tumor_score": 1.0 - majority_scores,
                "tumor_score": majority_scores,
                "predicted_label": [IDX_TO_CLASS[train_majority_class]] * len(y_true),
                "threshold": threshold,
            }
        )
    )

    predictions_path = output_dir / "test_predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    report = {
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "code_revision": _code_revision(),
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "scikit_learn": sklearn.__version__,
            "device": str(device),
        },
        "test_sample_count": len(test_rows),
        "test_set_used_for_selection": False,
        "predictions_file": str(predictions_path),
        "models": model_results,
    }
    report_path = output_dir / "evaluation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(result["degenerate_predictions"] for result in model_results.values()):
        print("[WARNING] At least one classifier/baseline predicted only one class at the fixed threshold.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen checkpoints on the untouched final test split."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RESULTS_DIR / "split_manifest.csv")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--compare-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    evaluate(
        args.manifest,
        args.checkpoint,
        args.results_dir,
        args.compare_checkpoint,
        args.legacy_checkpoint,
        args.threshold,
        args.batch_size,
        args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
