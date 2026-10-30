from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import cv2
from PIL import Image, __version__ as pillow_version
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import __version__ as torchvision_version

from .classifier_pipeline import (
    CLASS_TO_IDX,
    DEFAULT_CHECKPOINT,
    DEFAULT_RESULTS_DIR,
    IDX_TO_CLASS,
    PROJECT_ROOT,
    make_model,
    make_inference_transform,
    make_training_transform,
    save_checkpoint,
    seed_everything,
    sha256_file,
)


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: pd.DataFrame, training: bool = False) -> None:
        self.rows = rows.reset_index(drop=True)
        self.transform = make_training_transform() if training else make_inference_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / str(row["source_file"])
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load image listed in split manifest: {path}") from exc
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Image preprocessing did not return a tensor for {path}.")
        return tensor, int(row["label_idx"])


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _save_learning_curves(history: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_balanced_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["validation_balanced_accuracy"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Balanced accuracy", title="Balanced accuracy")
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _validate_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Create it with `python -m BrainTumorAI.preprocess`."
        )
    rows = pd.read_csv(path)
    required = {
        "source_file",
        "label",
        "label_idx",
        "split",
        "pixel_sha256",
        "duplicate_group",
        "status",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Split manifest is missing required columns: {sorted(missing)}")
    rows = rows[rows["status"] == "readable"].copy()
    if rows.empty:
        raise ValueError("The split manifest contains no readable images.")
    if set(rows["label"].unique()) != set(CLASS_TO_IDX):
        raise ValueError(f"Manifest labels must be {set(CLASS_TO_IDX)}, got {set(rows['label'].unique())}.")
    expected_indices = rows["label"].map(CLASS_TO_IDX)
    if not np.array_equal(expected_indices.to_numpy(), rows["label_idx"].to_numpy()):
        raise ValueError("Manifest label_idx values do not match the explicit class mapping.")
    for split in ("train", "validation", "test"):
        if split not in set(rows["split"]):
            raise ValueError(f"Manifest has no {split} images.")
    if rows["source_file"].duplicated().any():
        raise ValueError("The manifest contains duplicate source_file rows.")
    if rows["duplicate_group"].isna().any() or rows["duplicate_group"].astype(str).str.len().eq(0).any():
        raise ValueError("Readable manifest rows must have a non-empty duplicate_group.")
    group_split_counts = rows.groupby("duplicate_group")["split"].nunique()
    if group_split_counts.gt(1).any():
        raise ValueError("Exact or high-similarity duplicate groups cross split boundaries.")
    split_groups = rows.groupby("split")["pixel_sha256"].apply(set)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_groups[left] & split_groups[right]:
            raise ValueError(f"Exact duplicate pixel groups cross {left} and {right} splits.")
    return rows


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    targets: list[int] = []
    predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.item()) * labels.size(0)
        sample_count += labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if sample_count == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
    }


def train_model(
    manifest_path: Path,
    architecture: str,
    output_path: Path,
    results_dir: Path,
    seed: int = 42,
    epochs: int = 12,
    batch_size: int = 16,
    patience: int = 4,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    seed_everything(seed)
    rows = _validate_manifest(manifest_path)
    train_rows = rows[rows["split"] == "train"].copy()
    validation_rows = rows[rows["split"] == "validation"].copy()
    if set(train_rows["label"].unique()) != set(CLASS_TO_IDX):
        raise ValueError("Training split must contain both explicitly mapped classes.")
    if set(validation_rows["label"].unique()) != set(CLASS_TO_IDX):
        raise ValueError("Validation split must contain both explicitly mapped classes.")

    train_counts = Counter(train_rows["label"])
    validation_counts = Counter(validation_rows["label"])
    print(f"[DATA] Explicit mapping: {CLASS_TO_IDX}")
    print(f"[DATA] Train distribution: {dict(train_counts)}")
    print(f"[DATA] Validation distribution: {dict(validation_counts)}")
    print("[DATA] Class imbalance is modest; using unweighted cross-entropy without resampling.")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ManifestDataset(train_rows, training=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        ManifestDataset(validation_rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] Architecture: {architecture}; device: {device}; seed: {seed}")

    model = make_model(architecture, pretrained=(architecture == "resnet18")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history: list[dict[str, float | int]] = []
    best_balanced_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = _run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["balanced_accuracy"])
        record: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(
            f"[EPOCH {epoch:02d}/{epochs}] "
            f"train loss={train_metrics['loss']:.4f} bal_acc={train_metrics['balanced_accuracy']:.4f} | "
            f"validation loss={validation_metrics['loss']:.4f} "
            f"bal_acc={validation_metrics['balanced_accuracy']:.4f}"
        )
        _write_json(results_dir / f"{architecture}_training_history.json", {"epochs": history})
        pd.DataFrame(history).to_csv(results_dir / f"{architecture}_training_history.csv", index=False)

        score = validation_metrics["balanced_accuracy"]
        loss = validation_metrics["loss"]
        improved = score > best_balanced_accuracy + 1e-4 or (
            abs(score - best_balanced_accuracy) <= 1e-4 and loss < best_validation_loss
        )
        if improved:
            best_balanced_accuracy = score
            best_validation_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
            metadata = {
                "selected_epoch": best_epoch,
                "selection_metric": "validation_balanced_accuracy",
                "validation_balanced_accuracy": best_balanced_accuracy,
                "validation_loss": best_validation_loss,
                "seed": seed,
                "class_to_idx": CLASS_TO_IDX,
                "idx_to_class": IDX_TO_CLASS,
                "image_size": 224,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
                "augmentation": {
                    "training_only": ["random horizontal flip", "random rotation up to 10 degrees"],
                    "validation_test_inference": "resize, tensor conversion, ImageNet normalization",
                },
                "manifest_sha256": sha256_file(manifest_path),
                "code_revision": _code_revision(),
                "environment": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "torchvision": torchvision_version,
                    "scikit_learn": sklearn.__version__,
                    "numpy": np.__version__,
                    "pillow": pillow_version,
                    "opencv": cv2.__version__,
                    "device": str(device),
                    "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                },
                "training_config": {
                    "architecture": architecture,
                    "pretrained": architecture == "resnet18",
                    "optimizer": "Adam",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                    "batch_size": batch_size,
                    "max_epochs": epochs,
                    "early_stopping_patience": patience,
                    "loss": "CrossEntropyLoss",
                    "class_weighting": False,
                },
            }
            save_checkpoint(model, output_path, architecture, metadata)
            _write_json(results_dir / f"{architecture}_checkpoint_metadata.json", metadata)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[EARLY STOP] No validation improvement for {patience} epochs.")
            break

    duration_seconds = time.perf_counter() - started
    if best_epoch == 0 or not output_path.is_file():
        raise RuntimeError("Training ended without producing a validation-selected checkpoint.")
    summary = {
        "architecture": architecture,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "training_seconds": duration_seconds,
        "checkpoint": str(output_path),
        "checkpoint_sha256": sha256_file(output_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_class_counts": dict(train_counts),
        "validation_class_counts": dict(validation_counts),
        "test_set_accessed": False,
    }
    _save_learning_curves(history, results_dir / f"{architecture}_learning_curves.png")
    _write_json(results_dir / f"{architecture}_training_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-18 or small-CNN baseline using train/validation data only."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RESULTS_DIR / "split_manifest.csv")
    parser.add_argument("--architecture", choices=("resnet18", "small_cnn"), default="resnet18")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    output_path = args.checkpoint or (
        DEFAULT_CHECKPOINT
        if args.architecture == "resnet18"
        else Path(__file__).resolve().parent / "models" / "small_cnn_best.pth"
    )
    train_model(
        args.manifest,
        args.architecture,
        output_path,
        args.results_dir,
        args.seed,
        args.epochs,
        args.batch_size,
        args.patience,
    )


if __name__ == "__main__":
    main()
