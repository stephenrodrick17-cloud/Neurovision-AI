from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, __version__ as pillow_version
from torch.torch_version import TorchVersion
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from BrainTumorAI.classifier_pipeline import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    PROJECT_ROOT,
    make_model,
    make_inference_transform,
    make_training_transform,
    save_checkpoint,
    seed_everything,
    sha256_file,
)

ROOT = PROJECT_ROOT
MANIFEST_PATH = ROOT / "BrainTumorAI" / "results" / "split_manifest.csv"
PHASE1B_DIR = ROOT / "experiments_results" / "phase1b"
ARCHITECTURES = ("resnet18", "small_cnn", "densenet121", "efficientnet_b0")
SEEDS = (42, 43, 44, 45, 46)


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_learning_curves(history: list[dict[str, float | int]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_balanced_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["validation_balanced_accuracy"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Validation balanced accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: pd.DataFrame, training: bool = False) -> None:
        self.rows = rows.reset_index(drop=True)
        self.transform = make_training_transform() if training else make_inference_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        path = ROOT / str(row["source_file"])
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label_idx"])


class EpochRunner:
    def __init__(self, model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> None:
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.training = optimizer is not None

    def run(self) -> dict[str, float]:
        self.model.train(self.training)
        loss_sum = 0.0
        sample_count = 0
        targets: list[int] = []
        predictions: list[int] = []
        for images, labels in self.loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            if self.training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(self.training):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if self.training:
                    loss.backward()
                    self.optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0)
            sample_count += labels.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if sample_count == 0:
            raise ValueError("Cannot run an epoch with an empty loader.")
        return {
            "loss": loss_sum / sample_count,
            "accuracy": float(accuracy_score(targets, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        }


def _make_densenet121(pretrained: bool = True) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_TO_IDX))
    return model


def _make_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(CLASS_TO_IDX)),
    )
    return model


def build_model(architecture: str, pretrained: bool) -> nn.Module:
    if architecture == "resnet18":
        return make_model("resnet18", pretrained=pretrained)
    if architecture == "small_cnn":
        if pretrained:
            raise ValueError("small_cnn does not use pretrained weights.")
        return make_model("small_cnn", pretrained=False)
    if architecture == "densenet121":
        return _make_densenet121(pretrained=pretrained)
    if architecture == "efficientnet_b0":
        return _make_efficientnet_b0(pretrained=pretrained)
    raise ValueError(f"Unsupported architecture: {architecture}")


def read_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    rows = pd.read_csv(MANIFEST_PATH)
    required = {"source_file", "label", "label_idx", "split", "duplicate_group", "status"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    rows = rows[rows["status"] == "readable"].copy()
    return rows


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | int | list[int]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0]
    fn, tp = cm[1]
    metrics: dict[str, float | int | list[int]] = {
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
    return metrics


def evaluate_model(model: nn.Module, manifest_rows: pd.DataFrame, split: str = "test") -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = manifest_rows[manifest_rows["split"] == split].copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows found in split '{split}'.")
    device = next(model.parameters()).device
    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        path = ROOT / str(row["source_file"])
        with Image.open(path) as image:
            tensor = make_inference_transform()(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        tumor_score = float(probs[0, 1].item())
        no_tumor_score = float(probs[0, 0].item())
        predicted_label = 1 if tumor_score >= 0.5 else 0
        records.append(
            {
                "model": model.__class__.__name__,
                "source_file": row["source_file"],
                "duplicate_group": row.get("duplicate_group", ""),
                "true_label": row["label"],
                "true_label_idx": int(row["label_idx"]),
                "no_tumor_score": no_tumor_score,
                "tumor_score": tumor_score,
                "predicted_label": IDX_TO_CLASS[predicted_label],
                "threshold": 0.5,
            }
        )
    predictions = pd.DataFrame.from_records(records)
    true_labels = rows["label_idx"].astype(int).to_numpy()
    scores = predictions["tumor_score"].to_numpy(dtype=float)
    predicted = predictions["predicted_label"].map(CLASS_TO_IDX).to_numpy(dtype=int)
    metrics = _compute_binary_metrics(true_labels, predicted, scores)
    return predictions, metrics


def train_single_run(architecture: str, seed: int, output_dir: Path) -> dict[str, Any]:
    seed_everything(seed)
    rows = read_manifest()
    train_rows = rows[rows["split"] == "train"].copy().reset_index(drop=True)
    val_rows = rows[rows["split"] == "validation"].copy().reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        ManifestDataset(train_rows, training=True),
        batch_size=16,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        ManifestDataset(val_rows),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    model = build_model(architecture, pretrained=(architecture in {"resnet18", "densenet121", "efficientnet_b0"})).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history: list[dict[str, float | int]] = []
    best_balanced_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, 13):
        train_metrics = EpochRunner(model, train_loader, criterion, device, optimizer).run()
        val_metrics = EpochRunner(model, val_loader, criterion, device).run()
        scheduler.step(val_metrics["balanced_accuracy"])
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "validation_loss": float(val_metrics["loss"]),
            "validation_accuracy": float(val_metrics["accuracy"]),
            "validation_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(record)
        score = val_metrics["balanced_accuracy"]
        loss = val_metrics["loss"]
        improved = score > best_balanced_accuracy + 1e-4 or (abs(score - best_balanced_accuracy) <= 1e-4 and loss < best_validation_loss)
        if improved:
            best_balanced_accuracy = score
            best_validation_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path = output_dir / "checkpoint.pth"
            save_checkpoint(
                model,
                checkpoint_path,
                architecture,
                {
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "selection_metric": "validation_balanced_accuracy",
                    "validation_balanced_accuracy": best_balanced_accuracy,
                    "validation_loss": best_validation_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "manifest_sha256": sha256_file(MANIFEST_PATH),
                    "code_revision": _git_head(),
                    "training_config": {
                        "architecture": architecture,
                        "optimizer": "Adam",
                        "learning_rate": 1e-4,
                        "weight_decay": 1e-4,
                        "batch_size": 16,
                        "max_epochs": 12,
                        "early_stopping_patience": 4,
                        "loss": "CrossEntropyLoss",
                        "class_weighting": False,
                        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2)",
                        "augmentation": ["random horizontal flip", "random rotation up to 10 degrees"],
                    },
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "torchvision": __import__("torchvision").__version__,
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "pillow": pillow_version,
                        "device": str(device),
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    },
                },
            )
            _write_json(output_dir / "checkpoint_metadata.json", {
                "architecture": architecture,
                "seed": seed,
                "selected_epoch": best_epoch,
                "selection_metric": "validation_balanced_accuracy",
                "validation_balanced_accuracy": best_balanced_accuracy,
                "validation_loss": best_validation_loss,
                "threshold": 0.5,
                "class_to_idx": CLASS_TO_IDX,
            })
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 4:
            break

    _write_json(output_dir / "history.json", {"epochs": history})
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    _save_learning_curves(history, output_dir)

    if not (output_dir / "checkpoint.pth").is_file():
        raise RuntimeError(f"Training did not save a checkpoint for {architecture} seed {seed}.")

    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(output_dir / "checkpoint.pth", map_location="cpu", weights_only=True)
    model = build_model(architecture, pretrained=(architecture in {"resnet18", "densenet121", "efficientnet_b0"})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions, metrics = evaluate_model(model, rows, split="test")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "architecture": architecture,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "best_validation_loss": best_validation_loss,
        "training_seconds": round(time.perf_counter() - start_time, 3),
        "metrics": metrics,
        "train_class_counts": {label: int((train_rows["label"] == label).sum()) for label in ("no_tumor", "tumor")},
        "validation_class_counts": {label: int((val_rows["label"] == label).sum()) for label in ("no_tumor", "tumor")},
        "code_revision": _git_head(),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1B multi-seed training sweep.")
    parser.add_argument("--architecture", choices=("resnet18", "small_cnn", "densenet121", "efficientnet_b0", "all"), default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=PHASE1B_DIR)
    args = parser.parse_args()

    if args.architecture == "all":
        architectures = ARCHITECTURES
    else:
        architectures = (args.architecture,)
    seeds = (args.seed,) if args.seed is not None else SEEDS

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for architecture in architectures:
        for seed in seeds:
            output_dir = args.output_root / f"{architecture}_seed{seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Starting {architecture} seed {seed} -> {output_dir}")
            summary = train_single_run(architecture, seed, output_dir)
            results.append(summary)
    summary_path = args.output_root / "phase1b_summary.json"
    _write_json(summary_path, {"runs": results})
    print(json.dumps({"runs": results}, indent=2))


if __name__ == "__main__":
    main()
