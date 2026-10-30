from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.torch_version import TorchVersion
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "archive" / "Brain Tumor Data Set" / "Brain Tumor Data Set"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "tumor_classifier_reproducible.pth"

CLASS_TO_IDX = {"no_tumor": 0, "tumor": 1}
IDX_TO_CLASS = {index: label for label, index in CLASS_TO_IDX.items()}
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Raised when a research checkpoint is missing or incompatible."""


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def make_model(architecture: str = "resnet18", pretrained: bool = False) -> nn.Module:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_IDX))
        return model
    if architecture == "small_cnn":
        if pretrained:
            raise ValueError("The small_cnn baseline does not use pretrained weights.")
        return SmallCNN()
    raise ValueError(f"Unsupported model architecture: {architecture}")


def make_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def make_training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = make_inference_transform()(image.convert("RGB"))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Inference preprocessing must return a tensor.")
    return tensor


def predict_image(model: nn.Module, image: Image.Image) -> torch.Tensor:
    device = next(model.parameters()).device
    with torch.inference_mode():
        return model(preprocess_image(image).unsqueeze(0).to(device))[0]


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    architecture: str,
    metadata: dict[str, Any],
) -> None:
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.setdefault("architecture", architecture)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "class_to_idx": CLASS_TO_IDX,
        "model_state_dict": model.state_dict(),
        "metadata": checkpoint_metadata,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained classifier checkpoint not found: {checkpoint_path}. "
            "Train one with `python BrainTumorAI/train_classifier.py` or set "
            "NEUROVISION_CHECKPOINT to a valid checkpoint."
        )

    try:
            with torch.serialization.safe_globals([TorchVersion]):
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Could not read checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Checkpoint {checkpoint_path} does not use the supported research-checkpoint format. "
            "Retrain with BrainTumorAI/train_classifier.py."
        )
    if checkpoint.get("class_to_idx") != CLASS_TO_IDX:
        raise CheckpointError(
            f"Checkpoint class mapping must be {CLASS_TO_IDX}, got {checkpoint.get('class_to_idx')}."
        )

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, str):
        raise CheckpointError(f"Checkpoint {checkpoint_path} has no valid architecture name.")
    model = make_model(architecture)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise CheckpointError(f"Checkpoint {checkpoint_path} has no model_state_dict.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint {checkpoint_path} is incompatible with {architecture}: {exc}") from exc
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointError(f"Checkpoint {checkpoint_path} has no metadata object.")
    metadata = {**metadata, "architecture": architecture}
    model.eval()
    return model, metadata


def load_legacy_resnet18(path: str | Path, map_location: str | torch.device = "cpu") -> nn.Module:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Legacy checkpoint not found: {checkpoint_path}")
    try:
        with torch.serialization.safe_globals([TorchVersion]):
            state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Could not read legacy checkpoint {checkpoint_path}: {exc}") from exc
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise CheckpointError(f"Legacy checkpoint {checkpoint_path} is not a state dictionary.")
    model = make_model("resnet18")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Legacy checkpoint {checkpoint_path} is incompatible with ResNet-18: {exc}") from exc
    model.eval()
    return model
