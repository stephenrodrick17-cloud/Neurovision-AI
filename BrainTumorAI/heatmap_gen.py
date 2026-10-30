from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

    def generate_heatmap(self, input_image: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        if input_image.ndim != 4 or input_image.shape[0] != 1:
            raise ValueError("Grad-CAM expects one image with shape [1, channels, height, width].")
        if not torch.is_floating_point(input_image) or not torch.isfinite(input_image).all():
            raise ValueError("Grad-CAM input must contain finite floating-point values.")
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def save_activation(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            activations.append(output)
            if output.requires_grad:
                output.register_hook(gradients.append)

        forward_handle = self.target_layer.register_forward_hook(save_activation)
        self.model.zero_grad(set_to_none=True)
        try:
            model_input = input_image.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(model_input)
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError("The classifier must return logits with shape [1, classes].")
                selected_class = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
                if not 0 <= selected_class < logits.shape[1]:
                    raise ValueError(
                        f"target_class must be in [0, {logits.shape[1] - 1}], got {selected_class}."
                    )
                logits[0, selected_class].backward()
            if not activations or not gradients:
                raise RuntimeError("Grad-CAM target layer did not produce activations and gradients.")
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations[0]).sum(dim=1).squeeze(0)
            cam = F.relu(cam)
            if not torch.isfinite(cam).all():
                raise RuntimeError("Grad-CAM produced non-finite activation values.")
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max <= cam_min:
                raise RuntimeError(
                    "Grad-CAM produced no class-specific activation contrast; the heatmap is unavailable."
                )
            cam = (cam - cam_min) / (cam_max - cam_min)
            return cam.detach().cpu().numpy()
        finally:
            forward_handle.remove()
            self.model.zero_grad(set_to_none=True)


def apply_heatmap(image_path: str | Path, heatmap: np.ndarray) -> np.ndarray:
    if heatmap.ndim != 2 or heatmap.size == 0 or not np.isfinite(heatmap).all():
        raise ValueError("Heatmap overlay requires a finite, non-empty 2D heatmap.")
    if float(np.max(heatmap)) <= float(np.min(heatmap)):
        raise ValueError("Heatmap overlay is unavailable because the heatmap has no activation contrast.")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image for heatmap overlay: {image_path}")
    image = cv2.resize(image, (224, 224))
    resized_heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_image = cv2.applyColorMap(np.uint8(255 * np.clip(resized_heatmap, 0, 1)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_image, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
