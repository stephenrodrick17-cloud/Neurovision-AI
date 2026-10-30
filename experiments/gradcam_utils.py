from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def resize_heatmap_to_original_overlay(image: Image.Image | np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image dimensions without stretching to 224x224."""
    if heatmap.ndim != 2 or heatmap.size == 0 or not np.isfinite(heatmap).all():
        raise ValueError("The Grad-CAM heatmap must be a finite, non-empty 2D array.")

    if isinstance(image, Image.Image):
        image_array = np.asarray(image.convert("RGB"))
    else:
        image_array = np.asarray(image)
        if image_array.ndim == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[-1] == 1:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[-1] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    if image_array.shape[:2] == (0, 0):
        raise ValueError("The input image is empty.")

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value <= 0.0:
        raise ValueError("Grad-CAM heatmap has no activation contrast for overlay.")
    heatmap = heatmap / max_value

    resized_heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored_heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(resized_heatmap, 0, 1)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_array.astype(np.uint8), 1.0 - alpha, colored_heatmap, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_overlay_for_image(image_path: str | Path, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.4) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = resize_heatmap_to_original_overlay(image, heatmap, alpha=alpha)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)
