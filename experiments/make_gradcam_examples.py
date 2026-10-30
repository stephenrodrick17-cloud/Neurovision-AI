from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt

from BrainTumorAI.classifier_pipeline import DEFAULT_CHECKPOINT, PROJECT_ROOT, load_checkpoint, preprocess_image
from BrainTumorAI.heatmap_gen import GradCAM
from experiments.gradcam_utils import resize_heatmap_to_original_overlay

TARGET_DEFAULT_NAMES = [
    "Cancer (892).jpg",
    "Cancer (1).jpg",
    "Not Cancer  (739).jpg",
    "Not Cancer  (1809).jpg",
    "Not Cancer  (164).jpg",
    "Not Cancer  (121).jpg",
    "Cancer (882).jpg",
    "Cancer (2315).jpg",
]


def _make_fig_title(true_label: str, pred_label: str, tumor_score: float) -> str:
    return f"true={true_label}\npred={pred_label}\ntumor={tumor_score:.4f}"


def main() -> None:
    model, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    model.eval()

    predictions_path = PROJECT_ROOT / "BrainTumorAI" / "results" / "evaluation" / "test_predictions.csv"
    rows = pd.read_csv(predictions_path)
    rows_by_name: dict[str, pd.Series] = {}
    for _, row in rows.iterrows():
        rows_by_name[Path(row["source_file"]).name] = row

    out_dir = PROJECT_ROOT / "experiments_results" / "phase0" / "gradcam_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_items: list[tuple[str, np.ndarray, str, str, float, Path]] = []
    for filename in TARGET_DEFAULT_NAMES:
        row = rows_by_name[filename]
        source_file = PROJECT_ROOT / row["source_file"]
        image = Image.open(source_file).convert("RGB")
        tensor = preprocess_image(image).unsqueeze(0)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            tumor_score = float(probs[0, 1].item())

        grad_cam = GradCAM(model, model.layer4[-1])
        heatmap = grad_cam.generate_heatmap(tensor, target_class=pred_idx)
        overlay = resize_heatmap_to_original_overlay(np.array(image), heatmap, alpha=0.4)

        true_label = "tumor" if int(row["true_label_idx"]) == 1 else "no_tumor"
        pred_label = "tumor" if pred_idx == 1 else "no_tumor"
        output_path = out_dir / f"{Path(filename).stem}_overlay.png"
        Image.fromarray(overlay).save(output_path)
        figure_items.append((filename, overlay, true_label, pred_label, tumor_score, output_path))

    grid_path = out_dir / "gradcam_grid_2x4.png"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, item in zip(axes.flat, figure_items):
        filename, overlay, true_label, pred_label, tumor_score, _ = item
        ax.imshow(overlay)
        ax.set_title(_make_fig_title(true_label, pred_label, tumor_score), fontsize=9)
        ax.axis("off")
        ax.set_xlabel(filename, fontsize=7)
    fig.tight_layout()
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)

    metadata_path = out_dir / "gradcam_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "tumor_score": float(tumor_score),
                    "output_png": str(path),
                }
                for filename, _, true_label, pred_label, tumor_score, path in figure_items
            ],
            handle,
            indent=2,
        )

    print(f"Saved Grad-CAM grid: {grid_path}")
    print(f"Saved individual overlays: {len(figure_items)}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
