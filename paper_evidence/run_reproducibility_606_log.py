"""
Pre-Submission Analysis 5: Per-Image Reproducibility Log for the 606 Discovery Test Images

Tests and logs per-image tumor probability scores across all 606 held-out test images
(BrainTumorAI/results/split_manifest.csv, split == 'test') for both primary checkpoints:
  1. ResNet-18 (BrainTumorAI/models/tumor_classifier_reproducible.pth)
  2. SmallCNN  (BrainTumorAI/models/small_cnn_best.pth)

Compares:
  - stored_score: saved score in BrainTumorAI/results/evaluation/test_predictions.csv (evaluated at batch_size=32)
  - gpu_b32_run1: fresh GPU (cuda:0) inference at batch_size=32 (matching evaluate_classifier.py)
  - gpu_b32_run2: second independent GPU (cuda:0) inference pass at batch_size=32
  - cpu_b32:      fresh CPU inference at batch_size=32
  - gpu_b1:       fresh GPU (cuda:0) inference at batch_size=1
  - cpu_b1:       fresh CPU inference at batch_size=1

Saves:
  - paper_evidence/discovery_606_reproducibility_log.csv (all 606 images x 2 models = 1,212 rows with per-image scores and absolute differences)
  - paper_evidence/discovery_606_reproducibility_log.json (summary statistics: max_abs_diff, mean_abs_diff, median_abs_diff, p99_abs_diff, flipped_predictions)
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from BrainTumorAI.classifier_pipeline import load_checkpoint, sha256_file
from BrainTumorAI.train_classifier import ManifestDataset


def run_pass(model: torch.nn.Module, test_df: pd.DataFrame, device: torch.device, batch_size: int) -> np.ndarray:
    ds = ManifestDataset(test_df, training=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    scores: list[float] = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            scores.extend(probs.tolist())
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=np.float64)


def compare_pair(s_a: np.ndarray, s_b: np.ndarray, test_df: pd.DataFrame) -> dict[str, Any]:
    diffs = np.abs(s_a - s_b)
    p_a = (s_a >= 0.5).astype(np.int32)
    p_b = (s_b >= 0.5).astype(np.int32)
    flips_idx = np.where(p_a != p_b)[0]
    max_idx = int(np.argmax(diffs))
    return {
        "max_abs_diff": float(np.max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "median_abs_diff": float(np.median(diffs)),
        "p99_abs_diff": float(np.percentile(diffs, 99)),
        "max_diff_source_file": str(test_df.iloc[max_idx]["source_file"]),
        "max_diff_score_a": float(s_a[max_idx]),
        "max_diff_score_b": float(s_b[max_idx]),
        "flipped_predictions_at_0_5": int(len(flips_idx)),
        "flipped_details": [
            {
                "source_file": str(test_df.iloc[i]["source_file"]),
                "score_a": float(s_a[i]),
                "score_b": float(s_b[i]),
                "abs_diff": float(diffs[i]),
            }
            for i in flips_idx
        ],
    }


def main() -> None:
    t0 = time.time()
    gpu_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_dev = torch.device("cpu")

    split_df = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "split_manifest.csv")
    test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)
    stored_df = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "evaluation" / "test_predictions.csv")

    model_specs = [
        ("resnet18", ROOT / "BrainTumorAI" / "models" / "tumor_classifier_reproducible.pth", "resnet18"),
        ("small_cnn", ROOT / "BrainTumorAI" / "models" / "small_cnn_best.pth", "small_cnn_small_cnn_best"),
    ]

    csv_frames = []
    summary_models: dict[str, Any] = {}

    for m_name, ckpt_path, stored_key in model_specs:
        print(f"Running reproducibility passes for {m_name} ({ckpt_path.name})...", flush=True)
        sub_stored = stored_df[stored_df["model"] == stored_key].copy().reset_index(drop=True)
        assert list(sub_stored["source_file"]) == list(test_df["source_file"])
        s_stored = sub_stored["tumor_score"].to_numpy(dtype=np.float64)

        model, _ = load_checkpoint(ckpt_path, map_location="cpu")
        s_gpu_b32_1 = run_pass(model, test_df, gpu_dev, batch_size=32)
        s_gpu_b32_2 = run_pass(model, test_df, gpu_dev, batch_size=32)
        s_cpu_b32 = run_pass(model, test_df, cpu_dev, batch_size=32)
        s_gpu_b1 = run_pass(model, test_df, gpu_dev, batch_size=1)
        s_cpu_b1 = run_pass(model, test_df, cpu_dev, batch_size=1)

        comparisons = {
            "stored_b32_vs_gpu_b32_run1": compare_pair(s_stored, s_gpu_b32_1, test_df),
            "stored_b32_vs_cpu_b32": compare_pair(s_stored, s_cpu_b32, test_df),
            "gpu_b32_run1_vs_gpu_b32_run2": compare_pair(s_gpu_b32_1, s_gpu_b32_2, test_df),
            "gpu_b32_run1_vs_cpu_b32": compare_pair(s_gpu_b32_1, s_cpu_b32, test_df),
            "gpu_b1_vs_cpu_b1": compare_pair(s_gpu_b1, s_cpu_b1, test_df),
            "stored_b32_vs_gpu_b1": compare_pair(s_stored, s_gpu_b1, test_df),
            "gpu_b32_run1_vs_gpu_b1": compare_pair(s_gpu_b32_1, s_gpu_b1, test_df),
        }

        for comp_k, comp_v in comparisons.items():
            print(
                f"  [{m_name:9s} | {comp_k:28s}] "
                f"max_abs_diff={comp_v['max_abs_diff']:.8e}, mean_abs_diff={comp_v['mean_abs_diff']:.8e}, "
                f"flips={comp_v['flipped_predictions_at_0_5']}",
                flush=True,
            )

        summary_models[m_name] = {
            "checkpoint_path": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "n_test_images": int(len(test_df)),
            "comparisons": comparisons,
        }

        df_m = pd.DataFrame({
            "model": m_name,
            "source_file": test_df["source_file"],
            "duplicate_group": test_df["duplicate_group"],
            "true_label": test_df["label"],
            "true_label_idx": test_df["label_idx"],
            "score_stored_b32": s_stored,
            "score_gpu_b32_run1": s_gpu_b32_1,
            "score_gpu_b32_run2": s_gpu_b32_2,
            "score_cpu_b32": s_cpu_b32,
            "score_gpu_b1": s_gpu_b1,
            "score_cpu_b1": s_cpu_b1,
            "abs_diff_stored_vs_gpu_b32": np.abs(s_stored - s_gpu_b32_1),
            "abs_diff_stored_vs_cpu_b32": np.abs(s_stored - s_cpu_b32),
            "abs_diff_gpu_b32_vs_cpu_b32": np.abs(s_gpu_b32_1 - s_cpu_b32),
            "abs_diff_gpu_b1_vs_cpu_b1": np.abs(s_gpu_b1 - s_cpu_b1),
            "abs_diff_b32_vs_b1_gpu": np.abs(s_gpu_b32_1 - s_gpu_b1),
        })
        csv_frames.append(df_m)

    out_csv = ROOT / "paper_evidence" / "discovery_606_reproducibility_log.csv"
    pd.concat(csv_frames, ignore_index=True).to_csv(out_csv, index=False)

    out_json = ROOT / "paper_evidence" / "discovery_606_reproducibility_log.json"
    payload = {
        "analysis": "Per-image reproducibility log across all 606 Discovery test images",
        "script_path": "paper_evidence/run_reproducibility_606_log.py",
        "csv_log_path": "paper_evidence/discovery_606_reproducibility_log.csv",
        "csv_log_sha256": sha256_file(out_csv),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": summary_models,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved per-image reproducibility log to {out_csv} and {out_json}", flush=True)


if __name__ == "__main__":
    main()

