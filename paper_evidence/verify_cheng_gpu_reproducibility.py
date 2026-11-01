"""
Re-run Cheng screened (1,689) and unscreened (3,064) on GPU and compare with stored CPU results.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint
from BrainTumorAI.external_validation import _predict_cheng_archive

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

archive_path = Path("1512427.zip")
if not archive_path.exists():
    raise FileNotFoundError(f"Archive not found: {archive_path}")

ckpt_path = 'BrainTumorAI/models/tumor_classifier_reproducible.pth'
model, meta = load_checkpoint(ckpt_path, map_location=device)
model = model.to(device)
model.eval()

# 1. Unscreened Cheng (3,064)
print("\n" + "="*50)
print("Evaluating Cheng UNSCREENED (3,064 slices)...")
manifest_unscreened = pd.read_csv('BrainTumorAI/results/external_validation/cheng_figshare_v8/manifest.csv')
stored_unscreened = pd.read_csv('BrainTumorAI/results/external_validation/cheng_figshare_v8/predictions.csv')

t0 = time.time()
scores_unscreened = _predict_cheng_archive(
    model=model,
    rows=manifest_unscreened,
    archive_path=archive_path,
    device=device,
    batch_size=128
)
t1 = time.time()
runtime_unscreened = t1 - t0
print(f"Unscreened completed in {runtime_unscreened:.2f}s ({len(manifest_unscreened)/runtime_unscreened:.1f} imgs/s)")

stored_scores_u = stored_unscreened['tumor_score'].values
diffs_u = np.abs(scores_unscreened - stored_scores_u)
max_diff_u = float(np.max(diffs_u))
mean_diff_u = float(np.mean(diffs_u))

preds_live_u = (scores_unscreened >= 0.5).astype(int)
preds_stored_u = (stored_scores_u >= 0.5).astype(int)
flips_u = int(np.sum(preds_live_u != preds_stored_u))
pos_live_u = int(np.sum(preds_live_u == 1))
pos_stored_u = int(np.sum(preds_stored_u == 1))

print(f"Max abs diff: {max_diff_u:.8e}")
print(f"Mean abs diff: {mean_diff_u:.8e}")
print(f"Flipped count: {flips_u}")
print(f"Stored positive count: {pos_stored_u} / {len(scores_unscreened)} ({pos_stored_u/len(scores_unscreened)*100:.2f}%)")
print(f"Live GPU positive count: {pos_live_u} / {len(scores_unscreened)} ({pos_live_u/len(scores_unscreened)*100:.2f}%)")

# 2. Screened Cheng (1,689)
print("\n" + "="*50)
print("Evaluating Cheng SCREENED (1,689 slices)...")
stored_screened = pd.read_csv('BrainTumorAI/results/external_validation/cheng_figshare_v8_overlap_screened/predictions.csv')
manifest_screened = stored_screened

t0 = time.time()
scores_screened = _predict_cheng_archive(
    model=model,
    rows=manifest_screened,
    archive_path=archive_path,
    device=device,
    batch_size=128
)
t1 = time.time()
runtime_screened = t1 - t0
print(f"Screened completed in {runtime_screened:.2f}s ({len(manifest_screened)/runtime_screened:.1f} imgs/s)")

stored_scores_s = stored_screened['tumor_score'].values
diffs_s = np.abs(scores_screened - stored_scores_s)
max_diff_s = float(np.max(diffs_s))
mean_diff_s = float(np.mean(diffs_s))

preds_live_s = (scores_screened >= 0.5).astype(int)
preds_stored_s = (stored_scores_s >= 0.5).astype(int)
flips_s = int(np.sum(preds_live_s != preds_stored_s))
pos_live_s = int(np.sum(preds_live_s == 1))
pos_stored_s = int(np.sum(preds_stored_s == 1))

print(f"Max abs diff: {max_diff_s:.8e}")
print(f"Mean abs diff: {mean_diff_s:.8e}")
print(f"Flipped count: {flips_s}")
print(f"Stored positive count: {pos_stored_s} / {len(scores_screened)} ({pos_stored_s/len(scores_screened)*100:.2f}%)")
print(f"Live GPU positive count: {pos_live_s} / {len(scores_screened)} ({pos_live_s/len(scores_screened)*100:.2f}%)")

report = {
    "device": str(device),
    "device_name": torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
    "unscreened": {
        "n_slices": len(scores_unscreened),
        "runtime_seconds": runtime_unscreened,
        "throughput_imgs_per_s": len(scores_unscreened) / runtime_unscreened,
        "max_abs_score_diff": max_diff_u,
        "mean_abs_score_diff": mean_diff_u,
        "flipped_predictions": flips_u,
        "stored_positives": pos_stored_u,
        "live_positives": pos_live_u
    },
    "screened": {
        "n_slices": len(scores_screened),
        "runtime_seconds": runtime_screened,
        "throughput_imgs_per_s": len(scores_screened) / runtime_screened,
        "max_abs_score_diff": max_diff_s,
        "mean_abs_score_diff": mean_diff_s,
        "flipped_predictions": flips_s,
        "stored_positives": pos_stored_s,
        "live_positives": pos_live_s
    }
}

with open('paper_evidence/cheng_gpu_reproducibility.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\nSaved report to paper_evidence/cheng_gpu_reproducibility.json")
