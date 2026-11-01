"""
Batch size comparison:
(a) Primary ResNet-18 and SmallCNN on all 606 discovery test images at batch 1 vs batch 32:
    Count flipped predictions and max score difference.
(b) On a patient-stratified subset of 10,000 BraTS slices, compare batch 1 vs batch 128:
    Flips and change in each headline metric.
State which batch size each earlier result (Cheng, BraTS, phase1b, primary) used.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform
from BrainTumorAI.train_classifier import ManifestDataset
from BrainTumorAI.evaluate_classifier import compute_metrics
from paper_evidence.brats_dataset import BraTSPNGDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
transform = make_inference_transform()

# =====================================================================
# Part (a): Discovery Test Images (N=606) - Batch 1 vs Batch 32
# =====================================================================
print("\n" + "="*60)
print("Part (a): Discovery Test Set (N=606) - Batch 1 vs Batch 32")
print("="*60)

split_df = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

models = [
    ("resnet18", "BrainTumorAI/models/tumor_classifier_reproducible.pth"),
    ("small_cnn", "BrainTumorAI/models/small_cnn_best.pth")
]

part_a_results = {}

for model_name, ckpt_path in models:
    print(f"\n--- Model: {model_name} ---")
    model, _ = load_checkpoint(ckpt_path, map_location=device)
    model = model.to(device).eval()
    
    # 1. Batch size 1 (sequential)
    scores_b1 = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = row['source_file']
            img = Image.open(img_path).convert('RGB')
            t = transform(img).unsqueeze(0).to(device)
            prob = torch.softmax(model(t), dim=1)[0, 1].item()
            scores_b1.append(prob)
    scores_b1 = np.array(scores_b1)
    
    # 2. Batch size 32 (DataLoader)
    ds = ManifestDataset(test_df, training=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    scores_b32 = []
    with torch.no_grad():
        for imgs, _ in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
            scores_b32.extend(probs)
    scores_b32 = np.array(scores_b32)
    
    diffs = np.abs(scores_b1 - scores_b32)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    
    preds_b1 = (scores_b1 >= 0.5).astype(int)
    preds_b32 = (scores_b32 >= 0.5).astype(int)
    flips = int(np.sum(preds_b1 != preds_b32))
    
    flipped_details = []
    for fi in np.where(preds_b1 != preds_b32)[0]:
        flipped_details.append({
            "source_file": test_df.iloc[fi]['source_file'],
            "score_batch1": float(scores_b1[fi]),
            "score_batch32": float(scores_b32[fi]),
            "delta": float(scores_b1[fi] - scores_b32[fi])
        })
        
    part_a_results[model_name] = {
        "max_abs_score_diff": max_diff,
        "mean_abs_score_diff": mean_diff,
        "flipped_predictions_count": flips,
        "flipped_details": flipped_details
    }
    print(f"Max abs diff: {max_diff:.8e}")
    print(f"Mean abs diff: {mean_diff:.8e}")
    print(f"Flipped predictions: {flips}")
    if flips > 0:
        print(f"Details: {flipped_details}")

# =====================================================================
# Part (b): BraTS 10,000 Stratified Slices - Batch 1 vs Batch 128
# =====================================================================
print("\n" + "="*60)
print("Part (b): BraTS 10,000 Stratified Slices - Batch 1 vs Batch 128")
print("="*60)

brats_manifest = pd.read_csv('D:/datasets/prepared/brats/manifest.csv')
rng = np.random.default_rng(42)

# Sample 10,000 stratified by patient
p_summary = brats_manifest.groupby('patient_id').agg(
    n_slices=('slice_id', 'count'),
    tumor_slices=('label_idx', 'sum')
).reset_index()
p_summary['tumor_prop'] = p_summary['tumor_slices'] / p_summary['n_slices']
p_summary['prop_bin'] = pd.qcut(p_summary['tumor_prop'], q=5, labels=False, duplicates='drop')

sampled_pids = []
target_patients = 100
for b in p_summary['prop_bin'].unique():
    stratum_pids = p_summary[p_summary['prop_bin'] == b]['patient_id'].values
    n_s = len(stratum_pids) * target_patients // len(p_summary)
    sampled_pids.extend(rng.choice(stratum_pids, size=n_s, replace=False))

sub_brats = brats_manifest[brats_manifest['patient_id'].isin(sampled_pids)].head(10000).reset_index(drop=True)
print(f"Sub-BraTS slices: {len(sub_brats)} across {sub_brats['patient_id'].nunique()} patients.")

brats_root = 'D:/datasets/prepared/brats_oriented'
res_model, _ = load_checkpoint('BrainTumorAI/models/tumor_classifier_reproducible.pth', map_location=device)
res_model = res_model.to(device).eval()

# Batch 1 (sample of 10,000)
scores_brats_b1 = []
t0 = time.time()
with torch.no_grad():
    for idx, row in sub_brats.iterrows():
        p = os.path.join(brats_root, row['source_file'])
        im = Image.open(p).convert('RGB')
        t = transform(im).unsqueeze(0).to(device)
        prob = torch.softmax(res_model(t), dim=1)[0, 1].item()
        scores_brats_b1.append(prob)
        if (idx + 1) % 2500 == 0:
            print(f"  [Batch 1] {idx+1}/10000...")
scores_brats_b1 = np.array(scores_brats_b1)
t1 = time.time()
print(f"Batch 1 inference: {t1-t0:.2f}s")

# Batch 128
b_ds = BraTSPNGDataset(brats_root, sub_brats, transform)
b_loader = DataLoader(b_ds, batch_size=128, shuffle=False)
scores_brats_b128 = []
t0 = time.time()
with torch.no_grad():
    for imgs, _ in b_loader:
        probs = torch.softmax(res_model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
        scores_brats_b128.extend(probs)
scores_brats_b128 = np.array(scores_brats_b128)
t1 = time.time()
print(f"Batch 128 inference: {t1-t0:.2f}s")

diffs_brats = np.abs(scores_brats_b1 - scores_brats_b128)
max_diff_brats = float(np.max(diffs_brats))
mean_diff_brats = float(np.mean(diffs_brats))

preds_brats_b1 = (scores_brats_b1 >= 0.5).astype(int)
preds_brats_b128 = (scores_brats_b128 >= 0.5).astype(int)
flips_brats = int(np.sum(preds_brats_b1 != preds_brats_b128))

y_true_brats = sub_brats['label_idx'].values
metrics_b1 = compute_metrics(y_true_brats, scores_brats_b1)
metrics_b128 = compute_metrics(y_true_brats, scores_brats_b128)

metric_changes = {}
for k in ["accuracy", "balanced_accuracy", "tumor_sensitivity_recall", "no_tumor_specificity", "tumor_precision", "tumor_f1", "roc_auc", "average_precision"]:
    metric_changes[k] = {
        "batch_1": metrics_b1[k],
        "batch_128": metrics_b128[k],
        "delta": metrics_b128[k] - metrics_b1[k]
    }

part_b_results = {
    "n_slices": len(sub_brats),
    "max_abs_score_diff": max_diff_brats,
    "mean_abs_score_diff": mean_diff_brats,
    "flipped_predictions_count": flips_brats,
    "metric_changes": metric_changes
}

print(f"\nBraTS 10K Max abs diff: {max_diff_brats:.8e}")
print(f"BraTS 10K Mean abs diff: {mean_diff_brats:.8e}")
print(f"BraTS 10K Flips: {flips_brats}")
print(f"Metric deltas: {json.dumps(metric_changes, indent=2)}")

final_report = {
    "part_a_discovery_test_b1_vs_b32": part_a_results,
    "part_b_brats_10k_b1_vs_b128": part_b_results
}

with open('paper_evidence/batch_size_effects_report.json', 'w') as f:
    json.dump(final_report, f, indent=2)

print("\nSaved report to paper_evidence/batch_size_effects_report.json")

