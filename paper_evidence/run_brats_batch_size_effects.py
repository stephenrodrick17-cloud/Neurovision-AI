"""
Evaluate BraTS batch size effects (batch 1 vs 128) on a patient-stratified subset of 10,000 slices.
Headline metrics: Accuracy, Sensitivity, Specificity, AUC.
Compare flips and metrics.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

class BraTSSubsetDataset(Dataset):
    def __init__(self, slice_df, transform):
        self.slice_df = slice_df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.slice_df)
        
    def __getitem__(self, idx):
        row = self.slice_df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        # Oriented transform (np.rot90 1 counter-clockwise = anterior UP)
        img_np = np.array(img)
        img_rot = np.rot90(img_np, 1)
        img_pil = Image.fromarray(img_rot)
        return self.transform(img_pil), int(row['label']), row['filepath'], row['patient_id']

def compute_metrics(y_true, scores):
    preds = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float('nan')
    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "auc": float(auc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load metadata of brats
    manifest_path = "BrainTumorAI/results/external_validation/brats/slice_manifest.csv"
    if not os.path.exists(manifest_path):
        manifest_path = "BrainTumorAI/results/brats/slice_manifest.csv"
    if not os.path.exists(manifest_path):
        # find from existing predictions
        pred_path = "BrainTumorAI/results/external_validation/brats/predictions.csv"
        df_all = pd.read_csv(pred_path)
    else:
        df_all = pd.read_csv(manifest_path)
        
    print(f"Total BraTS slices available: {len(df_all)}")
    
    # Sample 10,000 slices stratified across patients
    # Group by patient_id, take ~8 slices per patient across all ~1251 patients
    patients = sorted(df_all['patient_id'].unique())
    np.random.seed(42)
    selected_indices = []
    
    # Calculate how many per patient
    target_total = 10000
    n_patients = len(patients)
    per_patient = target_total // n_patients
    remainder = target_total % n_patients
    
    grouped = df_all.groupby('patient_id')
    for i, pid in enumerate(patients):
        group_indices = grouped.get_group(pid).index.values
        k = per_patient + (1 if i < remainder else 0)
        k = min(k, len(group_indices))
        chosen = np.random.choice(group_indices, size=k, replace=False)
        selected_indices.extend(chosen)
        
    subset_df = df_all.iloc[selected_indices].copy().reset_index(drop=True)
    print(f"Selected subset: {len(subset_df)} slices from {len(subset_df['patient_id'].unique())} patients")
    
    transform = make_inference_transform()
    dataset = BraTSSubsetDataset(subset_df, transform)
    
    # Load primary ResNet-18 model
    ckpt_path = "BrainTumorAI/models/tumor_classifier_reproducible.pth"
    model, _ = load_checkpoint(ckpt_path, map_location=device)
    model = model.to(device).eval()
    
    print("\nEvaluating batch size 1 (num_workers=0)...")
    loader_b1 = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    scores_b1 = []
    labels_list = []
    with torch.no_grad():
        for imgs, lbls, _, _ in loader_b1:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
            scores_b1.extend(probs)
            labels_list.extend(lbls.numpy())
    scores_b1 = np.array(scores_b1)
    labels = np.array(labels_list)
    
    print("Evaluating batch size 128 (num_workers=0)...")
    loader_b128 = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    scores_b128 = []
    with torch.no_grad():
        for imgs, _, _, _ in loader_b128:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
            scores_b128.extend(probs)
    scores_b128 = np.array(scores_b128)
    
    diffs = np.abs(scores_b1 - scores_b128)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    
    preds_b1 = (scores_b1 >= 0.5).astype(int)
    preds_b128 = (scores_b128 >= 0.5).astype(int)
    flips = int(np.sum(preds_b1 != preds_b128))
    
    metrics_b1 = compute_metrics(labels, scores_b1)
    metrics_b128 = compute_metrics(labels, scores_b128)
    
    print(f"\n--- Results on 10,000 BraTS slices ---")
    print(f"Max abs diff: {max_diff:.8e}")
    print(f"Mean abs diff: {mean_diff:.8e}")
    print(f"Flips: {flips} / {len(subset_df)}")
    print(f"Batch 1 metrics:   acc={metrics_b1['accuracy']:.6f}, sens={metrics_b1['sensitivity']:.6f}, spec={metrics_b1['specificity']:.6f}, auc={metrics_b1['auc']:.6f}")
    print(f"Batch 128 metrics: acc={metrics_b128['accuracy']:.6f}, sens={metrics_b128['sensitivity']:.6f}, spec={metrics_b128['specificity']:.6f}, auc={metrics_b128['auc']:.6f}")
    
    result = {
        "n_samples": len(subset_df),
        "n_patients": len(subset_df['patient_id'].unique()),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "flips": flips,
        "metrics_batch_1": metrics_b1,
        "metrics_batch_128": metrics_b128,
        "metric_changes": {
            "accuracy_diff": metrics_b128['accuracy'] - metrics_b1['accuracy'],
            "sensitivity_diff": metrics_b128['sensitivity'] - metrics_b1['sensitivity'],
            "specificity_diff": metrics_b128['specificity'] - metrics_b1['specificity'],
            "auc_diff": metrics_b128['auc'] - metrics_b1['auc']
        }
    }
    
    os.makedirs('paper_evidence', exist_ok=True)
    with open('paper_evidence/brats_batch_size_effects.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nSaved to paper_evidence/brats_batch_size_effects.json")

if __name__ == '__main__':
    main()

