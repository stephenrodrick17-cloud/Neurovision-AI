"""
Evaluate BraTS batch size effects (batch 1 vs 128) on a patient-stratified subset of 10,000 slices.
Primary ResNet-18 model on brats_oriented slices.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score, confusion_matrix
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform
from paper_evidence.brats_dataset import BraTSPNGDataset

def compute_m(y_true, scores):
    preds = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "sensitivity": float(recall_score(y_true, preds, pos_label=1)),
        "specificity": float(recall_score(y_true, preds, pos_label=0)),
        "auc": float(roc_auc_score(y_true, scores)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    df_10k = pd.read_csv('paper_evidence/brats_10k_stratified_manifest.csv')
    print(f"Loaded 10k subset: {len(df_10k)} slices across {df_10k['patient_id'].nunique()} patients")
    
    transform = make_inference_transform()
    root_dir = 'D:/datasets/prepared/brats_oriented'
    dataset = BraTSPNGDataset(root_dir, df_10k, transform)
    
    ckpt_path = 'BrainTumorAI/models/tumor_classifier_reproducible.pth'
    model, _ = load_checkpoint(ckpt_path, map_location=device)
    model = model.to(device).eval()
    
    labels = df_10k['label_idx'].values
    
    print("Evaluating Batch Size 1...")
    loader_b1 = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    scores_b1 = np.zeros(len(df_10k), dtype=np.float32)
    
    with torch.no_grad():
        for imgs, idxs in loader_b1:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            scores_b1[idxs.numpy()] = probs
            
    print("Evaluating Batch Size 128...")
    loader_b128 = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    scores_b128 = np.zeros(len(df_10k), dtype=np.float32)
    
    with torch.no_grad():
        for imgs, idxs in loader_b128:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            scores_b128[idxs.numpy()] = probs
            
    diffs = np.abs(scores_b1 - scores_b128)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    
    preds_b1 = (scores_b1 >= 0.5).astype(int)
    preds_b128 = (scores_b128 >= 0.5).astype(int)
    flips = int(np.sum(preds_b1 != preds_b128))
    
    m_b1 = compute_m(labels, scores_b1)
    m_b128 = compute_m(labels, scores_b128)
    
    print(f"\n--- Results on 10,000 Stratified BraTS Slices ---")
    print(f"Max abs diff: {max_diff:.8e}")
    print(f"Mean abs diff: {mean_diff:.8e}")
    print(f"Flips: {flips} / 10000")
    print(f"Batch 1:   Acc={m_b1['accuracy']:.4f}, Sens={m_b1['sensitivity']:.4f}, Spec={m_b1['specificity']:.4f}, AUC={m_b1['auc']:.4f}")
    print(f"Batch 128: Acc={m_b128['accuracy']:.4f}, Sens={m_b128['sensitivity']:.4f}, Spec={m_b128['specificity']:.4f}, AUC={m_b128['auc']:.4f}")
    
    res = {
        "n_slices": len(df_10k),
        "n_patients": int(df_10k['patient_id'].nunique()),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "flips": flips,
        "batch_1_metrics": m_b1,
        "batch_128_metrics": m_b128,
        "metric_changes": {
            "accuracy_delta": m_b128['accuracy'] - m_b1['accuracy'],
            "sensitivity_delta": m_b128['sensitivity'] - m_b1['sensitivity'],
            "specificity_delta": m_b128['specificity'] - m_b1['specificity'],
            "auc_delta": m_b128['auc'] - m_b1['auc']
        }
    }
    
    with open('paper_evidence/brats_10k_batch_size_effects.json', 'w') as f:
        json.dump(res, f, indent=2)
    print("Saved to paper_evidence/brats_10k_batch_size_effects.json")

if __name__ == '__main__':
    main()
