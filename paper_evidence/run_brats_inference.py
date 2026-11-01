import os
import sys
import time
import gzip
import json
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)

class BraTSPNGDataset(Dataset):
    def __init__(self, root_dir, df, transform):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        self.files = [os.path.join(root_dir, f) for f in df['source_file']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx]).convert('RGB')
        return self.transform(im), idx

def run_cohort_inference(name, root_dir, manifest_df, model, device, transform, batch_size=128):
    print(f"\n==========================================")
    print(f"Starting inference for: {name}")
    print(f"Root: {root_dir} | Total slices: {len(manifest_df)}")
    print(f"==========================================")
    
    dataset = BraTSPNGDataset(root_dir, manifest_df, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    scores = np.zeros(len(manifest_df), dtype=np.float32)
    t0 = time.time()
    processed = 0
    
    with torch.no_grad():
        for batch_x, batch_idxs in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            tumor_probs = probs[:, 1].cpu().numpy()
            scores[batch_idxs.numpy()] = tumor_probs
            processed += len(batch_idxs)
            if processed % 10000 == 0 or processed == len(manifest_df):
                elapsed = time.time() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                rem = (len(manifest_df) - processed) / speed if speed > 0 else 0
                print(f"[{name}] [{processed}/{len(manifest_df)}] ({processed/len(manifest_df)*100:.1f}%) | {speed:.1f} slices/s | Elapsed: {elapsed:.1f}s | ETA: {rem:.1f}s")
                
    t1 = time.time()
    runtime = t1 - t0
    print(f"[{name}] Completed in {runtime:.2f} s ({len(manifest_df)/runtime:.1f} slices/s)")
    return scores, runtime

def compute_metrics(y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    sens = recall_score(y_true, preds, pos_label=1, zero_division=0)
    spec = recall_score(y_true, preds, pos_label=0, zero_division=0)
    prec = precision_score(y_true, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    cm = confusion_matrix(y_true, preds).tolist()
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm
    }

def patient_cluster_bootstrap(df, scores, n_resamples=2000, seed=42):
    print(f"Computing patient-cluster bootstrap (N={n_resamples}, seed={seed})...")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    
    unique_patients = df['patient_id'].unique()
    n_patients = len(unique_patients)
    patient_to_indices = df.groupby('patient_id').indices
    
    y_true_all = df['label_idx'].values
    
    boot_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": []
    }
    
    for b in range(n_resamples):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = np.concatenate([patient_to_indices[pid] for pid in sampled_pids])
        
        y_b = y_true_all[sampled_indices]
        s_b = scores[sampled_indices]
        m = compute_metrics(y_b, s_b)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])
            
    ci_results = {}
    for k, vals in boot_metrics.items():
        low = np.percentile(vals, 2.5)
        high = np.percentile(vals, 97.5)
        ci_results[k] = [float(low), float(high)]
        
    t1 = time.time()
    print(f"Bootstrap finished in {t1-t0:.2f} s")
    return ci_results

def compute_histograms(scores, y_true):
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    bin_labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f})" for i in range(len(bins)-1)]
    
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    
    pos_counts, _ = np.histogram(pos_scores, bins=bins)
    neg_counts, _ = np.histogram(neg_scores, bins=bins)
    
    return {
        "tumor": {
            "mean": float(np.mean(pos_scores)),
            "median": float(np.median(pos_scores)),
            "histogram": dict(zip(bin_labels, [int(c) for c in pos_counts]))
        },
        "no_tumor": {
            "mean": float(np.mean(neg_scores)),
            "median": float(np.median(neg_scores)),
            "histogram": dict(zip(bin_labels, [int(c) for c in neg_counts]))
        }
    }

if __name__ == '__main__':
    manifest_path = 'D:/datasets/prepared/brats/manifest.csv'
    df_manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df_manifest)} slices across {df_manifest['patient_id'].nunique()} patients.")
    
    ckpt_path = 'BrainTumorAI/models/tumor_classifier_reproducible.pth'
    device = torch.device('cpu')
    raw_model, meta = load_checkpoint(ckpt_path, map_location=device)
    raw_model.eval()
    
    print("Optimizing model with torch.jit for CPU inference...")
    script_model = torch.jit.script(raw_model)
    opt_model = torch.jit.optimize_for_inference(script_model)
    opt_model.eval()
    
    transform = make_inference_transform()
    
    cohorts = [
        ("brats_oriented", "D:/datasets/prepared/brats_oriented"),
        ("brats_old", "D:/datasets/prepared/brats"),
        ("brats_oriented_flipLR", "D:/datasets/prepared/brats_oriented_flipLR"),
    ]
    
    all_scores = {}
    runtimes = {}
    
    for name, root_dir in cohorts:
        scores, rt = run_cohort_inference(name, root_dir, df_manifest, opt_model, device, transform, batch_size=128)
        all_scores[name] = scores
        runtimes[name] = rt
        
        # Save per-slice predictions as .csv.gz
        pred_df = pd.DataFrame({
            "patient_id": df_manifest['patient_id'],
            "slice_id": df_manifest['slice_id'],
            "label": df_manifest['label_idx'],
            "score": scores
        })
        out_csv_gz = f"paper_evidence/{name}_predictions.csv.gz"
        pred_df.to_csv(out_csv_gz, index=False, compression="gzip")
        print(f"Saved per-slice predictions to: {out_csv_gz}")
        
    y_true = df_manifest['label_idx'].values
    
    final_report = {
        "environment": {
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 8,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "cohorts": {}
    }
    
    for name, _ in cohorts:
        scores = all_scores[name]
        metrics = compute_metrics(y_true, scores)
        cis = patient_cluster_bootstrap(df_manifest, scores, n_resamples=2000, seed=42)
        hists = compute_histograms(scores, y_true)
        
        final_report["cohorts"][name] = {
            "metrics": metrics,
            "bootstrap_ci_95": cis,
            "histograms": hists,
            "runtime_seconds": runtimes[name]
        }
        
    # Paired comparison: primary (brats_oriented) vs old (brats_old)
    s_primary = all_scores["brats_oriented"]
    s_old = all_scores["brats_old"]
    
    p_primary = (s_primary >= 0.5).astype(int)
    p_old = (s_old >= 0.5).astype(int)
    
    flips_old_tumor_to_notumor = int(np.sum((p_old == 1) & (p_primary == 0)))
    flips_old_notumor_to_tumor = int(np.sum((p_old == 0) & (p_primary == 1)))
    total_unchanged = int(np.sum(p_old == p_primary))
    total_changed = int(np.sum(p_old != p_primary))
    
    score_diff = s_primary - s_old
    
    final_report["paired_comparison_primary_vs_old"] = {
        "total_slices": len(y_true),
        "total_changed_predictions": total_changed,
        "total_unchanged_predictions": total_unchanged,
        "flips_old_tumor_to_new_notumor": flips_old_tumor_to_notumor,
        "flips_old_notumor_to_new_tumor": flips_old_notumor_to_tumor,
        "mean_score_difference": float(np.mean(score_diff)),
        "median_score_difference": float(np.median(score_diff)),
        "max_abs_score_difference": float(np.max(np.abs(score_diff)))
    }
    
    out_json = "paper_evidence/brats_orientation_comparison_report.json"
    with open(out_json, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n==========================================")
    print(f"Saved complete comparison report to: {out_json}")
    print(f"==========================================")
