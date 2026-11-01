"""
Full 3-cohort BraTS inference on GPU (RTX 3050 Laptop GPU) with PyTorch 2.7.1+cu118.
Cohorts in order:
1. brats_oriented (PRIMARY)
2. brats_old
3. brats_oriented_flipLR

Prints progress every 1,000 slices.
Saves per-slice predictions as .csv.gz.
Computes patient-cluster bootstrap (2,000 resamples, seed 42) for 95% CIs.
Computes paired comparison between brats_oriented and brats_old.
Saves final JSON report to paper_evidence/brats_gpu_comparison_report.json.
"""
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
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform
from paper_evidence.brats_dataset import BraTSPNGDataset

def run_cohort_inference(name, root_dir, manifest_df, model, device, transform, batch_size=128):
    print(f"\n==========================================", flush=True)
    print(f"Starting GPU inference for: {name}", flush=True)
    print(f"Root: {root_dir} | Total slices: {len(manifest_df)}", flush=True)
    print(f"==========================================", flush=True)
    
    dataset = BraTSPNGDataset(root_dir, manifest_df, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    scores = np.zeros(len(manifest_df), dtype=np.float32)
    t0 = time.time()
    processed = 0
    last_printed = 0
    total = len(manifest_df)
    
    with torch.no_grad():
        for batch_x, batch_idxs in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            tumor_probs = probs[:, 1].cpu().numpy()
            scores[batch_idxs.numpy()] = tumor_probs
            processed += len(batch_idxs)
            
            if processed - last_printed >= 1000 or processed == total:
                elapsed = time.time() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                rem = (total - processed) / speed if speed > 0 else 0
                print(f"[{name}] [{processed}/{total}] ({processed/total*100:.1f}%) | {speed:.1f} slices/s | Elapsed: {elapsed:.1f}s | ETA: {rem:.1f}s", flush=True)
                last_printed = processed
                
    runtime = time.time() - t0
    throughput = len(manifest_df) / runtime
    print(f"[{name}] Completed in {runtime:.2f} s ({throughput:.1f} slices/s)", flush=True)
    return scores, runtime, throughput

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
    print(f"Computing patient-cluster bootstrap (N={n_resamples}, seed={seed})...", flush=True)
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
    print(f"Bootstrap finished in {t1-t0:.2f} s", flush=True)
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
    print(f"Loaded BraTS manifest with {len(df_manifest)} slices across {df_manifest['patient_id'].nunique()} patients.", flush=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"Device Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/(1024**2):.1f} MB", flush=True)
    
    ckpt_path = 'BrainTumorAI/models/tumor_classifier_reproducible.pth'
    model, meta = load_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    transform = make_inference_transform()
    
    cohorts = [
        ("brats_oriented", "D:/datasets/prepared/brats_oriented"),
        ("brats_old", "D:/datasets/prepared/brats"),
        ("brats_oriented_flipLR", "D:/datasets/prepared/brats_oriented_flipLR"),
    ]
    
    all_scores = {}
    runtimes = {}
    throughputs = {}
    
    for name, root_dir in cohorts:
        scores, rt, tp = run_cohort_inference(name, root_dir, df_manifest, model, device, transform, batch_size=128)
        all_scores[name] = scores
        runtimes[name] = rt
        throughputs[name] = tp
        
        # Save per-slice predictions as .csv.gz
        pred_df = pd.DataFrame({
            "patient_id": df_manifest['patient_id'],
            "slice_id": df_manifest['slice_id'],
            "label": df_manifest['label_idx'],
            "score": scores
        })
        out_csv_gz = f"paper_evidence/{name}_predictions.csv.gz"
        pred_df.to_csv(out_csv_gz, index=False, compression="gzip")
        print(f"Saved per-slice predictions to: {out_csv_gz}", flush=True)
        
    y_true = df_manifest['label_idx'].values
    
    final_report = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": "0.22.1+cu118",
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
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
            "runtime_seconds": runtimes[name],
            "throughput_imgs_per_s": throughputs[name]
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
    
    out_json = "paper_evidence/brats_gpu_comparison_report.json"
    with open(out_json, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n==========================================", flush=True)
    print(f"Saved complete comparison report to: {out_json}", flush=True)
    print(f"==========================================", flush=True)
