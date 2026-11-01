"""
Items C, D, E — Complete script.
C: Cheng overlap detail (from overlap_audit.json)
D: Clean-test bootstrap CIs (465 images, seed 0, 2000 resamples)  
E: Clean-vs-removed per-checkpoint (13 checkpoints)
All output to paper_evidence/
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.abspath('.'))
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

torch.set_num_threads(8)

# ============================================================
# ITEM C: Cheng overlap detail
# ============================================================
print("=" * 60)
print("ITEM C: Cheng Overlap Detail")
print("=" * 60)

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    oa = json.load(f)

hsm = oa['matches']['high_similarity_matches']
split_manifest = pd.read_csv('BrainTumorAI/results/split_manifest.csv')

file_to_info = {}
for _, row in split_manifest.iterrows():
    file_to_info[row['source_file']] = {
        'split': row['split'],
        'label': row['label'],
        'label_idx': row['label_idx'],
        'duplicate_group': row.get('duplicate_group', None)
    }

match_records = []
for m in hsm:
    info = file_to_info.get(m['local_source_file'], {})
    match_records.append({
        'cheng_slice': m['cheng_slice'],
        'patient_id': m['patient_id'],
        'local_file': m['local_source_file'],
        'local_split': m['local_split'],
        'label': info.get('label', 'UNKNOWN'),
        'label_idx': info.get('label_idx', -1),
        'duplicate_group': info.get('duplicate_group', None),
        'phash_dist': m['phash_hamming_distance']
    })

df_matches = pd.DataFrame(match_records)

patient_splits = df_matches.groupby('patient_id')['local_split'].apply(set).to_dict()

both_train_test = sum(1 for s in patient_splits.values() if 'train' in s and 'test' in s)
both_train_val = sum(1 for s in patient_splits.values() if 'train' in s and 'validation' in s)
all_three = sum(1 for s in patient_splits.values() if 'train' in s and 'test' in s and 'validation' in s)
train_only = sum(1 for s in patient_splits.values() if s == {'train'})
test_only = sum(1 for s in patient_splits.values() if s == {'test'})
val_only = sum(1 for s in patient_splits.values() if s == {'validation'})

test_matches = df_matches[df_matches['local_split'] == 'test']
train_matched_pids = {pid for pid, s in patient_splits.items() if 'train' in s}
test_images_also_train = test_matches[test_matches['patient_id'].isin(train_matched_pids)]

item_c = {
    "source": "BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json",
    "total_match_edges": len(hsm),
    "unique_cheng_patients_matched": len(patient_splits),
    "patient_split_breakdown": {
        "both_train_and_test": both_train_test,
        "both_train_and_validation": both_train_val,
        "all_three_splits": all_three,
        "train_only": train_only,
        "test_only": test_only,
        "validation_only": val_only
    },
    "test_matched_images": {
        "total": len(test_matches),
        "unique_patients": len(test_matches['patient_id'].unique()),
        "belonging_to_patients_also_in_train": len(test_images_also_train)
    },
    "label_distribution_all": df_matches['label'].value_counts().to_dict(),
    "split_distribution": df_matches['local_split'].value_counts().to_dict(),
    "per_split_labels": {
        split: df_matches[df_matches['local_split'] == split]['label'].value_counts().to_dict()
        for split in ['train', 'validation', 'test']
    }
}

print(json.dumps(item_c, indent=2))

# ============================================================
# ITEM D: Clean-test bootstrap CIs
# duplicate-group stratified bootstrap, seed 0, 2000 resamples
# For ResNet-18 and SmallCNN
# ============================================================
print("\n" + "=" * 60)
print("ITEM D: Clean-Test Bootstrap CIs")
print("=" * 60)

# Identify clean test images (606 total test - 141 matched = 465 clean)
test_files = set(split_manifest[split_manifest['split'] == 'test']['source_file'])
matched_test_files = set(test_matches['local_file'])
clean_test_files = test_files - matched_test_files
print(f"Test: {len(test_files)} | Matched: {len(matched_test_files)} | Clean: {len(clean_test_files)}")

# Load primary test predictions
primary_preds = pd.read_csv('BrainTumorAI/results/evaluation/test_predictions.csv')

def compute_metrics_from_arrays(y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "sensitivity": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, preds, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float('nan'),
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float('nan'),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "n": int(len(y_true))
    }

def duplicate_group_stratified_bootstrap(df, score_col, n_resamples=2000, seed=0):
    """Bootstrap resampling by duplicate_group to maintain within-group correlation."""
    rng = np.random.default_rng(seed)
    
    # Get unique duplicate groups
    groups = df['duplicate_group'].unique()
    n_groups = len(groups)
    group_to_indices = df.groupby('duplicate_group').indices
    
    y_all = df['true_label_idx'].values
    s_all = df[score_col].values
    
    boot_metrics = {k: [] for k in [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "precision", "f1", "roc_auc", "pr_auc"
    ]}
    
    for _ in range(n_resamples):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        y_b = y_all[idx]
        s_b = s_all[idx]
        
        # Skip degenerate samples
        if len(np.unique(y_b)) < 2:
            continue
            
        m = compute_metrics_from_arrays(y_b, s_b)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])
    
    ci = {}
    for k, vals in boot_metrics.items():
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            ci[k] = {
                "point": float(np.mean(vals)),
                "ci_2.5": float(np.percentile(vals, 2.5)),
                "ci_97.5": float(np.percentile(vals, 97.5)),
                "valid_resamples": len(vals)
            }
    return ci

item_d = {"seed": 0, "n_resamples": 2000, "method": "duplicate_group_stratified_bootstrap", "models": {}}

for model_name, model_key in [("resnet18", "resnet18"), ("small_cnn", "small_cnn_small_cnn_best")]:
    model_preds = primary_preds[primary_preds['model'] == model_key].copy()
    # Filter to clean test only
    clean_preds = model_preds[model_preds['source_file'].isin(clean_test_files)].copy()
    print(f"\n{model_name}: {len(clean_preds)} clean test predictions")
    
    # Check duplicate_group column
    if 'duplicate_group' not in clean_preds.columns:
        # Merge from split_manifest
        sg = split_manifest[['source_file', 'duplicate_group']].drop_duplicates()
        clean_preds = clean_preds.merge(sg, on='source_file', how='left')
    
    n_groups = clean_preds['duplicate_group'].nunique()
    print(f"  Duplicate groups: {n_groups}")
    
    # Point estimates
    point = compute_metrics_from_arrays(
        clean_preds['true_label_idx'].values,
        clean_preds['tumor_score'].values
    )
    print(f"  Point: Acc={point['accuracy']:.4f} Sens={point['sensitivity']:.4f} Spec={point['specificity']:.4f} AUC={point['roc_auc']:.4f}")
    
    # Bootstrap CIs
    t0 = time.time()
    cis = duplicate_group_stratified_bootstrap(clean_preds, 'tumor_score', n_resamples=2000, seed=0)
    t1 = time.time()
    print(f"  Bootstrap: {t1-t0:.2f}s")
    
    item_d["models"][model_name] = {
        "n_clean_test": len(clean_preds),
        "n_duplicate_groups": n_groups,
        "point_estimates": point,
        "bootstrap_ci_95": cis
    }

print(json.dumps(item_d, indent=2))

# ============================================================
# ITEM E: Clean-vs-removed per-checkpoint
# 13 checkpoints: resnet18 (seeds 42-46), small_cnn (seeds 42-46), densenet121 (seeds 42-44)
# ============================================================
print("\n" + "=" * 60)
print("ITEM E: Clean vs Removed per Checkpoint")
print("=" * 60)

checkpoints = []
for model in ['resnet18', 'small_cnn', 'densenet121']:
    if model == 'densenet121':
        seeds = [42, 43, 44]
    else:
        seeds = [42, 43, 44, 45, 46]
    for s in seeds:
        checkpoints.append((model, s))

print(f"Total checkpoints: {len(checkpoints)}")

item_e = {"checkpoints": []}

for model, seed in checkpoints:
    seed_dir = f"experiments_results/phase1b/{model}_seed{seed}"
    pred_file = os.path.join(seed_dir, 'test_predictions.csv')
    
    if not os.path.exists(pred_file):
        print(f"  MISSING: {pred_file}")
        item_e["checkpoints"].append({
            "model": model,
            "seed": seed,
            "status": "MISSING",
            "pred_file": pred_file
        })
        continue
    
    df_pred = pd.read_csv(pred_file)
    
    # Separate clean and removed
    clean = df_pred[df_pred['source_file'].isin(clean_test_files)]
    removed = df_pred[df_pred['source_file'].isin(matched_test_files)]
    
    if len(clean) == 0:
        print(f"  {model}_seed{seed}: 0 clean predictions, skipping")
        continue
        
    clean_metrics = compute_metrics_from_arrays(
        clean['true_label_idx'].values,
        clean['tumor_score'].values
    )
    
    removed_metrics = None
    if len(removed) > 0:
        removed_metrics = compute_metrics_from_arrays(
            removed['true_label_idx'].values,
            removed['tumor_score'].values
        )
    
    entry = {
        "model": model,
        "seed": seed,
        "pred_file": pred_file,
        "clean": {
            "n": len(clean),
            "accuracy": clean_metrics["accuracy"],
            "sensitivity": clean_metrics["sensitivity"],
            "specificity": clean_metrics["specificity"],
            "roc_auc": clean_metrics["roc_auc"],
            "confusion_matrix": clean_metrics["confusion_matrix"]
        },
        "removed": None
    }
    
    if removed_metrics:
        entry["removed"] = {
            "n": len(removed),
            "accuracy": removed_metrics["accuracy"],
            "sensitivity": removed_metrics["sensitivity"],
            "specificity": removed_metrics["specificity"],
            "roc_auc": removed_metrics["roc_auc"],
            "confusion_matrix": removed_metrics["confusion_matrix"]
        }
    
    item_e["checkpoints"].append(entry)
    
    rm_acc = removed_metrics["accuracy"] if removed_metrics else "N/A"
    rm_sens = removed_metrics["sensitivity"] if removed_metrics else "N/A"
    print(f"  {model}_seed{seed}: Clean({len(clean)}) Acc={clean_metrics['accuracy']:.4f} Sens={clean_metrics['sensitivity']:.4f} | Removed({len(removed)}) Acc={rm_acc} Sens={rm_sens}")

# ============================================================
# Save combined report
# ============================================================
combined = {
    "environment": {
        "torch": torch.__version__,
        "device": "cpu",
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "item_c_cheng_overlap_detail": item_c,
    "item_d_clean_test_cis": item_d,
    "item_e_clean_vs_removed": item_e
}

out_path = "paper_evidence/items_CDE_report.json"
with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nSaved combined report to: {out_path}")
print("DONE")

