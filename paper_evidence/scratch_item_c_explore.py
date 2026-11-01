"""
Items C, D, E — Cheng overlap detail, clean-test CIs, clean-vs-removed per-checkpoint.
All output to paper_evidence/
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ============================================================
# ITEM C: Cheng overlap detail
# ============================================================
print("=" * 60)
print("ITEM C: Cheng Overlap Detail")
print("=" * 60)

with open('BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json') as f:
    oa = json.load(f)

hsm = oa['matches']['high_similarity_matches']
print(f"Total high-similarity match edges: {len(hsm)}")

# Load split manifest to get labels
split_manifest = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
print(f"Split manifest: {len(split_manifest)} entries")
print(f"Split manifest columns: {list(split_manifest.columns)}")

# Build lookup: source_file -> (split, label)
file_to_info = {}
for _, row in split_manifest.iterrows():
    file_to_info[row['source_file']] = {
        'split': row['split'],
        'label': row['label']
    }

# Extract per-match info
match_records = []
for m in hsm:
    cheng_slice = m['cheng_slice']
    patient_id = m['patient_id']
    local_file = m['local_source_file']
    local_split = m['local_split']
    phash_dist = m['phash_hamming_distance']
    
    # Get label from split manifest
    info = file_to_info.get(local_file, {})
    label = info.get('label', 'UNKNOWN')
    
    match_records.append({
        'cheng_slice': cheng_slice,
        'patient_id': patient_id,
        'local_file': local_file,
        'local_split': local_split,
        'label': label,
        'phash_dist': phash_dist
    })

df_matches = pd.DataFrame(match_records)
print(f"\nMatched edges: {len(df_matches)}")

# Unique Cheng patients
unique_patients = df_matches['patient_id'].unique()
print(f"Unique Cheng patients matched: {len(unique_patients)}")

# Per patient: which splits have matches?
patient_splits = df_matches.groupby('patient_id')['local_split'].apply(set).to_dict()

both_train_test = 0
both_train_val = 0
all_three = 0
train_only = 0
test_only = 0
val_only = 0
train_and_test_patients = []
train_and_val_patients = []
all_three_patients = []

for pid, splits in patient_splits.items():
    has_train = 'train' in splits
    has_test = 'test' in splits
    has_val = 'validation' in splits
    
    if has_train and has_test:
        both_train_test += 1
        train_and_test_patients.append(pid)
    if has_train and has_val:
        both_train_val += 1
        train_and_val_patients.append(pid)
    if has_train and has_test and has_val:
        all_three += 1
        all_three_patients.append(pid)
    if has_train and not has_test and not has_val:
        train_only += 1
    if has_test and not has_train and not has_val:
        test_only += 1
    if has_val and not has_train and not has_test:
        val_only += 1

print(f"\nPatients with matches in both train AND test: {both_train_test}")
print(f"Patients with matches in both train AND validation: {both_train_val}")
print(f"Patients with matches in all three splits: {all_three}")
print(f"Patients with matches ONLY in train: {train_only}")
print(f"Patients with matches ONLY in test: {test_only}")
print(f"Patients with matches ONLY in validation: {val_only}")

# For the 141 matched test images
test_matches = df_matches[df_matches['local_split'] == 'test']
print(f"\nTest-matched images: {len(test_matches)}")
test_patients = test_matches['patient_id'].unique()
print(f"Unique patients with test matches: {len(test_patients)}")

# How many test-matched images belong to patients ALSO matched in train
train_matched_patients = set()
for pid, splits in patient_splits.items():
    if 'train' in splits:
        train_matched_patients.add(pid)

test_images_also_in_train = test_matches[test_matches['patient_id'].isin(train_matched_patients)]
print(f"Test images belonging to patients also matched in train: {len(test_images_also_in_train)}")

# Label distribution of ALL matched images
print("\nLabel distribution of all matched images:")
label_dist = df_matches['label'].value_counts().to_dict()
print(label_dist)

# Split distribution
print("\nSplit distribution of all matched images:")
split_dist = df_matches['local_split'].value_counts().to_dict()
print(split_dist)

# Per-split label distribution
print("\nPer-split label distribution:")
for split in ['train', 'validation', 'test']:
    subset = df_matches[df_matches['local_split'] == split]
    if len(subset) > 0:
        ld = subset['label'].value_counts().to_dict()
        print(f"  {split}: {len(subset)} images, labels: {ld}")

item_c_report = {
    "source_file": "BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json",
    "total_match_edges": len(hsm),
    "unique_cheng_patients_matched": len(unique_patients),
    "patients_matched_in_both_train_and_test": both_train_test,
    "patients_matched_in_both_train_and_validation": both_train_val,
    "patients_matched_in_all_three_splits": all_three,
    "patients_matched_only_in_train": train_only,
    "patients_matched_only_in_test": test_only,
    "patients_matched_only_in_validation": val_only,
    "test_matched_images_total": len(test_matches),
    "test_matched_unique_patients": len(test_patients),
    "test_images_from_patients_also_in_train": len(test_images_also_in_train),
    "label_distribution_all_matches": label_dist,
    "split_distribution": split_dist,
    "per_split_label_distribution": {
        split: df_matches[df_matches['local_split'] == split]['label'].value_counts().to_dict()
        for split in ['train', 'validation', 'test']
    },
    "train_and_test_patient_ids": sorted(train_and_test_patients),
    "train_and_val_patient_ids": sorted(train_and_val_patients),
    "all_three_patient_ids": sorted(all_three_patients)
}

# ============================================================
# ITEM D: Clean-test bootstrap CIs (seed 0, 2000 resamples)
# ============================================================
print("\n" + "=" * 60)
print("ITEM D: Clean-Test Confidence Intervals")
print("=" * 60)

# Load phase3 clean test rescore
with open('paper_evidence/phase3_clean_test_rescore.json') as f:
    p3 = json.load(f)

# Identify clean vs removed images
# The test set minus overlap matches = clean
# Need: test split files from split_manifest, overlap matched test files
test_split = split_manifest[split_manifest['split'] == 'test'].copy()
print(f"Total test images: {len(test_split)}")

# Matched test files
matched_test_files = set(test_matches['local_file'].tolist())
test_split['is_removed'] = test_split['source_file'].isin(matched_test_files)
clean_test = test_split[~test_split['is_removed']]
removed_test = test_split[test_split['is_removed']]
print(f"Clean test: {len(clean_test)}, Removed: {len(removed_test)}")

# We need the predictions for these images
# Load the primary stored predictions
# Check which files have predictions
results_dir = 'BrainTumorAI/results'
print(f"\nLooking for test predictions in {results_dir}...")

# Find test prediction files
import glob
pred_files = glob.glob(os.path.join(results_dir, '*.csv'))
print(f"CSV files in results: {[os.path.basename(f) for f in pred_files]}")

# Also check experiments_results
exp_dirs = glob.glob('experiments_results/phase1b/*')
print(f"Phase1b dirs: {[os.path.basename(d) for d in exp_dirs]}")

# Load the primary ResNet-18 predictions (seed 42)
# From phase2 we know the primary stored preds are in results/
primary_preds_path = os.path.join(results_dir, 'test_predictions.csv')
if os.path.exists(primary_preds_path):
    preds_resnet = pd.read_csv(primary_preds_path)
    print(f"\nPrimary ResNet-18 preds: {len(preds_resnet)} rows")
    print(f"Columns: {list(preds_resnet.columns)}")
else:
    print(f"NOT FOUND: {primary_preds_path}")
    # Check alternative locations
    for f in glob.glob(os.path.join(results_dir, '**/*.csv'), recursive=True):
        print(f"  Found: {f}")


