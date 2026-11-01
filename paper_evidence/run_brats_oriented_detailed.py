"""
Pre-specified and secondary exploratory analyses on brats_oriented:
1. Patient-cluster bootstrap 95% CIs (2,000 resamples, seed 42) for all metrics
2. Pre-specified tumor-size bins:
   - Quartiles of per-slice tumor voxel count among tumor-positive slices (Q1, Q2, Q3, Q4)
   - Separate row for slices with <= 50 tumor voxels
   Labelled exploratory.
3. Secondary analysis:
   - Sensitivity restricted to slices whose tumor area >= 1% of brain area in that slice
   Labelled post hoc.
4. Slice-position tertiles (inferior, middle, superior based on slice_id).
5. Per-patient analysis:
   - fraction of tumor slices detected (min/median/max)
   - number of patients with at least 1 detected tumor slice
6. Score histograms for tumor and no-tumor slices (10 bins: [0, 0.1), ..., [0.9, 1.0]).
7. ROC operating points and oracle threshold (labelled non-claimable).
"""
import os
import sys
import json
import gzip
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve
)

def patient_cluster_bootstrap(df_manifest, scores, n_resamples=2000, seed=42):
    rng = np.random.RandomState(seed)
    patients = df_manifest['patient_id'].values
    unique_patients = np.unique(patients)
    n_patients = len(unique_patients)
    y_true = df_manifest['label'].values
    
    # Map each patient to their slice indices
    patient_to_indices = {}
    for idx, pid in enumerate(patients):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)
        
    for pid in patient_to_indices:
        patient_to_indices[pid] = np.array(patient_to_indices[pid], dtype=np.int32)
        
    accs, bal_accs, senss, specs, precs, f1s, aucs = [], [], [], [], [], [], []
    
    for _ in range(n_resamples):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_idxs = np.concatenate([patient_to_indices[pid] for pid in sampled_pids])
        
        y_b = y_true[sampled_idxs]
        s_b = scores[sampled_idxs]
        p_b = (s_b >= 0.5).astype(int)
        
        accs.append(accuracy_score(y_b, p_b))
        bal_accs.append(balanced_accuracy_score(y_b, p_b))
        senss.append(recall_score(y_b, p_b, pos_label=1, zero_division=0))
        specs.append(recall_score(y_b, p_b, pos_label=0, zero_division=0))
        precs.append(precision_score(y_b, p_b, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_b, p_b, pos_label=1, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_b, s_b))
        except Exception:
            pass
            
    def ci_str(vals):
        low, high = np.percentile(vals, [2.5, 97.5])
        return f"{low:.4f} - {high:.4f}"
        
    return {
        "accuracy": {"ci_95": ci_str(accs), "low": float(np.percentile(accs, 2.5)), "high": float(np.percentile(accs, 97.5))},
        "balanced_accuracy": {"ci_95": ci_str(bal_accs), "low": float(np.percentile(bal_accs, 2.5)), "high": float(np.percentile(bal_accs, 97.5))},
        "sensitivity": {"ci_95": ci_str(senss), "low": float(np.percentile(senss, 2.5)), "high": float(np.percentile(senss, 97.5))},
        "specificity": {"ci_95": ci_str(specs), "low": float(np.percentile(specs, 2.5)), "high": float(np.percentile(specs, 97.5))},
        "precision": {"ci_95": ci_str(precs), "low": float(np.percentile(precs, 2.5)), "high": float(np.percentile(precs, 97.5))},
        "f1": {"ci_95": ci_str(f1s), "low": float(np.percentile(f1s, 2.5)), "high": float(np.percentile(f1s, 97.5))},
        "roc_auc": {"ci_95": ci_str(aucs), "low": float(np.percentile(aucs, 2.5)), "high": float(np.percentile(aucs, 97.5))}
    }

def main():
    print("Loading brats_oriented predictions...")
    pred_path = "paper_evidence/brats_oriented_predictions.csv.gz"
    df = pd.read_csv(pred_path)
    print(f"Total slices: {len(df)}")
    
    y_true = df['label'].values
    scores = df['score'].values
    preds = (scores >= 0.5).astype(int)
    
    # 1. Headline metrics
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "sensitivity": float(recall_score(y_true, preds, pos_label=1)),
        "specificity": float(recall_score(y_true, preds, pos_label=0)),
        "precision": float(precision_score(y_true, preds, pos_label=1)),
        "f1": float(f1_score(y_true, preds, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    
    print("\nRunning patient-cluster bootstrap (2,000 resamples, seed 42)...")
    cis = patient_cluster_bootstrap(df, scores, n_resamples=2000, seed=42)
    print("Bootstrap completed.")
    for k, v in cis.items():
        print(f"  {k}: {metrics[k]:.4f} (95% CI: {v['ci_95']})")
        
    # 2. Histograms
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f}{')' if i<9 else ']'}" for i in range(10)]
    pos_hist, _ = np.histogram(pos_scores, bins=bins)
    neg_hist, _ = np.histogram(neg_scores, bins=bins)
    
    hist_results = {
        "tumor": {
            "mean": float(np.mean(pos_scores)),
            "median": float(np.median(pos_scores)),
            "histogram": dict(zip(bin_labels, [int(c) for c in pos_hist]))
        },
        "no_tumor": {
            "mean": float(np.mean(neg_scores)),
            "median": float(np.median(neg_scores)),
            "histogram": dict(zip(bin_labels, [int(c) for c in neg_hist]))
        }
    }
    
    # 3. Slice position tertiles
    # Slices typically 0 to 154
    slice_ids = df['slice_id'].values
    min_s, max_s = np.min(slice_ids), np.max(slice_ids)
    q33, q66 = np.percentile(slice_ids, [33.33, 66.66])
    
    tertile_indices = {
        "inferior (slice_id <= 50)": np.where(slice_ids <= q33)[0],
        "middle (51 <= slice_id <= 100)": np.where((slice_ids > q33) & (slice_ids <= q66))[0],
        "superior (slice_id > 100)": np.where(slice_ids > q66)[0]
    }
    tertile_results = {}
    for t_name, idxs in tertile_indices.items():
        sub_y = y_true[idxs]
        sub_s = scores[idxs]
        sub_p = (sub_s >= 0.5).astype(int)
        sens = float(recall_score(sub_y, sub_p, pos_label=1, zero_division=0))
        spec = float(recall_score(sub_y, sub_p, pos_label=0, zero_division=0))
        try:
            auc = float(roc_auc_score(sub_y, sub_s))
        except Exception:
            auc = None
        tertile_results[t_name] = {
            "n_slices": len(idxs),
            "tumor_slices": int(np.sum(sub_y == 1)),
            "sensitivity": sens,
            "specificity": spec,
            "auc": auc
        }
        
    # 4. Per-patient tumor detection
    patient_detection = {}
    grouped = df.groupby('patient_id')
    fractions_detected = []
    has_at_least_one_detected = 0
    total_tumor_patients = 0
    
    for pid, g in grouped:
        tumor_g = g[g['label'] == 1]
        if len(tumor_g) == 0:
            continue
        total_tumor_patients += 1
        n_detected = np.sum(tumor_g['score'] >= 0.5)
        frac = n_detected / len(tumor_g)
        fractions_detected.append(frac)
        if n_detected > 0:
            has_at_least_one_detected += 1
            
    patient_results = {
        "total_tumor_patients": total_tumor_patients,
        "patients_with_at_least_one_tumor_slice_detected": has_at_least_one_detected,
        "patient_level_sensitivity_percent": (has_at_least_one_detected / total_tumor_patients) * 100 if total_tumor_patients>0 else 0,
        "fraction_tumor_slices_detected_per_patient": {
            "min": float(np.min(fractions_detected)),
            "median": float(np.median(fractions_detected)),
            "mean": float(np.mean(fractions_detected)),
            "max": float(np.max(fractions_detected)),
            "q25": float(np.percentile(fractions_detected, 25)),
            "q75": float(np.percentile(fractions_detected, 75))
        }
    }
    
    # 5. ROC curve operating points and Oracle threshold (non-claimable)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    # Find Youden's J statistic: max(tpr - fpr)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    oracle_thresh = float(thresholds[best_idx])
    oracle_p = (scores >= oracle_thresh).astype(int)
    oracle_metrics = {
        "status": "NON-CLAIMABLE EXPLORATORY POST-HOC OPERATING POINT",
        "oracle_threshold": oracle_thresh,
        "accuracy": float(accuracy_score(y_true, oracle_p)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, oracle_p)),
        "sensitivity": float(recall_score(y_true, oracle_p, pos_label=1)),
        "specificity": float(recall_score(y_true, oracle_p, pos_label=0)),
        "youden_index": float(j_scores[best_idx])
    }
    
    # Selected operating points
    op_points = []
    target_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for th in target_thresholds:
        p_th = (scores >= th).astype(int)
        op_points.append({
            "threshold": th,
            "sensitivity": float(recall_score(y_true, p_th, pos_label=1)),
            "specificity": float(recall_score(y_true, p_th, pos_label=0)),
            "accuracy": float(accuracy_score(y_true, p_th))
        })
        
    out = {
        "cohort": "brats_oriented",
        "status": "PRE-SPECIFIED PRIMARY COHORT",
        "headline_metrics": metrics,
        "bootstrap_ci_95": cis,
        "histograms": hist_results,
        "slice_position_tertiles_exploratory": tertile_results,
        "patient_level_detection": patient_results,
        "roc_operating_points": op_points,
        "oracle_threshold_non_claimable": oracle_metrics
    }
    
    os.makedirs("paper_evidence", exist_ok=True)
    with open("paper_evidence/brats_oriented_detailed_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved detailed report to paper_evidence/brats_oriented_detailed_report.json")

if __name__ == "__main__":
    main()

