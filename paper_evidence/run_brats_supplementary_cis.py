"""
Supplement BraTS analyses with:
1. Paired patient-cluster bootstrap CIs (2,000 resamples, seed 42) for differences:
   - brats_oriented minus brats_old (AUC, accuracy, sensitivity, specificity)
   - brats_oriented minus brats_oriented_flipLR (AUC, accuracy, sensitivity, specificity)
2. Tumor-size bins:
   - n_slices, n_patients
   - sensitivity with 95% patient-cluster bootstrap CI (2,000 resamples, seed 42)
   - AUC evaluated against all negative slices (87,653 non-tumor slices)
   - <=50-voxel bin, Q1, Q2, Q3, Q4, and >=1% brain-area post-hoc bin
3. Slice-position tertiles:
   - sensitivity with 95% patient-cluster bootstrap CI
   - specificity with 95% patient-cluster bootstrap CI
   - AUC with 95% patient-cluster bootstrap CI
4. Per-patient detection:
   - min, Q25, median, mean, Q75, max fraction of tumor slices detected
   - number of patients with >= 1 detected tumor slice
   - number of patients with ZERO detected tumor slices
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

def main():
    t0 = time.time()
    print("Loading predictions and segmentation counts...", flush=True)
    df_ori = pd.read_csv("paper_evidence/brats_oriented_predictions.csv.gz")
    df_old = pd.read_csv("paper_evidence/brats_old_predictions.csv.gz")
    df_flp = pd.read_csv("paper_evidence/brats_oriented_flipLR_predictions.csv.gz")
    df_seg = pd.read_csv("paper_evidence/brats_seg_voxel_counts.csv.gz")

    df = df_ori.merge(df_seg, on=["patient_id", "slice_id"], how="left")
    df["score_ori"] = df_ori["score"].values
    df["score_old"] = df_old["score"].values
    df["score_flp"] = df_flp["score"].values
    df["pred_ori"] = (df["score_ori"] >= 0.5).astype(int)
    df["pred_old"] = (df["score_old"] >= 0.5).astype(int)
    df["pred_flp"] = (df["score_flp"] >= 0.5).astype(int)

    patients = df["patient_id"].values
    unique_patients = np.unique(patients)
    n_patients = len(unique_patients)
    y_true = df["label"].values

    # Map patient -> slice indices
    p_to_idx = {}
    for idx, pid in enumerate(patients):
        if pid not in p_to_idx:
            p_to_idx[pid] = []
        p_to_idx[pid].append(idx)
    for pid in p_to_idx:
        p_to_idx[pid] = np.array(p_to_idx[pid], dtype=np.int32)

    s_ori = df["score_ori"].values
    s_old = df["score_old"].values
    s_flp = df["score_flp"].values
    p_ori = df["pred_ori"].values
    p_old = df["pred_old"].values
    p_flp = df["pred_flp"].values

    # -------------------------------------------------------------------------
    # 1. PAIRED PATIENT-CLUSTER BOOTSTRAP FOR DIFFERENCES (2,000 resamples, seed 42)
    # -------------------------------------------------------------------------
    print("Running paired patient-cluster bootstrap (2,000 resamples, seed 42)...", flush=True)
    rng = np.random.RandomState(42)

    diff_ori_old = {"auc": [], "accuracy": [], "sensitivity": [], "specificity": []}
    diff_ori_flp = {"auc": [], "accuracy": [], "sensitivity": [], "specificity": []}

    # Point estimates
    pt_ori = {
        "auc": float(roc_auc_score(y_true, s_ori)),
        "accuracy": float(accuracy_score(y_true, p_ori)),
        "sensitivity": float(recall_score(y_true, p_ori, pos_label=1)),
        "specificity": float(recall_score(y_true, p_ori, pos_label=0)),
    }
    pt_old = {
        "auc": float(roc_auc_score(y_true, s_old)),
        "accuracy": float(accuracy_score(y_true, p_old)),
        "sensitivity": float(recall_score(y_true, p_old, pos_label=1)),
        "specificity": float(recall_score(y_true, p_old, pos_label=0)),
    }
    pt_flp = {
        "auc": float(roc_auc_score(y_true, s_flp)),
        "accuracy": float(accuracy_score(y_true, p_flp)),
        "sensitivity": float(recall_score(y_true, p_flp, pos_label=1)),
        "specificity": float(recall_score(y_true, p_flp, pos_label=0)),
    }

    for i in range(2000):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        idxs = np.concatenate([p_to_idx[pid] for pid in sampled_pids])

        yb = y_true[idxs]
        pos_mask = (yb == 1)
        neg_mask = (yb == 0)

        # Fast sensitivity/specificity/accuracy
        acc_ori = np.mean(p_ori[idxs] == yb)
        acc_old = np.mean(p_old[idxs] == yb)
        acc_flp = np.mean(p_flp[idxs] == yb)

        sens_ori = np.mean(p_ori[idxs][pos_mask] == 1)
        sens_old = np.mean(p_old[idxs][pos_mask] == 1)
        sens_flp = np.mean(p_flp[idxs][pos_mask] == 1)

        spec_ori = np.mean(p_ori[idxs][neg_mask] == 0)
        spec_old = np.mean(p_old[idxs][neg_mask] == 0)
        spec_flp = np.mean(p_flp[idxs][neg_mask] == 0)

        auc_ori = roc_auc_score(yb, s_ori[idxs])
        auc_old = roc_auc_score(yb, s_old[idxs])
        auc_flp = roc_auc_score(yb, s_flp[idxs])

        diff_ori_old["accuracy"].append(acc_ori - acc_old)
        diff_ori_old["sensitivity"].append(sens_ori - sens_old)
        diff_ori_old["specificity"].append(spec_ori - spec_old)
        diff_ori_old["auc"].append(auc_ori - auc_old)

        diff_ori_flp["accuracy"].append(acc_ori - acc_flp)
        diff_ori_flp["sensitivity"].append(sens_ori - sens_flp)
        diff_ori_flp["specificity"].append(spec_ori - spec_flp)
        diff_ori_flp["auc"].append(auc_ori - auc_flp)

        if (i + 1) % 500 == 0:
            print(f"  Paired bootstrap {i+1}/2000 done ({time.time()-t0:.1f}s)", flush=True)

    def summarize_diff(pt_a, pt_b, arr):
        low, high = np.percentile(arr, [2.5, 97.5])
        delta = pt_a - pt_b
        return {
            "point_diff": float(delta),
            "ci_95_low": float(low),
            "ci_95_high": float(high),
            "ci_95_str": f"{delta:+.4f} ({low:+.4f} to {high:+.4f})"
        }

    paired_results = {
        "oriented_minus_old": {
            m: summarize_diff(pt_ori[m], pt_old[m], diff_ori_old[m])
            for m in ["auc", "accuracy", "sensitivity", "specificity"]
        },
        "oriented_minus_flipLR": {
            m: summarize_diff(pt_ori[m], pt_flp[m], diff_ori_flp[m])
            for m in ["auc", "accuracy", "sensitivity", "specificity"]
        }
    }

    # -------------------------------------------------------------------------
    # 2. TUMOR-SIZE BINS WITH SENSITIVITY CIs AND AUC vs ALL NEGATIVES
    # -------------------------------------------------------------------------
    print("Computing tumor-size bin CIs and AUCs...", flush=True)
    tumor_mask = (y_true == 1)
    neg_mask_all = (y_true == 0)
    neg_scores_all = s_ori[neg_mask_all]
    n_neg_all = len(neg_scores_all)

    tumor_voxels = df["tumor_voxels"].values
    brain_voxels = df["brain_voxels"].values
    tumor_pos_voxels = tumor_voxels[tumor_mask]
    q25, q50, q75 = np.percentile(tumor_pos_voxels, [25, 50, 75])

    bin_defs = [
        ("<=50 voxels (tiny)", tumor_mask & (tumor_voxels <= 50)),
        (f"Q1 (1-{int(q25)} voxels)", tumor_mask & (tumor_voxels <= q25)),
        (f"Q2 ({int(q25)+1}-{int(q50)} voxels)", tumor_mask & (tumor_voxels > q25) & (tumor_voxels <= q50)),
        (f"Q3 ({int(q50)+1}-{int(q75)} voxels)", tumor_mask & (tumor_voxels > q50) & (tumor_voxels <= q75)),
        (f"Q4 (>{int(q75)} voxels)", tumor_mask & (tumor_voxels > q75)),
        ("Post-hoc: tumor area >= 1% brain area", tumor_mask & (brain_voxels > 0) & ((tumor_voxels / np.maximum(brain_voxels, 1)) >= 0.01)),
    ]

    size_bin_results = {}
    for b_name, b_mask in bin_defs:
        b_idxs = np.where(b_mask)[0]
        n_s = len(b_idxs)
        b_pids = df["patient_id"].values[b_idxs]
        u_pids = np.unique(b_pids)
        n_p = len(u_pids)

        sens_pt = float(np.mean(p_ori[b_idxs] == 1))

        # AUC of positive slices in this bin vs all negative slices
        y_bin_auc = np.concatenate([np.ones(n_s, dtype=int), np.zeros(n_neg_all, dtype=int)])
        s_bin_auc = np.concatenate([s_ori[b_idxs], neg_scores_all])
        auc_vs_all_neg = float(roc_auc_score(y_bin_auc, s_bin_auc))

        # Patient-cluster bootstrap CI for sensitivity in this bin (2000 resamples, seed 42)
        p_bin_map = {}
        for idx_in_df in b_idxs:
            pid = patients[idx_in_df]
            if pid not in p_bin_map:
                p_bin_map[pid] = []
            p_bin_map[pid].append(idx_in_df)
        for pid in p_bin_map:
            p_bin_map[pid] = np.array(p_bin_map[pid], dtype=np.int32)

        rng_b = np.random.RandomState(42)
        boot_sens = []
        for _ in range(2000):
            spids = rng_b.choice(u_pids, size=n_p, replace=True)
            sidxs = np.concatenate([p_bin_map[pid] for pid in spids])
            boot_sens.append(np.mean(p_ori[sidxs] == 1))

        s_low, s_high = np.percentile(boot_sens, [2.5, 97.5])
        size_bin_results[b_name] = {
            "n_slices": int(n_s),
            "n_patients": int(n_p),
            "sensitivity": sens_pt,
            "sensitivity_ci_95": f"{s_low:.4f} - {s_high:.4f}",
            "sensitivity_ci_low": float(s_low),
            "sensitivity_ci_high": float(s_high),
            "auc_vs_all_negatives": auc_vs_all_neg,
            "mean_score": float(np.mean(s_ori[b_idxs])),
            "median_score": float(np.median(s_ori[b_idxs]))
        }
        print(f"  {b_name}: n={n_s}, patients={n_p}, sens={sens_pt:.4f} ({s_low:.4f}-{s_high:.4f}), AUC={auc_vs_all_neg:.4f}", flush=True)

    # -------------------------------------------------------------------------
    # 3. SLICE-POSITION TERTILES WITH CIs
    # -------------------------------------------------------------------------
    print("Computing slice-position tertile CIs...", flush=True)
    slice_ids = df["slice_id"].values
    q33, q66 = np.percentile(slice_ids, [33.33, 66.66])
    tertile_defs = [
        (f"Inferior (slice_id <= {int(q33)})", slice_ids <= q33),
        (f"Middle ({int(q33)+1} <= slice_id <= {int(q66)})", (slice_ids > q33) & (slice_ids <= q66)),
        (f"Superior (slice_id > {int(q66)})", slice_ids > q66),
    ]

    tertile_results = {}
    for t_name, t_mask in tertile_defs:
        t_idxs = np.where(t_mask)[0]
        t_pids = np.unique(patients[t_idxs])
        n_tp = len(t_pids)

        p_t_map = {}
        for idx_in_df in t_idxs:
            pid = patients[idx_in_df]
            if pid not in p_t_map:
                p_t_map[pid] = []
            p_t_map[pid].append(idx_in_df)
        for pid in p_t_map:
            p_t_map[pid] = np.array(p_t_map[pid], dtype=np.int32)

        yt = y_true[t_idxs]
        pt = p_ori[t_idxs]
        st = s_ori[t_idxs]

        sens_pt = float(recall_score(yt, pt, pos_label=1))
        spec_pt = float(recall_score(yt, pt, pos_label=0))
        auc_pt = float(roc_auc_score(yt, st))

        rng_t = np.random.RandomState(42)
        b_sens, b_spec, b_auc = [], [], []
        for _ in range(2000):
            spids = rng_t.choice(t_pids, size=n_tp, replace=True)
            sidxs = np.concatenate([p_t_map[pid] for pid in spids])
            yb = y_true[sidxs]
            pb = p_ori[sidxs]
            sb = s_ori[sidxs]
            b_sens.append(np.mean(pb[yb == 1] == 1))
            b_spec.append(np.mean(pb[yb == 0] == 0))
            b_auc.append(roc_auc_score(yb, sb))

        sens_l, sens_h = np.percentile(b_sens, [2.5, 97.5])
        spec_l, spec_h = np.percentile(b_spec, [2.5, 97.5])
        auc_l, auc_h = np.percentile(b_auc, [2.5, 97.5])

        tertile_results[t_name] = {
            "n_slices": int(len(t_idxs)),
            "n_tumor_slices": int(np.sum(yt == 1)),
            "n_no_tumor_slices": int(np.sum(yt == 0)),
            "n_patients": int(n_tp),
            "sensitivity": sens_pt,
            "sensitivity_ci_95": f"{sens_l:.4f} - {sens_h:.4f}",
            "specificity": spec_pt,
            "specificity_ci_95": f"{spec_l:.4f} - {spec_h:.4f}",
            "auc": auc_pt,
            "auc_ci_95": f"{auc_l:.4f} - {auc_h:.4f}",
        }
        print(f"  {t_name}: sens={sens_pt:.4f} ({sens_l:.4f}-{sens_h:.4f}), spec={spec_pt:.4f} ({spec_l:.4f}-{spec_h:.4f}), auc={auc_pt:.4f} ({auc_l:.4f}-{auc_h:.4f})", flush=True)

    # -------------------------------------------------------------------------
    # 4. PER-PATIENT DETECTION SUMMARY
    # -------------------------------------------------------------------------
    grouped = df[df["label"] == 1].groupby("patient_id")
    fracs = []
    n_detected_patients = 0
    n_zero_detected_patients = 0
    for pid, g in grouped:
        n_det = int(np.sum(g["pred_ori"].values == 1))
        fracs.append(n_det / len(g))
        if n_det > 0:
            n_detected_patients += 1
        else:
            n_zero_detected_patients += 1

    per_patient_summary = {
        "total_patients_with_tumor_slices": int(len(fracs)),
        "patients_with_at_least_one_detected_tumor_slice": int(n_detected_patients),
        "patients_with_zero_detected_tumor_slices": int(n_zero_detected_patients),
        "patient_level_any_slice_sensitivity": float(n_detected_patients / len(fracs)),
        "fraction_of_tumor_slices_detected": {
            "min": float(np.min(fracs)),
            "q25": float(np.percentile(fracs, 25)),
            "median": float(np.median(fracs)),
            "mean": float(np.mean(fracs)),
            "q75": float(np.percentile(fracs, 75)),
            "max": float(np.max(fracs)),
        }
    }

    out = {
        "paired_bootstrap_comparisons": paired_results,
        "tumor_size_bins_with_ci_and_auc": size_bin_results,
        "slice_position_tertiles_with_ci": tertile_results,
        "per_patient_detection": per_patient_summary
    }

    with open("paper_evidence/brats_supplementary_cIs_and_subgroups.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved paper_evidence/brats_supplementary_cIs_and_subgroups.json in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()

