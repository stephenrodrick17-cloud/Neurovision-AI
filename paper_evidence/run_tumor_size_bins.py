"""
Compute per-slice tumor voxel counts from BraTS 2023 GLI segmentation NIfTI files.
Then join with brats_oriented predictions to produce tumor-size-bin metrics.

Pre-specified bins (EXPLORATORY):
  - Bin 0: <= 50 tumor voxels ("tiny")
  - Q1: bottom quartile of tumor voxel count among tumor-positive slices
  - Q2: second quartile
  - Q3: third quartile
  - Q4: top quartile

Secondary analysis (POST HOC):
  - Sensitivity restricted to slices with tumor area >= 1% of brain area
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics import recall_score, roc_auc_score

def main():
    seg_root = "D:/datasets/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    pred_path = "paper_evidence/brats_oriented_predictions.csv.gz"
    manifest_path = "D:/datasets/prepared/brats/manifest.csv"

    df_pred = pd.read_csv(pred_path)
    df_man = pd.read_csv(manifest_path)
    print(f"Predictions: {len(df_pred)}, Manifest: {len(df_man)}")

    patients = sorted(df_man['patient_id'].unique())
    print(f"Total patients: {len(patients)}")

    # Compute per-slice tumor voxel count and brain voxel count from seg masks
    records = []
    for pi, pid in enumerate(patients):
        seg_path = os.path.join(seg_root, pid, f"{pid}-seg.nii.gz")
        if not os.path.exists(seg_path):
            continue
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata()  # shape (H, W, D)

        # Also load a structural to get brain mask (non-zero in t2f)
        t2f_path = os.path.join(seg_root, pid, f"{pid}-t2f.nii.gz")
        if os.path.exists(t2f_path):
            brain_data = nib.load(t2f_path).get_fdata()
        else:
            brain_data = None

        n_slices = seg_data.shape[2]
        for s in range(n_slices):
            seg_slice = seg_data[:, :, s]
            tumor_voxels = int(np.sum(seg_slice > 0))
            if brain_data is not None:
                brain_voxels = int(np.sum(brain_data[:, :, s] > 0))
            else:
                brain_voxels = 0
            records.append({
                "patient_id": pid,
                "slice_id": s,
                "tumor_voxels": tumor_voxels,
                "brain_voxels": brain_voxels
            })

        if (pi + 1) % 100 == 0 or (pi + 1) == len(patients):
            print(f"Processed {pi+1}/{len(patients)} patients", flush=True)

    df_seg = pd.DataFrame(records)
    print(f"Segmentation slice records: {len(df_seg)}")

    # Merge with predictions
    df_merged = df_pred.merge(df_seg, on=["patient_id", "slice_id"], how="left")
    print(f"Merged: {len(df_merged)}, with tumor_voxels NaN: {df_merged['tumor_voxels'].isna().sum()}")

    # Only tumor-positive slices for quartile computation
    tumor_slices = df_merged[df_merged['label'] == 1].copy()
    tumor_pos_voxels = tumor_slices['tumor_voxels'].values
    print(f"Tumor-positive slices: {len(tumor_slices)}")
    print(f"Tumor voxel count stats: min={np.min(tumor_pos_voxels)}, median={np.median(tumor_pos_voxels)}, max={np.max(tumor_pos_voxels)}")

    # Pre-specified quartile boundaries
    q25, q50, q75 = np.percentile(tumor_pos_voxels, [25, 50, 75])
    print(f"Quartiles: Q25={q25:.1f}, Q50={q50:.1f}, Q75={q75:.1f}")

    # Define bins
    bins_spec = {
        "tiny (<=50 voxels)": tumor_slices[tumor_slices['tumor_voxels'] <= 50],
        f"Q1 (<=Q25={q25:.0f})": tumor_slices[tumor_slices['tumor_voxels'] <= q25],
        f"Q2 (Q25-Q50, {q25:.0f}-{q50:.0f})": tumor_slices[(tumor_slices['tumor_voxels'] > q25) & (tumor_slices['tumor_voxels'] <= q50)],
        f"Q3 (Q50-Q75, {q50:.0f}-{q75:.0f})": tumor_slices[(tumor_slices['tumor_voxels'] > q50) & (tumor_slices['tumor_voxels'] <= q75)],
        f"Q4 (>Q75={q75:.0f})": tumor_slices[tumor_slices['tumor_voxels'] > q75],
    }

    results = {
        "status": "EXPLORATORY PRE-SPECIFIED SUBGROUP ANALYSIS",
        "quartile_boundaries": {"Q25": float(q25), "Q50": float(q50), "Q75": float(q75)},
        "bins": {}
    }

    for name, sub_df in bins_spec.items():
        y = sub_df['label'].values
        s = sub_df['score'].values
        p = (s >= 0.5).astype(int)
        sens = float(recall_score(y, p, pos_label=1, zero_division=0))
        try:
            auc = float(roc_auc_score(y, s))
        except Exception:
            auc = None
        results["bins"][name] = {
            "n_slices": len(sub_df),
            "n_patients": int(sub_df['patient_id'].nunique()),
            "sensitivity": sens,
            "auc": auc,
            "mean_score": float(np.mean(s)),
            "median_score": float(np.median(s))
        }
        print(f"  {name}: n={len(sub_df)}, sens={sens:.4f}, auc={auc}")

    # Secondary post-hoc: tumor area >= 1% of brain area
    tumor_slices_with_brain = tumor_slices[tumor_slices['brain_voxels'] > 0].copy()
    tumor_slices_with_brain['tumor_frac'] = tumor_slices_with_brain['tumor_voxels'] / tumor_slices_with_brain['brain_voxels']
    gte_1pct = tumor_slices_with_brain[tumor_slices_with_brain['tumor_frac'] >= 0.01]

    y_ph = gte_1pct['label'].values
    s_ph = gte_1pct['score'].values
    p_ph = (s_ph >= 0.5).astype(int)
    sens_ph = float(recall_score(y_ph, p_ph, pos_label=1, zero_division=0))
    try:
        auc_ph = float(roc_auc_score(y_ph, s_ph))
    except Exception:
        auc_ph = None

    results["post_hoc_tumor_gte_1pct_brain"] = {
        "status": "POST HOC SECONDARY ANALYSIS",
        "n_slices": len(gte_1pct),
        "n_patients": int(gte_1pct['patient_id'].nunique()),
        "sensitivity": sens_ph,
        "auc": auc_ph,
        "mean_score": float(np.mean(s_ph)),
        "median_tumor_fraction": float(np.median(gte_1pct['tumor_frac']))
    }
    print(f"\nPost-hoc (tumor>=1% brain): n={len(gte_1pct)}, sens={sens_ph:.4f}, auc={auc_ph}")

    # Save segmentation data for later use
    df_seg.to_csv("paper_evidence/brats_seg_voxel_counts.csv.gz", index=False, compression="gzip")
    
    with open("paper_evidence/brats_tumor_size_bins.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to paper_evidence/brats_tumor_size_bins.json")

if __name__ == "__main__":
    main()

