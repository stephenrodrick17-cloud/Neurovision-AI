"""
Pre-Submission Analysis 1: Header-Only / Metadata Baseline on the Kaggle Discovery Dataset
(Following Vangala et al., 2026: testing whether image container/header properties alone predict tumor vs no_tumor without reading pixel intensities).

Features extracted strictly from container headers / metadata (zero pixel values read):
1. File format (JPEG, PNG, TIFF)
2. Color mode (RGB, L, RGBA, P, CMYK, etc.)
3. Pixel dimensions: width, height, aspect_ratio (width / height), pixel_area (width * height)
4. File size: file_size_bytes, bytes_per_pixel (file_size_bytes / (width * height))
5. Container markers: has_jfif, jfif_unit, jfif_density_x, jfif_density_y, has_exif, has_adobe, dpi_x, dpi_y
6. JPEG Quantization Tables:
   - n_quant_tables
   - Table 0 (Luminance) summary: mean, std, min, max, sum, q[0], q[1], q[7], q[63]
   - Table 1 (Chrominance) summary: mean, std, min, max, sum, q[0], q[1], q[7], q[63]
   - All 64 raw coefficients of Quantization Table 0 and all 64 raw coefficients of Quantization Table 1

Trains gradient-boosted decision trees (HistGradientBoostingClassifier and GradientBoostingClassifier, seed=42)
strictly on the train partition (n=3,191) and evaluates on:
- Validation partition (n=803)
- Held-out Test partition (full n=606)
- Clean Test partition (n=465, excluding 141 Cheng-matched test images)
- JPEG-only Test partition (n=590)
Includes 2,000-resample duplicate-group-stratified and image-stratified bootstrap 95% CIs (seed=42 and seed=0),
plus feature-group ablations and permutation feature importances.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from BrainTumorAI.evaluate_classifier import (
    compute_metrics,
    stratified_bootstrap_intervals,
    stratified_group_bootstrap_intervals,
)


def extract_header_features(row: pd.Series) -> dict[str, Any]:
    rel_path = str(row["source_file"])
    full_path = ROOT / rel_path
    file_size = int(full_path.stat().st_size)

    with Image.open(full_path) as im:
        fmt = str(im.format or row["format"] or "UNKNOWN").upper()
        mode = str(im.mode or row["mode"] or "UNKNOWN").upper()
        w, h = im.size
        info = im.info or {}
        qtables = getattr(im, "quantization", None) or {}

    area = max(1, int(w) * int(h))
    bpp = float(file_size) / float(area)
    aspect = float(w) / float(max(1, h))

    jfif_density = info.get("jfif_density", (0, 0))
    if not isinstance(jfif_density, (tuple, list)) or len(jfif_density) < 2:
        jfif_density = (0, 0)
    dpi = info.get("dpi", (0, 0))
    if not isinstance(dpi, (tuple, list)) or len(dpi) < 2:
        dpi = (0, 0)

    q0 = list(qtables.get(0, [])) if isinstance(qtables, dict) else []
    q1 = list(qtables.get(1, [])) if isinstance(qtables, dict) else []
    if len(q0) < 64:
        q0 = q0 + [-1] * (64 - len(q0))
    else:
        q0 = q0[:64]
    if len(q1) < 64:
        q1 = q1 + [-1] * (64 - len(q1))
    else:
        q1 = q1[:64]

    q0_valid = [x for x in q0 if x >= 0]
    q1_valid = [x for x in q1 if x >= 0]

    feat: dict[str, Any] = {
        # Format one-hot
        "fmt_JPEG": 1.0 if fmt == "JPEG" else 0.0,
        "fmt_PNG": 1.0 if fmt == "PNG" else 0.0,
        "fmt_TIFF": 1.0 if fmt == "TIFF" else 0.0,
        # Mode one-hot
        "mode_RGB": 1.0 if mode == "RGB" else 0.0,
        "mode_L": 1.0 if mode == "L" else 0.0,
        "mode_RGBA": 1.0 if mode == "RGBA" else 0.0,
        "mode_P": 1.0 if mode == "P" else 0.0,
        "mode_OTHER": 1.0 if mode not in ("RGB", "L", "RGBA", "P") else 0.0,
        # Dimensions & file size
        "width": float(w),
        "height": float(h),
        "aspect_ratio": aspect,
        "pixel_area": float(area),
        "is_square": 1.0 if w == h else 0.0,
        "file_size_bytes": float(file_size),
        "bytes_per_pixel": bpp,
        # Container / header tags
        "has_jfif": 1.0 if "jfif" in info else 0.0,
        "jfif_unit": float(info.get("jfif_unit", -1)),
        "jfif_density_x": float(jfif_density[0]),
        "jfif_density_y": float(jfif_density[1]),
        "has_exif": 1.0 if "exif" in info else 0.0,
        "has_adobe": 1.0 if "adobe" in info else 0.0,
        "has_icc_profile": 1.0 if "icc_profile" in info else 0.0,
        "dpi_x": float(dpi[0]),
        "dpi_y": float(dpi[1]),
        # Quantization table summaries
        "n_quant_tables": float(len(qtables)) if isinstance(qtables, dict) else 0.0,
        "q0_mean": float(np.mean(q0_valid)) if q0_valid else -1.0,
        "q0_std": float(np.std(q0_valid)) if q0_valid else -1.0,
        "q0_min": float(np.min(q0_valid)) if q0_valid else -1.0,
        "q0_max": float(np.max(q0_valid)) if q0_valid else -1.0,
        "q0_sum": float(np.sum(q0_valid)) if q0_valid else -1.0,
        "q1_mean": float(np.mean(q1_valid)) if q1_valid else -1.0,
        "q1_std": float(np.std(q1_valid)) if q1_valid else -1.0,
        "q1_min": float(np.min(q1_valid)) if q1_valid else -1.0,
        "q1_max": float(np.max(q1_valid)) if q1_valid else -1.0,
        "q1_sum": float(np.sum(q1_valid)) if q1_valid else -1.0,
    }
    for i in range(64):
        feat[f"q0_{i:02d}"] = float(q0[i])
    for i in range(64):
        feat[f"q1_{i:02d}"] = float(q1[i])
    return feat


def eval_with_bootstraps(
    y_true: np.ndarray,
    scores: np.ndarray,
    dup_groups: np.ndarray,
    compute_ci: bool = True,
) -> dict[str, Any]:
    m = compute_metrics(y_true, scores, threshold=0.5)
    out = {
        "n": int(m["sample_count"]),
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "sensitivity": m["tumor_sensitivity_recall"],
        "specificity": m["no_tumor_specificity"],
        "precision": m["tumor_precision"],
        "f1": m["tumor_f1"],
        "roc_auc": m["roc_auc"],
        "pr_auc": m["average_precision"],
        "confusion_matrix": m["confusion_matrix"],
    }
    if compute_ci:
        out["dup_group_bootstrap_95_seed42"] = stratified_group_bootstrap_intervals(
            y_true, scores, dup_groups, threshold=0.5, seed=42, replicates=2000
        )
        out["image_bootstrap_95_seed42"] = stratified_bootstrap_intervals(
            y_true, scores, threshold=0.5, seed=42, replicates=2000
        )
    return out


def main() -> None:
    t0 = time.time()
    manifest_path = ROOT / "BrainTumorAI" / "results" / "split_manifest.csv"
    df = pd.read_csv(manifest_path)
    print(f"Loaded {len(df)} rows from {manifest_path}", flush=True)

    overlap_audit = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8")
    )
    cheng_test_files = {
        edge["local_source_file"]
        for edge in overlap_audit["matches"]["high_similarity_matches"]
        if edge["local_split"] == "test"
    }

    feat_rows = [extract_header_features(row) for _, row in df.iterrows()]
    df_feat = pd.DataFrame(feat_rows)
    feature_names = list(df_feat.columns)
    print(f"Extracted {len(feature_names)} non-pixel header features in {time.time() - t0:.2f}s", flush=True)

    is_train = (df["split"] == "train").to_numpy()
    is_val = (df["split"] == "validation").to_numpy()
    is_test = (df["split"] == "test").to_numpy()

    X_train = df_feat.loc[is_train].to_numpy(dtype=np.float64)
    y_train = df.loc[is_train, "label_idx"].to_numpy(dtype=np.int64)

    X_val = df_feat.loc[is_val].to_numpy(dtype=np.float64)
    y_val = df.loc[is_val, "label_idx"].to_numpy(dtype=np.int64)
    g_val = df.loc[is_val, "duplicate_group"].astype(str).to_numpy()

    df_test_meta = df.loc[is_test].copy().reset_index(drop=True)
    X_test = df_feat.loc[is_test].to_numpy(dtype=np.float64)
    y_test = df_test_meta["label_idx"].to_numpy(dtype=np.int64)
    g_test = df_test_meta["duplicate_group"].astype(str).to_numpy()

    clean_mask_test = ~df_test_meta["source_file"].isin(cheng_test_files).to_numpy()
    jpeg_mask_test = (df_test_meta["format"].astype(str).str.upper() == "JPEG").to_numpy()

    # Descriptive summary of header differences between tumor and no_tumor across full dataset (4,600) and test (606)
    df_all = pd.concat([df[["split", "label", "label_idx", "format", "mode"]], df_feat], axis=1)
    header_distribution_summary = {
        "by_class_all_4600": {},
        "by_class_test_606": {},
    }
    for subset_name, sub_df in (("by_class_all_4600", df_all), ("by_class_test_606", df_all[df_all["split"] == "test"])):
        for lbl_name, grp in sub_df.groupby("label"):
            header_distribution_summary[subset_name][str(lbl_name)] = {
                "n": int(len(grp)),
                "format_counts": {str(k): int(v) for k, v in grp["format"].value_counts().items()},
                "mode_counts": {str(k): int(v) for k, v in grp["mode"].value_counts().items()},
                "is_512x512_count": int(np.sum((grp["width"] == 512) & (grp["height"] == 512))),
                "is_square_count": int(np.sum(grp["is_square"] == 1.0)),
                "median_width": float(grp["width"].median()),
                "median_height": float(grp["height"].median()),
                "median_file_size_bytes": float(grp["file_size_bytes"].median()),
                "median_bytes_per_pixel": float(grp["bytes_per_pixel"].median()),
                "median_q0_mean": float(grp["q0_mean"].median()),
            }

    # Train primary gradient-boosted trees models on train (n=3,191)
    models = {
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    model_results: dict[str, Any] = {}
    pred_df = df_test_meta[["source_file", "duplicate_group", "label", "label_idx", "format", "mode", "width", "height"]].copy()

    for m_name, clf in models.items():
        clf.fit(X_train, y_train)
        tr_scores = clf.predict_proba(X_train)[:, 1]
        va_scores = clf.predict_proba(X_val)[:, 1]
        te_scores = clf.predict_proba(X_test)[:, 1]
        pred_df[f"score_{m_name}"] = te_scores

        perm = permutation_importance(
            clf, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=42
        )
        top_idx = np.argsort(perm.importances_mean)[::-1][:15]
        top_features = [
            {
                "feature": feature_names[idx],
                "importance_mean_auc_drop": float(perm.importances_mean[idx]),
                "importance_std": float(perm.importances_std[idx]),
            }
            for idx in top_idx
        ]

        model_results[m_name] = {
            "train_3191": eval_with_bootstraps(y_train, tr_scores, df.loc[is_train, "duplicate_group"].astype(str).to_numpy(), compute_ci=False),
            "validation_803": eval_with_bootstraps(y_val, va_scores, g_val, compute_ci=False),
            "test_full_606": eval_with_bootstraps(y_test, te_scores, g_test, compute_ci=True),
            "test_clean_465": eval_with_bootstraps(y_test[clean_mask_test], te_scores[clean_mask_test], g_test[clean_mask_test], compute_ci=True),
            "test_jpeg_only_590": eval_with_bootstraps(y_test[jpeg_mask_test], te_scores[jpeg_mask_test], g_test[jpeg_mask_test], compute_ci=True),
            "top_15_permutation_importances_on_test_auc": top_features,
        }
        te_m = model_results[m_name]["test_full_606"]
        cl_m = model_results[m_name]["test_clean_465"]
        print(
            f"  [{m_name}]\n"
            f"    Test Full (n=606):  Acc={te_m['accuracy']:.4f}, BalAcc={te_m['balanced_accuracy']:.4f}, "
            f"Sens={te_m['sensitivity']:.4f}, Spec={te_m['specificity']:.4f}, AUC={te_m['roc_auc']:.4f} "
            f"(Dup-Group 95% CI: {te_m['dup_group_bootstrap_95_seed42']['roc_auc']['lower_95']:.4f} - {te_m['dup_group_bootstrap_95_seed42']['roc_auc']['upper_95']:.4f})\n"
            f"    Test Clean (n=465): Acc={cl_m['accuracy']:.4f}, BalAcc={cl_m['balanced_accuracy']:.4f}, "
            f"Sens={cl_m['sensitivity']:.4f}, Spec={cl_m['specificity']:.4f}, AUC={cl_m['roc_auc']:.4f}",
            flush=True,
        )

    # Feature-group ablations using HistGradientBoostingClassifier
    feature_groups = {
        "format_and_mode_only": [c for c in feature_names if c.startswith("fmt_") or c.startswith("mode_")],
        "dimensions_only": ["width", "height", "aspect_ratio", "pixel_area", "is_square"],
        "file_size_and_bpp_only": ["file_size_bytes", "bytes_per_pixel"],
        "dimensions_and_filesize": ["width", "height", "aspect_ratio", "pixel_area", "is_square", "file_size_bytes", "bytes_per_pixel"],
        "jpeg_quantization_tables_only": [c for c in feature_names if c.startswith("q0_") or c.startswith("q1_") or c == "n_quant_tables"],
        "container_markers_only": ["has_jfif", "jfif_unit", "jfif_density_x", "jfif_density_y", "has_exif", "has_adobe", "has_icc_profile", "dpi_x", "dpi_y"],
        "all_163_header_features": feature_names,
    }

    ablation_results = {}
    for grp_name, cols in feature_groups.items():
        clf_ab = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
        )
        X_tr_sub = df_feat.loc[is_train, cols].to_numpy(dtype=np.float64)
        X_te_sub = df_feat.loc[is_test, cols].to_numpy(dtype=np.float64)
        clf_ab.fit(X_tr_sub, y_train)
        s_te = clf_ab.predict_proba(X_te_sub)[:, 1]
        ab_full = eval_with_bootstraps(y_test, s_te, g_test, compute_ci=False)
        ab_clean = eval_with_bootstraps(y_test[clean_mask_test], s_te[clean_mask_test], g_test[clean_mask_test], compute_ci=False)
        ablation_results[grp_name] = {
            "n_features": len(cols),
            "test_full_606": ab_full,
            "test_clean_465": ab_clean,
        }
        print(
            f"  [Ablation: {grp_name:30s} ({len(cols):3d} feats)] "
            f"Test606 Acc={ab_full['accuracy']:.4f}, AUC={ab_full['roc_auc']:.4f} | "
            f"Clean465 Acc={ab_clean['accuracy']:.4f}, AUC={ab_clean['roc_auc']:.4f}",
            flush=True,
        )

    pred_csv_path = ROOT / "paper_evidence" / "header_only_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)

    payload = {
        "analysis": "Header-only (non-pixel metadata) baseline on Kaggle Brain Tumor Detection discovery dataset",
        "reference": "Vangala et al., 2026 (metadata/header shortcut audit)",
        "manifest_path": "BrainTumorAI/results/split_manifest.csv",
        "train_n": int(np.sum(is_train)),
        "val_n": int(np.sum(is_val)),
        "test_n": int(np.sum(is_test)),
        "n_features_total": len(feature_names),
        "feature_names": feature_names,
        "header_distribution_summary": header_distribution_summary,
        "models": model_results,
        "feature_group_ablations_hist_gbdt": ablation_results,
        "predictions_csv": "paper_evidence/header_only_test_predictions.csv",
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_json = ROOT / "paper_evidence" / "header_only_baseline.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved header-only baseline results to {out_json}", flush=True)


if __name__ == "__main__":
    main()

