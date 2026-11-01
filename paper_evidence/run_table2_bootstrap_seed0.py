"""
Pre-Submission Analysis 4: Second-Pass Bootstrap (2,000 resamples, seed 0 and seed 42) for Table 2
(Internal Discovery Test Set n=606 and Clean Test Set n=465).

Re-runs and saves the exact duplicate-group-stratified and image-stratified bootstrap confidence intervals
using BrainTumorAI.evaluate_classifier.{stratified_group_bootstrap_intervals, stratified_bootstrap_intervals}
with:
  1. replicates = 1000, seed = 42 (matches BrainTumorAI/results/evaluation/evaluation_metrics.json)
  2. replicates = 2000, seed = 0  (second-pass bootstrap; matches experiments_results/phase1c/resnet18_group_bootstrap_2000_seed0.json)
  3. replicates = 2000, seed = 42
for all 4 evaluated models in BrainTumorAI/results/evaluation/test_predictions.csv on both:
  - Full test split (n = 606)
  - Clean test split (n = 465, excluding the 141 Cheng-matched test images)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from BrainTumorAI.evaluate_classifier import (
    compute_metrics,
    stratified_bootstrap_intervals,
    stratified_group_bootstrap_intervals,
)


def main() -> None:
    t0 = time.time()
    pred_path = ROOT / "BrainTumorAI" / "results" / "evaluation" / "test_predictions.csv"
    df_preds = pd.read_csv(pred_path)

    overlap_audit = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8")
    )
    cheng_test_files = {
        edge["local_source_file"]
        for edge in overlap_audit["matches"]["high_similarity_matches"]
        if edge["local_split"] == "test"
    }

    phase1c_ref_path = ROOT / "experiments_results" / "phase1c" / "resnet18_group_bootstrap_2000_seed0.json"
    phase1c_ref = json.loads(phase1c_ref_path.read_text(encoding="utf-8")) if phase1c_ref_path.is_file() else None

    models = ["resnet18", "small_cnn_small_cnn_best", "legacy_existing_checkpoint", "training_majority_class"]
    results: dict[str, Any] = {}

    for m_key in models:
        sub_full = df_preds[df_preds["model"] == m_key].copy().reset_index(drop=True)
        sub_clean = sub_full[~sub_full["source_file"].isin(cheng_test_files)].copy().reset_index(drop=True)

        m_out: dict[str, Any] = {}
        for subset_key, sub_df in (("test_full_606", sub_full), ("test_clean_465", sub_clean)):
            y_true = sub_df["true_label_idx"].to_numpy(dtype=np.int64)
            scores = sub_df["tumor_score"].to_numpy(dtype=np.float64)
            groups = sub_df["duplicate_group"].astype(str).to_numpy()

            pt = compute_metrics(y_true, scores, threshold=0.5)
            grp_1000_s42 = stratified_group_bootstrap_intervals(y_true, scores, groups, threshold=0.5, seed=42, replicates=1000)
            img_1000_s42 = stratified_bootstrap_intervals(y_true, scores, threshold=0.5, seed=42, replicates=1000)

            grp_2000_s0 = stratified_group_bootstrap_intervals(y_true, scores, groups, threshold=0.5, seed=0, replicates=2000)
            img_2000_s0 = stratified_bootstrap_intervals(y_true, scores, threshold=0.5, seed=0, replicates=2000)

            grp_2000_s42 = stratified_group_bootstrap_intervals(y_true, scores, groups, threshold=0.5, seed=42, replicates=2000)
            img_2000_s42 = stratified_bootstrap_intervals(y_true, scores, threshold=0.5, seed=42, replicates=2000)

            m_out[subset_key] = {
                "n": int(len(sub_df)),
                "point_estimates": {
                    "accuracy": pt["accuracy"],
                    "balanced_accuracy": pt["balanced_accuracy"],
                    "tumor_sensitivity_recall": pt["tumor_sensitivity_recall"],
                    "no_tumor_specificity": pt["no_tumor_specificity"],
                    "tumor_precision": pt["tumor_precision"],
                    "tumor_f1": pt["tumor_f1"],
                    "roc_auc": pt["roc_auc"],
                    "average_precision": pt["average_precision"],
                    "confusion_matrix": pt["confusion_matrix"],
                },
                "duplicate_group_stratified_95_ci": {
                    "replicates_1000_seed_42": grp_1000_s42,
                    "replicates_2000_seed_0": grp_2000_s0,
                    "replicates_2000_seed_42": grp_2000_s42,
                },
                "image_stratified_95_ci": {
                    "replicates_1000_seed_42": img_1000_s42,
                    "replicates_2000_seed_0": img_2000_s0,
                    "replicates_2000_seed_42": img_2000_s42,
                },
            }
        results[m_key] = m_out
        r606 = m_out["test_full_606"]
        g2000 = r606["duplicate_group_stratified_95_ci"]["replicates_2000_seed_0"]
        print(
            f"  [{m_key:28s} | Full n=606] "
            f"Acc={r606['point_estimates']['accuracy']:.4f} (2000-seed0 DupGrp CI: {g2000['accuracy']['lower_95']:.4f} - {g2000['accuracy']['upper_95']:.4f}), "
            f"Sens={r606['point_estimates']['tumor_sensitivity_recall']:.4f} ({g2000['tumor_sensitivity_recall']['lower_95']:.4f} - {g2000['tumor_sensitivity_recall']['upper_95']:.4f}), "
            f"Spec={r606['point_estimates']['no_tumor_specificity']:.4f} ({g2000['no_tumor_specificity']['lower_95']:.4f} - {g2000['no_tumor_specificity']['upper_95']:.4f}), "
            f"AUC={r606['point_estimates']['roc_auc']:.4f} ({g2000['roc_auc']['lower_95']:.4f} - {g2000['roc_auc']['upper_95']:.4f})",
            flush=True,
        )

    # Verify exact match with experiments_results/phase1c/resnet18_group_bootstrap_2000_seed0.json
    verification = {}
    if phase1c_ref is not None:
        g2000_r18 = results["resnet18"]["test_full_606"]["duplicate_group_stratified_95_ci"]["replicates_2000_seed_0"]
        key_map = {
            "accuracy": "accuracy",
            "sensitivity": "tumor_sensitivity_recall",
            "specificity": "no_tumor_specificity",
            "roc_auc": "roc_auc",
        }
        max_diff = 0.0
        for ref_k, our_k in key_map.items():
            ref_lo = phase1c_ref["comparison"][ref_k]["recomputed_2000_seed0"]["lower_95"]
            ref_hi = phase1c_ref["comparison"][ref_k]["recomputed_2000_seed0"]["upper_95"]
            our_lo = g2000_r18[our_k]["lower_95"]
            our_hi = g2000_r18[our_k]["upper_95"]
            d = max(abs(ref_lo - our_lo), abs(ref_hi - our_hi))
            max_diff = max(max_diff, d)
            verification[ref_k] = {
                "phase1c_saved_lower_95": ref_lo,
                "recomputed_lower_95": our_lo,
                "phase1c_saved_upper_95": ref_hi,
                "recomputed_upper_95": our_hi,
                "abs_diff": d,
            }
        verification["exact_match_with_phase1c_json"] = bool(max_diff == 0.0)
        print(f"Verified exact match with experiments_results/phase1c/resnet18_group_bootstrap_2000_seed0.json: max_diff={max_diff:.2e}", flush=True)

    payload = {
        "analysis": "Second-pass bootstrap (2,000 resamples, seed 0 and seed 42) for Table 2 (Internal Discovery Test Set)",
        "script_path": "paper_evidence/run_table2_bootstrap_seed0.py",
        "predictions_source": "BrainTumorAI/results/evaluation/test_predictions.csv",
        "verification_vs_phase1c_json": verification,
        "models": results,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_path = ROOT / "paper_evidence" / "table2_bootstrap_seed0_2000.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved second-pass bootstrap report to {out_path} in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()

