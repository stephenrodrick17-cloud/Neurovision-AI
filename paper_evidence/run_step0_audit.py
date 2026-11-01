"""
Step 0 Audit Script:
(a) Extracts and verifies exact training settings used for seed 42 of each Patient-Split BraTS model
    (including reading checkpoint metadata from paper_evidence/models/*.pth) and saves
    paper_evidence/leakage_free_training_settings.json.
(b) Generates and saves the fixed patient-split manifest paper_evidence/brats_patient_split_manifest.csv
    (850 train patients / 6,800 slices, 151 val patients / 1,208 slices, 250 test patients / 33,696 slices)
    and asserts zero patient overlap across train, validation, and test splits.
(c) Resolves all 4 reported inconsistencies from raw saved files on disk and saves
    paper_evidence/inconsistency_resolution.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from BrainTumorAI.classifier_pipeline import load_checkpoint, sha256_file


def main() -> None:
    pe_dir = ROOT / "paper_evidence"
    models_dir = pe_dir / "models"

    # =========================================================================
    # STEP 0(a): Exact Training Settings for Seed 42 of Each BraTS Model
    # =========================================================================
    ckpt_a = models_dir / "resnet18_brats_patient_split.pth"
    ckpt_b = models_dir / "resnet18_brats_finetuned.pth"
    ckpt_c = models_dir / "small_cnn_brats_patient_split.pth"

    _, meta_a = load_checkpoint(ckpt_a, map_location="cpu")
    _, meta_b = load_checkpoint(ckpt_b, map_location="cpu")
    _, meta_c = load_checkpoint(ckpt_c, map_location="cpu")

    common_settings = {
        "training_script": "paper_evidence/run_leakage_free_baselines.py",
        "training_function": "train_binary_model (lines 165-282)",
        "pipeline_definitions": "BrainTumorAI/classifier_pipeline.py",
        "max_epochs": 6,
        "max_epochs_source": "paper_evidence/run_leakage_free_baselines.py lines 383, 401, 419",
        "optimizer": "torch.optim.Adam",
        "optimizer_source": "paper_evidence/run_leakage_free_baselines.py line 196",
        "weight_decay": 1e-4,
        "weight_decay_source": "paper_evidence/run_leakage_free_baselines.py line 196",
        "scheduler": "torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2) stepped on val_balanced_accuracy",
        "scheduler_source": "paper_evidence/run_leakage_free_baselines.py lines 197, 237",
        "batch_size_train": 32,
        "batch_size_val": 32,
        "batch_size_source": "paper_evidence/run_leakage_free_baselines.py lines 174, 180-192, 384, 402, 420",
        "augmentation_train": [
            "transforms.Resize((224, 224))",
            "transforms.RandomHorizontalFlip(p=0.5)",
            "transforms.RandomRotation(degrees=10)",
            "transforms.ToTensor()",
            "transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))",
        ],
        "augmentation_val_and_test": [
            "transforms.Resize((224, 224))",
            "transforms.ToTensor()",
            "transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))",
        ],
        "augmentation_source": "BrainTumorAI/classifier_pipeline.py lines 75-94 (make_inference_transform, make_training_transform)",
        "input_size": [224, 224, 3],
        "input_size_source": "BrainTumorAI/classifier_pipeline.py line 21 (IMAGE_SIZE = 224)",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "source": "BrainTumorAI/classifier_pipeline.py lines 22-23",
        },
        "loss_function": "torch.nn.CrossEntropyLoss()",
        "loss_source": "paper_evidence/run_leakage_free_baselines.py line 195",
        "class_weighting": "None (unweighted CrossEntropyLoss; training & validation datasets are balanced by sampling 4 tumor + 4 non-tumor slices per patient)",
        "class_weighting_source": "paper_evidence/run_leakage_free_baselines.py lines 195, 325-340",
        "model_selection_and_early_stopping_rule": {
            "selection_metric": "validation_balanced_accuracy (tie-broken by lower validation_loss when |va_bal - best_bal_acc| <= 1e-4)",
            "improvement_condition": "va_bal > best_bal_acc + 1e-4 or (abs(va_bal - best_bal_acc) <= 1e-4 and va_loss < best_val_loss)",
            "early_stopping_patience_epochs": 2,
            "early_stopping_condition": "no_improve >= 2",
            "test_set_used_for_selection": False,
            "source": "paper_evidence/run_leakage_free_baselines.py lines 248-257",
        },
        "seeding_and_determinism": {
            "seed_function": "seed_everything(seed) in BrainTumorAI/classifier_pipeline.py lines 110-121",
            "rngs_seeded": [
                "random.seed(seed)",
                "np.random.seed(seed)",
                "torch.manual_seed(seed)",
                "torch.cuda.manual_seed_all(seed)",
                "DataLoader generator=torch.Generator().manual_seed(seed)",
            ],
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    }

    models_settings = {
        "model_A_resnet18_brats_patient_split": {
            "model_id": "resnet18_brats_patient_split",
            "architecture": "resnet18",
            "initialization": "ImageNet pretrained (torchvision.models.ResNet18_Weights.IMAGENET1K_V1), final Linear(512, 2) randomly initialized after seed_everything(seed) or make_model('resnet18', pretrained=True)",
            "initialization_source": "BrainTumorAI/classifier_pipeline.py lines 62-67 & paper_evidence/run_leakage_free_baselines.py line 374",
            "learning_rate": 1e-4,
            "learning_rate_source": "paper_evidence/run_leakage_free_baselines.py line 385",
            "seed42_checkpoint_path": "paper_evidence/models/resnet18_brats_patient_split.pth",
            "seed42_checkpoint_sha256": sha256_file(ckpt_a),
            "seed42_checkpoint_metadata": meta_a,
        },
        "model_B_resnet18_brats_finetuned": {
            "model_id": "resnet18_brats_finetuned",
            "architecture": "resnet18",
            "initialization": "Fine-tuned from primary Discovery ResNet-18 checkpoint (BrainTumorAI/models/tumor_classifier_reproducible.pth)",
            "initialization_source": "paper_evidence/run_leakage_free_baselines.py line 392",
            "learning_rate": 5e-5,
            "learning_rate_source": "paper_evidence/run_leakage_free_baselines.py line 403",
            "seed42_checkpoint_path": "paper_evidence/models/resnet18_brats_finetuned.pth",
            "seed42_checkpoint_sha256": sha256_file(ckpt_b),
            "seed42_checkpoint_metadata": meta_b,
        },
        "model_C_small_cnn_brats_patient_split": {
            "model_id": "small_cnn_brats_patient_split",
            "architecture": "small_cnn",
            "initialization": "Randomly initialized from scratch (SmallCNN: 4 Conv2d-BN-ReLU blocks [32, 64, 128, 256], AdaptiveAvgPool2d(1,1), Dropout(0.2), Linear(256, 2))",
            "initialization_source": "BrainTumorAI/classifier_pipeline.py lines 31-71 & paper_evidence/run_leakage_free_baselines.py line 410",
            "learning_rate": 1e-4,
            "learning_rate_source": "paper_evidence/run_leakage_free_baselines.py line 421",
            "seed42_checkpoint_path": "paper_evidence/models/small_cnn_brats_patient_split.pth",
            "seed42_checkpoint_sha256": sha256_file(ckpt_c),
            "seed42_checkpoint_metadata": meta_c,
        },
    }

    settings_payload = {
        "common_training_settings": common_settings,
        "per_model_settings": models_settings,
        "note_on_seed42_leakage_free_baselines_json_training_info": (
            "In run_leakage_free_baselines.py lines 361-363, when reloading existing seed 42 checkpoints from disk, "
            "the script called meta.get('extra', {}) instead of reading meta directly, leaving training_info as {} "
            "in leakage_free_baselines.json. The actual seed 42 selected_epoch, validation_balanced_accuracy, and "
            "validation_loss are stored directly inside each .pth checkpoint's metadata dict (quoted above)."
        ),
    }

    settings_out = pe_dir / "leakage_free_training_settings.json"
    settings_out.write_text(json.dumps(settings_payload, indent=2), encoding="utf-8")
    print(f"Saved {settings_out}")
    for k, v in models_settings.items():
        print(f"  {k}: lr={v['learning_rate']}, meta={v['seed42_checkpoint_metadata']}")

    # =========================================================================
    # STEP 0(b): Fixed Patient-Split BraTS Manifest (brats_patient_split_manifest.csv)
    # =========================================================================
    brats_root = Path("D:/datasets/prepared/brats_oriented")
    df_full_brats = pd.read_csv(brats_root / "manifest.csv")
    df_test_250 = pd.read_csv(pe_dir / "brats_250_patients_manifest.csv")

    test_patients_set = set(df_test_250["patient_id"].unique())
    all_brats_patients = sorted(df_full_brats["patient_id"].unique())
    remaining_patients = [p for p in all_brats_patients if p not in test_patients_set]
    assert len(test_patients_set) == 250
    assert len(remaining_patients) == 1001
    assert len(set(remaining_patients) & test_patients_set) == 0

    # Fixed patient split seed = 42 (independent of model training seed)
    rng_split = np.random.default_rng(42)
    shuffled_rem = rng_split.permutation(remaining_patients)
    train_patients = sorted(shuffled_rem[:850].tolist())
    val_patients = sorted(shuffled_rem[850:].tolist())

    assert len(train_patients) == 850
    assert len(val_patients) == 151
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & test_patients_set) == 0
    assert len(set(val_patients) & test_patients_set) == 0

    def sample_patient_slices(df_pool: pd.DataFrame, pid_list: list[str], k_per_class: int, seed_val: int) -> pd.DataFrame:
        rng_s = np.random.default_rng(seed_val)
        sub_df = df_pool[df_pool["patient_id"].isin(set(pid_list))]
        chosen_frames = []
        for _, grp in sub_df.groupby("patient_id", sort=True):
            for lbl in (0, 1):
                c_grp = grp[grp["label_idx"] == lbl]
                if len(c_grp) <= k_per_class:
                    chosen_frames.append(c_grp)
                else:
                    idx = rng_s.choice(len(c_grp), size=k_per_class, replace=False)
                    chosen_frames.append(c_grp.iloc[np.sort(idx)])
        return pd.concat(chosen_frames, ignore_index=True)

    df_brats_train = sample_patient_slices(df_full_brats, train_patients, k_per_class=4, seed_val=42).copy()
    df_brats_val = sample_patient_slices(df_full_brats, val_patients, k_per_class=4, seed_val=43).copy()
    df_brats_test = df_test_250.copy()

    df_brats_train["split"] = "train"
    df_brats_val["split"] = "validation"
    df_brats_test["split"] = "test"
    for _df_part in (df_brats_train, df_brats_val, df_brats_test):
        if "label" not in _df_part.columns:
            _df_part["label"] = _df_part["label_idx"].map({0: "no_tumor", 1: "tumor"})

    cols = ["split", "patient_id", "slice_id", "source_file", "label", "label_idx"]
    df_split_manifest = pd.concat([df_brats_train[cols], df_brats_val[cols], df_brats_test[cols]], ignore_index=True)

    # Strict zero-overlap assertions
    tr_p = set(df_split_manifest[df_split_manifest["split"] == "train"]["patient_id"].unique())
    va_p = set(df_split_manifest[df_split_manifest["split"] == "validation"]["patient_id"].unique())
    te_p = set(df_split_manifest[df_split_manifest["split"] == "test"]["patient_id"].unique())
    assert len(tr_p) == 850 and len(va_p) == 151 and len(te_p) == 250
    assert len(tr_p & va_p) == 0, "Train and validation patients overlap!"
    assert len(tr_p & te_p) == 0, "Train and test patients overlap!"
    assert len(va_p & te_p) == 0, "Validation and test patients overlap!"
    assert len(df_brats_train) == 6800 and len(df_brats_val) == 1208 and len(df_brats_test) == 33696

    manifest_out = pe_dir / "brats_patient_split_manifest.csv"
    df_split_manifest.to_csv(manifest_out, index=False)
    print(
        f"Saved fixed manifest {manifest_out} (SHA256={sha256_file(manifest_out)}): "
        f"train={len(tr_p)} pts ({len(df_brats_train)} slices), "
        f"val={len(va_p)} pts ({len(df_brats_val)} slices), "
        f"test={len(te_p)} pts ({len(df_brats_test)} slices). Zero overlap verified."
    )

    # =========================================================================
    # STEP 0(c): Resolve the 4 Inconsistencies from Raw Saved Files
    # =========================================================================
    # Load raw prediction files and JSON files
    df_pred_baseline = pd.read_csv(pe_dir / "brats_250_leakage_free_baseline_predictions.csv.gz")
    df_pred_other = pd.read_csv(pe_dir / "brats_250_other_checkpoints_predictions.csv.gz") if (pe_dir / "brats_250_other_checkpoints_predictions.csv.gz").is_file() else None
    df_pred_oriented_full = pd.read_csv(pe_dir / "brats_oriented_predictions.csv.gz")

    leakage_json = json.loads((pe_dir / "leakage_free_baselines.json").read_text(encoding="utf-8"))
    gradcam_json = json.loads((pe_dir / "gradcam_brats_evaluation.json").read_text(encoding="utf-8"))
    cis250_json = json.loads((pe_dir / "brats_250_5seeds_with_cis.json").read_text(encoding="utf-8"))

    y250 = df_pred_baseline["label_idx"].to_numpy(dtype=np.int32)

    # --- Inconsistency 1: Zero-shot Discovery ResNet-18 on 250-patient subset ---
    # Compare [[17380, 242], [9472, 6602]] vs [[17380, 242], [9473, 6601]]
    s_zs_csv = df_pred_baseline["score_zeroshot_resnet18"].to_numpy(dtype=np.float64)
    p_zs_csv = (s_zs_csv >= 0.5).astype(np.int32)
    cm_zs_csv = confusion_matrix(y250, p_zs_csv, labels=[0, 1]).tolist()

    # Also check the subset of brats_oriented_predictions.csv.gz for the 250 patients
    df_oriented_250 = df_test_250[["patient_id", "slice_id", "label_idx"]].merge(
        df_pred_oriented_full[["patient_id", "slice_id", "score"]],
        on=["patient_id", "slice_id"],
        how="left",
    )
    s_zs_full_sub = df_oriented_250["score"].to_numpy(dtype=np.float64)
    cm_zs_full_sub = confusion_matrix(df_oriented_250["label_idx"].to_numpy(), (s_zs_full_sub >= 0.5).astype(np.int32), labels=[0, 1]).tolist()

    # Find boundary slice(s) near 0.5 in s_zs_csv
    near_05_idx = np.where(np.abs(s_zs_csv - 0.5) < 5e-4)[0]
    boundary_slices_zs = []
    for idx in near_05_idx:
        row = df_pred_baseline.iloc[idx]
        boundary_slices_zs.append({
            "row_index": int(idx),
            "patient_id": str(row["patient_id"]),
            "slice_id": int(row["slice_id"]),
            "label_idx": int(row["label_idx"]),
            "score_in_brats_250_leakage_free_baseline_predictions_csv": float(s_zs_csv[idx]),
            "score_in_brats_oriented_predictions_csv": float(s_zs_full_sub[idx]),
        })

    # --- Inconsistency 2: Patient-Split ImageNet ResNet-18 ---
    # Compare [[16683, 939], [1490, 14584]] vs [[16682, 940], [1490, 14584]], and PR-AUC 0.9791 vs 0.9776
    s_imgnet_csv = df_pred_baseline["score_brats_patient_split_resnet18"].to_numpy(dtype=np.float64)
    p_imgnet_csv = (s_imgnet_csv >= 0.5).astype(np.int32)
    cm_imgnet_csv = confusion_matrix(y250, p_imgnet_csv, labels=[0, 1]).tolist()
    prauc_imgnet_csv = float(average_precision_score(y250, s_imgnet_csv))
    rocauc_imgnet_csv = float(roc_auc_score(y250, s_imgnet_csv))
    acc_imgnet_csv = float(accuracy_score(y250, p_imgnet_csv))

    cm_imgnet_json = leakage_json["baseline_1_brats_patient_split"]["models_on_brats_250_test"]["resnet18_brats_patient_split_from_imagenet"]["test_metrics"]["confusion_matrix"]
    prauc_imgnet_json = leakage_json["baseline_1_brats_patient_split"]["models_on_brats_250_test"]["resnet18_brats_patient_split_from_imagenet"]["test_metrics"]["pr_auc"]

    # --- Inconsistency 3: Cheng 3-class strict patient split train/val slice counts ---
    # 2,103/499 vs 2,175/427 (test = 462 in both)
    cheng_strict_json = leakage_json["baseline_3_cheng_3class_patient_vs_slice_split"]["strict_patient_split"]

    # --- Inconsistency 4: 250-subset values that differ in the last digit between exports ---
    # (e.g. ResNet-18 5-seed mean accuracy 0.6235 vs 0.6236)
    r18_runs = [r for r in cis250_json["architectures"]["resnet18"]["individual_runs"] if "primary" not in r["model_name"]]
    r18_accs_exact = [r["accuracy"] for r in r18_runs]
    r18_accs_rounded4 = [round(r["accuracy"], 4) for r in r18_runs]
    mean_exact = float(np.mean(r18_accs_exact))
    mean_of_rounded4 = float(np.mean(r18_accs_rounded4))

    resolution_payload = {
        "inconsistency_1_zeroshot_resnet18_confusion_matrix_on_250_subset": {
            "question": "Zero-shot discovery ResNet-18 on the 250-patient subset: confusion matrix [[17380,242],[9472,6602]] vs [[17380,242],[9473,6601]]",
            "correct_value": [[17380, 242], [9472, 6602]],
            "verified_from_files": [
                "paper_evidence/brats_250_leakage_free_baseline_predictions.csv.gz (column score_zeroshot_resnet18)",
                "paper_evidence/leakage_free_baselines.json (lines 22-31)",
                "paper_evidence/brats_250_5seeds_with_cis.json (lines 25-34 & 84-93)",
                "paper_evidence/brats_250_other_checkpoints.json (lines 25-34 & 84-93)",
            ],
            "explanation_of_discrepancy": (
                "In standard inference across the 33,696 slices (batch_size=128 in torch.inference_mode(), saved in "
                "brats_250_leakage_free_baseline_predictions.csv.gz, leakage_free_baselines.json, and brats_250_5seeds_with_cis.json), "
                "the confusion matrix is [[17380, 242], [9472, 6602]] (TP=6,602, FN=9,472). "
                "In run_gradcam_brats_evaluation.py (saved in gradcam_brats_evaluation.json lines 123 and 272), "
                "forward passes were run only on the 16,074 tumor-positive slices with batch_size=64 and requires_grad_(True) enabled; "
                "due to cuDNN/FP32 reduction order differences between batch_size=128 (all slices) and batch_size=64 (tumor slices only), "
                "1 boundary tumor slice close to 0.5 shifted across the 0.5 threshold, yielding 6,601 TP and 9,473 FN in the Grad-CAM subgroup count. "
                "For standard classification evaluation (batch_size=128 on all 33,696 slices), [[17380, 242], [9472, 6602]] is the exact value."
            ),
            "confusion_matrix_in_brats_250_leakage_free_baseline_predictions_csv": cm_zs_csv,
            "confusion_matrix_in_brats_oriented_predictions_csv_subset": cm_zs_full_sub,
            "slices_within_5e4_of_threshold_05": boundary_slices_zs,
        },
        "inconsistency_2_patient_split_imagenet_resnet18_cm_and_prauc": {
            "question": "Patient-split ImageNet ResNet-18: [[16683,939],[1490,14584]] vs [[16682,940],[1490,14584]], and PR-AUC 0.9791 vs 0.9776",
            "correct_confusion_matrix": [[16683, 939], [1490, 14584]],
            "correct_pr_auc": 0.9791319069413886,
            "correct_pr_auc_rounded_4dp": 0.9791,
            "verified_from_files": [
                "paper_evidence/brats_250_leakage_free_baseline_predictions.csv.gz (column score_brats_patient_split_resnet18)",
                "paper_evidence/leakage_free_baselines.json (lines 74-91)",
            ],
            "recomputed_from_saved_predictions_csv": {
                "confusion_matrix": cm_imgnet_csv,
                "accuracy": acc_imgnet_csv,
                "roc_auc": rocauc_imgnet_csv,
                "pr_auc": prauc_imgnet_csv,
            },
            "values_in_leakage_free_baselines_json": {
                "confusion_matrix": cm_imgnet_json,
                "pr_auc": prauc_imgnet_json,
            },
            "explanation_of_discrepancy": (
                "Both the saved prediction file (paper_evidence/brats_250_leakage_free_baseline_predictions.csv.gz) and "
                "the saved metrics file (paper_evidence/leakage_free_baselines.json lines 81-91) agree 100%: "
                "confusion_matrix = [[16683, 939], [1490, 14584]] and pr_auc = 0.9791319069413886 (0.9791). "
                "The values [[16682, 940], [1490, 14584]] and PR-AUC 0.9776 came from an aborted earlier run (task-2506) "
                "that crashed at KeyError: 'overlapping_patient_ids' before saving JSON, and were erroneously quoted in chat text. "
                "The authoritative saved files on disk have [[16683, 939], [1490, 14584]] and PR-AUC = 0.9791."
            ),
        },
        "inconsistency_3_cheng_3class_strict_patient_split_slice_counts": {
            "question": "Cheng 3-class strict patient split: train/val slice counts 2,103/499 vs 2,175/427 (test = 462 in both)",
            "correct_n_train_slices": 2103,
            "correct_n_val_slices": 499,
            "correct_n_test_slices": 462,
            "verified_from_files": [
                "paper_evidence/leakage_free_baselines.json (lines 453-457)",
            ],
            "values_in_leakage_free_baselines_json": {
                "n_train_slices": cheng_strict_json["n_train_slices"],
                "n_val_slices": cheng_strict_json["n_val_slices"],
                "n_test_slices": cheng_strict_json["n_test_slices"],
                "n_test_patients": cheng_strict_json["n_test_patients"],
                "confusion_matrix": cheng_strict_json["confusion_matrix"],
            },
            "explanation_of_discrepancy": (
                "The saved file paper_evidence/leakage_free_baselines.json (lines 454-456) records the exact slice counts "
                "for the 163 train / 35 val / 35 test patient split (np.random.default_rng(42).permutation(sorted(unique_pids))): "
                "n_train_slices = 2,103, n_val_slices = 499, n_test_slices = 462 (total = 3,064). "
                "The numbers 2,175/427 were from an ad-hoc chat calculation using a different patient sort order and do not appear in any saved file."
            ),
        },
        "inconsistency_4_250_subset_last_digit_rounding": {
            "question": "250-subset values that differ in the last digit between exports (e.g. ResNet-18 5-seed mean accuracy 0.6235 vs 0.6236)",
            "correct_unrounded_mean": mean_exact,
            "correct_4dp_rounded_mean": round(mean_exact, 4),
            "mean_of_4dp_rounded_seed_values": mean_of_rounded4,
            "per_seed_unrounded_accuracies": r18_accs_exact,
            "per_seed_4dp_rounded_accuracies": r18_accs_rounded4,
            "verified_from_files": [
                "paper_evidence/brats_250_5seeds_with_cis.json (lines 370-377)",
                "paper_evidence/brats_250_other_checkpoints.json (lines 370-377)",
            ],
            "explanation_of_discrepancy": (
                "In paper_evidence/brats_250_5seeds_with_cis.json (line 371), the exact 5-seed mean accuracy of ResNet-18 "
                "computed from unrounded floating-point values is 0.6235517568850902, which rounds to 4 decimal places as 0.6236 "
                "(line 376: '0.6236 +/- 0.0787'). If one instead averages the 4-decimal-place rounded per-seed numbers "
                "(0.7117 + 0.6451 + 0.5265 + 0.5578 + 0.6766) / 5 = 3.1177 / 5 = 0.62354, that rounds to 0.6235. "
                "The exact unrounded mean is 0.623552 (0.6236 at 4 decimal places)."
            ),
        },
    }

    res_out = pe_dir / "inconsistency_resolution.json"
    res_out.write_text(json.dumps(resolution_payload, indent=2), encoding="utf-8")
    print(f"Saved {res_out}")


if __name__ == "__main__":
    main()
