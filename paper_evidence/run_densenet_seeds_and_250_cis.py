"""
Step 1:
1. Train DenseNet-121 seeds 45 and 46 on Discovery using the exact Phase 1B protocol
   (experiments/phase1b_training.py), raising DenseNet-121 from 3 to 5 seeds (42, 43, 44, 45, 46).
2. Update the Phase 1B internal 5-seed summary across all 3 architectures (ResNet-18, SmallCNN, DenseNet-121)
   with 95% confidence intervals across seeds.
3. Evaluate all 17 checkpoints (ResNet-18 primary + seeds 42-46, SmallCNN primary + seeds 42-46,
   DenseNet-121 seeds 42-46) on the 250-patient BraTS subset (33,696 slices) and compute
   2,000-resample patient-cluster 95% bootstrap CIs (seed 42) for every checkpoint, every
   architecture multi-seed mean, and paired architecture differences.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, __version__ as pillow_version
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
from torch.utils.data import DataLoader, Dataset

from BrainTumorAI.classifier_pipeline import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    load_checkpoint,
    make_inference_transform,
    make_training_transform,
    save_checkpoint,
    seed_everything,
    sha256_file,
)
from experiments.phase1b_training import (
    EpochRunner,
    MANIFEST_PATH,
    PHASE1B_DIR,
    _compute_binary_metrics,
    _git_head,
    _save_learning_curves,
    _write_json,
    build_model,
    evaluate_model,
    read_manifest,
)


class CachedManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Caches decoded RGB PIL images in memory to avoid repeated disk I/O while keeping exact RNG transforms."""
    def __init__(self, rows: pd.DataFrame, image_cache: dict[str, Image.Image], training: bool = False) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_cache = image_cache
        self.transform = make_training_transform() if training else make_inference_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        img = self.image_cache[str(row["source_file"])]
        tensor = self.transform(img)
        return tensor, int(row["label_idx"])


def train_densenet_seed(seed: int, rows: pd.DataFrame, image_cache: dict[str, Image.Image]) -> dict[str, Any]:
    architecture = "densenet121"
    output_dir = PHASE1B_DIR / f"{architecture}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    ckpt_path = output_dir / "checkpoint.pth"
    if summary_path.is_file() and ckpt_path.is_file():
        print(f"Found existing {architecture} seed {seed} at {output_dir}, loading summary...", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"\nTraining {architecture} seed {seed} -> {output_dir} ...", flush=True)
    seed_everything(seed)
    train_rows = rows[rows["split"] == "train"].copy().reset_index(drop=True)
    val_rows = rows[rows["split"] == "validation"].copy().reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        CachedManifestDataset(train_rows, image_cache, training=True),
        batch_size=16,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        CachedManifestDataset(val_rows, image_cache, training=False),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(architecture, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history: list[dict[str, float | int]] = []
    best_balanced_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, 13):
        t_ep = time.perf_counter()
        train_metrics = EpochRunner(model, train_loader, criterion, device, optimizer).run()
        val_metrics = EpochRunner(model, val_loader, criterion, device).run()
        scheduler.step(val_metrics["balanced_accuracy"])
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "validation_loss": float(val_metrics["loss"]),
            "validation_accuracy": float(val_metrics["accuracy"]),
            "validation_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(record)
        score = val_metrics["balanced_accuracy"]
        loss = val_metrics["loss"]
        print(
            f"  [Seed {seed} | Epoch {epoch:02d}/12] train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={loss:.4f}, val_bal_acc={score:.4f} ({time.perf_counter()-t_ep:.1f}s)",
            flush=True,
        )
        improved = score > best_balanced_accuracy + 1e-4 or (
            abs(score - best_balanced_accuracy) <= 1e-4 and loss < best_validation_loss
        )
        if improved:
            best_balanced_accuracy = score
            best_validation_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                ckpt_path,
                architecture,
                {
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "selection_metric": "validation_balanced_accuracy",
                    "validation_balanced_accuracy": best_balanced_accuracy,
                    "validation_loss": best_validation_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "manifest_sha256": sha256_file(MANIFEST_PATH),
                    "code_revision": _git_head(),
                    "training_config": {
                        "architecture": architecture,
                        "optimizer": "Adam",
                        "learning_rate": 1e-4,
                        "weight_decay": 1e-4,
                        "batch_size": 16,
                        "max_epochs": 12,
                        "early_stopping_patience": 4,
                        "loss": "CrossEntropyLoss",
                        "class_weighting": False,
                        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2)",
                        "augmentation": ["random horizontal flip", "random rotation up to 10 degrees"],
                    },
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "torchvision": __import__("torchvision").__version__,
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "pillow": pillow_version,
                        "device": str(device),
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    },
                },
            )
            _write_json(
                output_dir / "checkpoint_metadata.json",
                {
                    "architecture": architecture,
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "selection_metric": "validation_balanced_accuracy",
                    "validation_balanced_accuracy": best_balanced_accuracy,
                    "validation_loss": best_validation_loss,
                    "threshold": 0.5,
                    "class_to_idx": CLASS_TO_IDX,
                },
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 4:
            print(f"  Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})", flush=True)
            break

    _write_json(output_dir / "history.json", {"epochs": history})
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    _save_learning_curves(history, output_dir)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(architecture, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions, metrics = evaluate_model(model, rows, split="test")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "architecture": architecture,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "best_validation_loss": best_validation_loss,
        "training_seconds": round(time.perf_counter() - start_time, 3),
        "metrics": metrics,
        "train_class_counts": {label: int((train_rows["label"] == label).sum()) for label in ("no_tumor", "tumor")},
        "validation_class_counts": {label: int((val_rows["label"] == label).sum()) for label in ("no_tumor", "tumor")},
        "code_revision": _git_head(),
    }
    _write_json(summary_path, summary)
    print(
        f"Finished {architecture} seed {seed} in {summary['training_seconds']}s: "
        f"Test Acc={metrics['accuracy']:.4f}, BalAcc={metrics['balanced_accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}",
        flush=True,
    )
    return summary


def load_any_checkpoint(ckpt_path: str | Path, device: torch.device) -> nn.Module:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = payload.get("architecture", "resnet18")
    if arch in ("densenet121", "efficientnet_b0"):
        model = build_model(arch, pretrained=False)
        model.load_state_dict(payload["model_state_dict"])
        return model.to(device).eval()
    else:
        model, _ = load_checkpoint(ckpt_path, map_location=device)
        return model.to(device).eval()


def fast_metrics_from_arrays(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    preds = (scores >= 0.5).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    auc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "auc": float(auc),
        "pr_auc": float(ap),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # -------------------------------------------------------------------------
    # Part 1: Train DenseNet-121 seeds 45 and 46 on Discovery (if not already trained)
    # -------------------------------------------------------------------------
    missing_seeds = [
        s for s in (45, 46)
        if not (PHASE1B_DIR / f"densenet121_seed{s}" / "checkpoint.pth").is_file()
        or not (PHASE1B_DIR / f"densenet121_seed{s}" / "summary.json").is_file()
    ]
    if missing_seeds:
        rows = read_manifest()
        print(f"Pre-caching {len(rows)} Discovery images in RAM for fast training...", flush=True)

        def _read_pil(rel_path: str) -> tuple[str, Image.Image]:
            p = ROOT / rel_path
            with Image.open(p) as im:
                return rel_path, im.convert("RGB")

        with ThreadPoolExecutor(max_workers=16) as ex:
            pairs = list(ex.map(_read_pil, rows["source_file"].astype(str).tolist()))
        image_cache = dict(pairs)
        print(f"Cached {len(image_cache)} Discovery images in {time.time() - t0:.2f}s", flush=True)

        for s in missing_seeds:
            train_densenet_seed(s, rows, image_cache)

        image_cache.clear()
        torch.cuda.empty_cache()
    else:
        print("DenseNet-121 seeds 45 and 46 already trained and saved.", flush=True)

    # Summarize all 5 seeds (42-46) on internal Discovery test set (N=606)
    internal_5seed_summary = {}
    for arch in ("resnet18", "small_cnn", "densenet121"):
        seed_metrics = []
        for s in (42, 43, 44, 45, 46):
            sum_file = PHASE1B_DIR / f"{arch}_seed{s}" / "summary.json"
            data = json.loads(sum_file.read_text(encoding="utf-8"))
            m = data["metrics"]
            seed_metrics.append({
                "seed": s,
                "best_epoch": data["best_epoch"],
                "best_validation_balanced_accuracy": data["best_validation_balanced_accuracy"],
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "average_precision": m["average_precision"],
            })
        agg = {}
        # t_crit for df=4 (n=5 seeds) at 95% two-sided confidence is 2.776445
        t_crit_df4 = 2.7764451051977987
        for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "average_precision"):
            vals = [x[k] for x in seed_metrics]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1))
            se_v = sd_v / np.sqrt(len(vals))
            ci_low = max(0.0, mean_v - t_crit_df4 * se_v)
            ci_high = min(1.0, mean_v + t_crit_df4 * se_v)
            agg[k] = {
                "mean": mean_v,
                "sd_sample": sd_v,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "ci_95_t_low": float(ci_low),
                "ci_95_t_high": float(ci_high),
                "mean_sd_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
                "ci_95_t_str": f"{ci_low:.4f} - {ci_high:.4f}",
            }
        internal_5seed_summary[arch] = {
            "n_seeds": 5,
            "seeds": [42, 43, 44, 45, 46],
            "per_seed": seed_metrics,
            "aggregate": agg,
        }

    with open(ROOT / "paper_evidence" / "phase1b_5seeds_internal_summary.json", "w", encoding="utf-8") as f:
        json.dump(internal_5seed_summary, f, indent=2)
    print("Saved paper_evidence/phase1b_5seeds_internal_summary.json", flush=True)

    # -------------------------------------------------------------------------
    # Part 2: Evaluate all 17 checkpoints on the 250-patient BraTS subset (33,696 slices)
    # -------------------------------------------------------------------------
    df_sub = pd.read_csv(ROOT / "paper_evidence" / "brats_250_patients_manifest.csv")
    total_len = len(df_sub)
    y_true = df_sub["label_idx"].to_numpy(dtype=np.int32)
    patients_arr = df_sub["patient_id"].to_numpy()
    unique_patients = np.array(sorted(df_sub["patient_id"].unique()))
    n_patients = len(unique_patients)
    print(f"\nPre-loading {total_len} BraTS 250-patient subset slices into 224x224 uint8 RAM buffer...", flush=True)

    root_dir = Path("D:/datasets/prepared/brats_oriented")
    file_paths = [str(root_dir / f) for f in df_sub["source_file"]]

    resize_op = make_inference_transform().transforms[0]  # Resize((224, 224))

    def _load_u8_224(path_str: str) -> np.ndarray:
        with Image.open(path_str) as im:
            rgb = resize_op(im.convert("RGB"))
            return np.asarray(rgb, dtype=np.uint8)[:, :, 0].copy()

    t_load = time.time()
    cache_npy = root_dir / "cache_250_u8_224.npy"
    if cache_npy.is_file():
        imgs_u8 = np.load(cache_npy)
    else:
        with ThreadPoolExecutor(max_workers=16) as ex:
            u8_list = list(ex.map(_load_u8_224, file_paths))
        imgs_u8 = np.stack(u8_list, axis=0)  # shape (33696, 224, 224), 1.69 GB
        del u8_list
        np.save(cache_npy, imgs_u8)
    print(f"Loaded {imgs_u8.shape} uint8 buffer ({imgs_u8.nbytes / 1e9:.2f} GB) in {time.time() - t_load:.2f}s", flush=True)

    mean_t = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    checkpoints = {
        "resnet18": [
            ("seed42 (primary reference)", "BrainTumorAI/models/tumor_classifier_reproducible.pth"),
            ("seed42 (phase1b)", "experiments_results/phase1b/resnet18_seed42/checkpoint.pth"),
            ("seed43", "experiments_results/phase1b/resnet18_seed43/checkpoint.pth"),
            ("seed44", "experiments_results/phase1b/resnet18_seed44/checkpoint.pth"),
            ("seed45", "experiments_results/phase1b/resnet18_seed45/checkpoint.pth"),
            ("seed46", "experiments_results/phase1b/resnet18_seed46/checkpoint.pth"),
        ],
        "smallcnn": [
            ("seed42 (primary)", "BrainTumorAI/models/small_cnn_best.pth"),
            ("seed42 (phase1b)", "experiments_results/phase1b/small_cnn_seed42/checkpoint.pth"),
            ("seed43", "experiments_results/phase1b/small_cnn_seed43/checkpoint.pth"),
            ("seed44", "experiments_results/phase1b/small_cnn_seed44/checkpoint.pth"),
            ("seed45", "experiments_results/phase1b/small_cnn_seed45/checkpoint.pth"),
            ("seed46", "experiments_results/phase1b/small_cnn_seed46/checkpoint.pth"),
        ],
        "densenet121": [
            ("seed42", "experiments_results/phase1b/densenet121_seed42/checkpoint.pth"),
            ("seed43", "experiments_results/phase1b/densenet121_seed43/checkpoint.pth"),
            ("seed44", "experiments_results/phase1b/densenet121_seed44/checkpoint.pth"),
            ("seed45", "experiments_results/phase1b/densenet121_seed45/checkpoint.pth"),
            ("seed46", "experiments_results/phase1b/densenet121_seed46/checkpoint.pth"),
        ],
    }

    existing_pred_csv = ROOT / "paper_evidence" / "brats_250_other_checkpoints_predictions.csv.gz"
    df_existing_preds = pd.read_csv(existing_pred_csv) if existing_pred_csv.is_file() else None

    all_scores: dict[tuple[str, str], np.ndarray] = {}
    for arch, ckpt_list in checkpoints.items():
        batch_sz = 64 if arch == "densenet121" else 128
        for name, ckpt_rel in ckpt_list:
            clean_col = f"score_{arch}_{name.replace(' ', '_').replace('(', '').replace(')', '')}"
            if df_existing_preds is not None and clean_col in df_existing_preds.columns and len(df_existing_preds) == total_len:
                scores = df_existing_preds[clean_col].to_numpy(dtype=np.float32)
                all_scores[(arch, name)] = scores
                m_now = fast_metrics_from_arrays(y_true, scores)
                print(
                    f"  [Cached | {arch:11s} | {name:26s}] Acc={m_now['accuracy']:.4f}, BalAcc={m_now['balanced_accuracy']:.4f}, "
                    f"Sens={m_now['sensitivity']:.4f}, Spec={m_now['specificity']:.4f}, AUC={m_now['auc']:.4f}",
                    flush=True,
                )
                continue
            t_m = time.time()
            model = load_any_checkpoint(ROOT / ckpt_rel, device)
            scores = np.zeros(total_len, dtype=np.float32)
            with torch.inference_mode():
                for start_i in range(0, total_len, batch_sz):
                    end_i = min(total_len, start_i + batch_sz)
                    batch_u8 = torch.from_numpy(imgs_u8[start_i:end_i]).to(device, non_blocking=True)
                    batch_fp = (batch_u8.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1) - mean_t) / std_t
                    probs = torch.softmax(model(batch_fp), dim=1)[:, 1].cpu().numpy()
                    scores[start_i:end_i] = probs
            all_scores[(arch, name)] = scores
            m_now = fast_metrics_from_arrays(y_true, scores)
            print(
                f"  [GPU    | {arch:11s} | {name:26s}] Acc={m_now['accuracy']:.4f}, BalAcc={m_now['balanced_accuracy']:.4f}, "
                f"Sens={m_now['sensitivity']:.4f}, Spec={m_now['specificity']:.4f}, AUC={m_now['auc']:.4f} ({time.time()-t_m:.1f}s)",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()

    del imgs_u8

    # -------------------------------------------------------------------------
    # Part 3: 2,000-Resample Patient-Cluster Bootstrap CIs (seed 42)
    # -------------------------------------------------------------------------
    print("\nRunning 2,000-resample patient-cluster bootstrap (seed 42) across all 17 checkpoints...", flush=True)
    t_boot = time.time()

    # Pre-index slices by patient for ultra-fast patient-cluster bootstrap
    patient_to_indices = {pid: np.where(patients_arr == pid)[0] for pid in unique_patients}
    pid_index_list = [patient_to_indices[pid] for pid in unique_patients]

    # Notice: for confusion-matrix metrics (acc, bal_acc, sens, spec, prec, f1), we can precompute
    # per-patient (tp, tn, fp, fn) counts for each checkpoint! Then resampling patients is just a matrix multiply!
    # And for AUC, we can compute patient-cluster bootstrap on the resampled slice arrays!
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_patient_Choices = rng.integers(0, n_patients, size=(n_boot, n_patients))

    # Precompute multinomial weights per patient for each bootstrap replicate: shape (n_boot, n_patients)
    patient_weights = np.zeros((n_boot, n_patients), dtype=np.int32)
    for b in range(n_boot):
        patient_weights[b] = np.bincount(boot_patient_Choices[b], minlength=n_patients)

    # Precompute resampled slice index arrays for AUC computation
    # To make AUC fast across 17 checkpoints x 2,000 replicates, we also compute exact patient-cluster
    # bootstrap for every checkpoint's confusion-matrix metrics AND AUC!
    boot_metrics_per_ckpt: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key in all_scores:
        boot_metrics_per_ckpt[key] = {
            m: np.zeros(n_boot, dtype=np.float64)
            for m in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc")
        }

    # 1. Exact vectorized bootstrap for confusion-matrix metrics using per-patient TP, TN, FP, FN
    for key, scores in all_scores.items():
        preds = (scores >= 0.5).astype(np.int32)
        p_tp = np.array([np.sum((preds[idx] == 1) & (y_true[idx] == 1)) for idx in pid_index_list], dtype=np.float64)
        p_tn = np.array([np.sum((preds[idx] == 0) & (y_true[idx] == 0)) for idx in pid_index_list], dtype=np.float64)
        p_fp = np.array([np.sum((preds[idx] == 1) & (y_true[idx] == 0)) for idx in pid_index_list], dtype=np.float64)
        p_fn = np.array([np.sum((preds[idx] == 0) & (y_true[idx] == 1)) for idx in pid_index_list], dtype=np.float64)

        b_tp = patient_weights @ p_tp
        b_tn = patient_weights @ p_tn
        b_fp = patient_weights @ p_fp
        b_fn = patient_weights @ p_fn

        b_tot = b_tp + b_tn + b_fp + b_fn
        b_acc = (b_tp + b_tn) / b_tot
        b_sens = b_tp / np.maximum(1.0, b_tp + b_fn)
        b_spec = b_tn / np.maximum(1.0, b_tn + b_fp)
        b_bal = 0.5 * (b_sens + b_spec)
        b_prec = b_tp / np.maximum(1.0, b_tp + b_fp)
        b_f1 = (2.0 * b_tp) / np.maximum(1.0, 2.0 * b_tp + b_fp + b_fn)

        boot_metrics_per_ckpt[key]["accuracy"] = b_acc
        boot_metrics_per_ckpt[key]["balanced_accuracy"] = b_bal
        boot_metrics_per_ckpt[key]["sensitivity"] = b_sens
        boot_metrics_per_ckpt[key]["specificity"] = b_spec
        boot_metrics_per_ckpt[key]["precision"] = b_prec
        boot_metrics_per_ckpt[key]["f1"] = b_f1

    # 2. Fast weighted ROC-AUC across the 2,000 replicates using sample_weight!
    # Notice: resampling patients with replacement is mathematically identical to passing
    # per-slice sample_weight = patient_weights[b, patient_idx_of_slice] to roc_auc_score!
    slice_patient_idx = np.searchsorted(unique_patients, patients_arr)
    for b in range(n_boot):
        w_slice = patient_weights[b, slice_patient_idx]
        mask_nz = w_slice > 0
        y_b = y_true[mask_nz]
        w_b = w_slice[mask_nz]
        for key, scores in all_scores.items():
            boot_metrics_per_ckpt[key]["auc"][b] = roc_auc_score(y_b, scores[mask_nz], sample_weight=w_b)
        if (b + 1) % 500 == 0:
            print(f"  Completed {b+1}/{n_boot} AUC bootstrap replicates ({time.time() - t_boot:.1f}s)", flush=True)

    # Assemble output structure
    results: dict[str, Any] = {
        "subset_info": {
            "n_patients": int(n_patients),
            "n_slices": int(total_len),
            "n_tumor_slices": int(np.sum(y_true == 1)),
            "n_no_tumor_slices": int(np.sum(y_true == 0)),
            "manifest_path": "paper_evidence/brats_250_patients_manifest.csv",
            "bootstrap_method": "Patient-cluster bootstrap (2,000 resamples of 250 patients with replacement, seed 42)",
        },
        "architectures": {},
        "paired_architecture_comparisons_5seed_mean": {},
    }

    t_crit_df4 = 2.7764451051977987  # df=4 for 5 seeds
    t_crit_df3 = 3.182446305284263   # df=3 for 4 seeds (43-46)

    arch_5seed_boot_means: dict[str, dict[str, np.ndarray]] = {}

    for arch, ckpt_list in checkpoints.items():
        arch_runs = []
        for name, ckpt_rel in ckpt_list:
            key = (arch, name)
            m_pt = fast_metrics_from_arrays(y_true, all_scores[key])
            b_dict = boot_metrics_per_ckpt[key]
            ci_dict = {}
            for m_name in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc"):
                low, high = np.percentile(b_dict[m_name], [2.5, 97.5])
                ci_dict[m_name] = {
                    "low": float(low),
                    "high": float(high),
                    "ci_95_str": f"{low:.4f} - {high:.4f}",
                }
            arch_runs.append({
                "model_name": name,
                "checkpoint": ckpt_rel,
                "accuracy": m_pt["accuracy"],
                "balanced_accuracy": m_pt["balanced_accuracy"],
                "sensitivity": m_pt["sensitivity"],
                "specificity": m_pt["specificity"],
                "precision": m_pt["precision"],
                "f1": m_pt["f1"],
                "auc": m_pt["auc"],
                "pr_auc": m_pt["pr_auc"],
                "confusion_matrix": [[m_pt["tn"], m_pt["fp"]], [m_pt["fn"], m_pt["tp"]]],
                "bootstrap_ci_95": ci_dict,
            })

        seed_runs_all = [r for r in arch_runs if "primary" not in r["model_name"]]
        seed_keys_all = [(arch, r["model_name"]) for r in seed_runs_all]
        seed_runs_43_46 = [r for r in arch_runs if r["model_name"] in ("seed43", "seed44", "seed45", "seed46")]
        seed_keys_43_46 = [(arch, r["model_name"]) for r in seed_runs_43_46]

        def summarize_group(runs: list[dict[str, Any]], keys: list[tuple[str, str]], t_crit: float) -> dict[str, Any]:
            out_s = {}
            for m_name in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc"):
                vals = [r[m_name] for r in runs]
                mean_v = float(np.mean(vals))
                sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                sd_pop = float(np.std(vals, ddof=0))
                se_v = sd_v / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                t_low = float(max(0.0, mean_v - t_crit * se_v))
                t_high = float(min(1.0, mean_v + t_crit * se_v))

                # Patient-cluster bootstrap distribution of the multi-seed mean
                boot_mean_arr = np.mean(np.stack([boot_metrics_per_ckpt[k][m_name] for k in keys], axis=0), axis=0)
                b_low, b_high = np.percentile(boot_mean_arr, [2.5, 97.5])

                out_s[m_name] = {
                    "mean": mean_v,
                    "sd_sample": sd_v,
                    "sd_pop": sd_pop,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean_sd_sample_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
                    "mean_sd_pop_str": f"{mean_v:.4f} +/- {sd_pop:.4f}",
                    "seed_t_ci_95": [t_low, t_high],
                    "seed_t_ci_95_str": f"{t_low:.4f} - {t_high:.4f}",
                    "patient_bootstrap_ci_95": [float(b_low), float(b_high)],
                    "patient_bootstrap_ci_95_str": f"{b_low:.4f} - {b_high:.4f}",
                }
            return out_s

        arch_5seed_boot_means[arch] = {
            m_name: np.mean(np.stack([boot_metrics_per_ckpt[k][m_name] for k in seed_keys_all], axis=0), axis=0)
            for m_name in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc")
        }

        results["architectures"][arch] = {
            "n_phase1b_seeds": len(seed_runs_all),
            "individual_runs": arch_runs,
            "summary_all_phase1b_seeds": summarize_group(seed_runs_all, seed_keys_all, t_crit_df4),
            "summary_seeds_43_to_46": summarize_group(seed_runs_43_46, seed_keys_43_46, t_crit_df3),
        }

    # Paired comparisons between architectures (5-seed mean) across the 250 patients
    for pair_name, (a1, a2) in [
        ("densenet121_minus_resnet18", ("densenet121", "resnet18")),
        ("resnet18_minus_smallcnn", ("resnet18", "smallcnn")),
        ("densenet121_minus_smallcnn", ("densenet121", "smallcnn")),
    ]:
        pair_out = {}
        for m_name in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc"):
            pt_diff = (
                results["architectures"][a1]["summary_all_phase1b_seeds"][m_name]["mean"]
                - results["architectures"][a2]["summary_all_phase1b_seeds"][m_name]["mean"]
            )
            diff_boot = arch_5seed_boot_means[a1][m_name] - arch_5seed_boot_means[a2][m_name]
            low, high = np.percentile(diff_boot, [2.5, 97.5])
            pair_out[m_name] = {
                "point_diff": float(pt_diff),
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "ci_95_str": f"{pt_diff:+.4f} ({low:+.4f} to {high:+.4f})",
            }
        results["paired_architecture_comparisons_5seed_mean"][pair_name] = pair_out

    out_path1 = ROOT / "paper_evidence" / "brats_250_other_checkpoints.json"
    out_path2 = ROOT / "paper_evidence" / "brats_250_5seeds_with_cis.json"
    out_path1.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_path2.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved complete 5-seed + CI results to {out_path1} and {out_path2} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

