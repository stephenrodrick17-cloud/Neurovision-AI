"""
Multi-Seed (Seeds 42-46) Training, Evaluation, Aggregation, and Grad-CAM Evaluation
for the Three Patient-Split BraTS Reference Models:
  - Model A: resnet18_brats_patient_split (ResNet-18 initialized from ImageNet, lr=1e-4)
  - Model B: resnet18_brats_finetuned (ResNet-18 fine-tuned from Discovery primary checkpoint, lr=5e-5)
  - Model C: small_cnn_brats_patient_split (SmallCNN trained from scratch, lr=1e-4)

Follows Steps 1, 2, 3, and 4 of the audit specification:
- Reuses existing seed 42 checkpoints as-is (copying to {model}_seed42.pth without modifying originals).
- Reads the fixed patient split from paper_evidence/brats_patient_split_manifest.csv (850 train / 151 val / 250 test patients)
  and asserts zero patient overlap.
- Trains seeds 43, 44, 45, 46 with identical hyperparameters, saving checkpoints to
  paper_evidence/models/{model}_seed{S}.pth and per-epoch training logs to paper_evidence/logs/{model}_seed{S}.json.
- Evaluates all 15 checkpoints on all 33,696 slices of the 250 held-out BraTS test patients (eval_batch_size=128)
  and on Cheng unscreened (3,064 slices, 233 patients) and Cheng overlap-screened (1,689 slices, 152 patients)
  with 2,000-resample patient-cluster bootstrap 95% CIs (np.random.default_rng(42)).
- Saves per-slice predictions (505,440 rows) to paper_evidence/brats_250_leakage_free_multiseed_predictions.csv.gz.
- Computes 5-seed aggregates (mean +/- sample SD ddof=1, min, max, 95% t-CI df=4, and patient-cluster bootstrap CI),
  paired differences of 5-seed mean predictions (A - zeroshot, B - zeroshot, C - zeroshot, A - B),
  and seed-wise paired differences A - B across seeds 42-46.
- Runs quantitative Grad-CAM evaluation on the 16,074 tumor-positive slices for seeds 43-46 of Models A and B
  (combining with seed 42 from gradcam_brats_evaluation.json) and saves paper_evidence/gradcam_brats_multiseed.json.
"""
from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    DEFAULT_CHECKPOINT,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    load_checkpoint,
    make_inference_transform,
    make_model,
    make_training_transform,
    save_checkpoint,
    seed_everything,
    sha256_file,
)
from BrainTumorAI.external_validation import _decode_cheng_mat
from paper_evidence.run_gradcam_brats_evaluation import (
    compute_batched_gradcam_7x7,
    patient_cluster_bootstrap_mean,
    summarize_slice_df,
)

T_CRIT_DF4 = 2.7764451051977987  # two-sided 95% t-critical value for df=4 (n=5 seeds)


class InMemoryImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, images: list[Image.Image], labels: list[int], training: bool = False) -> None:
        self.images = images
        self.labels = labels
        self.transform = make_training_transform() if training else make_inference_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), int(self.labels[idx])


def seed_all_strict(seed: int) -> dict[str, Any]:
    """Set python, numpy, torch, cuda seeds and deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed_everything(seed)
    det_alg_enabled = False
    det_note = (
        "random, numpy, torch CPU/CUDA, and DataLoader torch.Generator seeded with seed; "
        "torch.backends.cudnn.deterministic=True, torch.backends.cudnn.benchmark=False."
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        det_alg_enabled = True
        det_note += " torch.use_deterministic_algorithms(True, warn_only=True) enabled."
    except Exception as e:
        det_note += f" torch.use_deterministic_algorithms could not be set: {e}"
    return {
        "seed": seed,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "use_deterministic_algorithms_warn_only": det_alg_enabled,
        "determinism_note": det_note,
    }


def train_brats_model_seed(
    model_key: str,
    arch_name: str,
    init_type: str,
    lr: float,
    seed: int,
    train_images: list[Image.Image],
    train_labels: list[int],
    val_images: list[Image.Image],
    val_labels: list[int],
    device: torch.device,
    ckpt_path: Path,
    log_path: Path,
    manifest_sha256: str,
    max_epochs: int = 6,
    batch_size: int = 32,
) -> tuple[nn.Module, dict[str, Any]]:
    det_info = seed_all_strict(seed)

    # Initialize model AFTER seeding so any random weight initialization (e.g. Linear head or SmallCNN) uses `seed`
    if init_type == "imagenet":
        model = make_model("resnet18", pretrained=True)
    elif init_type == "discovery_finetune":
        model, _ = load_checkpoint(DEFAULT_CHECKPOINT, map_location="cpu")
    elif init_type == "scratch":
        model = make_model("small_cnn", pretrained=False)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    # Re-seed right before DataLoader creation so DataLoader shuffle order & augmentations match `train_binary_model`
    seed_everything(seed)
    train_loader = DataLoader(
        InMemoryImageDataset(train_images, train_labels, training=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        InMemoryImageDataset(val_images, val_labels, training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_bal_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        lr_before = float(optimizer.param_groups[0]["lr"])
        model.train()
        tr_loss = 0.0
        tr_preds, tr_targets = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item()) * y.size(0)
            tr_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            tr_targets.extend(y.detach().cpu().tolist())
        tr_loss /= len(train_images)
        tr_acc = float(accuracy_score(tr_targets, tr_preds))
        tr_bal = float(balanced_accuracy_score(tr_targets, tr_preds))

        model.eval()
        va_loss = 0.0
        va_preds, va_targets, va_scores = [], [], []
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                probs = torch.softmax(logits, dim=1)[:, 1]
                va_loss += float(loss.item()) * y.size(0)
                va_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
                va_targets.extend(y.detach().cpu().tolist())
                va_scores.extend(probs.detach().cpu().tolist())
        va_loss /= len(val_images)
        va_acc = float(accuracy_score(va_targets, va_preds))
        va_bal = float(balanced_accuracy_score(va_targets, va_preds))
        va_sens = float(recall_score(va_targets, va_preds, pos_label=1, zero_division=0))
        va_spec = float(recall_score(va_targets, va_preds, pos_label=0, zero_division=0))
        va_auc = float(roc_auc_score(va_targets, va_scores))
        va_ap = float(average_precision_score(va_targets, va_scores))
        scheduler.step(va_bal)

        ep_record = {
            "epoch": epoch,
            "learning_rate": lr_before,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "train_balanced_accuracy": tr_bal,
            "val_loss": va_loss,
            "val_accuracy": va_acc,
            "val_balanced_accuracy": va_bal,
            "val_sensitivity": va_sens,
            "val_specificity": va_spec,
            "val_roc_auc": va_auc,
            "val_pr_auc": va_ap,
            "epoch_seconds": round(time.time() - ep_t0, 2),
        }
        history.append(ep_record)
        print(
            f"    [{model_key} | seed {seed} | Epoch {epoch:02d}/{max_epochs}] "
            f"tr_loss={tr_loss:.4f}, tr_bal={tr_bal:.4f}, va_loss={va_loss:.4f}, va_bal={va_bal:.4f}, va_auc={va_auc:.4f}",
            flush=True,
        )

        if va_bal > best_bal_acc + 1e-4 or (abs(va_bal - best_bal_acc) <= 1e-4 and va_loss < best_val_loss):
            best_bal_acc = va_bal
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 2:
            print(f"    Early stopping triggered at epoch {epoch} (selected epoch {best_epoch})", flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()
    torch.cuda.empty_cache()

    train_secs = round(time.time() - t0, 2)
    meta_payload = {
        "model_key": model_key,
        "architecture": arch_name,
        "initialization": init_type,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selection_metric": "validation_balanced_accuracy",
        "validation_balanced_accuracy": best_bal_acc,
        "validation_loss": best_val_loss,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": 2,
        "manifest_sha256": manifest_sha256,
        "training_seconds": train_secs,
        "determinism": det_info,
    }
    save_checkpoint(model, ckpt_path, arch_name, meta_payload)

    log_payload = {
        **meta_payload,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "history": history,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    return model, log_payload


def compute_binary_metrics_and_cluster_bootstrap(
    y_true: np.ndarray,
    scores: np.ndarray,
    unique_patients: np.ndarray,
    pid_index_list: list[np.ndarray],
    patient_weights: np.ndarray,
    slice_patient_idx: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    preds = (scores >= 0.5).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tot = tp + tn + fp + fn

    acc = (tp + tn) / tot
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    bal_acc = 0.5 * (sens + spec)
    prec = tp / max(1, tp + fp)
    f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    auc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))

    n_boot = patient_weights.shape[0]
    p_tp = np.array([np.sum((preds[idx] == 1) & (y_true[idx] == 1)) for idx in pid_index_list], dtype=np.float64)
    p_tn = np.array([np.sum((preds[idx] == 0) & (y_true[idx] == 0)) for idx in pid_index_list], dtype=np.float64)
    p_fp = np.array([np.sum((preds[idx] == 1) & (y_true[idx] == 0)) for idx in pid_index_list], dtype=np.float64)
    p_fn = np.array([np.sum((preds[idx] == 0) & (y_true[idx] == 1)) for idx in pid_index_list], dtype=np.float64)

    b_tp = patient_weights @ p_tp
    b_tn = patient_weights @ p_tn
    b_fp = patient_weights @ p_fp
    b_fn = patient_weights @ p_fn
    b_tot = b_tp + b_tn + b_fp + b_fn

    boot_dist = {
        "accuracy": (b_tp + b_tn) / b_tot,
        "sensitivity": b_tp / np.maximum(1.0, b_tp + b_fn),
        "specificity": b_tn / np.maximum(1.0, b_tn + b_fp),
        "precision": b_tp / np.maximum(1.0, b_tp + b_fp),
        "f1": (2.0 * b_tp) / np.maximum(1.0, 2.0 * b_tp + b_fp + b_fn),
        "roc_auc": np.zeros(n_boot, dtype=np.float64),
        "pr_auc": np.zeros(n_boot, dtype=np.float64),
    }
    boot_dist["balanced_accuracy"] = 0.5 * (boot_dist["sensitivity"] + boot_dist["specificity"])

    for b in range(n_boot):
        w_slice = patient_weights[b, slice_patient_idx]
        mask_nz = w_slice > 0
        yb = y_true[mask_nz]
        sb = scores[mask_nz]
        wb = w_slice[mask_nz]
        boot_dist["roc_auc"][b] = roc_auc_score(yb, sb, sample_weight=wb)
        boot_dist["pr_auc"][b] = average_precision_score(yb, sb, sample_weight=wb)

    ci_95 = {}
    for k, arr in boot_dist.items():
        low, high = np.percentile(arr, [2.5, 97.5])
        ci_95[k] = {
            "low": float(low),
            "high": float(high),
            "ci_95_str": f"{low:.4f} - {high:.4f}",
        }

    summary = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "bootstrap_ci_95": ci_95,
    }
    return summary, boot_dist


def main() -> None:
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    pe_dir = ROOT / "paper_evidence"
    models_dir = pe_dir / "models"
    logs_dir = pe_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # VERIFY FIXED PATIENT-SPLIT MANIFEST (STEP 0b / STEP 1)
    # =========================================================================
    manifest_path = pe_dir / "brats_patient_split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}. Run paper_evidence/run_step0_audit.py first.")
    manifest_sha = sha256_file(manifest_path)
    df_manifest = pd.read_csv(manifest_path)

    df_train = df_manifest[df_manifest["split"] == "train"].copy().reset_index(drop=True)
    df_val = df_manifest[df_manifest["split"] == "validation"].copy().reset_index(drop=True)
    df_test = df_manifest[df_manifest["split"] == "test"].copy().reset_index(drop=True)

    tr_pids = set(df_train["patient_id"].unique())
    va_pids = set(df_val["patient_id"].unique())
    te_pids = set(df_test["patient_id"].unique())
    assert len(tr_pids) == 850 and len(df_train) == 6800
    assert len(va_pids) == 151 and len(df_val) == 1208
    assert len(te_pids) == 250 and len(df_test) == 33696
    assert len(tr_pids & va_pids) == 0
    assert len(tr_pids & te_pids) == 0
    assert len(va_pids & te_pids) == 0
    print(
        f"Verified fixed manifest {manifest_path.name} (SHA256={manifest_sha[:16]}...): "
        f"850 train pts (6,800 slices), 151 val pts (1,208 slices), 250 test pts (33,696 slices). Zero patient overlap.",
        flush=True,
    )

    # =========================================================================
    # STEP 1: REUSE SEED 42 & TRAIN SEEDS 43, 44, 45, 46 FOR MODELS A, B, C
    # =========================================================================
    model_specs = [
        {
            "code": "A",
            "model_key": "resnet18_brats_patient_split",
            "arch_name": "resnet18",
            "init_type": "imagenet",
            "lr": 1e-4,
            "seed42_orig_ckpt": models_dir / "resnet18_brats_patient_split.pth",
        },
        {
            "code": "B",
            "model_key": "resnet18_brats_finetuned",
            "arch_name": "resnet18",
            "init_type": "discovery_finetune",
            "lr": 5e-5,
            "seed42_orig_ckpt": models_dir / "resnet18_brats_finetuned.pth",
        },
        {
            "code": "C",
            "model_key": "small_cnn_brats_patient_split",
            "arch_name": "small_cnn",
            "init_type": "scratch",
            "lr": 1e-4,
            "seed42_orig_ckpt": models_dir / "small_cnn_brats_patient_split.pth",
        },
    ]
    seeds = [42, 43, 44, 45, 46]

    # Ensure seed 42 checkpoints are copied to {model}_seed42.pth (preserving originals untouched)
    for spec in model_specs:
        mkey = spec["model_key"]
        orig_p = spec["seed42_orig_ckpt"]
        s42_p = models_dir / f"{mkey}_seed42.pth"
        if not s42_p.is_file():
            shutil.copy2(orig_p, s42_p)
        log42_p = logs_dir / f"{mkey}_seed42.json"
        if not log42_p.is_file():
            _, meta42 = load_checkpoint(orig_p, map_location="cpu")
            log42_payload = {
                "model_key": mkey,
                "architecture": spec["arch_name"],
                "initialization": spec["init_type"],
                "seed": 42,
                "selected_epoch": meta42.get("selected_epoch"),
                "selection_metric": "validation_balanced_accuracy",
                "validation_balanced_accuracy": meta42.get("validation_balanced_accuracy"),
                "validation_loss": meta42.get("validation_loss"),
                "learning_rate": spec["lr"],
                "batch_size": 32,
                "max_epochs": 6,
                "early_stopping_patience": 2,
                "manifest_sha256": manifest_sha,
                "reused_from_checkpoint": str(orig_p.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_path": str(s42_p.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": sha256_file(s42_p),
                "history": "NOT VERIFIED (seed 42 was trained in task-2518 and reused as-is; checkpoint metadata stores selected_epoch, validation_balanced_accuracy, and validation_loss)",
            }
            log42_p.write_text(json.dumps(log42_payload, indent=2), encoding="utf-8")

    # Check if any seeds 43..46 need training
    missing_jobs = []
    for spec in model_specs:
        mkey = spec["model_key"]
        for s in (43, 44, 45, 46):
            ckpt_p = models_dir / f"{mkey}_seed{s}.pth"
            log_p = logs_dir / f"{mkey}_seed{s}.json"
            if not ckpt_p.is_file() or not log_p.is_file():
                missing_jobs.append((spec, s, ckpt_p, log_p))

    brats_root = Path("D:/datasets/prepared/brats_oriented")

    def _load_rgb_pil(path_str: str) -> Image.Image:
        with Image.open(path_str) as im:
            return im.convert("RGB")

    if missing_jobs:
        print(f"\nLoading 6,800 train and 1,208 val BraTS images into RAM for {len(missing_jobs)} training runs...", flush=True)
        t_io = time.time()
        with ThreadPoolExecutor(max_workers=16) as ex:
            brats_train_imgs = list(ex.map(_load_rgb_pil, [str(brats_root / f) for f in df_train["source_file"]]))
            brats_val_imgs = list(ex.map(_load_rgb_pil, [str(brats_root / f) for f in df_val["source_file"]]))
        brats_train_lbls = df_train["label_idx"].astype(int).tolist()
        brats_val_lbls = df_val["label_idx"].astype(int).tolist()
        print(f"Loaded train/val images in {time.time() - t_io:.2f}s", flush=True)

        for spec, s, ckpt_p, log_p in missing_jobs:
            print(f"\n--- Training Model {spec['code']} ({spec['model_key']}) | Seed {s} ---", flush=True)
            train_brats_model_seed(
                model_key=spec["model_key"],
                arch_name=spec["arch_name"],
                init_type=spec["init_type"],
                lr=spec["lr"],
                seed=s,
                train_images=brats_train_imgs,
                train_labels=brats_train_lbls,
                val_images=brats_val_imgs,
                val_labels=brats_val_lbls,
                device=device,
                ckpt_path=ckpt_p,
                log_path=log_p,
                manifest_sha256=manifest_sha,
            )

        del brats_train_imgs, brats_val_imgs
        torch.cuda.empty_cache()
    else:
        print("\nAll 15 checkpoints (seeds 42-46 x 3 models) and logs already exist on disk.", flush=True)

    # =========================================================================
    # STEP 2: EVALUATE ALL 15 CHECKPOINTS ON BraTS-250 (33,696 SLICES) & CHENG
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("STEP 2: EVALUATING ALL 15 CHECKPOINTS ON BraTS-250 (33,696 SLICES)", flush=True)
    print("======================================================================", flush=True)

    eval_batch_size = 128
    mean_t = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    resize_op = make_inference_transform().transforms[0]

    def _load_u8_224(path_str: str) -> np.ndarray:
        with Image.open(path_str) as im:
            rgb = resize_op(im.convert("RGB"))
            return np.asarray(rgb, dtype=np.uint8)[:, :, 0].copy()

    cache_npy = brats_root / "cache_250_u8_224.npy"
    if cache_npy.is_file():
        test250_u8 = np.load(cache_npy)
    else:
        with ThreadPoolExecutor(max_workers=16) as ex:
            test250_u8 = np.stack(list(ex.map(_load_u8_224, [str(brats_root / f) for f in df_test["source_file"]])), axis=0)
        np.save(cache_npy, test250_u8)

    def eval_on_u8_buffer(model_obj: nn.Module, buf_u8: np.ndarray, batch_sz: int = eval_batch_size) -> np.ndarray:
        model_obj = model_obj.to(device)
        model_obj.eval()
        out_scores = np.zeros(len(buf_u8), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(buf_u8), batch_sz):
                j = min(len(buf_u8), i + batch_sz)
                b_u8 = torch.from_numpy(buf_u8[i:j]).to(device, non_blocking=True)
                b_fp = (b_u8.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1) - mean_t) / std_t
                out_scores[i:j] = torch.softmax(model_obj(b_fp), dim=1)[:, 1].cpu().numpy()
        model_obj.cpu()
        torch.cuda.empty_cache()
        return out_scores

    y_test250 = df_test["label_idx"].to_numpy(dtype=np.int32)
    p_test250 = df_test["patient_id"].to_numpy()
    s_test250 = df_test["slice_id"].to_numpy(dtype=np.int32)

    unique_patients = np.array(sorted(np.unique(p_test250)))
    n_patients = len(unique_patients)
    pid_index_list = [np.where(p_test250 == pid)[0] for pid in unique_patients]
    slice_patient_idx = np.searchsorted(unique_patients, p_test250)

    n_boot = 2000
    rng_boot = np.random.default_rng(42)
    boot_choices = rng_boot.integers(0, n_patients, size=(n_boot, n_patients))
    patient_weights = np.zeros((n_boot, n_patients), dtype=np.int32)
    for b in range(n_boot):
        patient_weights[b] = np.bincount(boot_choices[b], minlength=n_patients)

    # Load seed 42 cached scores from brats_250_leakage_free_baseline_predictions.csv.gz to guarantee exact reuse
    df_seed42_cached = pd.read_csv(pe_dir / "brats_250_leakage_free_baseline_predictions.csv.gz")
    scores_zeroshot = df_seed42_cached["score_zeroshot_resnet18"].to_numpy(dtype=np.float32)
    seed42_score_map = {
        "resnet18_brats_patient_split": df_seed42_cached["score_brats_patient_split_resnet18"].to_numpy(dtype=np.float32),
        "resnet18_brats_finetuned": df_seed42_cached["score_brats_finetuned_resnet18"].to_numpy(dtype=np.float32),
        "small_cnn_brats_patient_split": df_seed42_cached["score_brats_patient_split_smallcnn"].to_numpy(dtype=np.float32),
    }

    res_zeroshot, boot_zeroshot = compute_binary_metrics_and_cluster_bootstrap(
        y_test250, scores_zeroshot, unique_patients, pid_index_list, patient_weights, slice_patient_idx
    )

    scores_by_model_seed: dict[tuple[str, int], np.ndarray] = {}
    metrics_by_model_seed: dict[tuple[str, int], dict[str, Any]] = {}
    boot_by_model_seed: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    pred_dfs: list[pd.DataFrame] = []

    for spec in model_specs:
        mkey = spec["model_key"]
        for s in seeds:
            ckpt_p = models_dir / f"{mkey}_seed{s}.pth"
            if s == 42 and mkey in seed42_score_map:
                sc = seed42_score_map[mkey]
            else:
                model_obj, _ = load_checkpoint(ckpt_p, map_location="cpu")
                sc = eval_on_u8_buffer(model_obj, test250_u8, batch_sz=eval_batch_size)
                del model_obj
            scores_by_model_seed[(mkey, s)] = sc
            m_sum, b_dist = compute_binary_metrics_and_cluster_bootstrap(
                y_test250, sc, unique_patients, pid_index_list, patient_weights, slice_patient_idx
            )
            log_data = json.loads((logs_dir / f"{mkey}_seed{s}.json").read_text(encoding="utf-8"))
            m_sum["checkpoint_path"] = str(ckpt_p.relative_to(ROOT)).replace("\\", "/")
            m_sum["checkpoint_sha256"] = sha256_file(ckpt_p)
            m_sum["selected_epoch"] = log_data.get("selected_epoch")
            m_sum["validation_balanced_accuracy"] = log_data.get("validation_balanced_accuracy")
            m_sum["validation_loss"] = log_data.get("validation_loss")
            m_sum["eval_batch_size"] = eval_batch_size
            metrics_by_model_seed[(mkey, s)] = m_sum
            boot_by_model_seed[(mkey, s)] = b_dist

            print(
                f"  [{mkey:30s} | seed {s}] Acc={m_sum['accuracy']:.4f} ({m_sum['bootstrap_ci_95']['accuracy']['ci_95_str']}), "
                f"BalAcc={m_sum['balanced_accuracy']:.4f}, Sens={m_sum['sensitivity']:.4f}, Spec={m_sum['specificity']:.4f}, "
                f"AUC={m_sum['roc_auc']:.4f} ({m_sum['bootstrap_ci_95']['roc_auc']['ci_95_str']}), PR-AUC={m_sum['pr_auc']:.4f}",
                flush=True,
            )

            pred_dfs.append(
                pd.DataFrame({
                    "model": mkey,
                    "seed": s,
                    "patient_id": p_test250,
                    "slice_id": s_test250,
                    "label": y_test250,
                    "score": sc,
                })
            )

    # Save 15-checkpoint per-slice predictions (columns: model, seed, patient_id, slice_id, label, score)
    multiseed_pred_path = pe_dir / "brats_250_leakage_free_multiseed_predictions.csv.gz"
    df_all_preds = pd.concat(pred_dfs, ignore_index=True)
    df_all_preds.to_csv(multiseed_pred_path, index=False, compression="gzip")
    multiseed_pred_sha256 = sha256_file(multiseed_pred_path)
    print(f"\nSaved {len(df_all_preds)} prediction rows to {multiseed_pred_path} (SHA256={multiseed_pred_sha256})", flush=True)
    del df_all_preds, pred_dfs

    # =========================================================================
    # EVALUATE ALL 15 CHECKPOINTS ON CHENG UNSCREENED (3,064) & SCREENED (1,689)
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("STEP 2B: EVALUATING ALL 15 CHECKPOINTS ON CHENG (3,064 & 1,689 SLICES)", flush=True)
    print("======================================================================", flush=True)

    cheng_zip_path = ROOT / "1512427.zip"
    cheng_records = []
    cheng_u8_224_list: list[np.ndarray] = []
    part_names = [
        "brainTumorDataPublic_1-766.zip",
        "brainTumorDataPublic_767-1532.zip",
        "brainTumorDataPublic_1533-2298.zip",
        "brainTumorDataPublic_2299-3064.zip",
    ]
    with zipfile.ZipFile(cheng_zip_path, "r") as outer_z:
        for pname in part_names:
            with outer_z.open(pname) as part_stream:
                with zipfile.ZipFile(part_stream, "r") as inner_z:
                    mat_names = sorted(
                        [n for n in inner_z.namelist() if n.lower().endswith(".mat")],
                        key=lambda x: int(Path(x).stem),
                    )
                    for mname in mat_names:
                        raw_bytes = inner_z.read(mname)
                        img_arr, pid, label_num = _decode_cheng_mat(raw_bytes, f"{pname}!{mname}")
                        mn, mx = float(img_arr.min()), float(img_arr.max())
                        scaled = np.rint((img_arr.astype(np.float64) - mn) * (255.0 / (mx - mn)))
                        pil_im = Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), mode="L").convert("RGB")
                        rgb_224 = resize_op(pil_im)
                        cheng_u8_224_list.append(np.asarray(rgb_224, dtype=np.uint8)[:, :, 0].copy())
                        cheng_records.append({
                            "cheng_slice": f"{pname}!{mname}",
                            "patient_id": str(pid),
                            "subtype_idx": int(label_num) - 1,
                        })

    df_cheng = pd.DataFrame(cheng_records)
    cheng_u8 = np.stack(cheng_u8_224_list, axis=0)
    del cheng_u8_224_list

    overlap_audit = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8")
    )
    overlapping_cheng_pids = {edge["patient_id"] for edge in overlap_audit["matches"]["high_similarity_matches"]}
    screened_mask = ~df_cheng["patient_id"].isin(overlapping_cheng_pids).to_numpy()
    p_cheng_all = df_cheng["patient_id"].to_numpy()
    p_cheng_scr = p_cheng_all[screened_mask]

    uniq_ch_all = np.array(sorted(np.unique(p_cheng_all)))
    uniq_ch_scr = np.array(sorted(np.unique(p_cheng_scr)))
    assert len(uniq_ch_all) == 233 and len(p_cheng_all) == 3064
    assert len(uniq_ch_scr) == 152 and len(p_cheng_scr) == 1689

    rng_ch_all = np.random.default_rng(42)
    w_ch_all = np.zeros((2000, len(uniq_ch_all)), dtype=np.int32)
    for b in range(2000):
        w_ch_all[b] = np.bincount(rng_ch_all.integers(0, len(uniq_ch_all), size=len(uniq_ch_all)), minlength=len(uniq_ch_all))

    rng_ch_scr = np.random.default_rng(42)
    w_ch_scr = np.zeros((2000, len(uniq_ch_scr)), dtype=np.int32)
    for b in range(2000):
        w_ch_scr[b] = np.bincount(rng_ch_scr.integers(0, len(uniq_ch_scr), size=len(uniq_ch_scr)), minlength=len(uniq_ch_scr))

    def cheng_sens_with_ci(scores_arr: np.ndarray, pids_arr: np.ndarray, uniq_p: np.ndarray, weights_mat: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        preds = (scores_arr >= 0.5).astype(np.int32)
        sens = float(np.mean(preds == 1))
        p_hits = np.array([np.sum(preds[pids_arr == pid] == 1) for pid in uniq_p], dtype=np.float64)
        p_cnts = np.array([np.sum(pids_arr == pid) for pid in uniq_p], dtype=np.float64)
        boot_s = (weights_mat @ p_hits) / (weights_mat @ p_cnts)
        lo, hi = np.percentile(boot_s, [2.5, 97.5])
        return {
            "n_slices": int(len(scores_arr)),
            "n_patients": int(len(uniq_p)),
            "sensitivity": sens,
            "mean_score": float(np.mean(scores_arr)),
            "ci_95_low": float(lo),
            "ci_95_high": float(hi),
            "ci_95_str": f"{lo:.4f} - {hi:.4f}",
        }, boot_s

    cheng_by_model_seed: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in model_specs:
        mkey = spec["model_key"]
        for s in seeds:
            ckpt_p = models_dir / f"{mkey}_seed{s}.pth"
            model_obj, _ = load_checkpoint(ckpt_p, map_location="cpu")
            c_scores = eval_on_u8_buffer(model_obj, cheng_u8, batch_sz=64)
            del model_obj
            unscr_res, _ = cheng_sens_with_ci(c_scores, p_cheng_all, uniq_ch_all, w_ch_all)
            scr_res, _ = cheng_sens_with_ci(c_scores[screened_mask], p_cheng_scr, uniq_ch_scr, w_ch_scr)
            cheng_by_model_seed[(mkey, s)] = {
                "unscreened_3064": unscr_res,
                "overlap_screened_1689": scr_res,
            }
            metrics_by_model_seed[(mkey, s)]["cheng_unscreened_3064"] = unscr_res
            metrics_by_model_seed[(mkey, s)]["cheng_overlap_screened_1689"] = scr_res
            print(
                f"  [Cheng | {mkey:30s} | seed {s}] Unscreened Sens={unscr_res['sensitivity']:.4f} ({unscr_res['ci_95_str']}), "
                f"Screened Sens={scr_res['sensitivity']:.4f} ({scr_res['ci_95_str']})",
                flush=True,
            )
    del cheng_u8

    # =========================================================================
    # STEP 3: AGGREGATE ACROSS 5 SEEDS & PAIRED COMPARISONS
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("STEP 3: 5-SEED AGGREGATION & PAIRED COMPARISONS", flush=True)
    print("======================================================================", flush=True)

    metric_names = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc"]
    per_model_results: dict[str, Any] = {}
    mean_scores_by_model: dict[str, np.ndarray] = {}
    mean_boot_by_model: dict[str, dict[str, np.ndarray]] = {}
    ensemble_res_by_model: dict[str, dict[str, Any]] = {}
    ensemble_boot_by_model: dict[str, dict[str, np.ndarray]] = {}

    summary_csv_rows = []

    for spec in model_specs:
        mkey = spec["model_key"]
        mcode = spec["code"]

        # Ensemble (mean predicted probability across the 5 seeds for each slice)
        ens_scores = np.mean(np.stack([scores_by_model_seed[(mkey, s)] for s in seeds], axis=0), axis=0)
        mean_scores_by_model[mkey] = ens_scores
        ens_res, ens_boot = compute_binary_metrics_and_cluster_bootstrap(
            y_test250, ens_scores, unique_patients, pid_index_list, patient_weights, slice_patient_idx
        )
        ensemble_res_by_model[mkey] = ens_res
        ensemble_boot_by_model[mkey] = ens_boot

        # 5-seed average of bootstrap distributions (mean of the 5 seed metrics per bootstrap replicate)
        mean_boot_by_model[mkey] = {}
        agg_metrics = {}
        for m_name in metric_names:
            vals = [metrics_by_model_seed[(mkey, s)][m_name] for s in seeds]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1))
            se_v = sd_v / np.sqrt(len(vals))
            t_lo = float(max(0.0, mean_v - T_CRIT_DF4 * se_v))
            t_hi = float(min(1.0, mean_v + T_CRIT_DF4 * se_v))

            b_stack = np.mean(np.stack([boot_by_model_seed[(mkey, s)][m_name] for s in seeds], axis=0), axis=0)
            mean_boot_by_model[mkey][m_name] = b_stack
            b_lo, b_hi = np.percentile(b_stack, [2.5, 97.5])

            agg_metrics[m_name] = {
                "mean": mean_v,
                "sd_sample": sd_v,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "seed_values": {str(s): float(metrics_by_model_seed[(mkey, s)][m_name]) for s in seeds},
                "mean_sd_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
                "t_ci_95_df4": [t_lo, t_hi],
                "t_ci_95_str": f"{t_lo:.4f} - {t_hi:.4f}",
                "patient_bootstrap_ci_95": [float(b_lo), float(b_hi)],
                "patient_bootstrap_ci_95_str": f"{b_lo:.4f} - {b_hi:.4f}",
            }

        # Also aggregate Cheng unscreened & screened sensitivity across 5 seeds
        for ch_key in ("cheng_unscreened_3064", "cheng_overlap_screened_1689"):
            vals = [metrics_by_model_seed[(mkey, s)][ch_key]["sensitivity"] for s in seeds]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1))
            se_v = sd_v / np.sqrt(len(vals))
            t_lo = float(max(0.0, mean_v - T_CRIT_DF4 * se_v))
            t_hi = float(min(1.0, mean_v + T_CRIT_DF4 * se_v))
            agg_metrics[f"{ch_key}_sensitivity"] = {
                "mean": mean_v,
                "sd_sample": sd_v,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "seed_values": {str(s): float(metrics_by_model_seed[(mkey, s)][ch_key]["sensitivity"]) for s in seeds},
                "mean_sd_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
                "t_ci_95_df4": [t_lo, t_hi],
                "t_ci_95_str": f"{t_lo:.4f} - {t_hi:.4f}",
            }

        per_model_results[mkey] = {
            "model_code": mcode,
            "architecture": spec["arch_name"],
            "initialization": spec["init_type"],
            "learning_rate": spec["lr"],
            "seeds": seeds,
            "per_seed": {str(s): metrics_by_model_seed[(mkey, s)] for s in seeds},
            "aggregate_5seed": agg_metrics,
            "ensemble_5seed_mean_score_metrics": ens_res,
        }

        for s in seeds:
            ms = metrics_by_model_seed[(mkey, s)]
            summary_csv_rows.append({
                "row_type": "seed",
                "model_code": mcode,
                "model": mkey,
                "seed": str(s),
                "accuracy": ms["accuracy"],
                "accuracy_ci_low": ms["bootstrap_ci_95"]["accuracy"]["low"],
                "accuracy_ci_high": ms["bootstrap_ci_95"]["accuracy"]["high"],
                "balanced_accuracy": ms["balanced_accuracy"],
                "sensitivity": ms["sensitivity"],
                "specificity": ms["specificity"],
                "precision": ms["precision"],
                "f1": ms["f1"],
                "roc_auc": ms["roc_auc"],
                "roc_auc_ci_low": ms["bootstrap_ci_95"]["roc_auc"]["low"],
                "roc_auc_ci_high": ms["bootstrap_ci_95"]["roc_auc"]["high"],
                "pr_auc": ms["pr_auc"],
                "cheng_unscreened_sens": ms["cheng_unscreened_3064"]["sensitivity"],
                "cheng_screened_sens": ms["cheng_overlap_screened_1689"]["sensitivity"],
            })
        summary_csv_rows.append({
            "row_type": "5seed_mean",
            "model_code": mcode,
            "model": mkey,
            "seed": "mean_42_46",
            "accuracy": agg_metrics["accuracy"]["mean"],
            "accuracy_sd": agg_metrics["accuracy"]["sd_sample"],
            "accuracy_t_ci_low": agg_metrics["accuracy"]["t_ci_95_df4"][0],
            "accuracy_t_ci_high": agg_metrics["accuracy"]["t_ci_95_df4"][1],
            "accuracy_ci_low": agg_metrics["accuracy"]["patient_bootstrap_ci_95"][0],
            "accuracy_ci_high": agg_metrics["accuracy"]["patient_bootstrap_ci_95"][1],
            "balanced_accuracy": agg_metrics["balanced_accuracy"]["mean"],
            "balanced_accuracy_sd": agg_metrics["balanced_accuracy"]["sd_sample"],
            "sensitivity": agg_metrics["sensitivity"]["mean"],
            "sensitivity_sd": agg_metrics["sensitivity"]["sd_sample"],
            "specificity": agg_metrics["specificity"]["mean"],
            "specificity_sd": agg_metrics["specificity"]["sd_sample"],
            "precision": agg_metrics["precision"]["mean"],
            "precision_sd": agg_metrics["precision"]["sd_sample"],
            "f1": agg_metrics["f1"]["mean"],
            "f1_sd": agg_metrics["f1"]["sd_sample"],
            "roc_auc": agg_metrics["roc_auc"]["mean"],
            "roc_auc_sd": agg_metrics["roc_auc"]["sd_sample"],
            "roc_auc_t_ci_low": agg_metrics["roc_auc"]["t_ci_95_df4"][0],
            "roc_auc_t_ci_high": agg_metrics["roc_auc"]["t_ci_95_df4"][1],
            "roc_auc_ci_low": agg_metrics["roc_auc"]["patient_bootstrap_ci_95"][0],
            "roc_auc_ci_high": agg_metrics["roc_auc"]["patient_bootstrap_ci_95"][1],
            "pr_auc": agg_metrics["pr_auc"]["mean"],
            "pr_auc_sd": agg_metrics["pr_auc"]["sd_sample"],
            "cheng_unscreened_sens": agg_metrics["cheng_unscreened_3064_sensitivity"]["mean"],
            "cheng_screened_sens": agg_metrics["cheng_overlap_screened_1689_sensitivity"]["mean"],
        })

    # Paired patient-cluster bootstrap differences (2,000 resamples, seed 42):
    # We compute both:
    # (1) 5-seed mean metric difference (average of the 5 seed metrics in each bootstrap replicate)
    # (2) 5-seed ensemble mean prediction difference (metrics of the 5-seed average score in each bootstrap replicate)
    def compute_paired_diff(
        m1_dict: dict[str, float],
        b1_dict: dict[str, np.ndarray],
        m2_dict: dict[str, float],
        b2_dict: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        out = {}
        for m_name in metric_names:
            pt = float(m1_dict[m_name] - m2_dict[m_name])
            d_arr = b1_dict[m_name] - b2_dict[m_name]
            lo, hi = np.percentile(d_arr, [2.5, 97.5])
            out[m_name] = {
                "point_diff": pt,
                "ci_95_low": float(lo),
                "ci_95_high": float(hi),
                "ci_95_str": f"{pt:+.4f} ({lo:+.4f} to {hi:+.4f})",
            }
        return out

    mA = "resnet18_brats_patient_split"
    mB = "resnet18_brats_finetuned"
    mC = "small_cnn_brats_patient_split"

    mean_pt_by_model = {
        mk: {mn: per_model_results[mk]["aggregate_5seed"][mn]["mean"] for mn in metric_names}
        for mk in (mA, mB, mC)
    }
    zs_pt = {mn: res_zeroshot[mn] for mn in metric_names}

    paired_5seed_mean_metric_diffs = {
        "A_minus_zeroshot_discovery_resnet18": compute_paired_diff(mean_pt_by_model[mA], mean_boot_by_model[mA], zs_pt, boot_zeroshot),
        "B_minus_zeroshot_discovery_resnet18": compute_paired_diff(mean_pt_by_model[mB], mean_boot_by_model[mB], zs_pt, boot_zeroshot),
        "C_minus_zeroshot_discovery_resnet18": compute_paired_diff(mean_pt_by_model[mC], mean_boot_by_model[mC], zs_pt, boot_zeroshot),
        "A_minus_B": compute_paired_diff(mean_pt_by_model[mA], mean_boot_by_model[mA], mean_pt_by_model[mB], mean_boot_by_model[mB]),
    }

    paired_5seed_ensemble_prediction_diffs = {
        "A_minus_zeroshot_discovery_resnet18": compute_paired_diff(
            {mn: ensemble_res_by_model[mA][mn] for mn in metric_names},
            ensemble_boot_by_model[mA],
            zs_pt,
            boot_zeroshot,
        ),
        "B_minus_zeroshot_discovery_resnet18": compute_paired_diff(
            {mn: ensemble_res_by_model[mB][mn] for mn in metric_names},
            ensemble_boot_by_model[mB],
            zs_pt,
            boot_zeroshot,
        ),
        "C_minus_zeroshot_discovery_resnet18": compute_paired_diff(
            {mn: ensemble_res_by_model[mC][mn] for mn in metric_names},
            ensemble_boot_by_model[mC],
            zs_pt,
            boot_zeroshot,
        ),
        "A_minus_B": compute_paired_diff(
            {mn: ensemble_res_by_model[mA][mn] for mn in metric_names},
            ensemble_boot_by_model[mA],
            {mn: ensemble_res_by_model[mB][mn] for mn in metric_names},
            ensemble_boot_by_model[mB],
        ),
    }

    # Seed-wise paired differences A - B across seeds 42, 43, 44, 45, 46
    seedwise_A_minus_B = {}
    for m_name in metric_names + ["cheng_unscreened_3064_sensitivity", "cheng_overlap_screened_1689_sensitivity"]:
        per_seed_diffs = {}
        diff_list = []
        for s in seeds:
            if m_name.startswith("cheng_"):
                ck = m_name.replace("_sensitivity", "")
                va = metrics_by_model_seed[(mA, s)][ck]["sensitivity"]
                vb = metrics_by_model_seed[(mB, s)][ck]["sensitivity"]
            else:
                va = metrics_by_model_seed[(mA, s)][m_name]
                vb = metrics_by_model_seed[(mB, s)][m_name]
            d = float(va - vb)
            per_seed_diffs[str(s)] = d
            diff_list.append(d)
        mean_d = float(np.mean(diff_list))
        sd_d = float(np.std(diff_list, ddof=1))
        se_d = sd_d / np.sqrt(len(diff_list))
        t_lo = float(mean_d - T_CRIT_DF4 * se_d)
        t_hi = float(mean_d + T_CRIT_DF4 * se_d)
        seedwise_A_minus_B[m_name] = {
            "mean_diff": mean_d,
            "sd_sample": sd_d,
            "min_diff": float(np.min(diff_list)),
            "max_diff": float(np.max(diff_list)),
            "t_ci_95_df4": [t_lo, t_hi],
            "mean_sd_str": f"{mean_d:+.4f} +/- {sd_d:.4f}",
            "t_ci_95_str": f"{t_lo:+.4f} to {t_hi:+.4f}",
            "five_seed_values": per_seed_diffs,
        }

    # ROC-AUC dominance check across all 15 checkpoints
    all_15_aucs = [
        (mkey, s, float(metrics_by_model_seed[(mkey, s)]["roc_auc"]))
        for spec in model_specs
        for mkey in [spec["model_key"]]
        for s in seeds
    ]
    min_auc_entry = min(all_15_aucs, key=lambda x: x[2])
    max_auc_entry = max(all_15_aucs, key=lambda x: x[2])
    ref_zeroshot_resnet18_auc = 0.7087767625548996
    ref_best_discovery_densenet_seed42_auc = 0.7254803587771558

    auc_dominance_check = {
        "zero_shot_discovery_resnet18_seed42_auc": ref_zeroshot_resnet18_auc,
        "best_discovery_densenet121_seed42_auc": ref_best_discovery_densenet_seed42_auc,
        "all_15_checkpoints_above_zeroshot_resnet18_07088": bool(all(x[2] > ref_zeroshot_resnet18_auc for x in all_15_aucs)),
        "all_15_checkpoints_above_best_discovery_densenet121_07255": bool(all(x[2] > ref_best_discovery_densenet_seed42_auc for x in all_15_aucs)),
        "n_checkpoints_above_07088": int(sum(1 for x in all_15_aucs if x[2] > ref_zeroshot_resnet18_auc)),
        "n_checkpoints_above_07255": int(sum(1 for x in all_15_aucs if x[2] > ref_best_discovery_densenet_seed42_auc)),
        "minimum_roc_auc_among_15_checkpoints": {
            "model": min_auc_entry[0],
            "seed": min_auc_entry[1],
            "roc_auc": min_auc_entry[2],
        },
        "maximum_roc_auc_among_15_checkpoints": {
            "model": max_auc_entry[0],
            "seed": max_auc_entry[1],
            "roc_auc": max_auc_entry[2],
        },
        "all_15_roc_aucs": [
            {"model": m, "seed": s, "roc_auc": a} for (m, s, a) in all_15_aucs
        ],
    }
    print(
        f"  ROC-AUC Dominance Check: All 15 > 0.7088? {auc_dominance_check['all_15_checkpoints_above_zeroshot_resnet18_07088']} "
        f"| All 15 > 0.7255? {auc_dominance_check['all_15_checkpoints_above_best_discovery_densenet121_07255']} "
        f"| Min ROC-AUC = {min_auc_entry[2]:.4f} ({min_auc_entry[0]} seed {min_auc_entry[1]})",
        flush=True,
    )

    multiseed_payload = {
        "task": "Multi-seed (seeds 42-46) evaluation of Patient-Split BraTS reference models A, B, C",
        "manifest_path": "paper_evidence/brats_patient_split_manifest.csv",
        "manifest_sha256": manifest_sha,
        "predictions_path": "paper_evidence/brats_250_leakage_free_multiseed_predictions.csv.gz",
        "predictions_sha256": multiseed_pred_sha256,
        "eval_batch_size_brats": eval_batch_size,
        "eval_batch_size_cheng": 64,
        "threshold": 0.5,
        "bootstrap_method": "Patient-cluster bootstrap (2,000 resamples of patients with replacement, np.random.default_rng(42))",
        "zeroshot_discovery_resnet18_reference": res_zeroshot,
        "models": per_model_results,
        "paired_bootstrap_differences_5seed_mean_metric": paired_5seed_mean_metric_diffs,
        "paired_bootstrap_differences_5seed_ensemble_score": paired_5seed_ensemble_prediction_diffs,
        "seedwise_paired_differences_A_minus_B": seedwise_A_minus_B,
        "roc_auc_dominance_check": auc_dominance_check,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "opencv": cv2.__version__,
            "nibabel": nib.__version__,
            "pillow": pillow_version,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "runtime_seconds_up_to_step3": round(time.time() - t_start, 2),
    }

    multiseed_json_path = pe_dir / "leakage_free_multiseed.json"
    multiseed_json_path.write_text(json.dumps(multiseed_payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary_csv_rows).to_csv(pe_dir / "leakage_free_multiseed_summary.csv", index=False)
    print(f"Saved {multiseed_json_path} and paper_evidence/leakage_free_multiseed_summary.csv", flush=True)

    # =========================================================================
    # STEP 4: GRAD-CAM PER SEED (SEEDS 42-46 OF MODELS A AND B ON 16,074 SLICES)
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("STEP 4: GRAD-CAM EVALUATION ACROSS SEEDS 42-46 FOR MODELS A AND B", flush=True)
    print("======================================================================", flush=True)
    t_gc0 = time.time()

    tumor_mask_rows = (df_test["label_idx"] == 1).to_numpy()
    df_tumor = df_test[tumor_mask_rows].copy().reset_index(drop=True)
    n_tumor_slices = len(df_tumor)
    imgs_tumor_u8_224 = test250_u8[tumor_mask_rows].copy()
    del test250_u8

    brain_masks_240 = np.zeros((n_tumor_slices, 240, 240), dtype=bool)
    for i in range(n_tumor_slices):
        brain_masks_240[i] = cv2.resize(imgs_tumor_u8_224[i], (240, 240), interpolation=cv2.INTER_NEAREST) > 0

    seg_root = Path("D:/datasets/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    patient_to_rows: dict[str, list[tuple[int, int]]] = {}
    for row_idx, row in df_tumor.iterrows():
        patient_to_rows.setdefault(str(row["patient_id"]), []).append((int(row_idx), int(row["slice_id"])))

    seg_masks_240 = np.zeros((n_tumor_slices, 240, 240), dtype=bool)
    tumor_voxel_counts = np.zeros(n_tumor_slices, dtype=np.int32)

    def _load_patient_seg(pid: str) -> tuple[str, list[tuple[int, np.ndarray, int]]]:
        seg_path = seg_root / pid / f"{pid}-seg.nii.gz"
        seg_img = nib.as_closest_canonical(nib.load(str(seg_path)))
        seg_vol = np.asarray(seg_img.dataobj)
        out_items = []
        for r_idx, s_id in patient_to_rows[pid]:
            raw_slice = seg_vol[:, :, s_id] > 0
            oriented_mask = np.rot90(raw_slice, 1)
            v_cnt = int(np.count_nonzero(oriented_mask))
            out_items.append((r_idx, oriented_mask, v_cnt))
        return pid, out_items

    with ThreadPoolExecutor(max_workers=16) as ex:
        for _, items in ex.map(_load_patient_seg, unique_patients.tolist()):
            for r_idx, m240, v_cnt in items:
                seg_masks_240[r_idx] = m240
                tumor_voxel_counts[r_idx] = v_cnt

    df_tumor["tumor_voxels"] = tumor_voxel_counts
    df_tumor["brain_voxels"] = np.sum(brain_masks_240, axis=(1, 2))
    df_tumor["tumor_brain_frac"] = df_tumor["tumor_voxels"] / np.maximum(1, df_tumor["brain_voxels"])

    kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    seg_masks_dilated15 = np.zeros_like(seg_masks_240)
    for i in range(n_tumor_slices):
        seg_masks_dilated15[i] = cv2.dilate(seg_masks_240[i].astype(np.uint8), kernel_15) > 0

    existing_gc = json.loads((pe_dir / "gradcam_brats_evaluation.json").read_text(encoding="utf-8"))
    seed42_gc_map = {
        "resnet18_brats_patient_split": existing_gc["models"]["brats_patient_split_resnet18_imagenet"],
        "resnet18_brats_finetuned": existing_gc["models"]["brats_patient_split_resnet18_finetuned"],
    }

    gc_models_multiseed: dict[str, Any] = {}
    gc_metric_keys = ["pointing_hit_exact", "pointing_hit_15px", "ebpg_tumor", "ebpg_brain", "dice_05", "iou_05", "dice_07", "iou_07", "dice_topk", "iou_topk", "soft_dice"]

    for mkey in ("resnet18_brats_patient_split", "resnet18_brats_finetuned"):
        per_seed_gc: dict[str, Any] = {}
        for s in seeds:
            if s == 42:
                per_seed_gc[str(s)] = seed42_gc_map[mkey]
                om42 = per_seed_gc[str(s)]["overall_all_tumor_slices"]["metrics"]
                print(
                    f"  [Grad-CAM | {mkey:30s} | seed 42 (reused)] "
                    f"Point(exact)={om42['pointing_hit_exact']['mean']:.4f}, Point(15px)={om42['pointing_hit_15px']['mean']:.4f}, "
                    f"EBPG={om42['ebpg_tumor']['mean']:.4f}, Dice@0.5={om42['dice_05']['mean']:.4f}, Dice@TopK={om42['dice_topk']['mean']:.4f}",
                    flush=True,
                )
                continue

            ckpt_p = models_dir / f"{mkey}_seed{s}.pth"
            t_m = time.time()
            model, _ = load_checkpoint(ckpt_p, map_location=device)
            model = model.to(device)
            model.eval()

            cams_7x7 = np.zeros((n_tumor_slices, 7, 7), dtype=np.float32)
            probs_all = np.zeros(n_tumor_slices, dtype=np.float32)
            batch_sz = 64
            for start_i in range(0, n_tumor_slices, batch_sz):
                end_i = min(n_tumor_slices, start_i + batch_sz)
                b_u8 = torch.from_numpy(imgs_tumor_u8_224[start_i:end_i]).to(device, non_blocking=True)
                b_fp = (b_u8.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1) - mean_t) / std_t
                c7, pr = compute_batched_gradcam_7x7(model, model.layer4[-1], b_fp, target_class=1)
                cams_7x7[start_i:end_i] = c7
                probs_all[start_i:end_i] = pr
            del model
            torch.cuda.empty_cache()

            pointing_exact = np.zeros(n_tumor_slices, dtype=np.float64)
            pointing_15px = np.zeros(n_tumor_slices, dtype=np.float64)
            ebpg_tumor = np.zeros(n_tumor_slices, dtype=np.float64)
            ebpg_brain = np.zeros(n_tumor_slices, dtype=np.float64)
            null_img = tumor_voxel_counts / float(240 * 240)
            null_brain = df_tumor["tumor_brain_frac"].to_numpy(dtype=np.float64)
            dice_05 = np.zeros(n_tumor_slices, dtype=np.float64)
            iou_05 = np.zeros(n_tumor_slices, dtype=np.float64)
            dice_07 = np.zeros(n_tumor_slices, dtype=np.float64)
            iou_07 = np.zeros(n_tumor_slices, dtype=np.float64)
            dice_topk = np.zeros(n_tumor_slices, dtype=np.float64)
            iou_topk = np.zeros(n_tumor_slices, dtype=np.float64)
            soft_dice = np.zeros(n_tumor_slices, dtype=np.float64)

            for i in range(n_tumor_slices):
                c7 = cams_7x7[i]
                if float(c7.max()) <= 0.0:
                    continue
                cam240 = cv2.resize(c7, (240, 240), interpolation=cv2.INTER_LINEAR)
                c_min, c_max = float(cam240.min()), float(cam240.max())
                if c_max > c_min:
                    cam240 = (cam240 - c_min) / (c_max - c_min)

                t_mask = seg_masks_240[i]
                b_mask = brain_masks_240[i]
                t_voxels = int(tumor_voxel_counts[i])

                peak_flat = int(np.argmax(cam240))
                py, px = divmod(peak_flat, 240)
                pointing_exact[i] = 1.0 if t_mask[py, px] else 0.0
                pointing_15px[i] = 1.0 if seg_masks_dilated15[i, py, px] else 0.0

                total_energy = float(np.sum(cam240))
                if total_energy > 0.0:
                    ebpg_tumor[i] = float(np.sum(cam240[t_mask])) / total_energy
                    ebpg_brain[i] = float(np.sum(cam240[b_mask])) / total_energy

                inter_soft = float(np.sum(cam240[t_mask]))
                soft_dice[i] = (2.0 * inter_soft) / (total_energy + t_voxels)

                for thr, d_arr, j_arr in ((0.5, dice_05, iou_05), (0.7, dice_07, iou_07)):
                    pred_m = cam240 >= thr
                    p_cnt = int(np.count_nonzero(pred_m))
                    inter = int(np.count_nonzero(pred_m & t_mask))
                    union = p_cnt + t_voxels - inter
                    d_arr[i] = (2.0 * inter) / max(1, p_cnt + t_voxels)
                    j_arr[i] = inter / max(1, union)

                flat_cam = cam240.ravel()
                flat_mask = t_mask.ravel()
                topk_idx = np.argpartition(flat_cam, -t_voxels)[-t_voxels:]
                inter_k = int(np.count_nonzero(flat_mask[topk_idx]))
                union_k = 2 * t_voxels - inter_k
                dice_topk[i] = inter_k / float(t_voxels)
                iou_topk[i] = inter_k / float(max(1, union_k))

            df_m = df_tumor.copy()
            df_m["score"] = probs_all
            df_m["pred"] = (probs_all >= 0.5).astype(np.int32)
            df_m["pointing_hit_exact"] = pointing_exact
            df_m["pointing_hit_15px"] = pointing_15px
            df_m["ebpg_tumor"] = ebpg_tumor
            df_m["ebpg_brain"] = ebpg_brain
            df_m["null_tumor_area_frac_image"] = null_img
            df_m["null_tumor_area_frac_brain"] = null_brain
            df_m["enrichment_vs_image"] = ebpg_tumor / np.maximum(1e-8, null_img)
            df_m["enrichment_vs_brain"] = ebpg_tumor / np.maximum(1e-8, null_brain)
            df_m["dice_05"] = dice_05
            df_m["iou_05"] = iou_05
            df_m["dice_07"] = dice_07
            df_m["iou_07"] = iou_07
            df_m["dice_topk"] = dice_topk
            df_m["iou_topk"] = iou_topk
            df_m["soft_dice"] = soft_dice

            overall_summary = summarize_slice_df(df_m, unique_patients, patient_weights)
            tp_summary = summarize_slice_df(df_m[df_m["pred"] == 1], unique_patients, patient_weights)
            fn_summary = summarize_slice_df(df_m[df_m["pred"] == 0], unique_patients, patient_weights)

            per_seed_gc[str(s)] = {
                "checkpoint": str(ckpt_p.relative_to(ROOT)).replace("\\", "/"),
                "overall_all_tumor_slices": overall_summary,
                "by_detection_status": {
                    "detected_true_positives_score_gte_05": tp_summary,
                    "missed_false_negatives_score_lt_05": fn_summary,
                },
            }
            om = overall_summary["metrics"]
            print(
                f"  [Grad-CAM | {mkey:30s} | seed {s}] "
                f"Point(exact)={om['pointing_hit_exact']['mean']:.4f}, Point(15px)={om['pointing_hit_15px']['mean']:.4f}, "
                f"EBPG={om['ebpg_tumor']['mean']:.4f}, Dice@0.5={om['dice_05']['mean']:.4f}, Dice@TopK={om['dice_topk']['mean']:.4f} ({time.time()-t_m:.1f}s)",
                flush=True,
            )

        # Aggregate Grad-CAM metrics across the 5 seeds
        agg_gc = {}
        for gk in gc_metric_keys:
            vals = [per_seed_gc[str(s)]["overall_all_tumor_slices"]["metrics"][gk]["mean"] for s in seeds]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1))
            se_v = sd_v / np.sqrt(len(vals))
            t_lo = float(max(0.0, mean_v - T_CRIT_DF4 * se_v))
            t_hi = float(min(1.0, mean_v + T_CRIT_DF4 * se_v))
            agg_gc[gk] = {
                "mean": mean_v,
                "sd_sample": sd_v,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "seed_values": {str(s): float(per_seed_gc[str(s)]["overall_all_tumor_slices"]["metrics"][gk]["mean"]) for s in seeds},
                "mean_sd_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
                "t_ci_95_df4": [t_lo, t_hi],
                "t_ci_95_str": f"{t_lo:.4f} - {t_hi:.4f}",
            }

        gc_models_multiseed[mkey] = {
            "seeds": seeds,
            "per_seed": per_seed_gc,
            "aggregate_5seed_overall": agg_gc,
        }

    gc_multiseed_payload = {
        "cohort": "brats_oriented (250-patient stratified subset, all 16,074 tumor-positive slices)",
        "n_patients": int(n_patients),
        "n_tumor_slices": int(n_tumor_slices),
        "gradcam_layer": "model.layer4[-1] (7x7 upsampled to 240x240 via bilinear interpolation)",
        "target_class": 1,
        "metric_definitions_from_code": {
            "source_file": "paper_evidence/run_gradcam_brats_evaluation.py",
            "pointing_rule_15px": {
                "lines": "264-267 and 359-362",
                "code": (
                    "kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)); "
                    "seg_masks_dilated15[i] = cv2.dilate(seg_masks_240[i].astype(np.uint8), kernel_15) > 0; "
                    "peak_flat = int(np.argmax(cam240)); py, px = divmod(peak_flat, 240); "
                    "pointing_15px[i] = 1.0 if seg_masks_dilated15[i, py, px] else 0.0"
                ),
                "description": (
                    "The ground-truth binary tumor mask (240x240) is morphologically dilated by an elliptical structuring element "
                    "of diameter 31x31 pixels (radius = 15 pixels). A slice is a hit (1.0) if the peak Grad-CAM pixel (py, px) = argmax(cam240) "
                    "falls within <=15 pixels of any ground-truth tumor voxel."
                ),
            },
            "ebpg_tumor_energy": {
                "lines": "365-368",
                "code": (
                    "total_energy = float(np.sum(cam240)); "
                    "ebpg_tumor[i] = float(np.sum(cam240[t_mask])) / total_energy"
                ),
                "description": (
                    "Fraction of total min-max normalized 240x240 Grad-CAM activation energy that falls strictly inside the "
                    "ground-truth 2D binary tumor mask t_mask (seg_masks_240[i])."
                ),
            },
            "topk_threshold_K": {
                "lines": "356 and 384-390",
                "code": (
                    "t_voxels = int(tumor_voxel_counts[i]); "
                    "flat_cam = cam240.ravel(); flat_mask = t_mask.ravel(); "
                    "topk_idx = np.argpartition(flat_cam, -t_voxels)[-t_voxels:]; "
                    "inter_k = int(np.count_nonzero(flat_mask[topk_idx])); "
                    "union_k = 2 * t_voxels - inter_k; "
                    "dice_topk[i] = inter_k / float(t_voxels); "
                    "iou_topk[i] = inter_k / float(max(1, union_k))"
                ),
                "description": (
                    "Adaptive per-slice threshold K equal to the exact ground-truth tumor voxel count K = |M| = t_voxels in that 240x240 slice. "
                    "The top K highest-intensity pixels in cam240 are selected as the binary prediction P_K (so |P_K| = |M| = K), "
                    "where Dice@TopK = 2|P_K cap M| / (|P_K| + |M|) = |P_K cap M| / K."
                ),
            },
        },
        "zeroshot_discovery_resnet18_reference_overall": existing_gc["models"]["zeroshot_discovery_resnet18"]["overall_all_tumor_slices"],
        "models_multiseed": gc_models_multiseed,
        "gradcam_runtime_seconds": round(time.time() - t_gc0, 2),
        "total_pipeline_runtime_seconds": round(time.time() - t_start, 2),
    }

    gc_out_path = pe_dir / "gradcam_brats_multiseed.json"
    gc_out_path.write_text(json.dumps(gc_multiseed_payload, indent=2), encoding="utf-8")
    print(f"\nSaved {gc_out_path} in {time.time() - t_gc0:.1f}s (total runtime: {time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

