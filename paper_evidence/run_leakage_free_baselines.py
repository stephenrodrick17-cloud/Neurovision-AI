"""
Step 2: Leakage-Free Baselines (BraTS Patient-Split, Discovery Patient-Decontaminated, and Cheng Patient-Split vs Slice-Split)

Produces three rigorous leakage-free baselines with 95% patient-cluster bootstrap CIs (2,000 resamples, seed 42):
1. Patient-Split BraTS (brats_oriented):
   - Train (850 patients) / Validation (151 patients) strictly disjoint from the held-out 250-patient test subset (33,696 slices).
   - Trains:
     (a) ResNet-18 from ImageNet pretrained weights
     (b) ResNet-18 fine-tuned from the Discovery primary checkpoint
     (c) SmallCNN from scratch
   - Evaluates all models on all 33,696 slices (250 patients) of the held-out BraTS test set (with paired bootstrap CIs vs zero-shot)
     AND externally on Cheng (3,064 slices, 233 patients).
2. Patient-Decontaminated Discovery ResNet-18:
   - Removes all train/validation slices belonging to the 59 Cheng patients that appear in the Discovery test split (zero train/val-to-test patient leakage).
   - Evaluates on Discovery test (full n=606, clean n=465, Cheng-matched n=141) and Overlap-Screened Cheng (1,689 slices, 152 unseen patients).
3. Cheng 233-Patient Split vs Random Slice Split (3,064 slices, 3-class tumor subtype: Meningioma, Glioma, Pituitary):
   - Directly quantifies the optimism of random slice-level splitting vs strict patient-level splitting on Cheng with 95% patient-cluster bootstrap CIs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile
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
from PIL import Image
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
from torchvision import models

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
)
from BrainTumorAI.external_validation import _decode_cheng_mat


class InMemoryImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, images: list[Image.Image], labels: list[int], training: bool = False) -> None:
        self.images = images
        self.labels = labels
        self.transform = make_training_transform() if training else make_inference_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), int(self.labels[idx])


def fast_binary_metrics_and_bootstrap(
    y_true: np.ndarray,
    scores: np.ndarray,
    patients_arr: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
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

    unique_patients = np.array(sorted(np.unique(patients_arr)))
    n_patients = len(unique_patients)
    pid_index_list = [np.where(patients_arr == pid)[0] for pid in unique_patients]

    rng = np.random.default_rng(seed)
    boot_choices = rng.integers(0, n_patients, size=(n_boot, n_patients))
    patient_weights = np.zeros((n_boot, n_patients), dtype=np.int32)
    for b in range(n_boot):
        patient_weights[b] = np.bincount(boot_choices[b], minlength=n_patients)

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
        "auc": np.zeros(n_boot, dtype=np.float64),
    }
    boot_dist["balanced_accuracy"] = 0.5 * (boot_dist["sensitivity"] + boot_dist["specificity"])

    slice_patient_idx = np.searchsorted(unique_patients, patients_arr)
    for b in range(n_boot):
        w_slice = patient_weights[b, slice_patient_idx]
        mask_nz = w_slice > 0
        boot_dist["auc"][b] = roc_auc_score(y_true[mask_nz], scores[mask_nz], sample_weight=w_slice[mask_nz])

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
        "auc": float(auc),
        "pr_auc": float(ap),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "bootstrap_ci_95": ci_95,
    }
    return summary, boot_dist


def train_binary_model(
    model: nn.Module,
    train_images: list[Image.Image],
    train_labels: list[int],
    val_images: list[Image.Image],
    val_labels: list[int],
    device: torch.device,
    seed: int = 42,
    max_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_path: Path | None = None,
    arch_name: str = "resnet18",
) -> tuple[nn.Module, dict[str, Any]]:
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
        tr_bal = float(balanced_accuracy_score(tr_targets, tr_preds))

        model.eval()
        va_loss = 0.0
        va_preds, va_targets = [], []
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += float(loss.item()) * y.size(0)
                va_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
                va_targets.extend(y.detach().cpu().tolist())
        va_loss /= len(val_images)
        va_bal = float(balanced_accuracy_score(va_targets, va_preds))
        scheduler.step(va_bal)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_balanced_accuracy": tr_bal,
            "val_loss": va_loss,
            "val_balanced_accuracy": va_bal,
        })
        print(f"    Epoch {epoch:02d}/{max_epochs}: tr_loss={tr_loss:.4f}, tr_bal={tr_bal:.4f}, va_loss={va_loss:.4f}, va_bal={va_bal:.4f}", flush=True)

        if va_bal > best_bal_acc + 1e-4 or (abs(va_bal - best_bal_acc) <= 1e-4 and va_loss < best_val_loss):
            best_bal_acc = va_bal
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 2:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()
    torch.cuda.empty_cache()
    if save_path is not None:
        save_checkpoint(
            model,
            save_path,
            arch_name,
            {
                "seed": seed,
                "selected_epoch": best_epoch,
                "validation_balanced_accuracy": best_bal_acc,
                "validation_loss": best_val_loss,
            },
        )
    return model, {
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_bal_acc,
        "best_validation_loss": best_val_loss,
        "training_seconds": round(time.time() - t0, 2),
        "history": history,
    }


def main() -> None:
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    models_dir = ROOT / "paper_evidence" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    mean_t = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    resize_op = make_inference_transform().transforms[0]

    # =========================================================================
    # PART 1: STRICT PATIENT-SPLIT BraTS BASELINE (brats_oriented)
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("PART 1: STRICT PATIENT-SPLIT BraTS BASELINE (brats_oriented)", flush=True)
    print("======================================================================", flush=True)

    brats_root = Path("D:/datasets/prepared/brats_oriented")
    df_full_brats = pd.read_csv(brats_root / "manifest.csv")
    df_test_250 = pd.read_csv(ROOT / "paper_evidence" / "brats_250_patients_manifest.csv")

    test_patients_set = set(df_test_250["patient_id"].unique())
    all_brats_patients = sorted(df_full_brats["patient_id"].unique())
    remaining_patients = [p for p in all_brats_patients if p not in test_patients_set]
    assert len(test_patients_set) == 250
    assert len(remaining_patients) == 1001
    assert len(set(remaining_patients) & test_patients_set) == 0

    rng_split = np.random.default_rng(42)
    shuffled_rem = rng_split.permutation(remaining_patients)
    train_patients = sorted(shuffled_rem[:850].tolist())
    val_patients = sorted(shuffled_rem[850:].tolist())
    assert len(train_patients) == 850 and len(val_patients) == 151
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & test_patients_set) == 0
    assert len(set(val_patients) & test_patients_set) == 0

    # Sample up to 6 tumor and 6 no-tumor slices per patient for balanced patient-weighted training/validation
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

    df_brats_train = sample_patient_slices(df_full_brats, train_patients, k_per_class=4, seed_val=42)
    df_brats_val = sample_patient_slices(df_full_brats, val_patients, k_per_class=4, seed_val=43)
    print(
        f"Patient-Split BraTS: Train={len(train_patients)} patients ({len(df_brats_train)} slices), "
        f"Val={len(val_patients)} patients ({len(df_brats_val)} slices), "
        f"Held-Out Test={len(test_patients_set)} patients ({len(df_test_250)} slices)",
        flush=True,
    )

    def _load_rgb_pil(path_str: str) -> Image.Image:
        with Image.open(path_str) as im:
            return im.convert("RGB")

    ckpt_r18_imgnet = models_dir / "resnet18_brats_patient_split.pth"
    ckpt_r18_ft = models_dir / "resnet18_brats_finetuned.pth"
    ckpt_scnn_brats = models_dir / "small_cnn_brats_patient_split.pth"

    if ckpt_r18_imgnet.is_file() and ckpt_r18_ft.is_file() and ckpt_scnn_brats.is_file():
        print("Loading already-trained Patient-Split BraTS checkpoints from disk...", flush=True)
        m_r18_imgnet, meta_imgnet = load_checkpoint(ckpt_r18_imgnet, map_location="cpu")
        m_r18_ft, meta_ft = load_checkpoint(ckpt_r18_ft, map_location="cpu")
        m_scnn_brats, meta_scnn = load_checkpoint(ckpt_scnn_brats, map_location="cpu")
        info_r18_imgnet = meta_imgnet.get("extra", {})
        info_r18_ft = meta_ft.get("extra", {})
        info_scnn_brats = meta_scnn.get("extra", {})
    else:
        print("Loading BraTS train/val slices into RAM...", flush=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            brats_train_imgs = list(ex.map(_load_rgb_pil, [str(brats_root / f) for f in df_brats_train["source_file"]]))
            brats_val_imgs = list(ex.map(_load_rgb_pil, [str(brats_root / f) for f in df_brats_val["source_file"]]))
        brats_train_lbls = df_brats_train["label_idx"].astype(int).tolist()
        brats_val_lbls = df_brats_val["label_idx"].astype(int).tolist()

        # Train 3 models on Patient-Split BraTS
        print("\n1A. Training ResNet-18 (from ImageNet) on Patient-Split BraTS...", flush=True)
        m_r18_imgnet = make_model("resnet18", pretrained=True)
        m_r18_imgnet, info_r18_imgnet = train_binary_model(
            m_r18_imgnet,
            brats_train_imgs,
            brats_train_lbls,
            brats_val_imgs,
            brats_val_lbls,
            device,
            seed=42,
            max_epochs=6,
            batch_size=32,
            lr=1e-4,
            save_path=ckpt_r18_imgnet,
            arch_name="resnet18",
        )
        torch.cuda.empty_cache()

        print("\n1B. Fine-tuning ResNet-18 (initialized from Discovery checkpoint) on Patient-Split BraTS...", flush=True)
        m_r18_ft, _ = load_checkpoint(DEFAULT_CHECKPOINT, map_location="cpu")
        m_r18_ft, info_r18_ft = train_binary_model(
            m_r18_ft,
            brats_train_imgs,
            brats_train_lbls,
            brats_val_imgs,
            brats_val_lbls,
            device,
            seed=42,
            max_epochs=6,
            batch_size=32,
            lr=5e-5,
            save_path=ckpt_r18_ft,
            arch_name="resnet18",
        )
        torch.cuda.empty_cache()

        print("\n1C. Training SmallCNN (from scratch) on Patient-Split BraTS...", flush=True)
        m_scnn_brats = make_model("small_cnn", pretrained=False)
        m_scnn_brats, info_scnn_brats = train_binary_model(
            m_scnn_brats,
            brats_train_imgs,
            brats_train_lbls,
            brats_val_imgs,
            brats_val_lbls,
            device,
            seed=42,
            max_epochs=6,
            batch_size=32,
            lr=1e-4,
            save_path=ckpt_scnn_brats,
            arch_name="small_cnn",
        )
        torch.cuda.empty_cache()

        del brats_train_imgs, brats_val_imgs
        torch.cuda.empty_cache()

    # Evaluate on all 33,696 slices of the held-out 250-patient BraTS test set
    pred_csv_path = ROOT / "paper_evidence" / "brats_250_leakage_free_baseline_predictions.csv.gz"
    y_test250 = df_test_250["label_idx"].to_numpy(dtype=np.int32)
    p_test250 = df_test_250["patient_id"].to_numpy()

    if pred_csv_path.is_file():
        print("Loading cached BraTS 250-patient baseline predictions from disk...", flush=True)
        df_pred_brats_baseline = pd.read_csv(pred_csv_path)
        scores_zeroshot = df_pred_brats_baseline["score_zeroshot_resnet18"].to_numpy(dtype=np.float32)
        scores_r18_imgnet = df_pred_brats_baseline["score_brats_patient_split_resnet18"].to_numpy(dtype=np.float32)
        scores_r18_ft = df_pred_brats_baseline["score_brats_finetuned_resnet18"].to_numpy(dtype=np.float32)
        scores_scnn_brats = df_pred_brats_baseline["score_brats_patient_split_smallcnn"].to_numpy(dtype=np.float32)
    else:
        print("\nEvaluating Patient-Split BraTS models on all 33,696 slices of held-out 250-patient test set...", flush=True)

        def _load_u8_224(path_str: str) -> np.ndarray:
            with Image.open(path_str) as im:
                rgb = resize_op(im.convert("RGB"))
                return np.asarray(rgb, dtype=np.uint8)[:, :, 0].copy()

        cache_npy = brats_root / "cache_250_u8_224.npy"
        if cache_npy.is_file():
            test250_u8 = np.load(cache_npy)
        else:
            with ThreadPoolExecutor(max_workers=16) as ex:
                test250_u8 = np.stack(list(ex.map(_load_u8_224, [str(brats_root / f) for f in df_test_250["source_file"]])), axis=0)
            np.save(cache_npy, test250_u8)

        def eval_on_u8_buffer(model_obj: nn.Module, buf_u8: np.ndarray, batch_sz: int = 128) -> np.ndarray:
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

        # Also get zero-shot Discovery ResNet-18 scores on the same 250 patients for paired comparison
        m_zeroshot, _ = load_checkpoint(DEFAULT_CHECKPOINT, map_location=device)
        scores_zeroshot = eval_on_u8_buffer(m_zeroshot, test250_u8)
        del m_zeroshot

        scores_r18_imgnet = eval_on_u8_buffer(m_r18_imgnet, test250_u8)
        scores_r18_ft = eval_on_u8_buffer(m_r18_ft, test250_u8)
        scores_scnn_brats = eval_on_u8_buffer(m_scnn_brats, test250_u8)
        del test250_u8

        # Save BraTS patient-split predictions on the 250-patient test set
        df_pred_brats_baseline = df_test_250[["patient_id", "slice_id", "source_file", "label_idx"]].copy()
        df_pred_brats_baseline["score_zeroshot_resnet18"] = scores_zeroshot
        df_pred_brats_baseline["score_brats_patient_split_resnet18"] = scores_r18_imgnet
        df_pred_brats_baseline["score_brats_finetuned_resnet18"] = scores_r18_ft
        df_pred_brats_baseline["score_brats_patient_split_smallcnn"] = scores_scnn_brats
        df_pred_brats_baseline.to_csv(
            pred_csv_path,
            index=False,
            compression="gzip",
        )

    res_zeroshot, boot_zeroshot = fast_binary_metrics_and_bootstrap(y_test250, scores_zeroshot, p_test250)
    res_r18_imgnet, boot_r18_imgnet = fast_binary_metrics_and_bootstrap(y_test250, scores_r18_imgnet, p_test250)
    res_r18_ft, boot_r18_ft = fast_binary_metrics_and_bootstrap(y_test250, scores_r18_ft, p_test250)
    res_scnn_brats, boot_scnn_brats = fast_binary_metrics_and_bootstrap(y_test250, scores_scnn_brats, p_test250)

    def paired_diff(m_new: dict[str, Any], b_new: dict[str, np.ndarray], m_ref: dict[str, Any], b_ref: dict[str, np.ndarray]) -> dict[str, Any]:
        out = {}
        for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc"):
            pt = m_new[k] - m_ref[k]
            d_arr = b_new[k] - b_ref[k]
            low, high = np.percentile(d_arr, [2.5, 97.5])
            out[k] = {
                "point_diff": float(pt),
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "ci_95_str": f"{pt:+.4f} ({low:+.4f} to {high:+.4f})",
            }
        return out

    paired_vs_zeroshot = {
        "resnet18_brats_patient_split_minus_zeroshot": paired_diff(res_r18_imgnet, boot_r18_imgnet, res_zeroshot, boot_zeroshot),
        "resnet18_brats_finetuned_minus_zeroshot": paired_diff(res_r18_ft, boot_r18_ft, res_zeroshot, boot_zeroshot),
        "small_cnn_brats_patient_split_minus_zeroshot": paired_diff(res_scnn_brats, boot_scnn_brats, res_zeroshot, boot_zeroshot),
    }

    print(
        f"  Zero-shot Discovery ResNet-18 on BraTS-250: Acc={res_zeroshot['accuracy']:.4f}, "
        f"Sens={res_zeroshot['sensitivity']:.4f}, Spec={res_zeroshot['specificity']:.4f}, AUC={res_zeroshot['auc']:.4f}",
        flush=True,
    )
    print(
        f"  Patient-Split BraTS ResNet-18 (ImageNet):   Acc={res_r18_imgnet['accuracy']:.4f} ({res_r18_imgnet['bootstrap_ci_95']['accuracy']['ci_95_str']}), "
        f"Sens={res_r18_imgnet['sensitivity']:.4f}, Spec={res_r18_imgnet['specificity']:.4f}, AUC={res_r18_imgnet['auc']:.4f} ({res_r18_imgnet['bootstrap_ci_95']['auc']['ci_95_str']})",
        flush=True,
    )
    print(
        f"  Patient-Split BraTS ResNet-18 (Fine-Tuned): Acc={res_r18_ft['accuracy']:.4f} ({res_r18_ft['bootstrap_ci_95']['accuracy']['ci_95_str']}), "
        f"Sens={res_r18_ft['sensitivity']:.4f}, Spec={res_r18_ft['specificity']:.4f}, AUC={res_r18_ft['auc']:.4f} ({res_r18_ft['bootstrap_ci_95']['auc']['ci_95_str']})",
        flush=True,
    )
    print(
        f"  Patient-Split BraTS SmallCNN (Scratch):     Acc={res_scnn_brats['accuracy']:.4f} ({res_scnn_brats['bootstrap_ci_95']['accuracy']['ci_95_str']}), "
        f"Sens={res_scnn_brats['sensitivity']:.4f}, Spec={res_scnn_brats['specificity']:.4f}, AUC={res_scnn_brats['auc']:.4f} ({res_scnn_brats['bootstrap_ci_95']['auc']['ci_95_str']})",
        flush=True,
    )

    # =========================================================================
    # PART 2: PATIENT-DECONTAMINATED DISCOVERY BASELINE & CHENG EVALUATION
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("PART 2: PATIENT-DECONTAMINATED DISCOVERY RESNET-18 BASELINE", flush=True)
    print("======================================================================", flush=True)

    df_disc = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "split_manifest.csv")
    overlap_audit = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8")
    )
    high_sim_edges = overlap_audit["matches"]["high_similarity_matches"]

    # Map each matched Discovery file to its Cheng patient_id(s)
    disc_file_to_pids: dict[str, set[str]] = {}
    for edge in high_sim_edges:
        f_rel = edge["local_source_file"]
        disc_file_to_pids.setdefault(f_rel, set()).add(edge["patient_id"])

    test_cheng_pids = set()
    test_matched_files = set()
    for edge in high_sim_edges:
        if edge["local_split"] == "test":
            test_cheng_pids.add(edge["patient_id"])
            test_matched_files.add(edge["local_source_file"])

    # Remove from train and validation any Discovery image matched to a test Cheng patient
    leaked_train_val_files = {
        f_rel
        for f_rel, pids in disc_file_to_pids.items()
        if (pids & test_cheng_pids) and f_rel not in test_matched_files
    }
    df_disc_clean_train = df_disc[(df_disc["split"] == "train") & (~df_disc["source_file"].isin(leaked_train_val_files))].copy()
    df_disc_clean_val = df_disc[(df_disc["split"] == "validation") & (~df_disc["source_file"].isin(leaked_train_val_files))].copy()
    df_disc_test = df_disc[df_disc["split"] == "test"].copy().reset_index(drop=True)

    print(
        f"Discovery Decontamination: removed {len(leaked_train_val_files)} train/val images sharing a patient with test.\n"
        f"  Clean Train: {len(df_disc_clean_train)} (was {len(df_disc[df_disc['split']=='train'])})\n"
        f"  Clean Val:   {len(df_disc_clean_val)} (was {len(df_disc[df_disc['split']=='validation'])})\n"
        f"  Test:        {len(df_disc_test)} (465 clean + 141 Cheng-matched)",
        flush=True,
    )

    ckpt_disc_decontam = models_dir / "resnet18_discovery_decontaminated.pth"
    with ThreadPoolExecutor(max_workers=16) as ex:
        disc_te_imgs = list(ex.map(_load_rgb_pil, [str(ROOT / f) for f in df_disc_test["source_file"]]))

    if ckpt_disc_decontam.is_file():
        print("Loading already-trained Decontaminated Discovery ResNet-18 checkpoint from disk...", flush=True)
        m_disc_decontam, meta_decontam = load_checkpoint(ckpt_disc_decontam, map_location="cpu")
        info_disc_decontam = meta_decontam.get("extra", {})
    else:
        with ThreadPoolExecutor(max_workers=16) as ex:
            disc_tr_imgs = list(ex.map(_load_rgb_pil, [str(ROOT / f) for f in df_disc_clean_train["source_file"]]))
            disc_va_imgs = list(ex.map(_load_rgb_pil, [str(ROOT / f) for f in df_disc_clean_val["source_file"]]))

        m_disc_decontam = make_model("resnet18", pretrained=True)
        m_disc_decontam, info_disc_decontam = train_binary_model(
            m_disc_decontam,
            disc_tr_imgs,
            df_disc_clean_train["label_idx"].astype(int).tolist(),
            disc_va_imgs,
            df_disc_clean_val["label_idx"].astype(int).tolist(),
            device,
            seed=42,
            max_epochs=10,
            batch_size=32,
            lr=1e-4,
            save_path=ckpt_disc_decontam,
            arch_name="resnet18",
        )
        del disc_tr_imgs, disc_va_imgs

    # Evaluate decontaminated Discovery model on Discovery Test (Full 606, Clean 465, Leaked 141)
    m_disc_decontam = m_disc_decontam.to(device)
    te_loader = DataLoader(InMemoryImageDataset(disc_te_imgs, df_disc_test["label_idx"].astype(int).tolist(), training=False), batch_size=32, shuffle=False)
    decontam_te_scores = []
    with torch.inference_mode():
        for x, _ in te_loader:
            probs = torch.softmax(m_disc_decontam(x.to(device)), dim=1)[:, 1].cpu().numpy()
            decontam_te_scores.extend(probs.tolist())
    m_disc_decontam = m_disc_decontam.cpu()
    torch.cuda.empty_cache()
    decontam_te_scores_arr = np.array(decontam_te_scores, dtype=np.float64)
    y_disc_te = df_disc_test["label_idx"].to_numpy(dtype=np.int32)
    is_cheng_matched = df_disc_test["source_file"].isin(test_matched_files).to_numpy()

    def eval_disc_subset(y_sub: np.ndarray, s_sub: np.ndarray) -> dict[str, Any]:
        p_sub = (s_sub >= 0.5).astype(np.int32)
        cm = confusion_matrix(y_sub, p_sub, labels=[0, 1])
        tn, fp = cm[0]
        fn, tp = cm[1]
        has_both = len(np.unique(y_sub)) > 1
        return {
            "n": int(len(y_sub)),
            "accuracy": float(accuracy_score(y_sub, p_sub)),
            "balanced_accuracy": float(balanced_accuracy_score(y_sub, p_sub)) if has_both else None,
            "sensitivity": float(recall_score(y_sub, p_sub, pos_label=1, zero_division=0)),
            "specificity": float(recall_score(y_sub, p_sub, pos_label=0, zero_division=0)) if has_both else None,
            "roc_auc": float(roc_auc_score(y_sub, s_sub)) if has_both else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }

    disc_decontam_results = {
        "training_info": info_disc_decontam,
        "removed_train_val_slices": len(leaked_train_val_files),
        "clean_train_count": len(df_disc_clean_train),
        "clean_val_count": len(df_disc_clean_val),
        "test_full_606": eval_disc_subset(y_disc_te, decontam_te_scores_arr),
        "test_clean_465": eval_disc_subset(y_disc_te[~is_cheng_matched], decontam_te_scores_arr[~is_cheng_matched]),
        "test_cheng_matched_141_now_unseen_patients": eval_disc_subset(y_disc_te[is_cheng_matched], decontam_te_scores_arr[is_cheng_matched]),
    }
    print(
        f"  Decontaminated ResNet-18 on Discovery Test:\n"
        f"    Full (n=606): Acc={disc_decontam_results['test_full_606']['accuracy']:.4f}, AUC={disc_decontam_results['test_full_606']['roc_auc']:.4f}\n"
        f"    Clean (n=465): Acc={disc_decontam_results['test_clean_465']['accuracy']:.4f}, AUC={disc_decontam_results['test_clean_465']['roc_auc']:.4f}\n"
        f"    141 Cheng-Matched Test Images (previously 100.0% leaked -> now unseen patients): "
        f"Sens={disc_decontam_results['test_cheng_matched_141_now_unseen_patients']['sensitivity']:.4f} "
        f"({disc_decontam_results['test_cheng_matched_141_now_unseen_patients']['confusion_matrix'][1][1]}/141)",
        flush=True,
    )
    del disc_te_imgs

    # =========================================================================
    # PART 3: DECODE CHENG ARCHIVE (3,064 SLICES, 233 PATIENTS) FOR:
    #   (A) External Sensitivity of BraTS Patient-Split & Decontaminated Models
    #   (B) Cheng 3-Class Subtype Patient-Split vs Random Slice-Split Baseline
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("PART 3: CHENG 3,064-SLICE EXTERNAL EVALUATION & PATIENT-SPLIT BASELINE", flush=True)
    print("======================================================================", flush=True)

    cheng_zip_path = ROOT / "1512427.zip"
    cheng_records = []
    cheng_pil_images: list[Image.Image] = []
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
                        cheng_pil_images.append(pil_im)
                        cheng_records.append({
                            "cheng_slice": f"{pname}!{mname}",
                            "patient_id": str(pid),
                            "subtype_idx": int(label_num) - 1,  # 0: meningioma, 1: glioma, 2: pituitary
                            "subtype": {1: "meningioma", 2: "glioma", 3: "pituitary"}[int(label_num)],
                        })

    df_cheng = pd.DataFrame(cheng_records)
    print(f"Loaded {len(df_cheng)} Cheng slices across {df_cheng['patient_id'].nunique()} patients.", flush=True)

    # Evaluate binary models on Cheng (3,064 slices, all positive class)
    cheng_loader = DataLoader(InMemoryImageDataset(cheng_pil_images, [1] * len(cheng_pil_images), training=False), batch_size=64, shuffle=False)

    def eval_cheng_sensitivity(model_obj: nn.Module, pid_mask: np.ndarray | None = None) -> dict[str, Any]:
        model_obj = model_obj.to(device)
        model_obj.eval()
        scores_list = []
        with torch.inference_mode():
            for x, _ in cheng_loader:
                probs = torch.softmax(model_obj(x.to(device)), dim=1)[:, 1].cpu().numpy()
                scores_list.extend(probs.tolist())
        model_obj.cpu()
        torch.cuda.empty_cache()
        s_all = np.array(scores_list, dtype=np.float64)
        p_all = df_cheng["patient_id"].to_numpy()
        if pid_mask is not None:
            s_all = s_all[pid_mask]
            p_all = p_all[pid_mask]
        preds = (s_all >= 0.5).astype(np.int32)
        sens = float(np.mean(preds == 1))
        uniq_p = np.array(sorted(np.unique(p_all)))
        n_p = len(uniq_p)
        p_hits = np.array([np.sum(preds[p_all == pid] == 1) for pid in uniq_p], dtype=np.float64)
        p_cnts = np.array([np.sum(p_all == pid) for pid in uniq_p], dtype=np.float64)
        rng_c = np.random.default_rng(42)
        boot_sens = np.zeros(2000, dtype=np.float64)
        for b in range(2000):
            ch = rng_c.integers(0, n_p, size=n_p)
            w = np.bincount(ch, minlength=n_p)
            boot_sens[b] = (w @ p_hits) / (w @ p_cnts)
        low, high = np.percentile(boot_sens, [2.5, 97.5])
        return {
            "n_slices": int(len(s_all)),
            "n_patients": int(n_p),
            "sensitivity": sens,
            "mean_score": float(np.mean(s_all)),
            "ci_95_low": float(low),
            "ci_95_high": float(high),
            "ci_95_str": f"{low:.4f} - {high:.4f}",
        }

    overlapping_cheng_pids = {edge["patient_id"] for edge in high_sim_edges}
    screened_mask = ~df_cheng["patient_id"].isin(overlapping_cheng_pids).to_numpy()

    cheng_external_eval = {
        "resnet18_brats_patient_split_unscreened_3064": eval_cheng_sensitivity(m_r18_imgnet),
        "resnet18_brats_finetuned_unscreened_3064": eval_cheng_sensitivity(m_r18_ft),
        "resnet18_discovery_decontaminated_unscreened_3064": eval_cheng_sensitivity(m_disc_decontam),
        "resnet18_discovery_decontaminated_screened_1689": eval_cheng_sensitivity(m_disc_decontam, screened_mask),
    }

    # Now train 3-class subtype ResNet-18 on Cheng under:
    # (1) Strict Patient-Level Split (70% train = 163 patients, 15% val = 35 patients, 15% test = 35 patients)
    # (2) Random Slice-Level Split (70% train = 2,144 slices, 15% val = 460 slices, 15% test = 460 slices)
    print("\n3B. Training 3-class ResNet-18 on Cheng: Strict Patient-Split vs Random Slice-Split...", flush=True)
    uniq_cheng_pids = sorted(df_cheng["patient_id"].unique())
    rng_ch = np.random.default_rng(42)
    perm_pids = rng_ch.permutation(uniq_cheng_pids)
    ch_tr_pids = set(perm_pids[:163].tolist())
    ch_va_pids = set(perm_pids[163:198].tolist())
    ch_te_pids = set(perm_pids[198:].tolist())

    idx_p_tr = np.where(df_cheng["patient_id"].isin(ch_tr_pids))[0]
    idx_p_va = np.where(df_cheng["patient_id"].isin(ch_va_pids))[0]
    idx_p_te = np.where(df_cheng["patient_id"].isin(ch_te_pids))[0]

    # Random slice split (same seed 42)
    perm_slices = rng_ch.permutation(len(df_cheng))
    idx_s_tr = perm_slices[:2144]
    idx_s_va = perm_slices[2144:2604]
    idx_s_te = perm_slices[2604:]

    def train_and_eval_cheng_3class(tr_idx: np.ndarray, va_idx: np.ndarray, te_idx: np.ndarray, split_name: str) -> dict[str, Any]:
        seed_everything(42)
        tr_im = [cheng_pil_images[i] for i in tr_idx]
        tr_lb = df_cheng.iloc[tr_idx]["subtype_idx"].astype(int).tolist()
        va_im = [cheng_pil_images[i] for i in va_idx]
        va_lb = df_cheng.iloc[va_idx]["subtype_idx"].astype(int).tolist()
        te_im = [cheng_pil_images[i] for i in te_idx]
        te_lb = df_cheng.iloc[te_idx]["subtype_idx"].to_numpy(dtype=np.int32)
        te_pids = df_cheng.iloc[te_idx]["patient_id"].to_numpy()

        model_3c = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model_3c.fc = nn.Linear(model_3c.fc.in_features, 3)
        model_3c = model_3c.to(device)

        tr_ld = DataLoader(InMemoryImageDataset(tr_im, tr_lb, training=True), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42))
        va_ld = DataLoader(InMemoryImageDataset(va_im, va_lb, training=False), batch_size=32, shuffle=False)
        te_ld = DataLoader(InMemoryImageDataset(te_im, te_lb.tolist(), training=False), batch_size=32, shuffle=False)

        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model_3c.parameters(), lr=1e-4, weight_decay=1e-4)
        best_val = -1.0
        best_st = None
        for ep in range(1, 7):
            model_3c.train()
            for x, y in tr_ld:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model_3c(x), y)
                loss.backward()
                opt.step()
            model_3c.eval()
            vp, vt = [], []
            with torch.inference_mode():
                for x, y in va_ld:
                    vp.extend(model_3c(x.to(device)).argmax(dim=1).cpu().tolist())
                    vt.extend(y.tolist())
            v_bacc = float(balanced_accuracy_score(vt, vp))
            if v_bacc > best_val:
                best_val = v_bacc
                best_st = {k: v.detach().cpu().clone() for k, v in model_3c.state_dict().items()}

        assert best_st is not None
        model_3c.load_state_dict(best_st)
        model_3c.eval()
        probs_list = []
        with torch.inference_mode():
            for x, _ in te_ld:
                probs_list.append(torch.softmax(model_3c(x.to(device)), dim=1).cpu().numpy())
        probs_arr = np.concatenate(probs_list, axis=0)
        preds_arr = probs_arr.argmax(axis=1)
        del model_3c
        torch.cuda.empty_cache()

        acc = float(accuracy_score(te_lb, preds_arr))
        bacc = float(balanced_accuracy_score(te_lb, preds_arr))
        f1_m = float(f1_score(te_lb, preds_arr, average="macro"))
        auc_m = float(roc_auc_score(te_lb, probs_arr, multi_class="ovr", average="macro"))

        # Patient-cluster bootstrap (2,000 resamples, seed 42)
        u_p = np.array(sorted(np.unique(te_pids)))
        n_p = len(u_p)
        p_idx_map = [np.where(te_pids == pid)[0] for pid in u_p]
        rng_b = np.random.default_rng(42)
        b_acc, b_bacc, b_f1, b_auc = [], [], [], []
        for _ in range(2000):
            ch = rng_b.integers(0, n_p, size=n_p)
            idx_cat = np.concatenate([p_idx_map[i] for i in ch])
            yb = te_lb[idx_cat]
            pb = preds_arr[idx_cat]
            prb = probs_arr[idx_cat]
            b_acc.append(accuracy_score(yb, pb))
            b_bacc.append(balanced_accuracy_score(yb, pb))
            b_f1.append(f1_score(yb, pb, average="macro"))
            if len(np.unique(yb)) == 3:
                b_auc.append(roc_auc_score(yb, prb, multi_class="ovr", average="macro"))

        def _ci(arr_vals: list[float]) -> dict[str, Any]:
            lo, hi = np.percentile(arr_vals, [2.5, 97.5])
            return {"low": float(lo), "high": float(hi), "ci_95_str": f"{lo:.4f} - {hi:.4f}"}

        print(f"    [{split_name}] Test n={len(te_lb)} ({n_p} patients): Acc={acc:.4f} ({_ci(b_acc)['ci_95_str']}), Macro-BalAcc={bacc:.4f}, Macro-F1={f1_m:.4f}, Macro-AUC={auc_m:.4f}", flush=True)
        return {
            "split_type": split_name,
            "n_train_slices": int(len(tr_idx)),
            "n_val_slices": int(len(va_idx)),
            "n_test_slices": int(len(te_idx)),
            "n_test_patients": int(n_p),
            "accuracy": acc,
            "macro_balanced_accuracy": bacc,
            "macro_f1": f1_m,
            "macro_roc_auc_ovr": auc_m,
            "confusion_matrix": confusion_matrix(te_lb, preds_arr, labels=[0, 1, 2]).tolist(),
            "bootstrap_ci_95": {
                "accuracy": _ci(b_acc),
                "macro_balanced_accuracy": _ci(b_bacc),
                "macro_f1": _ci(b_f1),
                "macro_roc_auc_ovr": _ci(b_auc),
            },
        }

    cheng_patient_split_res = train_and_eval_cheng_3class(idx_p_tr, idx_p_va, idx_p_te, "Strict Patient-Level Split (163/35/35 patients)")
    cheng_slice_split_res = train_and_eval_cheng_3class(idx_s_tr, idx_s_va, idx_s_te, "Random Slice-Level Split (70/15/15 slices)")

    all_baseline_results = {
        "baseline_1_brats_patient_split": {
            "description": (
                f"Strict patient-level split of BraTS (brats_oriented): 850 train patients ({len(df_brats_train)} balanced slices), "
                f"151 validation patients ({len(df_brats_val)} balanced slices), evaluated on all 33,696 slices of the held-out "
                "250-patient stratified test set (paper_evidence/brats_250_patients_manifest.csv). Zero patient overlap."
            ),
            "split_counts": {
                "train_patients": 850,
                "train_slices_sampled": int(len(df_brats_train)),
                "val_patients": 151,
                "val_slices_sampled": int(len(df_brats_val)),
                "test_patients": 250,
                "test_slices_full": int(len(df_test_250)),
            },
            "models_on_brats_250_test": {
                "zeroshot_discovery_resnet18_reference": res_zeroshot,
                "resnet18_brats_patient_split_from_imagenet": {
                    "training_info": info_r18_imgnet,
                    "checkpoint": "paper_evidence/models/resnet18_brats_patient_split.pth",
                    "test_metrics": res_r18_imgnet,
                },
                "resnet18_brats_finetuned_from_discovery": {
                    "training_info": info_r18_ft,
                    "checkpoint": "paper_evidence/models/resnet18_brats_finetuned.pth",
                    "test_metrics": res_r18_ft,
                },
                "small_cnn_brats_patient_split_from_scratch": {
                    "training_info": info_scnn_brats,
                    "checkpoint": "paper_evidence/models/small_cnn_brats_patient_split.pth",
                    "test_metrics": res_scnn_brats,
                },
            },
            "paired_bootstrap_comparisons_vs_zeroshot": paired_vs_zeroshot,
            "external_evaluation_on_cheng": cheng_external_eval,
        },
        "baseline_2_discovery_patient_decontaminated": disc_decontam_results,
        "baseline_3_cheng_3class_patient_vs_slice_split": {
            "strict_patient_split": cheng_patient_split_res,
            "random_slice_split_leaked": cheng_slice_split_res,
            "leakage_optimism_delta": {
                "accuracy_inflation": float(cheng_slice_split_res["accuracy"] - cheng_patient_split_res["accuracy"]),
                "macro_balanced_accuracy_inflation": float(cheng_slice_split_res["macro_balanced_accuracy"] - cheng_patient_split_res["macro_balanced_accuracy"]),
                "macro_f1_inflation": float(cheng_slice_split_res["macro_f1"] - cheng_patient_split_res["macro_f1"]),
                "macro_auc_inflation": float(cheng_slice_split_res["macro_roc_auc_ovr"] - cheng_patient_split_res["macro_roc_auc_ovr"]),
            },
        },
        "runtime_seconds": round(time.time() - t_start, 2),
    }

    out_json = ROOT / "paper_evidence" / "leakage_free_baselines.json"
    out_json.write_text(json.dumps(all_baseline_results, indent=2), encoding="utf-8")
    print(f"\nSaved all leakage-free baseline results to {out_json} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

