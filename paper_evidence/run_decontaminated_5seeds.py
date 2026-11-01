"""
Pre-Submission Analysis 3, Analysis 2 (T1-CE Sequence Check), & 5-Seed Grad-CAM Size-Bin Aggregation:
1. Trains seeds 43, 44, 45, 46 (reusing seed 42) of the Patient-Decontaminated Discovery ResNet-18
   (2,752 clean train images, 663 clean val images; 579 train/val slices sharing a Cheng patient with test removed).
   Evaluates all 5 seeds on:
   - Discovery Test Full (n=606), Clean (n=465), and Cheng-Matched (n=141, now unseen patients)
   - Cheng Unscreened (3,064 slices, 233 patients) and Overlap-Screened (1,689 slices, 152 patients)
   - BraTS-250 Held-Out Test (33,696 slices, 250 patients)
   Saves: paper_evidence/decontaminated_discovery_5seeds.json
2. Computes 5-seed (seeds 42-46) Grad-CAM tumor-size bin breakdowns (<=50 voxels, Q1, Q2, Q3, Q4)
   for Model A (resnet18_brats_patient_split) and Model B (resnet18_brats_finetuned) on all 16,074 tumor slices
   of the 250 held-out BraTS patients (required for Fig 6b).
   Saves: paper_evidence/gradcam_brats_multiseed_size_bins.json (and updates paper_evidence/gradcam_brats_multiseed.json)
3. Evaluates the frozen Discovery ResNet-18 on BraTS T1-CE (t1c.nii.gz, oriented rot90(k=1), matching Cheng's T1-CE sequence)
   across the 250 held-out BraTS patients (33,696 slices: 16,074 tumor + 17,622 tumor-free) and documents local disk status
   for standalone healthy-control external cohorts.
   Saves: paper_evidence/healthy_control_and_t1c_check.json
"""
from __future__ import annotations

import hashlib
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

from BrainTumorAI.classifier_pipeline import (
    DEFAULT_CHECKPOINT,
    IMAGE_MEAN,
    IMAGE_STD,
    load_checkpoint,
    make_inference_transform,
    make_model,
    make_training_transform,
    save_checkpoint,
    sha256_file,
)
from BrainTumorAI.evaluate_classifier import (
    compute_metrics,
    stratified_bootstrap_intervals,
    stratified_group_bootstrap_intervals,
)
from BrainTumorAI.external_validation import _decode_cheng_mat
from paper_evidence.run_gradcam_brats_evaluation import compute_batched_gradcam_7x7

T_CRIT_DF4 = 2.7764451051977987


class InMemoryImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, images: list[Image.Image], labels: list[int], training: bool = False) -> None:
        self.images = images
        self.labels = labels
        self.transform = make_training_transform() if training else make_inference_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), int(self.labels[idx])


def seed_all_strict(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _worker_init_fn(worker_id: int) -> None:
    w_seed = torch.initial_seed() % (2**32)
    np.random.seed(w_seed)
    random.seed(w_seed)


def train_decontaminated_seed(
    seed: int,
    train_images: list[Image.Image],
    train_labels: list[int],
    val_images: list[Image.Image],
    val_labels: list[int],
    device: torch.device,
    ckpt_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    seed_all_strict(seed)
    model = make_model("resnet18", pretrained=True)
    train_loader = DataLoader(
        InMemoryImageDataset(train_images, train_labels, training=True),
        batch_size=32,
        shuffle=True,
        num_workers=0,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        InMemoryImageDataset(val_images, val_labels, training=False),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_bal_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    t0 = time.time()
    max_epochs = 10

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
        va_auc = float(roc_auc_score(va_targets, va_scores))
        scheduler.step(va_bal)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_accuracy": tr_acc,
            "train_balanced_accuracy": tr_bal,
            "val_loss": va_loss,
            "val_accuracy": va_acc,
            "val_balanced_accuracy": va_bal,
            "val_roc_auc": va_auc,
        })
        print(
            f"    [resnet18_discovery_decontaminated seed={seed}] Epoch {epoch:02d}/{max_epochs}: "
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
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()
    torch.cuda.empty_cache()

    save_checkpoint(
        model,
        ckpt_path,
        "resnet18",
        {
            "seed": seed,
            "selected_epoch": best_epoch,
            "validation_balanced_accuracy": best_bal_acc,
            "validation_loss": best_val_loss,
        },
    )
    log_payload = {
        "model_key": "resnet18_discovery_decontaminated",
        "architecture": "resnet18",
        "initialization": "imagenet",
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation_balanced_accuracy": best_bal_acc,
        "validation_loss": best_val_loss,
        "training_seconds": round(time.time() - t0, 2),
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "history": history,
    }
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    return log_payload


def agg_5seeds(vals: list[float], seeds: list[int]) -> dict[str, Any]:
    mean_v = float(np.mean(vals))
    sd_v = float(np.std(vals, ddof=1))
    se_v = sd_v / np.sqrt(len(vals))
    t_lo = float(max(0.0, mean_v - T_CRIT_DF4 * se_v))
    t_hi = float(min(1.0, mean_v + T_CRIT_DF4 * se_v))
    return {
        "mean": mean_v,
        "sd_sample": sd_v,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "seed_values": {str(s): float(v) for s, v in zip(seeds, vals)},
        "mean_sd_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
        "t_ci_95_df4": [t_lo, t_hi],
        "t_ci_95_str": f"{t_lo:.4f} - {t_hi:.4f}",
    }


def main() -> None:
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    pe_dir = ROOT / "paper_evidence"
    models_dir = pe_dir / "models"
    logs_dir = pe_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: 5-SEED DECONTAMINATED DISCOVERY RESNET-18 (SEEDS 42-46)
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("PART 1: 5-SEED PATIENT-DECONTAMINATED DISCOVERY RESNET-18 (SEEDS 42-46)", flush=True)
    print("======================================================================", flush=True)

    df_disc = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "split_manifest.csv")
    overlap_audit = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8")
    )
    high_sim_edges = overlap_audit["matches"]["high_similarity_matches"]

    disc_file_to_pids: dict[str, set[str]] = {}
    for edge in high_sim_edges:
        disc_file_to_pids.setdefault(edge["local_source_file"], set()).add(edge["patient_id"])

    test_cheng_pids = {edge["patient_id"] for edge in high_sim_edges if edge["local_split"] == "test"}
    test_matched_files = {edge["local_source_file"] for edge in high_sim_edges if edge["local_split"] == "test"}

    leaked_train_val_files = {
        f_rel
        for f_rel, pids in disc_file_to_pids.items()
        if (pids & test_cheng_pids) and f_rel not in test_matched_files
    }
    df_disc_clean_train = df_disc[(df_disc["split"] == "train") & (~df_disc["source_file"].isin(leaked_train_val_files))].copy()
    df_disc_clean_val = df_disc[(df_disc["split"] == "validation") & (~df_disc["source_file"].isin(leaked_train_val_files))].copy()
    df_disc_test = df_disc[df_disc["split"] == "test"].copy().reset_index(drop=True)

    assert len(leaked_train_val_files) == 579
    assert len(df_disc_clean_train) == 2752
    assert len(df_disc_clean_val) == 663
    assert len(df_disc_test) == 606

    def _load_rgb_pil(path_str: str) -> Image.Image:
        with Image.open(path_str) as im:
            return im.convert("RGB")

    seeds = [42, 43, 44, 45, 46]
    orig_s42_ckpt = models_dir / "resnet18_discovery_decontaminated.pth"
    s42_ckpt = models_dir / "resnet18_discovery_decontaminated_seed42.pth"
    s42_log = logs_dir / "resnet18_discovery_decontaminated_seed42.json"
    if not s42_ckpt.is_file():
        shutil.copy2(orig_s42_ckpt, s42_ckpt)
    if not s42_log.is_file():
        _, meta42 = load_checkpoint(orig_s42_ckpt, map_location="cpu")
        lfb_json = json.loads((pe_dir / "leakage_free_baselines.json").read_text(encoding="utf-8"))
        tr_info42 = lfb_json["baseline_2_discovery_patient_decontaminated"]["training_info"]
        s42_log.write_text(
            json.dumps(
                {
                    "model_key": "resnet18_discovery_decontaminated",
                    "architecture": "resnet18",
                    "initialization": "imagenet",
                    "seed": 42,
                    "selected_epoch": tr_info42.get("selected_epoch", meta42.get("selected_epoch")),
                    "validation_balanced_accuracy": tr_info42.get("validation_balanced_accuracy", meta42.get("validation_balanced_accuracy")),
                    "validation_loss": tr_info42.get("validation_loss", meta42.get("validation_loss")),
                    "reused_from_checkpoint": "paper_evidence/models/resnet18_discovery_decontaminated.pth",
                    "checkpoint_path": "paper_evidence/models/resnet18_discovery_decontaminated_seed42.pth",
                    "checkpoint_sha256": sha256_file(s42_ckpt),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    missing_seeds = [
        s
        for s in (43, 44, 45, 46)
        if not (models_dir / f"resnet18_discovery_decontaminated_seed{s}.pth").is_file()
    ]
    if missing_seeds:
        print(f"Loading 2,752 clean train and 663 clean val Discovery images for seeds {missing_seeds}...", flush=True)
        with ThreadPoolExecutor(max_workers=16) as ex:
            tr_imgs = list(ex.map(_load_rgb_pil, [str(ROOT / f) for f in df_disc_clean_train["source_file"]]))
            va_imgs = list(ex.map(_load_rgb_pil, [str(ROOT / f) for f in df_disc_clean_val["source_file"]]))
        tr_lbls = df_disc_clean_train["label_idx"].astype(int).tolist()
        va_lbls = df_disc_clean_val["label_idx"].astype(int).tolist()
        for s in missing_seeds:
            train_decontaminated_seed(
                seed=s,
                train_images=tr_imgs,
                train_labels=tr_lbls,
                val_images=va_imgs,
                val_labels=va_lbls,
                device=device,
                ckpt_path=models_dir / f"resnet18_discovery_decontaminated_seed{s}.pth",
                log_path=logs_dir / f"resnet18_discovery_decontaminated_seed{s}.json",
            )
        del tr_imgs, va_imgs

    # Evaluate all 5 seeds of resnet18_discovery_decontaminated on Discovery test (606, 465, 141), Cheng (3064, 1689), and BraTS-250 (33696)
    with ThreadPoolExecutor(max_workers=16) as ex:
        disc_te_imgs = list(ex.map(_load_rgb_pil, [str(ROOT / f) for f in df_disc_test["source_file"]]))
    te_loader = DataLoader(InMemoryImageDataset(disc_te_imgs, df_disc_test["label_idx"].astype(int).tolist(), training=False), batch_size=32, shuffle=False)
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
            "f1": float(f1_score(y_sub, p_sub, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_sub, s_sub)) if has_both else None,
            "pr_auc": float(average_precision_score(y_sub, s_sub)) if has_both else None,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }

    # Load Cheng 3,064 slices
    resize_op = make_inference_transform().transforms[0]
    mean_t = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    cheng_zip_path = ROOT / "1512427.zip"
    cheng_records = []
    cheng_u8_list = []
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
                        cheng_u8_list.append(np.asarray(rgb_224, dtype=np.uint8)[:, :, 0].copy())
                        cheng_records.append({"patient_id": str(pid)})
    df_cheng = pd.DataFrame(cheng_records)
    cheng_u8 = np.stack(cheng_u8_list, axis=0)
    del cheng_u8_list

    overlapping_cheng_pids = {edge["patient_id"] for edge in high_sim_edges}
    screened_mask = ~df_cheng["patient_id"].isin(overlapping_cheng_pids).to_numpy()
    p_cheng_all = df_cheng["patient_id"].to_numpy()
    p_cheng_scr = p_cheng_all[screened_mask]
    uniq_ch_all = np.array(sorted(np.unique(p_cheng_all)))
    uniq_ch_scr = np.array(sorted(np.unique(p_cheng_scr)))

    rng_ch_all = np.random.default_rng(42)
    w_ch_all = np.zeros((2000, len(uniq_ch_all)), dtype=np.int32)
    for b in range(2000):
        w_ch_all[b] = np.bincount(rng_ch_all.integers(0, len(uniq_ch_all), size=len(uniq_ch_all)), minlength=len(uniq_ch_all))

    rng_ch_scr = np.random.default_rng(42)
    w_ch_scr = np.zeros((2000, len(uniq_ch_scr)), dtype=np.int32)
    for b in range(2000):
        w_ch_scr[b] = np.bincount(rng_ch_scr.integers(0, len(uniq_ch_scr), size=len(uniq_ch_scr)), minlength=len(uniq_ch_scr))

    def cheng_sens_ci(scores_arr: np.ndarray, pids_arr: np.ndarray, uniq_p: np.ndarray, weights_mat: np.ndarray) -> dict[str, Any]:
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
        }

    # Load BraTS-250 cache
    brats_root = Path("D:/datasets/prepared/brats_oriented")
    df_test250 = pd.read_csv(pe_dir / "brats_250_patients_manifest.csv")
    test250_u8 = np.load(brats_root / "cache_250_u8_224.npy")
    y_test250 = df_test250["label_idx"].to_numpy(dtype=np.int32)

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

    per_seed_decontam: dict[str, Any] = {}
    for s in seeds:
        ckpt_p = models_dir / f"resnet18_discovery_decontaminated_seed{s}.pth"
        log_p = logs_dir / f"resnet18_discovery_decontaminated_seed{s}.json"
        m_obj, _ = load_checkpoint(ckpt_p, map_location="cpu")

        # 1. Discovery Test
        m_obj = m_obj.to(device)
        m_obj.eval()
        te_scores = []
        with torch.inference_mode():
            for x, _ in te_loader:
                probs = torch.softmax(m_obj(x.to(device)), dim=1)[:, 1].cpu().numpy()
                te_scores.extend(probs.tolist())
        m_obj = m_obj.cpu()
        te_scores_arr = np.array(te_scores, dtype=np.float64)

        # 2. Cheng
        c_scores = eval_on_u8_buffer(m_obj, cheng_u8, batch_sz=64)
        ch_unscr = cheng_sens_ci(c_scores, p_cheng_all, uniq_ch_all, w_ch_all)
        ch_scr = cheng_sens_ci(c_scores[screened_mask], p_cheng_scr, uniq_ch_scr, w_ch_scr)

        # 3. BraTS-250
        b250_scores = eval_on_u8_buffer(m_obj, test250_u8, batch_sz=128)
        b250_eval = eval_disc_subset(y_test250, b250_scores)

        log_info = json.loads(log_p.read_text(encoding="utf-8"))
        per_seed_decontam[str(s)] = {
            "seed": s,
            "checkpoint_path": str(ckpt_p.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(ckpt_p),
            "selected_epoch": log_info.get("selected_epoch"),
            "validation_balanced_accuracy": log_info.get("validation_balanced_accuracy"),
            "validation_loss": log_info.get("validation_loss"),
            "test_full_606": eval_disc_subset(y_disc_te, te_scores_arr),
            "test_clean_465": eval_disc_subset(y_disc_te[~is_cheng_matched], te_scores_arr[~is_cheng_matched]),
            "test_cheng_matched_141_now_unseen_patients": eval_disc_subset(y_disc_te[is_cheng_matched], te_scores_arr[is_cheng_matched]),
            "cheng_unscreened_3064": ch_unscr,
            "cheng_overlap_screened_1689": ch_scr,
            "brats_250_oriented": b250_eval,
        }
        r = per_seed_decontam[str(s)]
        print(
            f"  [Decontaminated seed {s}] "
            f"Full606 Acc={r['test_full_606']['accuracy']:.4f} (AUC={r['test_full_606']['roc_auc']:.4f}) | "
            f"Clean465 Acc={r['test_clean_465']['accuracy']:.4f} (AUC={r['test_clean_465']['roc_auc']:.4f}) | "
            f"Matched141 Sens={r['test_cheng_matched_141_now_unseen_patients']['sensitivity']:.4f} | "
            f"Cheng1689 Sens={ch_scr['sensitivity']:.4f} | "
            f"BraTS250 AUC={b250_eval['roc_auc']:.4f}",
            flush=True,
        )

    del disc_te_imgs, cheng_u8

    agg_decontam = {
        "test_full_606": {
            m: agg_5seeds([per_seed_decontam[str(s)]["test_full_606"][m] for s in seeds], seeds)
            for m in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
        },
        "test_clean_465": {
            m: agg_5seeds([per_seed_decontam[str(s)]["test_clean_465"][m] for s in seeds], seeds)
            for m in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
        },
        "test_cheng_matched_141_now_unseen_patients": {
            "sensitivity": agg_5seeds([per_seed_decontam[str(s)]["test_cheng_matched_141_now_unseen_patients"]["sensitivity"] for s in seeds], seeds),
            "tp_count": agg_5seeds([float(per_seed_decontam[str(s)]["test_cheng_matched_141_now_unseen_patients"]["confusion_matrix"][1][1]) for s in seeds], seeds),
        },
        "cheng_unscreened_3064": {
            "sensitivity": agg_5seeds([per_seed_decontam[str(s)]["cheng_unscreened_3064"]["sensitivity"] for s in seeds], seeds),
        },
        "cheng_overlap_screened_1689": {
            "sensitivity": agg_5seeds([per_seed_decontam[str(s)]["cheng_overlap_screened_1689"]["sensitivity"] for s in seeds], seeds),
        },
        "brats_250_oriented": {
            m: agg_5seeds([per_seed_decontam[str(s)]["brats_250_oriented"][m] for s in seeds], seeds)
            for m in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc")
        },
    }

    decontam_payload = {
        "analysis": "5-seed (seeds 42-46) retraining of Patient-Decontaminated Discovery ResNet-18 (Table 12 update)",
        "removed_train_val_slices": 579,
        "clean_train_count": 2752,
        "clean_val_count": 663,
        "test_full_count": 606,
        "test_clean_count": 465,
        "test_cheng_matched_count": 141,
        "seeds": seeds,
        "per_seed": per_seed_decontam,
        "aggregate_5seed": agg_decontam,
    }
    decontam_out = pe_dir / "decontaminated_discovery_5seeds.json"
    decontam_out.write_text(json.dumps(decontam_payload, indent=2), encoding="utf-8")
    print(f"Saved {decontam_out}", flush=True)

    # =========================================================================
    # PART 2: 5-SEED GRAD-CAM TUMOR-SIZE BIN BREAKDOWN (<=50, Q1-Q4) FOR FIG 6B
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("PART 2: 5-SEED GRAD-CAM TUMOR-SIZE BINS (<=50, Q1-Q4) FOR MODELS A & B", flush=True)
    print("======================================================================", flush=True)

    tumor_mask_rows = (df_test250["label_idx"] == 1).to_numpy()
    df_tumor = df_test250[tumor_mask_rows].copy().reset_index(drop=True)
    n_tumor_slices = len(df_tumor)
    imgs_tumor_u8_224 = test250_u8[tumor_mask_rows].copy()
    del test250_u8

    unique_patients = np.array(sorted(df_tumor["patient_id"].unique()))
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
    kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    seg_masks_dilated15 = np.zeros_like(seg_masks_240)
    for i in range(n_tumor_slices):
        seg_masks_dilated15[i] = cv2.dilate(seg_masks_240[i].astype(np.uint8), kernel_15) > 0

    q25, q50, q75 = 488.0, 1269.0, 2248.0
    bin_masks = {
        "<=50 voxels (tiny)": (tumor_voxel_counts <= 50),
        "Q1 (1-488 voxels)": (tumor_voxel_counts <= q25),
        "Q2 (489-1269 voxels)": (tumor_voxel_counts > q25) & (tumor_voxel_counts <= q50),
        "Q3 (1270-2248 voxels)": (tumor_voxel_counts > q50) & (tumor_voxel_counts <= q75),
        "Q4 (>2248 voxels)": (tumor_voxel_counts > q75),
    }

    existing_gc_eval = json.loads((pe_dir / "gradcam_brats_evaluation.json").read_text(encoding="utf-8"))
    existing_gc_multi = json.loads((pe_dir / "gradcam_brats_multiseed.json").read_text(encoding="utf-8"))

    seed42_size_bins = {
        "resnet18_brats_patient_split": existing_gc_eval["models"]["brats_patient_split_resnet18_imagenet"]["by_tumor_size_bin"],
        "resnet18_brats_finetuned": existing_gc_eval["models"]["brats_patient_split_resnet18_finetuned"]["by_tumor_size_bin"],
    }

    size_bin_multiseed: dict[str, Any] = {
        "zeroshot_discovery_resnet18": existing_gc_eval["models"]["zeroshot_discovery_resnet18"]["by_tumor_size_bin"]
    }

    for mkey in ("resnet18_brats_patient_split", "resnet18_brats_finetuned"):
        per_seed_bins: dict[str, Any] = {}
        for s in seeds:
            if s == 42:
                per_seed_bins["42"] = {
                    b_name: {
                        "n_slices": int(seed42_size_bins[mkey][b_name]["n_slices"]),
                        "pointing_hit_exact": float(seed42_size_bins[mkey][b_name]["metrics"]["pointing_hit_exact"]["mean"]),
                        "pointing_hit_15px": float(seed42_size_bins[mkey][b_name]["metrics"]["pointing_hit_15px"]["mean"]),
                        "ebpg_tumor": float(seed42_size_bins[mkey][b_name]["metrics"]["ebpg_tumor"]["mean"]),
                        "dice_05": float(seed42_size_bins[mkey][b_name]["metrics"]["dice_05"]["mean"]),
                        "dice_topk": float(seed42_size_bins[mkey][b_name]["metrics"]["dice_topk"]["mean"]),
                    }
                    for b_name in bin_masks
                }
                continue

            ckpt_p = models_dir / f"{mkey}_seed{s}.pth"
            model, _ = load_checkpoint(ckpt_p, map_location=device)
            model = model.to(device)
            model.eval()
            cams_7x7 = np.zeros((n_tumor_slices, 7, 7), dtype=np.float32)
            batch_sz = 64
            for start_i in range(0, n_tumor_slices, batch_sz):
                end_i = min(n_tumor_slices, start_i + batch_sz)
                b_u8 = torch.from_numpy(imgs_tumor_u8_224[start_i:end_i]).to(device, non_blocking=True)
                b_fp = (b_u8.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1) - mean_t) / std_t
                c7, _ = compute_batched_gradcam_7x7(model, model.layer4[-1], b_fp, target_class=1)
                cams_7x7[start_i:end_i] = c7
            del model
            torch.cuda.empty_cache()

            p_exact = np.zeros(n_tumor_slices, dtype=np.float64)
            p_15px = np.zeros(n_tumor_slices, dtype=np.float64)
            ebpg_t = np.zeros(n_tumor_slices, dtype=np.float64)
            d_05 = np.zeros(n_tumor_slices, dtype=np.float64)
            d_topk = np.zeros(n_tumor_slices, dtype=np.float64)

            for i in range(n_tumor_slices):
                c7 = cams_7x7[i]
                if float(c7.max()) <= 0.0:
                    continue
                cam240 = cv2.resize(c7, (240, 240), interpolation=cv2.INTER_LINEAR)
                c_min, c_max = float(cam240.min()), float(cam240.max())
                if c_max > c_min:
                    cam240 = (cam240 - c_min) / (c_max - c_min)
                t_mask = seg_masks_240[i]
                t_voxels = int(tumor_voxel_counts[i])
                peak_flat = int(np.argmax(cam240))
                py, px = divmod(peak_flat, 240)
                p_exact[i] = 1.0 if t_mask[py, px] else 0.0
                p_15px[i] = 1.0 if seg_masks_dilated15[i, py, px] else 0.0
                total_energy = float(np.sum(cam240))
                if total_energy > 0.0:
                    ebpg_t[i] = float(np.sum(cam240[t_mask])) / total_energy
                pred_m = cam240 >= 0.5
                p_cnt = int(np.count_nonzero(pred_m))
                inter = int(np.count_nonzero(pred_m & t_mask))
                d_05[i] = (2.0 * inter) / max(1, p_cnt + t_voxels)
                flat_cam = cam240.ravel()
                flat_mask = t_mask.ravel()
                topk_idx = np.argpartition(flat_cam, -t_voxels)[-t_voxels:]
                d_topk[i] = int(np.count_nonzero(flat_mask[topk_idx])) / float(t_voxels)

            per_seed_bins[str(s)] = {
                b_name: {
                    "n_slices": int(np.sum(b_mask)),
                    "pointing_hit_exact": float(np.mean(p_exact[b_mask])),
                    "pointing_hit_15px": float(np.mean(p_15px[b_mask])),
                    "ebpg_tumor": float(np.mean(ebpg_t[b_mask])),
                    "dice_05": float(np.mean(d_05[b_mask])),
                    "dice_topk": float(np.mean(d_topk[b_mask])),
                }
                for b_name, b_mask in bin_masks.items()
            }
            print(f"  Computed Grad-CAM size bins for {mkey} seed {s}", flush=True)

        agg_bins = {}
        for b_name, b_mask in bin_masks.items():
            agg_bins[b_name] = {
                "n_slices": int(np.sum(b_mask)),
                "metrics": {
                    mk: agg_5seeds([per_seed_bins[str(s)][b_name][mk] for s in seeds], seeds)
                    for mk in ("pointing_hit_exact", "pointing_hit_15px", "ebpg_tumor", "dice_05", "dice_topk")
                },
            }
        size_bin_multiseed[mkey] = {
            "per_seed": per_seed_bins,
            "aggregate_5seed_by_size_bin": agg_bins,
        }
        existing_gc_multi["models_multiseed"][mkey]["by_tumor_size_bin_5seed"] = agg_bins

    (pe_dir / "gradcam_brats_multiseed_size_bins.json").write_text(json.dumps(size_bin_multiseed, indent=2), encoding="utf-8")
    (pe_dir / "gradcam_brats_multiseed.json").write_text(json.dumps(existing_gc_multi, indent=2), encoding="utf-8")
    print("Saved paper_evidence/gradcam_brats_multiseed_size_bins.json and updated gradcam_brats_multiseed.json", flush=True)

    del imgs_tumor_u8_224, seg_masks_240, seg_masks_dilated15

    # =========================================================================
    # PART 3: BRA-TS T1-CE (t1c.nii.gz) SEQUENCE-MATCHED EVALUATION (250 PATIENTS)
    # =========================================================================
    print("\n======================================================================", flush=True)
    print("PART 3: BraTS T1-CE (t1c.nii.gz) SEQUENCE-MATCHED EVALUATION (250 PTS)", flush=True)
    print("======================================================================", flush=True)
    t_t1c = time.time()

    all_p250_to_rows: dict[str, list[tuple[int, int]]] = {}
    for r_i, row in df_test250.iterrows():
        all_p250_to_rows.setdefault(str(row["patient_id"]), []).append((int(r_i), int(row["slice_id"])))

    t1c_u8_224 = np.zeros((len(df_test250), 224, 224), dtype=np.uint8)

    def _load_patient_t1c(pid: str) -> list[tuple[int, np.ndarray]]:
        t1c_path = seg_root / pid / f"{pid}-t1c.nii.gz"
        img_nii = nib.as_closest_canonical(nib.load(str(t1c_path)))
        vol = np.asarray(img_nii.dataobj, dtype=np.float32)
        nz = vol[vol > 0]
        if len(nz) > 0:
            p1, p99 = np.percentile(nz, [1.0, 99.0])
        else:
            p1, p99 = 0.0, 1.0
        denom = max(1e-6, float(p99 - p1))
        out = []
        for r_i, s_id in all_p250_to_rows[pid]:
            sl = np.rot90(vol[:, :, s_id], 1)
            mask_pos = sl > 0
            scaled = np.clip((sl - p1) / denom, 0.0, 1.0) * 255.0
            scaled[~mask_pos] = 0.0
            u8_240 = np.rint(scaled).astype(np.uint8)
            pil_rgb = resize_op(Image.fromarray(u8_240, mode="L").convert("RGB"))
            out.append((r_i, np.asarray(pil_rgb, dtype=np.uint8)[:, :, 0].copy()))
        return out

    with ThreadPoolExecutor(max_workers=16) as ex:
        for items in ex.map(_load_patient_t1c, sorted(all_p250_to_rows.keys())):
            for r_i, arr224 in items:
                t1c_u8_224[r_i] = arr224

    m_zs, _ = load_checkpoint(DEFAULT_CHECKPOINT, map_location="cpu")
    scores_t1c_zs = eval_on_u8_buffer(m_zs, t1c_u8_224, batch_sz=128)
    del m_zs, t1c_u8_224

    t1c_metrics = eval_disc_subset(y_test250, scores_t1c_zs)
    print(
        f"  Frozen Discovery ResNet-18 on BraTS-250 T1-CE (t1c, oriented, n=33,696): "
        f"Acc={t1c_metrics['accuracy']:.4f}, BalAcc={t1c_metrics['balanced_accuracy']:.4f}, "
        f"Sens={t1c_metrics['sensitivity']:.4f}, Spec={t1c_metrics['specificity']:.4f}, "
        f"AUC={t1c_metrics['roc_auc']:.4f}, PR-AUC={t1c_metrics['pr_auc']:.4f} ({time.time() - t_t1c:.1f}s)",
        flush=True,
    )

    t1c_payload = {
        "analysis": "Pre-Submission Analysis 2: Healthy-control external cohort audit & BraTS T1-CE (t1c) sequence-matched evaluation",
        "standalone_healthy_control_dataset_on_local_disk": {
            "status": "NOT VERIFIED (No standalone healthy-volunteer brain MRI dataset such as IXI or OASIS is present on local disk D:/datasets or workspace)",
            "checked_paths": ["D:/datasets", "c:/Users/asus/Downloads/Brain Imaging"],
        },
        "sequence_matched_brats_t1c_evaluation_250_patients": {
            "description": (
                "Evaluates the frozen primary Discovery ResNet-18 on T1-weighted contrast-enhanced (t1c.nii.gz) BraTS 2023 volumes "
                "(oriented rot90(k=1), 1st-99th percentile non-zero brain intensity scaling matching prepare_oriented_brats.py) "
                "across the exact same 250 held-out patients (33,696 slices: 16,074 tumor, 17,622 tumor-free). "
                "Because Cheng (2015) is a T1-CE cohort, comparing BraTS T1-CE (t1c) vs BraTS T2-FLAIR (t2f) on the same 250 patients "
                "directly tests whether MRI sequence mismatch (T2-FLAIR vs T1-CE) explains the external performance drop."
            ),
            "brats_250_t1c_oriented_metrics": t1c_metrics,
            "brats_250_t2f_oriented_reference_metrics": json.loads((pe_dir / "leakage_free_multiseed.json").read_text(encoding="utf-8"))["zeroshot_discovery_resnet18_reference"],
        },
    }
    (pe_dir / "healthy_control_and_t1c_check.json").write_text(json.dumps(t1c_payload, indent=2), encoding="utf-8")
    print("Saved paper_evidence/healthy_control_and_t1c_check.json", flush=True)


if __name__ == "__main__":
    main()
