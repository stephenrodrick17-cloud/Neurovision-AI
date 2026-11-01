"""
Step 3: Quantitative Grad-CAM Localization Evaluation on BraTS Ground-Truth Tumor Segmentation Masks

Evaluates Grad-CAM heatmaps (layer4[-1], target_class=1 [tumor], upsampled to 240x240 via bilinear
interpolation matching experiments/gradcam_utils.py) against the ground-truth 2D BraTS segmentation
masks (canonical RAS+ rotated rot90(k=1) to match brats_oriented) across all 16,074 tumor-positive
slices (250 patients) of the held-out 250-patient stratified subset (paper_evidence/brats_250_patients_manifest.csv).

Models compared on the exact same 16,074 tumor-positive slices (250 patients):
1. Primary Zero-Shot Discovery ResNet-18 (BrainTumorAI/models/tumor_classifier_reproducible.pth)
2. Leakage-Free Patient-Split BraTS ResNet-18 trained from ImageNet (paper_evidence/models/resnet18_brats_patient_split.pth)
3. Leakage-Free Patient-Split BraTS ResNet-18 fine-tuned from Discovery (paper_evidence/models/resnet18_brats_finetuned.pth)

Metrics computed per slice and aggregated with 95% Patient-Cluster Bootstrap CIs (2,000 resamples, seed 42):
- Pointing-Game Accuracy (exact tumor mask hit rate: argmax(CAM) inside seg_mask > 0)
- Pointing-Game Accuracy with 15-px tolerance margin (standard Zhang et al. pointing game)
- Energy-Based Pointing Game (EBPG: fraction of total CAM activation energy inside tumor mask)
- Null/Random Baseline (tumor mask area fraction of 240x240 slice and of brain mask) & Enrichment Ratio
- Brain-Mask Energy Fraction (fraction of CAM energy inside brain parenchyma vs background border)
- Spatial Overlap:
  * Dice & IoU at threshold CAM >= 0.5
  * Dice & IoU at threshold CAM >= 0.7
  * Dice & IoU at adaptive Top-K pixels (K = ground-truth tumor voxel count |M|)
  * Soft Dice (continuous overlap between normalized CAM in [0,1] and binary mask M)
- Stratified breakdowns:
  * Detected tumor slices (True Positives, score >= 0.5) vs Missed tumor slices (False Negatives, score < 0.5)
  * Pre-specified tumor-size bins (<=50 voxels, Q1, Q2, Q3, Q4, and post-hoc >=1% brain area)
"""
from __future__ import annotations

import json
import os
import sys
import time
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

from BrainTumorAI.classifier_pipeline import (
    DEFAULT_CHECKPOINT,
    IMAGE_MEAN,
    IMAGE_STD,
    load_checkpoint,
    make_inference_transform,
)
from BrainTumorAI.heatmap_gen import GradCAM


def compute_batched_gradcam_7x7(
    model: nn.Module,
    target_layer: nn.Module,
    batch_tensor: torch.Tensor,
    target_class: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact Grad-CAM 7x7 heatmaps and class-1 probabilities for a batch of images.
    Mathematically identical to BrainTumorAI.heatmap_gen.GradCAM.generate_heatmap.
    """
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_act(_mod: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        activations.append(out)
        if out.requires_grad:
            out.register_hook(gradients.append)

    handle = target_layer.register_forward_hook(save_act)
    model.zero_grad(set_to_none=True)
    try:
        inp = batch_tensor.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(inp)
            probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            logits[:, target_class].sum().backward()
        act = activations[0].detach()   # [B, C, 7, 7]
        grad = gradients[0].detach()    # [B, C, 7, 7]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = F.relu((weights * act).sum(dim=1))       # [B, 7, 7]
        cam_np = cam.cpu().numpy()
        # Normalize each 7x7 map to [0, 1]
        for i in range(cam_np.shape[0]):
            c_min = float(cam_np[i].min())
            c_max = float(cam_np[i].max())
            if c_max > c_min:
                cam_np[i] = (cam_np[i] - c_min) / (c_max - c_min)
            else:
                cam_np[i] = 0.0
        return cam_np, probs
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def patient_cluster_bootstrap_mean(
    values: np.ndarray,
    patient_indices_list: list[np.ndarray],
    patient_weights: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """
    Exact patient-cluster bootstrap for a slice-level mean metric (e.g. Pointing-Game hit rate, EBPG, Dice).
    """
    point_est = float(np.mean(values))
    p_sums = np.array([np.sum(values[idx]) for idx in patient_indices_list], dtype=np.float64)
    p_cnts = np.array([len(idx) for idx in patient_indices_list], dtype=np.float64)
    boot_means = (patient_weights @ p_sums) / np.maximum(1.0, patient_weights @ p_cnts)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return point_est, float(low), float(high), boot_means


def summarize_slice_df(
    df_slices: pd.DataFrame,
    unique_patients: np.ndarray,
    patient_weights: np.ndarray,
    compute_ci: bool = True,
) -> dict[str, Any]:
    if len(df_slices) == 0:
        return {"n_slices": 0, "n_patients": 0}
    p_arr = df_slices["patient_id"].to_numpy()
    n_pat = int(len(np.unique(p_arr)))
    metric_cols = [
        "pointing_hit_exact",
        "pointing_hit_15px",
        "ebpg_tumor",
        "ebpg_brain",
        "enrichment_vs_image",
        "enrichment_vs_brain",
        "dice_05",
        "iou_05",
        "dice_07",
        "iou_07",
        "dice_topk",
        "iou_topk",
        "soft_dice",
        "null_tumor_area_frac_image",
        "null_tumor_area_frac_brain",
    ]
    out: dict[str, Any] = {
        "n_slices": int(len(df_slices)),
        "n_patients": n_pat,
        "metrics": {},
    }
    if compute_ci and len(unique_patients) == 250 and n_pat == 250:
        p_idx_list = [np.where(p_arr == pid)[0] for pid in unique_patients]
        for col in metric_cols:
            vals = df_slices[col].to_numpy(dtype=np.float64)
            pt, lo, hi, _ = patient_cluster_bootstrap_mean(vals, p_idx_list, patient_weights)
            out["metrics"][col] = {
                "mean": pt,
                "median": float(np.median(vals)),
                "ci_95_low": lo,
                "ci_95_high": hi,
                "ci_95_str": f"{lo:.4f} - {hi:.4f}",
            }
    else:
        # Subset where some patients may have 0 slices in bin: resample unique patients present or use full 250 weights
        p_idx_list = [np.where(p_arr == pid)[0] for pid in unique_patients]
        for col in metric_cols:
            vals = df_slices[col].to_numpy(dtype=np.float64)
            pt, lo, hi, _ = patient_cluster_bootstrap_mean(vals, p_idx_list, patient_weights)
            out["metrics"][col] = {
                "mean": pt,
                "median": float(np.median(vals)),
                "ci_95_low": lo,
                "ci_95_high": hi,
                "ci_95_str": f"{lo:.4f} - {hi:.4f}",
            }
    return out


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df_250 = pd.read_csv(ROOT / "paper_evidence" / "brats_250_patients_manifest.csv")
    df_tumor = df_250[df_250["label_idx"] == 1].copy().reset_index(drop=True)
    n_tumor_slices = len(df_tumor)
    unique_patients = np.array(sorted(df_tumor["patient_id"].unique()))
    n_patients = len(unique_patients)
    print(f"Evaluating Grad-CAM across {n_tumor_slices} tumor-positive slices from {n_patients} BraTS patients...", flush=True)

    # Pre-load 224x224 uint8 images AND 240x240 brain masks from brats_oriented PNGs
    brats_root = Path("D:/datasets/prepared/brats_oriented")
    resize_op = make_inference_transform().transforms[0]

    def _load_slice_img(rel_path: str) -> tuple[np.ndarray, np.ndarray]:
        p = brats_root / rel_path
        with Image.open(p) as im:
            arr240 = np.asarray(im.convert("L"), dtype=np.uint8)
            brain_mask = arr240 > 0
            rgb224 = resize_op(im.convert("RGB"))
            u8_224 = np.asarray(rgb224, dtype=np.uint8)[:, :, 0].copy()
            return u8_224, brain_mask

    t_io = time.time()
    cache_npy = brats_root / "cache_250_u8_224.npy"
    if cache_npy.is_file():
        all_u8_224 = np.load(cache_npy)
        tumor_mask_rows = (df_250["label_idx"] == 1).to_numpy()
        imgs_u8_224 = all_u8_224[tumor_mask_rows].copy()
        del all_u8_224
        brain_masks_240 = np.zeros((n_tumor_slices, 240, 240), dtype=bool)
        for i in range(n_tumor_slices):
            brain_masks_240[i] = cv2.resize(imgs_u8_224[i], (240, 240), interpolation=cv2.INTER_NEAREST) > 0
    else:
        with ThreadPoolExecutor(max_workers=16) as ex:
            loaded_pairs = list(ex.map(_load_slice_img, df_tumor["source_file"].astype(str).tolist()))
        imgs_u8_224 = np.stack([p[0] for p in loaded_pairs], axis=0)      # (16074, 224, 224)
        brain_masks_240 = np.stack([p[1] for p in loaded_pairs], axis=0)  # (16074, 240, 240) bool
        del loaded_pairs
    print(f"Loaded {n_tumor_slices} 224x224 images and 240x240 brain masks in {time.time() - t_io:.2f}s", flush=True)

    # Load ground-truth 240x240 segmentation masks for the 250 patients, applying exact rot90(k=1) matching brats_oriented
    seg_root = Path("D:/datasets/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    t_seg = time.time()
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
    print(
        f"Loaded and oriented {n_tumor_slices} 240x240 ground-truth tumor masks in {time.time() - t_seg:.2f}s "
        f"(min voxels={tumor_voxel_counts.min()}, median={np.median(tumor_voxel_counts):.0f}, max={tumor_voxel_counts.max()})",
        flush=True,
    )

    # Precompute 15-px dilated masks for Zhang et al. tolerance pointing game
    kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    seg_masks_dilated15 = np.zeros_like(seg_masks_240)
    for i in range(n_tumor_slices):
        seg_masks_dilated15[i] = cv2.dilate(seg_masks_240[i].astype(np.uint8), kernel_15) > 0

    mean_t = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGE_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    # Verify batched Grad-CAM == single-image GradCAM on first 5 slices
    m_check, _ = load_checkpoint(DEFAULT_CHECKPOINT, map_location=device)
    m_check = m_check.to(device)
    m_check.eval()
    b5 = torch.from_numpy(imgs_u8_224[:5]).to(device, dtype=torch.float32).unsqueeze(1).expand(-1, 3, -1, -1).div(255.0).sub(mean_t).div(std_t)
    cam5_batch, _ = compute_batched_gradcam_7x7(m_check, m_check.layer4[-1], b5, target_class=1)
    single_gc = GradCAM(m_check, m_check.layer4[-1])
    max_diff_gc = 0.0
    for i in range(5):
        c_single = single_gc.generate_heatmap(b5[i : i + 1], target_class=1)
        max_diff_gc = max(max_diff_gc, float(np.max(np.abs(cam5_batch[i] - c_single))))
    print(f"Verified batched Grad-CAM vs single-image GradCAM max abs diff on 5 slices: {max_diff_gc:.2e}", flush=True)
    del m_check

    # Prepare 2,000-resample patient-cluster bootstrap weights (seed 42)
    n_boot = 2000
    rng = np.random.default_rng(42)
    boot_choices = rng.integers(0, n_patients, size=(n_boot, n_patients))
    patient_weights = np.zeros((n_boot, n_patients), dtype=np.int32)
    for b in range(n_boot):
        patient_weights[b] = np.bincount(boot_choices[b], minlength=n_patients)

    models_to_evaluate = [
        ("zeroshot_discovery_resnet18", DEFAULT_CHECKPOINT),
        ("brats_patient_split_resnet18_imagenet", ROOT / "paper_evidence" / "models" / "resnet18_brats_patient_split.pth"),
        ("brats_patient_split_resnet18_finetuned", ROOT / "paper_evidence" / "models" / "resnet18_brats_finetuned.pth"),
    ]

    # Pre-specified quartile boundaries from the full BraTS cohort (488, 1269, 2248)
    q25, q50, q75 = 488.0, 1269.0, 2248.0

    all_model_gradcam_results: dict[str, Any] = {}
    boot_distributions_overall: dict[str, dict[str, np.ndarray]] = {}

    for model_key, ckpt_path in models_to_evaluate:
        if not Path(ckpt_path).is_file():
            print(f"Skipping {model_key}: checkpoint {ckpt_path} not found.", flush=True)
            continue
        print(f"\nEvaluating Grad-CAM for {model_key} ({ckpt_path})...", flush=True)
        t_m = time.time()
        model, _ = load_checkpoint(ckpt_path, map_location=device)
        model = model.to(device)
        model.eval()

        cams_7x7 = np.zeros((n_tumor_slices, 7, 7), dtype=np.float32)
        probs_all = np.zeros(n_tumor_slices, dtype=np.float32)
        batch_sz = 64
        for start_i in range(0, n_tumor_slices, batch_sz):
            end_i = min(n_tumor_slices, start_i + batch_sz)
            b_u8 = torch.from_numpy(imgs_u8_224[start_i:end_i]).to(device, non_blocking=True)
            b_fp = (b_u8.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1) - mean_t) / std_t
            c7, pr = compute_batched_gradcam_7x7(model, model.layer4[-1], b_fp, target_class=1)
            cams_7x7[start_i:end_i] = c7
            probs_all[start_i:end_i] = pr
        del model
        torch.cuda.empty_cache()
        print(f"  Computed {n_tumor_slices} Grad-CAM maps on GPU in {time.time() - t_m:.2f}s. Computing 240x240 mask overlap metrics...", flush=True)

        # Compute per-slice metrics at 240x240 resolution
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

            # 1. Pointing game (peak pixel location)
            peak_flat = int(np.argmax(cam240))
            py, px = divmod(peak_flat, 240)
            pointing_exact[i] = 1.0 if t_mask[py, px] else 0.0
            pointing_15px[i] = 1.0 if seg_masks_dilated15[i, py, px] else 0.0

            # 2. Energy-based pointing game
            total_energy = float(np.sum(cam240))
            if total_energy > 0.0:
                ebpg_tumor[i] = float(np.sum(cam240[t_mask])) / total_energy
                ebpg_brain[i] = float(np.sum(cam240[b_mask])) / total_energy

            # 3. Soft Dice
            inter_soft = float(np.sum(cam240[t_mask]))
            soft_dice[i] = (2.0 * inter_soft) / (total_energy + t_voxels)

            # 4. Thresholded Dice & IoU at 0.5 and 0.7
            for thr, d_arr, j_arr in ((0.5, dice_05, iou_05), (0.7, dice_07, iou_07)):
                pred_m = cam240 >= thr
                p_cnt = int(np.count_nonzero(pred_m))
                inter = int(np.count_nonzero(pred_m & t_mask))
                union = p_cnt + t_voxels - inter
                d_arr[i] = (2.0 * inter) / max(1, p_cnt + t_voxels)
                j_arr[i] = inter / max(1, union)

            # 5. Top-K overlap (where K = t_voxels)
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

        # Store bootstrap distributions for paired comparison across models
        p_arr = df_m["patient_id"].to_numpy()
        p_idx_list = [np.where(p_arr == pid)[0] for pid in unique_patients]
        boot_distributions_overall[model_key] = {}
        for col in ("pointing_hit_exact", "pointing_hit_15px", "ebpg_tumor", "ebpg_brain", "dice_05", "iou_05", "dice_topk", "soft_dice"):
            _, _, _, b_means = patient_cluster_bootstrap_mean(df_m[col].to_numpy(dtype=np.float64), p_idx_list, patient_weights)
            boot_distributions_overall[model_key][col] = b_means

        overall_summary = summarize_slice_df(df_m, unique_patients, patient_weights)
        tp_summary = summarize_slice_df(df_m[df_m["pred"] == 1], unique_patients, patient_weights)
        fn_summary = summarize_slice_df(df_m[df_m["pred"] == 0], unique_patients, patient_weights)

        size_bins_summary = {
            "<=50 voxels (tiny)": summarize_slice_df(df_m[df_m["tumor_voxels"] <= 50], unique_patients, patient_weights),
            "Q1 (1-488 voxels)": summarize_slice_df(df_m[df_m["tumor_voxels"] <= q25], unique_patients, patient_weights),
            "Q2 (489-1269 voxels)": summarize_slice_df(df_m[(df_m["tumor_voxels"] > q25) & (df_m["tumor_voxels"] <= q50)], unique_patients, patient_weights),
            "Q3 (1270-2248 voxels)": summarize_slice_df(df_m[(df_m["tumor_voxels"] > q50) & (df_m["tumor_voxels"] <= q75)], unique_patients, patient_weights),
            "Q4 (>2248 voxels)": summarize_slice_df(df_m[df_m["tumor_voxels"] > q75], unique_patients, patient_weights),
            "Post-hoc: tumor area >= 1% brain area": summarize_slice_df(df_m[df_m["tumor_brain_frac"] >= 0.01], unique_patients, patient_weights),
        }

        all_model_gradcam_results[model_key] = {
            "checkpoint": str(ckpt_path),
            "overall_all_tumor_slices": overall_summary,
            "by_detection_status": {
                "detected_true_positives_score_gte_05": tp_summary,
                "missed_false_negatives_score_lt_05": fn_summary,
            },
            "by_tumor_size_bin": size_bins_summary,
        }

        om = overall_summary["metrics"]
        tpm = tp_summary["metrics"]
        print(
            f"  [{model_key}] Overall (n={n_tumor_slices}):\n"
            f"    Pointing-Game (exact): {om['pointing_hit_exact']['mean']:.4f} ({om['pointing_hit_exact']['ci_95_str']})\n"
            f"    Pointing-Game (15px):  {om['pointing_hit_15px']['mean']:.4f} ({om['pointing_hit_15px']['ci_95_str']})\n"
            f"    EBPG (tumor energy):   {om['ebpg_tumor']['mean']:.4f} ({om['ebpg_tumor']['ci_95_str']}) vs Null={om['null_tumor_area_frac_image']['mean']:.4f}\n"
            f"    EBPG (brain energy):   {om['ebpg_brain']['mean']:.4f} ({om['ebpg_brain']['ci_95_str']})\n"
            f"    Dice@0.5:              {om['dice_05']['mean']:.4f} ({om['dice_05']['ci_95_str']}), Dice@TopK: {om['dice_topk']['mean']:.4f} ({om['dice_topk']['ci_95_str']})\n"
            f"  [{model_key}] Detected TP Slices (n={tp_summary['n_slices']}):\n"
            f"    Pointing-Game (exact): {tpm['pointing_hit_exact']['mean']:.4f} ({tpm['pointing_hit_exact']['ci_95_str']})\n"
            f"    Pointing-Game (15px):  {tpm['pointing_hit_15px']['mean']:.4f} ({tpm['pointing_hit_15px']['ci_95_str']})\n"
            f"    EBPG (tumor energy):   {tpm['ebpg_tumor']['mean']:.4f} ({tpm['ebpg_tumor']['ci_95_str']})\n"
            f"    Dice@0.5:              {tpm['dice_05']['mean']:.4f} ({tpm['dice_05']['ci_95_str']})",
            flush=True,
        )

    # Paired bootstrap comparisons of Grad-CAM localization (Leakage-Free BraTS vs Zero-Shot Discovery)
    paired_gradcam = {}
    if "zeroshot_discovery_resnet18" in boot_distributions_overall:
        for other_k in ("brats_patient_split_resnet18_imagenet", "brats_patient_split_resnet18_finetuned"):
            if other_k in boot_distributions_overall:
                p_comp = {}
                for col in ("pointing_hit_exact", "pointing_hit_15px", "ebpg_tumor", "ebpg_brain", "dice_05", "iou_05", "dice_topk", "soft_dice"):
                    pt_d = (
                        all_model_gradcam_results[other_k]["overall_all_tumor_slices"]["metrics"][col]["mean"]
                        - all_model_gradcam_results["zeroshot_discovery_resnet18"]["overall_all_tumor_slices"]["metrics"][col]["mean"]
                    )
                    d_arr = boot_distributions_overall[other_k][col] - boot_distributions_overall["zeroshot_discovery_resnet18"][col]
                    lo, hi = np.percentile(d_arr, [2.5, 97.5])
                    p_comp[col] = {
                        "point_diff": float(pt_d),
                        "ci_95_low": float(lo),
                        "ci_95_high": float(hi),
                        "ci_95_str": f"{pt_d:+.4f} ({lo:+.4f} to {hi:+.4f})",
                    }
                paired_gradcam[f"{other_k}_minus_zeroshot_discovery"] = p_comp

    final_payload = {
        "cohort": "brats_oriented (250-patient stratified subset, all tumor-positive slices)",
        "n_patients": int(n_patients),
        "n_tumor_slices": int(n_tumor_slices),
        "gradcam_layer": "model.layer4[-1] (7x7 upsampled to 240x240 via bilinear interpolation)",
        "target_class": 1,
        "batched_vs_single_verification_max_abs_diff": max_diff_gc,
        "bootstrap_method": "Patient-cluster bootstrap (2,000 resamples of 250 patients with replacement, seed 42)",
        "models": all_model_gradcam_results,
        "paired_bootstrap_comparisons": paired_gradcam,
        "runtime_seconds": round(time.time() - t0, 2),
    }

    out_path = ROOT / "paper_evidence" / "gradcam_brats_evaluation.json"
    out_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    print(f"\nSaved Grad-CAM evaluation report to {out_path} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

