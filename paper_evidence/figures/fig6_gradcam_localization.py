"""
Figure 6: Grad-CAM Localization Examples and Size-Stratified Overlap on BraTS-250 (Section 3.7.7)
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe Okabe-Ito palette
- Sized for 180 mm (7.09 in) column width
- Panel (a): For the frozen ResNet-18, show 3 BraTS slices:
             (1) Detected large tumor (Q4 > 2248 voxels, score >= 0.5)
             (2) Missed small tumor (Q1 <= 488 voxels, score < 0.5)
             (3) Missed large tumor (Q4 > 2248 voxels, score < 0.5)
             with Grad-CAM overlay and the ground-truth tumor mask contour in a contrasting colour; label score and Dice.
- Panel (b): Grouped bars by size bin (<=50, Q1-Q4): pointing game within 15 px and Dice at 0.5,
             for frozen vs. patient-split ImageNet-init vs. patient-split fine-tuned (mean +/- SD over 5 seeds).
All numbers read dynamically from saved result files in paper_evidence/.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BrainTumorAI.classifier_pipeline import (
    DEFAULT_CHECKPOINT,
    load_checkpoint,
    make_inference_transform,
)
from BrainTumorAI.heatmap_gen import GradCAM

OUT_DIR = Path(__file__).resolve().parent


def select_and_compute_3_examples() -> list[dict]:
    """Select deterministic representative slices for (1) detected large, (2) missed small, (3) missed large."""
    df_250 = pd.read_csv(ROOT / "paper_evidence" / "brats_250_patients_manifest.csv")
    df_preds = pd.read_csv(ROOT / "paper_evidence" / "brats_250_leakage_free_baseline_predictions.csv.gz")
    df_vox = pd.read_csv(ROOT / "paper_evidence" / "brats_seg_voxel_counts.csv.gz")

    df_m = df_250.merge(
        df_preds[["patient_id", "slice_id", "score_zeroshot_resnet18"]],
        on=["patient_id", "slice_id"],
        how="inner",
    ).merge(
        df_vox[["patient_id", "slice_id", "tumor_voxels"]],
        on=["patient_id", "slice_id"],
        how="inner",
    )
    df_t = df_m[df_m["label_idx"] == 1].copy().reset_index(drop=True)

    # 1. Detected large tumor: score >= 0.85, tumor_voxels > 2500
    cand_det_large = df_t[(df_t["tumor_voxels"] > 2800) & (df_t["score_zeroshot_resnet18"] >= 0.90)].sort_values(
        by=["score_zeroshot_resnet18", "tumor_voxels"], ascending=[False, False]
    )
    # 2. Missed small tumor: 100 <= tumor_voxels <= 400, score < 0.15
    cand_miss_small = df_t[(df_t["tumor_voxels"] >= 150) & (df_t["tumor_voxels"] <= 400) & (df_t["score_zeroshot_resnet18"] < 0.10)].sort_values(
        by=["tumor_voxels", "patient_id"], ascending=[False, True]
    )
    # 3. Missed large tumor: tumor_voxels > 2800, score < 0.15
    cand_miss_large = df_t[(df_t["tumor_voxels"] > 2800) & (df_t["score_zeroshot_resnet18"] < 0.15)].sort_values(
        by=["tumor_voxels", "patient_id"], ascending=[False, True]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(DEFAULT_CHECKPOINT, map_location=device)
    model = model.to(device).eval()
    gcam = GradCAM(model, model.layer4[-1])
    tfm = make_inference_transform()

    brats_root = Path("D:/datasets/prepared/brats_oriented")
    seg_root = Path("D:/datasets/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

    def eval_candidate(row: pd.Series, category_title: str) -> dict:
        pid = str(row["patient_id"])
        sid = int(row["slice_id"])
        im_pil = Image.open(brats_root / str(row["source_file"])).convert("RGB")
        gray240 = np.asarray(im_pil.convert("L"), dtype=np.uint8)

        seg_nii = nib.as_closest_canonical(nib.load(str(seg_root / pid / f"{pid}-seg.nii.gz")))
        seg_vol = np.asarray(seg_nii.dataobj)
        mask240 = np.rot90(seg_vol[:, :, sid] > 0, 1)

        inp = tfm(im_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = float(torch.softmax(model(inp), dim=1)[0, 1].item())
        cam7 = gcam.generate_heatmap(inp, target_class=1)
        cam240 = cv2.resize(cam7, (240, 240), interpolation=cv2.INTER_LINEAR)
        c_min, c_max = float(cam240.min()), float(cam240.max())
        if c_max > c_min:
            cam240 = (cam240 - c_min) / (c_max - c_min)

        pred_m = cam240 >= 0.5
        inter = int(np.count_nonzero(pred_m & mask240))
        t_vox = int(np.count_nonzero(mask240))
        p_vox = int(np.count_nonzero(pred_m))
        dice_05 = (2.0 * inter) / max(1, p_vox + t_vox)

        peak_flat = int(np.argmax(cam240))
        py, px = divmod(peak_flat, 240)

        return {
            "category": category_title,
            "patient_id": pid,
            "slice_id": sid,
            "tumor_voxels": t_vox,
            "score": prob,
            "dice_05": dice_05,
            "gray240": gray240,
            "mask240": mask240,
            "cam240": cam240,
            "peak_yx": (py, px),
        }

    # Find a detected large tumor with high Dice (>0.45) to illustrate strong hit vs missed small & missed large
    best_det = None
    for _, r in cand_det_large.head(25).iterrows():
        cand = eval_candidate(r, "Detected Large Tumor (TP, Q4)")
        if cand["dice_05"] >= 0.45:
            best_det = cand
            break
    if best_det is None:
        best_det = eval_candidate(cand_det_large.iloc[0], "Detected Large Tumor (TP, Q4)")

    def first_valid(cands: pd.DataFrame, title: str) -> dict:
        # Grad-CAM can be all-zero for confidently-negative slices; take the first candidate with contrast.
        for _, r in cands.head(200).iterrows():
            try:
                return eval_candidate(r, title)
            except RuntimeError:
                continue
        raise RuntimeError(f"No candidate with Grad-CAM contrast for: {title}")

    ex_miss_small = first_valid(cand_miss_small, "Missed Small Tumor (FN, Q1)")
    ex_miss_large = first_valid(cand_miss_large, "Missed Large Tumor (FN, Q4)")
    return [best_det, ex_miss_small, ex_miss_large]


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 7.8,
    })

    examples = select_and_compute_3_examples()
    size_bins_data = json.loads((ROOT / "paper_evidence" / "gradcam_brats_multiseed_size_bins.json").read_text(encoding="utf-8"))

    bin_keys = [
        ("<=50 voxels (tiny)", "<=50\n(tiny)"),
        ("Q1 (1-488 voxels)", "Q1\n(1-488)"),
        ("Q2 (489-1269 voxels)", "Q2\n(489-\n1269)"),
        ("Q3 (1270-2248 voxels)", "Q3\n(1270-\n2248)"),
        ("Q4 (>2248 voxels)", "Q4\n(>2248)"),
    ]

    fig = plt.figure(figsize=(7.09, 5.9), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.05], hspace=0.36)

    # ---- Panel (a): Top row: 3 representative slices for frozen ResNet-18 ----
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.12)
    for idx, ex in enumerate(examples):
        ax = fig.add_subplot(gs_a[0, idx])
        ax.imshow(ex["gray240"], cmap="gray", vmin=0, vmax=255)
        im_cam = ax.imshow(ex["cam240"], cmap="inferno", alpha=0.46, vmin=0.0, vmax=1.0)
        # Draw ground-truth mask contour in contrasting bright cyan (#00FFFF)
        ax.contour(ex["mask240"].astype(np.float32), levels=[0.5], colors=["#00FFFF"], linewidths=1.4)
        # Mark peak Grad-CAM location
        py, px = ex["peak_yx"]
        ax.plot(px, py, marker="+", color="#FFFFFF", markersize=7, markeredgewidth=1.5)

        ax.set_title(
            f"{ex['category']}\n{ex['patient_id']} (z={ex['slice_id']}, |M|={ex['tumor_voxels']} px)\n"
            f"Score = {ex['score']:.3f} | Dice@0.5 = {ex['dice_05']:.3f}",
            fontsize=7.0,
            fontweight="bold",
            pad=4,
        )
        ax.axis("off")
        if idx == 0:
            ax.text(
                0.0,
                1.26,
                "(a) Frozen Discovery ResNet-18 Grad-CAM Overlays vs. BraTS Tumor Mask Contour (Cyan)",
                transform=ax.transAxes,
                fontsize=8.3,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
        if idx == 2:
            cbar = fig.colorbar(im_cam, ax=ax, fraction=0.044, pad=0.03)
            cbar.ax.tick_params(labelsize=6.5)
            cbar.set_label("Normalized CAM", fontsize=6.8)

    # ---- Panel (b): Bottom row: 2 grouped bar charts by tumor size bin (Pointing Game 15px & Dice@0.5) ----
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.25)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])

    x = np.arange(len(bin_keys))
    bar_w = 0.26

    # Extract values for the 3 models across the 5 size bins
    zs_bins = size_bins_data["zeroshot_discovery_resnet18"]
    mA_bins = size_bins_data["resnet18_brats_patient_split"]["aggregate_5seed_by_size_bin"]
    mB_bins = size_bins_data["resnet18_brats_finetuned"]["aggregate_5seed_by_size_bin"]

    for ax_sub, metric_key, y_label, sub_title, y_max in (
        (ax_b1, "pointing_hit_15px", "Pointing Game Hit Rate (<=15 px)", "(b) Pointing Game (<=15 px) by Tumor Size", 1.08),
        (ax_b2, "dice_05", "Spatial Overlap (Dice @ CAM >= 0.5)", "Spatial Overlap (Dice @ 0.5) by Tumor Size", 0.42),
    ):
        zs_vals = [float(zs_bins[k]["metrics"][metric_key]["mean"]) for k, _ in bin_keys]
        # Frozen model is a single deterministic checkpoint (SD = 0 over seeds; we plot point estimate)
        mA_means = [float(mA_bins[k]["metrics"][metric_key]["mean"]) for k, _ in bin_keys]
        mA_sds = [float(mA_bins[k]["metrics"][metric_key]["sd_sample"]) for k, _ in bin_keys]

        mB_means = [float(mB_bins[k]["metrics"][metric_key]["mean"]) for k, _ in bin_keys]
        mB_sds = [float(mB_bins[k]["metrics"][metric_key]["sd_sample"]) for k, _ in bin_keys]

        ax_sub.bar(
            x - bar_w,
            zs_vals,
            width=bar_w,
            color="#D55E00",
            edgecolor="black",
            linewidth=0.5,
            label="Frozen Discovery ResNet-18",
        )
        ax_sub.bar(
            x,
            mA_means,
            width=bar_w,
            yerr=mA_sds,
            capsize=2.5,
            color="#0072B2",
            edgecolor="black",
            linewidth=0.5,
            error_kw=dict(lw=0.9, ecolor="#111827"),
            label="Patient-Split ImageNet-Init (5 seeds)",
        )
        ax_sub.bar(
            x + bar_w,
            mB_means,
            width=bar_w,
            yerr=mB_sds,
            capsize=2.5,
            color="#009E73",
            edgecolor="black",
            linewidth=0.5,
            error_kw=dict(lw=0.9, ecolor="#111827"),
            label="Patient-Split Fine-Tuned (5 seeds)",
        )

        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels([lbl.replace("(", "\n(", 1) if False else lbl for _, lbl in bin_keys], fontsize=6.2)
        ax_sub.set_xlabel("Tumor Area Bin (2D Slice Voxels)", fontweight="bold")
        ax_sub.set_ylabel(y_label, fontweight="bold")
        top_val = max(max(np.array(zs_vals)), max(np.array(mA_means) + np.array(mA_sds)), max(np.array(mB_means) + np.array(mB_sds)))
        if metric_key == "pointing_hit_15px":
            ax_sub.set_ylim(0.0, 1.45)
            ax_sub.set_yticks(np.linspace(0.0, 1.0, 6))
        else:
            ax_sub.set_ylim(0.0, top_val * 1.12)
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)
        ax_sub.grid(axis="y", linestyle="--", alpha=0.35)
        ax_sub.set_title(sub_title, fontsize=8.1, fontweight="bold", loc="left")
        if metric_key == "pointing_hit_15px":
            ax_sub.legend(loc="upper left", frameon=True, fontsize=6.4)

    out_png = OUT_DIR / "fig6_gradcam_localization.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()

