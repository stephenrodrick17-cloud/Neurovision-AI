"""
Figure 1: Audit Workflow Flowchart (Section 2.1 Study Design)
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe Okabe-Ito palette
- Sized for 180 mm (7.09 in) full-page width
- Reads all dataset/partition counts dynamically from saved result files on disk
- Does NOT mention the app or 3D view
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    # Load all numbers from saved files on disk
    df_split = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "split_manifest.csv")
    n_total = len(df_split)
    n_train = int((df_split["split"] == "train").sum())
    n_val = int((df_split["split"] == "validation").sum())
    n_test = int((df_split["split"] == "test").sum())

    p1b = json.loads((ROOT / "paper_evidence" / "phase1b_5seeds_internal_summary.json").read_text(encoding="utf-8"))
    n_seeds = int(p1b["resnet18"]["n_seeds"])

    cheng_unscr = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "metrics.json").read_text(encoding="utf-8")
    )
    cheng_slices = int(cheng_unscr["dataset_metadata"]["sample_count"])
    cheng_patients = int(cheng_unscr["dataset_metadata"]["patient_count"])

    cheng_scr = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8_overlap_screened" / "metrics.json").read_text(encoding="utf-8")
    )
    cheng_scr_slices = int(cheng_scr["metrics"]["sample_count"])
    cheng_scr_patients = int(cheng_scr["metrics"]["patient_count"])

    brats_rep = json.loads((ROOT / "paper_evidence" / "brats_oriented_detailed_report.json").read_text(encoding="utf-8"))
    cm_b = brats_rep["headline_metrics"]["confusion_matrix"]
    brats_slices = int(cm_b["tn"] + cm_b["fp"] + cm_b["fn"] + cm_b["tp"])
    brats_patients = int(brats_rep["patient_level_detection"]["total_tumor_patients"])

    gc_rep = json.loads((ROOT / "paper_evidence" / "gradcam_brats_multiseed.json").read_text(encoding="utf-8"))
    gc_tumor_slices = int(gc_rep["n_tumor_slices"])
    gc_patients = int(gc_rep["n_patients"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 8.5,
    })

    # 180 mm = 7.09 inches wide
    fig, ax = plt.subplots(figsize=(7.09, 6.2), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Okabe-Ito colour-blind-safe palette
    c_blue = "#0072B2"
    c_sky = "#56B4E9"
    c_green = "#009E73"
    c_orange = "#E69F00"
    c_vermilion = "#D55E00"
    c_purple = "#CC79A7"
    c_dark = "#1F2937"

    boxes = [
        (
            1,
            88.5,
            f"1. Discovery Cohort: Kaggle Brain Tumor Detection ({n_total:,} images)",
            "Two-dimensional raster brain MRI slices (2,513 tumor / 2,087 non-tumor; JPEG/PNG/TIFF)",
            c_blue,
            "#EFF6FF",
        ),
        (
            2,
            75.0,
            f"2. Image-Level Partition ({n_train:,} train / {n_val:,} validation / {n_test:,} test)",
            "Duplicate & near-duplicate connected components (pHash <= 4 & SSIM >= 0.95) grouped per split; no patient IDs",
            c_blue,
            "#EFF6FF",
        ),
        (
            3,
            61.5,
            f"3. Discovery Training: ResNet-18, SmallCNN, DenseNet-121 x {n_seeds} seeds (seeds 42-46)",
            "Selected by validation balanced accuracy; evaluated on full test (n=606) and Cheng-decontaminated clean test (n=465)",
            c_sky,
            "#F0F9FF",
        ),
        (
            4,
            48.0,
            f"4. Cross-Dataset Overlap Audit vs. Cheng ({cheng_slices:,} slices, {cheng_patients} patients) & BraTS 2023 ({brats_slices:,} slices)",
            "Two-stage pHash + SSIM screen verified with synthetic positive controls (100% recall) and negative controls (0% FPR)",
            c_orange,
            "#FFFBEB",
        ),
        (
            5,
            34.5,
            f"5. Frozen External Validation: Cheng Overlap-Screened ({cheng_scr_slices:,} slices, {cheng_scr_patients} patients) & BraTS T2-FLAIR ({brats_patients:,} patients)",
            "Zero-shot external tests without threshold tuning; orientation sensitivity (rot90, flipLR) & tumor-size stratification",
            c_vermilion,
            "#FEF2F2",
        ),
        (
            6,
            21.0,
            "6. Leakage-Free Reference Models (5 seeds each: seeds 42-46)",
            "Patient-split BraTS (850 train / 151 val / 250 test patients) | Decontaminated discovery model | Cheng patient- vs. slice-split",
            c_green,
            "#ECFDF5",
        ),
        (
            7,
            7.5,
            f"7. Quantitative Grad-CAM Localization Scoring vs. BraTS Ground-Truth Masks ({gc_tumor_slices:,} tumor slices, {gc_patients} patients)",
            "Pointing game (exact & 15 px tolerance), energy-based pointing game (EBPG), Dice@0.5, and adaptive Top-K overlap",
            c_purple,
            "#FDF2F8",
        ),
    ]

    box_w = 92.0
    box_h = 9.6
    x0 = 4.0

    for idx, y_center, title_str, sub_str, border_col, fill_col in boxes:
        y0 = y_center - box_h / 2.0
        patch = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            linewidth=1.5,
            edgecolor=border_col,
            facecolor=fill_col,
            zorder=2,
        )
        ax.add_patch(patch)

        # Left accent bar
        accent = FancyBboxPatch(
            (x0, y0),
            1.6,
            box_h,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            linewidth=0,
            facecolor=border_col,
            zorder=3,
        )
        ax.add_patch(accent)

        ax.text(
            x0 + 3.2,
            y_center + 1.65,
            title_str,
            fontsize=8.3,
            fontweight="bold",
            color=c_dark,
            va="center",
            ha="left",
            zorder=4,
        )
        ax.text(
            x0 + 3.2,
            y_center - 1.85,
            sub_str,
            fontsize=7.3,
            color="#374151",
            va="center",
            ha="left",
            zorder=4,
        )

    # Draw vertical arrows between consecutive boxes
    for i in range(len(boxes) - 1):
        y_top = boxes[i][1] - box_h / 2.0 - 0.3
        y_bot = boxes[i + 1][1] + box_h / 2.0 + 0.3
        arr = FancyArrowPatch(
            (50.0, y_top),
            (50.0, y_bot),
            arrowstyle="-|>,head_length=4.5,head_width=2.8",
            linewidth=1.4,
            color="#4B5563",
            zorder=5,
        )
        ax.add_patch(arr)

    plt.tight_layout(pad=0.2)
    out_png = OUT_DIR / "fig1_workflow.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
