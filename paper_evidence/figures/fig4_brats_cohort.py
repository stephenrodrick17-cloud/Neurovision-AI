"""
Figure 4: Primary Oriented BraTS Cohort Predictions (169,066 slices, 1,251 patients) — Section 3.7.3
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe Okabe-Ito palette
- Sized for 180 mm (7.09 in) column width
- Panel (a): Overlaid histograms of tumor score for tumor-containing (81,413) and tumor-free (87,653) slices,
             10 equal-width bins, shown in both linear count and log10-count scales.
- Panel (b): ROC curves for oriented (rot90) and unrotated versions with AUC in the legend (0.7067, 0.6623).
- Panel (c): Sensitivity by tumor-size bin (<=50, Q1-Q4) with patient-cluster 95% CIs from Table 9.
All numbers read directly from saved result files in paper_evidence/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 7.8,
    })

    # Load saved JSON reports
    rep_oriented = json.loads((ROOT / "paper_evidence" / "brats_oriented_detailed_report.json").read_text(encoding="utf-8"))
    rep_supp = json.loads((ROOT / "paper_evidence" / "brats_supplementary_cIs_and_subgroups.json").read_text(encoding="utf-8"))
    rep_comp = json.loads((ROOT / "paper_evidence" / "brats_gpu_comparison_report.json").read_text(encoding="utf-8"))

    # Load saved per-slice predictions for exact ROC curves
    df_orient = pd.read_csv(ROOT / "paper_evidence" / "brats_oriented_predictions.csv.gz")
    df_unrot = pd.read_csv(ROOT / "paper_evidence" / "brats_old_predictions.csv.gz")

    def _extract_y_s(df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lbl_col = next(c for c in ("label_idx", "true_label_idx", "label", "y_true") if c in df_in.columns)
        scr_col = next(c for c in ("tumor_prob", "tumor_score", "score", "prob") if c in df_in.columns)
        if df_in[lbl_col].dtype == object:
            y_arr = (df_in[lbl_col].astype(str) == "tumor").astype(np.int32).to_numpy()
        else:
            y_arr = df_in[lbl_col].to_numpy(dtype=np.int32)
        s_arr = df_in[scr_col].to_numpy(dtype=np.float64)
        return y_arr, s_arr

    y_orient, s_orient = _extract_y_s(df_orient)
    y_unrot, s_unrot = _extract_y_s(df_unrot)

    fpr_o, tpr_o, _ = roc_curve(y_orient, s_orient)
    fpr_u, tpr_u, _ = roc_curve(y_unrot, s_unrot)

    auc_o = float(rep_oriented["headline_metrics"]["roc_auc"])
    auc_u = float(rep_comp["cohorts"]["brats_old"]["metrics"]["roc_auc"])

    # Histograms (10 equal-width bins)
    bin_keys = [
        "[0.0, 0.1)", "[0.1, 0.2)", "[0.2, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)",
        "[0.5, 0.6)", "[0.6, 0.7)", "[0.7, 0.8)", "[0.8, 0.9)", "[0.9, 1.0]",
    ]
    tumor_counts = np.array([int(rep_oriented["histograms"]["tumor"]["histogram"][k]) for k in bin_keys], dtype=np.float64)
    notumor_counts = np.array([int(rep_oriented["histograms"]["no_tumor"]["histogram"][k]) for k in bin_keys], dtype=np.float64)
    assert int(tumor_counts.sum()) == 81413
    assert int(notumor_counts.sum()) == 87653

    bin_centers = np.linspace(0.05, 0.95, 10)
    bar_w = 0.040

    fig = plt.figure(figsize=(7.09, 5.8), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0], hspace=0.38, wspace=0.28)

    # ---- Panel (a): Top row spanning both columns (Left: linear counts; Right: log-scale counts) ----
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :], wspace=0.25)
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    c_tumor = "#D55E00"    # Okabe-Ito vermilion
    c_notumor = "#0072B2"  # Okabe-Ito blue

    ax_a1.bar(bin_centers - bar_w / 2, notumor_counts / 1000.0, width=bar_w, color=c_notumor, alpha=0.88, edgecolor="black", lw=0.4,
              label=f"Tumor-free ({int(notumor_counts.sum()):,})")
    ax_a1.bar(bin_centers + bar_w / 2, tumor_counts / 1000.0, width=bar_w, color=c_tumor, alpha=0.88, edgecolor="black", lw=0.4,
              label=f"Tumor-containing ({int(tumor_counts.sum()):,})")
    ax_a1.axvline(0.5, color="#111827", linestyle="--", lw=1.0, label="Threshold = 0.5")
    ax_a1.set_xlabel("Predicted Tumor Score", fontweight="bold")
    ax_a1.set_ylabel("Slice Count (x 1,000)", fontweight="bold")
    ax_a1.set_xlim(-0.02, 1.02)
    ax_a1.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_a1.spines["top"].set_visible(False)
    ax_a1.spines["right"].set_visible(False)
    ax_a1.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a1.legend(loc="upper right", frameon=True, fontsize=6.6)
    ax_a1.set_title("(a) Score Distribution (Linear Scale, 10 Bins)", fontsize=8.3, fontweight="bold", loc="left")

    ax_a2.bar(bin_centers - bar_w / 2, notumor_counts, width=bar_w, color=c_notumor, alpha=0.88, edgecolor="black", lw=0.4,
              label="Tumor-free (n=87,653)")
    ax_a2.bar(bin_centers + bar_w / 2, tumor_counts, width=bar_w, color=c_tumor, alpha=0.88, edgecolor="black", lw=0.4,
              label="Tumor-containing (n=81,413)")
    ax_a2.axvline(0.5, color="#111827", linestyle="--", lw=1.0)
    ax_a2.set_yscale("log")
    ax_a2.set_ylim(50, 1.2e5)
    ax_a2.set_xlabel("Predicted Tumor Score", fontweight="bold")
    ax_a2.set_ylabel("Slice Count (Log10 Scale)", fontweight="bold")
    ax_a2.set_xlim(-0.02, 1.02)
    ax_a2.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_a2.spines["top"].set_visible(False)
    ax_a2.spines["right"].set_visible(False)
    ax_a2.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a2.set_title("Score Distribution (Log-Count View)", fontsize=8.3, fontweight="bold", loc="left")

    # ---- Panel (b): Bottom-left: ROC curves for oriented vs unrotated ----
    ax_b = fig.add_subplot(gs[1, 0])
    # Subsample ROC points slightly for crisp vector/raster rendering while keeping exact endpoints
    step_o = max(1, len(fpr_o) // 1500)
    step_u = max(1, len(fpr_u) // 1500)
    ax_b.plot(fpr_o[::step_o], tpr_o[::step_o], color="#0072B2", lw=2.0, label=f"Oriented (AUC = {auc_o:.4f})")
    ax_b.plot(fpr_u[::step_u], tpr_u[::step_u], color="#D55E00", lw=1.8, linestyle="--", label=f"Unrotated (AUC = {auc_u:.4f})")
    ax_b.plot([0, 1], [0, 1], color="#6B7280", lw=1.0, linestyle=":", label="Chance (0.5000)")
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax_b.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower right", frameon=True, fontsize=7.0)
    ax_b.set_title("(b) ROC Curves: Oriented vs. Unrotated", fontsize=8.3, fontweight="bold", loc="left")

    # ---- Panel (c): Bottom-right: Sensitivity by tumor-size bin (<=50, Q1-Q4) with 95% CIs ----
    ax_c = fig.add_subplot(gs[1, 1])
    size_bins_order = [
        ("<=50 voxels (tiny)", "<=50\n(tiny)"),
        ("Q1 (1-488 voxels)", "Q1\n(1-488)"),
        ("Q2 (489-1269 voxels)", "Q2\n(489-1269)"),
        ("Q3 (1270-2248 voxels)", "Q3\n(1270-2248)"),
        ("Q4 (>2248 voxels)", "Q4\n(>2248)"),
    ]
    tb_data = rep_supp["tumor_size_bins_with_ci_and_auc"]
    sens_vals = [float(tb_data[k]["sensitivity"]) for k, _ in size_bins_order]
    ci_lows = [float(tb_data[k]["sensitivity_ci_low"]) for k, _ in size_bins_order]
    ci_highs = [float(tb_data[k]["sensitivity_ci_high"]) for k, _ in size_bins_order]
    yerr_low = [s - lo for s, lo in zip(sens_vals, ci_lows)]
    yerr_high = [hi - s for s, hi in zip(sens_vals, ci_highs)]

    x_pos = np.arange(len(size_bins_order))
    colors_c = ["#9CA3AF", "#56B4E9", "#009E73", "#E69F00", "#0072B2"]
    bars = ax_c.bar(
        x_pos,
        sens_vals,
        width=0.58,
        yerr=[yerr_low, yerr_high],
        capsize=3.5,
        color=colors_c,
        edgecolor="black",
        linewidth=0.6,
        error_kw=dict(lw=1.1, ecolor="#111827"),
    )
    for rect, s_val, hi_val in zip(bars, sens_vals, ci_highs):
        ax_c.text(
            rect.get_x() + rect.get_width() / 2.0,
            hi_val + 0.025,
            f"{s_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
        )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([lbl for _, lbl in size_bins_order], fontsize=7.0)
    ax_c.set_xlabel("Tumor Area Bin (2D Slice Voxels)", fontweight="bold")
    ax_c.set_ylabel("Tumor Sensitivity (95% CI)", fontweight="bold")
    ax_c.set_ylim(0.0, 0.90)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.set_title("(c) Sensitivity by Tumor-Size Bin (Table 9)", fontsize=8.3, fontweight="bold", loc="left")

    out_png = OUT_DIR / "fig4_brats_cohort.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
