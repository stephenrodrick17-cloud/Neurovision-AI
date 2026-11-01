"""
Figure 7: Summary of ROC-AUC Across Internal, Frozen External, and Patient-Split BraTS Settings (Section 3.9 Summary)
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe Okabe-Ito palette
- Sized for 180 mm (7.09 in) or 85 mm column width
- Dot/bar chart of ROC-AUC:
  1. Internal test (0.9996)
  2. Internal clean test (0.9994)
  3. BraTS oriented (0.7067)
  4. BraTS unrotated (0.6623)
  5. Patient-split BraTS ResNet-18 ImageNet-init (0.9738 +/- 0.0011) on 250 held-out patients
  6. Patient-split BraTS ResNet-18 Fine-tuned (0.9709 +/- 0.0014) on 250 held-out patients
- Note under the figure: "Note: Cheng has no negatives, so no AUC."
All numbers read directly from saved result files on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 8.0,
    })

    # Load exact numbers from saved files on disk
    eval_int = json.loads((ROOT / "BrainTumorAI" / "results" / "evaluation" / "evaluation_metrics.json").read_text(encoding="utf-8"))
    clean_int = json.loads((ROOT / "paper_evidence" / "phase3_clean_test_rescore.json").read_text(encoding="utf-8"))
    brats_ori = json.loads((ROOT / "paper_evidence" / "brats_oriented_detailed_report.json").read_text(encoding="utf-8"))
    brats_comp = json.loads((ROOT / "paper_evidence" / "brats_gpu_comparison_report.json").read_text(encoding="utf-8"))
    multiseed = json.loads((ROOT / "paper_evidence" / "leakage_free_multiseed.json").read_text(encoding="utf-8"))

    auc_int = float(eval_int["models"]["resnet18"]["roc_auc"])
    ci_int_lo = float(eval_int["models"]["resnet18"]["confidence_intervals_duplicate_group_stratified_bootstrap_95"]["roc_auc"]["lower_95"])
    ci_int_hi = float(eval_int["models"]["resnet18"]["confidence_intervals_duplicate_group_stratified_bootstrap_95"]["roc_auc"]["upper_95"])

    auc_clean = float(clean_int["resnet18"]["clean"]["roc_auc"])
    t2_boot_path = ROOT / "paper_evidence" / "table2_bootstrap_seed0_2000.json"
    if t2_boot_path.is_file():
        t2_boot = json.loads(t2_boot_path.read_text(encoding="utf-8"))
        ci_clean_dict = t2_boot["models"]["resnet18"]["test_clean_465"]["duplicate_group_stratified_95_ci"]["replicates_1000_seed_42"]["roc_auc"]
        ci_clean_lo = float(ci_clean_dict["lower_95"])
        ci_clean_hi = float(ci_clean_dict["upper_95"])
    else:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import pandas as pd
        from BrainTumorAI.evaluate_classifier import stratified_group_bootstrap_intervals
        df_p = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "evaluation" / "test_predictions.csv")
        ov_a = json.loads((ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8"))
        ch_files = {e["local_source_file"] for e in ov_a["matches"]["high_similarity_matches"] if e["local_split"] == "test"}
        sub_c = df_p[(df_p["model"] == "resnet18") & (~df_p["source_file"].isin(ch_files))].reset_index(drop=True)
        ci_c = stratified_group_bootstrap_intervals(
            sub_c["true_label_idx"].to_numpy(dtype=np.int64),
            sub_c["tumor_score"].to_numpy(dtype=np.float64),
            sub_c["duplicate_group"].astype(str).to_numpy(),
            threshold=0.5, seed=42, replicates=1000,
        )["roc_auc"]
        ci_clean_lo = float(ci_c["lower_95"])
        ci_clean_hi = float(ci_c["upper_95"])

    auc_b_ori = float(brats_ori["headline_metrics"]["roc_auc"])
    ci_b_ori_lo = float(brats_ori["bootstrap_ci_95"]["roc_auc"]["low"])
    ci_b_ori_hi = float(brats_ori["bootstrap_ci_95"]["roc_auc"]["high"])

    auc_b_unrot = float(rep_comp["cohorts"]["brats_old"]["metrics"]["roc_auc"]) if "rep_comp" in locals() else float(brats_comp["cohorts"]["brats_old"]["metrics"]["roc_auc"])
    ci_b_unrot_lo, ci_b_unrot_hi = [float(x) for x in brats_comp["cohorts"]["brats_old"]["bootstrap_ci_95"]["roc_auc"]]

    mA_agg = multiseed["models"]["resnet18_brats_patient_split"]["aggregate_5seed"]["roc_auc"]
    auc_mA = float(mA_agg["mean"])
    sd_mA = float(mA_agg["sd_sample"])
    ci_mA_lo, ci_mA_hi = [float(x) for x in mA_agg["patient_bootstrap_ci_95"]]

    mB_agg = multiseed["models"]["resnet18_brats_finetuned"]["aggregate_5seed"]["roc_auc"]
    auc_mB = float(mB_agg["mean"])
    sd_mB = float(mB_agg["sd_sample"])
    ci_mB_lo, ci_mB_hi = [float(x) for x in mB_agg["patient_bootstrap_ci_95"]]

    entries = [
        ("Internal Test\n(n = 606)", auc_int, auc_int - ci_int_lo, ci_int_hi - auc_int, f"{auc_int:.4f}", "#0072B2"),
        ("Internal Clean Test\n(n = 465, Cheng-screened)", auc_clean, auc_clean - ci_clean_lo, ci_clean_hi - auc_clean, f"{auc_clean:.4f}", "#56B4E9"),
        ("BraTS Oriented (rot90)\n(Frozen Zero-Shot, 169,066 slices)", auc_b_ori, auc_b_ori - ci_b_ori_lo, ci_b_ori_hi - auc_b_ori, f"{auc_b_ori:.4f}", "#E69F00"),
        ("BraTS Unrotated\n(Frozen Zero-Shot, 169,066 slices)", auc_b_unrot, auc_b_unrot - ci_b_unrot_lo, ci_b_unrot_hi - auc_b_unrot, f"{auc_b_unrot:.4f}", "#D55E00"),
        ("Patient-Split BraTS ResNet-18\n(ImageNet-init, 5 seeds, 250 pts)", auc_mA, auc_mA - ci_mA_lo, ci_mA_hi - auc_mA, f"{auc_mA:.4f} +/- {sd_mA:.4f}", "#009E73"),
        ("Patient-Split BraTS ResNet-18\n(Fine-tuned, 5 seeds, 250 pts)", auc_mB, auc_mB - ci_mB_lo, ci_mB_hi - auc_mB, f"{auc_mB:.4f} +/- {sd_mB:.4f}", "#CC79A7"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(entries))[::-1]
    vals = [e[1] for e in entries]
    err_lo = [e[2] for e in entries]
    err_hi = [e[3] for e in entries]
    labels = [e[0] for e in entries]
    val_strs = [e[4] for e in entries]
    colors = [e[5] for e in entries]

    bars = ax.barh(
        y_pos,
        vals,
        height=0.56,
        xerr=[err_lo, err_hi],
        capsize=3.2,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.90,
        error_kw=dict(lw=1.1, ecolor="#111827"),
    )
    ax.axvline(0.50, color="#4B5563", linestyle="--", lw=1.1, label="Chance (0.50)")

    for bar, v_str, v_num, hi_e in zip(bars, val_strs, vals, err_hi):
        ax.text(
            v_num + hi_e + 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            v_str,
            va="center",
            ha="left",
            fontsize=7.6,
            fontweight="bold",
            color="#111827",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.8, fontweight="bold")
    ax.set_xlabel("ROC-AUC", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.40, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Note under figure as required by prompt
    fig.text(
        0.02,
        -0.02,
        "Note: Cheng has no negatives, so no AUC.",
        fontsize=7.5,
        style="italic",
        color="#374151",
    )

    plt.tight_layout()
    out_png = OUT_DIR / "fig7_roc_auc_summary.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
