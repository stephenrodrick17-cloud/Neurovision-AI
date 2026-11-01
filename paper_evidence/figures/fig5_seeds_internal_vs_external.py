"""
Figure 5: Internal Test Accuracy vs. External BraTS-250 ROC-AUC Across Architectures and Seeds (Section 3.7.5)
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe Okabe-Ito palette
- Sized for 85 mm (3.35 in) / 120 mm single-to-1.5 column width (saved at 5.2 x 4.0 in so it is crystal clear at both 85 mm and 180 mm)
- Dot plot:
  x: internal test accuracy per seed (from paper_evidence/phase1b_5seeds_internal_summary.json)
  y: ROC-AUC on the 250-patient BraTS subset (from paper_evidence/brats_250_5seeds_with_cis.json)
- Three colours (ResNet-18, SmallCNN, DenseNet-121), five dots each (seeds 42-46)
- Marks primary seed-42 ResNet-18 and SmallCNN with a prominent outer ring
- Horizontal reference line at 0.5 (chance) and vertical error bars from per-checkpoint patient-cluster 95% CIs
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

    internal_data = json.loads((ROOT / "paper_evidence" / "phase1b_5seeds_internal_summary.json").read_text(encoding="utf-8"))
    external_data = json.loads((ROOT / "paper_evidence" / "brats_250_5seeds_with_cis.json").read_text(encoding="utf-8"))

    arch_configs = [
        ("resnet18", "resnet18", "ResNet-18 (5 seeds)", "#0072B2", "o"),
        ("small_cnn", "smallcnn", "SmallCNN (5 seeds)", "#D55E00", "s"),
        ("densenet121", "densenet121", "DenseNet-121 (5 seeds)", "#009E73", "^"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    # Horizontal chance reference line at 0.5
    ax.axhline(0.50, color="#4B5563", linestyle="--", lw=1.2, zorder=1, label="Chance ROC-AUC (0.50)")

    # Small horizontal jitter only when two seeds have identical x (e.g. ResNet-18 seeds 45 & 46 both have x=0.981848)
    for arch_key, ext_arch_key, label_str, col, marker_style in arch_configs:
        int_by_seed = {int(r["seed"]): float(r["accuracy"]) for r in internal_data[arch_key]["per_seed"]}
        ext_runs = external_data["architectures"][ext_arch_key]["individual_runs"]

        ext_by_seed = {}
        for r in ext_runs:
            mname = str(r["model_name"])
            if "primary reference" in mname or "small_cnn_best" in mname:
                continue
            for s in (42, 43, 44, 45, 46):
                if f"seed{s}" in mname:
                    ext_by_seed[s] = r
                    break

        xs, ys, y_lo, y_hi, seeds_list = [], [], [], [], []
        for s in (42, 43, 44, 45, 46):
            x_val = int_by_seed[s]
            if arch_key == "resnet18" and s == 45:
                x_val -= 0.0008
            elif arch_key == "resnet18" and s == 46:
                x_val += 0.0008
            elif arch_key == "densenet121" and s == 44:
                x_val -= 0.0008
            elif arch_key == "densenet121" and s == 45:
                x_val += 0.0008
            r_ext = ext_by_seed[s]
            auc_v = float(r_ext["auc"])
            ci_l = float(r_ext["bootstrap_ci_95"]["auc"]["low"])
            ci_h = float(r_ext["bootstrap_ci_95"]["auc"]["high"])
            xs.append(x_val)
            ys.append(auc_v)
            y_lo.append(auc_v - ci_l)
            y_hi.append(ci_h - auc_v)
            seeds_list.append(s)

        ax.errorbar(
            xs,
            ys,
            yerr=[y_lo, y_hi],
            fmt=marker_style,
            color=col,
            ecolor=col,
            elinewidth=1.0,
            capsize=2.8,
            markersize=6.2,
            alpha=0.90,
            label=label_str,
            zorder=3,
        )

        # Ring around primary seed-42 ResNet-18 and SmallCNN
        if arch_key in ("resnet18", "small_cnn"):
            idx42 = seeds_list.index(42)
            ax.scatter(
                [xs[idx42]],
                [ys[idx42]],
                s=135,
                facecolors="none",
                edgecolors="#111827",
                linewidths=1.6,
                zorder=5,
                label="Primary Seed-42 Checkpoint" if arch_key == "resnet18" else None,
            )
            ax.annotate(
                f"Seed 42\n({ys[idx42]:.4f})",
                xy=(xs[idx42], ys[idx42]),
                xytext=(-36 if arch_key == "resnet18" else 8, 8 if arch_key == "resnet18" else -14),
                textcoords="offset points",
                fontsize=6.8,
                fontweight="bold",
                color="#111827",
            )

    ax.set_xlabel("Internal Discovery Test Accuracy (n = 606)", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("External BraTS-250 ROC-AUC (95% Patient-Cluster CI)", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0.79, 1.005)
    ax.set_ylim(0.32, 0.80)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fontsize=7.2)

    plt.tight_layout()
    out_png = OUT_DIR / "fig5_seeds_internal_vs_external.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
