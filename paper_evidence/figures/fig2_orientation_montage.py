"""
Figure 2: Orientation Montage Across Discovery, Cheng, BraTS First Version (Unrotated), and BraTS Corrected
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe annotations
- Sized for 180 mm (7.09 in) column width
- 4 random images per column with fixed seed (seed=42)
- Columns: Discovery | Cheng | BraTS first version | BraTS corrected
- Includes explicit Anterior ('A') directional marker arrow on each column header
"""
from __future__ import annotations

from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BrainTumorAI.external_validation import _decode_cheng_mat

OUT_DIR = Path(__file__).resolve().parent


def load_cheng_sample_slices(zip_path: Path, n_samples: int = 4, seed: int = 42) -> list[tuple[np.ndarray, str]]:
    """Sample 4 axial Cheng slices with fixed seed."""
    rng = np.random.default_rng(seed)
    # Use axial slices from brainTumorDataPublic_1533-2298.zip & 2299-3064.zip
    chosen_entries = []
    with zipfile.ZipFile(zip_path, "r") as outer_z:
        with outer_z.open("brainTumorDataPublic_2299-3064.zip") as part_stream:
            with zipfile.ZipFile(part_stream, "r") as inner_z:
                mat_names = sorted(
                    [n for n in inner_z.namelist() if n.lower().endswith(".mat")],
                    key=lambda x: int(Path(x).stem),
                )
                picks = rng.choice(len(mat_names), size=n_samples, replace=False)
                for idx in np.sort(picks):
                    mname = mat_names[int(idx)]
                    raw = inner_z.read(mname)
                    arr, pid, _ = _decode_cheng_mat(raw, mname)
                    mn, mx = float(arr.min()), float(arr.max())
                    u8 = np.clip(np.rint((arr.astype(np.float64) - mn) * (255.0 / max(1e-6, mx - mn))), 0, 255).astype(np.uint8)
                    chosen_entries.append((u8, f"Slice {Path(mname).stem} ({pid})"))
    return chosen_entries


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 8.0,
    })

    # 1. Discovery (fixed seed 42, 4 random test images)
    df_disc = pd.read_csv(ROOT / "BrainTumorAI" / "results" / "split_manifest.csv")
    disc_samples = df_disc[df_disc["split"] == "test"].sample(4, random_state=42).reset_index(drop=True)

    # 2. Cheng (fixed seed 42, 4 random slices)
    cheng_samples = load_cheng_sample_slices(ROOT / "1512427.zip", n_samples=4, seed=42)

    # 3 & 4. BraTS first version (unrotated) and BraTS corrected (oriented), same 4 random mid-axial slices (seed 42)
    brats_unrot_root = Path("D:/datasets/prepared/brats")
    brats_corr_root = Path("D:/datasets/prepared/brats_oriented")
    df_brats = pd.read_csv(brats_unrot_root / "manifest.csv")
    brats_samples = df_brats[df_brats["slice_id"].between(65, 95) & (df_brats["label_idx"] == 1)].sample(4, random_state=42).reset_index(drop=True)

    fig, axes = plt.subplots(4, 4, figsize=(7.09, 7.2), dpi=300, facecolor="white")
    plt.subplots_adjust(wspace=0.08, hspace=0.22, top=0.88, bottom=0.03, left=0.03, right=0.97)

    col_headers = [
        ("Discovery\n(Kaggle)", "A ↑ (Up)", "#0072B2"),
        ("Cheng (2015)\n(Figshare T1-CE)", "A ↑ / Mixed", "#009E73"),
        ("BraTS First Version\n(Unrotated RAS+)", "← A (Left, -90°)", "#D55E00"),
        ("BraTS Corrected\n(Oriented rot90)", "A ↑ (Up)", "#0072B2"),
    ]

    for col_idx, (title_txt, ant_txt, col_color) in enumerate(col_headers):
        ax_top = axes[0, col_idx]
        ax_top.annotate(
            f"{title_txt}\nAnterior: {ant_txt}",
            xy=(0.5, 1.18),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=col_color,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor=col_color, lw=1.2),
        )

    for row_idx in range(4):
        # Col 0: Discovery
        d_row = disc_samples.iloc[row_idx]
        d_im = np.asarray(Image.open(ROOT / str(d_row["source_file"])).convert("L"))
        ax0 = axes[row_idx, 0]
        ax0.imshow(d_im, cmap="gray")
        ax0.set_title(Path(str(d_row["source_file"])).name, fontsize=6.8, pad=2)
        ax0.axis("off")
        # Draw small Anterior arrow at top center
        ax0.text(0.5, 0.92, "A ↑", transform=ax0.transAxes, color="#56B4E9", fontsize=7.5, fontweight="bold", ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, lw=0))

        # Col 1: Cheng
        c_im, c_lbl = cheng_samples[row_idx]
        ax1 = axes[row_idx, 1]
        ax1.imshow(c_im, cmap="gray")
        ax1.set_title(c_lbl, fontsize=6.8, pad=2)
        ax1.axis("off")
        ax1.text(0.5, 0.92, "A", transform=ax1.transAxes, color="#009E73", fontsize=7.5, fontweight="bold", ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, lw=0))

        # Col 2: BraTS first version (unrotated)
        b_row = brats_samples.iloc[row_idx]
        b_unrot_im = np.asarray(Image.open(brats_unrot_root / str(b_row["source_file"])).convert("L"))
        ax2 = axes[row_idx, 2]
        ax2.imshow(b_unrot_im, cmap="gray")
        ax2.set_title(f"{b_row['patient_id']} (z={b_row['slice_id']})", fontsize=6.5, pad=2)
        ax2.axis("off")
        ax2.text(0.14, 0.50, "← A", transform=ax2.transAxes, color="#E69F00", fontsize=7.5, fontweight="bold", ha="left", va="center",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, lw=0))

        # Col 3: BraTS corrected (oriented)
        b_corr_im = np.asarray(Image.open(brats_corr_root / str(b_row["source_file"])).convert("L"))
        ax3 = axes[row_idx, 3]
        ax3.imshow(b_corr_im, cmap="gray")
        ax3.set_title(f"{b_row['patient_id']} (z={b_row['slice_id']})", fontsize=6.5, pad=2)
        ax3.axis("off")
        ax3.text(0.5, 0.92, "A ↑", transform=ax3.transAxes, color="#56B4E9", fontsize=7.5, fontweight="bold", ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6, lw=0))

    out_png = OUT_DIR / "fig2_orientation_montage.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()

