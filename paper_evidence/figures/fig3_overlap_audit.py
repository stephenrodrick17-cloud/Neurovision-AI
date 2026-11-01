"""
Figure 3: Cross-Dataset Overlap Examples and Partition Counts (Section 3.7.1 Overlap)
- 300 dpi PNG, white background, sans-serif font, colour-blind-safe Okabe-Ito palette
- Sized for 180 mm (7.09 in) column width
- Panel (a): For Cheng patient MR053110B, show slices 2999, 3000, 3001 next to their matched discovery images,
             label each discovery image with its partition (validation, test, validation).
- Panel (b): Show 4 matched discovery-Cheng pairs with pHash Hamming distance and SSIM printed.
- Panel (c): Bar chart of matched discovery images by partition (543 / 162 / 141) and distinct Cheng patients
             by partition (75 / 61 / 59), read directly from saved overlap_audit.json and verified_leakage_claims.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BrainTumorAI.external_validation import _decode_cheng_mat
from BrainTumorAI.preprocess import _perceptual_hash, _structural_similarity

OUT_DIR = Path(__file__).resolve().parent


def load_specific_cheng_slices(zip_path: Path, target_keys: set[str]) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    part_names = [
        "brainTumorDataPublic_1-766.zip",
        "brainTumorDataPublic_767-1532.zip",
        "brainTumorDataPublic_1533-2298.zip",
        "brainTumorDataPublic_2299-3064.zip",
    ]
    with zipfile.ZipFile(zip_path, "r") as outer_z:
        for pname in part_names:
            needed_in_part = {k.split("!")[1] for k in target_keys if k.startswith(pname + "!")}
            if not needed_in_part:
                continue
            with outer_z.open(pname) as part_stream:
                with zipfile.ZipFile(part_stream, "r") as inner_z:
                    for mname in needed_in_part:
                        raw = inner_z.read(mname)
                        arr, _, _ = _decode_cheng_mat(raw, f"{pname}!{mname}")
                        mn, mx = float(arr.min()), float(arr.max())
                        u8 = np.clip(np.rint((arr.astype(np.float64) - mn) * (255.0 / max(1e-6, mx - mn))), 0, 255).astype(np.uint8)
                        found[f"{pname}!{mname}"] = u8
    return found


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 7.5,
    })

    audit_data = json.loads(
        (ROOT / "BrainTumorAI" / "results" / "external_validation" / "cheng_figshare_v8" / "overlap_audit.json").read_text(encoding="utf-8")
    )
    claims_data = json.loads((ROOT / "paper_evidence" / "verified_leakage_claims.json").read_text(encoding="utf-8"))
    matches = audit_data["matches"]["high_similarity_matches"]

    # Panel (a): Patient MR053110B, slices 2999.mat, 3000.mat, 3001.mat
    target_a_slices = [
        "brainTumorDataPublic_2299-3064.zip!2999.mat",
        "brainTumorDataPublic_2299-3064.zip!3000.mat",
        "brainTumorDataPublic_2299-3064.zip!3001.mat",
    ]
    panel_a_edges = []
    for t_sl in target_a_slices:
        edge = next(m for m in matches if m["cheng_slice"] == t_sl and m["patient_id"] == "MR053110B")
        panel_a_edges.append(edge)

    # Panel (b): 4 representative matched pairs across splits with varied pHash distance (0, 0, 2, 4) and high SSIM
    edge_d0_train = next(m for m in matches if int(m["phash_hamming_distance"]) == 0 and m["local_split"] == "train")
    edge_d0_test = next(m for m in matches if int(m["phash_hamming_distance"]) == 0 and m["local_split"] == "test")
    edge_d2 = next((m for m in matches if int(m["phash_hamming_distance"]) == 2), matches[10])
    edge_d4 = next((m for m in matches if int(m["phash_hamming_distance"]) == 4), matches[20])
    panel_b_edges = [edge_d0_train, edge_d0_test, edge_d2, edge_d4]

    needed_cheng = {e["cheng_slice"] for e in panel_a_edges + panel_b_edges}
    cheng_imgs = load_specific_cheng_slices(ROOT / "1512427.zip", needed_cheng)

    # Panel (c): Counts from saved audit & leakage verification files
    split_order = ["train", "validation", "test"]
    matched_imgs_by_split = {
        sp: sum(1 for m in matches if m["local_split"] == sp)
        for sp in split_order
    }
    patients_by_split = {
        "train": int(claims_data["patient_counts"]["patients_in_train"]),
        "validation": int(claims_data["patient_counts"]["patients_in_validation"]),
        "test": int(claims_data["patient_counts"]["patients_in_test"]),
    }
    assert [matched_imgs_by_split[s] for s in split_order] == [543, 162, 141]
    assert [patients_by_split[s] for s in split_order] == [75, 61, 59]

    fig = plt.figure(figsize=(7.09, 6.4), dpi=300, facecolor="white")
    gs_main = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.15, 0.90], width_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

    # ---- Panel (a): Top-left (2 rows x 3 cols: top row Cheng MR053110B 2999/3000/3001, bottom row Discovery match + split) ----
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[0, 0], hspace=0.32, wspace=0.08)
    for col_i, edge in enumerate(panel_a_edges):
        sl_num = Path(edge["cheng_slice"].split("!")[1]).stem
        ax_top = fig.add_subplot(gs_a[0, col_i])
        ax_top.imshow(cheng_imgs[edge["cheng_slice"]], cmap="gray")
        ax_top.set_title(f"Cheng #{sl_num}.mat\n(MR053110B)", fontsize=6.4, pad=2)
        ax_top.axis("off")
        if col_i == 0:
            ax_top.text(-0.05, 1.26, "(a) Patient MR053110B Across Splits",
                        transform=ax_top.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom")

        ax_bot = fig.add_subplot(gs_a[1, col_i])
        d_im = np.asarray(Image.open(ROOT / edge["local_source_file"]).convert("L"))
        ax_bot.imshow(d_im, cmap="gray")
        sp_lbl = edge["local_split"]
        sp_col = "#D55E00" if sp_lbl == "test" else "#0072B2"
        d_name = Path(edge["local_source_file"]).name
        ax_bot.set_title(f"{d_name}\n[{sp_lbl.upper()}]", fontsize=6.5, fontweight="bold", color=sp_col, pad=2)
        ax_bot.axis("off")

    # ---- Panel (b): Bottom-left & Top-right? Let's put Panel (b) on Bottom across 4 pairs and Panel (c) on Top-Right! ----
    # Wait: let's put Panel (c) in gs_main[0, 1] (top right) and Panel (b) spanning the entire bottom row gs_main[1, :] (2x4 grid)!
    ax_c = fig.add_subplot(gs_main[0, 1])
    x = np.arange(3)
    width = 0.36
    img_vals = [matched_imgs_by_split[s] for s in split_order]
    pat_vals = [patients_by_split[s] for s in split_order]

    rects1 = ax_c.bar(x - width / 2, img_vals, width, label="Matched Discovery Images (n=846)", color="#0072B2", edgecolor="black", linewidth=0.6)
    rects2 = ax_c.bar(x + width / 2, pat_vals, width, label="Distinct Cheng Patients (n=81 unique)", color="#E69F00", edgecolor="black", linewidth=0.6)

    for r in rects1:
        h = r.get_height()
        ax_c.text(r.get_x() + r.get_width() / 2.0, h + 10, f"{int(h)}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#0072B2")
    for r in rects2:
        h = r.get_height()
        ax_c.text(r.get_x() + r.get_width() / 2.0, h + 10, f"{int(h)}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#B45309")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["Train\n(3,191 imgs)", "Validation\n(803 imgs)", "Test\n(606 imgs)"], fontsize=7.5)
    ax_c.set_ylabel("Count", fontsize=8.0, fontweight="bold")
    ax_c.set_ylim(0, 630)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(axis="y", linestyle="--", alpha=0.4)
    ax_c.legend(loc="upper right", frameon=True, fontsize=6.6)
    ax_c.set_title("(c) Overlap Counts by Discovery Partition", fontsize=8.2, fontweight="bold", loc="left", pad=8)

    # ---- Panel (b): Bottom row spanning both columns (4 matched pairs: Cheng vs Discovery with pHash & SSIM) ----
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1, :], wspace=0.18)
    for pair_i, edge in enumerate(panel_b_edges):
        gs_pair = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_b[0, pair_i], wspace=0.05)
        ax_l = fig.add_subplot(gs_pair[0, 0])
        ax_r = fig.add_subplot(gs_pair[0, 1])

        d_im = np.asarray(Image.open(ROOT / edge["local_source_file"]).convert("L"))
        c_im = cheng_imgs[edge["cheng_slice"]]
        d_name = Path(edge["local_source_file"]).name
        c_num = Path(edge["cheng_slice"].split("!")[1]).stem

        ax_l.imshow(d_im, cmap="gray")
        ax_l.set_title(f"{d_name}\n[{edge['local_split']}]", fontsize=5.8, pad=2)
        ax_l.axis("off")

        ax_r.imshow(c_im, cmap="gray")
        ax_r.set_title(f"Cheng #{c_num}.mat\n({edge['patient_id']})", fontsize=5.8, pad=2)
        ax_r.axis("off")

        ph_d = int(edge["phash_hamming_distance"])
        _, _, d_g64 = _perceptual_hash(Image.open(ROOT / edge["local_source_file"]).convert("RGB"))
        _, _, c_g64 = _perceptual_hash(Image.fromarray(c_im, mode="L"))
        ssim_v = float(_structural_similarity(c_g64, d_g64))
        ax_l.text(
            1.02,
            -0.10,
            f"pHash={ph_d} | SSIM={ssim_v:.3f}",
            transform=ax_l.transAxes,
            ha="center",
            va="top",
            fontsize=6.2,
            fontweight="bold",
            color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#F3F4F6", edgecolor="#9CA3AF", lw=0.6),
        )
        if pair_i == 0:
            ax_l.text(0.0, 1.34, "(b) Matched Discovery-Cheng Pairs with Perceptual Hash Distance & SSIM",
                      transform=ax_l.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="bottom")

    out_png = OUT_DIR / "fig3_overlap_audit.png"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
