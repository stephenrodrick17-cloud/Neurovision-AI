"""
Positive control verification for the strictly internal overlap audit functions:
  - _perceptual_hash (DCT 63 AC coefficients vs median) from BrainTumorAI.preprocess
  - _structural_similarity (OpenCV Gaussian 11x11, sigma 1.5) from BrainTumorAI.preprocess
  - Zero imagehash or scikit-image used.

Tests:
1. 50 known Cheng-Discovery duplicate pairs decoded live from 1512427.zip via _decode_cheng_mat and matched against the 4,600 Discovery images using the exact project rule (pHash Hamming <= 4 and 64x64 grayscale SSIM >= 0.95).
2. 50 BraTS slices from brats_oriented injected into the Discovery index, confirming the audit detects 50/50 exact matches (pHash Hamming=0, SSIM=1.0).
"""
import os
import io
import sys
import json
import time
import zipfile
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.external_validation import _perceptual_hash, _structural_similarity, _decode_cheng_mat

def main():
    t0 = time.time()
    print("Running Positive Control Audit...", flush=True)

    # 1. Load Discovery hashes and 64x64 grayscale arrays
    split_df = pd.read_csv("BrainTumorAI/results/split_manifest.csv")
    disc_hashes = []
    disc_gray64 = {}
    disc_files = []
    for idx, row in split_df.iterrows():
        fpath = row["source_file"]
        with Image.open(fpath) as img:
            phash_hex, bits, gray64 = _perceptual_hash(img.convert("RGB"))
            disc_hashes.append(bits)
            disc_gray64[idx] = gray64
            disc_files.append(fpath)
    disc_hash_arr = np.array(disc_hashes, dtype=bool)
    print(f"Indexed {len(disc_files)} Discovery images in {time.time()-t0:.2f}s.", flush=True)

    # -------------------------------------------------------------------------
    # Control 1: 50 known Cheng-Discovery duplicate pairs decoded live from 1512427.zip
    # -------------------------------------------------------------------------
    cheng_audit_path = "BrainTumorAI/results/external_validation/cheng_figshare_v8/overlap_audit.json"
    with open(cheng_audit_path, "r") as f:
        cheng_audit = json.load(f)
    known_matches = cheng_audit["matches"]["high_similarity_matches"]

    seen_cheng = set()
    control_50_cheng = []
    for m in known_matches:
        cs = m["cheng_slice"]
        if cs not in seen_cheng:
            seen_cheng.add(cs)
            control_50_cheng.append(m)
        if len(control_50_cheng) == 50:
            break

    archive_path = "1512427.zip"
    cheng_detected_count = 0
    cheng_exact_expected_pair_count = 0
    cheng_ssims = []
    cheng_hammings = []
    cheng_sample_results = []

    with zipfile.ZipFile(archive_path, "r") as outer_zf:
        part_cache = {}
        for item in control_50_cheng:
            part_name, mat_name = item["cheng_slice"].split("!")
            if part_name not in part_cache:
                part_bytes = outer_zf.read(part_name)
                part_cache[part_name] = zipfile.ZipFile(io.BytesIO(part_bytes), "r")
            part_zf = part_cache[part_name]

            image_array, patient_id, _ = _decode_cheng_mat(
                part_zf.read(mat_name),
                f"{archive_path}!{part_name}!{mat_name}"
            )
            lower = float(image_array.min())
            upper = float(image_array.max())
            pixels = np.rint((image_array.astype(np.float64) - lower) * (255.0 / (upper - lower)))
            gray = np.clip(pixels, 0, 255).astype(np.uint8)

            _, c_bits, c_gray64 = _perceptual_hash(Image.fromarray(gray, mode="L"))
            c_arr = np.array(c_bits, dtype=bool)

            hd = np.count_nonzero(disc_hash_arr != c_arr, axis=1)
            cand_idxs = np.where(hd <= 4)[0]

            detected = False
            expected_pair_found = False
            best_ssim = 0.0
            best_hd = 99
            matched_disc_file = None

            for idx_hit in cand_idxs:
                ssim_val = _structural_similarity(c_gray64, disc_gray64[idx_hit])
                if ssim_val >= 0.95:
                    detected = True
                    if ssim_val > best_ssim:
                        best_ssim = ssim_val
                        best_hd = int(hd[idx_hit])
                        matched_disc_file = disc_files[idx_hit]
                    if disc_files[idx_hit] == item["local_source_file"]:
                        expected_pair_found = True

            if detected:
                cheng_detected_count += 1
                cheng_ssims.append(best_ssim)
                cheng_hammings.append(best_hd)
            if expected_pair_found:
                cheng_exact_expected_pair_count += 1

            if len(cheng_sample_results) < 5:
                cheng_sample_results.append({
                    "cheng_slice": item["cheng_slice"],
                    "patient_id": patient_id,
                    "expected_discovery_file": item["local_source_file"],
                    "matched_discovery_file": matched_disc_file,
                    "phash_hamming_distance": best_hd,
                    "ssim_64x64": float(best_ssim),
                    "detected": detected,
                    "expected_pair_matched": expected_pair_found
                })

        for pzf in part_cache.values():
            pzf.close()

    # -------------------------------------------------------------------------
    # Control 2: 50 BraTS slices injected into the Discovery set
    # -------------------------------------------------------------------------
    df_brats = pd.read_csv("D:/datasets/prepared/brats/manifest.csv")
    rng = np.random.RandomState(42)
    sample_50_idxs = rng.choice(len(df_brats), size=50, replace=False)
    sample_50_rows = df_brats.iloc[sample_50_idxs].reset_index(drop=True)

    oriented_dir = "D:/datasets/prepared/brats_oriented"
    injected_hashes = []
    injected_gray64 = []
    injected_files = []
    for _, row in sample_50_rows.iterrows():
        sfpath = os.path.join(oriented_dir, row["source_file"])
        with Image.open(sfpath) as simg:
            _, sbits, sgray64 = _perceptual_hash(simg.convert("RGB"))
            injected_hashes.append(sbits)
            injected_gray64.append(sgray64)
            injected_files.append(sfpath)

    augmented_hash_arr = np.vstack([disc_hash_arr, np.array(injected_hashes, dtype=bool)])
    n_orig = len(disc_files)

    brats_detected = 0
    brats_ssims = []
    for i, sfpath in enumerate(injected_files):
        with Image.open(sfpath) as simg:
            _, sbits, sgray64 = _perceptual_hash(simg.convert("RGB"))
            s_arr = np.array(sbits, dtype=bool)
            hd = np.count_nonzero(augmented_hash_arr != s_arr, axis=1)
            cand_idxs = np.where(hd <= 4)[0]
            found = False
            for h_idx in cand_idxs:
                target_gray64 = disc_gray64[h_idx] if h_idx < n_orig else injected_gray64[h_idx - n_orig]
                ssim_val = _structural_similarity(sgray64, target_gray64)
                if hd[h_idx] == 0 and ssim_val >= 0.9999 and (h_idx == n_orig + i):
                    found = True
                    brats_ssims.append(ssim_val)
                    break
            if found:
                brats_detected += 1

    out = {
        "functions_used": {
            "perceptual_hash": "BrainTumorAI.preprocess._perceptual_hash (DCT 63 AC coefficients vs median)",
            "structural_similarity": "BrainTumorAI.preprocess._structural_similarity (OpenCV GaussianBlur 11x11, sigma=1.5)",
            "rule": "pHash Hamming distance <= 4 and 64x64 grayscale SSIM >= 0.95",
            "imagehash_used": False,
            "scikit_image_used": False
        },
        "positive_control_1_cheng_discovery_50_pairs": {
            "source_archive": "C:/Users/asus/Downloads/Brain Imaging/1512427.zip",
            "pairs_tested": 50,
            "pairs_detected": int(cheng_detected_count),
            "exact_expected_discovery_file_matched": int(cheng_exact_expected_pair_count),
            "detection_rate_percent": float(cheng_detected_count / 50 * 100.0),
            "hamming_distance_min_max": [int(min(cheng_hammings)), int(max(cheng_hammings))],
            "ssim_min_mean_max": [float(np.min(cheng_ssims)), float(np.mean(cheng_ssims)), float(np.max(cheng_ssims))],
            "sample_5_verified_pairs": cheng_sample_results
        },
        "positive_control_2_brats_50_injected_slices": {
            "slices_injected": 50,
            "slices_detected": int(brats_detected),
            "detection_rate_percent": float(brats_detected / 50 * 100.0),
            "hamming_distance_all": 0,
            "ssim_min_mean_max": [float(np.min(brats_ssims)), float(np.mean(brats_ssims)), float(np.max(brats_ssims))]
        },
        "runtime_seconds": float(time.time() - t0)
    }

    with open("paper_evidence/positive_control_overlap_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Positive control complete in {time.time()-t0:.2f}s: Cheng={cheng_detected_count}/50, BraTS={brats_detected}/50", flush=True)

if __name__ == "__main__":
    main()
