"""
BraTS-vs-Discovery Overlap Audit using STRICTLY repository-internal functions:
- Perceptual hash: _perceptual_hash (DCT 63 AC coefficients vs median) from BrainTumorAI.external_validation
- Structural similarity: _structural_similarity (OpenCV Gaussian 11x11, sigma 1.5) from BrainTumorAI.external_validation
NO imagehash, NO scikit-image!
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import time
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from BrainTumorAI.external_validation import _perceptual_hash, _structural_similarity

def to_grayscale_array(img_pil):
    arr = np.array(img_pil.convert('L'))
    return arr

def main():
    print("====================================================================")
    print("BraTS vs Discovery Native Overlap Audit")
    print("Strictly internal functions: _perceptual_hash & _structural_similarity")
    print("Zero imagehash, zero scikit-image.")
    print("====================================================================")
    
    # 1. Load Discovery dataset images
    split_df = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
    print(f"Loaded Discovery dataset: {len(split_df)} images across splits: {split_df['split'].value_counts().to_dict()}")
    
    t0 = time.time()
    print("Computing hashes for Discovery images...")
    discovery_hashes = []
    discovery_records = []
    for idx, row in split_df.iterrows():
        fpath = row['source_file']
        with Image.open(fpath) as img:
            hex_str, bit_list, _ = _perceptual_hash(img)
            discovery_hashes.append(bit_list)
            discovery_records.append({
                "idx": idx,
                "file": fpath,
                "split": row['split'],
                "label": row['label'],
                "hex": hex_str
            })
    print(f"Computed {len(discovery_hashes)} Discovery hashes in {time.time() - t0:.2f} s")
    
    disc_hash_arr = np.array(discovery_hashes, dtype=bool) # shape (4600, 63)
    print(f"Discovery hash array shape: {disc_hash_arr.shape}")
    
    manifest_path = 'D:/datasets/prepared/brats/manifest.csv'
    df_manifest = pd.read_csv(manifest_path)
    patients = sorted(df_manifest['patient_id'].unique())
    print(f"Total BraTS patients: {len(patients)}, Total slices: {len(df_manifest)}")
    
    oriented_dir = 'D:/datasets/prepared/brats_oriented'
    
    exact_matches = []
    high_sim_matches = []
    
    t_scan_start = time.time()
    total_slices_audited = 0
    
    # Audit all 1,251 patients
    for p_idx, pid in enumerate(patients):
        p_df = df_manifest[df_manifest['patient_id'] == pid]
        
        for _, s_row in p_df.iterrows():
            s_fpath = os.path.join(oriented_dir, s_row['source_file'])
            if not os.path.exists(s_fpath):
                continue
                
            total_slices_audited += 1
            with Image.open(s_fpath) as s_img:
                _, s_bits, _ = _perceptual_hash(s_img)
                s_hash_arr = np.array(s_bits, dtype=bool)
                
                # Bitwise hamming distance
                hamming_dists = np.count_nonzero(disc_hash_arr != s_hash_arr, axis=1)
                
                # Exact match: hamming distance 0
                exact_hits = np.where(hamming_dists == 0)[0]
                for hit in exact_hits:
                    with Image.open(discovery_records[hit]['file']) as d_img:
                        d_arr = to_grayscale_array(d_img)
                        s_arr = to_grayscale_array(s_img)
                        if d_arr.shape != s_arr.shape:
                            d_arr = cv2.resize(d_arr, (s_arr.shape[1], s_arr.shape[0]))
                        ssim_val = _structural_similarity(d_arr, s_arr)
                    match_record = {
                        "patient_id": pid,
                        "slice_file": s_row['source_file'],
                        "discovery_file": discovery_records[hit]['file'],
                        "discovery_split": discovery_records[hit]['split'],
                        "hamming_dist": int(hamming_dists[hit]),
                        "ssim": float(ssim_val)
                    }
                    exact_matches.append(match_record)
                    print(f"EXACT HIT: {match_record}")
                    
                # Near duplicates (hamming <= 4)
                cand_hits = np.where((hamming_dists > 0) & (hamming_dists <= 4))[0]
                for hit in cand_hits:
                    with Image.open(discovery_records[hit]['file']) as d_img:
                        d_arr = to_grayscale_array(d_img)
                        s_arr = to_grayscale_array(s_img)
                        if d_arr.shape != s_arr.shape:
                            d_arr = cv2.resize(d_arr, (s_arr.shape[1], s_arr.shape[0]))
                        ssim_val = _structural_similarity(d_arr, s_arr)
                    if ssim_val >= 0.85:
                        match_record = {
                            "patient_id": pid,
                            "slice_file": s_row['source_file'],
                            "discovery_file": discovery_records[hit]['file'],
                            "discovery_split": discovery_records[hit]['split'],
                            "hamming_dist": int(hamming_dists[hit]),
                            "ssim": float(ssim_val)
                        }
                        high_sim_matches.append(match_record)
                        print(f"HIGH SIM HIT: {match_record}")
                        
        if (p_idx + 1) % 150 == 0 or (p_idx + 1) == len(patients):
            elapsed = time.time() - t_scan_start
            print(f"Audited {p_idx + 1}/{len(patients)} patients ({total_slices_audited} slices) in {elapsed:.1f}s | Hits: exact={len(exact_matches)}, high_sim={len(high_sim_matches)}", flush=True)
            
    # Multi-orientation on 30 patients
    print("\nRunning multi-orientation check on 30 patients across 4 rotations (0, 90, 180, 270)...")
    np.random.seed(42)
    sample_30 = np.random.choice(patients, size=30, replace=False)
    multi_rot_hits = []
    
    raw_dir = 'D:/datasets/prepared/brats'
    for pid in sample_30:
        p_df = df_manifest[df_manifest['patient_id'] == pid]
        for _, s_row in p_df.iterrows():
            s_fpath = os.path.join(raw_dir, s_row['source_file'])
            if not os.path.exists(s_fpath):
                continue
            with Image.open(s_fpath) as raw_img:
                raw_np = np.array(raw_img)
                for rot_k in [0, 1, 2, 3]:
                    rot_pil = Image.fromarray(np.rot90(raw_np, rot_k))
                    _, r_bits, _ = _perceptual_hash(rot_pil)
                    r_arr = np.array(r_bits, dtype=bool)
                    hd = np.count_nonzero(disc_hash_arr != r_arr, axis=1)
                    hits = np.where(hd <= 2)[0]
                    for hit in hits:
                        with Image.open(discovery_records[hit]['file']) as d_img:
                            d_arr = to_grayscale_array(d_img)
                            r_arr_gray = to_grayscale_array(rot_pil)
                            if d_arr.shape != r_arr_gray.shape:
                                d_arr = cv2.resize(d_arr, (r_arr_gray.shape[1], r_arr_gray.shape[0]))
                            ssim_val = _structural_similarity(d_arr, r_arr_gray)
                        if ssim_val >= 0.85:
                            multi_rot_hits.append({
                                "patient_id": pid, "rot_k": rot_k, "discovery_file": discovery_records[hit]['file'],
                                "ssim": float(ssim_val)
                            })
                            
    out = {
        "metadata": {
            "method": "Strictly internal repository functions: _perceptual_hash (DCT 63 AC coefficients) and _structural_similarity (OpenCV Gaussian 11x11, sigma 1.5)",
            "imagehash_used": False,
            "skimage_used": False,
            "discovery_images_indexed": len(discovery_records),
            "brats_patients_audited": len(patients),
            "brats_slices_audited": total_slices_audited,
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "all_patient_oriented_audit": {
            "exact_matches_count": len(exact_matches),
            "exact_matches": exact_matches,
            "high_similarity_matches_count": len(high_sim_matches),
            "high_similarity_matches": high_sim_matches
        },
        "multi_orientation_30_patient_audit": {
            "patients_audited": list(sample_30),
            "angles_tested": [0, 90, 180, 270],
            "hits_count": len(multi_rot_hits),
            "hits": multi_rot_hits
        }
    }
    
    with open('paper_evidence/brats_native_overlap_audit.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved native overlap audit to paper_evidence/brats_native_overlap_audit.json")

if __name__ == '__main__':
    main()
