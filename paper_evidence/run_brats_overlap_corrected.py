"""
Item 5: Full BraTS-vs-Discovery overlap audit on brats_oriented (corrected PNGs).
All 1,251 patients. pHash stage first, SSIM only on candidates.
Also: 30-patient sample on old PNGs under 4 orientations (0, 90CW, 90CCW, 180).

Output: paper_evidence/phase3_brats_overlap_audit_corrected.json
"""
import os, sys, json, time, hashlib
import numpy as np
import pandas as pd
from PIL import Image
from itertools import product as iterproduct

# pHash and SSIM dependencies
try:
    import imagehash
    print(f"imagehash version: {imagehash.__version__}")
except ImportError:
    print("ERROR: imagehash not installed")
    sys.exit(1)

try:
    from skimage.metrics import structural_similarity as ssim
    import skimage
    print(f"scikit-image version: {skimage.__version__}")
except ImportError:
    print("ERROR: scikit-image not installed")
    sys.exit(1)

PHASH_THRESHOLD = 4
SSIM_THRESHOLD = 0.95
SSIM_SIZE = 64  # resize for SSIM comparison

def compute_phash(img_path):
    """Compute perceptual hash of an image."""
    img = Image.open(img_path).convert('L')
    return imagehash.phash(img)

def compute_ssim(img1_path, img2_path, size=SSIM_SIZE):
    """Compute SSIM between two images after resizing to size x size grayscale."""
    img1 = Image.open(img1_path).convert('L').resize((size, size), Image.LANCZOS)
    img2 = Image.open(img2_path).convert('L').resize((size, size), Image.LANCZOS)
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    return ssim(arr1, arr2, data_range=255.0)

# ============================================================
# Load discovery test images
# ============================================================
print("\n" + "=" * 60)
print("Loading Discovery dataset images for overlap audit")
print("=" * 60)

split_manifest = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
discovery_all = split_manifest.copy()
print(f"Discovery images: {len(discovery_all)}")
print(f"Splits: {discovery_all['split'].value_counts().to_dict()}")

# Build pHash index for all discovery images
print("\nComputing pHash for all discovery images...")
t0 = time.time()
discovery_hashes = {}
discovery_root = 'BrainTumorAI'
errors = []

for i, row in discovery_all.iterrows():
    fpath = os.path.join(discovery_root, row['source_file'])
    if not os.path.exists(fpath):
        # Try alternative path
        fpath = row['source_file']
    if not os.path.exists(fpath):
        errors.append(row['source_file'])
        continue
    try:
        h = compute_phash(fpath)
        discovery_hashes[row['source_file']] = {
            'hash': h,
            'path': fpath,
            'split': row['split'],
            'label': row['label']
        }
    except Exception as e:
        errors.append(f"{row['source_file']}: {e}")
    
    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(discovery_all)}] hashed...")

t1 = time.time()
print(f"Hashed {len(discovery_hashes)} discovery images in {t1-t0:.1f}s ({len(errors)} errors)")

# ============================================================
# Full BraTS overlap audit on brats_oriented (all 1,251 patients)
# ============================================================
print("\n" + "=" * 60)
print("Full BraTS overlap audit: brats_oriented vs Discovery (all 1,251 patients)")
print("=" * 60)

brats_manifest = pd.read_csv('D:/datasets/prepared/brats/manifest.csv')
brats_oriented_root = 'D:/datasets/prepared/brats_oriented'
print(f"BraTS manifest: {len(brats_manifest)} slices, {brats_manifest['patient_id'].nunique()} patients")

# Sample strategy: computing pHash for 169K images would be slow.
# Use a representative sample: 50 slices per patient (random), or all if fewer.
# Actually, user wants ALL 1,251 patients. Let's use a stratified approach:
# - Compute pHash for every BraTS slice (this is the main bottleneck)
# - Then compare against discovery hashes
# 
# At ~100 hashes/s, 169K images = ~28 minutes. Let's do it.

print("\nComputing pHash for all BraTS oriented images...")
t0_brats = time.time()
brats_hashes = {}
n_brats = len(brats_manifest)

for i, row in brats_manifest.iterrows():
    fpath = os.path.join(brats_oriented_root, row['source_file'])
    try:
        h = compute_phash(fpath)
        brats_hashes[i] = {
            'hash': h,
            'patient_id': row['patient_id'],
            'source_file': row['source_file'],
            'label': row['label_idx']
        }
    except Exception as e:
        pass
    
    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0_brats
        speed = (i + 1) / elapsed
        eta = (n_brats - i - 1) / speed
        print(f"  [{i+1}/{n_brats}] ({(i+1)/n_brats*100:.1f}%) | {speed:.1f} imgs/s | ETA: {eta:.0f}s")

t1_brats = time.time()
print(f"Hashed {len(brats_hashes)} BraTS images in {t1_brats-t0_brats:.1f}s")

# Phase 1: pHash comparison
print("\nPhase 1: pHash comparison (threshold <= {})...".format(PHASH_THRESHOLD))
t0_compare = time.time()

phash_candidates = []
discovery_hash_list = [(sf, info['hash'], info) for sf, info in discovery_hashes.items()]

for bi, binfo in brats_hashes.items():
    bh = binfo['hash']
    for dsf, dh, dinfo in discovery_hash_list:
        dist = bh - dh  # Hamming distance
        if dist <= PHASH_THRESHOLD:
            phash_candidates.append({
                'brats_idx': bi,
                'brats_file': binfo['source_file'],
                'brats_patient': binfo['patient_id'],
                'discovery_file': dsf,
                'discovery_split': dinfo['split'],
                'discovery_label': dinfo['label'],
                'phash_distance': dist
            })

t1_compare = time.time()
print(f"Phase 1 complete: {len(phash_candidates)} candidates in {t1_compare-t0_compare:.1f}s")

# Phase 2: SSIM on candidates
print(f"\nPhase 2: SSIM verification on {len(phash_candidates)} candidates (threshold >= {SSIM_THRESHOLD})...")
t0_ssim = time.time()

confirmed_matches = []
for ci, cand in enumerate(phash_candidates):
    brats_path = os.path.join(brats_oriented_root, cand['brats_file'])
    disc_path = os.path.join(discovery_root, cand['discovery_file'])
    if not os.path.exists(disc_path):
        disc_path = cand['discovery_file']
    
    try:
        s = compute_ssim(brats_path, disc_path)
        if s >= SSIM_THRESHOLD:
            cand['ssim'] = float(s)
            confirmed_matches.append(cand)
    except Exception as e:
        pass
    
    if (ci + 1) % 100 == 0:
        print(f"  SSIM [{ci+1}/{len(phash_candidates)}]")

t1_ssim = time.time()
print(f"Phase 2 complete: {len(confirmed_matches)} confirmed matches in {t1_ssim-t0_ssim:.1f}s")

# Summarize
matched_patients = set(m['brats_patient'] for m in confirmed_matches)
matched_slices = set((m['brats_file'], m['discovery_file']) for m in confirmed_matches)

corrected_audit = {
    "dataset": "brats_oriented (np.rot90 corrected)",
    "brats_root": brats_oriented_root,
    "total_brats_slices": len(brats_hashes),
    "total_brats_patients": brats_manifest['patient_id'].nunique(),
    "total_discovery_images": len(discovery_hashes),
    "phash_threshold": PHASH_THRESHOLD,
    "ssim_threshold": SSIM_THRESHOLD,
    "ssim_resize": SSIM_SIZE,
    "phase1_phash_candidates": len(phash_candidates),
    "phase2_ssim_confirmed": len(confirmed_matches),
    "matched_patients": len(matched_patients),
    "matched_patient_ids": sorted(matched_patients),
    "matched_slice_pairs": len(matched_slices),
    "matches": confirmed_matches,
    "runtime_seconds": {
        "discovery_hashing": round(t1 - t0, 2),
        "brats_hashing": round(t1_brats - t0_brats, 2),
        "phash_comparison": round(t1_compare - t0_compare, 2),
        "ssim_verification": round(t1_ssim - t0_ssim, 2),
        "total": round(time.time() - t0, 2)
    }
}

print(f"\n--- Corrected BraTS Overlap Summary ---")
print(f"pHash candidates: {len(phash_candidates)}")
print(f"SSIM confirmed: {len(confirmed_matches)}")
print(f"Matched patients: {len(matched_patients)}")
print(f"Matched slice pairs: {len(matched_slices)}")

# ============================================================
# 30-patient sample: old PNGs under 4 orientations
# ============================================================
print("\n" + "=" * 60)
print("30-patient sample: old PNGs under 4 orientations")
print("=" * 60)

# Use the same 30 patients as the original Phase 3 audit
with open('paper_evidence/phase3_brats_overlap_audit.json') as f:
    old_audit = json.load(f)

# Get the sampled patients from the old audit
if 'sampled_patient_ids' in old_audit:
    sample_pids = old_audit['sampled_patient_ids']
elif 'sampled_patients' in old_audit:
    sample_pids = old_audit['sampled_patients']
else:
    # Pick first 30 patients
    sample_pids = sorted(brats_manifest['patient_id'].unique())[:30]

print(f"Using {len(sample_pids)} sampled patients")

sample_slices = brats_manifest[brats_manifest['patient_id'].isin(sample_pids)]
print(f"Sample slices: {len(sample_slices)}")

old_brats_root = 'D:/datasets/prepared/brats'

rotation_results = {}
rotations = {
    "0deg": lambda img: img,
    "90CW": lambda img: img.rotate(-90, expand=True),
    "90CCW": lambda img: img.rotate(90, expand=True),
    "180deg": lambda img: img.rotate(180)
}

for rot_name, rot_fn in rotations.items():
    print(f"\n  Rotation: {rot_name}")
    t0r = time.time()
    
    rot_candidates = []
    rot_confirmed = []
    
    for si, (_, srow) in enumerate(sample_slices.iterrows()):
        old_path = os.path.join(old_brats_root, srow['source_file'])
        if not os.path.exists(old_path):
            continue
        
        try:
            img = Image.open(old_path).convert('L')
            rotated = rot_fn(img)
            bh = imagehash.phash(rotated)
            
            for dsf, dh, dinfo in discovery_hash_list:
                dist = bh - dh
                if dist <= PHASH_THRESHOLD:
                    # SSIM check
                    rot_rgb = rotated.convert('RGB')
                    rot_resized = rotated.resize((SSIM_SIZE, SSIM_SIZE), Image.LANCZOS)
                    
                    disc_path = os.path.join(discovery_root, dsf)
                    if not os.path.exists(disc_path):
                        disc_path = dsf
                    disc_img = Image.open(disc_path).convert('L').resize((SSIM_SIZE, SSIM_SIZE), Image.LANCZOS)
                    
                    s = ssim(
                        np.array(rot_resized, dtype=np.float64),
                        np.array(disc_img, dtype=np.float64),
                        data_range=255.0
                    )
                    
                    rot_candidates.append({
                        'brats_file': srow['source_file'],
                        'brats_patient': srow['patient_id'],
                        'discovery_file': dsf,
                        'phash_distance': dist,
                        'ssim': float(s)
                    })
                    
                    if s >= SSIM_THRESHOLD:
                        rot_confirmed.append({
                            'brats_file': srow['source_file'],
                            'brats_patient': srow['patient_id'],
                            'discovery_file': dsf,
                            'phash_distance': dist,
                            'ssim': float(s)
                        })
        except Exception as e:
            pass
        
        if (si + 1) % 2000 == 0:
            print(f"    [{si+1}/{len(sample_slices)}]")
    
    t1r = time.time()
    rot_matched_patients = set(m['brats_patient'] for m in rot_confirmed)
    
    rotation_results[rot_name] = {
        "phash_candidates": len(rot_candidates),
        "ssim_confirmed": len(rot_confirmed),
        "matched_patients": len(rot_matched_patients),
        "matched_patient_ids": sorted(rot_matched_patients),
        "runtime_seconds": round(t1r - t0r, 2)
    }
    
    print(f"    Candidates: {len(rot_candidates)}, Confirmed: {len(rot_confirmed)}, Patients: {len(rot_matched_patients)}, Time: {t1r-t0r:.1f}s")

# ============================================================
# Save report
# ============================================================
full_report = {
    "environment": {
        "imagehash_version": imagehash.__version__,
        "skimage_version": skimage.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "corrected_full_audit": corrected_audit,
    "sample_30patient_rotation_audit": {
        "n_patients": len(sample_pids),
        "n_slices": len(sample_slices),
        "patient_ids": sorted([str(p) for p in sample_pids]),
        "rotations": rotation_results
    }
}

out_path = "paper_evidence/phase3_brats_overlap_audit_corrected.json"
with open(out_path, "w") as f:
    json.dump(full_report, f, indent=2)
print(f"\nSaved to: {out_path}")
print("DONE")

