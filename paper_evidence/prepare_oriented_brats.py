import os
import sys
import time
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

src_base = 'D:/datasets/prepared/brats'
src_images = os.path.join(src_base, 'images')

dst_rot_base = 'D:/datasets/prepared/brats_oriented'
dst_rot_images = os.path.join(dst_rot_base, 'images')

dst_flip_base = 'D:/datasets/prepared/brats_oriented_flipLR'
dst_flip_images = os.path.join(dst_flip_base, 'images')

os.makedirs(dst_rot_images, exist_ok=True)
os.makedirs(dst_flip_images, exist_ok=True)

# Copy manifest.csv
print("Copying manifests and metadata...")
shutil.copy2(os.path.join(src_base, 'manifest.csv'), os.path.join(dst_rot_base, 'manifest.csv'))
shutil.copy2(os.path.join(src_base, 'manifest.csv'), os.path.join(dst_flip_base, 'manifest.csv'))

# Copy metadata if exists and update notes
meta_path = os.path.join(src_base, 'dataset_metadata.json')
if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta_rot = dict(meta)
    meta_rot['orientation'] = 'rot90(1) applied to align anterior upward with Discovery standard'
    with open(os.path.join(dst_rot_base, 'dataset_metadata.json'), 'w') as f:
        json.dump(meta_rot, f, indent=2)
        
    meta_flip = dict(meta)
    meta_flip['orientation'] = 'rot90(1) + fliplr applied (radiological orientation sensitivity variant)'
    with open(os.path.join(dst_flip_base, 'dataset_metadata.json'), 'w') as f:
        json.dump(meta_flip, f, indent=2)

all_files = sorted(os.listdir(src_images))
total = len(all_files)
print(f"Total files to process: {total}")

def process_file(fname):
    src_p = os.path.join(src_images, fname)
    rot_p = os.path.join(dst_rot_images, fname)
    flip_p = os.path.join(dst_flip_images, fname)
    
    # Skip if both already exist and have nonzero size
    if os.path.exists(rot_p) and os.path.exists(flip_p) and os.path.getsize(rot_p) > 0 and os.path.getsize(flip_p) > 0:
        return 1
        
    arr = np.array(Image.open(src_p))
    rot = np.rot90(arr, 1)
    Image.fromarray(rot).save(rot_p)
    
    flip = np.fliplr(rot)
    Image.fromarray(flip).save(flip_p)
    return 1

t0 = time.time()
processed = 0

with ThreadPoolExecutor(max_workers=16) as ex:
    # Process in chunks of 5000 to track progress
    chunk_size = 5000
    for i in range(0, total, chunk_size):
        chunk = all_files[i:i+chunk_size]
        list(ex.map(process_file, chunk))
        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        rem = (total - processed) / rate if rate > 0 else 0
        print(f"[{processed}/{total}] ({processed/total*100:.1f}%) | {rate:.1f} files/s | Elapsed: {elapsed:.1f}s | Remaining: {rem:.1f}s")

t1 = time.time()
rot_count = len(os.listdir(dst_rot_images))
flip_count = len(os.listdir(dst_flip_images))

print(f"\nCompleted in {t1-t0:.2f} s")
print(f"Files in brats_oriented/images: {rot_count}")
print(f"Files in brats_oriented_flipLR/images: {flip_count}")

report = {
    "total_source_files": total,
    "brats_oriented_count": rot_count,
    "brats_oriented_flipLR_count": flip_count,
    "runtime_seconds": round(t1 - t0, 2),
    "date": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open('paper_evidence/brats_orientation_prep_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print("Saved report to paper_evidence/brats_orientation_prep_report.json")

