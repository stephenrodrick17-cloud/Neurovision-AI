import os
import io
import sys
import json
import zipfile
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.external_validation import _decode_cheng_mat

os.makedirs('paper_evidence/figures', exist_ok=True)

df_cheng = pd.read_csv('BrainTumorAI/results/external_validation/cheng_figshare_v8/manifest.csv')
seed = 42

meningioma_sample = df_cheng[df_cheng['subtype'] == 'meningioma'].sample(40, random_state=seed)
glioma_sample = df_cheng[df_cheng['subtype'] == 'glioma'].sample(40, random_state=seed)
pituitary_sample = df_cheng[df_cheng['subtype'] == 'pituitary'].sample(40, random_state=seed)

subtypes_data = [
    ("Meningioma", meningioma_sample),
    ("Glioma", glioma_sample),
    ("Pituitary", pituitary_sample)
]

# Map each target to its part and mat name, and store target grid position
targets = []
# targets: list of (part_name, mat_name, pid, subtype, r, c)
row_offset = 0
for subtype_name, sample_df in subtypes_data:
    for idx, (_, row) in enumerate(sample_df.reset_index(drop=True).iterrows()):
        r = row_offset + idx // 10
        c = idx % 10
        source_uri = row['source_file']
        part_name, _, mat_name = source_uri.removeprefix('archive://').partition('!')
        targets.append((part_name, mat_name, row['patient_id'], subtype_name, r, c))
    row_offset += 4

# Group targets by part_name
by_part = defaultdict(list)
for t in targets:
    by_part[t[0]].append(t)

decoded_images = {} # (r, c) -> (gray, mat_name, pid)

print("Reading Cheng archive efficiently by inner zip part...")
with zipfile.ZipFile('1512427.zip') as outer:
    for part_name, items in by_part.items():
        print(f"Reading {part_name} ({len(items)} slices)...")
        part_bytes = outer.read(part_name)
        with zipfile.ZipFile(io.BytesIO(part_bytes)) as inner:
            for _, mat_name, pid, subtype, r, c in items:
                mat_bytes = inner.read(mat_name)
                img_arr, _, _ = _decode_cheng_mat(mat_bytes, mat_name)
                low, high = float(img_arr.min()), float(img_arr.max())
                pixels = np.rint((img_arr.astype(np.float64) - low) * (255.0 / (high - low)))
                gray = np.clip(pixels, 0, 255).astype(np.uint8)
                decoded_images[(r, c)] = (gray, mat_name, pid, subtype)

fig, axes = plt.subplots(12, 10, figsize=(20, 24), constrained_layout=True)

for (r, c), (gray, mat_name, pid, subtype) in decoded_images.items():
    ax = axes[r, c]
    ax.imshow(gray, cmap='gray')
    ax.axis('off')
    if c == 0 and r % 4 == 0:
        ax.set_title(f"[{subtype}]\n{mat_name}\nPID:{pid}", fontsize=8, fontweight='bold', loc='left')
    else:
        ax.set_title(f"{mat_name}\nPID:{pid}", fontsize=7)

out_fig = 'paper_evidence/figures/cheng_plane_check.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved contact sheet to {out_fig}")
