import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs('paper_evidence/figures', exist_ok=True)

# 1. Load manifests
split_manifest = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = split_manifest[split_manifest['split'] == 'test']
disc_samples = test_df.sample(8, random_state=42)

df_brats = pd.read_csv('D:/datasets/prepared/brats/manifest.csv')
# Choose 8 slices across tumor and no-tumor
brats_samples = df_brats[df_brats['slice_id'].between(60, 100)].sample(8, random_state=42)

fig, axes = plt.subplots(3, 8, figsize=(16, 7), constrained_layout=True)

for i, (_, row) in enumerate(disc_samples.iterrows()):
    ax = axes[0, i]
    img = Image.open(row['source_file']).convert('RGB')
    ax.imshow(img)
    ax.axis('off')
    if i == 0:
        ax.set_title("Discovery Test (Ref)\nAnterior=UP", fontsize=10, fontweight='bold', loc='left')
    else:
        ax.set_title(f"Disc #{i+1}", fontsize=9)

for i, (_, row) in enumerate(brats_samples.iterrows()):
    src_p = os.path.join('D:/datasets/prepared/brats', row['source_file'])
    orig_im = Image.open(src_p)
    arr = np.array(orig_im)
    
    # Variant 1: rot90(1)
    rot_arr = np.rot90(arr, 1)
    ax1 = axes[1, i]
    ax1.imshow(rot_arr, cmap='gray')
    ax1.axis('off')
    if i == 0:
        ax1.set_title("BraTS (rot90 only)\nAnterior=UP", fontsize=10, fontweight='bold', loc='left')
    else:
        ax1.set_title(f"z={row['slice_id']}", fontsize=9)
        
    # Variant 2: rot90(1) + fliplr
    flip_arr = np.fliplr(rot_arr)
    ax2 = axes[2, i]
    ax2.imshow(flip_arr, cmap='gray')
    ax2.axis('off')
    if i == 0:
        ax2.set_title("BraTS (rot90 + flipLR)\nRadiological", fontsize=10, fontweight='bold', loc='left')
    else:
        ax2.set_title(f"z={row['slice_id']}", fontsize=9)

out_fig = 'paper_evidence/figures/orientation_corrected_montage.png'
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved corrected montage to {out_fig}")

