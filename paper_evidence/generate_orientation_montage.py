import os
import io
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.external_validation import _decode_cheng_mat

os.makedirs('paper_evidence/figures', exist_ok=True)

# 1. Load manifests
df_split = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = df_split[df_split['split'] == 'test']

df_cheng = pd.read_csv('BrainTumorAI/results/external_validation/cheng_figshare_v8/manifest.csv')
df_brats = pd.read_csv('D:/datasets/prepared/brats/manifest.csv')

# Fixed random state for reproducibility
seed = 42

disc_notumor = test_df[test_df['label'] == 'no_tumor'].sample(8, random_state=seed)
disc_tumor = test_df[test_df['label'] == 'tumor'].sample(8, random_state=seed)

cheng_meningioma = df_cheng[df_cheng['subtype'] == 'meningioma'].sample(8, random_state=seed)
cheng_glioma = df_cheng[df_cheng['subtype'] == 'glioma'].sample(8, random_state=seed)
cheng_pituitary = df_cheng[df_cheng['subtype'] == 'pituitary'].sample(8, random_state=seed)

brats_notumor = df_brats[df_brats['subtype'] == 'no-tumor slice'].sample(8, random_state=seed)
brats_tumor = df_brats[df_brats['subtype'] == 'tumor-containing slice'].sample(8, random_state=seed)

rows_config = [
    ("Discovery Test (No Tumor)", disc_notumor, "discovery"),
    ("Discovery Test (Tumor)", disc_tumor, "discovery"),
    ("Cheng (Meningioma)", cheng_meningioma, "cheng"),
    ("Cheng (Glioma)", cheng_glioma, "cheng"),
    ("Cheng (Pituitary)", cheng_pituitary, "cheng"),
    ("BraTS 2023 (No-Tumor)", brats_notumor, "brats"),
    ("BraTS 2023 (Tumor)", brats_tumor, "brats"),
]

fig, axes = plt.subplots(7, 8, figsize=(16, 15), constrained_layout=True)

# We will open 1512427.zip once
outer_zip = zipfile.ZipFile('1512427.zip')

print("=== SELECTED IMAGES DETAILS ===")
for r_idx, (label, sample_df, cohort) in enumerate(rows_config):
    print(f"\n--- Row {r_idx}: {label} ---")
    for c_idx, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes[r_idx, c_idx]
        if cohort == "discovery":
            img_path = row['source_file']
            img = Image.open(img_path).convert('RGB')
            fname = os.path.basename(img_path)
            print(f"Col {c_idx}: {fname} (size: {img.size})")
        elif cohort == "cheng":
            source_uri = row['source_file']
            part_name, _, mat_name = source_uri.removeprefix('archive://').partition('!')
            part_bytes = outer_zip.read(part_name)
            with zipfile.ZipFile(io.BytesIO(part_bytes)) as inner:
                mat_bytes = inner.read(mat_name)
                image_arr, pid, lbl = _decode_cheng_mat(mat_bytes, mat_name)
                low, high = float(image_arr.min()), float(image_arr.max())
                pixels = np.rint((image_arr.astype(np.float64) - low) * (255.0 / (high - low)))
                gray = np.clip(pixels, 0, 255).astype(np.uint8)
                img = Image.fromarray(gray, mode='L').convert('RGB')
                print(f"Col {c_idx}: Mat={mat_name}, PID={pid}, Subtype={row['subtype']} (arr shape: {image_arr.shape})")
        elif cohort == "brats":
            img_path = os.path.join('D:/datasets/prepared/brats', row['source_file'])
            img = Image.open(img_path).convert('RGB')
            fname = os.path.basename(img_path)
            print(f"Col {c_idx}: {fname}, PID={row['patient_id']}, Slice={row['slice_id']}")

        ax.imshow(img)
        ax.axis('off')
        if c_idx == 0:
            ax.set_title(f"{label}\n(Col 1)", fontsize=10, loc='left', fontweight='bold')
        else:
            ax.set_title(f"#{c_idx+1}", fontsize=9)

outer_zip.close()

out_path = 'paper_evidence/figures/orientation_check.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved montage to: {out_path}")
