"""
Verify GPU availability and ResNet-18 test set inference reproducibility.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

print(f"PyTorch version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA available: {cuda_avail}")

if cuda_avail:
    device = torch.device('cuda:0')
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device memory allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
else:
    device = torch.device('cpu')
    print("CUDA is NOT available.")

# Check ResNet-18 test set reproducibility
ckpt_path = 'BrainTumorAI/models/tumor_classifier_reproducible.pth'
model, meta = load_checkpoint(ckpt_path, map_location=device)
model.eval()

transform = make_inference_transform()

split_df = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

# Stored test predictions
stored_df = pd.read_csv('BrainTumorAI/results/evaluation/test_predictions.csv')
stored_resnet = stored_df[stored_df['model'] == 'resnet18'].reset_index(drop=True)
stored_dict = dict(zip(stored_resnet['source_file'], stored_resnet['tumor_score']))

scores = []
stored_scores = []
sources = []

with torch.no_grad():
    for idx, row in test_df.iterrows():
        src = row['source_file']
        path = os.path.join('BrainTumorAI', src)
        if not os.path.exists(path):
            path = src
        img = Image.open(path).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        out = model(t)
        prob = torch.softmax(out, dim=1)[0, 1].item()
        scores.append(prob)
        sources.append(src)
        stored_scores.append(stored_dict[src])

scores = np.array(scores)
stored_scores = np.array(stored_scores)

diffs = np.abs(scores - stored_scores)
max_diff = float(np.max(diffs))

flips = int(np.sum((scores >= 0.5) != (stored_scores >= 0.5)))

print("\n--- Discovery Test Set Reproducibility (ResNet-18) ---")
print(f"Device used: {device}")
print(f"Total test images: {len(scores)}")
print(f"Max absolute score difference: {max_diff:.8e}")
print(f"Mean absolute score difference: {np.mean(diffs):.8e}")
print(f"Number of flipped predictions (thresh=0.5): {flips}")

