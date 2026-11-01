"""
Verify ResNet-18 and SmallCNN test prediction reproducibility on GPU.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Executing on device: {device}")
if device.type == 'cuda':
    print(f"Device name: {torch.cuda.get_device_name(0)}")

split_df = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)
transform = make_inference_transform()

stored_df = pd.read_csv('BrainTumorAI/results/evaluation/test_predictions.csv')

models_to_test = [
    ("resnet18", "BrainTumorAI/models/tumor_classifier_reproducible.pth", "resnet18"),
    ("small_cnn", "BrainTumorAI/models/small_cnn_best.pth", "small_cnn_small_cnn_best")
]

results = {}

for model_name, ckpt_path, stored_model_key in models_to_test:
    print(f"\n--- Testing {model_name} ({ckpt_path}) ---")
    model, meta = load_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    stored_sub = stored_df[stored_df['model'] == stored_model_key].reset_index(drop=True)
    stored_dict = dict(zip(stored_sub['source_file'], stored_sub['tumor_score']))
    
    scores = []
    stored_scores = []
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            src = row['source_file']
            path = os.path.join('BrainTumorAI', src)
            if not os.path.exists(path):
                path = src
            img = Image.open(path).convert('RGB')
            t = transform(img).unsqueeze(0).to(device)
            out = model(t)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            scores.append(prob)
            stored_scores.append(stored_dict[src])
            
    scores = np.array(scores)
    stored_scores = np.array(stored_scores)
    
    diffs = np.abs(scores - stored_scores)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    
    preds_live = (scores >= 0.5).astype(int)
    preds_stored = (stored_scores >= 0.5).astype(int)
    flips = int(np.sum(preds_live != preds_stored))
    
    flipped_indices = np.where(preds_live != preds_stored)[0]
    flipped_details = []
    for fi in flipped_indices:
        flipped_details.append({
            "source_file": test_df.iloc[fi]['source_file'],
            "stored_score": float(stored_scores[fi]),
            "live_score": float(scores[fi]),
            "diff": float(diffs[fi])
        })
        
    near_001 = int(np.sum(np.abs(scores - 0.5) < 0.001))
    near_01 = int(np.sum(np.abs(scores - 0.5) < 0.01))
    
    results[model_name] = {
        "n_images": len(scores),
        "max_abs_score_diff": max_diff,
        "mean_abs_score_diff": mean_diff,
        "flipped_count": flips,
        "flipped_details": flipped_details,
        "near_threshold_0_001": near_001,
        "near_threshold_0_01": near_01
    }
    
    print(f"Max abs diff: {max_diff:.8e}")
    print(f"Mean abs diff: {mean_diff:.8e}")
    print(f"Flipped count: {flips}")
    print(f"Near threshold (<0.001): {near_001}")
    print(f"Near threshold (<0.01): {near_01}")
    if flips > 0:
        print(f"Flipped images: {flipped_details}")

with open('paper_evidence/discovery_gpu_reproducibility.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to paper_evidence/discovery_gpu_reproducibility.json")
