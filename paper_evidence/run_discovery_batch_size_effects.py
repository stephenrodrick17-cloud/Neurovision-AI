"""
Evaluate batch size effects:
1. ResNet-18 & SmallCNN on all 606 Discovery test images at batch 1 vs batch 32:
   - count flipped predictions
   - max score difference
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform
from BrainTumorAI.train_classifier import ManifestDataset

def evaluate_models():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    split_df = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
    test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)
    print(f"Discovery test images: {len(test_df)}")
    
    models_to_test = [
        ("resnet18", "BrainTumorAI/models/tumor_classifier_reproducible.pth", "resnet18_tumor_classifier_reproducible"),
        ("smallcnn", "BrainTumorAI/models/small_cnn_best.pth", "small_cnn_small_cnn_best")
    ]
    
    results = {}
    
    for display_name, ckpt_path, stored_name in models_to_test:
        print(f"\n==========================================")
        print(f"Testing {display_name} ({ckpt_path})")
        print(f"==========================================")
        
        model, _ = load_checkpoint(ckpt_path, map_location=device)
        model = model.to(device).eval()
        
        # 1. Batch size 1
        ds_b1 = ManifestDataset(test_df, training=False)
        loader_b1 = DataLoader(ds_b1, batch_size=1, shuffle=False, num_workers=0)
        scores_b1 = []
        with torch.no_grad():
            for imgs, _ in loader_b1:
                probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
                scores_b1.extend(probs)
        scores_b1 = np.array(scores_b1)
        
        # 2. Batch size 32
        ds_b32 = ManifestDataset(test_df, training=False)
        loader_b32 = DataLoader(ds_b32, batch_size=32, shuffle=False, num_workers=0)
        scores_b32 = []
        with torch.no_grad():
            for imgs, _ in loader_b32:
                probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
                scores_b32.extend(probs)
        scores_b32 = np.array(scores_b32)
        
        diffs = np.abs(scores_b1 - scores_b32)
        max_diff = float(np.max(diffs))
        mean_diff = float(np.mean(diffs))
        
        preds_b1 = (scores_b1 >= 0.5).astype(int)
        preds_b32 = (scores_b32 >= 0.5).astype(int)
        flips = int(np.sum(preds_b1 != preds_b32))
        
        flipped_cases = []
        for i in range(len(preds_b1)):
            if preds_b1[i] != preds_b32[i]:
                row = test_df.iloc[i]
                flipped_cases.append({
                    "image": os.path.basename(row['source_file']),
                    "source_file": str(row['source_file']),
                    "label": str(row['label']),
                    "score_b1": float(scores_b1[i]),
                    "score_b32": float(scores_b32[i]),
                    "pred_b1": int(preds_b1[i]),
                    "pred_b32": int(preds_b32[i])
                })
                
        print(f"Max abs diff (batch 1 vs 32): {max_diff:.8e}")
        print(f"Mean abs diff: {mean_diff:.8e}")
        print(f"Flipped predictions: {flips} / {len(test_df)}")
        if flipped_cases:
            print("Flipped cases:")
            for fc in flipped_cases:
                print(f"  {fc['image']}: b1={fc['score_b1']:.8f} (pred {fc['pred_b1']}) vs b32={fc['score_b32']:.8f} (pred {fc['pred_b32']})")
                
        results[display_name] = {
            "checkpoint": ckpt_path,
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "flips": flips,
            "flipped_cases": flipped_cases
        }
        
    os.makedirs('paper_evidence', exist_ok=True)
    with open('paper_evidence/discovery_batch_size_effects.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to paper_evidence/discovery_batch_size_effects.json")

if __name__ == '__main__':
    evaluate_models()
