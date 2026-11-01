import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.abspath('.'))
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

# Load manifest
split_manifest = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = split_manifest[split_manifest['split'] == 'test'].reset_index(drop=True)
print(f"Total test images: {len(test_df)}")

# Load model
ckpt_path = 'BrainTumorAI/models/tumor_classifier_reproducible.pth'
device = 'cpu'
model, meta = load_checkpoint(ckpt_path, map_location=device)
model.eval()

transform = make_inference_transform()

rotations = {
    "Original (0 deg)": None,
    "90 deg CW": Image.Transpose.ROTATE_270,
    "90 deg CCW": Image.Transpose.ROTATE_90,
    "180 deg": Image.Transpose.ROTATE_180
}

results = {}
y_true = (test_df['label'] == 'tumor').astype(int).values

for rot_name, rot_op in rotations.items():
    # Batch inference
    batch_tensors = []
    scores = []
    for idx, row in test_df.iterrows():
        img_path = row['source_file']
        img = Image.open(img_path).convert('RGB')
        if rot_op is not None:
            img = img.transpose(rot_op)
        tensor = transform(img)
        batch_tensors.append(tensor)
        
        if len(batch_tensors) == 64 or idx == len(test_df) - 1:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                tumor_scores = probs[:, 1].cpu().numpy().tolist()
                scores.extend(tumor_scores)
            batch_tensors = []
    
    scores = np.array(scores)
    preds = (scores >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, preds)
    sens = recall_score(y_true, preds, pos_label=1)
    spec = recall_score(y_true, preds, pos_label=0)
    auc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, preds).tolist()
    
    results[rot_name] = {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "sample_count": len(scores)
    }
    print(f"\n--- {rot_name} ---")
    print(f"Accuracy: {acc*100:.2f}% | Sensitivity: {sens*100:.2f}% | Specificity: {spec*100:.2f}% | ROC-AUC: {auc:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): {cm}")

out_path = 'paper_evidence/internal_rotation_sensitivity.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved internal rotation sensitivity results to {out_path}")
