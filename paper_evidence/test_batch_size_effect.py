import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from PIL import Image
import pandas as pd
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform
from BrainTumorAI.train_classifier import ManifestDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0')
model, _ = load_checkpoint('BrainTumorAI/models/small_cnn_best.pth', map_location=device)
model = model.to(device).eval()

split_df = pd.read_csv('BrainTumorAI/results/split_manifest.csv')
test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)
target_idx = test_df[test_df['source_file'].str.contains('Cancer \(2164\)', regex=True)].index[0]
print(f'Target index: {target_idx}')
target_file = test_df.iloc[target_idx]['source_file']
print(f'Target file: {target_file}')

transform = make_inference_transform()

# 1. Batch size 1 (as in Phase 1B evaluate_model)
img = Image.open(target_file).convert('RGB')
t1 = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    p1 = torch.softmax(model(t1), dim=1)[0, 1].item()
print(f'Batch size 1 (Phase 1B style) score: {p1:.8f}')

# 2. Batch size 32 via DataLoader (as in evaluate_classifier.py)
ds = ManifestDataset(test_df, training=False)
loader = DataLoader(ds, batch_size=32, shuffle=False)
batch_scores = []
with torch.no_grad():
    for imgs, _ in loader:
        probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().tolist()
        batch_scores.extend(probs)

p32 = batch_scores[target_idx]
print(f'Batch size 32 (evaluate_classifier.py style) score: {p32:.8f}')

stored_p = pd.read_csv('BrainTumorAI/results/evaluation/test_predictions.csv')
stored_score = stored_p[(stored_p['model']=='small_cnn_small_cnn_best') & (stored_p['source_file']==target_file)]['tumor_score'].values[0]
print(f'Stored primary score: {stored_score:.8f}')

p1b_p = pd.read_csv('experiments_results/phase1b/small_cnn_seed42/test_predictions.csv')
p1b_score = p1b_p[p1b_p['source_file']==target_file]['tumor_score'].values[0]
print(f'Phase 1B stored score: {p1b_score:.8f}')

