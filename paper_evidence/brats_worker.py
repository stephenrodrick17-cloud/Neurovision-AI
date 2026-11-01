import os
import time
import torch
import numpy as np
from PIL import Image
from BrainTumorAI.classifier_pipeline import load_checkpoint, make_inference_transform

def run_shard_inference(file_list, num_threads=2, batch_size=128):
    torch.set_num_threads(num_threads)
    device = torch.device('cpu')
    model, _ = load_checkpoint('BrainTumorAI/models/tumor_classifier_reproducible.pth', map_location=device)
    model.eval()
    tf = make_inference_transform()
    
    scores = []
    batch_tensors = []
    
    t0 = time.time()
    with torch.no_grad():
        for i, f in enumerate(file_list):
            im = Image.open(f).convert('RGB')
            batch_tensors.append(tf(im))
            if len(batch_tensors) == batch_size or i == len(file_list) - 1:
                b = torch.stack(batch_tensors).to(device)
                logits = model(b)
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
                scores.extend(probs.tolist())
                batch_tensors = []
    t1 = time.time()
    return scores, t1 - t0
