"""
Evaluate DenseNet-121 seeds 42-44 on the 250-patient stratified subset (33,696 slices)
using ThreadPoolExecutor for fast Windows I/O and batch_size=64 (avoiding 4GB VRAM paging),
and combine with the verified ResNet-18 (primary + seeds 42-46) and SmallCNN (primary + seeds 42-46)
results from task-1976 on the identical 33,696-slice manifest.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath('.'))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from BrainTumorAI.classifier_pipeline import make_inference_transform
from experiments.phase1b_training import build_model

def load_densenet_checkpoint(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = payload.get("architecture", "densenet121")
    model = build_model(arch, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()

def main():
    t0 = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    df_sub = pd.read_csv('paper_evidence/brats_250_patients_manifest.csv')
    total_len = len(df_sub)
    print(f"Loaded 250-patient subset: {total_len} slices across {df_sub['patient_id'].nunique()} patients", flush=True)
    y_true = df_sub['label_idx'].values

    transform = make_inference_transform()
    root_dir = 'D:/datasets/prepared/brats_oriented'
    files = [os.path.join(root_dir, f) for f in df_sub['source_file']]

    densenet_ckpts = [
        ("seed42", "experiments_results/phase1b/densenet121_seed42/checkpoint.pth"),
        ("seed43", "experiments_results/phase1b/densenet121_seed43/checkpoint.pth"),
        ("seed44", "experiments_results/phase1b/densenet121_seed44/checkpoint.pth"),
    ]

    models = {}
    scores_dict = {}
    for name, ckpt_path in densenet_ckpts:
        models[name] = load_densenet_checkpoint(ckpt_path, device)
        scores_dict[name] = np.zeros(total_len, dtype=np.float32)
    print(f"Loaded 3 DenseNet-121 checkpoints onto {device} in {time.time() - t0:.2f}s", flush=True)

    def load_one(path):
        im = Image.open(path).convert('RGB')
        return transform(im)

    batch_size = 64
    num_batches = (total_len + batch_size - 1) // batch_size
    t_pass = time.time()

    with ThreadPoolExecutor(max_workers=8) as pool, torch.no_grad():
        for b_idx in range(num_batches):
            start_i = b_idx * batch_size
            end_i = min(total_len, start_i + batch_size)
            batch_files = files[start_i:end_i]
            tensors = list(pool.map(load_one, batch_files))
            imgs_gpu = torch.stack(tensors, dim=0).to(device, non_blocking=True)

            for name, model in models.items():
                probs = torch.softmax(model(imgs_gpu), dim=1)[:, 1].cpu().numpy()
                scores_dict[name][start_i:end_i] = probs

            if (b_idx + 1) in (1, 25, 50) or (b_idx + 1) % 100 == 0 or (b_idx + 1) == num_batches:
                print(f"  Batch {b_idx+1}/{num_batches} ({time.time() - t_pass:.1f}s)", flush=True)

    # Verified ResNet-18 and SmallCNN results from task-1976 on the exact same 33,696 slices
    resnet18_runs = [
        {"model_name": "seed42 (primary reference)", "checkpoint": "BrainTumorAI/models/tumor_classifier_reproducible.pth", "accuracy": 0.7117, "balanced_accuracy": 0.6985, "sensitivity": 0.4107, "specificity": 0.9863, "auc": 0.7088},
        {"model_name": "seed42 (phase1b)", "checkpoint": "experiments_results/phase1b/resnet18_seed42/checkpoint.pth", "accuracy": 0.7117, "balanced_accuracy": 0.6985, "sensitivity": 0.4107, "specificity": 0.9863, "auc": 0.7088},
        {"model_name": "seed43", "checkpoint": "experiments_results/phase1b/resnet18_seed43/checkpoint.pth", "accuracy": 0.6451, "balanced_accuracy": 0.62835, "sensitivity": 0.2631, "specificity": 0.9936, "auc": 0.6587},
        {"model_name": "seed44", "checkpoint": "experiments_results/phase1b/resnet18_seed44/checkpoint.pth", "accuracy": 0.5265, "balanced_accuracy": 0.51665, "sensitivity": 0.3030, "specificity": 0.7303, "auc": 0.5050},
        {"model_name": "seed45", "checkpoint": "experiments_results/phase1b/resnet18_seed45/checkpoint.pth", "accuracy": 0.5578, "balanced_accuracy": 0.55405, "sensitivity": 0.4721, "specificity": 0.6360, "auc": 0.5705},
        {"model_name": "seed46", "checkpoint": "experiments_results/phase1b/resnet18_seed46/checkpoint.pth", "accuracy": 0.6766, "balanced_accuracy": 0.66805, "sensitivity": 0.4824, "specificity": 0.8537, "auc": 0.6651},
    ]

    smallcnn_runs = [
        {"model_name": "seed42 (primary)", "checkpoint": "BrainTumorAI/models/small_cnn_best.pth", "accuracy": 0.4146, "balanced_accuracy": 0.4278, "sensitivity": 0.7153, "specificity": 0.1403, "auc": 0.3914},
        {"model_name": "seed42 (phase1b)", "checkpoint": "experiments_results/phase1b/small_cnn_seed42/checkpoint.pth", "accuracy": 0.4146, "balanced_accuracy": 0.4278, "sensitivity": 0.7153, "specificity": 0.1403, "auc": 0.3914},
        {"model_name": "seed43", "checkpoint": "experiments_results/phase1b/small_cnn_seed43/checkpoint.pth", "accuracy": 0.5302, "balanced_accuracy": 0.5138, "sensitivity": 0.1566, "specificity": 0.8710, "auc": 0.3939},
        {"model_name": "seed44", "checkpoint": "experiments_results/phase1b/small_cnn_seed44/checkpoint.pth", "accuracy": 0.5405, "balanced_accuracy": 0.51965, "sensitivity": 0.0653, "specificity": 0.9740, "auc": 0.4418},
        {"model_name": "seed45", "checkpoint": "experiments_results/phase1b/small_cnn_seed45/checkpoint.pth", "accuracy": 0.5623, "balanced_accuracy": 0.5491, "sensitivity": 0.2624, "specificity": 0.8358, "auc": 0.5150},
        {"model_name": "seed46", "checkpoint": "experiments_results/phase1b/small_cnn_seed46/checkpoint.pth", "accuracy": 0.5291, "balanced_accuracy": 0.5235, "sensitivity": 0.4020, "specificity": 0.6450, "auc": 0.5005},
    ]

    densenet_runs = []
    print("\nResults for DENSENET121:", flush=True)
    for name, ckpt_path in densenet_ckpts:
        scores = scores_dict[name]
        preds = (scores >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, preds))
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        sens = float(recall_score(y_true, preds, pos_label=1))
        spec = float(recall_score(y_true, preds, pos_label=0))
        auc = float(roc_auc_score(y_true, scores))
        print(f"  {name} ({ckpt_path}): Acc={acc:.4f}, BalAcc={bal_acc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}, AUC={auc:.4f}", flush=True)
        densenet_runs.append({
            "model_name": name,
            "checkpoint": ckpt_path,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "sensitivity": sens,
            "specificity": spec,
            "auc": auc
        })

    def summarize_runs(runs):
        if not runs:
            return {}
        out_s = {}
        for m in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "auc"]:
            vals = [r[m] for r in runs]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sd_pop = float(np.std(vals, ddof=0))
            out_s[m] = {
                "mean": mean_v,
                "sd_sample": sd_v,
                "sd_pop": sd_pop,
                "mean_sd_sample_str": f"{mean_v:.4f} +/- {sd_v:.4f}",
                "mean_sd_pop_str": f"{mean_v:.4f} +/- {sd_pop:.4f}"
            }
        return out_s

    all_arch_runs = {
        "resnet18": resnet18_runs,
        "smallcnn": smallcnn_runs,
        "densenet121": densenet_runs
    }

    results = {
        "subset_info": {
            "n_patients": int(df_sub['patient_id'].nunique()),
            "n_slices": int(total_len),
            "n_tumor_slices": int(np.sum(y_true == 1)),
            "n_no_tumor_slices": int(np.sum(y_true == 0)),
            "manifest_path": "paper_evidence/brats_250_patients_manifest.csv"
        },
        "architectures": {}
    }

    for arch, arch_results in all_arch_runs.items():
        seed_runs_all = [r for r in arch_results if "primary" not in r["model_name"]]
        seed_runs_43_46 = [r for r in arch_results if r["model_name"] in ("seed43", "seed44", "seed45", "seed46")]
        results["architectures"][arch] = {
            "individual_runs": arch_results,
            "summary_all_phase1b_seeds": summarize_runs(seed_runs_all),
            "summary_seeds_43_to_46": summarize_runs(seed_runs_43_46) if seed_runs_43_46 else None
        }

    with open('paper_evidence/brats_250_other_checkpoints.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to paper_evidence/brats_250_other_checkpoints.json in {time.time() - t0:.2f}s", flush=True)

if __name__ == '__main__':
    main()
