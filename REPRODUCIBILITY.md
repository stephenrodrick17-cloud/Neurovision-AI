# Reproducibility record

## Experiment identity and caveat

This record describes the already completed local seed-42 image-level experiment and its preserved artifacts. The audit code and interface edits in the current working tree were made after that training run; no new training run is claimed here. The experiment metadata records Git base revision `0f4d663a327e8ab638b78e6425096a9fd84c9165`, but training ran with uncommitted working-tree changes. That commit alone is therefore not a complete immutable source snapshot. Archive the exact source tree and local artifacts before treating this run as independently reproducible.

Raw data, trained checkpoints, manifests, metrics, and plots are ignored by Git and are not present in a fresh clone. The source/version/license of the local dataset are **UNVERIFIED**; see [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md).

## Data and split

- Local source layout: `archive/Brain Tumor Data Set/Brain Tumor Data Set/{Brain Tumor,Healthy}`.
- Audited snapshot: 4,600 readable images; 2,513 tumor-folder (`tumor=1`) and 2,087 healthy-folder (`no_tumor=0`).
- Dataset source, release/version, licensing terms, and independent label review remain **UNVERIFIED**. Labels are inherited from folder names.
- No patient, study, series, or source-volume identifiers were found. The split is image-level, not patient-independent.
- Seed: `42`.
- Split method: stratified group splitting after grouping exact decoded-pixel duplicates and candidate near-duplicates.
- Exact grouping: decoded-pixel SHA-256.
- Candidate near-duplicate grouping: perceptual-hash Hamming distance `<= 4` and grayscale SSIM `>= 0.95` on 64x64 images, connected components.
- Partition counts: train 3,191 (1,490 no-tumor / 1,701 tumor); validation 803 (376 / 427); final test 606 (221 / 385).
- Split-manifest SHA-256: `c0d909039b77d1d2ccb3fb27105e411eafee792b868b8203794ac5bf979d66ab`.
- The legacy prepared split had 151 exact-duplicate groups crossing train/test. The corrected manifest groups detected exact and screened near-duplicates before partitioning; residual duplication and patient overlap cannot be ruled out.
- Source/acquisition confounding and shortcut learning remain unresolved. The JPEG-only sensitivity analysis does not show that the model is free of shortcuts.
- Duplicate grouping is a heuristic leak-screen, not patient identification. Group-bootstrap intervals are not patient-clustered.

## Model and training configuration

| Setting | Recorded value |
|---|---|
| Primary architecture | Torchvision ResNet-18 with ImageNet `ResNet18_Weights.IMAGENET1K_V1` initialization; final head replaced with two logits |
| Class mapping | `no_tumor=0`, `tumor=1` |
| Input | One 2D image, RGB conversion, resize 224x224 |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`; standard deviation `(0.229, 0.224, 0.225)` |
| Training-only augmentation | Random horizontal flip and random rotation up to 10 degrees |
| Loss / imbalance | Unweighted cross-entropy; no sampler (raw class ratio about 1.2:1) |
| Optimizer | Adam; learning rate `1e-4`; weight decay `1e-4` |
| Batch size | 16 |
| Maximum epochs / early stop | 12 epochs; patience 4 |
| Model selection | Validation balanced accuracy; validation loss breaks ties |
| Scheduler | `ReduceLROnPlateau` monitored on validation only |
| Final test threshold | Fixed 0.5; not tuned against test labels |
| Seed | 42, including seeded data-loader generator |

Training and validation history, model configuration, class map, preprocessing metadata, selected epoch, environment metadata, manifest checksum, checkpoint hash, and test predictions are retained in local ignored outputs under `BrainTumorAI/results/` and `BrainTumorAI/models/`. Regeneration requires the same authorized dataset snapshot and dependency environment.

## Recorded environment

- Windows; Python 3.13.1
- PyTorch 2.7.1+cu118
- Torchvision 0.22.1+cu118
- scikit-learn 1.6.1
- NumPy 2.2.6
- Pillow 11.1.0
- OpenCV 4.10.0
- NVIDIA GeForce RTX 3050 Laptop GPU

These are the recorded local experiment versions, not a fully locked or clean-room environment specification. The repository requirements file is not a complete lockfile.

## Artifact hashes and recorded results

- ResNet-18 checkpoint SHA-256: `0ce4741c3ce2cbc287c7b716395b1571bac143200e2f3489e48efd002310b40e`.
- Small-CNN checkpoint SHA-256: `f08d9e8bf8c3c01f06281909cd61052bcd465fee860c5c89d2941f44097da05c`.
- Seed-42 split-manifest SHA-256: `c0d909039b77d1d2ccb3fb27105e411eafee792b868b8203794ac5bf979d66ab`.
- Recorded source Git base revision: `0f4d663a327e8ab638b78e6425096a9fd84c9165` (not sufficient by itself to identify the uncommitted training source tree).
- ResNet-18 held-out image-level metrics: accuracy 98.51%, balanced accuracy 98.16%, tumor sensitivity 99.48%, no-tumor specificity 96.83%, tumor precision 98.21%, tumor F1 98.84%, ROC-AUC 0.999647, average precision 0.999800; confusion matrix `[no_tumor, tumor] = [[214, 7], [2, 383]]`; predicted counts 216 no-tumor / 390 tumor.
- Small-CNN baseline: accuracy 81.02%, balanced accuracy 78.70%, tumor sensitivity 87.27%, no-tumor specificity 70.14%, ROC-AUC 0.904472.
- Historical legacy checkpoint on the same 606-image test partition: all predictions were tumor; 63.53% accuracy, 50.00% balanced accuracy, 0.00% specificity, ROC-AUC 0.495152. This does not establish why its predictions collapsed.
- Confidence intervals use duplicate-group-stratified bootstrap, not patient clusters.
- A JPEG-only post hoc sensitivity analysis (590 images; all 16 TIFFs in the full test set were tumor) yielded accuracy 98.47%, balanced accuracy 98.15%, ROC-AUC 0.999644, AP 0.999790, and confusion matrix `[[214, 7], [2, 367]]`. It was not used to modify model selection, threshold, or preprocessing and does not rule out other source confounding.

These measurements are preliminary performance against folder-derived labels on an image-level split. They do not establish patient-level generalization, external validation, calibrated probability, clinical performance, or clinical readiness.

The legacy checkpoint predicted all 606 images as tumor on the corrected test split. Its original training failure remains unexplained because complete training logs and experiment provenance are unavailable. No expert localization annotations or reference segmentation masks were available, so Grad-CAM is qualitative only and no Dice, IoU, or localization metric is supported. 3D coordinates, geometry, and planning displays are simulated; they are not anatomically registered, do not establish physical tumor location or volume, and are not clinical treatment-planning capability.

## Saved results versus a new run

The saved metrics above are historical results from the preserved seed-42 run. They were not generated or changed during this audit. To inspect the saved evaluation without running training or overwriting it:

```powershell
Get-Content BrainTumorAI/results/evaluation/evaluation_metrics.json
```

To re-evaluate the existing checkpoints and manifest without retraining, use a new output directory. This is a new evaluation pass, not an independent reproduction of the historical training run:

```powershell
python -m BrainTumorAI.evaluate_classifier `
  --manifest BrainTumorAI/results/split_manifest.csv `
  --checkpoint BrainTumorAI/models/tumor_classifier_reproducible.pth `
  --compare-checkpoint BrainTumorAI/models/small_cnn_best.pth `
  --legacy-checkpoint BrainTumorAI/models/tumor_classifier.pth `
  --results-dir BrainTumorAI/results/evaluation_audit_rerun `
  --threshold 0.5 --batch-size 32 --bootstrap-replicates 1000
```

Omit `--legacy-checkpoint` if that optional checkpoint is unavailable. The evaluation command reads the held-out test split and writes new outputs; it must not be used to tune the model, threshold, preprocessing, or hyperparameters.

## Fresh experiment commands

The commands below create a separate manifest, result tree, and checkpoints, preserving the existing recorded artifacts. Run them from the repository root after installing the declared dependencies and placing the authorized dataset at the recorded source location. The dataset provenance and license are **UNVERIFIED**; do not redistribute it without establishing authorization.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m BrainTumorAI.preprocess `
  --data-root "archive/Brain Tumor Data Set/Brain Tumor Data Set" `
  --results-dir BrainTumorAI/results/reproduction_seed42 --seed 42

python -m BrainTumorAI.train_classifier `
  --manifest BrainTumorAI/results/reproduction_seed42/split_manifest.csv `
  --architecture resnet18 `
  --checkpoint BrainTumorAI/models/reproduction_seed42_resnet18.pth `
  --results-dir BrainTumorAI/results/reproduction_seed42 `
  --seed 42 --epochs 12 --batch-size 16 --patience 4

python -m BrainTumorAI.train_classifier `
  --manifest BrainTumorAI/results/reproduction_seed42/split_manifest.csv `
  --architecture small_cnn `
  --checkpoint BrainTumorAI/models/reproduction_seed42_small_cnn.pth `
  --results-dir BrainTumorAI/results/reproduction_seed42 `
  --seed 42 --epochs 12 --batch-size 16 --patience 4

python -m BrainTumorAI.evaluate_classifier `
  --manifest BrainTumorAI/results/reproduction_seed42/split_manifest.csv `
  --checkpoint BrainTumorAI/models/reproduction_seed42_resnet18.pth `
  --compare-checkpoint BrainTumorAI/models/reproduction_seed42_small_cnn.pth `
  --legacy-checkpoint BrainTumorAI/models/tumor_classifier.pth `
  --results-dir BrainTumorAI/results/reproduction_seed42/evaluation `
  --threshold 0.5 --batch-size 32 --bootstrap-replicates 1000

python -m pytest -q tests/test_classifier_pipeline.py
streamlit run BrainTumorAI/app.py
```

Omit the legacy-checkpoint option when the local legacy checkpoint is absent. ResNet-18 uses ImageNet V1 initialization; the first fresh training run may need network access to retrieve those weights if they are not cached. The recorded experiment used Windows/Python 3.13.1, while deployment metadata specifies Python 3.11.7. `requirements.txt` is not a lockfile, so the historical package environment is not fully pinned.

## Current software verification (2026-10-10)

This audit did not retrain either model or change checkpoints, source images, the split manifest, or saved evaluation results.

| Check | Actual result |
|---|---|
| Focused tests, plugin auto-load disabled | `19 passed in 15.86s`; no test failures, skips, or errors. Command: `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'; & 'C:\Users\asus\python.exe' -m pytest -q tests/test_classifier_pipeline.py`. All tests ran; this setting excluded unrelated globally installed pytest plugins. |
| Focused tests, normal plugin auto-load | Command `& 'C:\Users\asus\python.exe' -m pytest -q tests/test_classifier_pipeline.py` failed before test collection because a globally installed plugin dependency rejected an inherited extra environment setting during Pydantic settings initialization. No project tests ran in this invocation. |
| Python compilation | Passed with exit code 0: `& 'C:\Users\asus\python.exe' -m compileall -q BrainTumorAI tests`. |
| Git whitespace check | Passed after documentation changes: `git diff --check`. |
| Current Streamlit process | Launched with `& 'C:\Users\asus\python.exe' -m streamlit run BrainTumorAI/app.py --server.headless=true --server.port=18765 --server.address=127.0.0.1`; `/_stcore/health` returned HTTP 200 and `ok`; `/` returned HTTP 200. Automatic dotenv loading was disabled for this isolated runtime check. |
| Current app script/checkpoint loading | `streamlit.testing.v1.AppTest` rendered the current app on its About page with zero app exceptions and the saved ResNet-18 checkpoint available. AppTest emitted two bare-mode `ScriptRunContext` warnings. |
| Inference smoke test | The shared classifier inference path returned two finite logits for a synthetic 224x224 RGB image. This is a software smoke test, not a scientific metric or held-out-data evaluation. The prior 12-image evaluator/application parity check predates the latest UI-copy changes and was not repeated here. |
| Artifact checksums | Current ResNet-18 checkpoint, small-CNN checkpoint, and split manifest SHA-256 values match those recorded above. |

Application startup and the basic health/inference smoke checks were verified after the current UI-copy changes. The current check did not run an uploaded clinical image through the Streamlit browser widget and does not establish clinical or patient-independent function.

## Known reproduction gaps

1. Verify the dataset's authoritative source, release/version, license, and labeling procedure.
2. Archive an immutable source revision or patch corresponding exactly to the trained experiment.
3. Reproduce from a clean, fully pinned environment and an authorized, hashed input snapshot.
4. Establish patient/study/source-volume identifiers and use group-independent partitions; obtain a truly independent validation cohort.
5. Have qualified reviewers assess label validity and investigate source/acquisition shortcuts.
6. Add reference masks/localizations before claiming or quantitatively measuring Grad-CAM localization.
7. Replace simulated spatial/planning views with a separately validated, geometry-aware pipeline before making anatomical or treatment-planning claims.
