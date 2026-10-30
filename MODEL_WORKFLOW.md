# NeuroVision AI: Model and Application Workflow

This document describes the checked-in implementation and locally reproducible experiment workflow. The application is a research/education prototype, not a medical device or clinical decision-support system.

## 1. Scope and class definition

The classifier processes one two-dimensional image per inference. It predicts one of two folder-derived labels:

| Source directory | Model label | Output index |
|---|---|---:|
| `Brain Tumor` | `tumor` | 1 |
| `Healthy` | `no_tumor` | 0 |

This mapping is defined explicitly in `BrainTumorAI/classifier_pipeline.py` and verified when training, evaluation, and checkpoint loading. The model does not classify glioma/meningioma/pituitary subtypes, segment lesions, or process 3D volumes.

The available local source contains 2,513 tumor-labeled and 2,087 no-tumor-labeled images. It has 4,600 readable files in JPEG, PNG, and TIFF formats with varying dimensions and color modes. No patient or source-volume identifiers were found. The repository files do not establish the dataset's original source, release/version, license, or clinical label validity; obtain and verify these details before research redistribution or publication. This is not evidence of a BraTS experiment.

## 2. Pipeline at a glance

```text
archive/Brain Tumor Data Set/Brain Tumor Data Set/{Brain Tumor,Healthy}
    │
    ├─ python -m BrainTumorAI.preprocess
    │     ├─ image readability, dimensions, modes and format audit
    │     ├─ file and decoded-pixel hashes
    │     ├─ conservative pHash + SSIM near-duplicate grouping
    │     └─ seeded, stratified group train/validation/test manifest
    │
    ├─ python -m BrainTumorAI.train_classifier --architecture resnet18
    │     └─ ImageNet ResNet-18; validation-only scheduler/checkpoint/early stopping
    │
    ├─ python -m BrainTumorAI.train_classifier --architecture small_cnn
    │     └─ compact CNN comparison on the same train/validation split
    │
    ├─ python -m BrainTumorAI.evaluate_classifier
    │     ├─ fixed-checkpoint evaluation on the final test split
    │     ├─ majority-class, legacy-checkpoint and optional CNN comparisons
    │     └─ metrics, confidence intervals, predictions and plots
    │
    └─ streamlit run BrainTumorAI/app.py
          └─ strict checkpoint load; shared RGB, resize and normalization code
```

Generated manifests, metrics, plots, histories, metadata, and model weights are local files under `BrainTumorAI/results/` and `BrainTumorAI/models/`; they are ignored by Git and must be archived with an experiment when needed.

## 3. Data audit and split construction

`BrainTumorAI/preprocess.py` resolves paths from the repository, not the current working directory. It reads the raw source folders directly and keeps their paths, labels, hashes, dimensions, formats, and split assignments in `results/split_manifest.csv`. It does not create resized copies or modify source images. Unreadable images are listed in `results/dataset_audit.json` and excluded from the manifest used for training.

The audit found:

- 4,600 readable images and 0 unreadable images.
- 459 distinct width/height combinations (width range 150–1,920; height range 167–1,446).
- 432 exact decoded-pixel duplicate groups containing 1,017 images; there were no exact duplicate groups with conflicting folder labels.
- 2,782 images in multi-image duplicate/near-duplicate components after a conservative screen requiring both pHash Hamming distance ≤4 and 64×64 grayscale SSIM ≥0.95. This heuristic is a leakage screen, not patient identification or a substitute for manual data review.
- The legacy generated train/test folders had 151 exact-pixel duplicate groups crossing the split boundary (224 training files and 167 test files in the shared groups).

The new seeded `StratifiedGroupKFold` process creates training, validation, and final-test partitions while keeping each exact/near-duplicate component in only one partition. Group sizes mean the actual proportions need not equal exact 70/15/15. Review `dataset_audit.json` for the actual class counts and proportions. The split remains image-level: patient independence cannot be claimed without patient/source-volume IDs.

The old training script incorrectly evaluated the nominal test folder during every epoch, used its accuracy for scheduler updates, and selected the final checkpoint on that same folder. Thus the old test set was a validation set in practice. An earlier historical run loaded a shape-compatible two-output checkpoint strictly and predicted all 920 images in its old test folder as tumor: 399 no-tumor and 521 tumor. The recorded accuracy was 56.63%, balanced accuracy 50%, specificity 0%, and ROC-AUC 0.487. These old-folder values are not an unbiased performance estimate. The underlying checkpoint/training logs do not establish why it collapsed to one class; the corrected split and evaluation are intended to diagnose this without claiming an unverified cause.

## 4. Training

`BrainTumorAI/train_classifier.py` reads only `train` and `validation` rows. It verifies label indices against the explicit class map and refuses a manifest with duplicate hash groups spanning partitions. Test rows are not loaded or evaluated during training.

The default baseline is ResNet-18 initialized with `ResNet18_Weights.IMAGENET1K_V1`, with a replacement two-logit classifier head. The additional `small_cnn` architecture supports a reasonable smaller-model baseline. Both use cross-entropy, Adam, and identical split manifests. The dataset imbalance (about 1.2:1 in the raw source) is modest, so the implementation uses unweighted loss and no balanced sampler.

Training and validation preprocessing both convert to RGB, resize to 224×224, convert to a tensor, and normalize with ImageNet means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. Only training adds random horizontal flips and rotations up to 10 degrees. Evaluation and Streamlit inference share the same inference-transform factory. All random sources are seeded; the DataLoader uses a seeded generator and no worker processes.

Each epoch records train/validation loss, accuracy, balanced accuracy, and learning rate. `ReduceLROnPlateau`, best-checkpoint selection, and early stopping use validation balanced accuracy only (validation loss breaks ties). Checkpoint metadata records the class map, transform settings, architecture, seed, training settings, manifest hash, software versions, selected epoch, validation score, and code revision where Git can provide it. A separate summary records the checkpoint SHA-256 and training duration.

## 5. Evaluation

`BrainTumorAI/evaluate_classifier.py` loads the frozen checkpoint strictly and reads only the held-out `test` rows. The default classification threshold is fixed at 0.5; no threshold is selected using test labels. The tumor ROC-AUC and average precision use continuous tumor-class softmax scores, not hard predictions.

The report includes accuracy, balanced accuracy, tumor sensitivity/recall, no-tumor specificity, tumor precision and F1, confusion matrix, per-class classification report, ROC-AUC, average precision, and explicit predicted tumor/no-tumor counts. It flags one-class prediction collapse. The primary 95% percentile intervals use a stratified bootstrap that resamples complete exact/near-duplicate groups; a secondary image-level bootstrap is also saved. Neither can respect patient clustering because the dataset has no patient IDs. The evaluator and training manifest validator both reject duplicate-group leakage across partitions.

The same final split can compare:

1. A training-majority-class baseline (majority label determined from training data only).
2. The old local ResNet-18 state-dictionary checkpoint, when available.
3. The corrected ResNet-18 checkpoint.
4. The small-CNN checkpoint.

Run all planned comparisons together in one final evaluation command. Do not use final-test outputs to choose a model, threshold, preprocessing, or hyperparameters. Evaluation output is written to `BrainTumorAI/results/evaluation/`, including `evaluation_metrics.json`, `test_predictions.csv`, confusion-matrix images, and ROC/precision-recall curves.

### Measured seed-42 run

On the fixed 606-image final test set (221 `no_tumor`, 385 `tumor`), with threshold 0.5, the corrected ResNet-18 achieved 98.51% accuracy, 98.16% balanced accuracy, 99.48% tumor sensitivity, 96.83% no-tumor specificity, 0.9996 ROC-AUC, and 0.9998 average precision. It predicted 216 images as no-tumor and 390 as tumor; confusion matrix `[no_tumor, tumor]` was `[[214, 7], [2, 383]]`. Duplicate-group-bootstrap 95% intervals were 96.28–100.00% for accuracy, 95.08–100.00% for balanced accuracy, 98.40–100.00% for sensitivity, 90.95–100.00% for specificity, and 0.9987–1.0000 for ROC-AUC.

On this same split, the small CNN achieved 81.02% accuracy and 0.9045 ROC-AUC. The legacy checkpoint predicted tumor for all 606 images (63.53% accuracy, 50.00% balanced accuracy, 0% specificity, ROC-AUC 0.4952), matching the training-majority baseline's accuracy while providing no useful discrimination. These are measured image-level results against folder-derived labels, not patient-independent or clinical performance. Their unusually strong corrected-model values require independent reproduction and scrutiny for dataset/source confounding; they do not establish generalization.

## 6. Streamlit inference and Grad-CAM

The application loads `BrainTumorAI/models/tumor_classifier_reproducible.pth` by default. Set `NEUROVISION_CHECKPOINT` to select a different supported experiment checkpoint. The loader requires the research checkpoint format, explicit class mapping, and strict parameter matching. A missing, old-format, incompatible, or wrong-architecture checkpoint produces a visible error and training instruction; it never substitutes random classifier weights.

At inference, each image is converted to RGB and passed through the shared 224×224/ImageNet-normalization pipeline. The two logits are softmaxed along the class dimension and displayed in `no_tumor=0`, `tumor=1` order. These scores are uncalibrated and are not disease probabilities or certainty estimates. Sample browsers use only manifest rows assigned to the training partition; they do not browse the final test partition.

A prior runtime consistency check compared the evaluator's batch inference with the application's single-image inference on 12 final-test images: labels and scores matched exactly (maximum absolute score difference 0). The Streamlit health endpoint also returned `ok` with the trained checkpoint loaded. These checks predate the latest UI-copy changes; current app startup and smoke-test results are recorded in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Grad-CAM hooks are installed only for the heatmap call and removed in a `finally` block. The selected class is passed explicitly, including for either output class. Grad-CAM is a model-activation visualization, not a segmentation or proof of tumor location.

The interface labels all generated coordinates **Illustrative / Simulated Coordinates**. The heatmap-to-display mapping is deterministic and uses a zero placeholder for Z, but it has no voxel-to-world transform, physical image geometry, anatomical orientation, or MNI registration. Heatmap threshold area is not lesion volume. The 3D visualization, robotic trajectory and safety score, radiosurgery dose/DVH views, sample records, and task queue are synthetic demonstrations. Treatment-advice output has been removed and replaced by a neutral placeholder. These features are not validated plans, connected to hardware, or medical guidance. No patient MRI volume or segmentation mask is used to create spatial geometry.

Grad-CAM targets the final ResNet block and the selected classifier class. It is a qualitative model-activation visualization only: no expert-drawn masks or reference localizations are available, so no Dice, IoU, or localization metric is reported. Invalid inputs and zero-contrast maps now raise explicit errors instead of yielding a misleading empty overlay.

The test split has 590 JPEGs and 16 TIFFs; all 16 TIFFs are tumor-labeled. An exploratory JPEG-only calculation from the stored predictions gave 98.47% accuracy, 98.15% balanced accuracy, ROC-AUC 0.999644, average precision 0.999790, and confusion matrix `[[214, 7], [2, 367]]`. File format alone does not account for the unusually high scores. The subset check does not exclude other source/acquisition artifacts, patient overlap, or label leakage, and was not used to tune the model.

## 7. Commands

Run commands from the repository root after obtaining the authorized source data:

```powershell
python -m BrainTumorAI.preprocess --seed 42
python -m BrainTumorAI.train_classifier --architecture resnet18 --epochs 12 --batch-size 16 --patience 4
python -m BrainTumorAI.train_classifier --architecture small_cnn --epochs 12 --batch-size 16 --patience 4
python -m BrainTumorAI.evaluate_classifier `
  --checkpoint BrainTumorAI/models/tumor_classifier_reproducible.pth `
  --compare-checkpoint BrainTumorAI/models/small_cnn_best.pth `
  --legacy-checkpoint BrainTumorAI/models/tumor_classifier.pth
streamlit run BrainTumorAI/app.py
```

Omit `--legacy-checkpoint` if the old local checkpoint is not available. Training outputs can be tuned with `--seed`, `--epochs`, `--batch-size`, and `--patience`; record any non-default values and rerun only validation-driven model selection before the final evaluation.

Focused tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q tests/test_classifier_pipeline.py
```

The plugin setting avoids unrelated globally installed pytest plugins; it is not required by the model code. The current focused run completed with 19 passed tests and no failures, skips, or errors in 15.86 seconds when that auto-loading was disabled. A normal auto-loading attempt failed before test collection in an unrelated globally installed plugin dependency; no project tests ran in that invocation. Python compilation passed. The current Streamlit process returned HTTP 200 from both its health and root endpoints; AppTest rendered the current app with no app exceptions and the saved checkpoint available, and a synthetic RGB input produced finite logits through the shared inference path. AppTest emitted two bare-mode `ScriptRunContext` warnings. The earlier 12-image evaluator/application parity check predates the latest UI-copy changes and was not repeated. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands and exact outcomes.

## 8. Reporting limitations

The old checkpoint's collapsed predictions demonstrate that loading a tensor-compatible checkpoint alone is not evidence that it is a reliable classifier. Even a strong image-level result on the replacement split would remain preliminary: source/version/license and clinical label validity are unknown, patient identities are absent, near-duplicate screening is heuristic, and no independent external dataset has been tested. Do not claim clinical readiness, patient-independent performance, BraTS segmentation, calibrated probabilities, novelty, or treatment-planning validity from this prototype.
