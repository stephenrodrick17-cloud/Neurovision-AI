# NeuroVision AI

NeuroVision AI is a Streamlit research and educational prototype for binary brain-image classification and interactive visualization. The application also contains simulated stereotactic planning, radiosurgery plots, and a surgical-task dashboard. These interface features are not medical-device functions and are not validated for patient care.

## Important limitations

- The trained classifier is binary (`no_tumor` vs `tumor`); it does not identify tumor subtypes.
- The legacy checkpoint predicted tumor for all 606 images in the corrected test split (63.53% accuracy, 50.00% balanced accuracy, 0.00% specificity, ROC-AUC 0.4952). Its original training failure remains unexplained because full logs and experiment provenance are unavailable. Its older 920-image result is not an unbiased test.
- The locally available source contains 4,600 readable images and no patient/volume identifiers. Its original source, version, license, and clinical label validity cannot be established from the files present. The new split is duplicate-grouped but still image-level, not patient-independent.
- The legacy prepared split had 151 exact-duplicate groups crossing train/test. The corrected workflow groups detected exact and screened near-duplicate images, but cannot rule out residual duplication or patient overlap. Source/acquisition confounding remains unresolved; the JPEG-only sensitivity analysis does not prove freedom from shortcut learning.
- No expert localization annotations or reference segmentation masks are available. Grad-CAM is qualitative only; no validated localization, Dice, or IoU claim is supported.
- Any displayed coordinates are labeled **Illustrative / Simulated Coordinates**: they are deterministic display-only image-plane values with a zero placeholder Z, not voxel-to-world coordinates or MNI coordinates.
- 3D coordinates, spatial displays, robotic trajectories, safety scores, radiation-dose, and DVH views are simulated. They are not anatomically registered, do not establish physical tumor location or volume, and are not clinical treatment-planning capabilities.
- Treatment text is static, patient records are held in session memory, and report fields may contain placeholders.
- Do not use the software to diagnose, triage, or select treatment. Do not send identifiable medical images to external services without appropriate authorization and privacy review.

## Features

The application is in `BrainTumorAI/app.py` and provides:

1. **Home:** project landing view and navigation.
2. **Dashboard:** choose a demonstration record, upload one image, adjust image appearance, view binary class scores, qualitative Grad-CAM overlay, synthetic 3D visualization, a notice that treatment advice is not provided, demonstration PDF summary, and optional AI chat/image description.
3. **Advanced Analytics:** inspect intermediate CNN activations and heuristic peak-activation coordinates.
4. **Robotics Simulation:** browse local dataset folders or upload an image, view illustrative geometry and placeholder device telemetry, and manage a session-only demonstration queue.
5. **Demonstration Records:** add/list synthetic records in Streamlit session state (not an EHR and not persistent storage).
6. **About System:** architecture summary and disclaimer.

### Model workflow and provenance

For the full end-to-end explanation, see [MODEL_WORKFLOW.md](MODEL_WORKFLOW.md). For confirmed root causes and audit evidence, see [AUDIT_REPORT.md](AUDIT_REPORT.md).
Dataset evidence and unresolved source/license questions are recorded in [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md); exact run settings and hashes are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The frozen checkpoint has been evaluated on the overlap-screened Cheng Figshare v8 cohort: tumor-positive slice sensitivity was 72.41% (1,223/1,689; 95% patient-cluster bootstrap CI 66.90–77.99%) at a fixed threshold of 0.5. The screen excluded 81 candidate-matched patients (1,375 slices), so the retained subtype mix differs substantially from the full source cohort; Cheng has no no-tumor cases and image-similarity screening cannot prove patient independence. On the provided BraTS 2023 GLI archive, 2D slice accuracy was 60.19%, balanced accuracy 59.92%, sensitivity 52.60%, specificity 67.24%, and ROC-AUC 0.6623 at the same threshold. BraTS negatives are tumor-free slices from tumor-bearing patients, not healthy controls. BraTS patient-level and clinical performance are not established. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the complete report and [EXTERNAL_VALIDATION.md](EXTERNAL_VALIDATION.md) for methods and reproducible commands.

Images are converted to RGB, resized to 224 × 224, converted to tensors, and normalized with ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. A ResNet-18 produces two logits; softmax scores are displayed. The maximum score is not calibrated and must not be interpreted as diagnostic certainty.

The reproducibly trained ResNet-18 is saved at `BrainTumorAI/models/tumor_classifier_reproducible.pth`. Checkpoints, datasets, and generated results are ignored by Git, so a fresh clone will not contain local data or a trained checkpoint. The app loads a checkpoint strictly and stops with a clear training instruction if it is missing or incompatible; it never produces predictions from random weights. Displayed softmax scores are uncalibrated model outputs, not probabilities of disease.

### Simulation modules

| Module | What it implements |
|---|---|
| `BrainTumorAI/heatmap_gen.py` | Grad-CAM activation visualization. No lesion-annotation validation is included. |
| `BrainTumorAI/surgery_viz.py` | Plotly views based on image pixels and synthetic depth. |
| `BrainTumorAI/stereotactic_robot.py` | Heuristic centroid, entry point, trajectory, safety score, and inverse-kinematic calculations using assumed geometry. |
| `BrainTumorAI/radiosurgery.py` | Synthetic beam, isodose, and DVH calculations; not a treatment-planning system. |
| `BrainTumorAI/surgical_task_queue.py` | In-memory demo task lifecycle and audit-like event entries; not connected to operating-room systems. |
| `BrainTumorAI/medication_report.py` | Non-advisory placeholder text and non-clinical PDF summary. |
| `BrainTumorAI/gemini_assistant.py` | Optional remote OpenRouter text/image description and browser speech synthesis. |

## Data and training

`BrainTumorAI/preprocess.py` expects this source layout:

```text
archive/Brain Tumor Data Set/Brain Tumor Data Set/
├── Brain Tumor/
└── Healthy/
```

The audit found 2,513 images in `Brain Tumor` (mapped to `tumor`) and 2,087 in `Healthy` (mapped to `no_tumor`). All 4,600 files were readable, with 459 distinct width/height combinations and JPEG, PNG, and TIFF encodings. It identified 432 exact decoded-pixel duplicate groups containing 1,017 images; a conservative pHash/SSIM screen identified additional similar-image groups. No exact-duplicate or high-similarity group crossed source labels. Source, release/version, license, and label-review procedure remain **UNVERIFIED**; see [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md).

The previous generated train/test folders had 151 exact-pixel duplicate groups crossing their split boundary (224 train files and 167 test files in those shared groups). The previous training script also used the nominal test set every epoch for learning-rate scheduling and best-checkpoint selection. The new `python -m BrainTumorAI.preprocess` command writes a CSV manifest and JSON audit to `BrainTumorAI/results/`; the final test split is reserved for evaluation. Near-duplicate groups are formed only when the pHash candidate also has grayscale SSIM ≥0.95; this is a duplicate-leakage screen, not patient identity verification.

The source files do not include a verifiable dataset citation, release/version, license, or patient IDs. The labels are inherited from the two folder names and were not independently adjudicated. Do not describe this as BraTS or a patient-independent dataset.

The corrected pipeline uses an explicit class mapping (`no_tumor=0`, `tumor=1`), a seed-42 stratified group split, exact and high-similarity image groups, and consistent RGB/224×224/ImageNet normalization. The fixed manifest contains 3,191 training images (1,490 no-tumor / 1,701 tumor), 803 validation images (376 / 427), and 606 final-test images (221 / 385). ResNet-18 starts from ImageNet V1 weights; training-only augmentation is horizontal flipping and rotation up to 10 degrees. It uses Adam (`1e-4`, weight decay `1e-4`), unweighted cross-entropy (class imbalance is modest), and selects checkpoints/early stopping by validation balanced accuracy. Training records loss and metrics each epoch and never opens the final test split.

The exact commands for inspecting the preserved seed-42 results, re-evaluating existing checkpoints without retraining, and running a fresh experiment into isolated output paths are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The original provenance and license remain **UNVERIFIED**. Evaluation uses a fixed 0.5 threshold and reports continuous-score ROC-AUC and average precision, per-class results, prediction counts, confusion matrix, duplicate-group-stratified and image-stratified bootstrap confidence intervals, and a training-majority baseline. Do not use final-test outputs to tune the model or threshold.

### Measured local experiment (seed 42)

The frozen checkpoints were evaluated once on the manifest's 606-image final test partition at the prespecified 0.5 threshold. Neither checkpoint nor threshold was selected using test data. The bootstrap intervals below are 95% percentile intervals from 1,000 duplicate/near-duplicate-group-stratified resamples; these groups are not patient clusters.

| Model | Accuracy | Balanced accuracy | Tumor sensitivity | No-tumor specificity | Tumor precision | Tumor F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected ImageNet ResNet-18 | 98.51% | 98.16% | 99.48% | 96.83% | 98.21% | 98.84% | 0.9996 | 0.9998 |
| Small CNN baseline | 81.02% | 78.70% | 87.27% | 70.14% | 83.58% | 85.39% | 0.9045 | 0.9462 |
| Existing legacy ResNet-18 | 63.53% | 50.00% | 100.00% | 0.00% | 63.53% | 77.70% | 0.4952 | 0.6827 |
| Training-majority baseline | 63.53% | 50.00% | 100.00% | 0.00% | 63.53% | 77.70% | 0.5000 | 0.6353 |

For the corrected ResNet-18, the confusion matrix in `[no_tumor, tumor]` order was `[[214, 7], [2, 383]]`; it predicted 216 images as no-tumor and 390 as tumor. Duplicate-group bootstrap 95% intervals were: accuracy 96.28–100.00%, balanced accuracy 95.08–100.00%, tumor sensitivity 98.40–100.00%, no-tumor specificity 90.95–100.00%, ROC-AUC 0.9987–1.0000, and average precision 0.9992–1.0000. The legacy checkpoint predicted all 606 images as tumor on this same split, so its accuracy equals the test-set tumor prevalence and is not evidence of discrimination.

This is an unusually strong **preliminary image-level result against folder-derived labels**, not a validated medical-imaging result. The unknown data source/version/license, lack of patient IDs and label adjudication, possible non-duplicate source/acquisition confounding, and absence of external validation mean the score must not be presented as patient-independent or clinical performance. A small random visual spot-check cannot validate labels. The old-checkpoint result on the earlier 920-image folder (56.63% accuracy, 0% specificity, 50% balanced accuracy, ROC-AUC 0.487) is a separate historical result and is not directly comparable because the split differed and was used for checkpoint selection.

The split is image-level because patient identifiers are absent. Patient-level/external validation, original dataset provenance and license, and clinical label validity remain prerequisites for any clinical or patient-generalization claim.

A test-only JPEG subset sensitivity check (590/606 images; all 16 TIFFs were tumor-labeled) retained similar metrics: 98.47% accuracy, 98.15% balanced accuracy, ROC-AUC 0.999644, AP 0.999790, confusion matrix `[[214, 7], [2, 367]]`. This shows file format alone did not explain the high score, but does not rule out other source/acquisition confounding. The subset analysis is exploratory, performed after the stored experiment, and was not used to modify the model.

Run focused automated checks with:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q tests/test_classifier_pipeline.py
```

Current software verification (2026-10-10): the focused suite completed with 19 passed, no failures/skips/errors, in 15.86 seconds when unrelated global pytest plugin auto-loading was disabled. A normal pytest invocation failed before test collection because a globally installed plugin dependency rejected an inherited extra environment setting; no project tests ran in that invocation. `compileall` succeeded for `BrainTumorAI` and `tests`. The current Streamlit process started; both `/_stcore/health` and `/` returned HTTP 200. Streamlit AppTest rendered the current app with the saved ResNet-18 checkpoint and no app exceptions (two bare-mode `ScriptRunContext` warnings were emitted). A synthetic 224x224 RGB input produced two finite logits through the shared inference path. This is a software smoke test, not a held-out or clinical inference evaluation. The earlier exact evaluator/application comparison on 12 held-out images predates the latest UI-copy changes and was not repeated. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands and details. The project requires PyTorch 2.6 or later for safe, weights-only checkpoint loading.

## Installation and launch

Use Python 3.11 (the deployment files specify Python 3.11). From the repository root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run:

```bash
streamlit run BrainTumorAI/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). Training ResNet-18 may download pretrained ImageNet weights if they are not cached. Train a reproducible checkpoint before launching inference; an absent or incompatible model is reported as an error.

## Optional Gemini/OpenRouter integration

The AI assistant uses `google/gemini-2.0-flash-001` via OpenRouter. Set a key in a root `.env` file or the process environment:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env` or embed a credential in source code. Image-description requests send image data to a remote provider. Use only data that you are authorized to transmit, and follow applicable consent, institutional, and privacy requirements.

## Deployment and repository map

- `Dockerfile`, `Procfile`, and `railpack.json` configure Streamlit deployment.
- `requirements.txt` lists Python dependencies; versions are not fully locked.
- `runtime.txt` declares Python 3.11.7.
- `BrainTumorAI/assets/` contains decorative UI media.
- `archive/`, `BrainTumorAI/dataset/`, `BrainTumorAI/models/*.pth`, generated reports, and `.env` are local/ignored artifacts and are not part of a normal Git clone.

## Research reporting checklist

Before writing scientific claims, verify and record:

- Dataset source, citation, version, license, cohort, labels, and inclusion/exclusion criteria.
- Patient-level split, duplicate checks, counts of patients and images per partition.
- Exact checkpoint hash and training run, code revision, software versions, hardware, and random seeds.
- Confusion matrix, sensitivity, specificity, precision, F1, accuracy, confidence intervals, and ROC-AUC from the held-out image-level test split; patient-level testing is currently impossible.
- Calibration and external validation if making probability or generalization claims.
- Ethics approval/waiver, consent, data governance, funding, conflicts, and author contributions.

Do not describe the prototype as providing tumor-subtype classification, MNI registration, volumetric segmentation, clinically validated dose planning, safe surgical planning, or clinical decision support based on the current implementation.

## Disclaimer

This project is for research and educational exploration only. It is not a medical device and has not been established as safe or effective for clinical use. All predictions, visualizations, generated text, and simulated plans may be incorrect. Qualified professionals must independently review any medical information; no patient-care decision should be based on this software.


## Audit study: shortcut learning, leakage and external validity (`paper_evidence/`)

Beyond the application, this repository contains the full evidence base of an audit of the binary tumour classifier.
Everything below is regenerated by scripts in [`paper_evidence/`](paper_evidence/README.md); every number quoted in
this section is read from a saved file in that folder. Items we could not verify are labelled **NOT VERIFIED**.
Dataset sources and licences are in [DATASETS_AND_LICENSES.md](DATASETS_AND_LICENSES.md).

### Question

A ResNet-18 trained on a public 4,600-image Kaggle collection reaches near-ceiling internal test performance. We ask
whether that performance reflects tumour recognition or dataset artefacts, and how it behaves on independent cohorts.

### Study design

1. **Discovery cohort:** Kaggle collection, 4,600 images, duplicate-grouped image-level split
   (3,191 train / 803 validation / 606 test). No patient IDs exist, so the split is not patient-independent.
2. **Models:** ResNet-18 (ImageNet init), SmallCNN, DenseNet-121, five seeds each (42-46).
3. **Overlap audit:** perceptual hash (Hamming distance <= 4) plus 64x64 SSIM >= 0.95 against Cheng (3,064 slices,
   233 patients) and BraTS 2023, with positive and negative controls. Result against Cheng: 818 Cheng slices
   matched 846 discovery images (543 train / 162 validation / 141 test) from 81 Cheng patients; 55 patients appear in
   both train and test.
4. **Frozen external tests:** Cheng overlap-screened (1,689 slices, 152 patients; tumour sensitivity only) and
   BraTS 2023 T2-FLAIR (169,066 slices, 1,251 patients; 81,413 tumour / 87,653 tumour-free) after an orientation
   correction (`rot90`).
5. **Leakage-free reference models:** patient-split BraTS (850 train / 151 validation / 250 test patients, split seed 42,
   test = 33,696 slices), a patient-decontaminated discovery model, and a Cheng subtype model with patient- versus
   slice-level splits.
6. **Localisation:** Grad-CAM scored against BraTS tumour masks (pointing game, energy-based pointing game, Dice).
7. **Shortcut checks:** header-only (non-pixel) metadata baseline; sequence-matched T1-CE test.

### Key results

| Setting | Result | Source file |
|---|---|---|
| Internal test, ResNet-18 (n = 606) | accuracy 0.9851, ROC-AUC 0.9996 | `phase3_clean_test_rescore.json` |
| Internal clean test (n = 465, Cheng-matched images removed) | accuracy 0.9806, ROC-AUC 0.9994 | `phase3_clean_test_rescore.json` |
| Frozen model, BraTS oriented (169,066 slices) | sensitivity 0.4031, specificity 0.9871, ROC-AUC 0.7067 | `brats_oriented_detailed_report.json` |
| Frozen model, BraTS unrotated | ROC-AUC 0.6623 | `brats_gpu_comparison_report.json` |
| Frozen model, Cheng overlap-screened | sensitivity 0.7241 (patient-cluster 95% CI 0.6690-0.7799) | `BrainTumorAI/results/external_validation/` (local) / `verified_leakage_claims.json` |
| Frozen model, BraTS 250-patient subset | ROC-AUC 0.7088 (seed 42) | `brats_250_5seeds_with_cis.json` |
| Discovery-trained models across seeds 42-46, BraTS-250 ROC-AUC (internal accuracy is near-identical within each architecture) | ResNet-18: 0.7088, 0.6587, 0.5050, 0.5705, 0.6651; SmallCNN: 0.3914, 0.3939, 0.4418, 0.5150, 0.5005; DenseNet-121: 0.7255, 0.6963, 0.7051, 0.5837, 0.6481 | `brats_250_5seeds_with_cis.json`, Fig. 5 |
| Patient-split BraTS ResNet-18 (ImageNet init, 5 seeds) | ROC-AUC 0.9738 +/- 0.0011 | `leakage_free_multiseed.json` |
| Patient-split BraTS ResNet-18 (fine-tuned, 5 seeds) | ROC-AUC 0.9709 +/- 0.0014 | `leakage_free_multiseed.json` |
| Decontaminated discovery ResNet-18, 5 seeds (n = 606) | accuracy 0.9733 +/- 0.0067, ROC-AUC 0.9905 +/- 0.0037 | `decontaminated_discovery_5seeds.json` |
| Same models on Cheng overlap-screened | sensitivity per seed 0.8348, 0.8769, 0.5465, 0.8686, 0.5317 | `decontaminated_discovery_5seeds.json` |
| Same models on BraTS-250 | ROC-AUC per seed 0.6636, 0.6151, 0.5628, 0.6324, 0.7461 | `decontaminated_discovery_5seeds.json` |
| Frozen model on BraTS-250 **T1-CE** | ROC-AUC 0.5627, sensitivity 0.2270 (vs 0.7088 for T2-FLAIR) | `healthy_control_and_t1c_check.json` |
| Header-only metadata baseline (gradient-boosted trees), test n = 606 | accuracy 0.9406, ROC-AUC 0.9814 | `header_only_baseline.json` |
| Same, Cheng-matched images removed (n = 465) | accuracy 0.9226, ROC-AUC 0.9719 | `header_only_baseline.json` |
| Header features: image dimensions only | ROC-AUC 0.9744 (test 606) | `header_only_baseline.json` |
| Header features: JPEG quantisation tables only | ROC-AUC 0.6537 (test 606) | `header_only_baseline.json` |

### Interpretation

- Near-ceiling internal accuracy does not transfer: the frozen model detects 40% of tumour slices on BraTS with very
  high specificity, and sensitivity collapses for small tumours (Fig. 4c).
- Metadata that contains no anatomy, mainly **image dimensions and file size**, predicts the label with ROC-AUC 0.97-0.98.
  The collection therefore carries a strong acquisition/format shortcut, and internal performance is not evidence of
  tumour recognition.
- Training on patient-separated BraTS data recovers high within-cohort performance (ROC-AUC about 0.97), so the
  architecture is capable; the discovery data are the limiting factor.
- External performance of the discovery-trained models varies widely by seed even when internal accuracy is
  identical, so single-seed external results should not be reported.
- Matching Cheng's T1-CE sequence does not explain the BraTS drop (T1-CE is worse, ROC-AUC 0.5627).
- Grad-CAM localisation improves with patient-split training but remains weak for small tumours (Fig. 6).
- **Not established:** performance on healthy controls from a second cohort (no healthy-volunteer dataset is on disk;
  **NOT VERIFIED**), clinical validity of the original labels, and licence of the discovery collection.

### Figures

| Figure | File |
|---|---|
| 1 Audit workflow | [`fig1_workflow.png`](paper_evidence/figures/fig1_workflow.png) |
| 2 Orientation montage | [`fig2_orientation_montage.png`](paper_evidence/figures/fig2_orientation_montage.png) |
| 3 Overlap examples and counts | [`fig3_overlap_audit.png`](paper_evidence/figures/fig3_overlap_audit.png) |
| 4 BraTS score histogram, ROC, sensitivity by size | [`fig4_brats_cohort.png`](paper_evidence/figures/fig4_brats_cohort.png) |
| 5 Internal accuracy vs external ROC-AUC by seed | [`fig5_seeds_internal_vs_external.png`](paper_evidence/figures/fig5_seeds_internal_vs_external.png) |
| 6 Grad-CAM localisation | [`fig6_gradcam_localization.png`](paper_evidence/figures/fig6_gradcam_localization.png) |
| 7 ROC-AUC across settings | [`fig7_roc_auc_summary.png`](paper_evidence/figures/fig7_roc_auc_summary.png) |

Each figure has its generating script next to it (300 dpi, colour-blind-safe palette, numbers read from saved files).

### Reproducing

Datasets must be obtained from their providers first (see [DATASETS_AND_LICENSES.md](DATASETS_AND_LICENSES.md)); paths
in the scripts point to `D:/datasets/...` and the repository root and must be adapted. Run order and per-script
descriptions are in [`paper_evidence/README.md`](paper_evidence/README.md). Model checkpoints are not committed; the
SHA-256 inventory in `paper_evidence/release_audit_v7.json` lets you verify retrained or re-downloaded files.
Training uses fixed seeds and deterministic flags where feasible; some CUDA operations remain non-deterministic
(PyTorch warns about CuBLAS), and CPU versus GPU inference differs by up to about 1e-3 in score
(`discovery_606_reproducibility_log.json`), which flipped one SmallCNN prediction at the 0.5 threshold.

### Release status and limits

- Git revision audited: see `paper_evidence/release_audit_v7.json` (no release tag exists yet).
- Code licence: none declared yet. Dataset licences: see [DATASETS_AND_LICENSES.md](DATASETS_AND_LICENSES.md).
- This software is a research prototype and is not a medical device.
