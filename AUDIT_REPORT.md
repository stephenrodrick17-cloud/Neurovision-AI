# NeuroVision AI classifier audit

## Executive summary

The local legacy checkpoint is shape-compatible and loads strictly, but it classified all 920 images in the old local test folder as `tumor`. That yields 56.63% accuracy only because 521/920 test examples had the tumor label; sensitivity was 100%, specificity 0%, balanced accuracy 50%, and ROC-AUC 0.487. This is degenerate behavior, not useful discrimination. Its SHA-256 was `779d87161a8118df39a9518b8b02fc663ed95990ce1dc5450c0266fdca3f3a97`.

The exact optimization failure cannot be reconstructed: the legacy checkpoint contains only a state dictionary, and there are no corresponding training logs/configuration or validated source provenance. Confirmed evaluation-design, duplicate leakage, fallback-loading, and provenance problems have been addressed in the new pipeline. A high validation result alone is not a substitute for final or patient-independent testing.

## Confirmed findings and fixes

| Finding | Evidence | Fix / disposition |
|---|---|---|
| Degenerate legacy predictions | Strictly loaded local two-output checkpoint predicted every image as tumor on the old 920-image test folder; confusion matrix `[[0, 399], [0, 521]]`; ROC-AUC 0.487. | Train/evaluate fresh on a separate manifest; report one-class collapse explicitly and compare with a training-majority baseline. Root cause within the old optimization run remains unverified. |
| Test data used for model selection | Legacy `train_classifier.py` iterated over `test` every epoch, passed test accuracy to `ReduceLROnPlateau`, and saved the best test-accuracy state. | Replaced by separate train/validation/test partitions. Only validation balanced accuracy/loss control scheduling, early stopping, and checkpoint choice. |
| Exact duplicate leakage in legacy split | Audit found 151 exact decoded-pixel groups across old train/test folders (224 train images and 167 test images in shared groups). | New split groups exact matches and high-similarity near-duplicate components before assignment. |
| Near-duplicate risk | Source audit found 432 exact-pixel duplicate groups / 1,017 images; a pHash+SSIM screen produced 955 multi-image components / 2,782 images. No high-similarity cross-label edges were found. | Groups use pHash Hamming distance ≤4 plus grayscale SSIM ≥0.95. This is only an image-similarity screen; it cannot establish patient identity. |
| No patient-level identifiers or verified provenance | Raw folders and local metadata contain no patient/source-volume IDs or source citation, version, or license. | Report image-level grouped splits only; obtain provenance/license and patient-level data before patient-generalization claims or redistribution. |
| Labels inherited from folder names | Files are separated into `Brain Tumor` and `Healthy`; no independent label adjudication is available. No exact-duplicate cross-label groups were found. | Map those directories explicitly to `tumor=1` and `no_tumor=0`, retain source paths, and disclose the lack of label verification. |
| Class imbalance is modest, not a sufficient explanation by itself | Raw counts: tumor 2,513; no-tumor 2,087 (about 1.2:1). | Use unweighted cross-entropy and no sampler for this baseline; compare against a training-majority predictor. |
| No class-index mismatch found | Source preprocessing mapped Brain Tumor to tumor; alphabetical `ImageFolder` classes map no_tumor to index 0 and tumor to 1; inference reads index 1 as tumor. | Preserve a single explicit mapping in checkpoint metadata, training, evaluation, and inference. |
| App could silently use an untrained head | Legacy `app.py` caught checkpoint errors and fell back to an ImageNet feature extractor with a newly initialized classification head. This was not the path used for the historical strict-load evaluation, but it could yield misleading predictions when loading failed. | Require the reproducible checkpoint format and strict parameter matching; show an error and training instruction instead of fallback predictions. |
| Data-format variability | All 4,600 source files were readable; 4,496 JPEG, 18 PNG, 86 TIFF; 459 distinct dimensions and RGB, grayscale, RGBA, and palette modes. | Decode with Pillow, convert explicitly to RGB, and share the 224×224/ImageNet normalization transform. |
| Near-duplicate manifest guard was incomplete | Before the final guard was added, training/evaluation manifest validation checked exact pixel hashes across splits but did not independently reject a shared near-duplicate-group ID. The generated seed-42 manifest itself had no group overlap. | Require a non-empty `duplicate_group`, reject any such group crossing partitions, and test that malformed manifests fail. |

## Dataset and split facts

Source folders: `archive/Brain Tumor Data Set/Brain Tumor Data Set/{Brain Tumor,Healthy}`. Local files do not identify an authoritative dataset source/version/license; do not relabel this as BraTS. No unreadable images were found. The revised split manifest and complete machine-readable audit are written to ignored `BrainTumorAI/results/split_manifest.csv` and `BrainTumorAI/results/dataset_audit.json`. With seed 42, the current manifest assigns 3,191 images to training, 803 to validation, and 606 to final test. The test set is image-level and not patient-independent.

The previous 80/20 split also failed to clear old output directories before writing and did not stratify or group images. The revised preprocessing script does not mutate or duplicate the original data. The seed-42 manifest contains 3,191 training images (1,490 no-tumor / 1,701 tumor), 803 validation images (376 / 427), and 606 final-test images (221 / 385). The test partition has 396 duplicate/near-duplicate groups, including 121 groups with more than one image; all three partitions are disjoint by group ID and exact pixel hash. A small random visual spot-check showed image and acquisition heterogeneity; it cannot validate folder labels, and no clinical/domain-expert label adjudication was available.

## Reproduction and evaluation status

Commands are documented in [README.md](README.md) and [MODEL_WORKFLOW.md](MODEL_WORKFLOW.md). The ResNet-18 and small-CNN were trained for 12 epochs each using training and validation only; both selected epoch 11. The corrected ResNet-18's best validation balanced accuracy was 99.15%; the small CNN's was 84.17%. The final 606-image test evaluation used a fixed 0.5 threshold, 1,000 duplicate-group-stratified bootstrap replicates, and no test-based model or threshold selection (`test_set_used_for_selection=false` in the machine-readable report).

### Measured final-test comparison

| Model | Accuracy | Balanced accuracy | Tumor sensitivity | No-tumor specificity | Tumor precision | Tumor F1 | ROC-AUC | Average precision | Predicted no-tumor / tumor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected ImageNet ResNet-18 | 98.51% | 98.16% | 99.48% | 96.83% | 98.21% | 98.84% | 0.9996 | 0.9998 | 216 / 390 |
| Small CNN | 81.02% | 78.70% | 87.27% | 70.14% | 83.58% | 85.39% | 0.9045 | 0.9462 | 204 / 402 |
| Existing legacy ResNet-18 | 63.53% | 50.00% | 100.00% | 0.00% | 63.53% | 77.70% | 0.4952 | 0.6827 | 0 / 606 |
| Training-majority baseline | 63.53% | 50.00% | 100.00% | 0.00% | 63.53% | 77.70% | 0.5000 | 0.6353 | 0 / 606 |

For corrected ResNet-18, confusion matrix in `[no_tumor, tumor]` order was `[[214, 7], [2, 383]]`. Its duplicate-group-bootstrap 95% intervals were accuracy 96.28–100.00%, balanced accuracy 95.08–100.00%, tumor sensitivity 98.40–100.00%, specificity 90.95–100.00%, ROC-AUC 0.9987–1.0000, and average precision 0.9992–1.0000. On this same test split, the legacy checkpoint predicted all 606 images as tumor, exactly the training-majority prediction. The corrected model's unusually strong result should be treated as preliminary and independently reproduced; absence of patient IDs leaves patient overlap and source/acquisition confounding unresolved.

The raw reports include per-class precision, recall, and F1; all test scores and labels; image- and duplicate-group-bootstrap intervals; confusion matrices; ROC and precision-recall plots; checkpoint hashes; training histories; and metadata. These generated files are local/ignored under `BrainTumorAI/results/` and `BrainTumorAI/models/`, not committed or pushed with the source changes.

Prior classifier-pipeline validation, before this research-integrity follow-up, completed with 14 focused tests passing; evaluator and Streamlit inference labels/scores matched exactly on 12 held-out images (maximum absolute score difference 0); and the Streamlit health endpoint returned `ok`. Those checks predate the current UI and documentation changes. Follow-up validation is recorded below.

These results are suitable only for preliminary software/method development and clearly qualified image-level reporting. The unknown source/license, unverified folder labels, lack of patient identifiers, lack of external validation, and uncommitted working-tree changes must be resolved or disclosed before manuscript submission; no clinical or patient-independent claim is supported.

## Follow-up research-integrity audit

### High-score sensitivity and evaluation

The stored ResNet-18 prediction CSV was independently recalculated at the documented 0.5 threshold, with class order `[no_tumor=0, tumor=1]` and the continuous tumor score used for ROC-AUC and average precision. The recalculated confusion matrix and metrics match the recorded test report. The training code uses only train/validation partitions for scheduling, early stopping, and checkpoint selection; the final test set is not loaded by the training loop. Application and evaluator preprocessing share the inference transform. A prior consistency check on 12 held-out images found matching predictions and scores (maximum absolute score difference 0).

An exploratory audit-only JPEG subset contained 590 of the 606 test files. All 16 TIFF files were tumor-labeled. JPEG-only results were 98.47% accuracy, 98.15% balanced accuracy, ROC-AUC 0.999644, and average precision 0.999790, with confusion matrix `[[214, 7], [2, 367]]`. The similar performance indicates that TIFF-vs-JPEG encoding alone does not explain the high result; it does not rule out other source, acquisition, patient-overlap, or label-related confounding. This post hoc subset calculation was not used to change the checkpoint, threshold, or preprocessing.

### Provenance, independence, and annotation

The official dataset name, source/publication, release/version, license, and label-adjudication procedure remain **UNVERIFIED**. Local source folders and generic filenames contain no reliable patient, study, series, or volume identifiers. Therefore, the split and group-bootstrap intervals are image/duplicate-group level, not patient independent. No expert-annotated tumor masks or reference-localization labels are present; no Dice, IoU, or localization score is justified. The details and evidence status are tabulated in [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md).

### Explainability and spatial interface

The interface now identifies generated coordinates as **Illustrative / Simulated Coordinates**. The heatmap-peak mapping is deterministic, uses a zero placeholder for Z, and has no physical affine, orientation, or MNI registration. Grad-CAM is qualitative activation visualization, not segmentation or verified tumor localization; invalid and zero-contrast outputs fail explicitly. The 3D brain, activation-derived proxy geometry/volume, critical structures, trajectories, safety score, dose/DVH, robot/device status, and queue remain simulations without patient-specific inputs or hardware control.

PDF summaries and optional assistant wording were revised to avoid turning the binary class score into diagnosis, subtype, lesion size/location, or treatment advice. The prior treatment database was removed; its compatibility helper now returns only a neutral no-advice placeholder regardless of the supplied scenario. Sample records are labeled as synthetic/session-only, the robotics page is explicitly a simulation, and optional OpenRouter chat/image transmission is disclosed.

### Reproducibility and verification status

Experiment settings, environment, artifact hashes, split checksum, historical metrics, commands, and the uncommitted-source-snapshot caveat are recorded in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The experiment was not retrained; existing metrics remain the prior measured run, not new results. Regression coverage was expanded for simulated coordinates, non-clinical report behavior, and Grad-CAM error handling. The earlier focused checks reported 19 passing tests, successful compilation, and a clean whitespace check; current verification is recorded below. These software checks do not independently reproduce the experiment or validate medical performance.

### Current software verification (2026-10-10)

| Check | Result |
|---|---|
| Focused suite with plugin auto-load disabled | `19 passed in 15.86s`; zero failed, skipped, or errored tests. All focused tests ran. |
| Normal pytest plugin auto-load | Failed before collection when an unrelated globally installed plugin dependency rejected an inherited extra environment setting. No project tests ran in that invocation. |
| Python compilation | `& 'C:\Users\asus\python.exe' -m compileall -q BrainTumorAI tests` completed with exit code 0. |
| Whitespace validation | `git diff --check` passed after these documentation changes. |
| Streamlit startup and health | Launched the current `BrainTumorAI/app.py` process using Python 3.13.1; `/_stcore/health` returned HTTP 200 with `ok`, and `/` returned HTTP 200. Dotenv loading was disabled for the isolated test; local environment files were not opened. |
| Current app script and model load | Streamlit AppTest rendered the current About page with the saved ResNet-18 checkpoint and zero app exceptions. It emitted two bare-mode `ScriptRunContext` warnings. |
| Inference smoke check | The shared classifier inference path returned two finite logits for a synthetic 224x224 RGB input. This verifies software execution only, not scientific performance. |
| Saved artifact checksums | ResNet-18 checkpoint, small-CNN checkpoint, and split-manifest SHA-256 values match the checksums documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). |

The historical evaluator/application parity test on 12 held-out images remains evidence from before the latest UI-copy changes; it was not repeated during this audit. Current Streamlit startup, health, script rendering, and a synthetic inference smoke test were completed. The synthetic check did not submit an image through the browser upload widget and is not a held-out, patient-level, external, or clinical inference evaluation. No retraining, checkpoint or dataset edits, manifest regeneration, evaluation-output overwrite, commit, or push occurred during this audit.
