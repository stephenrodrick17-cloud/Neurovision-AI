# NeuroVision AI: Reproducible Image-Level Evaluation of a Brain MRI Classifier Prototype

> **Draft and scope.** This draft describes a local software experiment, not a clinical validation study. The dataset source, version, license, patient identities, and clinical validity of folder-derived labels have not been verified. The reported metrics describe prediction of those labels on an image-level split only.

## Abstract

### Background

Brain-image classifiers require reproducible data partitions, strict checkpoint handling, and evaluation that is separated from model selection. A working user interface or high image-level score alone does not establish patient-level or clinical validity.

### Objective

To audit and characterize the NeuroVision AI binary brain-image classifier, address defects in its legacy training/evaluation workflow, and measure the corrected baseline on a fixed image-level test partition.

### Methods

The available local dataset contains 4,600 readable images in `Brain Tumor` and `Healthy` folders, mapped to `tumor=1` and `no_tumor=0`. Its source citation, release, license, patient identifiers, and independent label validation could not be established. A deterministic seed-42 split grouped exact pixel duplicates and candidate near-duplicates identified using perceptual-hash distance and grayscale structural similarity. The resulting partitions contained 3,191 training, 803 validation, and 606 final-test images. An ImageNet-pretrained ResNet-18 was trained using training data only; validation balanced accuracy selected checkpoints and controlled early stopping. A small CNN, the legacy ResNet-18 checkpoint, and a training-majority baseline were compared on the same final test partition. Evaluation used a fixed 0.5 threshold and continuous tumor-class scores for ROC-AUC and average precision. Confidence intervals were estimated by duplicate-group-stratified bootstrap.

### Results

On the 606-image final test partition, the corrected ResNet-18 achieved 98.51% accuracy, 98.16% balanced accuracy, 99.48% tumor sensitivity, 96.83% no-tumor specificity, and ROC-AUC 0.9996. Its confusion matrix in `[no_tumor, tumor]` order was `[[214, 7], [2, 383]]`; 216 images were predicted as no-tumor and 390 as tumor. The duplicate-group-bootstrap 95% interval was 96.28–100.00% for accuracy and 0.9987–1.0000 for ROC-AUC. The small CNN achieved 81.02% accuracy and ROC-AUC 0.9045. The legacy checkpoint predicted all 606 test images as tumor, with 63.53% accuracy, 50.00% balanced accuracy, 0% specificity, and ROC-AUC 0.4952. The training-majority baseline had the same accuracy and balanced accuracy as that degenerate checkpoint.

### Conclusions

The corrected pipeline produced a substantially stronger result against the local folder-derived labels than the legacy checkpoint on the same image-level test partition. This is preliminary software/method evidence only. The lack of patient identifiers, verified source and license, independently adjudicated labels, and external validation prevents patient-generalization or clinical claims. Independent data and domain-expert review are needed before manuscript submission as a diagnostic-performance study.

**Keywords:** brain MRI; binary image classification; ResNet-18; reproducibility; image-level evaluation; research prototype

## 1. Introduction

Automated analysis of brain images is an active research area, but performance estimates depend on dataset provenance, label validity, independence of evaluation cases, and the match between the experiment and the intended use. Image-level splits can overstate generalization when related images from the same patient or source volume occur across partitions. A model's tensor-compatible checkpoint also does not establish that it was trained appropriately or that its predictions are informative.

NeuroVision AI is a Streamlit research and educational prototype with a binary image classifier and supplementary activation visualizations. Its verified classifier task is limited to classifying a single two-dimensional image as `tumor` or `no_tumor`. It does not classify tumor subtypes, segment lesions, analyze volumes, or provide validated anatomical localization. Simulated stereotactic, robotic, radiosurgery, and treatment-report features are demonstrations and are not clinical tools.

This work audits the legacy pipeline, implements a train/validation/final-test workflow with strict model loading, and reports a measured comparison on the available local data. The specific cause of the legacy checkpoint's all-tumor output could not be reconstructed because its training logs and full provenance are unavailable.

## 2. Materials and methods

### 2.1 Data source and labels

The local source directory is `archive/Brain Tumor Data Set/Brain Tumor Data Set/`, with `Brain Tumor` and `Healthy` subdirectories. Files were mapped by folder name to `tumor=1` and `no_tumor=0`, respectively. The available files do not identify an authoritative source, citation, dataset release/version, license, patient or volume IDs, or an independent label-adjudication process. The data were not identified as BraTS and are not described as such here.

The audit counted 2,513 tumor-folder images and 2,087 healthy-folder images. All 4,600 files were readable; 4,496 were JPEG, 18 PNG, and 86 TIFF, with 459 distinct image dimensions. Exact decoded-pixel hashing found 432 duplicate groups containing 1,017 images. A conservative near-duplicate screen grouped images only when perceptual-hash Hamming distance was at most 4 and 64×64 grayscale structural similarity was at least 0.95. This process yielded 955 multi-image connected components containing 2,782 images and found no high-similarity cross-label edges. Such groups are image-similarity units, not patient identities. A small visual spot-check was not a clinical label review.

### 2.2 Partitions and leakage checks

The fixed manifest was generated with seed 42 using stratified group splitting, keeping exact and screened near-duplicate components within one partition. It assigned 3,191 images to training (1,490 no-tumor; 1,701 tumor), 803 to validation (376; 427), and 606 to final testing (221; 385). The final test partition contains 396 duplicate/near-duplicate groups, including 121 multi-image groups. No group ID or exact decoded-pixel hash crossed partitions.

The legacy prepared train/test folders had 151 exact duplicate groups crossing the split boundary, representing 224 train-folder and 167 test-folder images in shared groups. The legacy training script also evaluated the nominal test folder each epoch for scheduler updates and checkpoint selection. Its reported old-folder result is therefore not an unbiased final-test estimate. The replacement training script reads only train and validation rows and asserts that exact and near-duplicate groups do not cross partitions. No patient-independent split was possible.

### 2.3 Preprocessing

Images are decoded with Pillow, converted to RGB, resized to 224×224, converted to tensors, and normalized using ImageNet means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. Training-only augmentation consists of random horizontal flips and rotations up to 10 degrees. Validation, final-test, and application inference use the same non-augmenting transform. The test threshold, preprocessing settings, and checkpoint were not selected using final-test labels.

### 2.4 Models and training

The primary baseline is Torchvision ResNet-18 initialized with ImageNet V1 weights, with its final layer replaced by two logits. An additional small CNN was trained on the same manifest. The legacy local ResNet-18 state dictionary was loaded strictly for comparison; its training configuration and data provenance remain unverified. A constant training-majority predictor was included as a non-learning baseline.

For both newly trained models, the training seed was 42, the loss was unweighted cross-entropy, and Adam used learning rate `1e-4` and weight decay `1e-4` with batch size 16 and a maximum of 12 epochs. The class ratio was modest enough that no class weighting or balanced sampler was used. `ReduceLROnPlateau`, early stopping (patience 4), and checkpoint selection used validation balanced accuracy only; validation loss broke ties. Both runs completed 12 epochs and selected epoch 11. Best validation balanced accuracy was 99.15% for ResNet-18 and 84.17% for the small CNN. Training summaries mark the final test split as not accessed.

Checkpoints store the architecture, strict class mapping, model state, seed, preprocessing/training configuration, software environment, manifest hash, selected epoch, and validation result. The ResNet-18 checkpoint SHA-256 is `0ce4741c3ce2cbc287c7b716395b1571bac143200e2f3489e48efd002310b40e`; the small-CNN checkpoint SHA-256 is `f08d9e8bf8c3c01f06281909cd61052bcd465fee860c5c89d2941f44097da05c`. The split-manifest SHA-256 is `c0d909039b77d1d2ccb3fb27105e411eafee792b868b8203794ac5bf979d66ab`.

### 2.5 Evaluation

Each frozen candidate was evaluated on the same 606-image final-test partition. The decision threshold was fixed at 0.5. Metrics included accuracy, balanced accuracy, tumor sensitivity/recall, no-tumor specificity, tumor precision/F1, per-class precision/recall/F1, confusion matrix, predicted-class counts, ROC-AUC, and average precision. ROC-AUC and average precision used continuous tumor-class softmax scores. One-class prediction collapse was explicitly flagged.

The primary 95% percentile confidence intervals used 1,000 stratified bootstrap replicates that sampled complete duplicate/near-duplicate groups within each class. A secondary image-stratified bootstrap was also recorded. Neither interval accounts for patient-level clustering because patient/volume identifiers are unavailable. No test-driven threshold selection, checkpoint selection, or hyperparameter tuning was performed.

The experiment used Python 3.13.1, PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118, scikit-learn 1.6.1, NumPy 2.2.6, Pillow 11.1.0, OpenCV 4.10.0, and an NVIDIA GeForce RTX 3050 Laptop GPU on Windows. The recorded Git base revision was `0f4d663a327e8ab638b78e6425096a9fd84c9165`; the experiment was run with local working-tree changes, so that commit alone does not identify a complete immutable source snapshot.

### 2.6 Application and visualization

The Streamlit application loads the reproducible ResNet-18 checkpoint with strict key matching and the same inference preprocessing. Missing or incompatible checkpoints produce an explicit error instead of random-head predictions. Displayed softmax values are uncalibrated model scores. Grad-CAM targets the final ResNet block and selected output class; it is not a validated lesion segmentation or localization method. A prior startup/health check succeeded, and a consistency check on 12 final-test images found identical evaluator and single-image application predictions and scores (maximum absolute score difference 0). These checks predate the latest UI-copy changes; current runtime verification is reported in Section 3.3. Coordinates, 3D views, surgical trajectories, safety scores, and radiosurgery views are synthetic demonstrations; no treatment advice is provided.

The interface's **Illustrative / Simulated Coordinates** are deterministic display-only image-plane values with a placeholder zero Z coordinate. They are not derived from image-affine metadata, masks, or an anatomical registration, and are not MNI coordinates. Heatmap-derived geometry and estimated-volume proxies are not lesion measurements. No annotated masks or expert localization references were available; Grad-CAM remains qualitative, with no localization scores reported. Invalid or zero-contrast Grad-CAM outputs now fail explicitly. Robotic, radiosurgery, and patient-record features are labeled as demonstrations, not clinical functions. Treatment advice was removed and replaced with a neutral placeholder. Optional OpenRouter chat/image-description features disclose remote processing and must not receive identifiable or restricted data without authorization.

### 2.7 Exploratory image-format sensitivity check

As an audit-only sensitivity analysis, stored final-test predictions were restricted to JPEG files. This subset contained 590 images; all 16 TIFF images in the full test partition were tumor-labeled. The JPEG-only confusion matrix was `[[214, 7], [2, 367]]`, with 98.47% accuracy, 98.15% balanced accuracy, ROC-AUC 0.999644, and average precision 0.999790. This post hoc calculation was not used to select or modify a model, threshold, or preprocessing. It suggests TIFF-vs-JPEG format alone does not explain the high performance, but cannot exclude other source/acquisition confounding or patient overlap.

## 3. Results

### 3.1 Final-test model comparison

| Model | Accuracy | Balanced accuracy | Tumor sensitivity | No-tumor specificity | Tumor precision | Tumor F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected ImageNet ResNet-18 | 98.51% | 98.16% | 99.48% | 96.83% | 98.21% | 98.84% | 0.9996 | 0.9998 |
| Small CNN | 81.02% | 78.70% | 87.27% | 70.14% | 83.58% | 85.39% | 0.9045 | 0.9462 |
| Existing legacy ResNet-18 | 63.53% | 50.00% | 100.00% | 0.00% | 63.53% | 77.70% | 0.4952 | 0.6827 |
| Training-majority baseline | 63.53% | 50.00% | 100.00% | 0.00% | 63.53% | 77.70% | 0.5000 | 0.6353 |

For corrected ResNet-18, the class-specific results were:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| No-tumor | 221 | 99.07% | 96.83% | 97.94% |
| Tumor | 385 | 98.21% | 99.48% | 98.84% |

The corrected ResNet-18 confusion matrix, with true classes as rows and predictions as columns in `[no_tumor, tumor]` order, was `[[214, 7], [2, 383]]`. It classified 216 images as no-tumor and 390 as tumor. Duplicate-group-bootstrap 95% intervals were 96.28–100.00% for accuracy, 95.08–100.00% for balanced accuracy, 98.40–100.00% for tumor sensitivity, 90.95–100.00% for no-tumor specificity, 0.9987–1.0000 for ROC-AUC, and 0.9992–1.0000 for average precision.

The legacy checkpoint classified all 606 images as tumor, the same class as the training-majority baseline. Its accuracy therefore reflected the test-set tumor prevalence rather than useful class discrimination. On a different historical 920-image test folder, the legacy checkpoint also predicted every image as tumor (56.63% accuracy, 0% specificity, 50.00% balanced accuracy, ROC-AUC 0.487); that folder was used for checkpoint selection and had duplicate leakage, so those values are not directly comparable to the fixed final-test metrics above.

### 3.2 Reproducibility outputs

The local run generated the manifest and data audit, train/validation histories, learning curves, checkpoint metadata and hashes, test predictions, per-model metrics, confidence intervals, confusion-matrix plots, and ROC/precision-recall plots under `BrainTumorAI/results/` and `BrainTumorAI/models/`. Those data and checkpoint artifacts are ignored by Git and are not included in a fresh clone.

### 3.3 Current software verification

On 2026-10-10, the focused suite completed with 19 passed tests and no failures, skips, or errors when unrelated global pytest plugin auto-loading was disabled. A normal auto-loading attempt failed before collection in an unrelated globally installed plugin dependency; no project tests ran in that invocation. Python compilation completed successfully, and `git diff --check` passed after the documentation edits.

The current Streamlit process started successfully: its health endpoint returned HTTP 200 with `ok`, and its root route returned HTTP 200. Streamlit AppTest rendered the current application with the saved ResNet-18 checkpoint and no app exceptions; it emitted two bare-mode `ScriptRunContext` warnings. A synthetic 224x224 RGB input produced two finite logits through the shared inference path. This is a software smoke check, not a result on held-out data. The 12-image evaluator/application parity check above was not repeated after the interface changes. These checks do not establish clinical validity, patient independence, or external generalization.

## 4. Discussion

The audit found two concrete defects that undermined the legacy evaluation: test data influenced epoch-wise scheduling and checkpoint selection, and exact duplicate images crossed its train/test boundary. The old app could also silently fall back to an untrained classifier head if checkpoint loading failed. No class-index mismatch was found, and the legacy checkpoint did load strictly; therefore, the specific training cause of its all-tumor predictions remains unknown.

The corrected ResNet-18 produced stronger image-level discrimination than both the small CNN and the legacy checkpoint on the same fixed test split. This comparison supports the technical value of separating train/validation/test data, explicit mappings, strict loading, and auditable metrics; it does not establish which individual correction caused the difference. The very high corrected-model scores warrant independent reproduction and investigation of source/acquisition confounding. Duplicate grouping cannot rule out patient overlap or shared source characteristics without patient and acquisition metadata.

Limitations include the unverified source, version, license, and label provenance; labels inferred only from folder names; no expert adjudication; no patient identifiers; an image-level rather than patient-level split; heuristic near-duplicate detection; no independent external cohort; and no probability calibration. The legacy split had 151 exact-duplicate groups crossing train/test; corrected grouping does not rule out residual duplication or patient overlap. Source/acquisition confounding remains unresolved, and the JPEG-only sensitivity analysis does not demonstrate freedom from shortcut learning. The bootstrap intervals group similar images but are not patient-clustered. No annotated lesion masks or expert localization references were available to validate Grad-CAM, which remains qualitative without Dice or IoU claims. The interface's coordinates, 3D geometry, and planning views are simulated, not anatomically registered; they do not establish physical tumor location or volume and are not treatment-planning capability. The legacy model's all-tumor prediction on the corrected split is confirmed, but its original training failure remains unexplained because complete logs and provenance are unavailable.

Accordingly, the results are suitable only as qualified preliminary image-level results for software/method development. Before a manuscript makes substantive scientific or generalization claims, the dataset must be traced to an authorized source, labels and inclusion criteria must be checked by qualified reviewers, patient-level splitting and external validation should be performed, and the exact experiment source tree must be archived or committed. Ethics, consent, data governance, and author declarations also require confirmation.

## 5. Conclusion

NeuroVision AI now has an auditable local workflow with a validation-selected ResNet-18, a small-CNN comparison, a held-out image-level test evaluation, strict checkpoint loading, and shared app/evaluator preprocessing. The measured corrected ResNet-18 result was substantially stronger than the legacy all-tumor checkpoint on the same final split. It remains a historical preliminary result against unverified folder labels; the available provenance and patient-level information do not support independent reproduction, patient-independent, external, or clinical performance claims.

## Declarations to complete before submission

- **Ethics approval and consent:** Not established by the repository; confirm the applicable approval, waiver, and consent requirements with the data provider and institution.
- **Data availability and license:** The local dataset source, citation, release, and license are unknown. Establish authorization and add the primary dataset citation.
- **Code availability:** Source repository: `https://github.com/stephenrodrick17-cloud/Neurovision-AI`. Archive the exact worktree/source revision and experiment artifacts before submission.
- **Funding, conflicts, author contributions, acknowledgments:** Not available in the inspected materials; authors must complete these statements.

## References

1. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2016:770–778. doi:10.1109/CVPR.2016.90.
2. Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L. ImageNet: A large-scale hierarchical image database. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 2009:248–255. doi:10.1109/CVPR.2009.5206848.
3. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. 2017:618–626. doi:10.1109/ICCV.2017.74.

The source dataset citation could not be identified from local repository contents and must be added after provenance is established. Verify reference details and format against the target journal.
