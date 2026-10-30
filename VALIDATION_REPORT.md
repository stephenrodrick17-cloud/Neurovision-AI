# External Validation Report

**Report date:** 2026-10-20  
**Model:** Frozen binary ResNet-18 (`no_tumor=0`, `tumor=1`)  
**Checkpoint SHA-256:** `0ce4741c3ce2cbc287c7b716395b1571bac143200e2f3489e48efd002310b40e`  
**Decision threshold:** 0.5, fixed before evaluation

## Executive summary

The frozen classifier was evaluated on two external image collections. On the overlap-screened Cheng Figshare v8 cohort, the model detected 72.41% of tumor-positive slices. This cohort contains no negative examples, so it cannot measure specificity or overall binary discrimination. On the provided BraTS 2023 GLI archive, slice-level accuracy was 60.19%, balanced accuracy 59.92%, and ROC-AUC 0.6623. These results indicate limited-to-moderate slice-level discrimination and do not establish patient-level or clinical performance.

| Dataset and evaluation subset | Images/slices | Patients | Accuracy | Balanced accuracy | Tumor sensitivity | No-tumor specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cheng Figshare v8, overlap-screened primary subset | 1,689 | 152 | Not estimable | Not estimable | 72.41% | Not estimable | Not estimable |
| BraTS 2023 GLI, provided archive | 169,066 | 1,251 | 60.19% | 59.92% | 52.60% | 67.24% | 0.6623 |

The evaluation unit for both datasets is a 2D image slice. Patient IDs were used to cluster bootstrap resampling for confidence intervals; slices from one patient must not be treated as independent observations.

## Model and evaluation method

- The checkpoint was loaded without retraining, fine-tuning, or threshold selection on either external dataset.
- Images were converted to RGB, resized to 224 × 224, and normalized with the model's ImageNet mean and standard deviation.
- The binary tumor score was the model's softmax score for the positive class; predictions used the fixed 0.5 threshold.
- Confidence intervals are 95% patient-cluster bootstrap intervals. Cheng intervals use the recorded patient bootstrap evaluation. BraTS intervals below use 50 patient-cluster bootstrap replicates and should be considered approximate; a larger replicate count is recommended before publication.
- The evaluator does not verify that external subjects were absent from the checkpoint's training data. The checkpoint's independence from both external cohorts is therefore not established by these results.

## Dataset 1: Cheng Figshare v8

**Source:** Cheng brain tumor dataset, Figshare v8, [DOI 10.6084/m9.figshare.1512427.v8](https://doi.org/10.6084/m9.figshare.1512427.v8).  
**Input:** T1-weighted contrast-enhanced tumor slices with patient IDs and tumor subtype labels. All three tumor subtypes map to the model's single `tumor` class.

### Primary result: overlap-screened cohort

An exact/near-image similarity screen against the local train, validation, and test images identified candidate matches for 81 Cheng patients. Those patients (1,375 slices) were excluded from the primary evaluation. The remaining subset has 1,689 positive slices from 152 patients.

| Metric | Result |
|---|---:|
| Tumor sensitivity | 72.41% (1,223 / 1,689) |
| 95% patient-cluster bootstrap CI | 66.90%–77.99% |
| Mean tumor score | 0.7210 |

Subtype-specific sensitivity is descriptive because of the small, unevenly distributed patient counts:

| Subtype | Slices | Patients | Sensitivity | 95% patient-cluster bootstrap CI |
|---|---:|---:|---:|---:|
| Glioma | 51 | 8 | 100.00% | 100.00%–100.00% |
| Meningioma | 708 | 82 | 96.47% | 94.28%–98.21% |
| Pituitary | 930 | 62 | 52.58% | 45.79%–59.86% |

The all-positive cohort has no no-tumor images. Specificity, accuracy, balanced accuracy, ROC-AUC, and average precision cannot be estimated. The overlap screen detects candidate image similarity only; it neither proves patient identity nor rules out residual patient overlap. The excluded patients substantially change the subtype composition, so these results are not an unbiased estimate for the full dataset.

### Supplementary unscreened result

For context only, the unscreened 3,064-slice, 233-patient cohort had sensitivity of 84.60% (2,592 / 3,064; 95% patient-cluster bootstrap CI 80.90%–88.24%). Because candidate image matches were detected, this is not the primary external-validation estimate and should not be presented as independent validation.

## Dataset 2: BraTS 2023 GLI

**Input supplied:** `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData (1).zip`. The archive name and file structure indicate BraTS 2023 GLI training data; the archive's acquisition provenance and applicable use terms were not independently verified as part of model evaluation.

The available T2-FLAIR (`t2f`) volume was used as the FLAIR input, paired with each subject's segmentation. Volumes were converted to closest canonical orientation; nonzero intensities were scaled per volume using the 0.5th and 99.5th percentiles. Axial slices with fewer than 256 nonzero pixels were omitted. A slice was labeled tumor-positive if its segmentation contained any nonzero voxel.

| Cohort detail | Count |
|---|---:|
| Subjects | 1,251 |
| Retained 2D slices | 169,066 |
| Tumor-free slices | 87,653 |
| Tumor-containing slices | 81,413 |

### Results at threshold 0.5

| Metric | Estimate | 95% patient-cluster bootstrap CI |
|---|---:|---:|
| Accuracy | 60.19% | 59.35%–60.92% |
| Balanced accuracy | 59.92% | 59.07%–60.65% |
| Tumor sensitivity (recall) | 52.60% | 50.94%–54.06% |
| No-tumor specificity | 67.24% | 66.61%–67.92% |
| Tumor precision | 59.86% | 58.82%–60.89% |
| Tumor F1 | 56.00% | 54.70%–57.09% |
| ROC-AUC | 0.6623 | 0.6537–0.6732 |
| Average precision | 0.6788 | 0.6678–0.6899 |

The negative class means a tumor-free axial slice from a subject in this tumor-bearing GLI cohort. It does not mean a healthy subject. These are slice-level metrics, not patient-level detection or segmentation results.

## Interpretation and limitations

1. **No clinical claim:** Neither result validates diagnosis, screening, triage, treatment selection, or patient outcomes.
2. **Slice-level dependence:** Slices from a volume are correlated. Patient-cluster bootstrap intervals account for grouping, but slice metrics do not translate directly to patient-level performance.
3. **No healthy controls in BraTS:** BraTS negative slices are tumor-free slices within tumor-bearing scans; specificity against healthy participants was not measured.
4. **Cheng is positive-only:** The Cheng cohort cannot provide specificity or full binary metrics. Its overlap-screened subset has shifted subtype mix, and small subtype patient counts make subgroup estimates uncertain.
5. **Overlap and independence:** Cheng candidate image matches were screened and excluded, but the screen cannot establish identity or rule out all overlap. No equivalent image-overlap screen against the supplied BraTS archive was performed. Training-data independence is not verified for either external cohort.
6. **Preprocessing/domain differences:** Both datasets are transformed to the classifier's 2D RGB input format. This does not establish equivalence of acquisition, modality, labeling, or clinical domain to the training data.
7. **BraTS confidence intervals:** The reported BraTS intervals were calculated with 50 bootstrap replicates due to the large evaluation workload. They are useful as an approximate uncertainty summary but should be recomputed with more replicates for publication.
8. **Provenance and licensing:** The Cheng source record states CC BY 4.0. BraTS archive authorization, version metadata, and terms were not independently audited in this run.

## Reproducibility

The Cheng JSON reports, predictions, and overlap audit are under `BrainTumorAI/results/external_validation/`. The prepared BraTS manifest and slices were generated under `D:\datasets\prepared\brats`; this report records the completed BraTS model-scoring output. The external-validation implementation and commands are documented in [EXTERNAL_VALIDATION.md](EXTERNAL_VALIDATION.md).
