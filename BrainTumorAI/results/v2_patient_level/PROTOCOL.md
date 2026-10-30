# V2 Patient-Level Brain MRI Classification Protocol

**Protocol status:** Amended; awaiting user approval before Phase 1  
**Protocol version:** 1.1  
**Original protocol date:** 2026-10-21  
**Amendment 1 date:** 2026-10-21  
**Superseded Protocol v1.0 SHA-256:** `B1B6DBC878E8BCFFBB0B5B0A01713E533328AF263BDB24CB8400956A7653AD98`  
**Current Protocol v1.1 SHA-256:** Recorded in the adjacent `PROTOCOL.sha256` file after this amendment is written. The digest sidecar is not part of the hashed protocol.

## Amendment 1 (2026-10-21)

This amendment was made before Phase 1 and before any V2 model training or model-output review. It records a mask-area-only descriptive analysis of the 1,251 available BraTS GLI segmentation volumes. No MRI intensities, classifier predictions, checkpoint results, or model outputs were read for this analysis. The saved counts and mask-area quantiles are in `mask_area_distribution.json`.

Across 193,905 canonical axial segmentation slices, 112,468 had zero mask area and 81,437 had nonzero mask area. Among nonzero slices, mask-area percentiles (pixels) were: p1=5, p5=43, p10=121, p25=487, median=1,268, p75=2,247, p90=3,109, p95=3,593, and p99=4,618. There were 1,416 nonzero slices with area below 10 pixels (1.74% of nonzero-mask slices). Therefore, **A_min = 10 pixels** is adopted: it removes only the smallest 1.74% of labeled positive slices by area while retaining 98.26%; this is a data-informed engineering cutoff, not a clinically validated lesion threshold.

Prespecified slice mask-area reporting bins (computed only for retained tumor-positive slices) are:

| Bin | Mask area |
|---|---:|
| Small | 10–49 pixels |
| Medium-small | 50–199 pixels |
| Medium-large | 200–999 pixels |
| Large | ≥1,000 pixels |

These are **2D slice mask-area bins**, not 3D tumor-volume or clinical tumor-size categories.

Amendment 1 supersedes any conflicting v1.0 statements below:

- **Primary experiment:** train only on BraTS GLI training patients and evaluate the one-time final test only on held-out BraTS GLI patients. No Cheng images are used for primary model fitting, validation, or threshold selection.
- **Cheng:** use as a positive-only external sensitivity cohort after excluding patients with candidate matches in the existing Cheng-to-Kaggle/local similarity audit. It has no negative class, so binary accuracy, specificity, and ROC-AUC are not estimable.
- **Leave-one-source-out:** remove the binary “train on Cheng, test on BraTS” experiment; Cheng-only training cannot support a binary model. “Train on BraTS, evaluate on Cheng” is retained only as positive-only external sensitivity analysis.
- **Optional mixed-source analysis:** mixing Cheng with BraTS for training is not part of the primary experiment. It may only be added after a separately approved amendment and must be explicitly labeled as carrying a source/modality shortcut risk (Cheng T1-contrast versus BraTS T2-FLAIR), with a source-predictor shortcut check.
- **Phase 2 source predictor:** this is a separate diagnostic classifier (source label only), not a tumor model and not part of primary fitting. Split Cheng patients deterministically by patient ID and seed 42 into 80% source-predictor training and 20% source-predictor validation; exclude the diagnostic-validation patients from the later Cheng sensitivity evaluation. Train the source predictor on Cheng diagnostic-training images plus BraTS primary-training images; validate on the Cheng diagnostic-validation patients plus BraTS primary-validation patients, using equal numbers of images per source sampled by the same seeded method. Never use BraTS primary-test patients. Report this as source separability/shortcut risk, not proof that the primary tumor model learned a source shortcut.
- **Sampling:** after patient split and slice eligibility filtering, sort patient IDs and slice indices. Within each patient and class, select uniformly without replacement using NumPy `Generator(PCG64(42))`, up to 20 positive and 20 negative slices (maximum 40 per patient). Do not fill an unfilled class quota from the other class. Preserve natural validation and test distributions. Compute inverse-frequency class weights from the sampled BraTS training labels only; do not weight validation/test metrics or use their labels to set weights. Record chosen slice indices and RNG seed.
- **Patient-level aggregation:** for each patient, use the arithmetic mean of the selected slice tumor probabilities. Define the patient label from all eligible slices before sampling: positive if any eligible slice has mask area ≥A_min, otherwise negative. If all BraTS subjects are positive, report patient-level tumor-bearing sensitivity only; patient specificity and balanced accuracy are then not estimable. If both patient labels occur, report the binary patient metrics. Cheng patient score is the arithmetic mean of its image tumor scores and is positive-only sensitivity.
- **Threshold:** select the decision threshold by maximizing slice-level balanced accuracy on BraTS validation data only; ties resolve to the threshold closest to 0.5, then the lower threshold. Freeze it before any test evaluation and external Cheng evaluation. Apply this same threshold to both slice scores and patient-mean scores; do not separately optimize a patient threshold on an all-positive patient cohort.
- **Patient split:** use 70/15/15 train/validation/test proportions for BraTS patients, seed 42, grouped by patient ID. The sole source is BraTS GLI and the subtype is glioma; stratify by the maximum eligible positive slice-area bin per patient (using segmentation labels only, before slice sampling) to balance tumor burden. Split before slice sampling or image preprocessing. Cheng is a separate external cohort, not a split member.
- **Amendment policy:** any change to this protocol after approval requires a new, dated, sequential amendment appended to this file before the changed procedure or any affected model output is inspected. Record the prior protocol SHA-256, reason, exact changed rules, and new SHA-256 sidecar. Never replace or delete prior hash records. Changes after test access require explicit user approval and must be labeled exploratory; the test set may not be rerun to tune choices.

## 1. Study objective and data sources

This study evaluates a binary tumor-slice classifier with patient-grouped partitions. The primary model is trained only on BraTS GLI training patients and evaluated on held-out BraTS GLI patients. It is a research evaluation, not a clinical validation.

1. **Cheng Figshare v8** (`1512427.zip`): 3,064 T1-weighted contrast-enhanced tumor images from 233 patients, with meningioma, glioma, and pituitary subtype labels. All three subtypes map to `tumor=1`. This collection has no healthy or no-tumor class.
2. **BraTS 2023 GLI**: subject MRI volumes and segmentation masks. Use only the T2-FLAIR volume (`t2f`) and its paired segmentation; do not use T1, T1-contrast, or T2-weighted volumes.
3. **Legacy Kaggle/local folder collection**: not used for primary training, validation, threshold selection, or V2 test evaluation. The existing exact/near-duplicate audit is used only to screen/exclude candidate-matched Cheng patients from the positive-only external evaluation. Any separate legacy comparison is outside the primary V2 experiment and is image-level because the legacy collection has no verified patient identifiers.

The source and imaging domains differ: Cheng provides 2D T1-contrast images while BraTS provides 2D axial T2-FLAIR slices. They will be resized and normalized consistently, but the modality difference remains a domain-shift limitation.

## 2. Labels, slice rules, and sampling

### 2.1 Cheng images

Convert each Cheng `cjdata.image` matrix to 8-bit grayscale using per-image min-max scaling, as in the existing external-validation preparation. Preserve its patient ID and subtype. Do not derive or synthesize negative examples for Cheng.

### 2.2 BraTS FLAIR slices

Canonicalize paired FLAIR and segmentation volumes to the closest canonical orientation, then use axial slices along canonical axis 2. Reject a pair if the arrays, affine transforms, or patient identifiers do not match.

For every axial slice, define:

- **Brain area:** count of nonzero FLAIR pixels in that slice.
- **Central brain range:** slices with brain area at least **50% of the maximum brain area in that patient's volume**.
- **Tumor area:** count of pixels with segmentation label greater than zero.
- **Positive (`tumor=1`):** tumor area is **at least A_min = 10 pixels**. Amendment 1 selected this cutoff from the observed BraTS mask-area distribution: it excludes 1.74% of nonzero-mask slices. This is not a clinically validated minimum lesion size. Positive slices are retained regardless of the central-range rule.
- **Negative (`no_tumor=0`):** tumor area is zero and the slice is within the central brain range.
- **Ambiguous/excluded:** tumor area is greater than zero but less than 10 pixels; also exclude zero-tumor slices outside the central brain range.

Do not alter these rules after viewing model outputs. Record per-patient candidate, retained, positive, negative, and excluded-slice counts in the preparation audit.

### 2.3 Per-patient sampling and cap

After the patient split and slice eligibility filtering, sort patient IDs and slice indices. Use one NumPy `Generator(PCG64(42))` instance, iterating in that sorted order. Within each patient and class, select uniformly without replacement, up to 20 tumor-positive and 20 tumor-free eligible slices (maximum **N = 40** per patient). Do not fill an unused class quota with the other class. This seeded, class-stratified sampling limits long-volume domination while keeping per-patient class contributions balanced where both classes are available. Keep validation/test at their natural sampled class distributions. Record candidate counts, retained counts, selected indices, and seed in the manifest/audit.

## 3. Healthy controls and negative-class interpretation

An inventory of `D:\datasets` found only `brats` and `prepared` at its top level and no dataset license or README files. No separately documented healthy-control dataset was identified; none will be used unless a source and license are verified before data preparation. BraTS tumor-free slices are negatives only for the **slice-level** task. They are not healthy participants, and no healthy-control specificity claim is permitted.

## 4. Patient-level partitions and leakage controls

Create a BraTS patient manifest for the primary experiment. Keep Cheng in a separate external-evaluation manifest. Use source-qualified IDs (`cheng:<source patient ID>` and `brats:<subject ID>`) to prevent accidental ID collisions. A patient's slices and all augmented/cached derivatives must remain in one partition.

Assign BraTS patients once to train/validation/test at a target **70% / 15% / 15%** using deterministic seed **42**. Because the primary cohort is one source and one subtype, stratify patients by their maximum eligible tumor-positive slice mask-area bin before slice sampling (mask labels only; not MRI intensities or model outputs). Preserve the partition across all models and seeds. Save and verify the patient-to-split mapping before slice sampling or image preprocessing. Explicitly assert that no BraTS patient ID appears in more than one partition. Cross-source identity cannot be established from the available identifiers.

The existing Cheng-to-local/Kaggle overlap audit is reused to identify candidate matching Cheng slices. Exclude every matched Cheng patient from V2 training, validation, and test (conservative patient-level exclusion); do not use the Kaggle collection for training. Record the audit file hash and excluded IDs/counts in the V2 audit. The existing audit is only a similarity screen, not proof of identity or proof that residual overlap is absent. If the audit cannot be mapped unambiguously to Cheng patient IDs, stop and report that blocker rather than silently using affected data.

Reserve the BraTS test partition. Do not load test images, labels, predictions, or metrics during data design, preprocessing choices, source-shortcut checks, training, or threshold selection. Cheng's positive-only external set is also evaluated only after the protocol, checkpoints, and validation-selected threshold are frozen. Any final BraTS test execution is a one-time evaluation only after explicit user approval. If that run fails, stop and ask before any retry.

## 5. Primary and external-source experiments

1. **Primary:** train/validate on BraTS GLI train/validation patients and evaluate once on held-out BraTS GLI test patients.
2. **Cheng external sensitivity:** evaluate the frozen BraTS-trained model on Cheng v8 after excluding every patient identified by the existing Cheng-to-Kaggle/local overlap audit. Select no parameter or threshold from Cheng. Report tumor-positive sensitivity only; no specificity or full binary metrics are estimable.
3. **Optional mixed-source analysis:** mixing Cheng and BraTS in training is excluded from the primary study. It requires a new approved amendment and a source-shortcut analysis because source and modality (T1-contrast vs T2-FLAIR) can be learned instead of tumor signal.

## 6. Models, training, and validation selection

Train **ResNet-18 with ImageNet initialization** and **DenseNet-121 with ImageNet initialization**, each with seeds **42, 43, 44, 45, and 46**. Use the existing `BrainTumorAI/train_classifier.py` recipe unless an implementation limitation requires a documented, approved amendment:

- RGB input, resize to 224 × 224, ImageNet mean/std normalization.
- Training-only horizontal flip and rotation up to 10 degrees.
- Adam, learning rate `1e-4`, weight decay `1e-4`; weighted cross-entropy with inverse-frequency class weights calculated from the sampled BraTS training labels only. This is a prespecified amendment to handle residual slice-label imbalance after per-patient stratified sampling; validation/test distributions remain unchanged.
- Batch size 16, maximum 12 epochs, early-stopping patience 4.
- Reduce-on-plateau scheduler on validation balanced accuracy, factor 0.5, patience 2.
- Select each seed's checkpoint by validation balanced accuracy; break ties within `1e-4` by lower validation loss.
- No oversampling. Do not calculate class weights from validation or test labels.

The training split is used for fitting. Validation data alone may be used for checkpoint selection, calibration, and threshold selection. Save epoch-level logs, validation predictions, environment/library versions, seed, split-manifest SHA-256, and SHA-256 for every checkpoint. Generate independent output names; never overwrite an existing result.

## 7. Threshold and metrics

For each selected checkpoint, choose a decision threshold by maximizing **slice-level balanced accuracy on BraTS validation patients only** over candidate thresholds formed from distinct validation slice scores plus endpoints 0 and 1. If tied, choose the threshold closest to 0.5; if still tied, choose the lower threshold. Freeze this threshold before test or Cheng evaluation and apply it to slice scores and patient-mean scores. Do not tune a patient-level threshold on the all-positive BraTS patient cohort.

Report slice-level metrics for BraTS and Cheng where estimable. At patient level, aggregate slice tumor scores by their **arithmetic mean** and assign truth from all eligible segmentation slices before sampling (positive if any slice area is ≥A_min). If the BraTS patient cohort has only positive patients, report patient-level sensitivity only and mark specificity, balanced accuracy, and full binary metrics not estimable. Cheng remains positive-only at the patient level.

- accuracy, balanced accuracy, sensitivity, specificity, precision, F1;
- ROC-AUC, average precision, Brier score, and 15-bin equal-width expected calibration error (ECE).

For confidence intervals, resample **patients**, retaining all selected slices for each sampled patient, with **1,000 bootstrap replicates** and fixed seed 42. Report 95% percentile intervals. Clearly label metrics not estimable because neither evaluation set contains a healthy/tumor-free subject cohort. Report Cheng subtype breakdowns and BraTS slice mask-area-bin breakdowns; do not interpret BraTS tumor-free slices as healthy controls.

For the final approved BraTS test and Cheng positive-only external evaluation, report each seed separately and mean ± standard deviation across the five seeds. Thresholds remain the per-checkpoint validation-selected thresholds; do not average or retune them on test/external data.

## 8. Phase-specific outputs and safeguards

All V2 outputs are new and belong under `BrainTumorAI\results\v2_patient_level\`. Do not overwrite legacy results. Cache 224 × 224 images only on `D:\`; do not store BraTS images or derived BraTS slices inside Git-tracked paths or release archives.

Before any training, add the relevant BraTS data/cache paths and checkpoint folders to `.gitignore`. Fix and record seeds, software versions, source archive hashes, split-manifest SHA-256, checkpoint SHA-256, and the exact commands used. Do not store, request, or display Synapse credentials.

Phase gates:

1. **Phase 0 — protocol:** no training; record the frozen protocol hash and wait for approval.
2. **Phase 1 — preparation:** build and verify the patient-level manifest, report split/source/class/type counts and hashes, then wait.
3. **Phase 2 — shortcut checks:** train only the source predictor on train/validation, report image-statistic analyses in `SHORTCUT_REPORT.md`, and wait. If source prediction validation accuracy exceeds 95%, flag a serious shortcut risk and propose mitigations without applying them until approved.
4. **Phase 3 — model training:** train the specified models using training data and validation-only selection. Report validation results only; do not inspect test.
5. **Phase 4 — final test:** proceed only after explicit approval of frozen checkpoints and thresholds. Access test once; if an error interrupts evaluation, stop and request approval before retrying.
6. **Phase 5 — quantitative Grad-CAM:** use only the approved test run and a localization threshold selected on validation; compare with masks, include random-heatmap and model-randomization baselines, and describe Grad-CAM as an explanation aid rather than segmentation.
7. **Phase 6 — reporting:** write the reproducibility report/checklist, exact commands, artifact inclusion/exclusion list, and run tests plus `git diff --check`.

## 9. Prespecified assumptions and constraints

- BraTS `_t2f` is treated as the challenge's T2-FLAIR modality.
- Cheng's available images are T1-contrast and are not converted into synthetic FLAIR.
- The BraTS 10-pixel threshold (informed by the saved mask-area distribution), 50%-of-maximum central range, 40-slice cap, and per-patient class-stratified sampling rule are fixed; do not tune them on validation or test.
- The Cheng-to-Kaggle overlap audit only detects exact/high-similarity image candidates. Its patient-level exclusion is conservative and is not a patient-identity determination.
- No separately licensed healthy-control dataset was found in the inspected `D:\datasets` inventory.
- Cross-source patient linkage cannot be checked from source-qualified IDs alone.
- BraTS patient-level specificity is not estimable because the GLI subjects are tumor-bearing; Cheng is a positive-only external cohort.
- This study does not establish clinical validity, patient-level diagnosis, segmentation performance, or patient independence from the original training collection.
