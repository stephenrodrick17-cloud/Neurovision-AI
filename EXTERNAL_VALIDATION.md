# External validation: Cheng and BraTS

The repository now has a separate, frozen-checkpoint validation pipeline for two candidate external sources. It does not train or tune the classifier on either dataset, and it never merges external examples into the existing training, validation, or test partitions.

## Current status

**Cheng and BraTS GLI external evaluations are complete for the locally available files.** The supplied `1512427.zip` was verified against the official MD5 values for all four Figshare archives. Preparation read the nested MAT files directly without extracting the approximately 880 MB archive. The user-provided `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData (1).zip` was prepared and evaluated as a 2D slice cohort using the available T2-FLAIR (`t2f`) volumes and segmentations. Summary metrics and limitations are in [VALIDATION_REPORT.md](VALIDATION_REPORT.md). The supplied archive's authorization/version provenance and licensing terms were not independently verified in this run.

An exact/near-duplicate screen against all 4,600 local train, validation, and test images found 818 high-similarity Cheng slices across 81 patients (846 matching edges; no exact-pixel matches). Excluding those patients removed 1,375 slices, all glioma. The remaining 1,689 tumor-positive slices from 152 patients contained 51 glioma slices (8 patients), 708 meningioma slices (82 patients), and 930 pituitary slices (62 patients). Frozen ResNet-18 sensitivity was **72.41% (1,223/1,689)** at the fixed 0.5 threshold, with a 95% patient-cluster bootstrap interval of **66.90–77.99%**. This is positive-case sensitivity only, not full binary external validation, and the subset's subtype composition was strongly changed by the exclusion. The full, unscreened cohort sensitivity was 84.60% (2,592/3,064; 95% patient-cluster bootstrap CI 80.90–88.24%) and is a supplementary descriptive result only because it includes detected image matches. The dataset has no no-tumor class, so specificity and full binary accuracy are not estimable. The similarity screen cannot prove patient identity or exclude residual overlap.

The provided BraTS GLI archive was extracted and prepared under `D:\datasets\brats`; prepared slices and manifest are under `D:\datasets\prepared\brats`. Use a D: path for BraTS downloads and generated slices because the C: drive has limited free space. Do not put a Synapse token in this repository or in chat.

## Cheng Figshare cohort

Official record: [Cheng brain tumor dataset, Figshare v8](https://doi.org/10.6084/m9.figshare.1512427.v8). Figshare identifies the license as **CC BY 4.0** and the data record describes 3,064 T1-weighted contrast-enhanced image slices from 233 patients. The accompanying README documents a patient identifier (`cjdata.PID`) and three labels: meningioma, glioma, and pituitary tumor.

The official Figshare README confirms label codes 1 = meningioma, 2 = glioma, and 3 = pituitary tumor, and demonstrates per-image min–max conversion to 8-bit grayscale. The pipeline follows this scaling, restores MATLAB matrix-axis order for v7.3/HDF5 files, and applies the project's shared RGB/resize/normalization transform; no anatomical orientation normalization is done. All three labels map to the classifier's positive `tumor` class. The cohort has no no-tumor class; therefore it can estimate **tumor-positive slice sensitivity only**. It cannot estimate specificity, balanced accuracy, binary accuracy, or ROC-AUC. Confidence intervals resample whole patient IDs, not individual slices. This result must not be presented as complete binary external validation or as subtype classification.

### Prepared cohort and evaluation

The verified outer archive can be used directly; it does not need to be extracted. Prepared manifests, metrics, and predictions are ignored by Git.

```powershell
python -m BrainTumorAI.external_validation prepare-cheng `
  --data-root "C:\Users\asus\Downloads\Brain Imaging\1512427.zip" `
  --output-dir "BrainTumorAI\results\external_validation\cheng_figshare_v8"
```

The preparation verified all four embedded Figshare archive checksums and wrote a manifest for 3,064 slices and 233 patients. It retains patient IDs and tumor subtype; image data are read from the nested archive during evaluation, so duplicate image copies are not written to disk. MAT v7.3 files require `h5py`.

Screen for candidate overlap against every local split:

```powershell
python -m BrainTumorAI.external_validation audit-cheng-overlap `
  --archive "C:\Users\asus\Downloads\Brain Imaging\1512427.zip" `
  --local-manifest "BrainTumorAI\results\split_manifest.csv"
```

### Evaluate the overlap-screened subset

```powershell
python -m BrainTumorAI.external_validation evaluate `
  --manifest "BrainTumorAI\results\external_validation\cheng_figshare_v8\manifest.csv" `
  --image-root "BrainTumorAI\results\external_validation\cheng_figshare_v8" `
  --dataset-name cheng_figshare_v8_overlap_screened `
  --overlap-audit "BrainTumorAI\results\external_validation\cheng_figshare_v8\overlap_audit.json"
```

The all-slice report and predictions are under `BrainTumorAI/results/external_validation/cheng_figshare_v8/`; do not present the unfiltered cohort as independent validation because of the detected overlap. The primary overlap-screened report and predictions are under `BrainTumorAI/results/external_validation/cheng_figshare_v8_overlap_screened/`. The screen excludes every matched glioma patient, so the retained subset is not representative of the source subtype composition. This report also gives subtype-specific, patient-clustered sensitivity. Specificity, binary accuracy, balanced accuracy, ROC-AUC, and average precision are not estimable on this all-tumor cohort.

## BraTS FLAIR slice cohort

Obtain the chosen BraTS release from the [official challenge portal on Synapse](https://www.synapse.org/Synapse:syn51156910) and follow its current access, registration, citation, and use terms. This preparation path expects one FLAIR NIfTI volume (named `_flair` or BraTS GLI `_t2f`) and a matching segmentation NIfTI per subject, with filenames ending in `_seg.nii[.gz]` (hyphen separators are also accepted). Install the optional parser if needed:

Authenticate Synapse locally first; do not share the token. Then inspect the account's available Synapse downloads:

```powershell
.\.venv\Scripts\synapse.exe login
.\.venv\Scripts\synapse.exe get-download-list --downloadLocation "D:\datasets\brats"
```

`get-download-list` itself does not fetch BraTS unless that dataset is already in the account's download list. Identify the BraTS Synapse entity/version authorized for the account, then retrieve that entity recursively with its official Synapse ID, for example:

```powershell
.\.venv\Scripts\synapse.exe get syn<OFFICIAL_ENTITY_ID> --recursive `
  --downloadLocation "D:\datasets\brats"
```

Replace the placeholder only with the ID returned by Synapse for the authorized BraTS release. Do not guess an ID or use another challenge release's terms.

```powershell
.\.venv\Scripts\python.exe -m pip install nibabel
```

Prepare the cohort:

```powershell
python -m BrainTumorAI.external_validation prepare-brats `
  --data-root "D:\datasets\brats" `
  --output-dir "D:\datasets\prepared\brats"
```

The converter canonicalizes image orientation, pairs each scan with its segmentation, and extracts axial FLAIR slices. A slice is positive if any segmentation voxel is nonzero and negative if it contains brain signal but no labeled tumor voxel. Slices with fewer than 256 nonzero FLAIR pixels are omitted by default. Images are scaled to 8-bit using per-volume nonzero-intensity percentiles before the shared classifier preprocessing. The manifest records each subject ID so all confidence intervals resample complete subjects.

This is **2D slice classification**, not patient-level diagnosis or tumor segmentation. A negative slice belongs to a BraTS tumor-bearing case; it is not a healthy participant. Accordingly, specificity is only against tumor-free slices in this cohort. This external test also cannot establish performance on healthy controls.

Evaluate:

```powershell
python -m BrainTumorAI.external_validation evaluate `
  --manifest "D:\datasets\prepared\brats\manifest.csv" `
  --image-root "D:\datasets\prepared\brats" `
  --dataset-name brats_flair_2d
```

Both evaluation commands apply the fixed 0.5 threshold without external-set tuning and write `metrics.json` and per-slice `predictions.csv` under `BrainTumorAI/results/external_validation/<dataset-name>/`. Reports include the manifest and checkpoint hashes. Review the source release, access terms, subject separation, preprocessing, and class counts before reporting results. Neither pipeline is a clinical validation.
