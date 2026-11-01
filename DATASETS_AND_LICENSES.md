# Datasets, sources and licences

This repository contains **code, manifests, derived metrics and figures only**. No raw images, volumes or model
checkpoints (`*.pth`) are distributed here (see `.gitignore`). Everyone who re-runs the experiments must obtain the
datasets from the original providers and accept their terms.

Verification status uses three labels:

- **VERIFIED**: confirmed from the provider's own machine-readable record on the date shown.
- **PARTIAL**: provider page confirms part of the information.
- **NOT VERIFIED**: could not be confirmed from files on disk or from the provider; do not rely on it.

Checked on 2026-11-01.

## Summary

| Dataset | Role in this study | Provider / ID | Licence | Status |
|---|---|---|---|---|
| Kaggle brain-tumour image collection (4,600 images) | Discovery / training set (image-level split 3,191 / 803 / 606) | Kaggle `preetviradiya/brian-tumor-dataset` (probable source) | Kaggle lists **GPL 2** for that listing | **NOT VERIFIED** that the local copy is that exact release; no licence file inside the local archive |
| Cheng brain tumour dataset v8 (3,064 slices, 233 patients, T1-CE) | External cohort (tumour-positive only) and 3-class subtype experiment | Figshare DOI 10.6084/m9.figshare.1512427.v8 | **CC BY 4.0** | **VERIFIED** (Figshare API, `license.name = "CC BY 4.0"`) |
| BraTS 2023 adult glioma (GLI) training data (1,251 patients) | External cohort, patient-split reference models, Grad-CAM localisation vs masks | Synapse project `syn51156910` (ASNR-MICCAI BraTS 2023) | Access governed by the Synapse challenge data-use terms; no SPDX licence confirmed | **PARTIAL / NOT VERIFIED** (see below) |

## 1. Kaggle discovery collection

- Local layout: `Brain Tumor Data Set/Brain Tumor Data Set/{Brain Tumor, Healthy}`; 4,600 readable images
  (2,513 tumour, 2,087 non-tumour). Formats: 4,496 JPEG, 86 TIFF, 18 PNG (from `header_only_baseline.json`).
- Probable source: <https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset> (uploader Preet Viradiya,
  version 1, 2021-05-16). Kaggle's API reports the licence as **GPL 2**.
- **Caveats (important):**
  - The archive we hold has no licence/readme file, so we cannot prove it is that exact release (**NOT VERIFIED**).
  - Kaggle's description of that listing is generic and carries no patient identifiers, acquisition protocol or
    labelling method. Label validity and origin of the underlying images are undocumented.
  - GPL 2 is a software licence and is an unusual choice for image data; its implications for redistribution are
    unclear. **We therefore do not redistribute the images.**
- `DATASET_PROVENANCE.md` and `paper_evidence/release_audit_v7.json` record the same unresolved status.

## 2. Cheng et al. brain tumour dataset (Figshare)

- Citation (as given by Figshare): Cheng, Jun (2017). *brain tumor dataset*. figshare. Dataset.
  <https://doi.org/10.6084/m9.figshare.1512427.v8>
- Content: 3,064 T1-weighted contrast-enhanced slices, 233 patients, three tumour types (meningioma, glioma,
  pituitary), MATLAB `.mat` files with image, label, patient ID, tumour border and mask. Version 8 published
  2024-12-21.
- Licence: **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>). Attribution required; redistribution and
  adaptation allowed.
- Integrity: the four archive parts match the Figshare-supplied MD5 checksums (verified by the Figshare API record
  and by our loader in `BrainTumorAI/external_validation.py`).
- Please also cite the original papers listed in the dataset README (Cheng et al., PLoS ONE 2015; Cheng et al.,
  Scientific Reports 2016).

## 3. BraTS 2023 adult glioma (GLI)

- Provider: Synapse (Sage Bionetworks), ASNR-MICCAI BraTS 2023 challenge, project `syn51156910`
  (<https://www.synapse.org/#!Synapse:syn51156910>). Challenge is closed; training data remain downloadable after
  registration.
- Content used: T2-FLAIR (`t2f`) and, for one control, T1-CE (`t1c`) volumes plus `seg` masks for 1,251 patients;
  169,066 axial slices were sliced from T2-FLAIR (81,413 tumour-containing, 87,653 tumour-free).
- Licence: the Synapse wiki text we fetched states participants obtain the data from Synapse and does not print a
  licence identifier. An earlier internal audit file (`release_audit_v7.json`) wrote "CC BY-SA 4.0 / Synapse
  Challenge Data Use Terms", and a web search summary suggested non-commercial restrictions. **These sources
  disagree and none is authoritative, so the licence is NOT VERIFIED.** Read the terms you accept on the Synapse
  registration/access page before any redistribution or commercial use.
- Required attribution (Synapse convention): *"Data used in this publication were obtained as part of the Brain
  Tumor Segmentation (BraTS) Challenge project through Synapse ID: syn51156910."* Also cite the BraTS overview
  papers (Menze et al. 2015; Bakas et al. 2017; Bakas et al. 2018; BraTS 2023 adult-glioma paper).
- Do not attempt to re-identify patients. We publish only patient-ID manifests (BraTS case IDs), predictions and
  aggregate metrics, not images or volumes.

## What this repository publishes from the datasets

| Artefact | Contains | Derived from |
|---|---|---|
| `paper_evidence/*_manifest.csv` | BraTS case IDs, slice indices, labels, split assignment | BraTS |
| `paper_evidence/*_predictions.csv.gz` | per-slice model scores | BraTS, Cheng |
| `paper_evidence/*.json` | aggregate metrics, confidence intervals, audits | all |
| `paper_evidence/figures/*.png` | figures; Fig. 2 and Fig. 3 embed small image crops for illustration | all |

If you are a rights holder and want a figure or manifest removed, please open an issue.

## Code licence

The source code in this repository currently has **no licence file**. Until the owner adds one, default copyright
applies. This is independent of the dataset licences above.
