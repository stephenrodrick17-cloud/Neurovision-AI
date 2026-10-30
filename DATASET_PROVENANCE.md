# Dataset provenance and integrity record

## Scope and status

This record describes only the local image collection audited for the NeuroVision-AI experiment. It does not establish a publication-grade dataset identity or permission to redistribute the images.

| Item | Evidence available locally | Source | Verification status | Remaining uncertainty |
|---|---|---|---|---|
| Local directory | `archive/Brain Tumor Data Set/Brain Tumor Data Set/{Brain Tumor,Healthy}` | Local repository data tree | Verified locally | The directory name is not an authoritative dataset citation. |
| Official dataset name | No matching source record, citation, or archive metadata was found. | Local documentation, filenames, and metadata search | **UNVERIFIED** | The original official name and whether this is a subset, mirror, or repackaging remain unknown. |
| Original source / publication / URL | No authoritative provenance record links the local files to a publication or repository. Similarly named public datasets were not sufficient to establish identity. | Local repository and prior catalog search | **UNVERIFIED** | Exact source, authors, publication, release page, and version. |
| Release/version | No version identifier or download record is present. | Local files and repository metadata | **UNVERIFIED** | Dataset release, collection date, and any changes made before local use. |
| License and redistribution permission | No license statement or source terms are present with the local images. | Local documentation and metadata | **UNVERIFIED** | Copyright, permitted research use, redistribution, and attribution requirements. Do not redistribute until verified. |
| Image count and readability | 4,600 files were readable: 2,513 under `Brain Tumor`, 2,087 under `Healthy`; zero unreadable files in the recorded audit. | `BrainTumorAI/results/dataset_audit.json` (local, ignored output) | Verified for the audited local snapshot | The local data could have changed since the audit; regenerate the audit to verify a new copy. |
| Folder labels | `Brain Tumor` maps to `tumor=1`; `Healthy` maps to `no_tumor=0`. | Folder names and explicit pipeline mapping | Verified as the software mapping | Independent clinical label review and the original label-generation process are not documented. These are folder-derived labels, not adjudicated diagnoses. |
| Image content / dimensionality | Files are two-dimensional raster images in JPEG, PNG, and TIFF formats; 4,496 JPEG, 18 PNG, and 86 TIFF. | Image decoder audit | Verified as file representation | It is not established whether each file is a slice, a separately acquired MRI, an image exported from a volume, or a derived/reformatted image. |
| Image properties | 459 distinct width/height combinations; RGB, grayscale, RGBA, and palette modes were found. | Image decoder audit | Verified for the local snapshot | Scanner, sequence, acquisition protocol, spatial orientation, and intensity calibration are not available. |
| Patient/study/source-volume identifiers | No reliable patient, study, series, or source-volume identifiers were found in filenames, folders, local metadata, or available sidecars. | Local source tree and metadata inspection | Not available | Patient overlap across image files and partitions cannot be assessed. IDs must not be inferred or manufactured from filenames. |
| Expert label review | No review records or adjudication annotations are present. | Local repository search | **UNVERIFIED / NOT DOCUMENTED** | Label correctness and reviewer qualifications are unknown. |
| Segmentation/localization references | No tumor masks or expert anatomical localization annotations were found. | Local repository search | Not available | Grad-CAM localization cannot be quantitatively validated against this collection. |
| BraTS relationship | No evidence identifies this collection as BraTS or provides BraTS volumes/masks. | Local files and repository documentation | Not established | Do not describe this as a BraTS experiment. |

## Integrity audit recorded

The seed-42 audit identified:

- 432 exact decoded-pixel duplicate groups containing 1,017 images; no exact-duplicate group had conflicting folder labels.
- A candidate near-duplicate screen requiring both perceptual-hash Hamming distance `<= 4` and 64x64 grayscale SSIM `>= 0.95`. It yielded 955 multi-image connected components containing 2,782 images, with no high-similarity cross-label edge recorded.
- The screening is a heuristic image-similarity method. It may miss related images or group unrelated similar-looking images; it does not identify patients or source volumes.
- The old prepared train/test folders had 151 exact-pixel duplicate groups crossing the split boundary, involving 224 training-folder and 167 test-folder images.
- The revised seed-42 manifest assigns entire detected duplicate components to one partition. The split is image-level, not patient-independent.
- In the 606-image final test partition, 590 images are JPEG and 16 are TIFF; all 16 TIFF files are tumor-labeled. The exploratory JPEG-only check remained similarly high, so file format alone does not explain the result. Other acquisition/source confounding remains unresolved.

The full image-level audit and file manifest are local ignored outputs under `BrainTumorAI/results/`. Review them before sharing: the manifest contains source paths and file hashes. Do not publish data or identifying metadata without authorization.

## Research-use restrictions and next steps

Until provenance and license are established, treat the image collection as locally held material of unknown redistribution status. Obtain the original source page/publication and applicable terms; document the exact downloaded release and file counts; confirm authorization; and preserve a provenance record for any transformed copy. Obtain reliable patient/study/source-volume grouping and independent label adjudication before claiming patient-independent performance or clinically meaningful labels.
