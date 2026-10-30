# Dataset provenance checklist

This checklist records only what the repository itself proves. Where the repo does not contain a reliable fact, it is marked as `TO FILL BY AUTHOR (check the Kaggle page)` and not guessed.

## Provenance record

| Field | Value recorded in repo | Evidence | Status |
|---|---|---|---|
| Dataset name | `TO FILL BY AUTHOR (check the Kaggle page)` | The local repository contains `archive/Brain Tumor Data Set/Brain Tumor Data Set`, but that directory name is not an authoritative citation. See `DATASET_PROVENANCE.md` and `BrainTumorAI/preprocess.py`. | Not proven |
| Source URL | `TO FILL BY AUTHOR (check the Kaggle page)` | No authoritative provenance record links the local files to a publication, repo, or source page. | Not proven |
| Original authors | `TO FILL BY AUTHOR (check the Kaggle page)` | No author or publication metadata is present in the repo. | Not proven |
| Version / release date | `TO FILL BY AUTHOR (check the Kaggle page)` | No version identifier or download record is present locally. | Not proven |
| License / redistribution terms | `TO FILL BY AUTHOR (check the Kaggle page)` | No license statement or source terms are present with the local images. | Not proven |
| How labels were created | Folder-derived labels: `Brain Tumor` -> `tumor=1`, `Healthy` -> `no_tumor=0`; no expert adjudication record found. | The repo mapping is defined in `BrainTumorAI/preprocess.py` and summarized in `DATASET_PROVENANCE.md`. | Partially proven |
| Patient IDs available | `No` | `DATASET_PROVENANCE.md` states there are no reliable patient, study, series, or source-volume identifiers in filenames, folders, or local metadata. | Proven |
| Download date | `TO FILL BY AUTHOR (check the Kaggle page)` | No download record or source-date metadata is stored in the repo. | Not proven |

## Repository evidence used

- `DATASET_PROVENANCE.md` — provenance and integrity record
- `BrainTumorAI/preprocess.py` — local source-class mapping and audit report fields
- `BrainTumorAI/classifier_pipeline.py` — local image directory root (`archive/Brain Tumor Data Set/Brain Tumor Data Set`)

## Notes from the repo

- The repo does not establish a publication-grade dataset identity or permission to redistribute the images.
- The dataset is treated as a local image collection of unknown origin and provenance.
- The split logic is image-level and uses duplicate grouping, but patient-level grouping is not possible because patient identifiers are absent.
- Until provenance and licensing are established, the project should not state a verified dataset citation or reuse permission.
