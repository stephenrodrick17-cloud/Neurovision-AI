# Manuscript insert pack: tables, figures, and declarations

This file collects material requested for the manuscript revision. Numeric values below are from the saved Phase 1B aggregate/comparison CSVs and Phase 1C bootstrap output. The local MRI image collection itself is not included.

## Table: training configuration

This can be added as a dedicated Methods table. If it is inserted as Table 2, renumber the later result tables and update in-text references.

| Setting | Value |
|---|---|
| Architectures | ResNet-18 (ImageNet `IMAGENET1K_V1`), SmallCNN (from scratch); DenseNet-121 (ImageNet `IMAGENET1K_V1`) in the additional Phase 1B runs |
| Input | RGB, 224 × 224 pixels |
| Class mapping | no-tumor = 0; tumor = 1 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 16 |
| Epoch limit | 12; early stopping could end a run sooner |
| Scheduler | `ReduceLROnPlateau`, monitor validation balanced accuracy, mode `max`, factor 0.5, patience 2 |
| Loss | Unweighted `CrossEntropyLoss` |
| Class imbalance | No class weights or resampling |
| Checkpoint selection | Highest validation balanced accuracy; validation loss breaks ties within the configured tolerance |
| Early stopping | Patience 4 |
| Seed runs | Primary seed 42; ResNet-18 and SmallCNN seeds 42–46; DenseNet-121 seeds 42–44 |
| Seed control | Python, NumPy, PyTorch and CUDA RNGs seeded; cuDNN deterministic enabled and benchmark disabled |
| Device | CUDA when available; saved primary evaluation metadata records NVIDIA GeForce RTX 3050 Laptop GPU |
| Mixed precision | Not used in the training loop |
| Additional-run implementation | Separate Phase 1B runner, not the primary training script; same fixed manifest and matched settings |

For code evidence, see `BrainTumorAI/train_classifier.py` around lines 189–243 and 303–317, `BrainTumorAI/classifier_pipeline.py` around lines 110–121, and `experiments/phase1b_training.py` around lines 259–285 and 324–329.

## Table: multi-seed aggregate performance

Values are mean ± sample standard deviation across runs. All seeds share one fixed test partition; these are repeated-run summaries, not independent external validation.

| Architecture | Seeds (n) | Accuracy (%) | Balanced accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) | F1 (%) | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 42–46 (5) | 98.18 ± 0.31 | 97.84 ± 0.28 | 99.12 ± 0.39 | 96.56 ± 0.25 | 98.05 ± 0.14 | 98.58 ± 0.24 | 0.9963 ± 0.0024 | 0.9974 ± 0.0017 |
| SmallCNN | 42–46 (5) | 82.21 ± 1.10 | 82.57 ± 2.55 | 81.25 ± 4.33 | 83.89 ± 9.02 | 90.15 ± 4.48 | 85.29 ± 0.86 | 0.9087 ± 0.0128 | 0.9436 ± 0.0087 |
| DenseNet-121 | 42–44 (3) | 98.51 ± 0.17 | 98.12 ± 0.14 | 99.57 ± 0.30 | 96.68 ± 0.26 | 98.12 ± 0.14 | 98.84 ± 0.13 | 0.9984 ± 0.0007 | 0.9990 ± 0.0005 |

## Table: paired matched-seed differences

Values are ResNet-18 minus comparator, mean ± sample SD. Accuracy and balanced accuracy are percentage-point differences. No p-values or significance tests were calculated.

| Comparison | Matched seeds (n) | Accuracy difference (pp) | Balanced accuracy difference (pp) | ROC-AUC difference |
|---|---|---:|---:|---:|
| ResNet-18 minus SmallCNN | 42–46 (5) | 15.97 ± 1.15 | 15.27 ± 2.74 | 0.0876 ± 0.0144 |
| ResNet-18 minus DenseNet-121 | 42–44 (3) | −0.33 ± 0.57 | −0.26 ± 0.54 | −0.0015 ± 0.0026 |

## Table: ResNet-18 duplicate-group bootstrap intervals

Intervals are percentile 95% CIs from duplicate-group resampling stratified by class. They are not patient-clustered.

| Metric | Stored: 1,000 resamples, seed 42 | Recomputed: 2,000 resamples, seed 0 |
|---|---:|---:|
| Accuracy | 96.28–100.00% | 96.01–100.00% |
| Tumor sensitivity | 98.40–100.00% | 98.37–100.00% |
| No-tumor specificity | 90.95–100.00% | 90.42–100.00% |
| ROC-AUC | 99.87–100.00% | 99.85–100.00% |

## Supplementary external-cohort sensitivity analysis

The primary manuscript result is the overlap-screened Cheng Figshare v8 subset: 72.41% sensitivity (1,223/1,689 slices from 152 patients; 95% patient-cluster bootstrap CI 66.90–77.99%) at the fixed threshold of 0.5. The unfiltered result is shown here as a secondary descriptive analysis only; it must not be interpreted as independent external validation because the overlap screen detected matching images.

| Cheng analysis | Slices / patients | Tumor sensitivity | Patient-cluster bootstrap 95% CI | Interpretation |
|---|---:|---:|---:|---|
| Full, unscreened cohort | 3,064 / 233 | 84.60% (2,592/3,064) | 80.90–88.24% | Secondary only; includes image matches to local splits |
| Primary overlap-screened subset | 1,689 / 152 | 72.41% (1,223/1,689) | 66.90–77.99% | Primary; no detected image match, but patient independence is not proven |

The exact/near-duplicate audit identified 818 Cheng slices in 846 match edges across 81 Cheng patient IDs. Excluding every patient with a match removed 1,375 slices; all excluded slices were glioma (1,375 of 1,426 glioma slices in the full cohort). The remaining subset contains 51 glioma slices from 8 patients, 708 meningioma slices from 82 patients, and 930 pituitary slices from 62 patients. Thus, the screening substantially changed subtype composition. The internal-to-screened sensitivity difference is descriptive and cannot be attributed solely to domain generalization. Cheng has no no-tumor cases, so neither analysis estimates specificity or full binary performance.

## Figures and captions

### Newly drawn, vector figures

1. [Figure 1: Study and application workflow](experiments_results/manuscript_assets/workflow.svg)  
   **Caption:** Image-level data are assigned by a fixed manifest. Training uses the training subset, checkpoint selection uses validation balanced accuracy, and final evaluation uses the held-out test subset. The application combines classification, optional qualitative Grad-CAM, and an illustrative simulated 3D display. The split is not patient-level.
2. [Figure 2: Dataset split counts](experiments_results/manuscript_assets/split_flow.svg)  
   **Caption:** Counts of folder-labeled images in the fixed training, validation, and test partitions. Patient and study identifiers were unavailable.
3. [Figure 3: Preprocessing sequence](experiments_results/manuscript_assets/preprocessing.svg)  
   **Caption:** Shared image preprocessing and training-only random horizontal flip (p = 0.5) and rotation (±10°). No crop or tumor-region selection was used.

### Existing analysis figures

4. [Figure 4: ResNet-18 reliability diagram](experiments_results/phase1c/resnet18_reliability_15bin.png)  
   **Caption:** Test-set reliability diagram using 15 equal-width bins, with bin counts. The sparse occupancy of several bins should be considered in interpretation.
5. [Figure 5: SmallCNN reliability diagram](experiments_results/phase1c/small_cnn_reliability_15bin.png)  
   **Caption:** SmallCNN test-set reliability diagram using 15 equal-width bins, with bin counts.
6. [Figure 6: Selected Grad-CAM examples](experiments_results/phase0/gradcam_examples/gradcam_grid_2x4.png)  
   **Caption:** Selected correctly and incorrectly classified images with predicted-class Grad-CAM overlays. Heatmaps are qualitative, min-max normalized per image, and are not tumor masks.
7. [Figure 7: Selected error montage](experiments_results/phase1c/requested_test_image_montage.png)  
   **Caption:** Dataset label, predicted label, and tumor score for three selected test errors. These are model outputs only and are not medical interpretations.
8. [Figure 8: Brain-inspired illustration of a simulated 3D display](experiments_results/manuscript_assets/illustrative_3d_view.svg)  
   **Caption:** Brain-inspired schematic illustrating the concept of the application's simulated 3D display. The brain shape and folds are drawn for illustration; depth and activation placement are synthetic. This is not a patient-specific reconstruction, validated localization, segmentation, or surgical plan.

## Reproducibility statement

> The primary evaluation metadata records code revision `0f4d663a327e8ab638b78e6425096a9fd84c9165` ([GitHub commit](https://github.com/stephenrodrick17-cloud/Neurovision-AI/commit/0f4d663a327e8ab638b78e6425096a9fd84c9165)). This revision is a code reference, not a complete archive of all local experiment outputs: run artifacts were stored separately, and the recorded commit does not by itself preserve those files or later working-tree changes. Per-run metrics, aggregate summaries, paired comparisons, calibration outputs, and figures are supplied separately with this manuscript material. The local dataset is not redistributed; its provenance and license remain unverified.

## Author declarations — complete only with verified information

The repository and supplied draft do not establish these declarations. Replace bracketed fields only after author/institution confirmation.

**Ethics approval:**  
> This study analyzed [identify the dataset and its source]. The [name of institutional review board or ethics committee] determined that [state the actual approval, exemption, or not-human-subjects determination], reference [approval/exemption number and date].  
> If no determination exists, do not claim exemption; obtain guidance from the responsible institution before submission.

**Informed consent:**  
> [State whether informed consent was obtained, waived by the ethics body, or not applicable, with the supporting basis.]  
> Do not infer consent status from the absence of patient identifiers in this repository.

**Funding:**  
> [List funder(s), grant number(s), and funder role.]  
> If accurate after author confirmation: “This work received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.”

**Competing interests:**  
> [Disclose actual financial or non-financial interests.]  
> If confirmed by all authors: “The authors declare no competing interests.”

**Data availability:**  
> The image files are not redistributed with this manuscript. The local collection's original source, version, and license have not been verified; therefore, availability and redistribution rights cannot currently be stated. The authors will provide a verified source citation and access information if these can be established and redistribution is permitted.

**Code and analysis outputs:**  
> The repository is available at https://github.com/stephenrodrick17-cloud/Neurovision-AI. Evaluation metadata records revision `0f4d663a327e8ab638b78e6425096a9fd84c9165`; experiment outputs are maintained separately and should be archived at a stable, citable location before publication.  
> Do not describe all experiment artifacts as archived in that commit.

## Source files

- [Phase 1B aggregate summary CSV](experiments_results/phase1b/phase1b_aggregate_summary.csv)
- [Paired seed comparison summary CSV](experiments_results/phase1b/paired_seed_comparison_summary.csv)
- [Per-seed metrics CSV](experiments_results/phase1b/phase1b_per_seed_metrics.csv)
- [SmallCNN calibration CSV](experiments_results/phase1c/small_cnn_calibration.csv)
- [ResNet-18 bootstrap JSON](experiments_results/phase1c/resnet18_group_bootstrap_2000_seed0.json)
- [Dataset provenance record](DATASET_PROVENANCE.md)
