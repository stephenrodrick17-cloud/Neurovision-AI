# `paper_evidence/` index

Scripts, manifests and saved results behind the audit study described in the root [README](../README.md).
Raw images, NIfTI volumes, `*.npy` caches and `*.pth` checkpoints are **not** committed (see `.gitignore`).
Datasets and licences: [../DATASETS_AND_LICENSES.md](../DATASETS_AND_LICENSES.md).

Scripts reference the repository root and `D:/datasets/{brats,prepared/brats_oriented}`; adjust paths for your machine.
Script-to-output pairings below follow file naming; each script's docstring and its `out_path` lines are authoritative.
On Windows with the sandboxed virtualenv, run with the project interpreter (`.venv_gpu_py313`).

## Data preparation

| Script | Purpose |
|---|---|
| `prepare_oriented_brats.py` | Slices BraTS T2-FLAIR volumes to 2D images with the corrected `rot90` orientation |
| `generate_orientation_montage.py`, `generate_corrected_montage.py`, `generate_cheng_plane_check.py` | Orientation checks |
| `audit_cheng_shapes_metadata.py` | Cheng image shape/plane audit |
| `brats_dataset.py`, `brats_worker.py` | Dataset and worker helpers |

## Leakage and overlap audits

| Script | Output |
|---|---|
| `run_step0_audit.py`, `verify_leakage_claims.py`, `resolve_patient_conflict.py` | `verified_leakage_claims.json`, `inconsistency_resolution.json` |
| `run_positive_control_audit.py` | `positive_control_overlap_audit.json` |
| `run_brats_overlap_native.py`, `run_brats_overlap_corrected.py` | `brats_native_overlap_audit.json`, `phase3_brats_overlap_audit.json` |
| `run_steps_2_and_3.py`, `make_paper_tables.py` | `phase3_clean_test_rescore.json`, paper tables |

## Frozen external validation (BraTS, Cheng)

| Script | Output |
|---|---|
| `run_brats_inference.py`, `run_brats_inference_gpu.py` | `brats_*_predictions.csv.gz`, `brats_gpu_comparison_report.json` |
| `run_brats_oriented_detailed.py`, `run_brats_supplementary_cis.py`, `run_tumor_size_bins.py` | `brats_oriented_detailed_report.json`, `brats_supplementary_cIs_and_subgroups.json`, `brats_tumor_size_bins.json` |
| `run_densenet_seeds_and_250_cis.py`, `run_other_checkpoints_250.py` | `brats_250_5seeds_with_cis.json`, `brats_250_other_checkpoints.json` |
| `run_*_batch_size_effects.py`, `run_brats_10k_batch_size.py` | batch-size sensitivity JSONs |
| `internal_rotation_sensitivity.py` | `internal_rotation_sensitivity.json` |

## Leakage-free reference models

| Script | Output |
|---|---|
| `run_leakage_free_baselines.py` | `leakage_free_baselines.json`, `brats_patient_split_manifest.csv`, `leakage_free_training_settings.json` |
| `run_leakage_free_multiseed_and_gradcam.py` | `leakage_free_multiseed.json`, `leakage_free_multiseed_summary.csv`, `brats_250_leakage_free_multiseed_predictions.csv.gz`, `gradcam_brats_multiseed.json`, `logs/*_seed*.json` |
| `run_decontaminated_5seeds.py` | `decontaminated_discovery_5seeds.json`, `gradcam_brats_multiseed_size_bins.json`, `healthy_control_and_t1c_check.json` |
| `run_gradcam_brats_evaluation.py` | `gradcam_brats_evaluation.json` |

Split rule: patient split read from the saved manifest (850 train / 151 validation / 250 test patients, split seed 42,
4 tumour + 4 tumour-free slices per patient for train/validation). Only weight initialisation, data order and
augmentation depend on the training seed (43-46); seed 42 is reused as saved.

## Pre-submission analyses

| Script | Output |
|---|---|
| `run_header_only_baseline.py` | `header_only_baseline.json`, `header_only_test_predictions.csv` |
| `run_table2_bootstrap_seed0.py` | `table2_bootstrap_seed0_2000.json` (reproduces the saved phase-1c intervals exactly) |
| `run_reproducibility_606_log.py` | `discovery_606_reproducibility_log.{csv,json}` |
| `run_release_audit.py` | `release_audit_v7.json` (git revision, SHA-256 inventory, licence audit) |
| `run_items_CDE.py` | `items_CDE_report.json` |

## Figures

`figures/fig1_*.py` ... `fig7_*.py` regenerate `fig1_*.png` ... `fig7_*.png` from the files above.

## Known issues

- `brats_250_5seeds_with_cis.json` lists the primary seed 42 as well as the phase-1b seed 42 run; do not recompute
  five-seed means by averaging every `individual_runs` entry. Per-seed values are in the README table.
- `release_audit_v7.json` records the BraTS licence as "CC BY-SA 4.0 / Synapse terms"; this is **NOT VERIFIED** and is
  superseded by [../DATASETS_AND_LICENSES.md](../DATASETS_AND_LICENSES.md).
- `scratch_*.py` files are exploratory and kept only for provenance.
