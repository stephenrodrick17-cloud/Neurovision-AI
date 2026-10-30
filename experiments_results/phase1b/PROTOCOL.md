# Phase 1B protocol: multi-seed training and additional baselines

Status: Draft only. No training runs have started. This protocol is a pre-approval plan for the next phase.

## Goal

Evaluate the existing fixed split and training setup across multiple seeds and additional architectures, while keeping the experimental design tied to the repo's established rules and saved outputs.

## Controlled constraints

- Do not modify any existing project code under `BrainTumorAI/` or protected output directories.
- Do not write checkpoints into `BrainTumorAI/models/`.
- Save all run artifacts under `experiments_results/phase1b/<arch>_seed<N>/`.
- Reuse the existing training path wherever possible; add new architecture support in a new module, not by editing `BrainTumorAI/classifier_pipeline.py`.
- Keep the same split manifest, transforms, and class mapping as the repository. No test-set tuning.
- Any threshold or selection rule must be recorded before the evaluation results are seen.

## Fixed training setup

The following settings are the controlled reference setup and will be kept constant across all runs unless specifically required by the architecture:

- Manifest: `BrainTumorAI/results/split_manifest.csv`
- Data split: fixed train / validation / test manifest from the repo
- Classes: `no_tumor=0`, `tumor=1`
- Input preprocessing: RGB conversion, resize to 224x224, ToTensor, ImageNet normalization
- Training transform: resize to 224x224, random horizontal flip, random rotation (10 degrees), ToTensor, normalize
- Model selection rule: best validation balanced accuracy, not test-set accuracy or loss
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 1e-4
- Batch size: 16
- Epochs: 12
- Loss: unweighted CrossEntropy
- Early stopping: patience 4
- Scheduler: ReduceLROnPlateau(mode=max, factor=0.5, patience=2) on validation balanced accuracy
- Seed: fixed per run (42, 43, 44, 45, 46)
- Device: CUDA if available; else CPU
- Random seed handling: Python, NumPy, PyTorch, and deterministic cuDNN flags set before training

## Architectures to test

1. `resnet18` (ImageNet pretrained)
   - Using the existing repo pipeline and checkpoint convention
   - The run must check whether seed-42 reproduces the repo's original epoch-11 result

2. `small_cnn` (scratch, no pretrained weights)
   - Match the repo baseline architecture already in `BrainTumorAI/classifier_pipeline.py`

3. `densenet121` (ImageNet pretrained, new 2-class head)
   - New architecture in a new module

4. `efficientnet_b0` (ImageNet pretrained, new 2-class head)
   - New architecture in a new module

## Seeds

- 42
- 43
- 44
- 45
- 46

Every architecture will be trained for all five seeds, using the same manifest and the same evaluation policy.

## Run and artifact layout

Each run will be saved to:

`experiments_results/phase1b/<arch>_seed<N>/`

For each run, save:

- checkpoint file
- training history JSON
- metadata JSON
- environment summary (Python, torch, torchvision, CUDA availability, GPU name if available)
- per-image predictions CSV on the test split, matching the repo's test output schema as closely as possible

The artifact directory must include at least:

- `checkpoint.pth` or equivalent
- `history.json`
- `metadata.json`
- `test_predictions.csv`
- `logs/` if a background process is used

## Evaluation outputs required

After all runs are complete, produce:

- Per-run metrics table
- Mean ± SD per architecture across seeds for:
  - accuracy
  - balanced accuracy
  - sensitivity
  - specificity
  - precision
  - F1
  - ROC-AUC
  - AP
- Paired per-seed comparison of ResNet-18 vs each other architecture
- Learning-curve figure across epochs for each run
- A summary showing which epoch was selected by validation balanced accuracy
- A reproduction check for the seed-42 ResNet-18 run against the original repository result, including the difference if it does not match exactly

## Timing estimate

Reference timing from the repo logs:

- ResNet-18: 506 s per run
- Small CNN: 445 s per run

Estimated training wall time for Phase 1B:

- ResNet-18: 5 seeds × 506 s = 2,530 s ≈ 42 min
- Small CNN: 5 seeds × 445 s = 2,225 s ≈ 37 min
- DenseNet-121: estimated ~ 600 s per run; 5 seeds ≈ 3,000 s ≈ 50 min
- EfficientNet-B0: estimated ~ 600 s per run; 5 seeds ≈ 3,000 s ≈ 50 min

Total estimated GPU time for the full phase:

- Approximately 10,755 s ≈ 3.0 hours, excluding validation/evaluation and logging overhead

This estimate is intentionally conservative. It assumes one GPU and does not include idle time, repeated runs, checkpoint reloads, or debugging.

## Commands expected for this phase

The exact commands will depend on the new training module created for the additional architectures, but the intended workflow is:

1. Create a new training module under `experiments/` or a new subpackage for the extra architectures.
2. Use the existing repo data split and training logic as the source of truth.
3. For each seed and architecture:
   - train
   - save checkpoint/history
   - evaluate on the fixed test split
   - save per-image test predictions
4. Aggregate metrics into a summary table
5. Produce learning-curves and comparisons

## What will be reported before training starts

Before any training begins, the project will explicitly report:

- architecture list
- seeds used
- final artifact locations
- training duration estimate
- whether the warm-start seed-42 ResNet-18 run is expected to match the original epoch-11 result
- any discrepancies that arise after the run (without hiding them)

## Approval gate

Training will not start until this protocol is approved.

This protocol is intentionally strict and documents the exact experimental scope, output format, and constraints before the runs begin.
