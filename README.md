# NeuroVision AI

NeuroVision AI is a Streamlit research and educational prototype for binary brain-image classification and interactive visualization. The application also contains simulated stereotactic planning, radiosurgery plots, and a surgical-task dashboard. These interface features are not medical-device functions and are not validated for patient care.

## Important limitations

- The trained classifier is binary (`no_tumor` vs `tumor`); it does not identify tumor subtypes.
- The local checkpoint evaluated during development classified all 920 images in the local test folder as tumor (399 no-tumor and 521 tumor). Image-level accuracy was 56.63%, specificity 0%, balanced accuracy 50%, and ROC-AUC 0.487. This is degenerate behavior, not evidence of a useful detector. The checkpoint and dataset are not tracked in Git; provenance and patient-level independence are unknown.
- Displayed MNI coordinates are heuristic, with a random third coordinate. They are not registered to MNI space.
- 3D, robotic trajectory, safety-score, radiation-dose, and DVH views are mathematical/UI simulations based on a 2D image and synthetic geometry. They are not clinical plans, calibrated dose calculations, or connected to surgical hardware.
- Treatment text is static, patient records are held in session memory, and report fields may contain placeholders.
- Do not use the software to diagnose, triage, or select treatment. Do not send identifiable medical images to external services without appropriate authorization and privacy review.

## Features

The application is in `BrainTumorAI/app.py` and provides:

1. **Home:** project landing view and navigation.
2. **Dashboard:** choose a sample patient, upload one JPG/PNG/JPEG image, adjust image appearance, view binary class scores, Grad-CAM overlay, simulated 3D visualization, fixed informational text, PDF report, and optional AI chat/image description.
3. **Advanced Analytics:** inspect intermediate CNN activations and heuristic peak-activation coordinates.
4. **Robotics Command Center:** browse local dataset folders or upload an image, view simulated trajectory/robot status, radiosurgery plots, and manage an in-memory surgical task queue.
5. **Patient Records:** add/list demonstration records in Streamlit session state (not persistent storage).
6. **About System:** architecture summary and disclaimer.

### Model workflow

Images are converted to RGB, resized to 224 × 224, converted to tensors, and normalized with ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. A ResNet-18 produces two logits; softmax scores are displayed. The maximum score is not calibrated and must not be interpreted as diagnostic certainty.

The model is saved at `BrainTumorAI/models/tumor_classifier.pth`. `.pth` files and datasets are ignored by Git, so a fresh clone will not contain local data or a trained checkpoint. When no usable checkpoint exists, the application attempts fallback initialization; output from a randomly initialized classification head is not meaningful.

### Simulation modules

| Module | What it implements |
|---|---|
| `BrainTumorAI/heatmap_gen.py` | Grad-CAM activation visualization. No lesion-annotation validation is included. |
| `BrainTumorAI/surgery_viz.py` | Plotly views based on image pixels and synthetic depth. |
| `BrainTumorAI/stereotactic_robot.py` | Heuristic centroid, entry point, trajectory, safety score, and inverse-kinematic calculations using assumed geometry. |
| `BrainTumorAI/radiosurgery.py` | Synthetic beam, isodose, and DVH calculations; not a treatment-planning system. |
| `BrainTumorAI/surgical_task_queue.py` | In-memory demo task lifecycle and audit-like event entries; not connected to operating-room systems. |
| `BrainTumorAI/medication_report.py` | Fixed text and PDF report generation. |
| `BrainTumorAI/gemini_assistant.py` | Optional remote OpenRouter text/image calls and browser speech synthesis. |

## Data and training

`BrainTumorAI/preprocess.py` expects this source layout (relative to the repository root):

```text
archive/Brain Tumor Data Set/Brain Tumor Data Set/
├── Brain Tumor/
└── Healthy/
```

It resizes readable images to 224 × 224 and makes an image-level 80/20 split with `random_state=42`, without stratification, patient grouping, or duplicate checking. It writes `BrainTumorAI/dataset/{train,test}/{tumor,no_tumor}/`. The original dataset citation, version, license, and patient-level composition are not documented in the repository; determine these before research use.

The current `BrainTumorAI/train_classifier.py` training configuration is:

- ResNet-18 initialized with ImageNet V1 weights; two-class output.
- Batch size 32; cross-entropy loss.
- Phase 1: freeze the backbone and train the classifier head for 5 epochs using Adam at 0.001.
- Phase 2: fine-tune all parameters for 20 epochs using Adam at 0.0001.
- Weight decay `1e-4`, `ReduceLROnPlateau` scheduling, and augmentation with horizontal flips, rotations up to 15 degrees, color jitter, and small translations.
- The script evaluates the `test` folder during training and selects the best observed test accuracy checkpoint. This uses the nominal test set for model selection, so it is not an unbiased final test evaluation.

Run preprocessing and training from the repository root after placing authorized data in the expected location:

```bash
python BrainTumorAI/preprocess.py
python BrainTumorAI/train_classifier.py
```

For a defensible study, create patient-level training, validation, and untouched test cohorts, prevent duplicate/patient leakage, record data provenance and preprocessing, seed all sources of randomness, and evaluate the final model once on the untouched test cohort. Report patient-level metrics and confidence intervals. The existing scripts do not implement this protocol.

## Installation and launch

Use Python 3.11 (the deployment files specify Python 3.11). From the repository root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run:

```bash
streamlit run BrainTumorAI/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). On first launch, Torchvision may need to download pretrained ImageNet weights if they are not cached.

## Optional Gemini/OpenRouter integration

The AI assistant uses `google/gemini-2.0-flash-001` via OpenRouter. Set a key in a root `.env` file or the process environment:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env` or embed a credential in source code. Image-description requests send image data to a remote provider. Use only data that you are authorized to transmit, and follow applicable consent, institutional, and privacy requirements.

## Deployment and repository map

- `Dockerfile`, `Procfile`, and `railpack.json` configure Streamlit deployment.
- `requirements.txt` lists Python dependencies; versions are not fully locked.
- `runtime.txt` declares Python 3.11.7.
- `BrainTumorAI/assets/` contains decorative UI media.
- `archive/`, `BrainTumorAI/dataset/`, `BrainTumorAI/models/*.pth`, generated reports, and `.env` are local/ignored artifacts and are not part of a normal Git clone.

## Research reporting checklist

Before writing scientific claims, verify and record:

- Dataset source, citation, version, license, cohort, labels, and inclusion/exclusion criteria.
- Patient-level split, duplicate checks, counts of patients and images per partition.
- Exact checkpoint hash and training run, code revision, software versions, hardware, and random seeds.
- Confusion matrix, sensitivity, specificity, precision, F1, accuracy, confidence intervals, and appropriate ROC-AUC from a held-out patient-level test set.
- Calibration and external validation if making probability or generalization claims.
- Ethics approval/waiver, consent, data governance, funding, conflicts, and author contributions.

Do not describe the prototype as providing tumor-subtype classification, MNI registration, volumetric segmentation, clinically validated dose planning, safe surgical planning, or clinical decision support based on the current implementation.

## Disclaimer

This project is for research and educational exploration only. It is not a medical device and has not been established as safe or effective for clinical use. All predictions, visualizations, generated text, and simulated plans may be incorrect. Qualified professionals must independently review any medical information; no patient-care decision should be based on this software.
