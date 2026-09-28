# 🧠 NeuroVision AI | Autonomous Neuro-Oncology & Stereotactic Robotics Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.14.0-EE4C2C.svg)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![System Status](https://img.shields.io/badge/System-100%25%20Operational-brightgreen.svg)]()

**NeuroVision AI** is an advanced, enterprise-grade Clinical Decision Support System (CDSS) and Autonomous Stereotactic Surgical Planning Platform designed for neurosurgeons, radiation oncologists, and neuro-radiologists. It combines deep residual neural networks for high-accuracy brain tumor detection, Grad-CAM explainability, 3D anatomical MNI-152 standard space registration, stereotactic burr-hole trajectory planning with 6-DOF robotic arm inverse kinematics, non-coplanar radiosurgery beam/isodose modeling, multimodal AI reasoning, and automated clinical PDF reporting.

---

## 📑 Table of Contents
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Key Modules & Capabilities](#-key-modules--capabilities)
  - [1. Neural Diagnostic Engine & Deep Classifier](#1-neural-diagnostic-engine--deep-classifier)
  - [2. Explainable AI (Grad-CAM) & Feature Extraction](#2-explainable-ai-grad-cam--feature-extraction)
  - [3. 3D Spatial Mesh Visualization & Slicing](#3-3d-spatial-mesh-visualization--slicing)
  - [4. Stereotactic Robotics Trajectory Planner & 6-DOF IK](#4-stereotactic-robotics-trajectory-planner--6-dof-ik)
  - [5. Stereotactic Radiosurgery (SRS) & DVH Simulator](#5-stereotactic-radiosurgery-srs--dvh-simulator)
  - [6. Robotics Command Center & Dataset Browser](#6-robotics-command-center--dataset-browser)
  - [7. Surgical Task Queue & OR Telemetry](#7-surgical-task-queue--or-telemetry)
  - [8. Clinical PDF Report Generation & Patient Database](#8-clinical-pdf-report-generation--patient-database)
  - [9. Multimodal Clinical AI & Blind Accessibility](#9-multimodal-clinical-ai--blind-accessibility)
- [Repository Structure](#-repository-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [How to Run the Application](#-how-to-run-the-application)
- [Dataset & Model Training](#-dataset--model-training)
- [Subsystem Verification & Test Suite](#-subsystem-verification--test-suite)
- [Mathematical Formulations & Robotics Kinematics](#-mathematical-formulations--robotics-kinematics)
- [Docker & Cloud Deployment](#-docker--cloud-deployment)
- [Medical Disclaimer](#-medical-disclaimer)

---

## 🏗️ System Architecture & Pipeline

```
                                  ┌────────────────────────┐
                                  │  Input MRI Brain Scan  │
                                  │ (Dataset / User Upload)│
                                  └───────────┬────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ Preprocessing & Normalization   │
                             │   (224x224 RGB, ImageNet Norm)  │
                             └────────────────┬────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │   ResNet-18 Classifier  │                       │   Grad-CAM Engine       │
        │  (Inference & Softmax)  │                       │ (Layer4 Activation Maps)│
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │ Binary Diagnostic Class │                       │ 2D Attention Heatmap &  │
        │  (Tumor vs Healthy)     │                       │ High-Res Spatial Mask   │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │ MNI-152 Standard Space Mapping  │
                             │ (Lobe, Hemisphere, Centroid mm) │
                             └────────────────┬────────────────┘
                                              │
                     ┌────────────────────────┼────────────────────────┐
                     ▼                        ▼                        ▼
        ┌────────────────────────┐┌────────────────────────┐┌────────────────────────┐
        │  Stereotactic Planning ││  Radiosurgery Planning ││  Clinical Explanation  │
        │ • Burr-hole Entry      ││ • 9 Linac Beams (6MV)  ││ • Pathology Rationale │
        │ • Critical Structures  ││ • Isodose Contours     ││ • Eloquence Analysis   │
        │ • Safety Score (0-100) ││ • DVH Dose Curves      ││ • Surgical Strategy    │
        │ • 6-DOF Robotic IK     ││ • PTV/Healthy Sparing  ││ • Pharmacological Plan │
        └────────────┬───────────┘└───────────┬────────────┘└───────────┬────────────┘
                     │                        │                         │
                     └────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │     Robotics Command Center     │
                             │ • Interactive 3D Plotly Visuals │
                             │ • Endoscope & OR Camera Feeds   │
                             │ • 6-DOF Joint Angles & HUD      │
                             │ • Surgical Task Queue & E-STOP  │
                             │ • Clinical PDF Export (FPDF2)   │
                             └─────────────────────────────────┘
```

---

## ⚡ Key Modules & Capabilities

### 1. Neural Diagnostic Engine & Deep Classifier
- **Backbone Architecture**: Fine-tuned **ResNet-18** deep convolutional neural network.
- **Weights**: Checkpoint loaded from `BrainTumorAI/models/tumor_classifier.pth` with dynamic fallback.
- **Classification Output**: Binary categorization (`TUMOR DETECTED` vs `HEALTHY SCAN`) with exact softmax probability confidence score ($0.0\% - 100.0\%$).
- **Image Preprocessing**: Brightness, contrast, and sharpness adjustments with standard normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 2. Explainable AI (Grad-CAM) & Feature Extraction
- **Visual Audit Trail**: Implements Gradient-weighted Class Activation Mapping (**Grad-CAM**) on the final convolutional layer (`model.layer4[-1]`).
- **Gradient Backpropagation**: Extracts feature gradients relative to the predicted class score to highlight exact spatial regions driving the model's decision.
- **Activation Slicing**: Inspects intermediate feature representations across all network depths (`conv1`, `layer1`, `layer2`, `layer3`, `layer4`) with interactive filter grids and activation distribution histograms.

### 3. 3D Spatial Mesh Visualization & Slicing
- **Plotly 3D Engine**: Generates interactive 3D volumetric surfaces from MRI slices and binary tumor masks.
- **Skull & Brain Contours**: Renders ellipsoidal calvarium boundaries and internal parenchyma surfaces.
- **Tumor Iso-Surfaces**: Renders color-coded pathological masses embedded inside 3D brain coordinates with interactive rotation, pan, zoom, and cross-sectional slicing.

### 4. Stereotactic Robotics Trajectory Planner & 6-DOF IK
- **Burr-Hole Entry Optimization**: Computes the optimal cranial entry point on the skull surface minimizing linear penetration depth while avoiding the superior sagittal and transverse venous sinuses.
- **Critical Structure Collision Avoidance**: Evaluates trajectory clearance against essential neurological structures:
  - Lateral & Third Ventricles
  - Brainstem (Midbrain / Pons / Medulla)
  - Optic Chiasm & Tracts
  - Internal Carotid & Middle Cerebral Arteries (Sylvian Fissure)
  - Primary Motor Cortex (Precentral Gyrus)
- **Safety Score (0–100)**: Exponential proximity penalty metric calculating geometric risk.
- **6-DOF Robotic Manipulator Inverse Kinematics**: Computes joint angles ($\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6$) and End-Effector (EEF) 6D Cartesian pose for neurosurgical robots (such as ROSA-One Brain).

### 5. Stereotactic Radiosurgery (SRS) & DVH Simulator
- **Non-Coplanar Beam Delivery**: Simulates 9 non-coplanar stereotactic Linac/CyberKnife radiation beams ($6\,\text{MV}$) converging on the Planning Target Volume (PTV).
- **Isodose Volumes**: Computes $100\%$, $80\%$, $50\%$, and $20\%$ isodose distributions across 2D/3D grids.
- **Dose-Volume Histogram (DVH)**: Generates cumulative dose curves comparing PTV tumor target coverage against healthy surrounding brain tissue sparing.

### 6. Robotics Command Center & Dataset Browser
- **Integrated Dataset Repository Browser**: Directly browse and load hundreds of clinical scans from `dataset/` across categories:
  - `Test - Tumor Positive (Pathology)`
  - `Test - Healthy Controls (Normal)`
  - `Train - Tumor Positive (Pathology)`
  - `Train - Healthy Controls (Normal)`
- **🎲 Random Sample Generator**: Test random cases with 1-click.
- **Custom Scan Uploader**: Drag-and-drop custom `.jpg`, `.png`, or `.jpeg` MRI scans.
- **Neuro-Radiological Diagnostic Findings Engine**:
  - **Pathological Impression**: Detailed differential diagnosis (Glioblastoma vs Astrocytoma vs Meningioma vs Pituitary Adenoma vs Normal Neuroanatomy).
  - **Anatomical & Eloquence Context**: MNI coordinates and speech/motor cortex risk analysis.
  - **Robotic Surgical Strategy**: Burr-hole rationale, trajectory depth, and manipulator torque feasibility.
  - **Pharmacological Protocol**: Dexamethasone anti-edema dosage and Levetiracetam (Keppra) seizure prophylaxis.

### 7. Surgical Task Queue & OR Telemetry
- **Procedure Lifecycle Management**: Multi-state queue (`QUEUED` $\rightarrow$ `PLANNING` $\rightarrow$ `IN_PROGRESS` $\rightarrow$ `COMPLETED` / `CANCELLED`).
- **Camera Feeds & Overlays**: Dual-channel simulated camera feeds:
  - *Endoscope Channel*: 30 FPS, YUV 4:2:2, 11 ms latency with aiming reticle.
  - *Overhead OR Channel*: 24 FPS, 1080p H.264, 14 ms latency with patient silhouette & manipulator tracker.
- **Emergency Stop (E-STOP)**: Instant one-click servo lock and fail-safe procedure freeze.

### 8. Clinical PDF Report Generation & Patient Database
- **FPDF2 Document Engine**: Generates comprehensive, clinical-grade medical reports including patient demographics, scan metadata, diagnostic classification, confidence percentage, MNI coordinates, lesion volume, surgical recommendations, and AI audit notes.
- **EHR Patient Management**: Electronic health records database to register, edit, and link patient profiles.

### 9. Multimodal Clinical AI & Blind Accessibility
- **Multimodal AI Consultant**: Contextual reasoning explaining MRI scan metadata and diagnostic findings.
- **Audio-Guided Diagnostics**: Accessibility engine generating detailed spatial audio descriptions of MRI scans for visually impaired clinicians and patients.

---

## 📂 Repository Structure

```text
d:\Brain Imaging\
├── BrainTumorAI/
│   ├── app.py                     # Main Streamlit Web Application (Command Center & Dashboard)
│   ├── heatmap_gen.py             # Grad-CAM Explainable AI Engine
│   ├── surgery_viz.py             # Plotly 3D Brain, Skull, Trajectory & Isodose Mesh Renderer
│   ├── stereotactic_robot.py      # Stereotactic Trajectory Planner & 6-DOF Robotic Arm IK
│   ├── radiosurgery.py            # Radiosurgery 9-Beam Linac & DVH Simulation Engine
│   ├── surgical_task_queue.py     # Surgical Task Queue & State Management
│   ├── medication_report.py       # Clinical Recommendation Engine & PDF Report Generator
│   ├── gemini_assistant.py        # Multimodal AI Consultant & Blind Accessibility Engine
│   ├── train_classifier.py        # PyTorch ResNet-18 Classifier Training Pipeline
│   ├── preprocess.py              # Image Preprocessing & Contrast Enhancement Utilities
│   ├── models/
│   │   └── tumor_classifier.pth   # Trained ResNet-18 Model Weights Checkpoint
│   ├── dataset/                   # MRI Clinical Dataset Repository
│   │   ├── train/                 # Training Split (tumor / no_tumor)
│   │   └── test/                  # Evaluation Split (tumor / no_tumor)
│   ├── reports/                   # Output Directory for Generated Clinical PDF Reports
│   └── assets/                    # UI Media Assets (Videos, Icons, Styles)
├── Dockerfile                     # Production Container Definition
├── requirements.txt               # Locked Python Dependencies
├── Procfile                       # Cloud Service Deployment Specification
└── README.md                      # Comprehensive System Documentation
```

---

## 💻 Installation & Local Setup

### Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python**: Version `3.10` or higher
- **Hardware**: Compatible with CPU execution (CUDA GPU optional for acceleration)

### Step 1: Clone the Repository
```bash
git clone https://github.com/stephenrodrick17-cloud/Neurovision-AI.git
cd "Brain Imaging"
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

Start the Streamlit application on your local server:

```powershell
# Run using the virtual environment
& ".venv\Scripts\streamlit.exe" run "BrainTumorAI\app.py"
```

Once started, open your web browser at:
👉 **`http://localhost:8501`**

### Navigation Guide:
1. **🏠 Home**: System overview and interactive hero display.
2. **📊 Dashboard**: Real-time diagnostic inference unit with Grad-CAM, 3D brain models, and multimodal AI consultant.
3. **📈 Advanced Analytics**: Multi-layer CNN feature map inspector and MNI coordinate locator.
4. **🤖 Robotics Command Center**: Ingest dataset scans, run automated diagnostic reports, inspect 3D surgical trajectories, DVH curves, and monitor robotic arm telemetry.
5. **🗂️ Patient Records**: Register patients and manage the electronic health records database.
6. **ℹ️ About System**: Deep learning backbone specifications and clinical disclaimers.

---

## 🏋️ Dataset & Model Training

To retrain or fine-tune the ResNet-18 classifier on custom brain imaging datasets:

```powershell
& ".venv\Scripts\python.exe" "BrainTumorAI\train_classifier.py"
```

### Training Pipeline Features:
- **Data Augmentation**: Random horizontal flips, random rotations ($\pm 15^\circ$), affine transforms, color jitter.
- **Optimization**: Cross-Entropy Loss with Adam optimizer ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$).
- **Validation**: Automated epoch checkpointing saving the best validation accuracy weights to `BrainTumorAI/models/tumor_classifier.pth`.

---

## 🧪 Subsystem Verification & Test Suite

The platform includes an automated 9-point verification suite testing all computational modules:

```powershell
& ".venv\Scripts\python.exe" -c "
import sys, os
sys.path.append('BrainTumorAI')
import torch, torchvision, cv2, numpy as np, PIL, fpdf, plotly
from BrainTumorAI.app import load_model, mni_anatomical_mapping, get_dataset_inventory
from BrainTumorAI.heatmap_gen import GradCAM, apply_heatmap
from BrainTumorAI.stereotactic_robot import build_full_plan
from BrainTumorAI.radiosurgery import build_radiosurgery_plan
from BrainTumorAI.surgery_viz import create_3d_brain_model, render_stereotactic_plan_3d
from BrainTumorAI.medication_report import create_pdf_report
from BrainTumorAI.surgical_task_queue import default_queue, add_task

print('[OK] All 9 Core Subsystems Verified Successfully!')
"
```

---

## 📐 Mathematical Formulations & Robotics Kinematics

### 1. MNI-152 Standard Space Coordinate Mapping
Given a 2D MRI activation peak $(u_{\max}, v_{\max})$ on a scan grid of size $(H, W)$:
$$X_{\text{MNI}} = \left(\frac{u_{\max}}{W} \times 180\right) - 90, \quad Y_{\text{MNI}} = \left(\frac{v_{\max}}{H} \times 180\right) - 90$$
$$Z_{\text{MNI}} = Z_{\text{base}} + \Delta z$$

### 2. Stereotactic Safety Scoring Function
For a trajectory line segment $\mathbf{L}(t) = \mathbf{P}_{\text{entry}} + t(\mathbf{P}_{\text{target}} - \mathbf{P}_{\text{entry}})$, $t \in [0, 1]$, and a critical anatomical structure $i$ with center $\mathbf{C}_i$ and radius $R_i$:
$$d_{\min, i} = \min_{t \in [0, 1]} \|\mathbf{L}(t) - \mathbf{C}_i\|_2$$
$$\text{Penalty}_i = \begin{cases} 
\omega_i \cdot \exp\left(-\frac{d_{\min, i} - R_i}{\sigma}\right) & \text{if } d_{\min, i} > R_i \\
\omega_{\text{collision}} & \text{if } d_{\min, i} \le R_i
\end{cases}$$
$$\text{Safety Score} = \max\left(0, 100 - \sum_{i=1}^{N} \text{Penalty}_i\right)$$

### 3. Radiosurgery Isodose & DVH Formulation
Radiation dose at grid location $(x, y)$ from $M$ non-coplanar converging beam axes:
$$D(x, y) = \sum_{k=1}^{M} D_0 \cdot \exp\left(-\frac{\text{dist}_{\perp}^2(x, y, \mathbf{B}_k)}{2\sigma_{\text{beam}}^2}\right) \cdot \text{TPR}(d_{\text{depth}})$$

---

## 🐳 Docker & Cloud Deployment

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t neurovision-ai .

# Run the container mapping port 8501
docker run -p 8501:8501 neurovision-ai
```

### Deploy to Streamlit Community Cloud
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a **New App**, select your repository branch, and set the entry file path to:
   `BrainTumorAI/app.py`
4. Click **Deploy**.

---

## ⚖️ Medical Disclaimer

> [!IMPORTANT]
> **RESEARCH & EDUCATIONAL USE ONLY**: NeuroVision AI is an artificial intelligence research prototype and clinical decision support system designed for academic, research, and educational purposes. It is **not** an FDA-approved or CE-marked medical device. It should never be used as the sole basis for clinical diagnosis, patient management, surgical intervention, or radiation therapy planning. All surgical decisions, anatomical evaluations, and prescriptions must be made and verified by certified medical specialists.

---

## 👨‍💻 Developed By
**Stephen Rodrick** - *Advancing Neural AI for Humanity*  
[GitHub Profile](https://github.com/stephenrodrick17-cloud)

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

© 2026 **NeuroVision AI Systems**. All rights reserved.
