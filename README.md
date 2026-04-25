# 🧠 NeuroVision AI: The Future of Autonomous Neuro-Diagnostics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neurovision-ai.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-blueviolet)](https://deepmind.google/technologies/gemini/)

> **"Decoding the human brain with voxel-level precision and multi-modal AI reasoning."**

NeuroVision AI is a medical-grade Clinical Decision Support System (CDSS) that bridges the gap between deep learning and clinical practice. Built for neuro-radiologists, it combines high-fidelity MRI analysis with generative AI to provide explainable, accessible, and actionable diagnostic insights.

---

## 🌟 Visionary Features

### 1. 🔬 Neural Diagnostic Engine
*   **Voxel-Level Classification**: Powered by a fine-tuned ResNet-18 architecture, providing real-time detection of Gliomas, Meningiomas, and Pituitary tumors.
*   **Explainable AI (XAI)**: Integrated **Grad-CAM** heatmaps that highlight specific anatomical regions of interest, building clinical trust through visual audit trails.
*   **3D Spatial Visualization**: Interactive 3D brain modeling to visualize the spatial extent of detected lesions.

### 2. 🤖 Gemini AI Clinical Consultant (Multi-modal)
*   **Contextual Reasoning**: Powered by **Gemini 2.0 Flash**, the assistant analyzes scan metadata and clinical history to provide diagnostic rationales.
*   **Autonomous Therapeutic Sequences**: Automated suggestions for clinical interventions based on tumor classification and standard protocols.

### 3. 🔊 Inclusive Design (Blind Accessibility)
*   **Audio-Guided Diagnostics**: A world-first for neuro-imaging apps—Gemini generates detailed, spatially-aware audio descriptions of MRI scans for visually impaired clinicians and patients.
*   **Native TTS Integration**: Integrated Web Speech API for seamless "Read Aloud" functionality across the diagnostic lifecycle.

### 4. 📊 Advanced Architecture Analytics
*   **Neural Activation Maps**: Real-time visualization of intermediate CNN layers to understand feature extraction.
*   **MNI-152 Mapping**: Autonomous coordinate transformation into standard MNI space for research-grade accuracy.

---

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (Premium Custom CSS / Glassmorphism)
*   **Deep Learning**: PyTorch, Torchvision (ResNet-18)
*   **GenAI**: Gemini 2.0 Flash (via OpenRouter/Google GenAI)
*   **Visualization**: Plotly, OpenCV, Grad-CAM
*   **Reporting**: FPDF (Automated Medical Reports)
*   **Accessibility**: Web Speech API (TTS)

---

## 🚀 Getting Started

### 1. Clone & Configure
```bash
git clone https://github.com/stephenrodrick17-cloud/Neurovision-AI.git
cd Neurovision-AI
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_key_here
```

### 3. Installation
```bash
pip install -r requirements.txt
```

### 4. Run the Engine
```bash
streamlit run BrainTumorAI/app.py
```

---

## 📅 Roadmap: The Path to Clinical Excellence
- [ ] **Multi-sequence MRI support**: Integrating T1, T2, and FLAIR data.
- [ ] **FHIR Integration**: Direct synchronization with Electronic Health Records (EHR).
- [ ] **Edge Deployment**: On-device inference for low-resource environments.
- [ ] **Segment-Anything (SAM)**: Precision voxel-level tumor segmentation.

---

## ⚖️ Medical Disclaimer
**NeuroVision AI is a research and educational tool.** It is designed to assist, not replace, the clinical judgment of qualified medical professionals. Final diagnostic decisions must be verified by a board-certified radiologist or neurosurgeon.

---

## 👨‍💻 Developed By
**Stephen Rodrick** - *Advancing Neural AI for Humanity*
[GitHub](https://github.com/stephenrodrick17-cloud) | [LinkedIn](https://www.linkedin.com/)

---
*Built with ❤️ for the intersection of Neuroscience and Artificial Intelligence.*
