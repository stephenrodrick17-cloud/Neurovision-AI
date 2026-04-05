# NeuroVision AI | Clinical Decision Support

NeuroVision AI is a state-of-the-art Clinical Decision Support System (CDSS) designed to assist radiologists and neurosurgeons in the detection and localization of brain tumors using deep learning.

## 🚀 Deployment Instructions

### 1. Deploy via Streamlit Community Cloud (Recommended)
This app is optimized for deployment on [Streamlit Cloud](https://streamlit.io/cloud).

1.  **Push your code** to a GitHub repository (done).
2.  **Sign in** to [Streamlit Cloud](https://share.streamlit.io/).
3.  Click **"New app"**.
4.  Select your repository (`Neurovision-AI`), the branch (`main`), and the main file path: `BrainTumorAI/app.py`.
5.  Click **"Deploy!"**.

### 2. Local Setup
If you want to run the app locally:

```bash
# Clone the repository
git clone https://github.com/stephenrodrick17-cloud/Neurovision-AI.git
cd Neurovision-AI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run BrainTumorAI/app.py
```

## 🧠 Key Features
- **Neural Diagnostic Dashboard**: Real-time tumor classification and voxel-level localization.
- **Advanced Architecture Analytics**: Deep layer activation maps and MNI-152 spatial mapping.
- **Clinical Records**: Integrated patient database and automated PDF report generation.
- **Explainable AI**: Grad-CAM visualization for clinical trust and verification.

## ⚖️ Disclaimer
This software is intended for research and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
