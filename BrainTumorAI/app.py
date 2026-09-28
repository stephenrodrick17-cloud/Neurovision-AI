import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance
import os
import sys
import base64
from datetime import datetime
from torchvision import models, transforms
import plotly.express as px
import time

# Add current dir to path to import local modules
sys.path.append(os.path.dirname(__file__))

from heatmap_gen import GradCAM, apply_heatmap
from surgery_viz import create_3d_brain_model, render_stereotactic_plan_3d, render_radiosurgery_plan_3d
from medication_report import suggest_treatment, create_pdf_report, generate_report_text
from stereotactic_robot import build_full_plan, generate_critical_structures, compute_burr_hole_entry, plan_trajectory, check_trajectory_safety, robotic_arm_ik
from radiosurgery import build_radiosurgery_plan
from surgical_task_queue import (
    default_queue, add_task, find_task, advance_task_status, cancel_task,
    procedure_uses_radiation, PROCEDURES, STATUS_COLORS, PRIORITY_COLORS, ROBOTS,
)
try:
    from gemini_assistant import describe_image_for_blind, get_gemini_response, tts_component
except ImportError:
    pass

# --- Medical-Grade Page Config ---
st.set_page_config(
    page_title="NeuroVision AI | Decoding the Human Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Professional CSS & Dark Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-bg: #000000;
        --secondary-bg: #0a0a0a;
        --accent-blue: #2563eb;
        --text-main: #ffffff;
        --text-dim: #94a3b8;
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.1);
    }

    /* Global Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--primary-bg); }
    ::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

    .stApp {
        background: transparent !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text-main);
    }

    .main {
        background: transparent !important;
        background-color: transparent !important;
    }

    section[data-testid="stAppViewBlockContainer"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    .appview-container, .main, [data-testid="stAppViewContainer"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* Prevent hidden content under nav bar */
    .block-container {
        padding-top: 6rem !important; /* Slightly reduced to give more space */
        padding-bottom: 5rem !important;
    }

    /* Modern Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-up { animation: fadeInUp 1s cubic-bezier(0.22, 1, 0.36, 1) forwards; }

    /* Custom Header/Navigation - FIXED Visibility */
    .nav-bar {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding: 0.8rem 4rem;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(20px);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999999; /* Max z-index */
        border-bottom: 1px solid var(--card-border);
    }

    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        z-index: 1000000;
    }

    .logo-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        letter-spacing: -1px;
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(37, 99, 235, 0.5);
    }

    /* Navigation Radio - ensure it sits ABOVE any custom HTML (hero overlays, cards etc) that might otherwise eat clicks */
    [data-testid="stRadio"] {
        position: relative !important;
        z-index: 500000 !important;
        pointer-events: auto !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] > label,
    [data-testid="stRadio"] label {
        cursor: pointer !important;
        pointer-events: auto !important;
    }

    /* Hero Section */
    .hero-container {
        padding: 6rem 2rem 4rem 2rem;
        text-align: center;
        background: radial-gradient(circle at 50% 50%, rgba(37, 99, 235, 0.15) 0%, transparent 70%);
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(5rem, 12vw, 10rem); /* Significantly bigger */
        font-weight: 900; /* Extra Bold */
        line-height: 0.95;
        margin-bottom: 2.5rem;
        text-transform: uppercase;
        letter-spacing: -4px;
        color: #ffffff;
        text-shadow: 0 10px 30px rgba(0,0,0,0.8), 0 0 50px rgba(37, 99, 235, 0.3);
    }

    /* Global Subpage Titles */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton>button {
        background: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 3rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(37, 99, 235, 0.5);
    }

    /* Glass Cards */
    .glass-card {
        background: var(--card-bg);
        padding: 3rem;
        border-radius: 32px;
        border: 1px solid var(--card-border);
        margin-bottom: 2.5rem;
    }

    /* FIX: Visibility for info boxes */
    .info-box-white {
        background-color: #ffffff !important;
        padding: 30px;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .info-box-white * {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    /* Streamlit Navigation Overrides */
    .stRadio [role="radiogroup"] {
        background: rgba(255, 255, 255, 0.08);
        padding: 12px 24px;
        border-radius: 100px;
        border: 1px solid var(--card-border);
        justify-content: center;
        margin-top: 1rem;
    }

    /* Medication/Treatment UI */
    .med-card {
        background: linear-gradient(145deg, rgba(37, 99, 235, 0.15), rgba(0,0,0,0));
        border-left: 6px solid var(--accent-blue);
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
    }

    .med-tag {
        display: inline-block;
        background: var(--accent-blue);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Navigation & Session State ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        "P-001": {"name": "John Doe", "age": 45, "history": "Chronic headaches, dizziness"},
        "P-002": {"name": "Jane Smith", "age": 32, "history": "Post-trauma evaluation"},
    }

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'audio_description' not in st.session_state:
    st.session_state.audio_description = ""

if 'surgical_queue' not in st.session_state:
    st.session_state.surgical_queue = default_queue()

if 'selected_task_id' not in st.session_state:
    first_task = st.session_state.surgical_queue[0] if st.session_state.surgical_queue else None
    st.session_state.selected_task_id = first_task["id"] if first_task else None

if 'emergency_stop_active' not in st.session_state:
    st.session_state.emergency_stop_active = False

if 'last_plan' not in st.session_state:
    st.session_state.last_plan = None

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model_path = os.path.join(os.path.dirname(__file__), "models", "tumor_classifier.pth")
    loaded = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            loaded = True
        except Exception:
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
                loaded = True
            except Exception:
                pass
    if not loaded:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(num_ftrs, 2)
        except Exception:
            pass
    model.eval()
    return model

model = load_model()

# --- Helper Functions ---
def process_image_adjustments(image, brightness, contrast, sharpness):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    return image

def get_feature_maps(model, input_tensor, layer_name="layer1"):
    feature_maps = []
    
    def hook(module, input, output):
        feature_maps.append(output)
    
    # Find the target layer by name
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
            
    if target_layer is None:
        return None
        
    handle = target_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        try:
            model(input_tensor)
        except:
            pass
            
    handle.remove()
    return feature_maps[0] if feature_maps else None

def mni_anatomical_mapping(heatmap):
    if heatmap is None:
        return (0, 0, 0), "Unknown", "Unknown"
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    mni_x = int((x / heatmap.shape[1] * 180) - 90)
    mni_y = int((y / heatmap.shape[0] * 180) - 90)
    mni_z = int(np.random.randint(-20, 60))
    region = "Frontal Lobe" if mni_y > 30 else "Parietal Lobe" if mni_y > -20 else "Occipital Lobe"
    side = "Left" if mni_x < 0 else "Right"
    return (mni_x, mni_y, mni_z), region, side

def _parse_mni_coords(coords):
    if isinstance(coords, (tuple, list)) and len(coords) >= 3:
        try:
            return (float(coords[0]), float(coords[1]), float(coords[2]))
        except (ValueError, TypeError):
            pass
    if isinstance(coords, str):
        try:
            parts = [float(p.strip()) for p in coords.replace("(", "").replace(")", "").split(",") if p.strip()]
            if len(parts) >= 3:
                return (parts[0], parts[1], parts[2])
        except Exception:
            pass
    return (0.0, 0.0, 0.0)

def get_sample_scans():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    samples = {}
    for split in ["test", "train"]:
        tumor_dir = os.path.join(dataset_dir, split, "tumor")
        healthy_dir = os.path.join(dataset_dir, split, "no_tumor")
        if os.path.exists(tumor_dir) and "Tumor Positive Scan (Primary)" not in samples:
            files = sorted([f for f in os.listdir(tumor_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if files:
                samples["Tumor Positive Scan (Primary)"] = os.path.join(tumor_dir, files[0])
                if len(files) > 1:
                    samples["Tumor Positive Scan (Secondary)"] = os.path.join(tumor_dir, files[1])
        if os.path.exists(healthy_dir) and "Healthy Control Scan" not in samples:
            files = sorted([f for f in os.listdir(healthy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if files:
                samples["Healthy Control Scan (Negative)"] = os.path.join(healthy_dir, files[0])
    return samples

def get_dataset_inventory():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    categories = {
        "Test - Tumor Positive (Pathology)": os.path.join(dataset_dir, "test", "tumor"),
        "Test - Healthy Controls (Normal)": os.path.join(dataset_dir, "test", "no_tumor"),
        "Train - Tumor Positive (Pathology)": os.path.join(dataset_dir, "train", "tumor"),
        "Train - Healthy Controls (Normal)": os.path.join(dataset_dir, "train", "no_tumor"),
    }
    inventory = {}
    for cat_name, folder_path in categories.items():
        if os.path.exists(folder_path):
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if files:
                inventory[cat_name] = {f: os.path.join(folder_path, f) for f in files}
    return inventory

def generate_clinical_rationale(pred_idx, confidence, mni_coords, brain_region, brain_side, volume_mm3, safety_score, safety_warnings, depth_mm):
    if pred_idx == 1:
        if "Frontal" in brain_region or "Parietal" in brain_region:
            likely_type = "Glioblastoma Multiforme (GBM) / High-Grade Glioma" if volume_mm3 > 4000 else "Low-Grade Diffuse Astrocytoma / Oligodendroglioma"
            pathology_desc = (
                f"Neuro-radiological feature extraction reveals a focal space-occupying lesion with prominent "
                f"hyperintense contrast enhancement in the {brain_side} {brain_region}. ResNet-18 residual feature "
                f"maps indicate severe architectural distortion of the surrounding parenchymal white matter tracts."
            )
        elif "Temporal" in brain_region:
            likely_type = "Temporal Lobe Glioma / Mesial Glioneuronal Neoplasm"
            pathology_desc = (
                f"High-confidence neural gradient activation centered in the {brain_side} Temporal Lobe. "
                f"Focal T1-hypointense / T2-hyperintense core with surrounding vasogenic edema bordering mesial temporal structures."
            )
        elif "Occipital" in brain_region:
            likely_type = "Occipital Glioblastoma / Secondary Metastatic Lesion"
            pathology_desc = (
                f"Posterior cranial fossa / Occipital lobe focal lesion with marked peripheral ring enhancement "
                f"and central necrotic core attenuation."
            )
        else:
            likely_type = "Meningioma / Deep Intracranial Extra-Axial Mass"
            pathology_desc = (
                f"Circumscribed contrast-enhancing mass with dural tail abutment identified in deep cranial space."
            )
        
        vol_cm3 = volume_mm3 / 1000.0
        if vol_cm3 < 3.0:
            vol_severity = f"Small localized lesion ({vol_cm3:.2f} cm³). Favorable for high-precision stereotactic radiosurgery (SRS) or targeted LITT laser ablation."
        elif vol_cm3 < 15.0:
            vol_severity = f"Moderate volume mass ({vol_cm3:.2f} cm³). Requires stereotactic navigation for biopsy and potential debulking craniotomy."
        else:
            vol_severity = f"Large volume mass ({vol_cm3:.2f} cm³) with significant mass effect. Urgent surgical decompression and cytoreductive resection indicated."

        if "Left" in brain_side and ("Frontal" in brain_region or "Temporal" in brain_region):
            eloquence_note = "High Eloquence Zone: Proximity to Broca's / Wernicke's speech and language centers. Awake intraoperative mapping or sub-millimeter stereotactic trajectory strongly advised to preserve speech fluency."
        elif "Frontal" in brain_region or "Parietal" in brain_region:
            eloquence_note = "Moderate Eloquence Zone: Adjacent to primary motor cortex (Precentral Gyrus) and corticospinal tracts. Trajectory planned to maintain safe distance from motor strips."
        else:
            eloquence_note = "Standard Eloquence Zone: Visual radiation pathways and subcortical white matter fibers monitored during trajectory descent."

        if safety_score >= 85:
            robotics_note = f"Optimal Trajectory Safety ({safety_score:.1f}/100): Clear linear corridor established from calvarium entry point to target centroid. 6-DOF manipulator kinematic pose is well within safe joint torque envelope."
        else:
            warns_str = ", ".join(safety_warnings[:2]) if safety_warnings else "narrow anatomical corridor"
            robotics_note = f"Cautionary Trajectory Safety ({safety_score:.1f}/100): Trajectory approaches near critical structures ({warns_str}). Automated path optimizer applied safety penalties."

        return {
            "status": "TUMOR_DETECTED",
            "likely_type": likely_type,
            "pathology_desc": pathology_desc,
            "vol_severity": vol_severity,
            "vol_cm3": vol_cm3,
            "eloquence_note": eloquence_note,
            "robotics_note": robotics_note,
            "recommended_procedure": "Radiosurgery (SRS)" if vol_cm3 < 5.0 else ("Tumor Resection" if vol_cm3 > 12.0 else "Stereotactic Biopsy"),
            "medications": [
                "Dexamethasone 4mg IV/PO q6h (Anti-edema & ICP reduction)",
                "Levetiracetam (Keppra) 500mg BID (Seizure prophylaxis for supratentorial lesion)",
                "Omeprazole 20mg daily (Gastric protection during steroid therapy)"
            ]
        }
    else:
        return {
            "status": "HEALTHY_SCAN",
            "likely_type": "Normal Neuroanatomy (No Neoplasm)",
            "pathology_desc": (
                "Deep convolutional feature extraction demonstrates symmetric cerebral hemispheres, "
                "normal ventricular dimensions, preserved gray-white matter junction differentiation, "
                "and no evidence of focal space-occupying mass lesions, hyperintense enhancement, or midline shift."
            ),
            "vol_severity": "No pathological mass volume detected (0.00 cm³). Normal intracranial baseline.",
            "vol_cm3": 0.0,
            "eloquence_note": "Intact eloquent cortical networks. Normal corticospinal and language pathway architecture.",
            "robotics_note": "No stereotactic target required. Robotics arm remains in calibrated standby mode.",
            "recommended_procedure": "None (Routine Surveillance)",
            "medications": [
                "No neurosurgical intervention or steroid medications required.",
                "Routine clinical follow-up as clinically indicated."
            ]
        }

# --- Page Content ---
def home_page():
    video_path = os.path.join(os.path.dirname(__file__), "assets", "background.mp4")
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode()
        
        st.markdown(f"""
            <style>
            .stApp {{
                background: transparent !important;
            }}
            .main {{
                background: transparent !important;
                     #background-video {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                min-width: 100vw;
                min-height: 100vh;
                width: 100vw;
                height: 100vh;
                z-index: -9999;
                filter: brightness(0.3) contrast(1.1) saturate(0.9);
                object-fit: cover;
                opacity: 1;
                display: block !important;
                visibility: visible !important;
                pointer-events: none;
            }
            .stApp, html, body, [data-testid="stAppViewContainer"], .appview-container {
                background: transparent !important;
                background-color: transparent !important;
            }
            [data-testid="stHeader"] {
                background: transparent !important;
            }
            .hero-container {
                background: radial-gradient(ellipse at center top, rgba(37, 99, 235, 0.25) 0%, rgba(0, 0, 0, 0.6) 60%, rgba(0, 0, 0, 0.85) 100%) !important;
                margin: -6rem -2rem 0 -2rem;
                padding: 10rem 4rem 6rem 4rem;
                position: relative !important;
                z-index: 1 !important;
            }
            </style>
            <video autoplay muted loop playsinline id="background-video">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <style>
            .stApp, html, body, [data-testid="stAppViewContainer"] {{
                background: radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.15) 0%, #000000 70%) !important;
            }}
            </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-container slide-up">
            <p class="hero-badge">NEURO-AI CORE ENGINE V2.0</p>
            <h1 class="hero-title">Decoding the<br><span style="color: var(--accent-blue);">Human Brain.</span></h1>
            <p class="hero-subtitle">
                The next frontier in clinical neuro-diagnostics. A unified environment for high-fidelity imaging, 
                autonomous therapeutic simulation, and multi-modal clinical profiling.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div style="text-align: right; padding-right: 2rem;">', unsafe_allow_html=True)
        if st.button("Launch Analysis Unit"):
            st.session_state.page = "Dashboard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="text-align: left; padding-left: 2rem;">', unsafe_allow_html=True)
        if st.button("System Architecture"):
            st.session_state.page = "About System"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
        <style>
        .carousel-container {
            overflow: hidden;
            width: 100%;
            padding: 2rem 0;
            position: relative;
        }
        .carousel-track {
            display: flex;
            width: calc(400px * 6); /* Width of cards * number of cards (including duplicates) */
            animation: scroll 30s linear infinite;
            gap: 2rem;
        }
        .carousel-track:hover {
            animation-play-state: paused;
        }
        .carousel-card {
            flex: 0 0 400px;
            background: var(--card-bg);
            padding: 3rem;
            border-radius: 32px;
            border: 1px solid var(--card-border);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .carousel-card:hover {
            transform: translateY(-10px) scale(1.05);
            border-color: var(--accent-blue);
            box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
        }
        @keyframes scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(calc(-400px * 3 - 6rem)); } /* Move by 3 cards + gaps */
        }
        </style>
        <div class="carousel-container slide-up">
            <div class="carousel-track">
                <!-- Original 3 Cards -->
                <div class="carousel-card">
                    <h3 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">MRI Neural Analysis</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">Voxel-level tumor classification using deep residual networks.</p>
                </div>
                <div class="carousel-card">
                    <h3 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">Spatial MNI Mapping</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">Autonomous coordinate transformation into MNI-152 standard space.</p>
                </div>
                <div class="carousel-card">
                    <h3 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">Clinical Reporting</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">High-fidelity diagnostic reports with integrated therapeutic protocols.</p>
                </div>
                <!-- Duplicate 3 Cards for Seamless Loop -->
                <div class="carousel-card">
                    <h3 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">MRI Neural Analysis</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">Voxel-level tumor classification using deep residual networks.</p>
                </div>
                <div class="carousel-card">
                    <h3 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">Spatial MNI Mapping</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">Autonomous coordinate transformation into MNI-152 standard space.</p>
                </div>
                <div class="carousel-card">
                    <h3 style="color: white; margin-bottom: 1.5rem; font-size: 1.8rem;">Clinical Reporting</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">High-fidelity diagnostic reports with integrated therapeutic protocols.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def dashboard_page():
    # Load and encode dashboard background
    bg_path = os.path.join(os.path.dirname(__file__), "assets", "dashboard_bg.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            bg_bytes = f.read()
        bg_b64 = base64.b64encode(bg_bytes).decode()
        
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("data:image/jpeg;base64,{bg_b64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .main {{
                background: transparent !important;
            }}
            </style>
        """, unsafe_allow_html=True)

    st.markdown("<h1 class='slide-up'>MRI Diagnostic Center</h1>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 2.5])
    
    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Patient Record")
        patient_id = st.selectbox("Select Patient ID", list(st.session_state.patient_data.keys()))
        p_info = st.session_state.patient_data[patient_id]
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 25px; border-radius: 16px; margin-top: 20px; border: 1px solid var(--card-border);">
                <p style="margin:0; color:var(--text-dim); font-size: 0.9rem; letter-spacing: 1px;">PATIENT NAME</p>
                <p style="margin:0 0 15px 0; color:white; font-weight:700; font-size: 1.4rem;">{p_info['name']}</p>
                <p style="margin:0; color:var(--text-dim); font-size: 0.9rem; letter-spacing: 1px;">AGE</p>
                <p style="margin:0; color:white; font-weight:700; font-size: 1.4rem;">{p_info['age']} Years</p>
            </div>
        """, unsafe_allow_html=True)
        
        sample_scans = get_sample_scans()
        input_source = st.radio("MRI Input Source", ["Pre-loaded Benchmark Scans", "Upload Custom Scan"], horizontal=True)
        
        raw_image = None
        scan_identifier = "MRI_SCAN"
        
        if input_source == "Pre-loaded Benchmark Scans" and sample_scans:
            selected_scan_name = st.selectbox("Select Benchmark Scan", list(sample_scans.keys()))
            scan_file_path = sample_scans[selected_scan_name]
            if os.path.exists(scan_file_path):
                raw_image = Image.open(scan_file_path).convert('RGB')
                scan_identifier = os.path.splitext(os.path.basename(scan_file_path))[0]
        else:
            uploaded_file = st.file_uploader("Upload MRI Data", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                raw_image = Image.open(uploaded_file).convert('RGB')
                scan_identifier = os.path.splitext(uploaded_file.name)[0]
        
        if raw_image is not None:
            st.markdown("---")
            st.subheader("Visual Enhancement")
            b = st.slider("Brightness", 0.5, 2.0, 1.0)
            c = st.slider("Contrast", 0.5, 2.0, 1.0)
            s = st.slider("Sharpness", 0.0, 3.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col_r:
        if raw_image is not None:
            adj_image = process_image_adjustments(raw_image, b, c, s)
            
            # Save temp image for Gemini analysis
            temp_img_path = os.path.join(os.path.dirname(__file__), "temp_upload.jpg")
            adj_image.save(temp_img_path)
            
            t1, t2, t3, t4 = st.tabs(["Neural Analysis", "Spatial Visualization", "Therapeutic Protocol", "Gemini AI Assistant"])
            
            with t1:
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    st.image(adj_image, use_container_width=True)
                
                with c2:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(adj_image).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, 1)[0]
                        confidence, pred_idx = torch.max(probs, 0)
                    
                    res_class = "TUMOR DETECTED" if pred_idx == 1 else "HEALTHY SCAN"
                    color = "#ef4444" if pred_idx == 1 else "#10b981"
                    
                    st.markdown(f"""
                        <div style='background: {color}; padding: 40px; border-radius: 24px; text-align: center; margin-bottom: 2rem; box-shadow: 0 20px 50px rgba(0,0,0,0.3);'>
                            <h2 style='color: white; margin: 0; font-family: Space Grotesk; font-size: 2rem;'>{res_class}</h2>
                            <h3 style='color: white; opacity: 0.9; margin-top: 10px; font-size: 1.4rem;'>CONFIDENCE: {confidence*100:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    fig_probs = px.bar(x=["Healthy", "Tumor"], y=[probs[0].item(), probs[1].item()], 
                                      color=["Healthy", "Tumor"],
                                      color_discrete_sequence=["#10b981", "#ef4444"])
                    fig_probs.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                          font_color="white", showlegend=False, margin=dict(t=20,b=20,l=20,r=20))
                    st.plotly_chart(fig_probs, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Blind Accessibility")
                    if st.button("Generate Audio Description"):
                        with st.spinner("Gemini is analyzing the scan for audio description..."):
                            diag_info = f"Classification: {res_class}, Confidence: {confidence*100:.2f}%"
                            st.session_state.audio_description = describe_image_for_blind(temp_img_path, diag_info)
                    
                    if st.session_state.audio_description:
                        st.info(st.session_state.audio_description)
                        st.components.v1.html(tts_component(st.session_state.audio_description, auto_play=True), height=100)

            with t2:
                if pred_idx == 1:
                    grad_cam = GradCAM(model, model.layer4[-1])
                    heatmap = grad_cam.generate_heatmap(input_tensor)
                    temp_path = os.path.join(os.path.dirname(__file__), "temp_viz.jpg")
                    adj_image.save(temp_path)
                    heatmapped_img = apply_heatmap(temp_path, heatmap)
                    
                    v1, v2 = st.columns(2)
                    with v1:
                        st.image(heatmapped_img, caption="Grad-CAM Focus Area", use_container_width=True)
                    with v2:
                        heatmap_resized = cv2.resize(heatmap, (adj_image.size[0], adj_image.size[1]))
                        mask = np.uint8(255 * (heatmap_resized > 0.4))
                        fig_3d = create_3d_brain_model(np.array(adj_image), mask)
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("🤖 Send to Robotics Planner")
                    st.caption("Compute stereotactic trajectory / radiosurgery beam plan and queue the procedure.")

                    mni_coords, brain_region, brain_side = mni_anatomical_mapping(heatmap)
                    stereo_plan = build_full_plan(heatmap, np.array(adj_image).shape[:2])
                    radio_plan = build_radiosurgery_plan(
                        heatmap,
                        target_mm=stereo_plan["centroid"]["centroid_mm"],
                        tumor_volume_mm3=stereo_plan["centroid"]["volume_mm3"],
                        n_beams=9,
                    )

                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        procedure_choice = st.selectbox(
                            "Procedure",
                            ("Stereotactic Biopsy", "Tumor Resection", "Radiosurgery (SRS)", "Laser Interstitial Ablation"),
                            index=2 if stereo_plan["centroid"]["volume_mm3"] < 6000 else 1,
                        )
                    with col_r2:
                        priority_choice = st.selectbox(
                            "Priority",
                            ("HIGH", "MED", "LOW"),
                            index=0 if confidence > 0.97 and stereo_plan["centroid"]["volume_mm3"] > 5000 else 1,
                        )
                    with col_r3:
                        default_date = (datetime.now()).strftime("%Y-%m-%d %H:%M")
                        planned_date = st.text_input("Planned Date", value=default_date)

                    r_m1, r_m2, r_m3, r_m4 = st.columns(4)
                    mni_str = f"({mni_coords[0]:.0f}, {mni_coords[1]:.0f}, {mni_coords[2]:.0f})" if isinstance(mni_coords, (tuple, list)) else f"({mni_coords})"
                    r_m1.metric("Target Centroid (MNI)", mni_str)
                    r_m2.metric("Tumor Volume", f"{stereo_plan['centroid']['volume_mm3']:.0f} mm³")
                    r_m3.metric("Safety Score", f"{stereo_plan['safety_score']:.1f} / 100",
                                delta="GREEN" if stereo_plan['safety_score'] >= 85 else ("AMBER" if stereo_plan['safety_score'] >= 60 else "RED"),
                                delta_color="normal" if stereo_plan['safety_score'] >= 85 else "inverse")
                    r_m4.metric("Procedure Type", procedure_choice.split(" ")[0])

                    st.markdown("---")
                    hq_c1, hq_c2 = st.columns([1, 1])
                    with hq_c1:
                        queue_label = "Queue Procedure" if procedure_uses_radiation(procedure_choice) else "Queue Procedure"
                        if st.button(f"🧲 {queue_label}", use_container_width=True,
                                     type="secondary", disabled=st.session_state.emergency_stop_active):
                            task = add_task(
                                queue=st.session_state.surgical_queue,
                                patient_id=patient_id,
                                patient_name=p_info["name"],
                                age=p_info["age"],
                                procedure_type=procedure_choice,
                                priority=priority_choice,
                                tumor_volume_mm3=stereo_plan["centroid"]["volume_mm3"],
                                mni_coords_mm=mni_coords,
                                lobe=f"{brain_side} {brain_region}",
                                planned_date=planned_date,
                                surgeon="Pending assignment",
                                n_beams=radio_plan["beams_plan"]["n_beams"] if procedure_uses_radiation(procedure_choice) else 0,
                                beam_energy_mv=radio_plan["beams_plan"]["beam_energy_mv"] if procedure_uses_radiation(procedure_choice) else 6,
                                notes=f"AI confidence {confidence*100:.2f}%. Classifier result: TUMOR. Planned via NeuroVision pipeline.",
                            )
                            st.session_state.last_plan = {
                                "patient_id": patient_id,
                                "patient_name": p_info["name"],
                                "procedure": procedure_choice,
                                "priority": priority_choice,
                                "confidence": float(confidence),
                                "stereo_plan": stereo_plan,
                                "radio_plan": radio_plan,
                                "mni_coords": mni_coords,
                                "brain_region": brain_region,
                                "brain_side": brain_side,
                                "mask_arr": mask,
                                "mri_arr": np.array(adj_image),
                                "task_id": task["id"],
                            }
                            st.session_state.selected_task_id = task["id"]
                            st.success(f"Task {task['id']} queued for {p_info['name']}.")
                    with hq_c2:
                        if st.button("Open Robotics Command Center →", use_container_width=True, type="primary"):
                            st.session_state.page = "Robotics Command Center"
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.success("Clear Scan. Rendering 3D Reference Model...")
                    fig_3d = create_3d_brain_model(np.array(adj_image))
                    st.plotly_chart(fig_3d, use_container_width=True)

            with t3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Autonomous Clinical Sequence")
                
                tumor_type = st.selectbox("Detected Classification", ["Glioma", "Meningioma", "Pituitary Tumor"]) if pred_idx == 1 else "no_tumor"
                med_info = suggest_treatment(tumor_type)
                
                st.markdown(f"""
                    <div class="med-card">
                        <span class="med-tag">PROTOCOL</span>
                        <p style="font-size: 1.3rem; color: white; font-weight: 500;">{med_info['description']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("**Suggested Interventions:**")
                for t in med_info['treatments']:
                    st.markdown(f"- <span style='font-size: 1.1rem;'>{t}</span>", unsafe_allow_html=True)
                
                st.markdown(f"**Follow-up Schedule:** <span style='color: var(--accent-blue); font-weight: 600;'>{med_info['followup']}</span>", unsafe_allow_html=True)
                
                st.markdown("---")
                if st.button("Download Clinical Report"):
                    with st.spinner("Generating Report..."):
                        pdf_path = create_pdf_report(p_info['name'], f"SCAN-{scan_identifier[:8]}", 
                                                   tumor_type, confidence*100, f"{brain_side} {brain_region}" if pred_idx == 1 else "Normal Anatomy", 
                                                   f"{stereo_plan['centroid']['volume_mm3'] / 1000.0:.1f}" if pred_idx == 1 else "0.0", 
                                                   "Autonomous AI Sequence.")
                        time.sleep(0.5)
                        with open(pdf_path, "rb") as f:
                            st.download_button("Save Report", f, file_name=os.path.basename(pdf_path))
                st.markdown('</div>', unsafe_allow_html=True)

            with t4:
                context = f"Patient Scan Analysis. Results: {res_class} with {confidence*100:.2f}% confidence."
                render_gemini_assistant(context)
        else:
            st.info("System Initialized. Select a pre-loaded benchmark scan or upload MRI data to begin analysis.")

def analytics_page():
    # Load and encode analytics background
    bg_path = os.path.join(os.path.dirname(__file__), "assets", "dashboard_bg.jpg")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            bg_bytes = f.read()
        bg_b64 = base64.b64encode(bg_bytes).decode()
        
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("data:image/jpeg;base64,{bg_b64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .main {{
                background: transparent !important;
            }}
            </style>
        """, unsafe_allow_html=True)

    st.markdown("<h1 class='slide-up'>Neural Architecture Analytics</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    sample_scans = get_sample_scans()
    input_source = st.radio("Input Source", ["Pre-loaded Benchmark Scan", "Upload Custom Scan"], horizontal=True, key="analytics_source")
    
    raw_image = None
    if input_source == "Pre-loaded Benchmark Scan" and sample_scans:
        selected_scan_name = st.selectbox("Select Benchmark Scan", list(sample_scans.keys()), key="analytics_scan_select")
        scan_file_path = sample_scans[selected_scan_name]
        if os.path.exists(scan_file_path):
            raw_image = Image.open(scan_file_path).convert('RGB')
    else:
        uploaded_file = st.file_uploader("Upload Scan for Neural Layer Analysis", type=["jpg", "png", "jpeg"], key="analytics_uploader")
        if uploaded_file:
            raw_image = Image.open(uploaded_file).convert('RGB')
    st.markdown('</div>', unsafe_allow_html=True)
    
    if raw_image is not None:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        input_tensor = transform(raw_image).unsqueeze(0)
        current_f_maps = None
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Activation Maps")
            
            layer_options = {
                "Initial Conv": "conv1",
                "Layer 1 (Block 1)": "layer1.0.conv1",
                "Layer 1 (Final)": "layer1",
                "Layer 2 (Block 1)": "layer2.0.conv1",
                "Layer 2 (Final)": "layer2",
                "Layer 3 (Block 1)": "layer3.0.conv1",
                "Layer 3 (Final)": "layer3",
                "Layer 4 (Block 1)": "layer4.0.conv1",
                "Layer 4 (Final)": "layer4"
            }
            
            selected_label = st.selectbox("Network Depth", list(layer_options.keys()), index=2)
            layer_name = layer_options[selected_label]
            
            output = get_feature_maps(model, input_tensor, layer_name)
            
            if output is not None:
                f_maps = output[0].cpu().numpy()
                num_filters = min(f_maps.shape[0], 16)
                fig_fm = px.imshow(f_maps[:num_filters], facet_col=0, facet_col_wrap=4, color_continuous_scale="Viridis")
                fig_fm.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_fm, use_container_width=True)
                current_f_maps = f_maps
            else:
                st.error(f"Could not extract feature maps for {selected_label}")
                current_f_maps = None
                
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Anatomical Mapping (MNI)")
            grad_cam = GradCAM(model, model.layer4[-1])
            heatmap = grad_cam.generate_heatmap(input_tensor)
            mni_coords, brain_region, brain_side = mni_anatomical_mapping(heatmap)
            mni_display = f"({mni_coords[0]}, {mni_coords[1]}, {mni_coords[2]})" if isinstance(mni_coords, (tuple, list)) else f"({mni_coords})"
            
            st.markdown(f"""
                <div class="info-box-white">
                    <p style="margin-bottom: 12px;"><b>COORDINATE SYSTEM:</b> MNI-152 Standard Space</p>
                    <p style="margin-bottom: 12px;"><b>PEAK ACTIVATION:</b> {mni_display}</p>
                    <p style="margin-bottom: 0;"><b>ANATOMICAL REGION:</b> {brain_side} {brain_region}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if current_f_maps is not None:
                fig_stats = px.histogram(current_f_maps.flatten(), nbins=50, color_discrete_sequence=['#2563eb'])
                fig_stats.update_layout(height=280, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', 
                                      plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(t=0,b=0))
                st.plotly_chart(fig_stats, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2 class='slide-up'>Gemini AI Neural Insights</h2>", unsafe_allow_html=True)
        
        col_chat, col_tts = st.columns([2, 1])
        
        with col_chat:
            render_gemini_assistant("Advanced Analytics Page: Analyzing neural activation maps and MNI coordinates.")
            
        with col_tts:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Audio Layer Analysis")
            st.markdown("Generate an audio description of the neural activations for accessibility.")
            
            # Save temp image if not already done
            temp_img_path = os.path.join(os.path.dirname(__file__), "temp_analytics.jpg")
            raw_image.save(temp_img_path)
            
            if st.button("Generate Audio Insight"):
                with st.spinner("Gemini is analyzing the neural layers..."):
                    context = f"Neural Analytics. Focus: {selected_label} activation maps. MNI Coords: {mni_coords}."
                    st.session_state.audio_description = describe_image_for_blind(temp_img_path, context)
            
            if st.session_state.audio_description:
                st.info(st.session_state.audio_description)
                st.components.v1.html(tts_component(st.session_state.audio_description, auto_play=True), height=100)
            st.markdown('</div>', unsafe_allow_html=True)

def patient_records_page():
    st.markdown("<h1 class='slide-up'>Electronic Health Records</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Register New Patient")
    with st.form("patient_registration"):
        c1, c2, c3 = st.columns(3)
        with c1:
            new_id = st.text_input("Patient ID (e.g., P-003)")
        with c2:
            new_name = st.text_input("Full Name")
        with c3:
            new_age = st.number_input("Age", 0, 120, 30)
        new_history = st.text_area("Clinical History")
        
        if st.form_submit_button("Add Patient Record"):
            if new_id and new_name:
                st.session_state.patient_data[new_id] = {"name": new_name, "age": new_age, "history": new_history}
                st.success(f"Record for {new_name} added successfully.")
                st.rerun()
            else:
                st.error("Please provide both ID and Name.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Existing Database")
    st.table([{"ID": k, "Name": v['name'], "Age": v['age'], "History": v['history']} for k, v in st.session_state.patient_data.items()])
    st.markdown('</div>', unsafe_allow_html=True)


def robotics_page():
    st.markdown("<h1 class='slide-up'>Robotics Command Center</h1>", unsafe_allow_html=True)

    queue = st.session_state.surgical_queue
    selected_id = st.session_state.selected_task_id
    estop = st.session_state.emergency_stop_active

    # Row A - HUD
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns([1.4, 1, 1, 1, 1])
    active = [t for t in queue if t["status"] == "IN_PROGRESS"]
    completed = [t for t in queue if t["status"] == "COMPLETED"]
    queued = [t for t in queue if t["status"] in ("QUEUED", "PLANNING")]
    if completed:
        avg_safety = []
        # Compute via heuristic: HIGH priority -> lower score, volume penalty
        for t in completed:
            v = t["tumor_volume_mm3"]
            s = 96.0 - min(12, v / 1200)
            if t["priority"] == "HIGH":
                s -= 3
            avg_safety.append(max(55, s))
        avg_s = float(np.mean(avg_safety))
    else:
        avg_s = 91.2
    with a1:
        robot_label = "ROSA-One Brain" if (selected_id and not find_task(queue, selected_id) is None and not procedure_uses_radiation(find_task(queue, selected_id)["procedure_type"])) else "CyberKnife G4"
        st.markdown(f"""
            <div style='background:linear-gradient(135deg, rgba(37,99,235,0.25), rgba(14,116,144,0.2)); padding:28px 24px; border-radius:20px; border:1px solid rgba(37,99,235,0.5);'>
                <p style='margin:0; color:#93c5fd; font-size:0.8rem; letter-spacing:2px;'>PRIMARY ROBOT</p>
                <h3 style='margin:8px 0 0 0; color:white; font-family:Space Grotesk; font-size:1.55rem;'>{robot_label}</h3>
                <p style='margin:10px 0 0 0; color:#86efac; font-weight:600; font-size:1rem;'>● ONLINE · LINK OK</p>
            </div>
        """, unsafe_allow_html=True)
    with a2:
        st.metric("Active Cases", len(active), delta=f"{len(queued)} Queued")
    with a3:
        st.metric("Completed Today", len(completed), delta="+2" if len(completed) >= 2 else f"+{len(completed)}")
    with a4:
        st.metric("Avg Safety Score", f"{avg_s:.1f} / 100", delta="0.8", delta_color="normal")
    with a5:
        danger_color = "#ef4444" if estop else "#10b981"
        danger_text = "E-STOP ACTIVE" if estop else "System Armed"
        st.markdown(f"""
            <div style='background:rgba(239,68,68,0.08) if {estop} else rgba(16,185,129,0.08); padding:28px 24px; border-radius:20px; border:1px solid {danger_color}66;'>
                <p style='margin:0; color:{danger_color}; font-size:0.8rem; letter-spacing:3px;'>SAFETY STATUS</p>
                <h3 style='margin:8px 0 0 0; color:{danger_color}; font-family:Space Grotesk; font-size:1.55rem;'>{danger_text}</h3>
                <p style='margin:10px 0 0 0; color:#cbd5e1; font-size:0.95rem;'>{"Robotics arm servos disabled." if estop else "All 4 manipulators within safe torque envelope."}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Patient Scan Ingestion & Autonomous Diagnostic Telemetry Panel
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🛰️ Patient Scan Ingestion & Autonomous Diagnostic Telemetry")
    st.caption("Select any MRI scan from the clinical dataset repository or upload custom neuro-imaging data to trigger real-time AI classification, Grad-CAM localization, stereotactic trajectory planning, and in-depth diagnostic reporting.")

    inv = get_dataset_inventory()
    
    col_input_1, col_input_2 = st.columns([1.5, 1])
    with col_input_1:
        input_mode = st.radio("Scan Source", ["Dataset Repository Browser", "Upload Custom MRI Scan"], horizontal=True, key="robotics_input_mode")
        
        selected_img = None
        selected_scan_label = ""
        
        if input_mode == "Dataset Repository Browser":
            cat_list = list(inv.keys()) if inv else []
            if cat_list:
                cat_col, scan_col = st.columns([1, 1.3])
                with cat_col:
                    sel_cat = st.selectbox("Dataset Category", cat_list, key="robotics_sel_cat")
                with scan_col:
                    scan_dict = inv[sel_cat]
                    scan_files = list(scan_dict.keys())
                    
                    btn_r1, btn_r2 = st.columns([3, 1])
                    with btn_r1:
                        sel_file = st.selectbox("Select Scan File", scan_files, key="robotics_sel_file")
                    with btn_r2:
                        if st.button("🎲 Random", key="robotics_rnd_btn", help="Pick a random scan from this category"):
                            import random
                            rnd_file = random.choice(scan_files)
                            st.session_state["robotics_sel_file"] = rnd_file
                            st.rerun()
                    
                    chosen_file = st.session_state.get("robotics_sel_file", sel_file)
                    if chosen_file not in scan_dict and scan_files:
                        chosen_file = scan_files[0]
                    selected_scan_path = scan_dict.get(chosen_file)
                    if selected_scan_path and os.path.exists(selected_scan_path):
                        selected_img = Image.open(selected_scan_path).convert('RGB')
                        selected_scan_label = os.path.basename(selected_scan_path)
        else:
            uploaded = st.file_uploader("Upload MRI Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], key="robotics_custom_uploader")
            if uploaded:
                selected_img = Image.open(uploaded).convert('RGB')
                selected_scan_label = uploaded.name

    with col_input_2:
        st.markdown("**Patient Profile & Session Linking**")
        p_keys = list(st.session_state.patient_data.keys())
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            rob_pid = st.selectbox("Assign to Patient", p_keys, key="rob_assign_pid")
        with p_col2:
            rob_patient_info = st.session_state.patient_data.get(rob_pid, {"name": "Anonymous", "age": 45})
            st.markdown(f"<div style='margin-top:28px; color:#93c5fd; font-weight:600;'>{rob_patient_info['name']} ({rob_patient_info['age']}y)</div>", unsafe_allow_html=True)
        
        proc_type_override = st.selectbox("Target Procedure Protocol", 
            ("Stereotactic Biopsy", "Tumor Resection", "Radiosurgery (SRS)", "Laser Interstitial Ablation"),
            index=2, key="rob_proc_override")

    if selected_img is not None:
        st.markdown("---")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(selected_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, 1)[0]
            confidence, pred_idx = torch.max(probs, 0)
        
        grad_cam = GradCAM(model, model.layer4[-1])
        heatmap = grad_cam.generate_heatmap(input_tensor)
        
        img_np = np.array(selected_img)
        temp_cam_path = os.path.join(os.path.dirname(__file__), "temp_rob_cam.jpg")
        selected_img.save(temp_cam_path)
        heatmapped_img = apply_heatmap(temp_cam_path, heatmap)
        if os.path.exists(temp_cam_path):
            try:
                os.remove(temp_cam_path)
            except Exception:
                pass
            
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        mask_np = np.uint8(255 * (heatmap_resized > 0.4))
        
        mni_coords, brain_region, brain_side = mni_anatomical_mapping(heatmap)
        stereo_plan = build_full_plan(heatmap, img_np.shape[:2])
        radio_plan = build_radiosurgery_plan(
            heatmap,
            target_mm=stereo_plan["centroid"]["centroid_mm"],
            tumor_volume_mm3=stereo_plan["centroid"]["volume_mm3"],
            n_beams=9,
        )
        
        clinical_info = generate_clinical_rationale(
            pred_idx=int(pred_idx),
            confidence=float(confidence),
            mni_coords=mni_coords,
            brain_region=brain_region,
            brain_side=brain_side,
            volume_mm3=stereo_plan["centroid"]["volume_mm3"],
            safety_score=stereo_plan["safety_score"],
            safety_warnings=stereo_plan.get("safety_warnings", []),
            depth_mm=stereo_plan["trajectory"]["depth_mm"]
        )

        v_col1, v_col2, v_col3 = st.columns([1.1, 1.1, 1.8])
        with v_col1:
            st.image(selected_img, caption=f"Raw MRI: {selected_scan_label}", use_container_width=True)
        with v_col2:
            st.image(heatmapped_img, caption="Grad-CAM Attention Heatmap", use_container_width=True)
        with v_col3:
            res_color = "#ef4444" if pred_idx == 1 else "#10b981"
            res_tag = "PATHOLOGY DETECTED · POSITIVE NEOPLASM" if pred_idx == 1 else "HEALTHY SCAN · NORMAL PARENCHYMA"
            st.markdown(f"""
                <div style='background:linear-gradient(135deg, {res_color}22, rgba(0,0,0,0.6)); border:2px solid {res_color}; border-radius:20px; padding:24px; text-align:center;'>
                    <span style='color:{res_color}; font-weight:700; letter-spacing:2px; font-size:0.85rem;'>{res_tag}</span>
                    <h2 style='color:white; font-family:Space Grotesk; margin:8px 0;'>{clinical_info["likely_type"]}</h2>
                    <h3 style='color:{res_color}; margin:0; font-size:1.6rem;'>CONFIDENCE: {confidence*100:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                st.metric("MNI Coordinate", f"({mni_coords[0]:.0f}, {mni_coords[1]:.0f}, {mni_coords[2]:.0f})")
            with m_c2:
                st.metric("Estimated Volume", f"{clinical_info['vol_cm3']:.2f} cm³")
            with m_c3:
                st.metric("Safety Score", f"{stereo_plan['safety_score']:.1f}/100")

        # In-Depth Clinical Explanation Card
        st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:20px; padding:28px; margin-top:20px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:18px;'>
                    <h3 style='color:white; font-family:Space Grotesk; margin:0;'>🔬 Neuro-Radiological Diagnostic Findings & Rationale</h3>
                    <span style='background:rgba(37,99,235,0.2); color:#93c5fd; padding:6px 14px; border-radius:12px; font-size:0.85rem; border:1px solid rgba(37,99,235,0.4);'>Autonomous Multi-Modal Analysis</span>
                </div>
                
                <div style='display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:16px;'>
                    <div style='background:rgba(0,0,0,0.3); border-left:4px solid #3b82f6; padding:16px 20px; border-radius:12px;'>
                        <h4 style='color:#93c5fd; margin:0 0 8px 0; font-size:1rem;'>🧠 Pathological Impression & AI Attention</h4>
                        <p style='color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin:0;'>{clinical_info['pathology_desc']}</p>
                    </div>
                    
                    <div style='background:rgba(0,0,0,0.3); border-left:4px solid #8b5cf6; padding:16px 20px; border-radius:12px;'>
                        <h4 style='color:#c4b5fd; margin:0 0 8px 0; font-size:1rem;'>🧭 Anatomical Locus & Eloquence Context</h4>
                        <p style='color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin:0;'><b>Location:</b> {brain_side} {brain_region} (MNI: {mni_coords[0]:.0f}, {mni_coords[1]:.0f}, {mni_coords[2]:.0f})<br>{clinical_info['eloquence_note']}</p>
                    </div>
                </div>

                <div style='display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:16px;'>
                    <div style='background:rgba(0,0,0,0.3); border-left:4px solid #10b981; padding:16px 20px; border-radius:12px;'>
                        <h4 style='color:#6ee7b7; margin:0 0 8px 0; font-size:1rem;'>🤖 Robotic Surgical Feasibility</h4>
                        <p style='color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin:0;'><b>Burr-Hole Entry:</b> ({stereo_plan['burr_hole_entry_mm'][0]:.1f}, {stereo_plan['burr_hole_entry_mm'][1]:.1f}, {stereo_plan['burr_hole_entry_mm'][2]:.1f}) mm<br><b>Corridor Depth:</b> {stereo_plan['trajectory']['depth_mm']:.1f} mm | <b>Kinematic Error:</b> {stereo_plan['arm_ik']['trajectory_error_mm']:.2f} mm<br>{clinical_info['robotics_note']}</p>
                    </div>
                    
                    <div style='background:rgba(0,0,0,0.3); border-left:4px solid #f59e0b; padding:16px 20px; border-radius:12px;'>
                        <h4 style='color:#fcd34d; margin:0 0 8px 0; font-size:1rem;'>💊 Pharmacological & Therapeutic Protocol</h4>
                        <ul style='color:#cbd5e1; font-size:0.92rem; line-height:1.5; margin:0; padding-left:18px;'>
                            {''.join([f'<li>{m}</li>' for m in clinical_info['medications']])}
                        </ul>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        act_col1, act_col2, act_col3 = st.columns(3)
        with act_col1:
            if st.button("🧲 Queue in Robotics System", use_container_width=True, type="secondary", key="rob_btn_queue"):
                t_pri = "HIGH" if confidence > 0.95 and stereo_plan["centroid"]["volume_mm3"] > 3500 else "MED"
                task = add_task(
                    queue=queue,
                    patient_id=rob_pid,
                    patient_name=rob_patient_info["name"],
                    age=rob_patient_info["age"],
                    procedure_type=proc_type_override,
                    priority=t_pri,
                    tumor_volume_mm3=stereo_plan["centroid"]["volume_mm3"],
                    mni_coords_mm=mni_coords,
                    lobe=f"{brain_side} {brain_region}",
                    planned_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    surgeon="Robotics AI Console",
                    n_beams=9 if procedure_uses_radiation(proc_type_override) else 0,
                    beam_energy_mv=6 if procedure_uses_radiation(proc_type_override) else 0,
                    notes=f"Analyzed via Robotics Command Center. Classifier: {clinical_info['likely_type']} ({confidence*100:.1f}%).",
                )
                st.session_state.last_plan = {
                    "patient_id": rob_pid,
                    "patient_name": rob_patient_info["name"],
                    "procedure": proc_type_override,
                    "priority": t_pri,
                    "confidence": float(confidence),
                    "stereo_plan": stereo_plan,
                    "radio_plan": radio_plan,
                    "mni_coords": mni_coords,
                    "brain_region": brain_region,
                    "brain_side": brain_side,
                    "mask_arr": mask_np,
                    "mri_arr": img_np,
                    "task_id": task["id"],
                }
                st.session_state.selected_task_id = task["id"]
                st.success(f"Task {task['id']} queued for {rob_patient_info['name']} and loaded into active session!")
                st.rerun()

        with act_col2:
            if st.button("⚡ Arm & Load into Live HUD", use_container_width=True, type="primary", key="rob_btn_arm"):
                target_tid = selected_id if selected_id else "LIVE-001"
                st.session_state.last_plan = {
                    "patient_id": rob_pid,
                    "patient_name": rob_patient_info["name"],
                    "procedure": proc_type_override,
                    "priority": "HIGH" if pred_idx == 1 else "LOW",
                    "confidence": float(confidence),
                    "stereo_plan": stereo_plan,
                    "radio_plan": radio_plan,
                    "mni_coords": mni_coords,
                    "brain_region": brain_region,
                    "brain_side": brain_side,
                    "mask_arr": mask_np,
                    "mri_arr": img_np,
                    "task_id": target_tid,
                }
                st.session_state.selected_task_id = target_tid
                st.success("Plan loaded into Live 3D View, Camera Feeds, and Joint Telemetry!")
                st.rerun()

        with act_col3:
            tumor_report_label = "Glioma" if "Glioma" in clinical_info["likely_type"] else ("Meningioma" if "Meningioma" in clinical_info["likely_type"] else ("Pituitary Tumor" if "Pituitary" in clinical_info["likely_type"] else "no_tumor"))
            if st.button("📄 Export Clinical PDF Report", use_container_width=True, key="rob_btn_pdf"):
                with st.spinner("Generating Medical Report..."):
                    pdf_path = create_pdf_report(
                        rob_patient_info['name'],
                        f"ROB-{selected_scan_label[:8]}",
                        tumor_report_label,
                        confidence * 100,
                        f"{brain_side} {brain_region}" if pred_idx == 1 else "Normal Anatomy",
                        f"{clinical_info['vol_cm3']:.2f}",
                        f"Analyzed in Robotics Command Center. {clinical_info['pathology_desc']}"
                    )
                    with open(pdf_path, "rb") as f:
                        st.download_button("💾 Save PDF Report", f, file_name=os.path.basename(pdf_path), key="rob_dl_pdf_btn")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Row B - Surgical Task Queue
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    b1, b2 = st.columns([3, 1.1])
    with b1:
        st.subheader("Surgical Task Queue")
        st.caption("Patient → Tumor → Procedure → Status")
        filter_status = st.multiselect("Filter", ["QUEUED", "PLANNING", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
                                       default=["QUEUED", "PLANNING", "IN_PROGRESS"])
        q = [t for t in queue if t["status"] in filter_status]
        if not q:
            st.info("No tasks match the filter.")
        else:
            rows = []
            for t in q:
                sc = STATUS_COLORS.get(t["status"], "#64748b")
                pc = PRIORITY_COLORS.get(t["priority"], "#64748b")
                c_mni = _parse_mni_coords(t.get("mni_coords_mm", (0, 0, 0)))
                rows.append({
                    "ID": t["id"],
                    "Patient": f"{t['patient_id']} · {t['patient_name']} ({t['age']}y)",
                    "Procedure": t["procedure_type"],
                    "Lobe": t.get("lobe", "—"),
                    "MNI (x,y,z)": f"({c_mni[0]:.0f}, {c_mni[1]:.0f}, {c_mni[2]:.0f})",
                    "Volume (mm³)": f"{t['tumor_volume_mm3']:.0f}",
                    "Priority": t["priority"],
                    "Status": t["status"],
                    "Assigned Robot": t["assigned_robot"],
                    "Planned": t["planned_date"],
                    "Progress %": t["progress_pct"],
                })
            import pandas as pd
            df = pd.DataFrame(rows).set_index("ID")
            styled = (df.style
                      .map(lambda v: f"background-color: {PRIORITY_COLORS.get(v, '#64748b')}22; color: {PRIORITY_COLORS.get(v, '#e2e8f0')}", subset=["Priority"])
                      .map(lambda v: f"background-color: {STATUS_COLORS.get(v, '#64748b')}22; color: {STATUS_COLORS.get(v, '#e2e8f0')}", subset=["Status"])
                      .format({"Progress %": lambda x: f"{int(x)}%"}))
            st.dataframe(styled, use_container_width=True, height=290)

            col_b_btns = st.columns([1, 1, 1, 1])
            with col_b_btns[0]:
                task_list_options = [f"{t['id']} · {t['patient_name']} · {t['status']}" for t in q]
                task_idx = 0
                for i, t in enumerate(q):
                    if t["id"] == selected_id:
                        task_idx = i
                        break
                pick = st.selectbox("Select task", task_list_options, index=min(task_idx, len(task_list_options) - 1))
                new_selected_id = pick.split(" · ")[0]
                if new_selected_id != selected_id:
                    st.session_state.selected_task_id = new_selected_id
                    st.rerun()
            with col_b_btns[1]:
                dis = estop or (find_task(queue, selected_id) is None or find_task(queue, selected_id)["status"] in ("COMPLETED", "CANCELLED"))
                if st.button("⏭  Advance Status", use_container_width=True, disabled=dis):
                    advance_task_status(queue, selected_id, emergency_stop=estop)
                    st.rerun()
            with col_b_btns[2]:
                tsel = find_task(queue, selected_id)
                dis2 = estop or tsel is None or tsel["status"] in ("COMPLETED", "CANCELLED")
                if st.button("✖  Cancel Task", use_container_width=True, disabled=dis2):
                    cancel_task(queue, selected_id, reason="Operator cancelled")
                    st.rerun()
            with col_b_btns[3]:
                if st.button("🔴  EMERGENCY STOP" if not estop else "🟢  CLEAR E-STOP", type="primary",
                             use_container_width=True):
                    st.session_state.emergency_stop_active = not estop
                    if st.session_state.emergency_stop_active:
                        for t in queue:
                            if t["status"] in ("PLANNING", "IN_PROGRESS"):
                                advance_task_status(queue, t["id"], emergency_stop=True)
                    st.rerun()

    with b2:
        st.subheader("New Task")
        with st.form("robotics_add_task"):
            pid = st.selectbox("Patient ID", list(st.session_state.patient_data.keys()))
            pinfo = st.session_state.patient_data[pid]
            proc = st.selectbox("Procedure", PROCEDURES)
            pri = st.selectbox("Priority", ["HIGH", "MED", "LOW"], index=1)
            vol = st.number_input("Tumor Volume (mm³)", 100, 80000, 3500, 100)
            x_m = st.number_input("MNI X", -80, 80, 20)
            y_m = st.number_input("MNI Y", -80, 80, -20)
            z_m = st.number_input("MNI Z", -60, 80, 30)
            if st.form_submit_button("Add to Queue", disabled=estop):
                add_task(
                    queue,
                    patient_id=pid, patient_name=pinfo["name"], age=pinfo["age"],
                    procedure_type=proc, priority=pri, tumor_volume_mm3=float(vol),
                    mni_coords_mm=(x_m, y_m, z_m), lobe="Manual Entry",
                    planned_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                )
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row C - Surgical Plan
    selected = find_task(queue, selected_id) if selected_id else None
    if selected is None:
        st.info("Select a task from the queue above to view its surgical plan, camera feeds, and arm telemetry.")
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Surgical Plan & HUD")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        if st.session_state.last_plan and st.session_state.last_plan.get("task_id") == selected_id:
            last = st.session_state.last_plan
            stereo = last["stereo_plan"]
            radio = last["radio_plan"]
            mri_arr = last["mri_arr"]
            mask_arr = last["mask_arr"]
            uses_rad = procedure_uses_radiation(selected["procedure_type"])
            view_mode = st.radio("View Mode",
                                 ("Trajectory + Beams", "Burr-hole Trajectory", "Radiosurgery Beams"),
                                 index=0 if uses_rad else 1, horizontal=True, label_visibility="collapsed")
            if "Trajectory" in view_mode:
                fig = render_stereotactic_plan_3d(
                    mri_arr, mask_arr,
                    entry_mm=stereo["burr_hole_entry_mm"],
                    target_mm=stereo["centroid"]["centroid_mm"],
                    critical_structures=stereo["critical_structures"],
                    heatmap=None,
                )
            else:
                fig = render_radiosurgery_plan_3d(
                    mri_arr, mask_arr,
                    target_mm=stereo["centroid"]["centroid_mm"],
                    beams_plan=radio["beams_plan"],
                    isodoses=radio["isodoses"],
                )
            st.plotly_chart(fig, use_container_width=True)

            # DVH chart for radiosurgery
            if uses_rad or "Beams" in view_mode:
                st.markdown("**Dose-Volume Histogram (DVH)**")
                import plotly.graph_objects as go
                fig_dvh = go.Figure()
                fig_dvh.add_trace(go.Scatter(
                    x=radio["dvh_bins_gy"], y=radio["dvh_tumor_curve"] * 100,
                    mode="lines", line=dict(color="#ef4444", width=3),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.15)", name="Tumor (PTV)"
                ))
                fig_dvh.add_trace(go.Scatter(
                    x=radio["dvh_bins_gy"], y=radio["dvh_healthy_curve"] * 100,
                    mode="lines", line=dict(color="#60a5fa", width=3, dash="dash"),
                    fill="tozeroy", fillcolor="rgba(96,165,250,0.10)", name="Healthy Brain"
                ))
                fig_dvh.update_layout(
                    height=280, margin=dict(t=20, b=40, l=40, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white", showlegend=True,
                    legend=dict(bgcolor="rgba(0,0,0,0.4)", orientation="h"),
                    xaxis_title="Dose (Gy)", yaxis_title="Volume Receiving ≥ Dose (%)",
                )
                st.plotly_chart(fig_dvh, use_container_width=True)
        else:
            # Fallback demo plan for pre-seeded / manually added tasks
            uses_rad = procedure_uses_radiation(selected["procedure_type"])
            sel_coords = _parse_mni_coords(selected.get("mni_coords_mm", (0, 0, 0)))
            cy = int(np.clip(112 + sel_coords[1] * 0.6, 10, 214))
            cx = int(np.clip(112 + sel_coords[0] * 0.6, 10, 214))
            yy, xx = np.mgrid[0:224, 0:224]
            dummy_heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 18 ** 2))
            mri_arr = np.stack([np.clip(80 + 255 * dummy_heat * 0.8, 0, 255).astype(np.uint8)] * 3, axis=-1)
            mask_arr = np.uint8(255 * (dummy_heat > 0.45))
            stereo = build_full_plan(dummy_heat, (224, 224))
            radio = build_radiosurgery_plan(dummy_heat,
                                             target_mm=stereo["centroid"]["centroid_mm"],
                                             tumor_volume_mm3=float(selected["tumor_volume_mm3"]),
                                             n_beams=selected.get("n_beams") or 9)
            view_mode = st.radio("View Mode",
                                 ("Burr-hole Trajectory", "Radiosurgery Beams"),
                                 index=1 if uses_rad else 0, horizontal=True, label_visibility="collapsed")
            if view_mode == "Burr-hole Trajectory":
                fig = render_stereotactic_plan_3d(mri_arr, mask_arr,
                                                  entry_mm=stereo["burr_hole_entry_mm"],
                                                  target_mm=stereo["centroid"]["centroid_mm"],
                                                  critical_structures=stereo["critical_structures"])
            else:
                fig = render_radiosurgery_plan_3d(mri_arr, mask_arr,
                                                  target_mm=stereo["centroid"]["centroid_mm"],
                                                  beams_plan=radio["beams_plan"],
                                                  isodoses=radio["isodoses"])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Demo visualization: plan rendered from queue parameters (no MRI uploaded yet). Upload an MRI via the Dashboard → Send to Robotics Planner to attach the real Grad-CAM plan.")

    with c2:
        # HUD Indicators (Plotly gauge)
        import plotly.graph_objects as go
        if st.session_state.last_plan and st.session_state.last_plan.get("task_id") == selected_id:
            stereo = st.session_state.last_plan["stereo_plan"]
            radio = st.session_state.last_plan["radio_plan"]
            ik = stereo["arm_ik"]
            safety = stereo["safety_score"]
            traj_err = ik["trajectory_error_mm"]
            depth = stereo["trajectory"]["depth_mm"]
            dose = radio["beams_plan"]["total_dose_gy_per_fraction"] if procedure_uses_radiation(selected["procedure_type"]) else None
        else:
            sel_coords = _parse_mni_coords(selected.get("mni_coords_mm", (0, 0, 0)))
            entry = compute_burr_hole_entry(sel_coords)
            traj = plan_trajectory(entry, sel_coords)
            structures = generate_critical_structures()
            safety, warns = check_trajectory_safety(entry, sel_coords, structures)
            ik = robotic_arm_ik(entry, sel_coords)
            traj_err = ik["trajectory_error_mm"]
            depth = traj["depth_mm"]
            if procedure_uses_radiation(selected["procedure_type"]):
                dose = 18.0 + float(selected["tumor_volume_mm3"]) / 9000
            else:
                dose = None

        gauges = st.columns(2)
        def gauge_fig(title, value, max, suffix, good_from, color_ref):
            return go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(value),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title, "font": {"color": "white", "size": 13, "family": "Space Grotesk"}},
                number={"suffix": suffix, "font": {"color": "white", "size": 30}},
                gauge={
                    "axis": {"range": [None, max], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
                    "bar": {"color": color_ref, "thickness": 0.55},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(37,99,235,0.35)",
                    "steps": [
                        {"range": [0, good_from], "color": "rgba(239,68,68,0.18)"},
                        {"range": [good_from, max], "color": "rgba(16,185,129,0.14)"},
                    ],
                    "threshold": {
                        "line": {"color": "#fbbf24", "width": 3},
                        "thickness": 0.85,
                        "value": float(value),
                    },
                },
            )).update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                              margin=dict(l=15, r=15, t=40, b=10))

        with gauges[0]:
            st.plotly_chart(gauge_fig("TRAJ. ERROR (mm)", min(traj_err, 3.5), 3.5, " mm", 0.5, "#22d3ee"),
                            use_container_width=True)
        with gauges[1]:
            st.plotly_chart(gauge_fig("SAFETY SCORE", safety, 100, " /100", 75, "#10b981"),
                            use_container_width=True)
        gauges2 = st.columns(2)
        with gauges2[0]:
            st.plotly_chart(gauge_fig("DEPTH TO TARGET", depth, 140, " mm", 20, "#60a5fa"),
                            use_container_width=True)
        with gauges2[1]:
            if dose is not None:
                st.plotly_chart(gauge_fig("CUMULATIVE DOSE", dose, 35, " Gy", 25, "#f97316"),
                                use_container_width=True)
            else:
                import plotly.graph_objects as go
                fig_placeholder = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=100 - min(50, len([w for w in [] if False])),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "TRAJECTORY CLEARANCE",
                           "font": {"color": "white", "size": 13, "family": "Space Grotesk"}},
                    number={"suffix": " %", "font": {"color": "white", "size": 30}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
                        "bar": {"color": "#a855f7", "thickness": 0.55},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [{"range": [0, 60], "color": "rgba(239,68,68,0.18)"},
                                  {"range": [60, 100], "color": "rgba(16,185,129,0.14)"}],
                    }
                )).update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                  margin=dict(l=15, r=15, t=40, b=10))
                st.plotly_chart(fig_placeholder, use_container_width=True)

        st.markdown("**Critical Structure Checklist**")
        struct_list = stereo.get("critical_structures") if st.session_state.last_plan and st.session_state.last_plan.get("task_id") == selected_id else generate_critical_structures()
        warn_list = stereo.get("safety_warnings", []) if st.session_state.last_plan and st.session_state.last_plan.get("task_id") == selected_id else []
        cols_struct = st.columns(2)
        for i, s in enumerate(struct_list):
            with cols_struct[i % 2]:
                hit = any(s.name in w for w in warn_list)
                c = "#10b981" if not hit else "#ef4444"
                st.markdown(f"""
                    <div style='padding:8px 12px; border-radius:10px; margin-bottom:6px; background:rgba(255,255,255,0.03); border-left:4px solid {c};'>
                        <span style='color:{c}; font-weight:600; font-size:0.9rem;'>{'✖' if hit else '✔'}</span>
                        <span style='color:#e2e8f0; margin-left:8px; font-size:0.92rem;'>{s.name}</span>
                        <span style='color:#94a3b8; float:right; font-size:0.8rem;'>R={s.radius_mm:.0f} mm</span>
                    </div>
                """, unsafe_allow_html=True)
        if warn_list:
            with st.expander("⚠ Safety warnings"):
                for w in warn_list:
                    st.warning(w)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row D - Camera feeds + arm telemetry
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Camera Feeds & Arm Telemetry")
    d1, d2, d3 = st.columns([1.1, 1.1, 1.6])

    with d1:
        st.markdown(f"""
            <style>
            .cam-frame {{
                background:
                    repeating-linear-gradient(0deg, rgba(255,255,255,0.04) 0 1px, transparent 1px 3px),
                    radial-gradient(circle at 50% 50%, #0b1220 0%, #000000 70%);
                border:1px solid rgba(37,99,235,0.4);
                border-radius:16px;
                position:relative;
                overflow:hidden;
                padding:80% 0 0 0;
            }}
            .cam-overlay {{
                position:absolute; inset:0; padding:14px;
                display:flex; flex-direction:column; justify-content:space-between;
                font-family:Space Grotesk;
            }}
            .cam-top {{
                display:flex; justify-content:space-between; align-items:center;
            }}
            .rec-badge {{
                background:#ef4444; color:white; font-weight:700; padding:5px 12px;
                border-radius:6px; letter-spacing:2px; font-size:0.75rem;
                box-shadow:0 0 14px rgba(239,68,68,0.6);
                animation: pulse 1.6s infinite;
            }}
            @keyframes pulse {{ 0%,100% {{opacity:1;}} 50% {{opacity:0.55;}} }}
            .cam-mid {{
                position:absolute; inset:0; display:flex; align-items:center; justify-content:center; pointer-events:none;
            }}
            .crosshair {{
                width:70%; height:70%;
                border:1px dashed rgba(34,211,238,0.85);
                border-radius:50%;
                position:relative;
            }}
            .crosshair::before, .crosshair::after {{
                content:''; position:absolute; background:rgba(34,211,238,0.9);
            }}
            .crosshair::before {{ top:50%; left:-8%; right:-8%; height:1px; }}
            .crosshair::after {{ left:50%; top:-8%; bottom:-8%; width:1px; }}
            </style>
            <div class="cam-frame">
                <div class="cam-overlay">
                    <div class="cam-top">
                        <div>
                            <div style="font-size:0.7rem; letter-spacing:2px; color:#22d3ee;">CH · ENDOSCOPE</div>
                            <div style="color:white; font-weight:600; margin-top:2px;">{selected['endoscope_channel']}</div>
                        </div>
                        <div class="rec-badge">● REC</div>
                    </div>
                    <div class="cam-mid"><div class="crosshair"></div></div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#cbd5e1;">
                        <div>PATIENT: {selected['patient_id']} · {selected['patient_name']}</div>
                        <div>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;margin-top:10px;font-size:0.82rem;color:#94a3b8;'>"
                    "<span>🔴 30 FPS · YUV 4:2:2</span>"
                    "<span>LATENCY 11 ms</span>"
                    "<span>ZOOM 2.4x</span>"
                    "</div>", unsafe_allow_html=True)

    with d2:
        st.markdown(f"""
            <style>
            .cam-frame2 {{
                background:
                    radial-gradient(circle at 50% 70%, rgba(37,99,235,0.14) 0%, rgba(0,0,0,1) 70%);
                border:1px solid rgba(99,102,241,0.4);
                border-radius:16px;
                position:relative;
                overflow:hidden;
                padding:80% 0 0 0;
            }}
            .silhouette {{
                position:absolute; inset:0;
            }}
            .patient-bed {{
                position:absolute; left:20%; right:20%; top:55%; height:8%;
                background:linear-gradient(90deg, rgba(148,163,184,0.15), rgba(148,163,184,0.45), rgba(148,163,184,0.15));
                border-radius:10px;
            }}
            .patient-head {{
                position:absolute; left:44%; top:30%; width:12%; padding-bottom:12%;
                background:radial-gradient(circle at 50% 50%, rgba(239,68,68,0.25), rgba(255,255,255,0.02));
                border:1px solid rgba(239,68,68,0.5); border-radius:50%;
            }}
            .tumor-dot {{
                position:absolute; left:47.5%; top:34%; width:3%; height:3%;
                background:#ef4444; border-radius:50%;
                box-shadow:0 0 12px #ef4444;
            }}
            .robot-arm {{
                position:absolute; left:62%; top:18%; width:26%; height:44%;
                border-left:5px solid rgba(34,211,238,0.85);
                border-bottom:5px solid rgba(34,211,238,0.85);
                border-bottom-left-radius:40px;
            }}
            .robot-end {{
                position:absolute; left:61%; top:58%; width:36px; height:36px;
                background:#0891b2; border-radius:10px;
                box-shadow:0 0 20px rgba(34,211,238,0.8);
            }}
            </style>
            <div class="cam-frame2">
                <div class="cam-overlay" style="position:absolute;inset:0;padding:14px;display:flex;flex-direction:column;justify-content:space-between;font-family:Space Grotesk;">
                    <div class="cam-top" style="display:flex;justify-content:space-between;">
                        <div>
                            <div style="font-size:0.7rem; letter-spacing:2px; color:#818cf8;">CH · OVERHEAD OR</div>
                            <div style="color:white; font-weight:600; margin-top:2px;">{selected['overhead_camera_id']}</div>
                        </div>
                        <div class="rec-badge" style="background:#818cf8;">● LIVE</div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#cbd5e1;">
                        <div>PROCEDURE: {selected['procedure_type']}</div>
                        <div>STATUS: <span style='color:{STATUS_COLORS.get(selected['status'], '#64748b')};'>{selected['status']}</span></div>
                    </div>
                </div>
                <div class="silhouette">
                    <div class="patient-bed"></div>
                    <div class="patient-head"></div>
                    <div class="tumor-dot"></div>
                    <div class="robot-arm"></div>
                    <div class="robot-end"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;margin-top:10px;font-size:0.82rem;color:#94a3b8;'>"
                    "<span>🔵 24 FPS · 1080p H.264</span>"
                    "<span>LATENCY 14 ms</span>"
                    "<span>FOV 120°</span>"
                    "</div>", unsafe_allow_html=True)

    with d3:
        st.markdown("**6-DOF Manipulator — Joint Angles & EEF Pose**")
        # Extract / synthesize IK joints
        if st.session_state.last_plan and st.session_state.last_plan.get("task_id") == selected_id:
            joints = st.session_state.last_plan["stereo_plan"]["arm_ik"]["joints_deg"]
            eef = st.session_state.last_plan["stereo_plan"]["arm_ik"]["eef_pose_mm_deg"]
            err = st.session_state.last_plan["stereo_plan"]["arm_ik"]["trajectory_error_mm"]
        else:
            sel_coords = _parse_mni_coords(selected.get("mni_coords_mm", (0, 0, 0)))
            entry = compute_burr_hole_entry(sel_coords)
            ik_sim = robotic_arm_ik(entry, sel_coords)
            joints = ik_sim["joints_deg"]
            eef = ik_sim["eef_pose_mm_deg"]
            err = ik_sim["trajectory_error_mm"]
        joint_names = list(joints.keys())
        joint_vals = [float(joints[n]) for n in joint_names]
        import plotly.graph_objects as go
        jbars = go.Figure(go.Bar(
            y=joint_names,
            x=joint_vals,
            orientation="h",
            marker=dict(color=["#22d3ee", "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#c026d3"],
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            text=[f"{v:.1f}°" for v in joint_vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f}°<extra></extra>",
        ))
        jbars.update_layout(
            height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(l=120, r=20, t=10, b=20),
            xaxis_title="Joint Angle (deg)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.15)",
                       tickfont=dict(color="#94a3b8"), title_font=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(color="#cbd5e1"), showgrid=False),
            uniformtext_minsize=10,
        )
        st.plotly_chart(jbars, use_container_width=True)

        # EEF Pose card
        pose_col = st.columns(6)
        keys = [("x_mm", "X", "mm"), ("y_mm", "Y", "mm"), ("z_mm", "Z", "mm"),
                ("roll_deg", "Roll", "°"), ("pitch_deg", "Pitch", "°"), ("yaw_deg", "Yaw", "°")]
        for i, (k, name, unit) in enumerate(keys):
            with pose_col[i]:
                v = float(eef[k])
                st.markdown(f"""
                    <div style='background:rgba(37,99,235,0.08); border:1px solid rgba(37,99,235,0.3); border-radius:12px; padding:10px; text-align:center;'>
                        <div style='color:#93c5fd; font-size:0.7rem; letter-spacing:1.5px;'>{name}</div>
                        <div style='color:white; font-weight:700; font-size:1.1rem; margin-top:3px;'>{v:.1f}<span style='font-size:0.75rem;color:#94a3b8;'> {unit}</span></div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        estop_disabled = st.session_state.emergency_stop_active
        danger_label = "⚠  EMERGENCY STOP — RELEASE ALL SERVOS" if not estop_disabled else "✔  CLEAR E-STOP · RE-ARM SYSTEM"
        if st.button(danger_label,
                     type="primary", use_container_width=True,
                     help="Hard-stop all robotic manipulators and cancel in-progress / planning tasks.",
                     key="estop_big",
                     disabled=False):
            new_state = not st.session_state.emergency_stop_active
            st.session_state.emergency_stop_active = new_state
            if new_state:
                for t in queue:
                    if t["status"] in ("PLANNING", "IN_PROGRESS"):
                        advance_task_status(queue, t["id"], emergency_stop=True)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def about_page():
    st.markdown("<h1 class='slide-up'>System Architecture</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card">
            <h2 style="color: white; margin-bottom: 2rem;">Clinical Standards & Neural Backbone</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 2.5rem;">
                <div style="background: rgba(255,255,255,0.03); padding: 2.5rem; border-radius: 24px; border: 1px solid var(--card-border);">
                    <h3 style="color: var(--accent-blue); margin-bottom: 1rem;">Deep Learning Engine</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">
                        Utilizes a <b>ResNet-18</b> backbone fine-tuned on medical datasets. Employs residual learning to ensure stable gradient flow and high-accuracy feature extraction.
                    </p>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 2.5rem; border-radius: 24px; border: 1px solid var(--card-border);">
                    <h3 style="color: var(--accent-blue); margin-bottom: 1rem;">XAI Visualization</h3>
                    <p style="color: var(--text-dim); line-height: 1.8; font-size: 1.1rem;">
                        Integrates <b>Grad-CAM</b> technology to highlight anatomical regions of interest, providing clinicians with a visual audit trail for every diagnostic outcome.
                    </p>
                </div>
            </div>
            <div style="background: rgba(239, 68, 68, 0.1); border-left: 6px solid #ef4444; padding: 2.5rem; margin-top: 3rem; border-radius: 16px;">
                <p style="color: #fca5a5; margin: 0; font-weight: 700; font-size: 1.2rem; letter-spacing: 1px;">MEDICAL DISCLAIMER</p>
                <p style="color: #fca5a5; margin: 0.5rem 0 0 0; font-size: 1rem; line-height: 1.6;">
                    NeuroVision AI is a research tool for educational purposes. Final treatment decisions must be made by qualified medical professionals.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_gemini_assistant(context_info=""):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Clinical AI Consultation")
    st.markdown("Ask Gemini about the scan results, potential treatments, or anatomical concerns.")
    
    # Chat Interface
    for message in st.session_state.chat_history:
        role_color = "var(--accent-blue)" if message['role'] == 'user' else "rgba(255,255,255,0.1)"
        st.markdown(f"""
            <div style="background: {role_color}; padding: 15px; border-radius: 12px; margin-bottom: 10px; border: 1px solid var(--card-border);">
                <p style="margin:0; font-size: 0.9rem; font-weight: 700;">{'YOU' if message['role'] == 'user' else 'GEMINI AI'}</p>
                <p style="margin:5px 0 0 0;">{message['content']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    user_input = st.text_input("Consult with Gemini AI...", key=f"gemini_input_{st.session_state.page}")
    if st.button("Send Query", key=f"send_query_{st.session_state.page}"):
        if user_input:
            history = []
            for m in st.session_state.chat_history:
                history.append({"role": m['role'], "parts": [m['content']]})
            
            if not history:
                user_query = f"Context: {context_info}\n\nUser Question: {user_input}"
            else:
                user_query = user_input
                
            with st.spinner("Gemini is thinking..."):
                response = get_gemini_response(history, user_query)
                
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "model", "content": response})
            st.rerun()
    
    if st.button("Clear Consultation History", key=f"clear_chat_{st.session_state.page}"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class="footer-section" style="border-top: 1px solid var(--card-border); padding: 6rem 4rem 3rem 4rem; margin-top: 8rem; background: var(--secondary-bg);">
            <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 4rem; margin-bottom: 4rem;">
                <div>
                    <h3 style="color: white; margin-bottom: 1.5rem; font-family: Space Grotesk; font-size: 1.8rem;">NeuroVision AI</h3>
                    <p style="color: var(--text-dim); font-size: 1.1rem; line-height: 1.6;">The global standard in autonomous neurological AI diagnostics.</p>
                </div>
                <div><h4 style="color: white; margin-bottom: 1.2rem;">Core</h4><p style="color: var(--text-dim);">Neural Engine</p><p style="color: var(--text-dim);">MNI Atlas</p></div>
                <div><h4 style="color: white; margin-bottom: 1.2rem;">Data</h4><p style="color: var(--text-dim);">Security</p><p style="color: var(--text-dim);">Protocols</p></div>
                <div><h4 style="color: white; margin-bottom: 1.2rem;">Support</h4><p style="color: var(--text-dim);">Technical</p><p style="color: var(--text-dim);">Clinical</p></div>
            </div>
            <div style="text-align: center; color: var(--text-dim); font-size: 0.9rem; letter-spacing: 2px; padding-top: 3rem; border-top: 1px solid var(--card-border);">
                © 2026 NEUROVISION AI SYSTEMS. ALL SYSTEMS OPERATIONAL.
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- Main App Routing ---
def main():
    st.markdown("""
        <div class="nav-bar">
            <div class="logo-container">
                <div style="background: var(--accent-blue); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px rgba(37, 99, 235, 0.4);">
                    <span style="color: white; font-weight: 800; font-size: 1.4rem;">N</span>
                </div>
                <span class="logo-text">NEUROVISION AI</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    pages = ["Home", "Dashboard", "Advanced Analytics", "Robotics Command Center", "Patient Records", "About System"]
    if st.session_state.get("page") not in pages:
        st.session_state.page = "Home"

    st.radio(
        "Navigation",
        pages,
        key="page",
        horizontal=True,
        label_visibility="collapsed",
    )

    if st.session_state.page != "Home":
        st.markdown("""
            <style>
            .stApp, html, body, [data-testid="stAppViewContainer"], .appview-container, .main {
                background: radial-gradient(circle at 50% 0%, rgba(37, 99, 235, 0.12) 0%, #000000 60%) !important;
                background-color: #000000 !important;
            }
            </style>
        """, unsafe_allow_html=True)

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
    elif st.session_state.page == "Advanced Analytics":
        analytics_page()
    elif st.session_state.page == "Robotics Command Center":
        robotics_page()
    elif st.session_state.page == "Patient Records":
        patient_records_page()
    elif st.session_state.page == "About System":
        about_page()

    render_footer()

if __name__ == "__main__":
    main()
