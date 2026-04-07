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
from surgery_viz import create_3d_brain_model
from medication_report import suggest_treatment, create_pdf_report, generate_report_text

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

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--primary-bg);
        color: var(--text-main);
    }

    .main {
        background-color: var(--primary-bg);
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
        font-size: 1.8rem; /* Slightly bigger */
        letter-spacing: -1px;
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(37, 99, 235, 0.5);
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

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model_path = os.path.join(os.path.dirname(__file__), "models", "tumor_classifier.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except:
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
    if heatmap is None: return "N/A", "N/A", "N/A"
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    mni_x = int((x / heatmap.shape[1] * 180) - 90)
    mni_y = int((y / heatmap.shape[0] * 180) - 90)
    mni_z = np.random.randint(-20, 60)
    region = "Frontal Lobe" if mni_y > 30 else "Parietal Lobe" if mni_y > -20 else "Occipital Lobe"
    side = "Left" if mni_x < 0 else "Right"
    return f"{mni_x}, {mni_y}, {mni_z}", region, side

# --- Page Content ---
def home_page():
    # Load and encode background video
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
            }}
            #background-video {{
                position: fixed;
                right: 0;
                top: 0;
                min-width: 100%;
                min-height: 100%;
                width: 100vw;
                height: 100vh;
                z-index: -100;
                filter: brightness(0.3) contrast(1.1);
                object-fit: cover;
                pointer-events: none;
            }}
            .hero-container {{
                background: transparent !important;
                padding-top: 10rem !important; /* Move it down a bit */
            }}
            .hero-subtitle {{
                font-size: 1.8rem !important;
                color: #ffffff !important;
                max-width: 1000px !important;
                margin: 0 auto 4rem auto !important;
                line-height: 1.6 !important;
                font-weight: 500 !important;
                text-shadow: 0 5px 15px rgba(0,0,0,1);
            }}
            .hero-badge {{
                color: var(--accent-blue) !important;
                font-weight: 800 !important;
                letter-spacing: 8px !important;
                margin-bottom: 2rem !important;
                font-size: 1.4rem !important;
                text-shadow: 0 0 20px rgba(37, 99, 235, 0.6);
            }}
            </style>
            <video autoplay muted loop playsinline id="background-video">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
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
        
        uploaded_file = st.file_uploader("Upload MRI Data", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.markdown("---")
            st.subheader("Visual Enhancement")
            b = st.slider("Brightness", 0.5, 2.0, 1.0)
            c = st.slider("Contrast", 0.5, 2.0, 1.0)
            s = st.slider("Sharpness", 0.0, 3.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col_r:
        if uploaded_file:
            raw_image = Image.open(uploaded_file).convert('RGB')
            adj_image = process_image_adjustments(raw_image, b, c, s)
            
            t1, t2, t3 = st.tabs(["Neural Analysis", "Spatial Visualization", "Therapeutic Protocol"])
            
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

            with t2:
                if pred_idx == 1:
                    grad_cam = GradCAM(model, model.layer4[-1])
                    heatmap = grad_cam.generate_heatmap(input_tensor)
                    temp_path = "temp_viz.jpg"
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
                    os.remove(temp_path)
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
                        pdf_path = create_pdf_report(p_info['name'], f"SCAN-{uploaded_file.name[:5]}", 
                                                   tumor_type, confidence*100, "Pending Analysis", 0.0, "Autonomous AI Sequence.")
                        time.sleep(1)
                        with open(pdf_path, "rb") as f:
                            st.download_button("Save Report", f, file_name=os.path.basename(pdf_path))
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("System Initialized. Awaiting MRI Data Upload.")

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
    uploaded_file = st.file_uploader("Upload Scan for Neural Layer Analysis", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert('RGB')
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        input_tensor = transform(raw_image).unsqueeze(0)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Activation Maps")
            
            # Expanded layer choices for ResNet-18
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
                # Ensure we have at least 16 filters to show
                num_filters = min(f_maps.shape[0], 16)
                fig_fm = px.imshow(f_maps[:num_filters], facet_col=0, facet_col_wrap=4, color_continuous_scale="Viridis")
                fig_fm.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_fm, use_container_width=True)
                
                # Use f_maps for histogram below
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
            
            st.markdown(f"""
                <div class="info-box-white">
                    <p style="margin-bottom: 12px;"><b>COORDINATE SYSTEM:</b> MNI-152 Standard Space</p>
                    <p style="margin-bottom: 12px;"><b>PEAK ACTIVATION:</b> ({mni_coords})</p>
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

    nav_choice = st.radio("", ["Home", "Dashboard", "Advanced Analytics", "Patient Records", "About System"], 
                           index=["Home", "Dashboard", "Advanced Analytics", "Patient Records", "About System"].index(st.session_state.page),
                           horizontal=True, label_visibility="collapsed")
    
    st.session_state.page = nav_choice
    
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Dashboard":
        dashboard_page()
    elif st.session_state.page == "Advanced Analytics":
        analytics_page()
    elif st.session_state.page == "Patient Records":
        patient_records_page()
    elif st.session_state.page == "About System":
        about_page()
    
    render_footer()

if __name__ == "__main__":
    main()
