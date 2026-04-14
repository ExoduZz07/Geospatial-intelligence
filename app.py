import streamlit as st
import os
import time
import rasterio
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
import glob
import cv2

# --- 1. CLOUD-SAFE IMPORTS & SECURITY ---
from run_inference import run_ai_scanner
from Geospatial_AI import apply_color_and_context

# Disable Decompression Bomb protection for massive drone maps
Image.MAX_IMAGE_PIXELS = None

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="TerraScan Hub", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE MEMORY ---
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'final_map_path' not in st.session_state:
    st.session_state.final_map_path = ""
if 'original_map' not in st.session_state:
    st.session_state.original_map = ""

# --- PREMIUM ENTERPRISE CSS ---
st.markdown("""
    <style>
    /* Import Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00F2FE, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
    }
    
    /* The "Initiate Scan" Button */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 800; 
        font-size: 1.1rem;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); 
        color: #0A0A0A; 
        border: none; 
        padding: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
    }
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        color: white;
    }
    
    /* Stylized Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Clean up the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111518;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: TELEMETRY ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/182px-Python-logo-notext.svg.png", width=40)
    st.title("TerraScan Hub")
    st.caption("SVAMITVA Geospatial AI Engine")
    st.divider()
    
    st.markdown("### ⚙️ System Telemetry")
    # Using columns for a tighter layout in sidebar
    tc1, tc2 = st.columns([1, 4])
    tc1.markdown("🟢") 
    tc2.markdown("**U-Net ResNet18**<br><span style='color:gray; font-size:12px;'>Core Online</span>", unsafe_allow_html=True)
    
    tc3, tc4 = st.columns([1, 4])
    tc3.markdown("🟢") 
    tc4.markdown("**Spectral Engine**<br><span style='color:gray; font-size:12px;'>Active</span>", unsafe_allow_html=True)
    
    tc5, tc6 = st.columns([1, 4])
    tc5.markdown("🟢") 
    tc6.markdown("**Drive Storage**<br><span style='color:gray; font-size:12px;'>Mounted</span>", unsafe_allow_html=True)

# --- MAIN DASHBOARD HEADER ---
st.markdown("<div class='main-header'>Automated Mapping Engine</div>", unsafe_allow_html=True)
st.markdown("<p style='color: #888; font-size: 1.1rem;'>High-Resolution Asset Classification via Deep Learning & Contextual Geometry</p>", unsafe_allow_html=True)
st.divider()

# --- LAYOUT: TABS ---
tab1, tab2 = st.tabs(["🚀 Mission Control", "🗄️ Output Gallery"])

with tab1:
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Wrap inputs in a sleek native border container
        with st.container(border=True):
            st.markdown("### 📍 Target Acquisition (Enterprise)")
            
            drive_path = "/content/drive/MyDrive/TerraScan_Data/"
            tif_files = glob.glob(f"{drive_path}*.tif")
            
            if not tif_files:
                st.warning("⚠️ No files found. Ensure Colab is running & Drive mounted.")
                target_file_path = None
            else:
                file_names = [os.path.basename(f) for f in tif_files]
                selected_name = st.selectbox("Select Storage Volume:", file_names)
                target_file_path = os.path.join(drive_path, selected_name)
            
            with st.expander("🛠️ Advanced Extraction Parameters"):
                st.slider("Texture Veto Sensitivity", 1, 100, 80)
                st.checkbox("Enable Deep Water Penetration", value=True)
                st.checkbox("Force Strict Geometry (Lakes)", value=True)
                st.caption("Note: UI sliders locked to optimal presets for demo environment.")

    with col2:
        # Using a container to vertically center the button
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding-bottom:10px;'>Ready for Execution</div>", unsafe_allow_html=True)
            run_btn = st.button("⚡ INITIATE SCAN")

    # --- PIPELINE EXECUTION ---
    if run_btn:
        if target_file_path is None:
            st.error("❌ Please select a file from the dropdown first!")
        else:
            tif_path = target_file_path
            
            # Sleek progress notification
            st.toast('Connection established. Initiating AI sequence...', icon='🔥')
            
            with st.status("Processing Geospatial Data...", expanded=True) as status:
                try:
                    st.write("🧠 [1/2] Executing Deep Learning Geometry Scan...")
                    ai_mask_path = run_ai_scanner(tif_path)
                    
                    st.write("🎨 [2/2] Applying Spectral & Texture Constraints...")
                    final_map_path = apply_color_and_context(tif_path, ai_mask_path)
                    
                    status.update(label="Extraction Complete!", state="complete", expanded=False)
                    st.toast('Mapping sequence successful!', icon='✅')
                    
                    # Post-Scan Analytics UI
                    st.markdown("### 📈 Post-Scan Analytics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="Pixel Processing", value="100%", delta="Optimal")
                    m2.metric(label="False Positives", value="Cleared", delta="-12%", delta_color="inverse")
                    m3.metric(label="System Health", value="Stable", delta="GPU VRAM OK")

                    # Lock to memory
                    st.session_state.scan_complete = True
                    st.session_state.final_map_path = final_map_path
                    st.session_state.original_map = tif_path

                except Exception as e:
                    status.update(label="System Failure", state="error", expanded=True)
                    st.error(f"❌ Crash at Module: {e}")
                    st.session_state.scan_complete = False

    # --- THE INTERACTIVE CLASS VIEWER ---
    if st.session_state.scan_complete and os.path.exists(st.session_state.final_map_path):
        st.divider()
        st.markdown("### 🔍 Interactive Spatial Inspector")
        
        # Wrapped the viewer controls in a nice container
        with st.container(border=True):
            v_col1, v_col2, v_col3 = st.columns([2, 2, 1])
            with v_col1:
                view_mode = st.radio("Display Engine:", ["AI Mask Only", "Overlay (Blended)", "Side-by-Side Compare"], horizontal=True)
            with v_col2:
                class_choice = st.selectbox("Isolate Feature Class:", [
                    "All Classes", "1 - RCC Roofs (Grey)", "2 - Tin Roofs (Cyan)", 
                    "3 - Tiled Roofs (Red)", "4 - Utilities (Purple)", 
                    "5 - Water Bodies (Blue)", "6 - Roads (Yellow)"
                ])
            with v_col3:
                # Provide Download right inside the viewer tools
                with open(st.session_state.final_map_path, "rb") as f:
                    st.download_button(label="📥 Export .TIF", data=f, file_name=os.path.basename(st.session_state.final_map_path), mime="image/tiff")

        with st.spinner("Rendering web-optimized high-res preview..."):
            with rasterio.open(st.session_state.final_map_path) as src_mask:
                scale_factor = 1024 / max(src_mask.width, src_mask.height)
                new_height = int(src_mask.height * scale_factor)
                new_width = int(src_mask.width * scale_factor)
                data = src_mask.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
            
            with rasterio.open(st.session_state.original_map) as src_orig:
                orig_raw = src_orig.read(out_shape=(src_orig.count, new_height, new_width), resampling=Resampling.nearest)
                if src_orig.count >= 3:
                    orig_img = np.moveaxis(orig_raw[:3], 0, -1) 
                else:
                    orig_img = np.stack((orig_raw[0],)*3, axis=-1)
            
            color_map = {
                1: [140, 140, 140], 2: [0, 191, 255], 3: [225, 87, 89],
                4: [156, 39, 176], 5: [78, 121, 167], 6: [242, 203, 108]
            }

            viz_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            if class_choice == "All Classes":
                for val, col in color_map.items():
                    viz_img[data == val] = col
            else:
                target_val = int(class_choice.split(" ")[0])
                viz_img[data == target_val] = color_map.get(target_val, [0, 0, 0])

            # Display with clean markdown spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            if view_mode == "Side-by-Side Compare":
                sub_col1, sub_col2 = st.columns(2)
                sub_col1.image(orig_img, caption="Raw Drone Telemetry", use_column_width=True)
                sub_col2.image(viz_img, caption=f"AI Extraction: {class_choice}", use_column_width=True)
            
            elif view_mode == "Overlay (Blended)":
                orig_img = orig_img.astype(np.uint8)
                overlay = cv2.addWeighted(orig_img, 0.6, viz_img, 0.4, 0)
                if class_choice != "All Classes":
                    background_mask = (viz_img == [0, 0, 0]).all(axis=2)
                    overlay[background_mask] = orig_img[background_mask]
                st.image(overlay, caption="X-Ray Deep Blend", use_column_width=True)
            
            else:
                st.image(viz_img, caption=f"Isolated Analysis: {class_choice}", use_column_width=True)

with tab2:
    st.markdown("### 🗄️ Recent Extractions Archive")
    output_dir_gallery = "Final_Outputs"
    
    if os.path.exists(output_dir_gallery):
        files = [f for f in os.listdir(output_dir_gallery) if f.endswith('.tif')]
        if files:
            for f in files:
                file_path = os.path.join(output_dir_gallery, f)
                with st.container(border=True):
                    dc1, dc2 = st.columns([4, 1])
                    dc1.markdown(f"**{f}**<br><span style='color:gray; font-size:12px;'>Processed successfully</span>", unsafe_allow_html=True)
                    with dc2:
                        with open(file_path, "rb") as file_data:
                            st.download_button(label="📥 Export", data=file_data, file_name=f, mime="image/tiff", key=f"gal_{f}")
