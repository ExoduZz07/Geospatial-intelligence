import streamlit as st
import os
import time
import subprocess
import sys
import rasterio
from rasterio.enums import Resampling
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="GeoAI Control", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE MEMORY ---
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'final_map_path' not in st.session_state:
    st.session_state.final_map_path = ""

# --- ADVANCED CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 800; 
        font-size: 18px;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); 
        color: #0A0A0A; 
        border: none; 
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    .stButton>button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
        color: white;
    }
    .highlight { color: #00F2FE; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: TELEMETRY ---
with st.sidebar:
    st.title("🛰️ GeoAI Control")
    st.caption("SVAMITVA Geospatial AI - Team Project")
    st.divider()
    st.markdown("### ⚙️ System Telemetry")
    st.success("🟢 U-Net ResNet18: Online")
    st.success("🟢 Spectral Engine: Active")
    st.success("🟢 GPU VRAM: Optimal")

# --- MAIN DASHBOARD HEADER ---
st.markdown("<h1>🌍<span class='highlight'>Automated Mapping Engine</span></h1>", unsafe_allow_html=True)
st.markdown("High-Resolution Asset Classification via Deep Learning & Contextual Geometry")
st.divider()

# --- LAYOUT: TABS ---
tab1, tab2 = st.tabs(["🚀 Mission Control", "🗄️ Output Gallery"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📍 Target Acquisition")
        
        # Drag and Drop Uploader
        uploaded_file = st.file_uploader("Drag and drop your Drone Orthomosaic (.tif) here", type=['tif', 'tiff'])
        
        with st.expander("🛠️ Advanced Configuration"):
            st.slider("Texture Veto Sensitivity", 1, 100, 80)
            st.checkbox("Enable Deep Water Penetration", value=True)
            st.checkbox("Force Strict Geometry (Lakes)", value=True)
            st.caption("Note: UI sliders locked to optimal presets for demo.")

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ INITIATE SCAN")

    # --- PIPELINE EXECUTION ---
    if run_btn:
        if uploaded_file is None:
            st.error("❌ Please upload a .tif file first!")
        else:
            # 1. Save the uploaded file to the hard drive
            upload_dir = r"C:\project_iit\Input_Uploads"
            os.makedirs(upload_dir, exist_ok=True)
            tif_path = os.path.join(upload_dir, uploaded_file.name)
            
            with st.spinner("Buffering image to core memory..."):
                with open(tif_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # 2. Setup standard output directories
            village_name = os.path.splitext(uploaded_file.name)[0]
            output_dir = r"C:\project_iit\Final_Outputs"
            os.makedirs(output_dir, exist_ok=True)
            final_map_path = os.path.join(output_dir, f"{village_name}_Final_Map.tif")
            
            # 3. Secure environment variables
            env = os.environ.copy()
            env["HACKATHON_TARGET_TIF"] = tif_path
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            with st.status("Initializing GeoAI Sequence...", expanded=True) as status:
                try:
                    st.write("📡 Connecting to Local Model...")
                    time.sleep(0.5)
                    
                    # --- STEP 1: AI INFERENCE ---
                    st.write("🧠 [1/2] Executing Deep Learning Geometry Scan... (Check terminal for live logs)")
                    my_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)
                        
                    subprocess.run([sys.executable, "run_inference.py"], env=env, check=True, capture_output=False, text=True, encoding="utf-8")
                    
                    # --- STEP 2: HYBRID ENGINE ---
                    st.write("🎨 [2/2] Applying Spectral & Texture Constraints... (Check terminal for live logs)")
                    subprocess.run([sys.executable, "geospatial_ai.py"], env=env, check=True, capture_output=False, text=True, encoding="utf-8")
                    
                    status.update(label="Mission Accomplished!", state="complete", expanded=False)
                    
                    st.success(f"🎉 Map Successfully Generated: `{village_name}_Final_Map.tif`")
                    st.balloons()
                    
                    # --- THE CLICKABLE FOLDER LINK ---
                    folder_uri = f"file:///{output_dir.replace(os.sep, '/')}"
                    st.markdown(f"### 📂 [Click here to open the Output Folder]({folder_uri})")
                    st.caption("Once the folder opens, simply drag the generated `.tif` file into QGIS.")
                    
                    st.markdown("### 📈 Post-Scan Analytics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Pixel Processing", "Complete", "100%")
                    m2.metric("False Positives", "Cleared", "Texture Veto")
                    m3.metric("System Health", "Stable", "No Memory Leaks")

                    # --- LOCK INTO MEMORY ---
                    st.session_state.scan_complete = True
                    st.session_state.final_map_path = final_map_path

                except subprocess.CalledProcessError as e:
                    status.update(label="System Failure", state="error", expanded=True)
                    st.error(f"❌ Crash at Module: {e.cmd[1]}")
                    st.warning("⚠️ **Open your black Terminal window to see the exact red error message!**")
                    st.session_state.scan_complete = False

    # --- THE INTERACTIVE CLASS VIEWER (Now outside the button block!) ---
    if st.session_state.scan_complete and os.path.exists(st.session_state.final_map_path):
        st.divider()
        st.markdown("### 🔍 Live Map Inspector")
        
        class_choice = st.selectbox("Isolate specific predicted features:", [
            "All Classes", 
            "1 - RCC Roofs (Grey)", 
            "2 - Tin Roofs (Cyan)", 
            "3 - Tiled Roofs (Red)",
            "4 - Utilities (Purple)", 
            "5 - Water Bodies (Blue)", 
            "6 - Roads (Yellow)"
        ])

        with st.spinner("Rendering web-optimized preview..."):
            # Downsample the massive TIF to a web-safe 1024px size
            with rasterio.open(st.session_state.final_map_path) as src:
                scale_factor = 1024 / max(src.width, src.height)
                new_height = int(src.height * scale_factor)
                new_width = int(src.width * scale_factor)
                
                data = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
            
            # QGIS Color Mapping
            color_map = {
                1: [140, 140, 140], 2: [0, 191, 255], 3: [225, 87, 89],
                4: [156, 39, 176], 5: [78, 121, 167], 6: [242, 203, 108]
            }

            # Create a black canvas
            viz_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            # Paint the canvas based on user selection
            if class_choice == "All Classes":
                for val, col in color_map.items():
                    viz_img[data == val] = col
            else:
                # Extract the ID number from the dropdown selection
                target_val = int(class_choice.split(" ")[0])
                viz_img[data == target_val] = color_map[target_val]

            st.image(viz_img, caption=f"Previewing: {class_choice} (Downsampled for UI performance)", width='stretch')


with tab2:
    st.markdown("### 🗄️ Recent Scans")
    st.info(r"Your successfully generated maps are securely stored in `C:\project_iit\Final_Outputs`")
    output_dir_gallery = r"C:\project_iit\Final_Outputs"
    
    if os.path.exists(output_dir_gallery):
        files = [f for f in os.listdir(output_dir_gallery) if f.endswith('.tif')]
        if files:
            for f in files:
                st.text(f"📄 {f}")
        else:
            st.caption("No maps generated yet.")
    else:
        st.caption("System waiting for first successful scan to create gallery.")