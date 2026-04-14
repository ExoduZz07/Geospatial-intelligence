# 🌍 TerraScan Hub (GeoAI Engine)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([YOUR_COLAB_LINK_HERE](https://colab.research.google.com/github/ExoduZz07/Geospatial-intelligence/blob/main/Geospatial_AI.ipynb))
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white)](#)

### **High-Resolution Asset Classification via Deep Learning & Contextual Geometry**

**TerraScan Hub** is an enterprise-grade AI pipeline built for the SVAMITVA Geospatial Hackathon. It performs automated semantic segmentation to extract building footprints, road networks (paved and unpaved), and hydrology from massive drone orthomosaics. 

By fusing a **ResNet18 U-Net backbone** with a custom **Geometric Subtraction Engine**, this architecture completely eliminates the false positives and spectral confusion commonly found in rural drone telemetry.

---

## 📺 Proof of Scale (Enterprise Demo)
Because traditional cloud deployments crash when handling massive geospatial files, we engineered a serverless bypass. **Watch our pipeline process a 10GB+ map instantaneously without browser timeouts:**
👉 **[Watch the Demo video Here](YOUR_YOUTUBE_LINK_HERE)**

---

## ✨ Core Engineering Innovations

Instead of relying purely on pixel color (which fails in complex terrains), we built a custom morphological backend to teach the AI *context*.

* **♾️ Infinite-Scale Architecture:** We bypassed standard Streamlit 200MB upload limits. By running Streamlit dynamically inside a GPU-backed Google Colab and mounting Google Drive as physical storage, the UI reads 10GB+ files instantly via a secure Ngrok tunnel. Zero uploads. Zero latency.
* **🛣️ "Long Streak" Geometric Subtraction:** Rural dirt roads and plowed agricultural fields share the exact same spectral signature. To fix this, our pipeline uses a custom `cv2.MORPH_TOPHAT` algorithm to isolate massive volumetric objects (farms) and literally subtract them from the AI mask, ensuring only high-aspect-ratio linear structures (roads) survive.
* **🧹 U-Net Seam Eradication:** Processing 10GB images requires 512x512 windowed chunking. To prevent the AI from hallucinating 1-pixel borders at the edges of these chunks, we implemented a targeted median filter that instantly heals grid seams while preserving sharp building geometry.
* **🔍 Native X-Ray Inspector:** Built an interactive spatial viewer directly in the browser using `rasterio` and `cv2.addWeighted`. Judges can instantly blend the AI mask with the raw drone telemetry using an adjustable "X-Ray" overlay—no QGIS installation required.

---

## 🚀 How to Run (Cloud Demo Mode)

The absolute easiest way to test this pipeline is via our serverless Google Colab environment. No local installation or manual downloading is required.

**Step 1: Prepare the Data**
1. Create a folder in your root Google Drive called exactly: `TerraScan_Data`
2. Drop your massive `.tif` drone maps into this folder.

**Step 2: Boot the Server**
1. Click the **Open in Colab** badge at the top of this README.
2. Go to the top menu and select `Runtime` > `Run all`.
3. Allow Colab access to your Google Drive when prompted (this connects the engine to your `TerraScan_Data` folder).
4. The notebook will automatically download our pre-trained AI weights (`mopr_hybrid_shape_3050.pth`), install dependencies, and generate a secure tunnel.

**Step 3: Access the Dashboard**
1. Scroll to the bottom of the Colab notebook.
2. Click the **STABLE ACCESS LINK** (Ngrok). 
3. Select your map from the dropdown and hit **Initiate Scan**.

---

## 💻 Local Installation (For Developers)

If you prefer to run the engine locally on a machine with a dedicated Nvidia GPU:

```bash
# 1. Clone the repository
git clone [https://github.com/ExoduZz07/Geospatial-intelligence.git](https://github.com/ExoduZz07/Geospatial-intelligence.git)
cd Geospatial-intelligence

# 2. Install Geospatial & ML Dependencies
pip install -r requirements.txt
pip install segmentation-models-pytorch pyngrok gdown

# 3. Download the AI Weights
# Place 'mopr_hybrid_shape_3050.pth' in the root directory
gdown 1INNelyEwutO9QAMD_XgNgWpCALZM5fXy -O mopr_hybrid_shape_3050.pth

# 4. Launch the Engine
streamlit run app.py --server.maxUploadSize 10000
```

---

## 🗺️ Feature Extraction Legend

The final output is a QGIS-ready `.tif` raster with the following locked colormap:

| ID | Feature Class | Hex Color | UI Indicator |
| :--- | :--- | :--- | :--- |
| **1** | RCC Structures (Concrete) | `#8C8C8C` | 🟢 Grey |
| **2** | Tin Roofing | `#00BFFF` | 🔵 Cyan |
| **3** | Tiled Roofing | `#E15759` | 🔴 Red |
| **4** | Utility Infrastructure | `#9C27B0` | 🟣 Purple |
| **5** | Hydrology / Water Bodies | `#4E79A7` | 💧 Blue |
| **6** | Road Networks (Paved & Unpaved)| `#F2CB6C` | 🟡 Yellow |
