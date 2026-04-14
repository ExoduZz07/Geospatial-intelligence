# TERRASCAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red?style=flat-square)](YOUR_YOUTUBE_LINK_HERE)

Geospatial semantic segmentation pipeline. Extracts structural and hydrological features from high-resolution orthomosaics via ResNet18 U-Net and morphological post-processing. Designed for arbitrary-scale inputs without memory exhaustion.

---

## ARCHITECTURE

* **I/O & Memory Management:** Mounts external drives for zero-copy read access. Bypasses standard HTTP payload limits via asynchronous Ngrok tunneling. Infers over 10GB+ rasters using 512x512 moving windows (`rasterio.windows.Window`) to maintain strict VRAM limits.
* **Inference Engine:** PyTorch ResNet18 U-Net backbone trained on rural geospatial telemetry.
* **Morphological Subtraction:** Resolves spectral confusion between unpaved networks and agricultural soil. Applies OpenCV top-hat transforms (`cv2.MORPH_TOPHAT`) to isolate and subtract volumetric artifacts, enforcing high-aspect-ratio geometry for linear infrastructure.
* **Artifact Eradication:** Spatial median filtering dynamically heals 1-pixel chunking seams at window boundaries prior to mask reconstruction.
* **Compositing:** Native matrix blending (`cv2.addWeighted`) renders real-time transparency overlays in the control dashboard.

---

## DEPLOYMENT: CLOUD RUNTIME

Executes in a serverless GPU environment. Zero local configuration required.

1. Initialize external storage: Create `TerraScan_Data` at the root of a Google Drive volume.
2. Transfer target `.tif` binaries into this directory.
3. Execute the `Open In Colab` directive.
4. Mount storage when prompted by the kernel.
5. Execute `Runtime > Run all`.
6. Access the generated Ngrok tunnel port via the stdout console.

---

## DEPLOYMENT: LOCAL BUILD

Requires an active CUDA environment and Python 3.8+.

```bash
# Clone source
git clone [https://github.com/ExoduZz07/Geospatial-intelligence.git](https://github.com/ExoduZz07/Geospatial-intelligence.git)
cd Geospatial-intelligence

# Resolve dependencies
pip install -r requirements.txt
pip install segmentation-models-pytorch pyngrok gdown

# Fetch model weights
gdown 1INNelyEwutO9QAMD_XgNgWpCALZM5fXy -O mopr_hybrid_shape_3050.pth

# Initialize server
streamlit run app.py --server.maxUploadSize 10000
## Feature Extraction

The final output is a QGIS-ready `.tif` raster with the following locked colormap:

| ID | Feature Class | Hex Color | UI Indicator |
| :--- | :--- | :--- | :--- |
| **1** | RCC Structures (Concrete) | `#8C8C8C` | 🟢 Grey |
| **2** | Tin Roofing | `#00BFFF` | 🔵 Cyan |
| **3** | Tiled Roofing | `#E15759` | 🔴 Red |
| **4** | Utility Infrastructure | `#9C27B0` | 🟣 Purple |
| **5** | Hydrology / Water Bodies | `#4E79A7` | 💧 Blue |
| **6** | Road Networks (Paved & Unpaved)| `#F2CB6C` | 🟡 Yellow |
