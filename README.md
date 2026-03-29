# Geospatial-intellignece
Automated AI pipeline for extracting building footprints, road networks, and waterbodies from drone orthomosaics. Uses a U-Net backbone with a custom Texture Veto Engine to eliminate false positives. Includes a Streamlit dashboard for real-time GIS-ready class filtering and 2GB+ file handling.
------------------------------------------------------------
🚨 CRITICAL: AI MODEL WEIGHTS REQUIRED
------------------------------------------------------------
Due to file size limits on GitHub, the trained PyTorch model 
weights are hosted on Google Drive. 

1. DOWNLOAD MODEL: https://drive.google.com/file/d/1qYywmgl9I_G2Qm8Q580r2tbtweZcWAd7/view?usp=sharing
2. FILE NAME: mopr_hybrid_shape_3050.pth
3. INSTALLATION: Place the downloaded .pth file directly 
   into the root 'project_iit' folder (same level as app.py).
   *EITHER MAKE THE FOLDER NAMES AND STRUCTURE SAME, OR CHANGE THE PATH INSIDE CODE*
------------------------------------------------------------
